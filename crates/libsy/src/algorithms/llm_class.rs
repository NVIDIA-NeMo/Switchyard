// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LLM-classifier router built on the [`Algorithm`] interfaces.
//!
//! Unlike a local ML classifier (which scores a prompt in-process), an LLM
//! classifier needs its own model call to classify the request. On the new
//! interfaces this is just two ordinary `driver.call_llm_target`s inside one
//! `create_run_task`: first the classifier target (to get a score), then the
//! routed strong/weak target. The multi-step nature is invisible to the caller —
//! it is the algorithm's own control flow.

use std::error::Error;
use std::sync::Arc;

use async_trait::async_trait;

use crate::{Algorithm, Context, Decision, Driver, LlmTargetSet, Request, Response};
use switchyard_protocol::{completion_text, LlmRequest, Message, Role};

/// The system prompt tells the classifier what to do
const CLASSIFIER_SYSTEM_PROMPT: &str = "Rate how strongly this request needs a frontier model. Reply with a single strong-win-rate score in [0, 1].";

/// The tier a classifier score selected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClassifierTier {
    Strong,
    Weak,
}

impl ClassifierTier {
    /// Stable string form of the tier, used in decision reasoning.
    pub fn as_str(self) -> &'static str {
        match self {
            ClassifierTier::Strong => "strong",
            ClassifierTier::Weak => "weak",
        }
    }
}

/// Decision produced at each step of the classifier flow. The classify step
/// leaves `score`/`tier` `None`; the routed step fills them in.
pub struct ClassifierDecision {
    /// The model this step selected (the classifier model, then the routed model).
    pub selected_model: String,
    /// Human-readable explanation of the step.
    pub reasoning: String,
    /// The classifier score, on the routed step; `None` on the classify step.
    pub score: Option<f64>,
    /// The tier chosen (strong/weak), on the routed step; `None` on the classify step.
    pub tier: Option<ClassifierTier>,
}

impl Decision for ClassifierDecision {
    fn selected_model(&self) -> &str {
        &self.selected_model
    }
    fn reasoning(&self) -> Option<&str> {
        Some(&self.reasoning)
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// LLM-classifier router: classify with one target, then route to strong/weak.
pub struct LlmClassifier {
    classifier_model: String,
    strong_model: String,
    weak_model: String,
    threshold: f64,
    target_set: LlmTargetSet,
}

impl LlmClassifier {
    /// Configure the classifier: the model that scores each request, the strong
    /// and weak models to route to, the score `threshold` at or above which the
    /// strong model is chosen, and the `target_set` to route among. That set must
    /// contain targets named `classifier_model`, `strong_model`, and `weak_model`.
    pub fn new(
        classifier_model: impl Into<String>,
        strong_model: impl Into<String>,
        weak_model: impl Into<String>,
        threshold: f64,
        target_set: LlmTargetSet,
    ) -> Self {
        Self {
            classifier_model: classifier_model.into(),
            strong_model: strong_model.into(),
            weak_model: weak_model.into(),
            threshold,
            target_set,
        }
    }
}

/// The first User message as well as the last <recent_turn_window> turns (User + Assistant).
/// If fewer than 5 turns have happened, include the whole message.
fn trim_messages(messages: &[Message], recent_turn_window: usize) -> Vec<Message> {
    let mut system = Vec::new();
    let mut first_user = None;
    let mut first_user_idx = None;
    for (idx, message) in messages.iter().enumerate() {
        match message.role {
            Role::System | Role::Developer => system.push(message.clone()),
            Role::User if first_user.is_none() => {
                first_user = Some(message.clone());
                first_user_idx = Some(idx);
            }
            _ => {}
        }
    }
    let Some(first_user) = first_user else {
        return system;
    };
    let tail = messages
        .iter()
        .enumerate()
        .filter(|(idx, message)| {
            *idx > first_user_idx.unwrap_or(0)
                && !matches!(message.role, Role::System | Role::Developer)
        })
        .map(|(_, message)| message.clone())
        .collect::<Vec<_>>();
    if recent_turn_window == 0 {
        let mut out = system;
        out.push(first_user);
        if let Some(last_user) = tail.iter().rev().find(|message| message.role == Role::User) {
            out.push(last_user.clone());
        }
        return out;
    }
    let mut out = system;
    out.push(first_user);
    let start = tail.len().saturating_sub(recent_turn_window);
    out.extend_from_slice(&tail[start..]);
    out
}

#[async_trait]
impl Algorithm for LlmClassifier {
    async fn create_run_task(
        self: Arc<Self>,
        ctx: Context,
        driver: Driver,
        request: Request,
    ) -> Result<Response, Box<dyn Error + Send + Sync>> {
        // The agent's inbound name rides through unchanged on every sub-call; the
        // model each sub-call actually hits is carried by its decision instead.
        let inbound = request.llm_request.model.clone();

        let mut just_the_key_messages = trim_messages(&request.llm_request.messages, 5);
        just_the_key_messages.insert(0, Message::text(Role::System, CLASSIFIER_SYSTEM_PROMPT));

        // 1. Classify: call the classifier target with the score-eliciting prompt.
        let classifier_target = self.target_set.get_target(&self.classifier_model)?;
        let llm_request = LlmRequest {
            model: inbound.clone(),
            messages: just_the_key_messages,
            ..LlmRequest::default()
        };
        let classify_request = Request {
            llm_request,
            raw_request: request.raw_request.clone(),
            metadata: request.metadata.clone(),
        };
        let classify_decision: Arc<dyn Decision> = Arc::new(ClassifierDecision {
            selected_model: self.classifier_model.clone(),
            reasoning: format!("classifying request via {}", self.classifier_model),
            score: None,
            tier: None,
        });
        driver.info(ctx.clone(), classify_decision.clone()).await?;
        let classify_response = driver
            .call_llm_target(
                ctx.clone(),
                &classifier_target,
                classify_request,
                classify_decision,
            )
            .await?;
        // Drain a streamed classifier response to its aggregate so a streamed
        // score is read instead of silently dropped. Keep only a valid
        // probability in [0.0, 1.0]: NaN, ±inf, and out-of-range values parse
        // as f64 but are not usable scores, so they collapse to None and fail
        // open to strong below rather than being treated as a real verdict.
        let score = completion_text(&classify_response.llm_response.into_agg().await?)
            .trim()
            .parse::<f64>()
            .ok()
            .filter(|s| (0.0..=1.0).contains(s));

        // 2. Route: pick strong/weak. Fail open — an unparseable or out-of-range
        //    score routes strong.
        let (tier, model) = match score {
            Some(s) if s >= self.threshold => (ClassifierTier::Strong, self.strong_model.clone()),
            Some(_) => (ClassifierTier::Weak, self.weak_model.clone()),
            None => (ClassifierTier::Strong, self.strong_model.clone()),
        };
        let routed_target = self.target_set.get_target(&model)?;
        let route_decision: Arc<dyn Decision> = Arc::new(ClassifierDecision {
            reasoning: format!(
                "classifier score {score:?} vs threshold {}; selected {model} ({})",
                self.threshold,
                tier.as_str()
            ),
            selected_model: model.clone(),
            score,
            tier: Some(tier),
        });
        driver.info(ctx.clone(), route_decision.clone()).await?;
        driver
            .call_llm_target(ctx, &routed_target, request, route_decision)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LlmResponse, LlmResponseChunk, LlmTarget, Response, RoutedLlmClient};
    use futures::StreamExt;
    use std::sync::Mutex;
    use switchyard_protocol::text_response;

    /// Returns `score` for the classifier target, an answer tagged with the model
    /// otherwise; records the requests it saw so a test can inspect the classifier
    /// prompt.
    struct ScoringClient {
        classifier_model: String,
        score: String,
        /// When true, the classifier target returns its score as a live stream
        /// rather than a buffered aggregate, exercising the drain-the-stream path.
        stream: bool,
        seen: Arc<Mutex<Vec<Request>>>,
    }

    #[async_trait]
    impl RoutedLlmClient for ScoringClient {
        async fn call(
            &self,
            _ctx: Context,
            request: Request,
            decision: Arc<dyn Decision>,
        ) -> Result<Response, Box<dyn Error + Send + Sync>> {
            let name = decision.selected_model().to_string();
            let is_classifier = name == self.classifier_model;
            let completion = if is_classifier {
                self.score.clone()
            } else {
                format!("answer from {name}")
            };
            self.seen.lock().map_err(|_| "lock poisoned")?.push(request);
            // A streaming classifier target emits its score as a live TextDelta
            // stream; the algorithm must drain it (into_agg) to read the score.
            let llm_response = if is_classifier && self.stream {
                let chunks = vec![Ok(LlmResponseChunk::TextDelta {
                    index: 0,
                    text: completion,
                })];
                LlmResponse::Stream(futures::stream::iter(chunks).boxed())
            } else {
                LlmResponse::Agg(text_response(None, completion))
            };
            Ok(Response {
                llm_response,
                metadata: None,
            })
        }
    }

    /// Build a classifier algo whose three targets share a scoring client.
    fn algo(threshold: f64, score: &str) -> (LlmClassifier, Arc<Mutex<Vec<Request>>>) {
        build_algo(threshold, score, false)
    }

    /// Like [`algo`], but the classifier target returns its score as a live stream.
    fn algo_streaming(threshold: f64, score: &str) -> (LlmClassifier, Arc<Mutex<Vec<Request>>>) {
        build_algo(threshold, score, true)
    }

    fn build_algo(
        threshold: f64,
        score: &str,
        stream: bool,
    ) -> (LlmClassifier, Arc<Mutex<Vec<Request>>>) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let client = Arc::new(ScoringClient {
            classifier_model: "router/classifier".to_string(),
            score: score.to_string(),
            stream,
            seen: Arc::clone(&seen),
        }) as Arc<dyn RoutedLlmClient>;
        let target = |name: &str| LlmTarget {
            semantic_name: name.to_string(),
            llm_client: Some(client.clone()),
        };
        let target_set = LlmTargetSet::new(vec![
            target("router/classifier"),
            target("frontier/model"),
            target("cheap/model"),
        ]);
        let algo = LlmClassifier {
            classifier_model: "router/classifier".to_string(),
            strong_model: "frontier/model".to_string(),
            weak_model: "cheap/model".to_string(),
            threshold,
            target_set,
        };
        (algo, seen)
    }

    fn request(prompt: &str) -> Request {
        Request {
            llm_request: switchyard_protocol::text_request(Some("auto".to_string()), prompt),
            raw_request: None,
            metadata: None,
        }
    }

    /// Wrap a classifier algo as `Arc<dyn Algorithm>` we can drive to completion.
    fn orch(algo: LlmClassifier) -> Arc<dyn Algorithm> {
        Arc::new(algo)
    }

    /// Downcast a trace entry to the concrete classifier decision.
    fn as_classifier(
        d: &Arc<dyn Decision>,
    ) -> Result<&ClassifierDecision, Box<dyn Error + Send + Sync>> {
        d.as_any()
            .downcast_ref::<ClassifierDecision>()
            .ok_or_else(|| "expected a ClassifierDecision".into())
    }

    #[tokio::test]
    async fn score_at_or_above_threshold_routes_strong() -> Result<(), Box<dyn Error + Send + Sync>>
    {
        let (algo, _) = algo(0.5, "0.9");
        let (trace, response) = orch(algo)
            .run(Context::default(), request("solve this proof"))
            .await?;
        assert_eq!(
            response
                .llm_response
                .as_agg()
                .map(completion_text)
                .unwrap_or_default(),
            "answer from frontier/model"
        );
        // Trace: [classify, route].
        assert_eq!(trace[0].selected_model(), "router/classifier");
        let routed = as_classifier(&trace[1])?;
        assert_eq!(routed.selected_model, "frontier/model");
        assert_eq!(routed.tier, Some(ClassifierTier::Strong));
        assert_eq!(routed.score, Some(0.9));
        Ok(())
    }

    #[tokio::test]
    async fn score_below_threshold_routes_weak() -> Result<(), Box<dyn Error + Send + Sync>> {
        let (algo, _) = algo(0.5, "0.2");
        let (trace, response) = orch(algo)
            .run(Context::default(), request("say hello"))
            .await?;
        assert_eq!(
            response
                .llm_response
                .as_agg()
                .map(completion_text)
                .unwrap_or_default(),
            "answer from cheap/model"
        );
        let routed = as_classifier(&trace[1])?;
        assert_eq!(routed.tier, Some(ClassifierTier::Weak));
        assert_eq!(routed.score, Some(0.2));
        Ok(())
    }

    #[tokio::test]
    async fn score_exactly_at_threshold_routes_strong() -> Result<(), Box<dyn Error + Send + Sync>>
    {
        let (algo, _) = algo(0.5, "0.5");
        let (_, response) = orch(algo)
            .run(Context::default(), request("borderline"))
            .await?;
        assert_eq!(
            response
                .llm_response
                .as_agg()
                .map(completion_text)
                .unwrap_or_default(),
            "answer from frontier/model"
        );
        Ok(())
    }

    #[tokio::test]
    async fn unparseable_score_defaults_to_strong() -> Result<(), Box<dyn Error + Send + Sync>> {
        let (algo, _) = algo(0.5, "not-a-number");
        let (trace, response) = orch(algo).run(Context::default(), request("hi")).await?;
        assert_eq!(
            response
                .llm_response
                .as_agg()
                .map(completion_text)
                .unwrap_or_default(),
            "answer from frontier/model"
        );
        let routed = as_classifier(&trace[1])?;
        assert_eq!(routed.tier, Some(ClassifierTier::Strong));
        assert_eq!(routed.score, None);
        Ok(())
    }

    #[tokio::test]
    async fn out_of_range_score_defaults_to_strong() -> Result<(), Box<dyn Error + Send + Sync>> {
        // "NaN" parses as an f64 but is not a usable probability, so it must fail
        // open to strong rather than be treated as a below-threshold verdict and
        // silently downgraded to weak (NvBug 6485976).
        let (algo, _) = algo(0.5, "NaN");
        let (trace, response) = orch(algo).run(Context::default(), request("hi")).await?;
        assert_eq!(
            response
                .llm_response
                .as_agg()
                .map(completion_text)
                .unwrap_or_default(),
            "answer from frontier/model"
        );
        let routed = as_classifier(&trace[1])?;
        assert_eq!(routed.tier, Some(ClassifierTier::Strong));
        assert_eq!(routed.score, None);
        Ok(())
    }

    #[tokio::test]
    async fn streamed_score_below_threshold_routes_weak() -> Result<(), Box<dyn Error + Send + Sync>>
    {
        // A streaming classifier target must have its score drained and read, not
        // dropped — otherwise every streamed verdict falls open to strong
        // regardless of value (NvBug 6485975).
        let (algo, _) = algo_streaming(0.5, "0.2");
        let (trace, response) = orch(algo)
            .run(Context::default(), request("say hello"))
            .await?;
        assert_eq!(
            response
                .llm_response
                .as_agg()
                .map(completion_text)
                .unwrap_or_default(),
            "answer from cheap/model"
        );
        let routed = as_classifier(&trace[1])?;
        assert_eq!(routed.tier, Some(ClassifierTier::Weak));
        assert_eq!(routed.score, Some(0.2));
        Ok(())
    }
}
