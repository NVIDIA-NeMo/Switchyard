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

use libsy::{
    Algorithm, ContentBlock, Context, Decision, Driver, LlmRequest, LlmTargetSet, Message, Request,
    Response, Role, Signals,
};

/// Preamble prepended to the user prompt when asking the classifier target for a
/// strong-win-rate score.
const CLASSIFIER_PROMPT_PREAMBLE: &str = "Rate how strongly this request needs a frontier model. \
     Reply with a single strong-win-rate score in [0, 1]:\n";

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
pub struct LlmClassifierOrchAlgo {
    classifier_model: String,
    strong_model: String,
    weak_model: String,
    threshold: f64,
    target_set: LlmTargetSet,
}

impl LlmClassifierOrchAlgo {
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

#[async_trait]
impl Algorithm for LlmClassifierOrchAlgo {
    async fn create_run_task(
        self: Arc<Self>,
        _ctx: Context,
        driver: Driver,
        request: Request,
    ) -> Result<Response, Box<dyn Error + Send + Sync>> {
        let user_prompt = request
            .llm_request
            .instructions
            .iter()
            .flat_map(|instruction| instruction.content.iter())
            .chain(
                request
                    .llm_request
                    .messages
                    .iter()
                    .flat_map(|message| message.content.iter()),
            )
            .filter_map(|block| match block {
                ContentBlock::Text { text }
                | ContentBlock::Refusal { text }
                | ContentBlock::Reasoning { text, .. } => Some(text.as_str()),
                ContentBlock::Unknown { raw, .. } => raw.as_str(),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");

        // 1. Classify: call the classifier target with the score-eliciting prompt.
        let classifier_target = self.target_set.get_target(&self.classifier_model)?;
        let classify_request = Request {
            llm_request: LlmRequest {
                model: request.llm_request.model.clone(),
                messages: vec![Message::text(
                    Role::User,
                    format!("{CLASSIFIER_PROMPT_PREAMBLE}{user_prompt}"),
                )],
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: request.metadata.clone(),
        };
        let classify_decision: Arc<dyn Decision> = Arc::new(ClassifierDecision {
            selected_model: self.classifier_model.clone(),
            reasoning: format!("classifying request via {}", self.classifier_model),
            score: None,
            tier: None,
        });
        driver.info(classify_decision.clone()).await?;
        let classify_response = driver
            .call_llm_target(&classifier_target, classify_request, classify_decision)
            .await?;
        let score_text = classify_response
            .llm_response
            .outputs
            .iter()
            .flat_map(|output| output.content.iter())
            .filter_map(|block| match block {
                ContentBlock::Text { text }
                | ContentBlock::Refusal { text }
                | ContentBlock::Reasoning { text, .. } => Some(text.as_str()),
                ContentBlock::Unknown { raw, .. } => raw.as_str(),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        let score = score_text.trim().parse::<f64>().ok();

        // 2. Route: pick strong/weak. Fail open — an unparseable score routes strong.
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
        let routed_request = Request {
            llm_request: request.llm_request,
            raw_request: request.raw_request,
            metadata: request.metadata,
        };
        driver.info(route_decision.clone()).await?;
        driver
            .call_llm_target(&routed_target, routed_request, route_decision)
            .await
    }

    async fn process_signals(
        self: Arc<Self>,
        _signals: Signals,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Stateless classification; agent-system signals are ignored.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libsy::{LlmClient, LlmResponse, LlmTarget, Response, ResponseOutput, RoutedRequest};
    use std::sync::Mutex;

    /// Returns `score` for the classifier target, an answer tagged with the model
    /// otherwise; records the requests it saw so a test can inspect the classifier
    /// prompt.
    struct ScoringClient {
        classifier_model: String,
        score: String,
        seen: Arc<Mutex<Vec<Request>>>,
    }

    #[async_trait]
    impl LlmClient for ScoringClient {
        async fn call(
            &self,
            routed: RoutedRequest,
        ) -> Result<Response, Box<dyn Error + Send + Sync>> {
            let name = routed.decision.selected_model().to_string();
            let outputs = if name == self.classifier_model {
                vec![ResponseOutput {
                    role: Role::Assistant,
                    content: vec![ContentBlock::Text {
                        text: self.score.clone(),
                    }],
                    stop_reason: None,
                }]
            } else {
                Vec::new()
            };
            self.seen
                .lock()
                .map_err(|_| "lock poisoned")?
                .push(routed.request);
            Ok(Response {
                llm_response: LlmResponse {
                    model: Some(name),
                    outputs,
                    ..LlmResponse::default()
                },
                metadata: None,
            })
        }
    }

    /// Build a classifier algo whose three targets share a scoring client.
    fn algo(threshold: f64, score: &str) -> (LlmClassifierOrchAlgo, Arc<Mutex<Vec<Request>>>) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let client = Arc::new(ScoringClient {
            classifier_model: "router/classifier".to_string(),
            score: score.to_string(),
            seen: Arc::clone(&seen),
        }) as Arc<dyn LlmClient>;
        let target = |name: &str| LlmTarget {
            semantic_name: name.to_string(),
            llm_client: Some(client.clone()),
        };
        let target_set = LlmTargetSet::new(vec![
            target("router/classifier"),
            target("frontier/model"),
            target("cheap/model"),
        ]);
        let algo = LlmClassifierOrchAlgo {
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
            llm_request: LlmRequest {
                model: Some("auto".to_string()),
                messages: vec![Message::text(Role::User, prompt)],
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: None,
        }
    }

    /// Wrap a classifier algo as `Arc<dyn Algorithm>` we can drive to completion.
    fn orch(algo: LlmClassifierOrchAlgo) -> Arc<dyn Algorithm> {
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
            response.llm_response.model.as_deref(),
            Some("frontier/model")
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
        assert_eq!(response.llm_response.model.as_deref(), Some("cheap/model"));
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
            response.llm_response.model.as_deref(),
            Some("frontier/model")
        );
        Ok(())
    }

    #[tokio::test]
    async fn unparseable_score_defaults_to_strong() -> Result<(), Box<dyn Error + Send + Sync>> {
        let (algo, _) = algo(0.5, "not-a-number");
        let (trace, response) = orch(algo).run(Context::default(), request("hi")).await?;
        assert_eq!(
            response.llm_response.model.as_deref(),
            Some("frontier/model")
        );
        let routed = as_classifier(&trace[1])?;
        assert_eq!(routed.tier, Some(ClassifierTier::Strong));
        assert_eq!(routed.score, None);
        Ok(())
    }

    #[tokio::test]
    async fn classifier_prompt_includes_the_user_text() -> Result<(), Box<dyn Error + Send + Sync>>
    {
        let (algo, seen) = algo(0.5, "0.9");
        orch(algo)
            .run(Context::default(), request("prove it"))
            .await?;
        let seen = seen.lock().map_err(|_| "lock poisoned")?;
        // Two calls: the classifier (preamble + user text), then the routed model.
        assert_eq!(seen.len(), 2);
        let classifier_prompt = seen[0].llm_request.messages[0]
            .text_content("\n")
            .ok_or("classifier prompt is not text")?;
        assert!(classifier_prompt.contains("prove it"));
        assert!(classifier_prompt.contains("frontier model"));
        assert_eq!(
            seen[1].llm_request.messages[0].text_content("\n"),
            Some("prove it".to_string())
        );
        Ok(())
    }
}
