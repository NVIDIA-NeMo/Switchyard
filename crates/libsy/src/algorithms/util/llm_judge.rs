// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A generic LLM *judge* and the classifier that routes on its verdict.
//!
//! A judge is the reusable half of an LLM classifier: it turns a request into a model call
//! and the model's reply into a structured verdict, but holds no routing policy of its own.
//! Concretely an [`LlmJudge<T>`] is a [`JudgeConfig`] (the prompt and optional response
//! schema — everything that is *data*) plus two pure functions:
//!
//! 1. [`build_request`](LlmJudge::build_request) — given the session [`State`] and the
//!    inbound [`Request`], produce the [`Request`] to send to the judge model.
//! 2. [`parse`](LlmJudge::parse) — turn the judge model's aggregated reply into a
//!    caller-defined structured verdict `T`.
//!
//! The *policy* — how a verdict becomes a route — lives in a [`JudgePolicy`], and
//! [`JudgeClassifier`] ties a judge, the model target that answers it, and a policy into a
//! [`Classifier`] for the [`FallThrough`](crate::algorithms::FallThrough) cascade. Keeping
//! the two apart lets one judge (prompt + schema) back many policies — thresholds, floors,
//! or a direct recommendation — without re-deriving the prompt.

// A reusable building block that no profile wires in yet; its consumers are the tests until
// a concrete judge-backed router adopts it.
#![allow(dead_code)]

use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde_json::Value;
use switchyard_protocol::{
    completion_text, AggLlmResponse, LlmRequest, Message, OutputParams, Role,
};

use crate::{
    Classification, Classifier, Context, Decision, Driver, LibsyError, LlmTarget, Request, Result,
    State,
};

/// The prompt and response schema that define a judge — everything about a judge that is
/// data rather than logic.
#[derive(Clone, Debug, Default)]
pub struct JudgeConfig {
    /// System prompt prepended to the judge request; tells the model what to assess and in
    /// what shape to answer.
    pub system_prompt: String,
    /// Optional JSON schema for structured output, attached as the request's
    /// [`response_format`](OutputParams::response_format). `None` leaves the model
    /// free-form (the judge still parses its reply as JSON).
    pub response_schema: Option<Value>,
}

/// A generic LLM judge: builds a judge request and parses the reply into a verdict `T`.
///
/// Stateless and cheap to share. `T` is the caller's structured verdict — any
/// [`DeserializeOwned`] type: a struct of signals, a bare score, an enum. The judge is
/// deliberately policy-free; pair it with a [`JudgePolicy`] through [`JudgeClassifier`].
pub struct LlmJudge<T> {
    config: JudgeConfig,
    // `fn() -> T` keeps the judge `Send + Sync` and covariant in `T` without owning one.
    _verdict: PhantomData<fn() -> T>,
}

impl<T: DeserializeOwned> LlmJudge<T> {
    /// Creates a judge from its prompt-and-schema [`JudgeConfig`].
    pub fn new(config: JudgeConfig) -> Self {
        Self {
            config,
            _verdict: PhantomData,
        }
    }

    /// Builds the request to send to the judge model.
    ///
    /// Prepends the configured system prompt to the inbound conversation and attaches the
    /// response schema (if any) as the request's structured-output format, preserving the
    /// inbound model name. `state` is offered for stateful judges; the base judge ignores
    /// it.
    pub fn build_request(&self, _state: &State, request: &Request) -> Request {
        let mut messages = Vec::with_capacity(request.llm_request.messages.len() + 1);
        messages.push(Message::text(Role::System, self.config.system_prompt.clone()));
        messages.extend(request.llm_request.messages.iter().cloned());
        let llm_request = LlmRequest {
            model: request.llm_request.model.clone(),
            messages,
            output: OutputParams {
                response_format: self.config.response_schema.clone(),
                ..OutputParams::default()
            },
            ..LlmRequest::default()
        };
        Request {
            llm_request,
            raw_request: None,
            metadata: request.metadata.clone(),
        }
    }

    /// Parses the judge model's aggregated reply into the verdict `T`.
    ///
    /// The completion is read as JSON — a bare number and a `{ … }` object are both valid —
    /// tolerating a Markdown ```` ```json ```` fence. Returns [`LibsyError::AlgorithmError`]
    /// when the reply is not valid `T`; judge output is untrusted, so callers typically
    /// treat this failure as an abstention rather than a hard error.
    pub fn parse(&self, response: &AggLlmResponse) -> Result<T> {
        let completion = completion_text(response);
        let json = strip_json_fence(completion.trim());
        serde_json::from_str::<T>(json).map_err(|err| LibsyError::AlgorithmError {
            message: format!(
                "judge reply did not parse as {}: {err}",
                std::any::type_name::<T>()
            ),
        })
    }
}

/// The routing *policy* layered on a judge's verdict: how a parsed `T` (or its absence)
/// becomes a [`Classification`].
///
/// `verdict` is `None` when the judge could not be consulted or its reply failed to parse.
/// A policy should **fail closed** there — never route untrusted or missing output to a
/// less-capable target.
pub trait JudgePolicy<T>: Send + Sync {
    /// Classifies from the parsed verdict, or its absence.
    fn classify(&self, verdict: Option<&T>) -> Classification;
}

/// A [`Classifier`] built from a judge, the model target that answers it, and a policy.
///
/// On each turn it builds the judge request, offloads the judge model call on the
/// per-request [`Driver`], parses the reply into `T`, and hands it to the [`JudgePolicy`].
/// A parse failure becomes `None` so the policy fails closed; a model-call failure is
/// surfaced as an error. Without a driver it cannot consult the judge and applies the
/// policy's fail-closed branch.
pub struct JudgeClassifier<T, P> {
    judge: LlmJudge<T>,
    target: LlmTarget,
    policy: P,
}

impl<T, P> JudgeClassifier<T, P>
where
    T: DeserializeOwned,
    P: JudgePolicy<T>,
{
    /// Ties a `judge` to the `target` model that answers it and the `policy` that routes on
    /// its verdict.
    pub fn new(judge: LlmJudge<T>, target: LlmTarget, policy: P) -> Self {
        Self {
            judge,
            target,
            policy,
        }
    }
}

#[async_trait]
impl<T, P> Classifier for JudgeClassifier<T, P>
where
    T: DeserializeOwned + Send + Sync,
    P: JudgePolicy<T>,
{
    async fn score(
        &self,
        state: &mut State,
        request: &Request,
        driver: Option<&Driver>,
    ) -> Result<Classification> {
        // Without a driver there is no way to offload the judge's model call; the policy
        // decides how to fail closed.
        let Some(driver) = driver else {
            return Ok(self.policy.classify(None));
        };

        let judge_request = self.judge.build_request(state, request);
        let decision: Arc<dyn Decision> = Arc::new(JudgeDecision {
            model: self.target.semantic_name.clone(),
        });
        let response = driver
            .call_llm_target(Context::default(), &self.target, judge_request, decision)
            .await?;
        let agg = response
            .llm_response
            .into_agg()
            .await
            .map_err(|err| LibsyError::external("judge model call", err))?;

        // The judge's reply is untrusted external data: a parse failure is an abstention,
        // not a hard error, so the policy fails closed instead of the run aborting.
        let verdict = self.judge.parse(&agg).ok();
        Ok(self.policy.classify(verdict.as_ref()))
    }
}

/// The decision published for the judge's own model call, so the offloaded call is
/// attributed to the judge target in the trace.
struct JudgeDecision {
    model: String,
}

impl Decision for JudgeDecision {
    fn selected_model(&self) -> &str {
        &self.model
    }
    fn reasoning(&self) -> Option<&str> {
        Some("llm judge consultation")
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Strips a Markdown ```` ```json … ``` ```` (or bare ```` ``` … ``` ````) fence, so a judge
/// that wraps its JSON in a code block still parses. Returns the inner text.
fn strip_json_fence(text: &str) -> &str {
    let Some(rest) = text.strip_prefix("```") else {
        return text;
    };
    let rest = rest.strip_prefix("json").unwrap_or(rest);
    let rest = rest.trim_start_matches(['\n', '\r']);
    rest.strip_suffix("```").map(str::trim).unwrap_or(rest)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::algorithms::FallThrough;
    use crate::{Algorithm, LlmResponse, LlmTargetSet, Response, RoutedLlmClient, Score};
    use switchyard_protocol::{text_request, text_response, LlmClientError};

    /// A structured judge verdict for the tests: a target name plus a confidence.
    #[derive(serde::Deserialize)]
    struct Verdict {
        route: String,
        confidence: f64,
    }

    /// Routes to the verdict's target at its confidence; fails closed to `"strong"` when
    /// the judge produced nothing parseable.
    struct RoutePolicy;

    impl JudgePolicy<Verdict> for RoutePolicy {
        fn classify(&self, verdict: Option<&Verdict>) -> Classification {
            match verdict {
                Some(v) => Classification::Scores(vec![Score {
                    target: v.route.clone(),
                    confidence: v.confidence,
                }]),
                None => Classification::Scores(vec![Score {
                    target: "strong".to_string(),
                    confidence: 1.0,
                }]),
            }
        }
    }

    fn judge() -> LlmJudge<Verdict> {
        LlmJudge::new(JudgeConfig {
            system_prompt: "assess".to_string(),
            response_schema: Some(serde_json::json!({ "type": "object" })),
        })
    }

    fn request(prompt: &str) -> Request {
        Request {
            llm_request: text_request(Some("auto".to_string()), prompt),
            raw_request: None,
            metadata: None,
        }
    }

    #[test]
    fn build_request_prepends_prompt_and_sets_schema() {
        let built = judge().build_request(&State::default(), &request("do the task"));
        // System prompt first, then the inbound conversation, model name preserved.
        assert_eq!(built.llm_request.messages[0].role, Role::System);
        assert_eq!(
            built.llm_request.messages[0].text_content(" ").as_deref(),
            Some("assess")
        );
        assert_eq!(built.llm_request.messages.len(), 2);
        assert_eq!(built.llm_request.model.as_deref(), Some("auto"));
        assert!(built.llm_request.output.response_format.is_some());
    }

    #[test]
    fn parse_reads_a_json_verdict() -> Result<()> {
        let v = judge().parse(&text_response(None, r#"{"route":"weak","confidence":0.8}"#))?;
        assert_eq!(v.route, "weak");
        assert_eq!(v.confidence, 0.8);
        Ok(())
    }

    #[test]
    fn parse_tolerates_a_json_fence() -> Result<()> {
        let fenced = "```json\n{\"route\":\"weak\",\"confidence\":0.5}\n```";
        let v = judge().parse(&text_response(None, fenced))?;
        assert_eq!(v.route, "weak");
        Ok(())
    }

    #[test]
    fn parse_rejects_malformed_output() {
        assert!(judge().parse(&text_response(None, "not json")).is_err());
    }

    /// Returns a fixed JSON verdict for the judge target, a model-tagged answer otherwise.
    struct JudgeClient {
        judge_model: String,
        verdict_json: String,
    }

    #[async_trait]
    impl RoutedLlmClient for JudgeClient {
        async fn call(
            &self,
            _ctx: Context,
            _request: Request,
            decision: Arc<dyn Decision>,
        ) -> std::result::Result<Response, LlmClientError> {
            let name = decision.selected_model().to_string();
            let completion = if name == self.judge_model {
                self.verdict_json.clone()
            } else {
                format!("answer from {name}")
            };
            Ok(Response {
                llm_response: LlmResponse::Agg(text_response(None, completion)),
                metadata: None,
            })
        }
    }

    /// Drives a `FallThrough` whose sole classifier is a judge classifier over a client
    /// that answers the judge with `verdict_json`, returning the completion text and the
    /// routed model.
    async fn route_via_judge(verdict_json: &str) -> Result<(String, String)> {
        let client = Arc::new(JudgeClient {
            judge_model: "judge/model".to_string(),
            verdict_json: verdict_json.to_string(),
        }) as Arc<dyn RoutedLlmClient>;
        let target = |name: &str| LlmTarget {
            semantic_name: name.to_string(),
            llm_client: Some(client.clone()),
        };
        let classifier = JudgeClassifier::new(judge(), target("judge/model"), RoutePolicy);
        let router = Arc::new(
            FallThrough::new(LlmTargetSet::new(vec![target("strong"), target("weak")]))
                .with_classifier(Arc::new(classifier)),
        );
        let (trace, response) = router.run(Context::default(), request("do the task")).await?;
        let text = response
            .llm_response
            .as_agg()
            .map(completion_text)
            .unwrap_or_default();
        let routed = trace
            .last()
            .map(|d| d.selected_model().to_string())
            .unwrap_or_default();
        Ok((text, routed))
    }

    #[tokio::test]
    async fn judge_classifier_routes_on_the_parsed_verdict() -> Result<()> {
        let (text, routed) = route_via_judge(r#"{"route":"weak","confidence":0.9}"#).await?;
        assert_eq!(routed, "weak");
        assert_eq!(text, "answer from weak");
        Ok(())
    }

    #[tokio::test]
    async fn malformed_verdict_fails_closed_to_strong() -> Result<()> {
        // The judge returns junk; the policy's `None` branch routes to the capable target.
        let (text, routed) = route_via_judge("garbage, not json").await?;
        assert_eq!(routed, "strong");
        assert_eq!(text, "answer from strong");
        Ok(())
    }

    #[tokio::test]
    async fn without_a_driver_the_policy_fails_closed() -> Result<()> {
        // A classifier called with no driver cannot consult the judge; it must fail closed.
        let classifier = JudgeClassifier::new(
            judge(),
            LlmTarget {
                semantic_name: "judge/model".to_string(),
                llm_client: None,
            },
            RoutePolicy,
        );
        let classification = classifier
            .score(&mut State::default(), &request("hi"), None)
            .await?;
        assert_eq!(
            classification.argmax(false)?.map(|s| s.target),
            Some("strong".to_string())
        );
        Ok(())
    }
}
