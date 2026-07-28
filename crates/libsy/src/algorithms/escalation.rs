// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Response-quality escalation: call the efficient model first, evaluate its response
//! with a judge, and escalate to the capable model only when the response is insufficient.
//!
//! Once a session escalates, it is pinned to the capable model for all remaining turns —
//! avoiding repeated weak attempts on a task the efficient model already failed.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use switchyard_protocol::{completion_text, LlmRequest, Message, OutputParams, Role};

use crate::{
    algorithms::util::{
        load_judge_config, AffinityRouter, Judge, JudgeClassifier, JudgeConfig, JudgePolicy,
    },
    Algorithm, Classification, Classifier, Context, Decision, Driver, Event, LibsyError,
    LlmResponse, LlmTarget, Processor, Request, Response, Result, RoutedLlmClient, Score,
    SharedState, State, StateValue,
};

const PROMPT_TEMPLATE: &str = include_str!("../prompts/escalation/prompt.md");
const SCHEMA_TEMPLATE: &str = include_str!("../prompts/escalation/schema.json");

/// Key in [`State::extra`] that carries the efficient model's response to the judge.
const CANDIDATE_KEY: &str = "escalation.candidate";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EscalationVerdict {
    should_escalate: bool,
    #[allow(dead_code)]
    confidence: f64,
    #[allow(dead_code)]
    reason: String,
}

struct EscalationDecision {
    model: String,
    tier: &'static str,
    reason: &'static str,
}

impl Decision for EscalationDecision {
    fn selected_model(&self) -> &str {
        &self.model
    }

    fn routing_tier(&self) -> Option<&'static str> {
        Some(self.tier)
    }

    fn reasoning(&self) -> Option<&str> {
        Some(self.reason)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Builds the judge request from the original request and the efficient model's response,
/// stored in [`State::extra`] under [`CANDIDATE_KEY`] before the judge is called.
struct EscalationJudge {
    config: JudgeConfig,
}

impl Judge for EscalationJudge {
    type Verdict = EscalationVerdict;

    fn build_request(&self, state: &State, request: &Request) -> Request {
        let candidate = state
            .extra
            .get(CANDIDATE_KEY)
            .and_then(|v| {
                if let StateValue::String(s) = v {
                    Some(s.as_str())
                } else {
                    None
                }
            })
            .unwrap_or("");

        let mut messages =
            vec![Message::text(Role::System, self.config.system_prompt.clone())];
        messages.extend(
            request
                .llm_request
                .messages
                .iter()
                .filter(|m| matches!(m.role, Role::System | Role::Developer))
                .cloned(),
        );
        if let Some(last_user) = request
            .llm_request
            .messages
            .iter()
            .rfind(|m| m.role == Role::User)
        {
            messages.push(last_user.clone());
        }
        messages.push(Message::text(Role::Assistant, candidate));
        Request {
            llm_request: LlmRequest {
                model: request.llm_request.model.clone(),
                messages,
                output: OutputParams {
                    response_format: self.config.response_schema.clone(),
                    ..OutputParams::default()
                },
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: request.metadata.clone(),
        }
    }
}

/// Routes to the capable target when the judge says to escalate, and abstains otherwise.
struct EscalationPolicy {
    capable: String,
}

impl JudgePolicy for EscalationPolicy {
    type Verdict = EscalationVerdict;

    fn to_classification(&self, verdict: Option<&EscalationVerdict>) -> Classification {
        if verdict.is_some_and(|v| v.should_escalate) {
            Classification::Scores(vec![Score {
                target: self.capable.clone(),
                confidence: 1.0,
            }])
        } else {
            Classification::Scores(Vec::new())
        }
    }
}

/// Routes to an efficient model first, escalates to a capable model when the judge finds
/// the response insufficient, and pins the session to the capable model thereafter.
pub struct EscalationRouter {
    efficient_target: LlmTarget,
    capable_target: LlmTarget,
    judge_classifier: JudgeClassifier<EscalationJudge, EscalationPolicy>,
    /// Latches sessions to the capable model after their first escalation.
    affinity: AffinityRouter,
}

impl EscalationRouter {
    /// Routes to `efficient_target` first; escalates to `capable_target` when the
    /// `judge_target` finds the response insufficient.
    pub fn new(
        efficient_target: LlmTarget,
        capable_target: LlmTarget,
        judge_target: LlmTarget,
    ) -> Result<Self> {
        let config = load_judge_config(PROMPT_TEMPLATE, SCHEMA_TEMPLATE)?;
        let capable_name = capable_target.semantic_name.clone();
        Ok(Self {
            efficient_target,
            capable_target,
            judge_classifier: JudgeClassifier::new(
                EscalationJudge { config },
                judge_target,
                EscalationPolicy { capable: capable_name.clone() },
            ),
            affinity: AffinityRouter::new().with_latch_only([capable_name]),
        })
    }
}

#[async_trait]
impl Algorithm<SharedState> for EscalationRouter {
    fn name(&self) -> &str {
        "escalation"
    }

    fn count_tokens_client(&self) -> Option<Arc<dyn RoutedLlmClient>> {
        [&self.capable_target, &self.efficient_target]
            .iter()
            .find_map(|t| {
                t.llm_client
                    .as_ref()
                    .filter(|c| c.supports_count_tokens())
                    .cloned()
            })
    }

    async fn create_run_task(
        self: Arc<Self>,
        ctx: Context<SharedState>,
        driver: Driver,
        mut request: Request,
    ) -> Result<Response> {
        let bare_ctx = ctx.without_state();

        // 1. Check whether this session is already pinned to the capable model.
        let is_pinned = {
            let mut state = ctx.state.lock().await;
            let classification = self.affinity.score(&mut state, &mut request, None).await?;
            matches!(classification, Classification::Scores(ref s) if !s.is_empty())
        };

        if is_pinned {
            let decision: Arc<dyn Decision> = Arc::new(EscalationDecision {
                model: self.capable_target.semantic_name.clone(),
                tier: "strong",
                reason: "session pinned to capable after prior escalation",
            });
            driver.info(bare_ctx.clone(), decision.clone()).await?;
            return driver
                .call_llm_target(bare_ctx, &self.capable_target, request, decision)
                .await;
        }

        // 2. Call the efficient model.
        let efficient_decision: Arc<dyn Decision> = Arc::new(EscalationDecision {
            model: self.efficient_target.semantic_name.clone(),
            tier: "weak",
            reason: "initial efficient-model attempt",
        });
        driver
            .info(bare_ctx.clone(), efficient_decision.clone())
            .await?;
        let efficient_response = driver
            .call_llm_target(
                bare_ctx.clone(),
                &self.efficient_target,
                request.clone(),
                efficient_decision,
            )
            .await?;

        // 3. Aggregate the efficient response so the judge can read it.
        let efficient_agg = efficient_response
            .llm_response
            .into_agg()
            .await
            .map_err(|e| LibsyError::external("aggregating efficient response", e))?;

        // 4. Store the efficient response in state and consult the judge.
        let should_escalate = {
            let mut state = ctx.state.lock().await;
            state.extra.insert(
                CANDIDATE_KEY.to_string(),
                StateValue::String(completion_text(&efficient_agg)),
            );
            let classification = self
                .judge_classifier
                .score(&mut state, &mut request, Some(&driver))
                .await?;
            matches!(classification, Classification::Scores(ref s) if !s.is_empty())
        };

        // 5. No escalation: return the efficient response.
        if !should_escalate {
            return Ok(Response {
                llm_response: LlmResponse::Agg(efficient_agg),
                metadata: None,
            });
        }

        // 6. Escalate: latch the session to the capable model so later turns skip the efficient attempt.
        let capable_decision: Arc<dyn Decision> = Arc::new(EscalationDecision {
            model: self.capable_target.semantic_name.clone(),
            tier: "strong",
            reason: "quality escalation to capable model",
        });
        {
            let mut state = ctx.state.lock().await;
            self.affinity
                .process(&mut state, Event::Request(&mut request))
                .await?;
            self.affinity
                .process(&mut state, Event::Decision(&*capable_decision))
                .await?;
        }

        driver
            .info(bare_ctx.clone(), capable_decision.clone())
            .await?;
        driver
            .call_llm_target(bare_ctx, &self.capable_target, request, capable_decision)
            .await
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Arc;

    use parking_lot::Mutex;
    use switchyard_protocol::{completion_text, text_request, text_response, LlmClientError, Metadata};

    use crate::{
        Algorithm, Context, Decision, LlmResponse, LlmTarget, Request, Response, RoutedLlmClient,
        SharedState,
    };

    use super::EscalationRouter;

    fn router(efficient: &str, capable: &str, judge: &str) -> Arc<EscalationRouter> {
        Arc::new(
            EscalationRouter::new(
                target(efficient, None),
                target(capable, None),
                target(judge, None),
            )
            .expect("schema and prompt must load"),
        )
    }

    fn target(name: &str, client: Option<Arc<dyn RoutedLlmClient>>) -> LlmTarget {
        LlmTarget {
            semantic_name: name.to_string(),
            llm_client: client,
        }
    }

    fn request(session_id: Option<&str>) -> Request {
        Request {
            llm_request: text_request(Some("auto".to_string()), "What is 2+2?"),
            raw_request: None,
            metadata: session_id.map(|id| Metadata {
                session_id: Some(id.to_string()),
                ..Metadata::default()
            }),
        }
    }

    fn verdict_json(should_escalate: bool) -> String {
        format!(
            r#"{{"should_escalate": {should_escalate}, "confidence": 0.9, "reason": "test"}}"#
        )
    }

    /// Records which models were called (in order) and returns pre-supplied text replies.
    ///
    /// All three targets share one instance; calls are served in arrival order regardless
    /// of which target the driver is calling. This mirrors the shared `RecordingClient`
    /// pattern used throughout the libsy test suite.
    struct RecordingClient {
        /// Text replies (or `Err(())` for a simulated transport failure) in call order.
        replies: Mutex<VecDeque<Result<String, ()>>>,
        /// Model names in the order each `call()` arrived.
        calls: Arc<Mutex<Vec<String>>>,
    }

    impl RecordingClient {
        fn new(
            replies: impl IntoIterator<Item = Result<String, ()>>,
        ) -> (Arc<Self>, Arc<Mutex<Vec<String>>>) {
            let calls = Arc::new(Mutex::new(Vec::new()));
            let client = Arc::new(Self {
                replies: Mutex::new(replies.into_iter().collect()),
                calls: calls.clone(),
            });
            (client, calls)
        }
    }

    #[async_trait::async_trait]
    impl RoutedLlmClient for RecordingClient {
        async fn call(
            &self,
            _ctx: Context,
            _request: Request,
            decision: Arc<dyn Decision>,
        ) -> std::result::Result<Response, LlmClientError> {
            self.calls.lock().push(decision.selected_model().to_string());
            match self.replies.lock().pop_front().unwrap_or(Ok("fallback".into())) {
                Ok(text) => Ok(Response {
                    llm_response: LlmResponse::Agg(text_response(None, &text)),
                    metadata: None,
                }),
                Err(()) => Err(LlmClientError::Configuration {
                    message: "test error".into(),
                }),
            }
        }
    }

    /// Router where all three targets share one `RecordingClient`.
    /// Replies are served in the order they are supplied, matching the call order:
    /// first the efficient call, then the judge call, then (if escalating) the capable call.
    fn instrumented_router(
        replies: impl IntoIterator<Item = Result<String, ()>>,
    ) -> (Arc<EscalationRouter>, Arc<Mutex<Vec<String>>>) {
        let (client, calls) = RecordingClient::new(replies);
        let c: Arc<dyn RoutedLlmClient> = client;
        let router = Arc::new(
            EscalationRouter::new(
                target("efficient", Some(c.clone())),
                target("capable", Some(c.clone())),
                target("judge", Some(c.clone())),
            )
            .expect("schema and prompt must load"),
        );
        (router, calls)
    }

    /// Convenience: a normal text reply.
    fn ok(text: &str) -> Result<String, ()> {
        Ok(text.to_string())
    }

    #[tokio::test]
    async fn pass_judge_returns_efficient_response() -> crate::Result<()> {
        let (router, calls) = instrumented_router([
            ok("4"),                            // efficient
            ok(&verdict_json(false)),           // judge: no escalation
        ]);

        let (_, response) = router
            .run(Context::<SharedState>::default(), request(Some("s1")))
            .await?;

        let agg = response.llm_response.as_agg().expect("should be Agg");
        assert_eq!(completion_text(agg), "4");
        assert_eq!(*calls.lock(), vec!["efficient", "judge"]);
        Ok(())
    }

    #[tokio::test]
    async fn fail_judge_escalates_to_capable() -> crate::Result<()> {
        let (router, calls) = instrumented_router([
            ok("I don't know"),                 // efficient
            ok(&verdict_json(true)),            // judge: escalate
            ok("The answer is 4"),              // capable
        ]);

        let (trace, response) = router
            .run(Context::<SharedState>::default(), request(Some("s2")))
            .await?;

        let agg = response.llm_response.as_agg().expect("should be Agg");
        assert_eq!(completion_text(agg), "The answer is 4");
        assert_eq!(*calls.lock(), vec!["efficient", "judge", "capable"]);
        assert_eq!(trace.len(), 2);
        assert_eq!(trace[0].selected_model(), "efficient");
        assert_eq!(trace[1].selected_model(), "capable");
        Ok(())
    }

    #[tokio::test]
    async fn pinned_session_skips_efficient_and_judge() -> crate::Result<()> {
        let (router, calls) = instrumented_router([
            ok("wrong"),                        // efficient (first request)
            ok(&verdict_json(true)),            // judge: escalate
            ok("right"),                        // capable (first request)
            ok("right again"),                  // capable (second request — pinned)
        ]);

        // First request escalates and pins the session.
        router
            .clone()
            .run(Context::<SharedState>::default(), request(Some("s3")))
            .await?;

        // Second request on the same session goes straight to capable.
        let (trace, _) = router
            .run(Context::<SharedState>::default(), request(Some("s3")))
            .await?;

        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].selected_model(), "capable");
        assert_eq!(
            *calls.lock(),
            vec!["efficient", "judge", "capable", "capable"]
        );
        Ok(())
    }

    #[tokio::test]
    async fn judge_failure_skips_escalation() -> crate::Result<()> {
        let (router, calls) = instrumented_router([
            ok("4"),                            // efficient
            ok("sorry, I cannot help"),         // judge: unparseable → fail safe
        ]);

        let (_, response) = router
            .run(Context::<SharedState>::default(), request(Some("s4")))
            .await?;

        let agg = response.llm_response.as_agg().expect("should be Agg");
        assert_eq!(completion_text(agg), "4");
        assert_eq!(*calls.lock(), vec!["efficient", "judge"]);
        Ok(())
    }

    #[tokio::test]
    async fn no_session_id_routes_without_pinning() -> crate::Result<()> {
        let (router, calls) = instrumented_router([
            ok("4"),                            // efficient (request 1)
            ok(&verdict_json(true)),            // judge: escalate (request 1)
            ok("4"),                            // capable (request 1)
            ok("4"),                            // efficient (request 2 — not pinned)
            ok(&verdict_json(false)),           // judge: no escalation (request 2)
        ]);

        router
            .clone()
            .run(Context::<SharedState>::default(), request(None))
            .await?;

        router
            .run(Context::<SharedState>::default(), request(None))
            .await?;

        assert_eq!(
            *calls.lock(),
            vec!["efficient", "judge", "capable", "efficient", "judge"]
        );
        Ok(())
    }

    #[tokio::test]
    async fn schema_and_prompt_load_at_construction() {
        router("a", "b", "c");
    }

    #[tokio::test]
    async fn routing_tiers_are_labelled() -> crate::Result<()> {
        let (router, _) = instrumented_router([
            ok("4"),
            ok(&verdict_json(false)),
        ]);

        let (trace, _) = router
            .run(Context::<SharedState>::default(), request(Some("s5")))
            .await?;

        assert_eq!(trace[0].routing_tier(), Some("weak"));
        Ok(())
    }
}
