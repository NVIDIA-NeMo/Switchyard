// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Response-quality escalation: call the efficient model first, evaluate its response
//! with a judge, and escalate to the capable model only when the response is insufficient.
//!
//! Once a session escalates, it is pinned to the capable model for all remaining turns —
//! avoiding repeated weak attempts on a task the efficient model already failed.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::Value;
use switchyard_protocol::{completion_text, LlmRequest, Message, OutputParams, Role};

use crate::{
    algorithms::util::JudgeConfig, Algorithm, Context, Decision, Driver, LibsyError, LlmResponse,
    LlmTarget, Request, Response, Result, RoutedLlmClient, SharedState,
};

const PROMPT_TEMPLATE: &str = include_str!("../prompts/escalation/prompt.md");
const SCHEMA_TEMPLATE: &str = include_str!("../prompts/escalation/schema.json");

/// Upper bound on retained session pins, keeping the process-local map from growing
/// without limit; an arbitrary entry is evicted once the bound is reached.
const MAX_PINS: usize = 4096;

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

struct JudgeCallDecision {
    model: String,
}

impl Decision for JudgeCallDecision {
    fn selected_model(&self) -> &str {
        &self.model
    }

    fn is_routed_call(&self) -> bool {
        false
    }

    fn reasoning(&self) -> Option<&str> {
        Some("escalation quality judge")
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Routes to an efficient model first, escalates to a capable model when the judge finds
/// the response insufficient, and pins the session to the capable model thereafter.
pub struct EscalationRouter {
    efficient_target: LlmTarget,
    capable_target: LlmTarget,
    judge_target: LlmTarget,
    judge_config: JudgeConfig,
    /// Session IDs pinned to the capable model after their first escalation.
    pins: Mutex<HashMap<String, ()>>,
}

impl EscalationRouter {
    /// Routes to `efficient_target` first; escalates to `capable_target` when the
    /// `judge_target` finds the response insufficient.
    pub fn new(
        efficient_target: LlmTarget,
        capable_target: LlmTarget,
        judge_target: LlmTarget,
    ) -> Result<Self> {
        Ok(Self {
            efficient_target,
            capable_target,
            judge_target,
            judge_config: Self::load_judge_config()?,
            pins: Mutex::new(HashMap::new()),
        })
    }

    fn load_judge_config() -> Result<JudgeConfig> {
        let response_schema: Value =
            serde_json::from_str(SCHEMA_TEMPLATE).map_err(|error| LibsyError::AlgorithmError {
                message: format!("escalation response schema is invalid: {error}"),
            })?;
        let prompt_schema = response_schema
            .pointer("/json_schema/schema")
            .ok_or_else(|| LibsyError::AlgorithmError {
                message: "escalation response schema has no json_schema.schema".to_string(),
            })?;
        let prompt_schema = serde_json::to_string_pretty(prompt_schema).map_err(|error| {
            LibsyError::AlgorithmError {
                message: format!("escalation prompt schema could not be rendered: {error}"),
            }
        })?;
        Ok(JudgeConfig {
            system_prompt: PROMPT_TEMPLATE.replace("{{RESPONSE_SCHEMA}}", &prompt_schema),
            response_schema: Some(response_schema),
        })
    }

    fn is_pinned(&self, session_id: &str) -> bool {
        self.pins.lock().contains_key(session_id)
    }

    fn pin(&self, session_id: &str) {
        let mut pins = self.pins.lock();
        if pins.len() >= MAX_PINS {
            if let Some(evicted) = pins.keys().next().cloned() {
                pins.remove(&evicted);
            }
        }
        pins.insert(session_id.to_string(), ());
    }

    /// Builds the judge request: judge system prompt, original system/developer instructions,
    /// last user message, and the efficient model's response as an assistant turn.
    fn build_judge_request(
        &self,
        original: &Request,
        efficient_text: &str,
    ) -> Request {
        let mut messages = vec![Message::text(
            Role::System,
            self.judge_config.system_prompt.clone(),
        )];
        // Retain the original system/developer instructions so the judge shares context.
        messages.extend(
            original
                .llm_request
                .messages
                .iter()
                .filter(|m| matches!(m.role, Role::System | Role::Developer))
                .cloned(),
        );
        // The last user message is what was actually asked.
        if let Some(last_user) = original
            .llm_request
            .messages
            .iter()
            .rfind(|m| m.role == Role::User)
        {
            messages.push(last_user.clone());
        }
        // The efficient model's response is the candidate answer to evaluate.
        messages.push(Message::text(Role::Assistant, efficient_text));
        Request {
            llm_request: LlmRequest {
                model: original.llm_request.model.clone(),
                messages,
                output: OutputParams {
                    response_format: self.judge_config.response_schema.clone(),
                    ..OutputParams::default()
                },
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: original.metadata.clone(),
        }
    }

    /// Calls the judge and returns whether escalation is warranted.
    /// Any failure (transport, parse) returns `false` — the judge is an optimization,
    /// not a gatekeeper, so failing safe means skipping escalation.
    async fn consult_judge(
        &self,
        ctx: Context,
        driver: &Driver,
        request: &Request,
        efficient_text: &str,
    ) -> bool {
        let judge_model = self.judge_target.semantic_name.as_str();
        let warn = |error: &dyn std::fmt::Display| {
            tracing::warn!(
                target: "libsy",
                judge_model,
                error = %error,
                "escalation judge unavailable; skipping escalation"
            );
        };

        let judge_request = self.build_judge_request(request, efficient_text);
        let judge_decision: Arc<dyn Decision> = Arc::new(JudgeCallDecision {
            model: self.judge_target.semantic_name.clone(),
        });

        let response = match driver
            .call_llm_target(ctx, &self.judge_target, judge_request, judge_decision)
            .await
        {
            Ok(r) => r,
            Err(e) => {
                warn(&e);
                return false;
            }
        };

        let agg = match response.llm_response.into_agg().await {
            Ok(a) => a,
            Err(e) => {
                warn(&e);
                return false;
            }
        };

        let text = completion_text(&agg);
        match parse_verdict(text.trim()) {
            Some(v) => v.should_escalate,
            None => {
                tracing::warn!(
                    target: "libsy",
                    judge_model,
                    "escalation judge verdict did not parse; skipping escalation"
                );
                false
            }
        }
    }
}

fn parse_verdict(text: &str) -> Option<EscalationVerdict> {
    serde_json::from_str(strip_json_fence(text)).ok()
}

fn strip_json_fence(text: &str) -> &str {
    let Some(rest) = text.strip_prefix("```") else {
        return text;
    };
    let rest = rest.strip_prefix("json").unwrap_or(rest);
    let rest = rest.trim_start_matches(['\n', '\r']);
    rest.strip_suffix("```").map(str::trim).unwrap_or(rest)
}

#[async_trait]
impl Algorithm<SharedState> for EscalationRouter {
    fn name(&self) -> &str {
        "escalation"
    }

    fn count_tokens_client(&self) -> Option<Arc<dyn RoutedLlmClient>> {
        [&self.capable_target, &self.efficient_target, &self.judge_target]
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
        request: Request,
    ) -> Result<Response> {
        let bare_ctx = ctx.without_state();

        // 1. Check whether this session is already pinned to the capable model.
        let session_id = request
            .metadata
            .as_ref()
            .and_then(|m| m.session_id.as_deref())
            .map(str::to_string);

        if session_id.as_deref().is_some_and(|id| self.is_pinned(id)) {
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

        let efficient_text = completion_text(&efficient_agg);

        // 4. Consult the judge; any failure returns false (skip escalation).
        let should_escalate = self
            .consult_judge(bare_ctx.clone(), &driver, &request, &efficient_text)
            .await;

        // 5. No escalation: return the efficient response.
        if !should_escalate {
            return Ok(Response {
                llm_response: LlmResponse::Agg(efficient_agg),
                metadata: None,
            });
        }

        // 6. Escalate: pin the session so later turns skip the efficient attempt.
        if let Some(ref id) = session_id {
            self.pin(id);
        }

        let capable_decision: Arc<dyn Decision> = Arc::new(EscalationDecision {
            model: self.capable_target.semantic_name.clone(),
            tier: "strong",
            reason: "quality escalation to capable model",
        });
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

    #[tokio::test]
    async fn pins_are_bounded_by_max_pins() {
        let router = router("e", "c", "j");
        for i in 0..=super::MAX_PINS {
            router.pin(&format!("session-{i}"));
        }
        assert_eq!(router.pins.lock().len(), super::MAX_PINS);
    }
}
