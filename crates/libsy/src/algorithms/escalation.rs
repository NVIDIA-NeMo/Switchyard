// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Escalation routing that judges an efficient model's answer before selecting a serving tier.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use switchyard_protocol::{
    AggLlmResponse, LlmClientError, LlmResponse, Message, ModelId, Request, Response, Role,
};

use super::util::classifier_contract::ClassifierContractConfig;
use super::util::decisive;
use super::util::escalation::{
    self, DeescalationConfig, EscalationJudge, EscalationJudgeConfig, EscalationPolicy,
    EvaluationPhase,
};
use super::util::llm_judge::JudgeClassifier;
use crate::core::algorithm::Driver;
use crate::core::classifier::{Classification, Classifier};
use crate::core::state::{State, StateValue};
use crate::{LibsyError, Result};

const STREAK_KEY: &str = "escalation_streak";
const STRONG_CALLS_KEY: &str = "escalation_strong_calls";
const RELEASE_STREAK_KEY: &str = "escalation_release_streak";
const WEAK_COOLDOWN_KEY: &str = "escalation_weak_cooldown";

fn count(state: &State, key: &str) -> u32 {
    match state.extra.get(key) {
        Some(StateValue::Count(n)) => *n,
        _ => 0,
    }
}

fn set_count(state: &mut State, key: &str, value: u32) {
    state
        .extra
        .insert(key.to_string(), StateValue::Count(value));
}

fn assistant_message(response: &AggLlmResponse) -> Message {
    Message {
        role: Role::Assistant,
        content: response
            .first_output()
            .map(|output| output.content.clone())
            .unwrap_or_default(),
    }
}

/// Calls the efficient model, judges its response, and latches to capable once the streak
/// confirms. Returns the efficient response directly when not escalating so the caller does
/// not pay for a second model call.
struct EscalationClassifier {
    escalation_judge: JudgeClassifier<EscalationJudge, EscalationPolicy>,
    capable: ModelId,
    efficient: ModelId,
    /// Consecutive escalate verdicts required to latch.
    confirmations: u32,
    deescalation: Option<DeescalationPolicy>,
    missing_session_warning_emitted: AtomicBool,
}

struct DeescalationPolicy {
    config: DeescalationConfig,
    judge: JudgeClassifier<EscalationJudge, EscalationPolicy>,
}

/// Builds the escalation classifier used by the shared LLM classifier route shell.
pub(super) fn build_classifier(
    judge_target: ModelId,
    efficient_target: &ModelId,
    capable_target: &ModelId,
    contract_config: ClassifierContractConfig,
    config: EscalationJudgeConfig,
    max_output_tokens: u64,
) -> Result<Arc<dyn Classifier<State>>> {
    let confirmations = config.confirmations;
    let deescalation = match config.deescalation {
        Some(deescalation) => Some(DeescalationPolicy {
            config: deescalation,
            judge: escalation::build_judge(
                judge_target.clone(),
                capable_target.clone(),
                efficient_target.clone(),
                &contract_config,
                config.clone(),
                Some(EvaluationPhase::Strong),
                max_output_tokens,
            )?,
        }),
        None => None,
    };
    let is_phase_aware = deescalation.is_some();
    let classifier: Arc<dyn Classifier<State>> = Arc::new(EscalationClassifier {
        escalation_judge: escalation::build_judge(
            judge_target,
            capable_target.clone(),
            efficient_target.clone(),
            &contract_config,
            config,
            is_phase_aware.then_some(EvaluationPhase::Efficient),
            max_output_tokens,
        )?,
        capable: capable_target.clone(),
        efficient: efficient_target.clone(),
        confirmations,
        deescalation,
        missing_session_warning_emitted: AtomicBool::new(false),
    });
    Ok(classifier)
}

impl EscalationClassifier {
    async fn review_capable(
        &self,
        deescalation: &DeescalationPolicy,
        state: &mut State,
        request: &Request,
        driver: &Driver,
        strong_calls: u32,
    ) -> Result<(Classification, Option<Response>)> {
        let next_strong_call = strong_calls.saturating_add(1);

        // The call that confirms escalation is the first capable call.
        if next_strong_call < deescalation.config.strong_min_calls {
            set_count(state, STRONG_CALLS_KEY, next_strong_call);
            driver.set_evidence(serde_json::json!({"source": "escalation", "verdict": "latched"}));
            return Ok((decisive(&self.capable), None));
        }

        let capable_response = driver
            .call_model(
                request.clone(),
                vec![self.capable.clone(), self.efficient.clone()],
            )
            .await?;
        if capable_response.served_model() == Some(&self.efficient) {
            set_count(state, RELEASE_STREAK_KEY, 0);
            driver.set_evidence(serde_json::json!({"source": "fallback"}));
            return Ok((decisive(&self.efficient), Some(capable_response)));
        }
        let agg = match capable_response.llm_response.into_agg().await {
            Ok(agg) => agg,
            Err(LlmClientError::Transport { .. }) => {
                set_count(state, RELEASE_STREAK_KEY, 0);
                driver.set_evidence(
                    serde_json::json!({"source": "fallback", "reason_code": "transport"}),
                );
                return Ok((decisive(&self.efficient), None));
            }
            Err(source) => {
                return Err(LibsyError::client_call(self.capable.clone(), source));
            }
        };
        let mut judge_request = request.clone();
        judge_request
            .llm_request
            .messages
            .push(assistant_message(&agg));
        let capable_response = Response {
            llm_response: if request.llm_request.stream {
                LlmResponse::Stream(agg.into_stream())
            } else {
                LlmResponse::Agg(agg)
            },
            metadata: capable_response.metadata,
        };

        let (classification, _) = deescalation
            .judge
            .score(state, &mut judge_request, Some(driver))
            .await?;
        let best = classification.argmax(false)?;
        let release_streak = match &best {
            Some(score) if score.target == self.efficient => {
                count(state, RELEASE_STREAK_KEY).saturating_add(1)
            }
            _ => 0,
        };
        set_count(state, STRONG_CALLS_KEY, next_strong_call);
        set_count(state, RELEASE_STREAK_KEY, release_streak);

        if release_streak >= deescalation.config.confirmations {
            set_count(state, STREAK_KEY, 0);
            set_count(state, STRONG_CALLS_KEY, 0);
            set_count(state, RELEASE_STREAK_KEY, 0);
            tracing::debug!(
                target = %self.efficient,
                "de-escalation policy released session to efficient tier"
            );
        } else if release_streak > 0 {
            driver.set_evidence(serde_json::json!({"source": "escalation", "verdict": "pending"}));
        }

        // A confirmed release applies on the next request; this turn is already complete.
        Ok((decisive(&self.capable), Some(capable_response)))
    }
}

#[async_trait]
impl Classifier<State> for EscalationClassifier {
    async fn score(
        &self,
        state: &mut State,
        request: &mut Request,
        driver: Option<&Driver>,
    ) -> Result<(Classification, Option<Response>)> {
        let Some(driver) = driver else {
            return Err(LibsyError::AlgorithmError {
                message: "escalation classifier requires a driver".into(),
            });
        };

        let has_session_id = request
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.session_id.as_deref())
            .is_some_and(|session_id| !session_id.is_empty());
        if (self.deescalation.is_some() || self.confirmations > 1)
            && !has_session_id
            && !self
                .missing_session_warning_emitted
                .swap(true, Ordering::Relaxed)
        {
            tracing::warn!(
                target: "libsy",
                confirmations = self.confirmations,
                deescalation = self.deescalation.is_some(),
                required_header = "x-switchyard-session-id",
                "stateful escalation has no session ID; routing state will not persist"
            );
        }

        let mut strong_calls = count(state, STRONG_CALLS_KEY);
        if let Some(deescalation) = &self.deescalation
            && deescalation
                .config
                .strong_max_calls
                .is_some_and(|strong_max_calls| strong_calls >= strong_max_calls)
        {
            set_count(state, STREAK_KEY, 0);
            set_count(state, STRONG_CALLS_KEY, 0);
            set_count(state, RELEASE_STREAK_KEY, 0);
            set_count(
                state,
                WEAK_COOLDOWN_KEY,
                deescalation.config.weak_cooldown_calls,
            );
            strong_calls = 0;
            tracing::debug!(
                target = %self.efficient,
                "de-escalation policy reached its hard limit and returned to efficient tier"
            );
        }
        if let Some(deescalation) = &self.deescalation
            && strong_calls > 0
        {
            return self
                .review_capable(deescalation, state, request, driver, strong_calls)
                .await;
        }

        // A confirmed permanent escalation stays capable without a judge call.
        if self.deescalation.is_none() && count(state, STREAK_KEY) >= self.confirmations {
            driver.set_evidence(serde_json::json!({
                "source": "escalation",
                "verdict": "latched",
            }));
            return Ok((decisive(&self.capable), None));
        }

        // Call efficient model and buffer the response so the judge can read it.
        //
        // If the efficient model exceeds its context window, fall through to capable. This call
        // deliberately has one candidate so the classifier sees the efficient model's error.
        tracing::info!(
            target = %self.efficient,
            "escalation classifier selected efficient tier"
        );
        let efficient_response = match driver
            .call_model(request.clone(), vec![self.efficient.clone()])
            .await
        {
            Ok(r) => r,
            Err(LibsyError::ClientCall {
                source: LlmClientError::ContextWindowExceeded { .. },
                ..
            }) => {
                driver.set_evidence(serde_json::json!({
                    "source": "fallback",
                    "reason_code": "context_window",
                }));
                return Ok((decisive(&self.capable), None));
            }
            Err(e) => return Err(e),
        };
        // The call resolves when its stream handle arrives; transport can still fail while
        // buffering. Fall back only for that availability failure and keep other errors typed.
        let agg = match efficient_response.llm_response.into_agg().await {
            Ok(agg) => agg,
            Err(LlmClientError::Transport { .. }) => {
                driver.set_evidence(serde_json::json!({
                    "source": "fallback",
                    "reason_code": "transport",
                }));
                return Ok((decisive(&self.capable), None));
            }
            Err(source) => {
                return Err(LibsyError::client_call(self.efficient.clone(), source));
            }
        };
        // Append the efficient reply so the judge reads this turn's completed trajectory.
        let mut judge_request = request.clone();
        judge_request
            .llm_request
            .messages
            .push(assistant_message(&agg));
        let efficient_response = Response {
            llm_response: if request.llm_request.stream {
                LlmResponse::Stream(agg.into_stream())
            } else {
                LlmResponse::Agg(agg)
            },
            metadata: efficient_response.metadata,
        };

        let weak_cooldown = count(state, WEAK_COOLDOWN_KEY);
        if weak_cooldown > 0 {
            set_count(state, WEAK_COOLDOWN_KEY, weak_cooldown - 1);
            driver
                .set_evidence(serde_json::json!({"source": "deescalation", "verdict": "cooldown"}));
            return Ok((decisive(&self.efficient), Some(efficient_response)));
        }

        let (classification, _) = self
            .escalation_judge
            .score(state, &mut judge_request, Some(driver))
            .await?;

        let held = count(state, STREAK_KEY);
        let best = classification.argmax(false)?;
        let (escalate, pending) = match &best {
            Some(score) if score.target == self.capable => (true, held.saturating_add(1)),
            Some(_) => (false, 0),
            None => (false, held),
        };
        set_count(state, STREAK_KEY, pending);

        if escalate && pending >= self.confirmations {
            driver.set_evidence(serde_json::json!({
                "source": "escalation",
                "verdict": "escalate",
            }));
            if self.deescalation.is_some() {
                set_count(state, STRONG_CALLS_KEY, 1);
                set_count(state, RELEASE_STREAK_KEY, 0);
            }
            return Ok((decisive(&self.capable), None));
        }

        if escalate {
            driver.set_evidence(serde_json::json!({
                "source": "escalation",
                "verdict": "pending",
            }));
        }

        Ok((decisive(&self.efficient), Some(efficient_response)))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Arc;

    use parking_lot::Mutex;
    use switchyard_protocol::{
        ContentBlock, LlmClientError, LlmResponse, LlmResponseChunk, Metadata, Request, Response,
        completion_text, text_request, text_response,
    };

    use super::*;
    use crate::algorithms::llm_class::{LlmClassifierConfig, LlmTaskClassifier};
    use crate::algorithms::util::DEFAULT_JUDGE_MAX_OUTPUT_TOKENS;
    use crate::core::testing::{Serve, reply, test_drive};

    /// A queue of replies, drained in order.
    struct Queue(Mutex<VecDeque<String>>);

    impl Queue {
        fn new(replies: impl IntoIterator<Item = &'static str>) -> Arc<Self> {
            Arc::new(Self(Mutex::new(
                replies.into_iter().map(String::from).collect(),
            )))
        }

        fn take(&self) -> String {
            self.0
                .lock()
                .pop_front()
                .unwrap_or_else(|| "unexpected call".to_string())
        }
    }

    /// Serves the judge and model targets from separate reply queues.
    fn queued(model: Arc<Queue>, judge: Arc<Queue>) -> impl Serve {
        move |target: ModelId, request: Request| {
            let queue = if target == "judge" {
                Arc::clone(&judge)
            } else {
                Arc::clone(&model)
            };
            async move {
                Ok(Response {
                    llm_response: LlmResponse::Agg(text_response(None, queue.take())),
                    metadata: request.metadata,
                })
            }
        }
    }

    fn classify_request() -> Request {
        Request {
            llm_request: text_request(Some("auto".to_string()), "classify this task"),
            raw_request: None,
            metadata: None,
        }
    }

    fn classify_session_request() -> Request {
        Request {
            metadata: Some(Metadata {
                session_id: Some("session-1".to_string()),
                ..Metadata::default()
            }),
            ..classify_request()
        }
    }

    /// Returns a stream that emits partial content before failing during aggregation.
    fn streamed_then_error(error: LlmClientError) -> Response {
        Response {
            llm_response: LlmResponse::Stream(Box::pin(futures::stream::iter([
                Ok(LlmResponseChunk::TextDelta {
                    index: 0,
                    text: "partial".to_string(),
                }
                .into()),
                Err(error),
            ]))),
            metadata: None,
        }
    }

    /// Builds a router with escalation enabled (`confirmations=1` latches immediately).
    fn escalation_router() -> Result<Arc<LlmTaskClassifier>> {
        Ok(Arc::new(LlmTaskClassifier::new(
            LlmClassifierConfig::Escalation {
                judge_target: ModelId::from("judge"),
                efficient_target: ModelId::from("efficient"),
                capable_target: ModelId::from("capable"),
                contract: ClassifierContractConfig::default(),
                config: EscalationJudgeConfig {
                    confirmations: 1,
                    ..EscalationJudgeConfig::default()
                },
                max_output_tokens: DEFAULT_JUDGE_MAX_OUTPUT_TOKENS,
            },
        )?))
    }

    fn deescalation_router(config: DeescalationConfig) -> Result<Arc<LlmTaskClassifier>> {
        Ok(Arc::new(LlmTaskClassifier::new(
            LlmClassifierConfig::Escalation {
                judge_target: ModelId::from("judge"),
                efficient_target: ModelId::from("efficient"),
                capable_target: ModelId::from("capable"),
                contract: ClassifierContractConfig::default(),
                config: EscalationJudgeConfig {
                    confirmations: 1,
                    deescalation: Some(config),
                    ..EscalationJudgeConfig::default()
                },
                max_output_tokens: DEFAULT_JUDGE_MAX_OUTPUT_TOKENS,
            },
        )?))
    }

    async fn selected_over(
        router: Arc<LlmTaskClassifier>,
        turns: usize,
        model: Arc<Queue>,
        judge: Arc<Queue>,
    ) -> Result<Vec<ModelId>> {
        let request = classify_session_request();
        let mut selected = Vec::with_capacity(turns);
        for _ in 0..turns {
            selected.push(
                test_drive(
                    router.clone(),
                    request.clone(),
                    queued(Arc::clone(&model), Arc::clone(&judge)),
                )
                .await?
                .0,
            );
        }
        Ok(selected)
    }

    #[tokio::test]
    async fn serves_efficient_when_judge_declines() -> Result<()> {
        let judge = Queue::new([r#"{"escalate":false,"reason":"progressing"}"#]);
        let model = Queue::new(["efficient answer"]);

        let (selected_model, response) = test_drive(
            escalation_router()?,
            classify_request(),
            queued(model, judge),
        )
        .await?;

        assert_eq!(selected_model, "efficient");
        assert_eq!(
            response.llm_response.as_agg().map(completion_text),
            Some("efficient answer".to_string())
        );
        Ok(())
    }

    #[tokio::test]
    async fn config_overrides_the_packaged_prompt() -> Result<()> {
        let prompts = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&prompts);
        let serve = move |target: ModelId, request: Request| {
            if target == "judge" {
                let prompt = request
                    .llm_request
                    .instructions
                    .first()
                    .and_then(|instruction| {
                        instruction.content.iter().find_map(|block| match block {
                            ContentBlock::Text { text } => Some(text.clone()),
                            _ => None,
                        })
                    });
                recorded.lock().extend(prompt);
                std::future::ready(Ok(reply(r#"{"escalate":false,"reason":"progressing"}"#)))
            } else {
                std::future::ready(Ok(reply("efficient answer")))
            }
        };
        let router = Arc::new(LlmTaskClassifier::new(LlmClassifierConfig::Escalation {
            judge_target: ModelId::from("judge"),
            efficient_target: ModelId::from("efficient"),
            capable_target: ModelId::from("capable"),
            contract: ClassifierContractConfig::default().with_prompt("Custom trajectory rubric."),
            config: EscalationJudgeConfig {
                confirmations: 1,
                ..EscalationJudgeConfig::default()
            },
            max_output_tokens: DEFAULT_JUDGE_MAX_OUTPUT_TOKENS,
        })?);

        test_drive(router, classify_request(), serve).await?;

        assert_eq!(&*prompts.lock(), &["Custom trajectory rubric."]);
        Ok(())
    }

    #[tokio::test]
    async fn upgrades_to_capable_when_judge_escalates() -> Result<()> {
        let judge = Queue::new([r#"{"escalate":true,"reason":"stuck in a loop"}"#]);
        let model = Queue::new(["efficient draft", "capable answer"]);

        let (selected_model, response) = test_drive(
            escalation_router()?,
            classify_request(),
            queued(model, judge),
        )
        .await?;

        assert_eq!(selected_model, "capable");
        assert_eq!(
            response.llm_response.as_agg().map(completion_text),
            Some("capable answer".to_string())
        );
        Ok(())
    }

    #[tokio::test]
    async fn stays_capable_after_latch() -> Result<()> {
        let judge = Queue::new([r#"{"escalate":true,"reason":"stuck"}"#]);
        let model = Queue::new(["efficient draft", "capable t1", "capable t2"]);
        let router = escalation_router()?;
        let request = classify_session_request();

        test_drive(
            router.clone(),
            request.clone(),
            queued(Arc::clone(&model), Arc::clone(&judge)),
        )
        .await?;
        let (selected_model, _) = test_drive(router, request, queued(model, judge)).await?;

        assert_eq!(selected_model, "capable");
        Ok(())
    }

    #[tokio::test]
    async fn deescalation_holds_then_returns_to_efficient() -> Result<()> {
        let judge = Queue::new([
            r#"{"escalate":true,"reason":"stuck"}"#,
            r#"{"escalate":false,"reason":"recovered"}"#,
            r#"{"escalate":false,"reason":"routine"}"#,
            r#"{"escalate":false,"reason":"progressing"}"#,
        ]);
        let model = Queue::new([
            "efficient draft",
            "capable t1",
            "capable t2",
            "capable t3",
            "capable t4",
            "efficient resumed",
        ]);
        let router = deescalation_router(DeescalationConfig {
            strong_min_calls: 3,
            strong_max_calls: None,
            confirmations: 2,
            weak_cooldown_calls: 0,
        })?;
        assert_eq!(
            selected_over(router, 5, model, judge).await?,
            ["capable", "capable", "capable", "capable", "efficient"].map(ModelId::from)
        );
        Ok(())
    }

    #[tokio::test]
    async fn deescalation_hard_limit_forces_a_weak_cooldown() -> Result<()> {
        let judge = Queue::new([
            r#"{"escalate":true,"reason":"stuck"}"#,
            r#"{"escalate":true,"reason":"still hard"}"#,
            r#"{"escalate":true,"reason":"still hard"}"#,
            r#"{"escalate":false,"reason":"progressing"}"#,
        ]);
        let model = Queue::new([
            "efficient draft",
            "capable t1",
            "capable t2",
            "capable t3",
            "efficient cooldown t1",
            "efficient cooldown t2",
            "efficient judged",
        ]);
        let router = deescalation_router(DeescalationConfig {
            strong_min_calls: 2,
            strong_max_calls: Some(3),
            confirmations: 2,
            weak_cooldown_calls: 2,
        })?;
        assert_eq!(
            selected_over(router, 6, model, judge).await?,
            [
                "capable",
                "capable",
                "capable",
                "efficient",
                "efficient",
                "efficient",
            ]
            .map(ModelId::from)
        );
        Ok(())
    }

    #[tokio::test]
    async fn strong_review_returns_an_efficient_fallback_without_judging_it() -> Result<()> {
        let judge = Queue::new([r#"{"escalate":true,"reason":"stuck"}"#]);
        let model = Queue::new(["efficient draft", "capable t1", "efficient fallback"]);
        let router = deescalation_router(DeescalationConfig {
            strong_min_calls: 1,
            strong_max_calls: None,
            confirmations: 1,
            weak_cooldown_calls: 0,
        })?;
        let request = classify_session_request();

        test_drive(
            router.clone(),
            request.clone(),
            queued(Arc::clone(&model), Arc::clone(&judge)),
        )
        .await?;
        let serve = move |target: ModelId, _request: Request| {
            let model = Arc::clone(&model);
            async move {
                assert_ne!(target, "judge", "fallback answer must not be judged");
                let mut response = reply(model.take());
                response.set_served_model(&ModelId::from("efficient"));
                Ok(response)
            }
        };

        let (selected_model, response) = test_drive(router, request, serve).await?;

        assert_eq!(selected_model, "efficient");
        assert_eq!(
            response.llm_response.as_agg().map(completion_text),
            Some("efficient fallback".to_string())
        );
        Ok(())
    }

    #[tokio::test]
    async fn falls_back_to_capable_when_efficient_overflows() -> Result<()> {
        let serve = |target: ModelId, _request: Request| async move {
            match target.as_str() {
                "efficient" => Err(LlmClientError::ContextWindowExceeded {
                    model: target,
                    message: "prompt is too long".to_string(),
                }),
                "judge" => panic!("the judge must not be consulted when efficient overflows"),
                _ => Ok(reply("capable answer")),
            }
        };

        let (selected_model, response) =
            test_drive(escalation_router()?, classify_request(), serve).await?;

        assert_eq!(selected_model, "capable");
        assert_eq!(
            response.llm_response.as_agg().map(completion_text),
            Some("capable answer".to_string())
        );
        Ok(())
    }

    /// A transport failure while buffering efficient must bypass the judge and serve capable.
    #[tokio::test]
    async fn falls_back_when_efficient_stream_transport_fails() -> Result<()> {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let serve = {
            let calls = Arc::clone(&calls);
            move |model: ModelId, _request: Request| {
                let calls = Arc::clone(&calls);
                async move {
                    let model = model.to_string();
                    calls.lock().push(model.clone());
                    match model.as_str() {
                        "efficient" => Ok(streamed_then_error(LlmClientError::Transport {
                            source: Box::new(std::io::Error::other("stream disconnected")),
                        })),
                        "judge" => {
                            panic!("the judge must not be consulted after a transport failure")
                        }
                        _ => Ok(reply("capable answer")),
                    }
                }
            }
        };
        let mut request = classify_request();
        request.llm_request.stream = true;

        let result = test_drive(escalation_router()?, request, serve).await;

        assert_eq!(&*calls.lock(), &["efficient", "capable"]);
        let (_, response) = result?;
        assert_eq!(
            response.llm_response.as_agg().map(completion_text),
            Some("capable answer".to_string())
        );
        Ok(())
    }

    /// Non-transport aggregation failures remain typed and do not silently change targets.
    #[tokio::test]
    async fn preserves_non_transport_stream_errors() -> Result<()> {
        let serve = |target: ModelId, _request: Request| async move {
            match target.as_str() {
                "efficient" => Ok(streamed_then_error(LlmClientError::InvalidResponse {
                    source: Box::new(std::io::Error::other("invalid stream event")),
                })),
                other => panic!("unexpected call to {other}"),
            }
        };
        let mut request = classify_request();
        request.llm_request.stream = true;

        match test_drive(escalation_router()?, request, serve).await {
            Err(LibsyError::ClientCall {
                target,
                source: LlmClientError::InvalidResponse { .. },
            }) => {
                assert_eq!(target, "efficient");
                Ok(())
            }
            Err(other) => panic!("expected InvalidResponse client error, got {other:?}"),
            Ok(_) => panic!("expected stream aggregation to fail"),
        }
    }
}
