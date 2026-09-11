// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Escalation routing that judges an efficient model's answer before selecting a serving tier.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use switchyard_protocol::{
    AggLlmResponse, LlmClientError, LlmResponse, Message, ModelId, Request, Response, Role,
};

use super::llm_class;
use super::util::classifier_contract::ClassifierContractConfig;
use super::util::decisive;
use super::util::escalation::{self, EscalationJudge, EscalationJudgeConfig, EscalationPolicy};
use super::util::llm_judge::JudgeClassifier;
use super::util::prompts;
use crate::core::algorithm::Driver;
use crate::core::classifier::{Classification, Classifier};
use crate::core::state::{State, StateValue};
use crate::{LibsyError, Result};

/// Session-state key holding the consecutive-escalate streak.
const STREAK_KEY: &str = "escalation_streak";
/// Session-state key recording that the up-front capability gate has run for this session.
const GATE_KEY: &str = "escalation_gate_done";

fn streak(state: &State) -> u32 {
    match state.extra.get(STREAK_KEY) {
        Some(StateValue::Count(n)) => *n,
        _ => 0,
    }
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
    judge: JudgeClassifier<EscalationJudge, EscalationPolicy>,
    /// Optional capability forecast run once per session before the first efficient call.
    gate: Option<Arc<dyn Classifier<State>>>,
    capable: ModelId,
    efficient: ModelId,
    /// Consecutive escalate verdicts required to latch.
    confirmations: u32,
    /// Note spliced into every turn the judge has sent to the capable tier.
    handoff_note: Option<String>,
}

impl EscalationClassifier {
    /// Hands the capable tier the configured note. Only judge-driven turns call this; a
    /// fallback to capable is not a verdict and must not tell the model the other tier failed.
    fn apply_handoff_note(&self, request: &mut Request) {
        if let Some(note) = &self.handoff_note {
            prompts::append_note(request, note);
        }
    }
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
    let handoff_note = config.handoff_note.clone();
    let gate = match &config.gate {
        Some(gate) => Some(llm_class::build_capability_gate(
            judge_target.clone(),
            efficient_target,
            capable_target,
            gate,
            contract_config.response_format_type(),
            max_output_tokens,
        )?),
        None => None,
    };
    let classifier: Arc<dyn Classifier<State>> = Arc::new(EscalationClassifier {
        gate,
        judge: escalation::build_judge(
            judge_target,
            capable_target.clone(),
            efficient_target.clone(),
            &contract_config,
            config,
            max_output_tokens,
        )?,
        capable: capable_target.clone(),
        efficient: efficient_target.clone(),
        confirmations,
        handoff_note,
    });
    Ok(classifier)
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

        // A confirmed session stays capable without a judge call.
        if streak(state) >= self.confirmations {
            driver.set_evidence(serde_json::json!({
                "source": "escalation",
                "verdict": "latched",
            }));
            self.apply_handoff_note(request);
            return Ok((decisive(&self.capable), None));
        }

        // The gate forecasts once per session, from the task framing alone, before the efficient
        // tier spends anything. A verdict below the threshold latches immediately; anything else
        // (including an unusable verdict or a failed call) falls open to the efficient tier and
        // leaves the trajectory judge in charge.
        if let Some(gate) = &self.gate
            && !matches!(state.extra.get(GATE_KEY), Some(StateValue::Count(_)))
        {
            state
                .extra
                .insert(GATE_KEY.to_string(), StateValue::Count(1));
            let mut gate_request = request.clone();
            match gate.score(state, &mut gate_request, Some(driver)).await {
                Ok((classification, _)) => {
                    let to_capable = classification
                        .argmax(false)?
                        .is_some_and(|score| score.target == self.capable);
                    // The forecaster's evidence carries `score` (p_solve) and `threshold`; log it
                    // so an operator can read the forecast distribution and tune the threshold.
                    let forecast = driver.evidence().unwrap_or(Value::Null);
                    tracing::info!(
                        latched = to_capable,
                        forecast = %forecast,
                        "escalation gate verdict"
                    );
                    if to_capable {
                        state.extra.insert(
                            STREAK_KEY.to_string(),
                            StateValue::Count(self.confirmations),
                        );
                        tracing::info!(
                            target = %self.capable,
                            "escalation gate latched the session to the capable tier"
                        );
                        let mut evidence = serde_json::json!({
                            "source": "escalation",
                            "verdict": "gate",
                        });
                        if let (Some(evidence), Some(forecast)) =
                            (evidence.as_object_mut(), forecast.as_object())
                        {
                            for key in ["score", "threshold"] {
                                if let Some(value) = forecast.get(key) {
                                    evidence.insert(key.to_string(), value.clone());
                                }
                            }
                        }
                        driver.set_evidence(evidence);
                        return Ok((decisive(&self.capable), None));
                    }
                }
                Err(error) => {
                    tracing::warn!(%error, "escalation gate failed; serving the efficient tier");
                }
            }
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

        let (classification, _) = self
            .judge
            .score(state, &mut judge_request, Some(driver))
            .await?;

        let held = streak(state);
        let best = classification.argmax(false)?;
        let (escalate, pending) = match &best {
            Some(score) if score.target == self.capable => (true, held + 1),
            Some(_) => (false, 0),
            None => (false, held),
        };
        state
            .extra
            .insert(STREAK_KEY.to_string(), StateValue::Count(pending));

        if escalate && pending >= self.confirmations {
            // Streak confirmed: drop the efficient response, caller will serve capable.
            driver.set_evidence(serde_json::json!({
                "source": "escalation",
                "verdict": "escalate",
            }));
            self.apply_handoff_note(request);
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
    use crate::algorithms::util::escalation::EscalationGateConfig;
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
        escalation_router_with(EscalationJudgeConfig {
            confirmations: 1,
            ..EscalationJudgeConfig::default()
        })
    }

    fn escalation_router_with(config: EscalationJudgeConfig) -> Result<Arc<LlmTaskClassifier>> {
        Ok(Arc::new(LlmTaskClassifier::new(
            LlmClassifierConfig::Escalation {
                judge_target: ModelId::from("judge"),
                efficient_target: ModelId::from("efficient"),
                capable_target: ModelId::from("capable"),
                contract: ClassifierContractConfig::default(),
                config,
                max_output_tokens: DEFAULT_JUDGE_MAX_OUTPUT_TOKENS,
            },
        )?))
    }

    /// The handoff note reaches the capable tier on the latching turn and on every confirmed
    /// turn after it, and never reaches the efficient tier or the judge.
    #[tokio::test]
    async fn handoff_note_reaches_only_the_capable_tier() -> Result<()> {
        const NOTE: &str = "You are taking over mid-task; verify before continuing.";
        let seen = Arc::new(Mutex::new(Vec::new()));
        let serve = {
            let seen = Arc::clone(&seen);
            move |model: ModelId, request: Request| {
                let seen = Arc::clone(&seen);
                async move {
                    let noted = request.llm_request.messages.iter().any(|message| {
                        message.role == switchyard_protocol::Role::User
                            && message.content.iter().any(|block| {
                                matches!(block, ContentBlock::Text { text } if text.contains(NOTE))
                            })
                    });
                    let model = model.to_string();
                    seen.lock().push((model.clone(), noted));
                    Ok(match model.as_str() {
                        "judge" => reply(r#"{"escalate":true,"reason":"stuck"}"#),
                        "efficient" => reply("efficient draft"),
                        _ => reply("capable answer"),
                    })
                }
            }
        };
        let router = escalation_router_with(EscalationJudgeConfig {
            confirmations: 1,
            handoff_note: Some(NOTE.to_string()),
            ..EscalationJudgeConfig::default()
        })?;
        let request = classify_session_request();

        test_drive(router.clone(), request.clone(), serve.clone()).await?;
        let (selected_model, _) = test_drive(router, request, serve).await?;

        assert_eq!(selected_model, "capable");
        assert_eq!(
            &*seen.lock(),
            &[
                ("efficient".to_string(), false),
                ("judge".to_string(), false),
                ("capable".to_string(), true),
                ("capable".to_string(), true),
            ]
        );
        Ok(())
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

    /// Builds a router with the up-front capability gate at the given threshold.
    fn gated_router(base_threshold: f64) -> Result<Arc<LlmTaskClassifier>> {
        Ok(Arc::new(LlmTaskClassifier::new(
            LlmClassifierConfig::Escalation {
                judge_target: ModelId::from("judge"),
                efficient_target: ModelId::from("efficient"),
                capable_target: ModelId::from("capable"),
                contract: ClassifierContractConfig::default(),
                config: EscalationJudgeConfig {
                    confirmations: 1,
                    gate: Some(EscalationGateConfig {
                        base_threshold,
                        threshold_step: 0.0,
                        prompt: None,
                    }),
                    ..EscalationJudgeConfig::default()
                },
                max_output_tokens: DEFAULT_JUDGE_MAX_OUTPUT_TOKENS,
            },
        )?))
    }

    const GATE_LOW: &str = r#"{"crux":"protocol framing","primary_rule":"LIM-2","capability_boundary":"unsupported","p_solve":0.2}"#;
    const GATE_HIGH: &str = r#"{"crux":"local change","primary_rule":"SUP-1","capability_boundary":"supported","p_solve":0.9}"#;

    #[tokio::test]
    async fn gate_latches_before_the_efficient_tier_is_called() -> Result<()> {
        // Only the gate verdict is queued for the judge and only one model reply exists: had the
        // efficient tier been called first, capable would have received "unexpected call".
        let judge = Queue::new([GATE_LOW]);
        let model = Queue::new(["capable answer"]);

        let (selected_model, response) = test_drive(
            gated_router(0.5)?,
            classify_session_request(),
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
    async fn gate_passes_then_trajectory_judge_decides() -> Result<()> {
        let judge = Queue::new([GATE_HIGH, r#"{"escalate":false,"reason":"progressing"}"#]);
        let model = Queue::new(["efficient answer"]);

        let (selected_model, response) = test_drive(
            gated_router(0.5)?,
            classify_session_request(),
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
    async fn gate_runs_once_per_session() -> Result<()> {
        // Second request: the judge queue holds only an escalate verdict. If the gate ran again
        // it would consume that verdict (invalid for the gate, so fail-open) and the trajectory
        // judge would then read "unexpected call" and keep efficient.
        let judge = Queue::new([
            GATE_HIGH,
            r#"{"escalate":false,"reason":"progressing"}"#,
            r#"{"escalate":true,"reason":"stuck"}"#,
        ]);
        let model = Queue::new(["efficient t1", "efficient t2", "capable t2"]);
        let router = gated_router(0.5)?;
        let request = classify_session_request();

        let (first, _) = test_drive(
            router.clone(),
            request.clone(),
            queued(Arc::clone(&model), Arc::clone(&judge)),
        )
        .await?;
        let (second, _) = test_drive(router, request, queued(model, judge)).await?;

        assert_eq!(first, "efficient");
        assert_eq!(second, "capable");
        Ok(())
    }

    #[tokio::test]
    async fn unusable_gate_verdict_falls_open_to_efficient() -> Result<()> {
        let judge = Queue::new([
            "not a verdict",
            r#"{"escalate":false,"reason":"progressing"}"#,
        ]);
        let model = Queue::new(["efficient answer"]);

        let (selected_model, _) = test_drive(
            gated_router(0.5)?,
            classify_session_request(),
            queued(model, judge),
        )
        .await?;

        assert_eq!(selected_model, "efficient");
        Ok(())
    }

    #[test]
    fn gate_threshold_is_validated() {
        let build = |base_threshold: f64| {
            LlmTaskClassifier::new(LlmClassifierConfig::Escalation {
                judge_target: ModelId::from("judge"),
                efficient_target: ModelId::from("efficient"),
                capable_target: ModelId::from("capable"),
                contract: ClassifierContractConfig::default(),
                config: EscalationJudgeConfig {
                    gate: Some(EscalationGateConfig {
                        base_threshold,
                        threshold_step: 0.0,
                        prompt: None,
                    }),
                    ..EscalationJudgeConfig::default()
                },
                max_output_tokens: DEFAULT_JUDGE_MAX_OUTPUT_TOKENS,
            })
        };
        assert!(build(0.4).is_ok());
        assert!(build(1.5).is_err());
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
