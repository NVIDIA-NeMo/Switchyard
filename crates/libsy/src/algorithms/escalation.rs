// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Escalation routing that judges an efficient model's answer before selecting a serving tier.

use std::sync::Arc;

use async_trait::async_trait;
use switchyard_protocol::{
    AggLlmResponse, Category, LlmClientError, Message, Request, Response, Role,
};

use super::util::buffered_response::buffer_response;
use super::util::classifier_contract::ClassifierContractConfig;
use super::util::decisive;
use super::util::escalation::{
    self, EscalationCategory, EscalationJudge, EscalationJudgeConfig, EscalationPolicy,
};
use super::util::llm_judge::JudgeClassifier;
use crate::core::algorithm::Driver;
use crate::core::classifier::{Classification, Classifier};
use crate::core::state::{State, StateValue};
use crate::{LibsyError, Result};

/// Session-state key holding the consecutive-escalate streak.
const STREAK_KEY: &str = "escalation_streak";
/// Session-state key holding the category currently being confirmed.
const CATEGORY_KEY: &str = "escalation_category";

fn streak(state: &State) -> u32 {
    match state.extra.get(STREAK_KEY) {
        Some(StateValue::Count(n)) => *n,
        _ => 0,
    }
}

fn category(state: &State) -> Option<&str> {
    match state.extra.get(CATEGORY_KEY) {
        Some(StateValue::String(category)) => Some(category),
        _ => None,
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
    /// Consecutive escalate verdicts required to latch.
    confirmations: u32,
}

/// Builds the escalation classifier used by the shared LLM classifier route shell.
pub(super) fn build_classifier(
    contract_config: ClassifierContractConfig,
    config: EscalationJudgeConfig,
    max_output_tokens: u64,
) -> Result<Arc<dyn Classifier<State>>> {
    let confirmations = config.confirmations;
    let classifier: Arc<dyn Classifier<State>> = Arc::new(EscalationClassifier {
        judge: escalation::build_judge(&contract_config, config, max_output_tokens)?,
        confirmations,
    });
    Ok(classifier)
}

#[async_trait]
impl Classifier<State> for EscalationClassifier {
    async fn score(
        &self,
        state: &mut State,
        request: &mut Request,
        driver: &Driver,
    ) -> Result<(Classification, Option<Response>)> {
        let capable = driver.first_model_for(&Category::Capable)?.clone();
        let efficient = driver.first_model_for(&Category::Efficient)?.clone();

        // A confirmed session stays capable without a judge call.
        if streak(state) >= self.confirmations {
            driver.set_evidence(serde_json::json!({
                "source": "escalation",
                "verdict": "latched",
            }));
            return Ok((decisive(&capable), None));
        }

        // Call efficient model and buffer the response so the judge can read it.
        //
        // If the efficient model exceeds its context window, fall through to capable. This call
        // deliberately has one candidate so the classifier sees the efficient model's error.
        tracing::info!(
            target = %efficient,
            "escalation classifier selected efficient tier"
        );
        let efficient_response = match driver
            .call_model(request.clone(), vec![efficient.clone()])
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
                return Ok((decisive(&capable), None));
            }
            Err(e) => return Err(e),
        };
        // The call resolves when its stream handle arrives; context and transport failures can
        // still occur while buffering.
        let efficient_response = match buffer_response(efficient.as_str(), efficient_response).await
        {
            Ok(response) => response,
            Err(LibsyError::ClientCall {
                source: LlmClientError::ContextWindowExceeded { .. },
                ..
            }) => {
                driver.set_evidence(serde_json::json!({
                    "source": "fallback",
                    "reason_code": "context_window",
                }));
                return Ok((decisive(&capable), None));
            }
            Err(LibsyError::ClientCall {
                source: LlmClientError::Transport { .. },
                ..
            }) => {
                driver.set_evidence(serde_json::json!({
                    "source": "fallback",
                    "reason_code": "transport",
                }));
                return Ok((decisive(&capable), None));
            }
            Err(error) => return Err(error),
        };
        // Append the efficient reply so the judge reads this turn's completed trajectory.
        let mut judge_request = request.clone();
        judge_request
            .llm_request
            .messages
            .push(assistant_message(&efficient_response.agg));

        let judge_models = driver.models_for(&Category::Judge);
        if judge_models.is_empty() {
            return Err(LibsyError::AlgorithmError {
                message: "no models available for category Judge".to_string(),
            });
        }
        let verdict = self
            .judge
            .verdict(state, &judge_request, driver, judge_models)
            .await;

        let held = streak(state);
        let held_category = category(state).map(str::to_string);
        let (escalate, pending, pending_category) = match verdict.as_ref() {
            Some(verdict) => {
                let category = verdict.category.label();
                tracing::info!(
                    escalate = verdict.escalate,
                    category,
                    new_evidence = verdict.new_evidence,
                    "escalation judge verdict"
                );
                if verdict.escalate
                    && verdict.new_evidence
                    && verdict.category != EscalationCategory::None
                {
                    let next = if held_category.as_deref() == Some(category) {
                        held + 1
                    } else {
                        1
                    };
                    (true, next, Some(category.to_string()))
                } else {
                    (false, 0, None)
                }
            }
            None => (false, held, held_category),
        };
        state
            .extra
            .insert(STREAK_KEY.to_string(), StateValue::Count(pending));
        if let Some(category) = pending_category {
            state
                .extra
                .insert(CATEGORY_KEY.to_string(), StateValue::String(category));
        } else {
            state.extra.remove(CATEGORY_KEY);
        }

        if escalate && pending >= self.confirmations {
            // Streak confirmed: drop the efficient response, caller will serve capable.
            driver.set_evidence(serde_json::json!({
                "source": "escalation",
                "verdict": "escalate",
            }));
            return Ok((decisive(&capable), None));
        }

        if escalate {
            driver.set_evidence(serde_json::json!({
                "source": "escalation",
                "verdict": "pending",
            }));
        }

        Ok((
            decisive(&efficient),
            Some(efficient_response.into_response()),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, VecDeque};
    use std::sync::Arc;

    use parking_lot::Mutex;
    use switchyard_protocol::{
        ContentBlock, LlmClientError, LlmResponse, LlmResponseChunk, Metadata, ModelId, Request,
        Response, completion_text, text_request, text_response,
    };

    use super::*;
    use crate::algorithms::llm_class::{LlmClassifierConfig, LlmTaskClassifier};
    use crate::algorithms::util::DEFAULT_JUDGE_MAX_OUTPUT_TOKENS;
    use crate::core::testing::{Serve, reply, test_drive_with_models};

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
                    upstream_headers: http::HeaderMap::new(),
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

    fn runtime_models() -> HashMap<Category, Vec<ModelId>> {
        [
            (Category::Judge, vec![ModelId::from("judge")]),
            (Category::Efficient, vec![ModelId::from("efficient")]),
            (Category::Capable, vec![ModelId::from("capable")]),
            (
                Category::Any,
                vec![ModelId::from("capable"), ModelId::from("efficient")],
            ),
        ]
        .into()
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
            upstream_headers: http::HeaderMap::new(),
        }
    }

    /// Builds a router with escalation enabled.
    fn escalation_router_with_confirmations(confirmations: u32) -> Result<Arc<LlmTaskClassifier>> {
        Ok(Arc::new(LlmTaskClassifier::new(
            LlmClassifierConfig::Escalation {
                contract: ClassifierContractConfig::default(),
                config: EscalationJudgeConfig {
                    confirmations,
                    ..EscalationJudgeConfig::default()
                },
                max_output_tokens: DEFAULT_JUDGE_MAX_OUTPUT_TOKENS,
            },
        )?))
    }

    /// Builds a router that latches on its first supported escalation verdict.
    fn escalation_router() -> Result<Arc<LlmTaskClassifier>> {
        escalation_router_with_confirmations(1)
    }

    #[tokio::test]
    async fn serves_efficient_when_judge_declines() -> Result<()> {
        let judge = Queue::new([
            r#"{"escalate":false,"category":"none","new_evidence":false,"reason":"progressing"}"#,
        ]);
        let model = Queue::new(["efficient answer"]);

        let (selected_model, response) = test_drive_with_models(
            escalation_router()?,
            classify_request(),
            runtime_models(),
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
                std::future::ready(Ok(reply(
                    r#"{"escalate":false,"category":"none","new_evidence":false,"reason":"progressing"}"#,
                )))
            } else {
                std::future::ready(Ok(reply("efficient answer")))
            }
        };
        let router = Arc::new(LlmTaskClassifier::new(LlmClassifierConfig::Escalation {
            contract: ClassifierContractConfig::default().with_prompt("Custom trajectory rubric."),
            config: EscalationJudgeConfig {
                confirmations: 1,
                ..EscalationJudgeConfig::default()
            },
            max_output_tokens: DEFAULT_JUDGE_MAX_OUTPUT_TOKENS,
        })?);

        test_drive_with_models(router, classify_request(), runtime_models(), serve).await?;

        assert_eq!(&*prompts.lock(), &["Custom trajectory rubric."]);
        Ok(())
    }

    #[tokio::test]
    async fn upgrades_to_capable_when_judge_escalates() -> Result<()> {
        let judge = Queue::new([
            r#"{"escalate":true,"category":"repetition","new_evidence":true,"reason":"stuck in a loop"}"#,
        ]);
        let model = Queue::new(["efficient draft", "capable answer"]);

        let (selected_model, response) = test_drive_with_models(
            escalation_router()?,
            classify_request(),
            runtime_models(),
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
    async fn confirmation_streak_requires_the_same_category() -> Result<()> {
        let judge = Queue::new([
            r#"{"escalate":true,"category":"repetition","new_evidence":true,"reason":"repeated command"}"#,
            r#"{"escalate":true,"category":"drift","new_evidence":true,"reason":"off task"}"#,
            r#"{"escalate":true,"category":"drift","new_evidence":true,"reason":"still off task"}"#,
        ]);
        let model = Queue::new(["efficient t1", "efficient t2", "efficient t3", "capable t3"]);
        let router = escalation_router_with_confirmations(2)?;
        let request = classify_session_request();

        let (first, _) = test_drive_with_models(
            router.clone(),
            request.clone(),
            runtime_models(),
            queued(Arc::clone(&model), Arc::clone(&judge)),
        )
        .await?;
        let (second, _) = test_drive_with_models(
            router.clone(),
            request.clone(),
            runtime_models(),
            queued(Arc::clone(&model), Arc::clone(&judge)),
        )
        .await?;
        let (third, _) =
            test_drive_with_models(router, request, runtime_models(), queued(model, judge)).await?;

        assert_eq!(first, "efficient");
        assert_eq!(second, "efficient");
        assert_eq!(third, "capable");
        Ok(())
    }

    #[tokio::test]
    async fn verdict_without_new_evidence_resets_the_streak() -> Result<()> {
        let judge = Queue::new([
            r#"{"escalate":true,"category":"repetition","new_evidence":true,"reason":"repeated command"}"#,
            r#"{"escalate":true,"category":"repetition","new_evidence":false,"reason":"only old evidence remains"}"#,
            r#"{"escalate":true,"category":"repetition","new_evidence":true,"reason":"new repeated command"}"#,
        ]);
        let model = Queue::new(["efficient t1", "efficient t2", "efficient t3"]);
        let router = escalation_router_with_confirmations(2)?;
        let request = classify_session_request();

        for _ in 0..3 {
            let (selected, _) = test_drive_with_models(
                router.clone(),
                request.clone(),
                runtime_models(),
                queued(Arc::clone(&model), Arc::clone(&judge)),
            )
            .await?;
            assert_eq!(selected, "efficient");
        }
        Ok(())
    }

    #[tokio::test]
    async fn unavailable_judge_preserves_the_category_streak() -> Result<()> {
        let judge = Queue::new([
            r#"{"escalate":true,"category":"repetition","new_evidence":true,"reason":"repeated command"}"#,
            "not json",
            r#"{"escalate":true,"category":"repetition","new_evidence":true,"reason":"another repeated command"}"#,
        ]);
        let model = Queue::new(["efficient t1", "efficient t2", "efficient t3", "capable t3"]);
        let router = escalation_router_with_confirmations(2)?;
        let request = classify_session_request();

        let (first, _) = test_drive_with_models(
            router.clone(),
            request.clone(),
            runtime_models(),
            queued(Arc::clone(&model), Arc::clone(&judge)),
        )
        .await?;
        let (second, _) = test_drive_with_models(
            router.clone(),
            request.clone(),
            runtime_models(),
            queued(Arc::clone(&model), Arc::clone(&judge)),
        )
        .await?;
        let (third, _) =
            test_drive_with_models(router, request, runtime_models(), queued(model, judge)).await?;

        assert_eq!(first, "efficient");
        assert_eq!(second, "efficient");
        assert_eq!(third, "capable");
        Ok(())
    }

    #[tokio::test]
    async fn stays_capable_after_latch() -> Result<()> {
        let judge = Queue::new([
            r#"{"escalate":true,"category":"repetition","new_evidence":true,"reason":"stuck"}"#,
        ]);
        let model = Queue::new(["efficient draft", "capable t1", "capable t2"]);
        let router = escalation_router()?;
        let request = classify_session_request();

        test_drive_with_models(
            router.clone(),
            request.clone(),
            runtime_models(),
            queued(Arc::clone(&model), Arc::clone(&judge)),
        )
        .await?;
        let (selected_model, _) =
            test_drive_with_models(router, request, runtime_models(), queued(model, judge)).await?;

        assert_eq!(selected_model, "capable");
        Ok(())
    }

    #[tokio::test]
    async fn falls_back_to_capable_when_efficient_overflows() -> Result<()> {
        for streamed in [false, true] {
            let serve = move |target: ModelId, _request: Request| async move {
                match target.as_str() {
                    "efficient" if streamed => {
                        Ok(streamed_then_error(LlmClientError::ContextWindowExceeded {
                            model: target,
                            message: "prompt is too long".to_string(),
                        }))
                    }
                    "efficient" => Err(LlmClientError::ContextWindowExceeded {
                        model: target,
                        message: "prompt is too long".to_string(),
                    }),
                    "judge" => {
                        panic!("the judge must not be consulted when efficient overflows")
                    }
                    _ => Ok(reply("capable answer")),
                }
            };
            let mut request = classify_request();
            request.llm_request.stream = streamed;

            let (selected_model, response) =
                test_drive_with_models(escalation_router()?, request, runtime_models(), serve)
                    .await?;

            assert_eq!(selected_model, "capable");
            assert_eq!(
                response.llm_response.as_agg().map(completion_text),
                Some("capable answer".to_string())
            );
        }
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

        let result =
            test_drive_with_models(escalation_router()?, request, runtime_models(), serve).await;

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

        match test_drive_with_models(escalation_router()?, request, runtime_models(), serve).await {
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
