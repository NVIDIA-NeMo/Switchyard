// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Node bindings for routing algorithms and buffered JSON translation.

mod algorithm;
mod translation;

use std::future::ready;
use std::sync::Arc;

use futures::StreamExt;
use futures::future::{AbortHandle, Abortable};
use napi_derive::napi;
use parking_lot::Mutex;
use serde::Deserialize;
use switchyard_libsy::{
    Algorithm, CallDecision, CapabilityJudgeConfig, ClassifyTrigger, CompositeJudgeConfig,
    CompositeRouter, DecisionJudgeConfig, LibsyError, PickerMode, RuntimeModels, StageRouterConfig,
    State, Step, StepStream, TaskClassifierConfig,
};
use switchyard_protocol::{
    Category, ContentBlock, DecisionResponse, LlmClientError, LlmRequest, ModelId, Request,
};

// The JS wrapper turns these prefixes into public Error codes.
fn error(code: &str, message: impl std::fmt::Display) -> napi::Error {
    napi::Error::from_reason(format!("{code}: {message}"))
}

fn routing_error(message: impl std::fmt::Display) -> napi::Error {
    error("ERR_ROUTING", message)
}

fn parse<T: serde::de::DeserializeOwned>(json: &str) -> napi::Result<T> {
    serde_json::from_str(json).map_err(|err| error("ERR_INVALID_REQUEST", err))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Options {
    targets: Targets,
    classifier: ClassifierOptions,
    stage: StageOptions,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Targets {
    capable: ModelId,
    efficient: ModelId,
    classifier: ModelId,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ClassifierOptions {
    classify_trigger: ClassifyTrigger,
    fail_open: bool,
    cutoff: Option<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StageOptions {
    efficient_first: StageProfile,
    capable_first: StageProfile,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StageProfile {
    threshold: f64,
    recent_window: Option<usize>,
    capable_hold_turns: Option<i64>,
    efficient_hold_turns: Option<i64>,
    capable_system_prompt: Option<String>,
}

impl StageProfile {
    fn config(self, mode: PickerMode) -> StageRouterConfig {
        let mut config = StageRouterConfig::new(mode, self.threshold);
        config.recent_window = self.recent_window;
        if let Some(turns) = self.capable_hold_turns {
            config.capable_hold_turns = turns;
        }
        config.efficient_hold_turns = self.efficient_hold_turns;
        config.capable_system_prompt = self.capable_system_prompt;
        config
    }
}

#[napi]
pub struct NativeComposite {
    router: Arc<CompositeRouter>,
    models: Arc<RuntimeModels>,
}

#[napi]
impl NativeComposite {
    #[napi(constructor)]
    pub fn new(options_json: String) -> napi::Result<Self> {
        let Options {
            targets,
            classifier,
            stage,
        } = serde_json::from_str(&options_json).map_err(|err| error("ERR_CONFIG", err))?;
        if targets.capable == targets.efficient {
            return Err(error(
                "ERR_CONFIG",
                "capable and efficient targets must differ",
            ));
        }
        // The judge's efficient tier selects profile 0 and its capable tier profile 1.
        let router = CompositeRouter::new_with_profiles(
            CompositeJudgeConfig::Capability(TaskClassifierConfig {
                judge: CapabilityJudgeConfig::Decision(DecisionJudgeConfig {
                    cutoff: classifier.cutoff,
                    ..Default::default()
                }),
                classify_trigger: classifier.classify_trigger,
                fail_open: classifier.fail_open,
                ..Default::default()
            }),
            stage.efficient_first.config(PickerMode::EfficientFirst),
            stage.capable_first.config(PickerMode::CapableFirst),
        )
        .map_err(|err| error("ERR_CONFIG", err))?;
        let models = RuntimeModels::new(
            [
                (Category::Judge, vec![targets.classifier]),
                (Category::Capable, vec![targets.capable.clone()]),
                (Category::Efficient, vec![targets.efficient.clone()]),
                (Category::Any, vec![targets.capable, targets.efficient]),
            ]
            .into(),
        );
        Ok(Self {
            router: Arc::new(router),
            models: Arc::new(models),
        })
    }

    #[napi]
    pub fn start(
        &self,
        messages_json: String,
        state_json: Option<String>,
    ) -> napi::Result<NativeRun> {
        let mut request = Request {
            llm_request: LlmRequest {
                model: Some("switchyard/auto".into()),
                messages: parse(&messages_json)?,
                ..Default::default()
            },
            ..Default::default()
        };
        switchyard_translation::normalize_compaction(&mut request.llm_request, None);
        let initial_state: State = state_json
            .as_deref()
            .map(parse)
            .transpose()?
            .unwrap_or_default();
        let state = initial_state.clone();
        let router = self.router.clone();
        let models = self.models.clone();
        // Defer run_stream until next() polls on napi's Tokio runtime.
        let stream =
            futures::stream::once(async move { router.run_stream(request, models, state) })
                .flatten();
        Ok(NativeRun {
            slot: Mutex::new(RunSlot {
                run: Some(Run {
                    initial_state,
                    stream: Box::pin(stream),
                    pending: None,
                }),
                abort: None,
            }),
        })
    }
}

#[napi(object, use_nullable = true)]
#[derive(Default)]
pub struct NativeStep {
    pub kind: String,
    pub request: Option<String>,
    pub model: Option<String>,
    pub state: Option<String>,
    pub system_prompt: Option<String>,
}

struct Run {
    initial_state: State,
    stream: StepStream,
    pending: Option<Box<CallDecision>>,
}

impl Run {
    async fn next(
        &mut self,
        response: Option<String>,
        failure: Option<String>,
    ) -> napi::Result<NativeStep> {
        if let Some(call) = self.pending.take() {
            let result = match (response, failure) {
                (Some(json), None) => Ok(parse::<DecisionResponse>(&json)?),
                (None, Some(message)) => Err(LibsyError::client_call(
                    call.model.clone(),
                    LlmClientError::General(message),
                )),
                _ => {
                    return Err(error(
                        "ERR_INVALID_REQUEST",
                        "Expected a decision response or error",
                    ));
                }
            };
            call.respond(ready(result)).await.map_err(routing_error)?;
        } else if response.is_some() || failure.is_some() {
            return Err(error("ERR_INVALID_REQUEST", "No decision is pending"));
        }
        let step = self
            .stream
            .next()
            .await
            .ok_or_else(|| error("ERR_ROUTING", "Routing ended without an outcome"))?
            .map_err(routing_error)?;
        let mut result = NativeStep::default();
        match step {
            Step::CallDecision(call) => {
                result.kind = "decision".into();
                result.request = Some(serde_json::to_string(&call.request).map_err(routing_error)?);
                self.pending = Some(call);
            }
            Step::Done(outcome) => {
                result.kind = "done".into();
                result.model = Some(
                    outcome
                        .selected_model_id()
                        .map_err(routing_error)?
                        .to_string(),
                );
                // `start` sends no instructions, so a first instruction block is the
                // routing prompt that libsy prepended for the selected model.
                result.system_prompt = outcome
                    .request
                    .llm_request
                    .instructions
                    .into_iter()
                    .next()
                    .and_then(|block| {
                        block.content.into_iter().find_map(|content| match content {
                            ContentBlock::Text { text } => Some(text),
                            _ => None,
                        })
                    });
                if outcome.state != self.initial_state {
                    result.state =
                        Some(serde_json::to_string(&outcome.state).map_err(routing_error)?);
                }
            }
            Step::CallModel(_) => return Err(error("ERR_ROUTING", "Unexpected model call")),
        }
        Ok(result)
    }
}

struct RunSlot {
    run: Option<Run>,
    abort: Option<AbortHandle>,
}

#[napi]
pub struct NativeRun {
    slot: Mutex<RunSlot>,
}

#[napi]
impl NativeRun {
    #[napi]
    pub async fn next(
        &self,
        response_json: Option<String>,
        error_message: Option<String>,
    ) -> napi::Result<NativeStep> {
        let (abort, registration) = AbortHandle::new_pair();
        let mut run = {
            let mut slot = self.slot.lock();
            let run = slot.run.take().ok_or_else(|| {
                error(
                    "ERR_INVALID_REQUEST",
                    "Run is closed or next is already pending",
                )
            })?;
            slot.abort = Some(abort.clone());
            run
        };
        // Move ownership into the future so close() also drops an in-flight stream.
        let (run, result) = Abortable::new(
            async move {
                let result = run.next(response_json, error_message).await;
                (run, result)
            },
            registration,
        )
        .await
        .map_err(|_| error("ABORT_ERR", "Routing cancelled"))?;
        let mut slot = self.slot.lock();
        if abort.is_aborted() {
            return Err(error("ABORT_ERR", "Routing cancelled"));
        }
        slot.abort = None;
        if result.as_ref().is_ok_and(|step| step.kind == "decision") {
            slot.run = Some(run);
        }
        result
    }

    #[napi]
    pub fn close(&self) {
        let mut slot = self.slot.lock();
        if let Some(abort) = slot.abort.take() {
            abort.abort();
        }
        slot.run = None;
    }
}
