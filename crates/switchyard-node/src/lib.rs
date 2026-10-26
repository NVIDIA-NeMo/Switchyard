// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Routing algorithms with caller-owned model calls and state.

use std::collections::BTreeMap;
use std::future::ready;
use std::sync::Arc;

use futures::StreamExt;
use futures::future::{AbortHandle, Abortable};
use napi_derive::napi;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use switchyard_libsy::{
    Algorithm, Call, CapabilityJudgeConfig, ClassifyTrigger, CompositeJudgeConfig, CompositeRouter,
    CompositeRouterConfig, DecisionJudgeConfig, LibsyError, PickerMode, RuntimeModels,
    StageRouterConfig, State, Step, StepStream, TaskClassifierConfig,
};
use switchyard_protocol::{
    AggLlmResponse, Category, DecisionResponse, LlmClientError, LlmRequest, LlmResponse, Metadata,
    ModelId, Request, Response,
};
use switchyard_runner::AlgorithmFactory;

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

#[napi]
/// Constructs configured algorithms without provider clients or credentials.
pub struct NativeAlgorithmFactory {
    factory: AlgorithmFactory,
}

#[napi]
impl NativeAlgorithmFactory {
    /// Parses deployment TOML; routing configuration is checked by [`Self::build`].
    #[napi(constructor)]
    pub fn new(source: String) -> napi::Result<Self> {
        Ok(Self {
            factory: AlgorithmFactory::from_toml(&source)
                .map_err(|err| error("ERR_CONFIG", err))?,
        })
    }

    /// Builds reusable algorithm handles with their configured route IDs.
    #[napi]
    pub fn build(&self) -> napi::Result<Vec<NativeConfiguredAlgorithm>> {
        let routes = self
            .factory
            .build()
            .map_err(|err| error("ERR_CONFIG", err))?;
        Ok(routes
            .into_iter()
            .map(|(id, configured)| NativeConfiguredAlgorithm {
                id,
                algorithm: configured.algorithm,
                models: Arc::new(configured.models),
            })
            .collect())
    }
}

#[napi]
/// An algorithm and its ordered model groups, shared across runs.
pub struct NativeConfiguredAlgorithm {
    id: String,
    algorithm: Arc<dyn Algorithm>,
    models: Arc<RuntimeModels>,
}

#[napi]
impl NativeConfiguredAlgorithm {
    /// The public route ID from the deployment TOML.
    #[napi(getter)]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Starts routing a normalized request with caller-owned state and model calls.
    #[napi]
    pub fn start(
        &self,
        request_json: String,
        state_json: Option<String>,
        headers_json: Option<String>,
    ) -> napi::Result<NativeRun> {
        let metadata = headers_json
            .as_deref()
            .map(|json| -> napi::Result<Metadata> {
                let mut headers = http::HeaderMap::new();
                for (name, value) in parse::<BTreeMap<String, String>>(json)? {
                    headers.insert(
                        http::HeaderName::from_bytes(name.as_bytes())
                            .map_err(|err| error("ERR_INVALID_REQUEST", err))?,
                        http::HeaderValue::from_str(&value)
                            .map_err(|err| error("ERR_INVALID_REQUEST", err))?,
                    );
                }
                Ok(Metadata::from_headers(&headers))
            })
            .transpose()?;
        let request = Request {
            llm_request: parse(&request_json)?,
            metadata,
            ..Default::default()
        };
        NativeRun::new(
            self.algorithm.clone(),
            request,
            self.models.clone(),
            state_json.as_deref(),
            true,
        )
    }
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
    decision: DecisionOptions,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DecisionOptions {
    cutoff: f64,
    candidates: BTreeMap<String, Candidate>,
    evidence: Value,
    instructions: Option<Value>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Candidate {
    Capable,
    Efficient,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StageOptions {
    efficient_first_threshold: f64,
    capable_first_threshold: f64,
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
        let decision = classifier.decision;
        let candidates = decision
            .candidates
            .into_iter()
            .map(|(id, tier)| {
                let model = match tier {
                    Candidate::Capable => &targets.capable,
                    Candidate::Efficient => &targets.efficient,
                };
                (id, model.clone())
            })
            .collect();
        let mut stage_config =
            StageRouterConfig::new(PickerMode::EfficientFirst, stage.efficient_first_threshold);
        stage_config.capable_first_threshold = stage.capable_first_threshold;
        let router = CompositeRouter::new(CompositeRouterConfig {
            judge: CompositeJudgeConfig::Capability(TaskClassifierConfig {
                judge: CapabilityJudgeConfig::Decision(DecisionJudgeConfig {
                    cutoff: decision.cutoff,
                    candidates,
                    evidence: decision.evidence,
                    instructions: decision.instructions,
                }),
                classify_trigger: classifier.classify_trigger,
                fail_open: classifier.fail_open,
                ..Default::default()
            }),
            stage: stage_config,
        })
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
        let request = Request {
            llm_request: LlmRequest {
                model: Some("switchyard/auto".into()),
                messages: parse(&messages_json)?,
                ..Default::default()
            },
            ..Default::default()
        };
        NativeRun::new(
            self.router.clone(),
            request,
            self.models.clone(),
            state_json.as_deref(),
            false,
        )
    }
}

#[napi(object, use_nullable = true)]
#[derive(Default)]
pub struct NativeStep {
    pub kind: String,
    pub request: Option<String>,
    pub model: Option<String>,
    /// Ordered candidates for a model-call step.
    pub models: Option<Vec<String>>,
    pub state: Option<String>,
    /// Full normalized outcome for configured algorithms.
    pub outcome: Option<String>,
}

#[derive(Serialize)]
struct OutcomeView<'a> {
    selected_model_ids: &'a [ModelId],
    request: &'a LlmRequest,
    response: Option<&'a AggLlmResponse>,
    metadata: Option<OutcomeMetadataView<'a>>,
    state: &'a State,
}

#[derive(Serialize)]
struct OutcomeMetadataView<'a> {
    outcome_id: &'a str,
    algorithm: &'a str,
    evidence: &'a Option<Value>,
}

#[derive(Deserialize)]
struct ModelReply {
    response: AggLlmResponse,
    #[serde(rename = "servedModel")]
    served_model: ModelId,
}

struct Run {
    // Composite's compact result reports state only when it changed.
    initial_state: Option<State>,
    stream: StepStream,
    pending: Option<Call>,
}

impl Run {
    async fn next(
        &mut self,
        response: Option<String>,
        failure: Option<String>,
    ) -> napi::Result<NativeStep> {
        if let Some(call) = self.pending.take() {
            match call {
                Call::Decision(call) => {
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
                }
                Call::Model(call) => {
                    let result = match (response, failure) {
                        (Some(json), None) => {
                            let reply: ModelReply = parse(&json)?;
                            if !call.models.contains(&reply.served_model) {
                                return Err(error(
                                    "ERR_INVALID_REQUEST",
                                    "servedModel must be one of the model call's candidates",
                                ));
                            }
                            Ok(Response {
                                llm_response: LlmResponse::Agg(reply.response),
                                metadata: Some(Metadata {
                                    served_model: Some(reply.served_model),
                                    ..Default::default()
                                }),
                                upstream_headers: http::HeaderMap::new(),
                            })
                        }
                        (None, Some(message)) => Err(LibsyError::client_call(
                            call.models
                                .first()
                                .cloned()
                                .ok_or_else(|| routing_error("Model call has no candidates"))?,
                            LlmClientError::General(message),
                        )),
                        _ => {
                            return Err(error(
                                "ERR_INVALID_REQUEST",
                                "Expected a model response or error",
                            ));
                        }
                    };
                    call.respond(ready(result)).await.map_err(routing_error)?;
                }
            }
        } else if response.is_some() || failure.is_some() {
            return Err(error("ERR_INVALID_REQUEST", "No call is pending"));
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
                self.pending = Some(Call::Decision(call));
            }
            Step::Done(outcome) => {
                result.kind = "done".into();
                if let Some(initial_state) = &self.initial_state {
                    result.model = Some(
                        outcome
                            .selected_model_id()
                            .map_err(routing_error)?
                            .to_string(),
                    );
                    if &outcome.state != initial_state {
                        result.state =
                            Some(serde_json::to_string(&outcome.state).map_err(routing_error)?);
                    }
                } else {
                    let response = match outcome.response.as_ref().map(|r| &r.llm_response) {
                        Some(LlmResponse::Agg(response)) => Some(response),
                        Some(LlmResponse::Stream(_)) => {
                            return Err(routing_error(
                                "Streaming model responses are not supported",
                            ));
                        }
                        None => None,
                    };
                    result.outcome = Some(
                        serde_json::to_string(&OutcomeView {
                            selected_model_ids: &outcome.selected_model_ids,
                            request: &outcome.request.llm_request,
                            response,
                            metadata: outcome.metadata.as_ref().map(|metadata| {
                                OutcomeMetadataView {
                                    outcome_id: metadata.outcome_id(),
                                    algorithm: &metadata.algorithm,
                                    evidence: &metadata.evidence,
                                }
                            }),
                            state: &outcome.state,
                        })
                        .map_err(routing_error)?,
                    );
                }
            }
            Step::CallModel(call) => {
                result.kind = "model".into();
                result.request =
                    Some(serde_json::to_string(&call.request.llm_request).map_err(routing_error)?);
                result.models = Some(call.models.iter().map(ToString::to_string).collect());
                self.pending = Some(Call::Model(call));
            }
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

impl NativeRun {
    fn new(
        algorithm: Arc<dyn Algorithm>,
        request: Request,
        models: Arc<RuntimeModels>,
        state_json: Option<&str>,
        full_outcome: bool,
    ) -> napi::Result<Self> {
        let state: State = state_json.map(parse).transpose()?.unwrap_or_default();
        let initial_state = (!full_outcome).then(|| state.clone());
        // Defer run_stream until next() polls on napi's Tokio runtime.
        let stream =
            futures::stream::once(async move { algorithm.run_stream(request, models, state) })
                .flatten();
        Ok(Self {
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
        if result
            .as_ref()
            .is_ok_and(|step| matches!(step.kind.as_str(), "decision" | "model"))
        {
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

#[cfg(test)]
mod tests;
