// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configured routing steps and normalized responses for host-owned execution.

use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use futures::future::{AbortHandle, Abortable};
use futures::{Stream, StreamExt, TryStreamExt};
use http::{HeaderMap, HeaderName, HeaderValue};
use napi::bindgen_prelude::Promise;
use napi_derive::napi;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use switchyard_libsy::{
    Algorithm, CallDecision, CallModel, LibsyError, RuntimeModels, State, Step,
};
use switchyard_protocol::{
    AggLlmResponse, DecisionRequest, DecisionResponse, LlmClientError, LlmRequest, LlmResponse,
    LlmResponseStream, LlmResponseStreamEvent, Metadata, ModelId, Request, Response,
};
use switchyard_runner::{
    AlgorithmFactory, ConfiguredAlgorithm, RunnerError, require_target_labels,
};
use tokio::sync::mpsc;

use crate::{error, parse, routing_error};

#[napi]
pub struct NativeAlgorithmFactory {
    inner: AlgorithmFactory,
}

#[napi]
impl NativeAlgorithmFactory {
    #[napi(constructor)]
    pub fn new(source: String) -> napi::Result<Self> {
        Ok(Self {
            inner: AlgorithmFactory::from_toml(&source).map_err(|err| error("ERR_CONFIG", err))?,
        })
    }

    #[napi]
    pub fn build(&self) -> napi::Result<BTreeMap<String, NativeConfiguredAlgorithm>> {
        Ok(self
            .inner
            .build()
            .map_err(|err| error("ERR_CONFIG", err))?
            .into_iter()
            .map(|(id, (algorithm, models))| {
                (
                    id,
                    NativeConfiguredAlgorithm {
                        inner: AlgorithmHandle::Bare {
                            algorithm,
                            models: Arc::new(models),
                        },
                    },
                )
            })
            .collect())
    }

    #[napi]
    pub fn build_configured(&self) -> napi::Result<BTreeMap<String, NativeConfiguredAlgorithm>> {
        Ok(self
            .inner
            .build_configured()
            .map_err(|err| error("ERR_CONFIG", err))?
            .into_iter()
            .map(|(id, algorithm)| {
                (
                    id,
                    NativeConfiguredAlgorithm {
                        inner: AlgorithmHandle::Governed(Arc::new(algorithm)),
                    },
                )
            })
            .collect())
    }
}

#[derive(Clone)]
enum AlgorithmHandle {
    Bare {
        algorithm: Arc<dyn Algorithm>,
        models: Arc<RuntimeModels>,
    },
    Governed(Arc<ConfiguredAlgorithm>),
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RunOptions {
    required_labels: Option<Vec<String>>,
    task_id: Option<String>,
}

fn admission_error(err: RunnerError) -> napi::Error {
    let code = match &err {
        RunnerError::Configuration { .. } => "ERR_CONFIG",
        RunnerError::Client(LlmClientError::InvalidRequest { .. }) => "ERR_INVALID_REQUEST",
        _ => "ERR_ROUTING",
    };
    error(code, err)
}

#[napi]
pub struct NativeConfiguredAlgorithm {
    inner: AlgorithmHandle,
}

#[napi(object)]
pub struct NativeTargetModel {
    pub client: String,
    pub id: String,
}

#[napi]
impl NativeConfiguredAlgorithm {
    #[napi]
    pub fn target_models(&self) -> napi::Result<BTreeMap<String, NativeTargetModel>> {
        let AlgorithmHandle::Governed(algorithm) = &self.inner else {
            return Err(error(
                "ERR_INVALID_REQUEST",
                "target discovery requires buildConfigured()",
            ));
        };
        Ok(algorithm
            .target_models()
            .iter()
            .map(|(name, target)| {
                (
                    name.to_string(),
                    NativeTargetModel {
                        client: target.client.clone(),
                        id: target.id.to_string(),
                    },
                )
            })
            .collect())
    }

    #[napi]
    pub fn start(
        &self,
        request_json: String,
        headers_json: Option<String>,
        options_json: Option<String>,
    ) -> napi::Result<NativeAlgorithmRun> {
        let options: RunOptions = options_json
            .as_deref()
            .map(parse)
            .transpose()?
            .unwrap_or_default();
        if (options.required_labels.is_some() || options.task_id.is_some())
            && matches!(&self.inner, AlgorithmHandle::Bare { .. })
        {
            return Err(error(
                "ERR_INVALID_REQUEST",
                "target requirements or task IDs require buildConfigured()",
            ));
        }
        let headers: Option<BTreeMap<String, String>> =
            headers_json.as_deref().map(parse).transpose()?;
        let headers = headers
            .map(|headers| {
                let mut result = HeaderMap::new();
                for (name, value) in headers {
                    let name = HeaderName::from_bytes(name.as_bytes())
                        .map_err(|err| error("ERR_INVALID_REQUEST", err))?;
                    let value = HeaderValue::from_str(&value)
                        .map_err(|err| error("ERR_INVALID_REQUEST", err))?;
                    result
                        .try_append(name, value)
                        .map_err(|err| error("ERR_INVALID_REQUEST", err))?;
                }
                Ok::<_, napi::Error>(result)
            })
            .transpose()?;
        let mut llm_request = parse(&request_json)?;
        switchyard_translation::normalize_compaction(&mut llm_request, headers.as_ref());
        let mut request = Request {
            llm_request,
            raw_request: None,
            metadata: headers.map(|headers| Metadata::from_headers(&headers)),
        };
        if let Some(task_id) = options.task_id {
            request.metadata.get_or_insert_default().task_id = Some(task_id);
        }
        if let Some(labels) = options.required_labels {
            require_target_labels(&mut request, labels)
                .map_err(|err| error("ERR_INVALID_REQUEST", err))?;
        }
        let handle = self.inner.clone();
        // Algorithm startup must run on napi's Tokio runtime.
        let stream = futures::stream::once(async move {
            let stream = match handle {
                AlgorithmHandle::Bare { algorithm, models } => {
                    Ok(algorithm.run_stream(request, models, State::default()))
                }
                AlgorithmHandle::Governed(algorithm) => algorithm
                    .run_stream(request, State::default())
                    .map_err(admission_error),
            }?;
            Ok::<_, napi::Error>(stream.map(|step| step.map_err(routing_error)))
        })
        .try_flatten();
        Ok(NativeAlgorithmRun {
            reader: Reader::new(stream),
        })
    }
}

type Items<T> = Pin<Box<dyn Stream<Item = napi::Result<T>> + Send>>;

struct ReaderSlot<T> {
    stream: Option<Items<T>>,
    abort: Option<AbortHandle>,
}

struct Reader<T> {
    slot: Mutex<ReaderSlot<T>>,
}

impl<T: Send + 'static> Reader<T> {
    fn new(stream: impl Stream<Item = napi::Result<T>> + Send + 'static) -> Self {
        Self {
            slot: Mutex::new(ReaderSlot {
                stream: Some(Box::pin(stream)),
                abort: None,
            }),
        }
    }

    async fn next(&self) -> napi::Result<Option<T>> {
        let (abort, registration) = AbortHandle::new_pair();
        let mut stream = {
            let mut slot = self.slot.lock();
            let stream = match slot.stream.take() {
                Some(stream) => stream,
                None if slot.abort.is_none() => return Ok(None),
                None => return Err(error("ERR_INVALID_REQUEST", "next is already pending")),
            };
            slot.abort = Some(abort.clone());
            stream
        };
        // close() aborts the future that owns the reader, including a pending next().
        let (stream, item) = Abortable::new(
            async move {
                let item = stream.next().await;
                (stream, item)
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
        if item.as_ref().is_some_and(Result::is_ok) {
            slot.stream = Some(stream);
        }
        item.transpose()
    }

    fn close(&self) {
        let mut slot = self.slot.lock();
        if let Some(abort) = slot.abort.take() {
            abort.abort();
        }
        slot.stream = None;
    }
}

#[napi]
pub struct NativeAlgorithmRun {
    reader: Reader<Step>,
}

#[napi(object, object_from_js = false, use_nullable = true)]
pub struct NativeAlgorithmStep {
    pub kind: String,
    pub data: String,
    pub call: Option<NativeAlgorithmCall>,
    pub stream: Option<NativeResponseStream>,
}

#[napi]
impl NativeAlgorithmRun {
    #[napi]
    pub async fn next(&self) -> napi::Result<Option<NativeAlgorithmStep>> {
        self.reader.next().await?.map(step_to_native).transpose()
    }

    #[napi]
    pub fn close(&self) {
        self.reader.close();
    }
}

enum HostCall {
    Model(Box<CallModel>),
    Decision(Box<CallDecision>),
}

type InputItem = Option<Result<LlmResponseStreamEvent, LlmClientError>>;

#[napi]
pub struct NativeAlgorithmCall {
    inner: Mutex<Option<HostCall>>,
    input: Mutex<Option<mpsc::Receiver<InputItem>>>,
    model: ModelId,
    is_complete: AtomicBool,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Reply {
    Agg {
        response: AggLlmResponse,
    },
    Stream,
    Decision {
        response: DecisionResponse,
    },
    Error {
        error: String,
        #[serde(default)]
        context_window_exceeded: bool,
    },
}

#[derive(Debug)]
struct ForeignError(String);

impl std::fmt::Display for ForeignError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ForeignError {}

fn client_error(model: &ModelId, message: String, is_context_window: bool) -> LlmClientError {
    if is_context_window {
        LlmClientError::ContextWindowExceeded {
            model: model.clone(),
            message,
        }
    } else {
        LlmClientError::Ffi {
            source: Box::new(ForeignError(message)),
        }
    }
}

#[napi]
impl NativeAlgorithmCall {
    #[napi]
    pub fn input(&self) -> napi::Result<NativeInputStream> {
        let mut input = self.input.lock();
        if self.is_complete.load(Ordering::Acquire) {
            return Err(error("ABORT_ERR", "Call is no longer waiting"));
        }
        if input.is_some() {
            return Err(error(
                "ERR_INVALID_REQUEST",
                "Response stream already supplied",
            ));
        }
        let (sender, receiver) = mpsc::channel(1);
        *input = Some(receiver);
        Ok(NativeInputStream {
            sender,
            model: self.model.clone(),
        })
    }

    #[napi]
    pub async fn respond(&self, work: Promise<String>) -> napi::Result<()> {
        let call = self
            .inner
            .lock()
            .take()
            .ok_or_else(|| routing_error("Call has already been completed"))?;
        let model = &self.model;
        let work = async {
            let json = work.await.map_err(|err| {
                LibsyError::client_call(model.clone(), client_error(model, err.to_string(), false))
            })?;
            let reply: Reply = serde_json::from_str(&json).map_err(|err| {
                LibsyError::client_call(model.clone(), client_error(model, err.to_string(), false))
            })?;
            if let Reply::Error {
                error,
                context_window_exceeded,
            } = reply
            {
                return Err(LibsyError::client_call(
                    model.clone(),
                    client_error(model, error, context_window_exceeded),
                ));
            }
            Ok(reply)
        };
        let result = match call {
            HostCall::Decision(call) => {
                call.respond(async {
                    match work.await? {
                        Reply::Decision { response } => Ok(response),
                        _ => Err(LibsyError::client_call(
                            model.clone(),
                            client_error(model, "Expected a decision response".into(), false),
                        )),
                    }
                })
                .await
            }
            HostCall::Model(call) => {
                let metadata = call.request.metadata.clone();
                call.respond(async {
                    let llm_response = match work.await? {
                        Reply::Agg { response } => LlmResponse::Agg(response),
                        Reply::Stream => {
                            let receiver = self.input.lock().take().ok_or_else(|| {
                                LibsyError::client_call(
                                    model.clone(),
                                    client_error(model, "Response stream is missing".into(), false),
                                )
                            })?;
                            let stream = futures::stream::unfold(Some(receiver), |state| async {
                                let mut receiver = state?;
                                let item = receiver.recv().await??;
                                let state = item.is_ok().then_some(receiver);
                                Some((item, state))
                            });
                            LlmResponse::Stream(Box::pin(stream))
                        }
                        _ => {
                            return Err(LibsyError::client_call(
                                model.clone(),
                                client_error(
                                    model,
                                    "Expected LlmResponse.Agg or LlmResponse.Stream".into(),
                                    false,
                                ),
                            ));
                        }
                    };
                    Ok(Response {
                        llm_response,
                        metadata,
                        upstream_headers: HeaderMap::new(),
                    })
                })
                .await
            }
        };
        self.is_complete.store(true, Ordering::Release);
        self.input.lock().take();
        result.map_err(routing_error)
    }
}

#[napi]
pub struct NativeInputStream {
    sender: mpsc::Sender<InputItem>,
    model: ModelId,
}

#[napi]
impl NativeInputStream {
    #[napi]
    pub async fn push(&self, json: String) -> napi::Result<()> {
        let event = serde_json::from_str(&json).map_err(routing_error)?;
        self.sender
            .send(Some(Ok(event)))
            .await
            .map_err(|_| error("ABORT_ERR", "Response stream closed"))
    }

    #[napi]
    pub async fn finish(
        &self,
        message: Option<String>,
        is_context_window: bool,
    ) -> napi::Result<()> {
        let item =
            message.map(|message| Err(client_error(&self.model, message, is_context_window)));
        self.sender
            .send(item)
            .await
            .map_err(|_| error("ABORT_ERR", "Response stream closed"))
    }

    #[napi]
    pub async fn closed(&self) {
        self.sender.closed().await;
    }
}

#[napi]
pub struct NativeResponseStream {
    reader: Reader<LlmResponseStreamEvent>,
}

impl NativeResponseStream {
    fn new(stream: LlmResponseStream) -> Self {
        Self {
            reader: Reader::new(stream.map(|item| item.map_err(routing_error))),
        }
    }
}

#[napi]
impl NativeResponseStream {
    #[napi]
    pub async fn next(&self) -> napi::Result<Option<String>> {
        self.reader
            .next()
            .await?
            .map(|event| serde_json::to_string(&event).map_err(routing_error))
            .transpose()
    }

    #[napi]
    pub fn close(&self) {
        self.reader.close();
    }
}

#[derive(Serialize)]
#[serde(untagged)]
enum StepPayload<'a> {
    Decision {
        algorithm: &'a str,
        request: &'a DecisionRequest,
        model: &'a ModelId,
    },
    Model {
        algorithm: &'a str,
        request: &'a LlmRequest,
        models: &'a [ModelId],
    },
    Done {
        selected_model_ids: &'a [ModelId],
        request: &'a LlmRequest,
        fallback_request: Option<&'a LlmRequest>,
        candidate_requests: BTreeMap<&'a str, &'a LlmRequest>,
        response: Option<ResponsePayload<'a>>,
        metadata: Option<MetadataPayload<'a>>,
    },
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ResponsePayload<'a> {
    Agg { response: &'a AggLlmResponse },
    Stream,
}

#[derive(Serialize)]
struct MetadataPayload<'a> {
    outcome_id: &'a str,
    algorithm: &'a str,
    evidence: Option<&'a Value>,
}

fn step_to_native(step: Step) -> napi::Result<NativeAlgorithmStep> {
    let mut result = NativeAlgorithmStep {
        kind: String::new(),
        data: String::new(),
        call: None,
        stream: None,
    };
    match step {
        Step::CallDecision(call) => {
            result.kind = "decision".into();
            result.data = serde_json::to_string(&StepPayload::Decision {
                algorithm: &call.algorithm,
                request: &call.request,
                model: &call.model,
            })
            .map_err(routing_error)?;
            result.call = Some(NativeAlgorithmCall {
                model: call.model.clone(),
                inner: Mutex::new(Some(HostCall::Decision(call))),
                input: Mutex::new(None),
                is_complete: AtomicBool::new(false),
            });
        }
        Step::CallModel(call) => {
            result.kind = "model".into();
            result.data = serde_json::to_string(&StepPayload::Model {
                algorithm: &call.algorithm,
                request: &call.request.llm_request,
                models: &call.models,
            })
            .map_err(routing_error)?;
            result.call = Some(NativeAlgorithmCall {
                model: call
                    .request
                    .llm_request
                    .model
                    .as_ref()
                    .map(ModelId::new)
                    .ok_or_else(|| routing_error("Model call has no selected model"))?,
                inner: Mutex::new(Some(HostCall::Model(call))),
                input: Mutex::new(None),
                is_complete: AtomicBool::new(false),
            });
        }
        Step::Done(outcome) => {
            result.kind = "done".into();
            result.data = serde_json::to_string(&StepPayload::Done {
                selected_model_ids: &outcome.selected_model_ids,
                request: &outcome.request.llm_request,
                fallback_request: outcome
                    .fallback_request
                    .as_ref()
                    .map(|request| &request.llm_request),
                candidate_requests: outcome
                    .candidate_requests
                    .iter()
                    .map(|(id, request)| (id.as_str(), &request.llm_request))
                    .collect(),
                response: outcome
                    .response
                    .as_ref()
                    .map(|response| match &response.llm_response {
                        LlmResponse::Agg(response) => ResponsePayload::Agg { response },
                        LlmResponse::Stream(_) => ResponsePayload::Stream,
                    }),
                metadata: outcome.metadata.as_ref().map(|metadata| MetadataPayload {
                    outcome_id: metadata.outcome_id(),
                    algorithm: &metadata.algorithm,
                    evidence: metadata.evidence.as_ref(),
                }),
            })
            .map_err(routing_error)?;
            if let Some(Response {
                llm_response: LlmResponse::Stream(stream),
                ..
            }) = outcome.response
            {
                result.stream = Some(NativeResponseStream::new(stream));
            }
        }
    }
    Ok(result)
}
