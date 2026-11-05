// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configured routing steps and normalized responses for host-owned execution.

use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use futures::future::{AbortHandle, Abortable};
use futures::{Stream, StreamExt};
use http::{HeaderMap, HeaderName, HeaderValue};
use napi::bindgen_prelude::Promise;
use napi_derive::napi;
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{Value, json};
use switchyard_libsy::{
    Algorithm, CallDecision, CallModel, LibsyError, RuntimeModels, State, Step,
};
use switchyard_protocol::{
    AggLlmResponse, DecisionResponse, LlmClientError, LlmResponse, LlmResponseStream,
    LlmResponseStreamEvent, Metadata, ModelId, Request, Response,
};
use switchyard_runner::AlgorithmFactory;
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
                        algorithm,
                        models: Arc::new(models),
                    },
                )
            })
            .collect())
    }
}

#[napi]
pub struct NativeConfiguredAlgorithm {
    algorithm: Arc<dyn Algorithm>,
    models: Arc<RuntimeModels>,
}

#[napi]
impl NativeConfiguredAlgorithm {
    #[napi]
    pub fn start(
        &self,
        request_json: String,
        headers_json: Option<String>,
    ) -> napi::Result<NativeAlgorithmRun> {
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
        let request = Request {
            llm_request,
            raw_request: None,
            metadata: headers.map(|headers| Metadata::from_headers(&headers)),
        };
        let algorithm = Arc::clone(&self.algorithm);
        let models = Arc::clone(&self.models);
        // Algorithm startup must run on napi's Tokio runtime.
        let stream = futures::stream::once(async move {
            algorithm.run_stream(request, models, State::default())
        })
        .flatten()
        .map(|step| step.map_err(routing_error));
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
            result.data = json!({
                "algorithm": call.algorithm, "request": call.request, "model": call.model,
            })
            .to_string();
            result.call = Some(NativeAlgorithmCall {
                model: call.model.clone(),
                inner: Mutex::new(Some(HostCall::Decision(call))),
                input: Mutex::new(None),
                is_complete: AtomicBool::new(false),
            });
        }
        Step::CallModel(call) => {
            result.kind = "model".into();
            result.data = json!({
                "algorithm": call.algorithm, "request": call.request.llm_request, "models": call.models,
            }).to_string();
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
            let response = match outcome.response {
                Some(response) => match response.llm_response {
                    LlmResponse::Agg(response) => json!({"kind": "agg", "response": response}),
                    LlmResponse::Stream(stream) => {
                        result.stream = Some(NativeResponseStream::new(stream));
                        json!({"kind": "stream"})
                    }
                },
                None => Value::Null,
            };
            let candidate_requests: BTreeMap<_, _> = outcome
                .candidate_requests
                .into_iter()
                .map(|(id, request)| (id.to_string(), request.llm_request))
                .collect();
            let metadata = outcome.metadata.map(|metadata| {
                json!({
                    "outcome_id": metadata.outcome_id(),
                    "algorithm": metadata.algorithm,
                    "evidence": metadata.evidence,
                })
            });
            result.data = json!({
                "selected_model_ids": outcome.selected_model_ids,
                "request": outcome.request.llm_request,
                "fallback_request": outcome.fallback_request.map(|request| request.llm_request),
                "candidate_requests": candidate_requests,
                "response": response,
                "metadata": metadata,
            })
            .to_string();
        }
    }
    Ok(result)
}
