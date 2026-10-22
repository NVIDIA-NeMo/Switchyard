// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pull-based C bridge to libsy.

use std::collections::{BTreeMap, HashMap};
use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use futures::{FutureExt, StreamExt};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use switchyard_libsy::{Algorithm, Call, LibsyError, RuntimeModels, State, Step, StepStream};
use switchyard_protocol::{
    AggLlmResponse, Category, DecisionRequest, LlmRequest, LlmResponse, Metadata, ModelId, Request,
    Response, WireFormat,
};
use switchyard_runner::AlgorithmSpec;
use tokio::runtime::Runtime;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
type ModelGroups = HashMap<String, Option<Vec<ModelId>>>;

/// Returns the compatibility generation for C signatures and JSON payloads.
#[unsafe(no_mangle)]
pub extern "C" fn sy_abi_version() -> u32 {
    1
}

#[derive(Deserialize)]
struct RouterConfig {
    algorithm: AlgorithmSpec,
    targets: BTreeMap<String, ModelId>,
}

#[derive(Deserialize)]
struct RunInput {
    request: LlmRequest,
    models: ModelGroups,
    subagent_models: Option<ModelGroups>,
    metadata: Option<RequestMetadata>,
}

#[derive(Deserialize)]
struct RequestMetadata {
    session_id: Option<String>,
    agent_id: Option<String>,
    #[serde(default)]
    is_subagent: bool,
    #[serde(default)]
    is_delegated_work: bool,
    session_final: Option<bool>,
}

#[derive(Deserialize)]
struct DecodeInput {
    format: WireFormat,
    body: serde_json::Value,
}

#[derive(Deserialize)]
struct EncodeInput {
    format: WireFormat,
    request: LlmRequest,
    model: ModelId,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum CallReply {
    Error {
        error: String,
    },
    Response {
        response: serde_json::Value,
        served_model: Option<ModelId>,
    },
}

#[derive(Serialize)]
struct OutcomeMetadataRef<'a> {
    outcome_id: &'a str,
    algorithm: &'a str,
    evidence: Option<&'a serde_json::Value>,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RunStep<'a> {
    CallModel {
        id: u64,
        models: &'a [ModelId],
        request: &'a LlmRequest,
    },
    CallDecision {
        id: u64,
        model: &'a ModelId,
        request: &'a DecisionRequest,
    },
    Done {
        selected_model_ids: &'a [ModelId],
        request: &'a LlmRequest,
        response: Option<&'a AggLlmResponse>,
        metadata: Option<OutcomeMetadataRef<'a>>,
    },
}

struct Engine {
    algorithm: Arc<dyn Algorithm>,
    runtime: Runtime,
}

/// Algorithm handle; routing runs retain its engine.
pub struct SyRouter {
    engine: Arc<Engine>,
}

struct Reader {
    stream: Option<StepStream>,
    next_id: u64,
}

enum RunStatus {
    Active,
    Completed,
    Cancelled,
    Failed,
}

impl RunStatus {
    fn require_active(&self) -> Result<()> {
        match self {
            Self::Active => Ok(()),
            Self::Completed => Err("routing already completed".into()),
            Self::Cancelled => Err("routing cancelled".into()),
            Self::Failed => Err("routing failed".into()),
        }
    }
}

struct Lifecycle {
    status: RunStatus,
    is_reading: bool,
    pending: HashMap<u64, Call>,
}

struct RunState {
    reader: tokio::sync::Mutex<Reader>,
    lifecycle: Mutex<Lifecycle>,
    cancellation: CancellationToken,
}

/// Routing run that owns pending calls and callback cleanup.
pub struct SyRun {
    state: Arc<RunState>,
    tasks: TaskTracker,
    // Tasks retain only run state; free joins them before dropping the runtime.
    engine: Arc<Engine>,
}

impl SyRun {
    fn new(stream: StepStream, engine: Arc<Engine>) -> Self {
        Self {
            state: Arc::new(RunState {
                reader: tokio::sync::Mutex::new(Reader {
                    stream: Some(stream),
                    next_id: 1,
                }),
                lifecycle: Mutex::new(Lifecycle {
                    status: RunStatus::Active,
                    is_reading: false,
                    pending: HashMap::new(),
                }),
                cancellation: CancellationToken::new(),
            }),
            tasks: TaskTracker::new(),
            engine,
        }
    }
}

impl Drop for SyRun {
    fn drop(&mut self) {
        self.state.stop(RunStatus::Cancelled);
        self.tasks.close();
        self.engine.runtime.block_on(self.tasks.wait());
    }
}

struct ReadGuard {
    state: Arc<RunState>,
}

impl Drop for ReadGuard {
    fn drop(&mut self) {
        self.state.lifecycle.lock().is_reading = false;
    }
}

impl RunState {
    fn begin_read(self: &Arc<Self>) -> Result<ReadGuard> {
        let mut lifecycle = self.lifecycle.lock();
        lifecycle.status.require_active()?;
        if lifecycle.is_reading {
            return Err("routing read already in progress".into());
        }
        lifecycle.is_reading = true;
        Ok(ReadGuard {
            state: Arc::clone(self),
        })
    }

    // Serialize cancellation with reply delivery.
    fn stop(&self, status: RunStatus) {
        {
            let mut lifecycle = self.lifecycle.lock();
            if matches!(lifecycle.status, RunStatus::Active) {
                lifecycle.status = status;
            }
            lifecycle.pending.clear();
        }
        self.cancellation.cancel();
        if let Ok(mut reader) = self.reader.try_lock() {
            reader.stream.take();
        }
    }

    async fn next_step(&self, reader: &mut Reader) -> Result<CString> {
        self.lifecycle.lock().status.require_active()?;
        let stream = reader.stream.as_mut().ok_or("routing stream is closed")?;
        let step = tokio::select! {
            biased;
            _ = self.cancellation.cancelled() => None,
            step = stream.next() => step,
        };
        self.lifecycle.lock().status.require_active()?;
        let step = step.ok_or("routing stream ended without an outcome")??;
        let id = reader.next_id;
        reader.next_id += 1;
        match step {
            Step::CallModel(call) => {
                let output = encode_json(&RunStep::CallModel {
                    id,
                    models: &call.models,
                    request: &call.request.llm_request,
                })?;
                let mut lifecycle = self.lifecycle.lock();
                lifecycle.status.require_active()?;
                lifecycle.pending.insert(id, Call::Model(call));
                Ok(output)
            }
            Step::CallDecision(call) => {
                let output = encode_json(&RunStep::CallDecision {
                    id,
                    model: &call.model,
                    request: &call.request,
                })?;
                let mut lifecycle = self.lifecycle.lock();
                lifecycle.status.require_active()?;
                lifecycle.pending.insert(id, Call::Decision(call));
                Ok(output)
            }
            Step::Done(outcome) => {
                let response = match outcome.response.as_ref() {
                    None => None,
                    Some(Response {
                        llm_response: LlmResponse::Agg(response),
                        ..
                    }) => Some(response),
                    Some(_) => {
                        return Err(
                            "streaming routing outcomes are not supported by this spike".into()
                        );
                    }
                };
                let output = encode_json(&RunStep::Done {
                    selected_model_ids: &outcome.selected_model_ids,
                    request: &outcome.request.llm_request,
                    response,
                    metadata: outcome
                        .metadata
                        .as_ref()
                        .map(|metadata| OutcomeMetadataRef {
                            outcome_id: metadata.outcome_id(),
                            algorithm: &metadata.algorithm,
                            evidence: metadata.evidence.as_ref(),
                        }),
                })?;
                let mut lifecycle = self.lifecycle.lock();
                lifecycle.status.require_active()?;
                lifecycle.status = RunStatus::Completed;
                lifecycle.pending.clear();
                reader.stream.take();
                Ok(output)
            }
        }
    }
}

impl ReadGuard {
    async fn next(self) -> Result<CString> {
        let mut reader = self.state.reader.lock().await;
        let result = AssertUnwindSafe(self.state.next_step(&mut reader))
            .catch_unwind()
            .await
            .unwrap_or_else(|_| Err("Switchyard binding panicked".into()));
        if result.is_err() {
            self.state.stop(RunStatus::Failed);
            reader.stream.take();
        }
        result
    }
}

fn error_string(message: String) -> *mut c_char {
    match CString::new(message) {
        Ok(message) => message.into_raw(),
        Err(_) => c"Switchyard error contained a NUL byte"
            .to_owned()
            .into_raw(),
    }
}

// The input must be NUL-terminated; parsed values do not borrow it.
unsafe fn parse_json_input<T: serde::de::DeserializeOwned>(input: *const c_char) -> Result<T> {
    if input.is_null() {
        return Err("missing JSON input".into());
    }
    let input = unsafe { CStr::from_ptr(input) };
    Ok(serde_json::from_slice(input.to_bytes())?)
}

fn encode_json(output: &impl Serialize) -> Result<CString> {
    Ok(CString::new(serde_json::to_vec(output)?)?)
}

/// Decodes provider JSON into a normalized request.
/// # Safety
/// JSON must be NUL-terminated and `out_json` writable. Free the output with `sy_string_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sy_decode_request(
    input_json: *const c_char,
    out_json: *mut *mut c_char,
) -> *mut c_char {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<()> {
        let out = unsafe { out_json.as_mut() }.ok_or("missing request output")?;
        *out = std::ptr::null_mut();
        let input: DecodeInput = unsafe { parse_json_input(input_json) }?;
        *out = encode_json(&switchyard_translation::decode_request(
            input.format,
            &input.body,
        )?)?
        .into_raw();
        Ok(())
    }));
    match result {
        Ok(Ok(())) => std::ptr::null_mut(),
        Ok(Err(error)) => error_string(error.to_string()),
        Err(_) => error_string("Switchyard binding panicked".to_string()),
    }
}

/// Applies the selected model and encodes the request.
/// # Safety
/// JSON must be NUL-terminated and `out_json` writable. Free the output with `sy_string_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sy_encode_request(
    input_json: *const c_char,
    out_json: *mut *mut c_char,
) -> *mut c_char {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<()> {
        let out = unsafe { out_json.as_mut() }.ok_or("missing request output")?;
        *out = std::ptr::null_mut();
        let mut input: EncodeInput = unsafe { parse_json_input(input_json) }?;
        switchyard_translation::prepare_request_for_target(&mut input.request, &input.model, None);
        *out = encode_json(&switchyard_translation::encode_request(
            &input.request,
            input.format,
        )?)?
        .into_raw();
        Ok(())
    }));
    match result {
        Ok(Ok(())) => std::ptr::null_mut(),
        Ok(Err(error)) => error_string(error.to_string()),
        Err(_) => error_string("Switchyard binding panicked".to_string()),
    }
}

/// Decodes a buffered provider response.
/// # Safety
/// JSON must be NUL-terminated and `out_json` writable. Free the output with `sy_string_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sy_decode_response(
    input_json: *const c_char,
    out_json: *mut *mut c_char,
) -> *mut c_char {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<()> {
        let out = unsafe { out_json.as_mut() }.ok_or("missing response output")?;
        *out = std::ptr::null_mut();
        let input: DecodeInput = unsafe { parse_json_input(input_json) }?;
        *out = encode_json(&switchyard_translation::decode_aggregated_response(
            &input.body,
            input.format,
        )?)?
        .into_raw();
        Ok(())
    }));
    match result {
        Ok(Ok(())) => std::ptr::null_mut(),
        Ok(Err(error)) => error_string(error.to_string()),
        Err(_) => error_string("Switchyard binding panicked".to_string()),
    }
}

fn parse_model_groups(groups: ModelGroups) -> Result<HashMap<Category, Vec<ModelId>>> {
    groups
        .into_iter()
        .map(|(name, models)| Ok((name.parse()?, models.unwrap_or_default())))
        .collect()
}

/// Builds an algorithm without provider clients.
/// # Safety
/// JSON must be NUL-terminated and `out` writable. Free the returned handle with `sy_router_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sy_router_new(
    config_json: *const c_char,
    out: *mut *mut SyRouter,
) -> *mut c_char {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<()> {
        let out = unsafe { out.as_mut() }.ok_or("missing router output")?;
        *out = std::ptr::null_mut();
        let config: RouterConfig = unsafe { parse_json_input(config_json) }?;
        let algorithm = config.algorithm.build("embedded", &config.targets)?;
        let runtime = Runtime::new()?;
        *out = Box::into_raw(Box::new(SyRouter {
            engine: Arc::new(Engine { algorithm, runtime }),
        }));
        Ok(())
    }));
    match result {
        Ok(Ok(())) => std::ptr::null_mut(),
        Ok(Err(error)) => error_string(error.to_string()),
        Err(_) => error_string("Switchyard binding panicked".to_string()),
    }
}

/// Starts a run with host-approved model groups.
/// # Safety
/// Handles must be live, JSON NUL-terminated, and `out` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sy_run_new(
    router: *const SyRouter,
    input_json: *const c_char,
    out: *mut *mut SyRun,
) -> *mut c_char {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<()> {
        let router = unsafe { router.as_ref() }.ok_or("missing router")?;
        let out = unsafe { out.as_mut() }.ok_or("missing run output")?;
        *out = std::ptr::null_mut();
        let input: RunInput = unsafe { parse_json_input(input_json) }?;
        let mut models = RuntimeModels::new(parse_model_groups(input.models)?);
        if let Some(subagent_models) = input.subagent_models {
            models = models.with_subagent(parse_model_groups(subagent_models)?);
        }
        let request = Request {
            llm_request: input.request,
            metadata: input.metadata.map(|metadata| Metadata {
                session_id: metadata.session_id,
                agent_id: metadata.agent_id,
                is_subagent: metadata.is_subagent,
                is_delegated_work: metadata.is_delegated_work,
                session_final: metadata.session_final,
                ..Default::default()
            }),
            ..Default::default()
        };
        let _entered = router.engine.runtime.enter();
        let stream = Arc::clone(&router.engine.algorithm).run_stream(
            request,
            Arc::new(models),
            State::default(),
        );
        *out = Box::into_raw(Box::new(SyRun::new(stream, Arc::clone(&router.engine))));
        Ok(())
    }));
    match result {
        Ok(Ok(())) => std::ptr::null_mut(),
        Ok(Err(error)) => error_string(error.to_string()),
        Err(_) => error_string("Switchyard binding panicked".to_string()),
    }
}

/// Requests the next step asynchronously. Only one read may be pending per run.
/// NULL accepts the read (one callback); an owned error rejects it (no callback).
/// Other runs, model replies, and cancellation can proceed independently.
/// # Safety
/// Keep the run live during this call and the callback state live until its callback returns.
/// Callbacks may overlap or run before this call returns. Free their strings with `sy_string_free`.
/// Callbacks must return promptly without unwinding, calling `sy_run_respond`, or freeing handles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sy_run_next_async(
    run: *const SyRun,
    callback: Option<unsafe extern "C" fn(usize, *mut c_char, *mut c_char)>,
    user_data: usize,
) -> *mut c_char {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<()> {
        let run = unsafe { run.as_ref() }.ok_or("missing run")?;
        let callback = callback.ok_or("missing completion callback")?;
        let read = run.state.begin_read()?;
        run.tasks.spawn_on(
            async move {
                let (error, output) = match read.next().await {
                    Ok(output) => (std::ptr::null_mut(), output.into_raw()),
                    Err(error) => (error_string(error.to_string()), std::ptr::null_mut()),
                };
                unsafe { callback(user_data, error, output) };
            },
            run.engine.runtime.handle(),
        );
        Ok(())
    }));
    match result {
        Ok(Ok(())) => std::ptr::null_mut(),
        Ok(Err(error)) => error_string(error.to_string()),
        Err(_) => error_string("Switchyard binding panicked".to_string()),
    }
}

/// Delivers a buffered host reply to its pending call.
/// # Safety
/// The run must remain live and JSON must be NUL-terminated. May overlap an async read.
/// Do not call from a completion callback or Tokio task.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sy_run_respond(
    run: *const SyRun,
    call_id: u64,
    reply_json: *const c_char,
) -> *mut c_char {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<()> {
        let run = unsafe { run.as_ref() }.ok_or("missing run")?;
        let reply: CallReply = unsafe { parse_json_input(reply_json) }?;
        let mut lifecycle = run.state.lifecycle.lock();
        lifecycle.status.require_active()?;
        let call = lifecycle
            .pending
            .remove(&call_id)
            .ok_or("unknown call ID")?;
        let result = match call {
            Call::Model(call) => {
                let response = match reply {
                    CallReply::Error { error } => Ok(Err(LibsyError::external(
                        "host model call",
                        std::io::Error::other(error),
                    ))),
                    CallReply::Response {
                        response,
                        served_model,
                    } => serde_json::from_value(response).map(|response| {
                        Ok(Response {
                            llm_response: LlmResponse::Agg(response),
                            metadata: served_model.map(|model| Metadata {
                                served_model: Some(model),
                                ..Default::default()
                            }),
                            upstream_headers: Default::default(),
                        })
                    }),
                };
                let response = match response {
                    Ok(response) => response,
                    Err(error) => {
                        lifecycle.pending.insert(call_id, Call::Model(call));
                        return Err(error.into());
                    }
                };
                run.engine
                    .runtime
                    .block_on(call.respond(std::future::ready(response)))
            }
            Call::Decision(call) => {
                let response = match reply {
                    CallReply::Error { error } => Ok(Err(LibsyError::external(
                        "host decision call",
                        std::io::Error::other(error),
                    ))),
                    CallReply::Response { response, .. } => {
                        serde_json::from_value(response).map(Ok)
                    }
                };
                let response = match response {
                    Ok(response) => response,
                    Err(error) => {
                        lifecycle.pending.insert(call_id, Call::Decision(call));
                        return Err(error.into());
                    }
                };
                run.engine
                    .runtime
                    .block_on(call.respond(std::future::ready(response)))
            }
        };
        if result.is_err() {
            lifecycle.status = RunStatus::Failed;
            lifecycle.pending.clear();
        }
        drop(lifecycle);
        if result.is_err() {
            run.state.stop(RunStatus::Failed);
        }
        Ok(result?)
    }));
    match result {
        Ok(Ok(())) => std::ptr::null_mut(),
        Ok(Err(error)) => error_string(error.to_string()),
        Err(_) => error_string("Switchyard binding panicked".to_string()),
    }
}

/// Signals cancellation without waiting for callbacks.
/// # Safety
/// The run must remain live for the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sy_run_cancel(run: *const SyRun) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if let Some(run) = unsafe { run.as_ref() } {
            run.state.stop(RunStatus::Cancelled);
        }
    }));
}

/// Cancels a run, waits for accepted callbacks to return, and frees it.
/// # Safety
/// Pass a handle from `sy_run_new` once, or NULL. Other calls using it must have returned.
/// Accepted callbacks may still be active; keep their state live until they return.
/// Do not call from a completion callback or Tokio task.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sy_run_free(run: *mut SyRun) {
    if !run.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(unsafe { Box::from_raw(run) })));
    }
}

/// Frees the router handle; existing runs retain its engine.
/// # Safety
/// Pass a handle from `sy_router_new` once, with no concurrent calls using it, or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sy_router_free(router: *mut SyRouter) {
    if !router.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(unsafe { Box::from_raw(router) })));
    }
}

/// Frees a library-owned error or JSON string.
/// # Safety
/// Pass a returned string once, with no remaining readers, or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sy_string_free(string: *mut c_char) {
    if !string.is_null() {
        drop(unsafe { CString::from_raw(string) });
    }
}

#[cfg(test)]
mod tests;
