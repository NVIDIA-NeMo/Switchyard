// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::c_void;
use std::future::Future;
use std::pin::Pin;
use std::ptr;
use std::task::{Context, Poll};

use futures::channel::oneshot;
use futures::Stream;
use nemo_relay_plugin::{
    LlmCallErrorV2, LlmCallOutcomeV2, LlmDispatchRequestV2, LlmRequest, LlmStreamEventV2,
    NemoRelayNativeAsyncCompletion, NemoRelayNativeAsyncNext, NemoRelayNativeAsyncNextStreamCb,
    NemoRelayNativeAsyncStream, NemoRelayNativeHostApiV1, NemoRelayNativeHostApiV4,
    NemoRelayNativeLlmStreamV2, NemoRelayNativeString, NemoRelayStatus,
};
use serde::Serialize;
use serde_json::Value as Json;

pub struct HostString {
    host: NemoRelayNativeHostApiV1,
    ptr: *mut NemoRelayNativeString,
}

impl HostString {
    pub fn json(host: &NemoRelayNativeHostApiV1, value: &impl Serialize) -> Result<Self, String> {
        let value = serde_json::to_string(value).map_err(|error| error.to_string())?;
        Self::text(host, &value)
    }

    pub fn text(host: &NemoRelayNativeHostApiV1, value: &str) -> Result<Self, String> {
        let mut ptr: *mut NemoRelayNativeString = ptr::null_mut();
        let status = unsafe { (host.string_new)(value.as_ptr(), value.len(), &mut ptr as *mut _) };
        if status == NemoRelayStatus::Ok && !ptr.is_null() {
            Ok(Self { host: *host, ptr })
        } else {
            Err(format!("Relay host string allocation failed: {status:?}"))
        }
    }

    pub fn as_ptr(&self) -> *const NemoRelayNativeString {
        self.ptr
    }
}

impl Drop for HostString {
    fn drop(&mut self) {
        unsafe { (self.host.string_free)(self.ptr) };
    }
}

pub fn read_string(
    host: &NemoRelayNativeHostApiV1,
    value: *const NemoRelayNativeString,
) -> Result<String, String> {
    if value.is_null() {
        return Err("Relay passed a null native string".into());
    }
    let len = unsafe { (host.string_len)(value) };
    let data = unsafe { (host.string_data)(value) };
    if data.is_null() && len != 0 {
        return Err("Relay passed an invalid native string".into());
    }
    let bytes = if len == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(data, len) }
    };
    std::str::from_utf8(bytes)
        .map(str::to_owned)
        .map_err(|error| error.to_string())
}

pub fn read_json(
    host: &NemoRelayNativeHostApiV1,
    value: *const NemoRelayNativeString,
) -> Result<Json, String> {
    serde_json::from_str(&read_string(host, value)?).map_err(|error| error.to_string())
}

pub async fn dispatch_buffered(
    host: &NemoRelayNativeHostApiV4,
    next: *const NemoRelayNativeAsyncNext,
    dispatch: &LlmDispatchRequestV2,
) -> Result<Json, LlmCallErrorV2> {
    let dispatch = HostString::json(&host.v3.v1, dispatch).map_err(internal_error)?;
    let (sender, receiver) = oneshot::channel::<LlmCallOutcomeV2>();
    let sender = Box::into_raw(Box::new(sender)).cast::<c_void>();
    let status = unsafe {
        (host.async_llm_next_invoke_result_v2)(next, dispatch.as_ptr(), buffered_result, sender)
    };
    if status != NemoRelayStatus::Ok {
        unsafe {
            drop(Box::from_raw(
                sender.cast::<oneshot::Sender<LlmCallOutcomeV2>>(),
            ))
        };
        return Err(internal_error(format!(
            "Relay rejected buffered dispatch: {status:?}"
        )));
    }
    match receiver.await {
        Ok(LlmCallOutcomeV2::Success { response }) => Ok(response),
        Ok(LlmCallOutcomeV2::Failure { error }) => Err(error),
        Err(_) => Err(internal_error(
            "Relay dropped the buffered dispatch callback".into(),
        )),
    }
}

pub async fn dispatch_passthrough_buffered(
    host: &NemoRelayNativeHostApiV4,
    next: *const NemoRelayNativeAsyncNext,
    request: &LlmRequest,
) -> Result<Json, String> {
    let request = HostString::json(&host.v3.v1, request)?;
    let (sender, receiver) = oneshot::channel::<Result<Json, String>>();
    let sender = Box::into_raw(Box::new(sender)).cast::<c_void>();
    let status = unsafe {
        (host.v3.async_next_invoke_result)(
            next,
            request.as_ptr(),
            passthrough_buffered_result,
            sender,
        )
    };
    if status != NemoRelayStatus::Ok {
        unsafe {
            drop(Box::from_raw(
                sender.cast::<oneshot::Sender<Result<Json, String>>>(),
            ))
        };
        return Err(format!("Relay rejected passthrough dispatch: {status:?}"));
    }
    receiver
        .await
        .map_err(|_| "Relay dropped the passthrough callback".to_string())?
}

pub async fn dispatch_passthrough_stream(
    host: &NemoRelayNativeHostApiV4,
    next: *const NemoRelayNativeAsyncNext,
    output: *const NemoRelayNativeAsyncStream,
    request: &LlmRequest,
) -> Result<(), String> {
    let request = HostString::json(&host.v3.v1, request)?;
    let (sender, receiver) = oneshot::channel();
    let state = Box::into_raw(Box::new(PassthroughStreamState {
        output: output as usize,
        sender: Some(sender),
    }))
    .cast::<c_void>();
    let status = unsafe {
        (host.v3.async_next_invoke_stream)(
            next,
            request.as_ptr(),
            output,
            passthrough_stream_result as NemoRelayNativeAsyncNextStreamCb,
            state,
        )
    };
    if status != NemoRelayStatus::Ok {
        unsafe { drop(Box::from_raw(state.cast::<PassthroughStreamState>())) };
        return Err(format!("Relay rejected passthrough stream: {status:?}"));
    }
    receiver
        .await
        .map_err(|_| "Relay dropped the passthrough stream callback".to_string())?
}

struct PassthroughStreamState {
    output: usize,
    sender: Option<oneshot::Sender<Result<(), String>>>,
}

unsafe extern "C" fn passthrough_stream_result(
    user_data: *mut c_void,
    chunk_json: *const NemoRelayNativeString,
    error: *const NemoRelayNativeString,
    done: bool,
) -> bool {
    let host = crate::host();
    let state = unsafe { &mut *user_data.cast::<PassthroughStreamState>() };
    let output = state.output as *const NemoRelayNativeAsyncStream;
    let result = if !error.is_null() {
        let message = read_string(&host.v3.v1, error)
            .unwrap_or_else(|_| "Relay passthrough stream failed".into());
        Some(Err(message))
    } else if done {
        let status = finish_stream(host, output);
        Some(if status == NemoRelayStatus::Ok {
            Ok(())
        } else {
            Err(format!(
                "Relay rejected passthrough stream finish: {status:?}"
            ))
        })
    } else {
        let result =
            read_json(&host.v3.v1, chunk_json).and_then(|chunk| push_stream(host, output, &chunk));
        if result.is_err() {
            Some(result)
        } else {
            None
        }
    };
    if let Some(result) = result {
        let mut state = unsafe { Box::from_raw(user_data.cast::<PassthroughStreamState>()) };
        if let Some(sender) = state.sender.take() {
            let _ = sender.send(result);
        }
        false
    } else {
        true
    }
}

unsafe extern "C" fn passthrough_buffered_result(
    user_data: *mut c_void,
    value_json: *const NemoRelayNativeString,
    error: *const NemoRelayNativeString,
) {
    let sender =
        unsafe { Box::from_raw(user_data.cast::<oneshot::Sender<Result<Json, String>>>()) };
    let host = crate::host();
    let result = if error.is_null() {
        read_json(&host.v3.v1, value_json)
    } else {
        Err(read_string(&host.v3.v1, error)
            .unwrap_or_else(|_| "Relay passthrough dispatch failed".into()))
    };
    let _ = sender.send(result);
}

unsafe extern "C" fn buffered_result(
    user_data: *mut c_void,
    outcome_json: *const NemoRelayNativeString,
) {
    let sender = unsafe { Box::from_raw(user_data.cast::<oneshot::Sender<LlmCallOutcomeV2>>()) };
    let host = crate::host();
    let outcome = read_json(&host.v3.v1, outcome_json)
        .and_then(|value| serde_json::from_value(value).map_err(|error| error.to_string()))
        .unwrap_or_else(|error| LlmCallOutcomeV2::Failure {
            error: internal_error(format!("invalid Relay buffered outcome: {error}")),
        });
    let _ = sender.send(outcome);
}

pub async fn dispatch_stream(
    host: &NemoRelayNativeHostApiV4,
    next: *const NemoRelayNativeAsyncNext,
    output_stream: *const NemoRelayNativeAsyncStream,
    dispatch: &LlmDispatchRequestV2,
) -> Result<ProviderJsonStream, LlmCallErrorV2> {
    let dispatch = HostString::json(&host.v3.v1, dispatch).map_err(internal_error)?;
    let (sender, receiver) = oneshot::channel::<Result<usize, LlmCallErrorV2>>();
    let sender = Box::into_raw(Box::new(sender)).cast::<c_void>();
    let status = unsafe {
        (host.async_llm_next_open_stream_v2)(
            next,
            dispatch.as_ptr(),
            output_stream,
            provider_stream_open,
            sender,
        )
    };
    if status != NemoRelayStatus::Ok {
        unsafe {
            drop(Box::from_raw(
                sender.cast::<oneshot::Sender<Result<usize, LlmCallErrorV2>>>(),
            ))
        };
        return Err(internal_error(format!(
            "Relay rejected streaming dispatch: {status:?}"
        )));
    }
    let stream = receiver
        .await
        .map_err(|_| internal_error("Relay dropped the stream-open callback".into()))??;
    Ok(ProviderJsonStream {
        host: *host,
        stream: stream as *const NemoRelayNativeLlmStreamV2,
        pending: None,
        done: false,
    })
}

unsafe extern "C" fn provider_stream_open(
    user_data: *mut c_void,
    stream: *const NemoRelayNativeLlmStreamV2,
    error_json: *const NemoRelayNativeString,
) {
    let sender = unsafe {
        Box::from_raw(user_data.cast::<oneshot::Sender<Result<usize, LlmCallErrorV2>>>())
    };
    let host = crate::host();
    let result = if error_json.is_null() {
        if stream.is_null() {
            Err(internal_error(
                "Relay returned neither a provider stream nor an error".into(),
            ))
        } else {
            Ok(stream as usize)
        }
    } else {
        read_json(&host.v3.v1, error_json)
            .and_then(|value| serde_json::from_value(value).map_err(|error| error.to_string()))
            .map_err(|error| internal_error(format!("invalid Relay stream-open error: {error}")))
    };
    let _ = sender.send(result);
}

pub struct ProviderJsonStream {
    host: NemoRelayNativeHostApiV4,
    stream: *const NemoRelayNativeLlmStreamV2,
    pending: Option<oneshot::Receiver<LlmStreamEventV2>>,
    done: bool,
}

unsafe impl Send for ProviderJsonStream {}

impl Stream for ProviderJsonStream {
    type Item = Result<Json, LlmCallErrorV2>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.done {
            return Poll::Ready(None);
        }
        if self.pending.is_none() {
            let (sender, receiver) = oneshot::channel();
            let sender = Box::into_raw(Box::new(sender)).cast::<c_void>();
            let status = unsafe {
                (self.host.async_llm_stream_next_v2)(self.stream, provider_stream_next, sender)
            };
            if status != NemoRelayStatus::Ok {
                unsafe {
                    drop(Box::from_raw(
                        sender.cast::<oneshot::Sender<LlmStreamEventV2>>(),
                    ))
                };
                self.done = true;
                return Poll::Ready(Some(Err(internal_error(format!(
                    "Relay rejected provider stream next: {status:?}"
                )))));
            }
            self.pending = Some(receiver);
        }
        let receiver = self.pending.as_mut().expect("pending receiver was set");
        match Pin::new(receiver).poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Ok(LlmStreamEventV2::Chunk { chunk })) => {
                self.pending = None;
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Ok(LlmStreamEventV2::Failure { error })) => {
                self.pending = None;
                self.done = true;
                Poll::Ready(Some(Err(error)))
            }
            Poll::Ready(Ok(LlmStreamEventV2::Done)) => {
                self.pending = None;
                self.done = true;
                Poll::Ready(None)
            }
            Poll::Ready(Err(_)) => {
                self.pending = None;
                self.done = true;
                Poll::Ready(Some(Err(internal_error(
                    "Relay dropped the provider stream next callback".into(),
                ))))
            }
        }
    }
}

impl Drop for ProviderJsonStream {
    fn drop(&mut self) {
        if !self.done {
            unsafe { (self.host.async_llm_stream_cancel_v2)(self.stream) };
        }
        unsafe { (self.host.async_llm_stream_release_v2)(self.stream) };
    }
}

unsafe extern "C" fn provider_stream_next(
    user_data: *mut c_void,
    event_json: *const NemoRelayNativeString,
) {
    let sender = unsafe { Box::from_raw(user_data.cast::<oneshot::Sender<LlmStreamEventV2>>()) };
    let host = crate::host();
    let event = read_json(&host.v3.v1, event_json)
        .and_then(|value| serde_json::from_value(value).map_err(|error| error.to_string()))
        .unwrap_or_else(|error| LlmStreamEventV2::Failure {
            error: internal_error(format!("invalid Relay stream event: {error}")),
        });
    let _ = sender.send(event);
}

pub fn resolve_completion(
    host: &NemoRelayNativeHostApiV4,
    completion: *const NemoRelayNativeAsyncCompletion,
    value: &Json,
) -> NemoRelayStatus {
    match HostString::json(&host.v3.v1, value) {
        Ok(value) => unsafe { (host.v3.async_completion_resolve_json)(completion, value.as_ptr()) },
        Err(_) => NemoRelayStatus::Internal,
    }
}

pub fn reject_completion(
    host: &NemoRelayNativeHostApiV4,
    completion: *const NemoRelayNativeAsyncCompletion,
    message: &str,
) -> NemoRelayStatus {
    match HostString::text(&host.v3.v1, message) {
        Ok(message) => unsafe { (host.v3.async_completion_reject)(completion, message.as_ptr()) },
        Err(_) => NemoRelayStatus::Internal,
    }
}

pub fn push_stream(
    host: &NemoRelayNativeHostApiV4,
    stream: *const NemoRelayNativeAsyncStream,
    value: &Json,
) -> Result<(), String> {
    let value = HostString::json(&host.v3.v1, value)?;
    loop {
        if unsafe { (host.v3.async_stream_is_cancelled)(stream) } {
            return Err("Relay caller cancelled the output stream".into());
        }
        match unsafe { (host.v3.async_stream_push_json)(stream, value.as_ptr()) } {
            NemoRelayStatus::Ok => return Ok(()),
            NemoRelayStatus::Internal => std::thread::yield_now(),
            status => return Err(format!("Relay rejected output stream event: {status:?}")),
        }
    }
}

pub fn finish_stream(
    host: &NemoRelayNativeHostApiV4,
    stream: *const NemoRelayNativeAsyncStream,
) -> NemoRelayStatus {
    unsafe { (host.v3.async_stream_finish)(stream) }
}

pub fn reject_stream(
    host: &NemoRelayNativeHostApiV4,
    stream: *const NemoRelayNativeAsyncStream,
    message: &str,
) -> NemoRelayStatus {
    match HostString::text(&host.v3.v1, message) {
        Ok(message) => unsafe { (host.v3.async_stream_reject)(stream, message.as_ptr()) },
        Err(_) => NemoRelayStatus::Internal,
    }
}

pub unsafe fn release_next(host: &NemoRelayNativeHostApiV4, next: *const NemoRelayNativeAsyncNext) {
    unsafe { (host.v3.async_next_release)(next) };
}

pub unsafe fn release_stream(
    host: &NemoRelayNativeHostApiV4,
    stream: *const NemoRelayNativeAsyncStream,
) {
    unsafe { (host.v3.async_stream_release)(stream) };
}

pub fn internal_error(message: String) -> LlmCallErrorV2 {
    LlmCallErrorV2::Internal { message }
}
