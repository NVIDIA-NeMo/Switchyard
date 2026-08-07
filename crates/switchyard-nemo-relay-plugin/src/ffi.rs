// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Small ownership wrapper around Relay's generic C host-table v3 hooks.
//!
//! The plugin manifest remains native API v1. Relay 0.7 supplies the appended
//! v3 host table to rebuilt v1 plugins, which lets this crate return `Pending`
//! and settle work from its own runtime without a targeted-continuation ABI.

use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use nemo_relay_plugin::{
    Json, LlmRequest, NemoRelayNativeAsyncCompletion, NemoRelayNativeAsyncNext,
    NemoRelayNativeAsyncNextStreamCb, NemoRelayNativeAsyncStream, NemoRelayNativeHostApiV1,
    NemoRelayNativeHostApiV3, NemoRelayNativeScopeHandle, NemoRelayNativeString, NemoRelayStatus,
};
use serde::Serialize;
use tokio::sync::{mpsc, oneshot};

const BACKPRESSURE_POLL: Duration = Duration::from_millis(1);
const CANCELLATION_POLL: Duration = Duration::from_millis(10);
const MAX_PASSTHROUGH_BUFFER_BYTES: usize = 8 * 1024 * 1024;
const MAX_PASSTHROUGH_BUFFER_EVENTS: usize = 256;

pub(crate) struct HostString {
    host: NemoRelayNativeHostApiV1,
    ptr: *mut NemoRelayNativeString,
}

// Host strings are immutable allocations owned by Relay's thread-safe host table.
unsafe impl Send for HostString {}

impl HostString {
    pub(crate) fn json(
        host: &NemoRelayNativeHostApiV1,
        value: &impl Serialize,
    ) -> Result<Self, String> {
        let value = serde_json::to_string(value).map_err(|error| error.to_string())?;
        Self::text(host, &value)
    }

    pub(crate) fn text(host: &NemoRelayNativeHostApiV1, value: &str) -> Result<Self, String> {
        let mut ptr = ptr::null_mut();
        let status = unsafe { (host.string_new)(value.as_ptr(), value.len(), &mut ptr) };
        if status == NemoRelayStatus::Ok && !ptr.is_null() {
            Ok(Self { host: *host, ptr })
        } else {
            Err(format!("Relay host string allocation failed: {status:?}"))
        }
    }

    pub(crate) fn as_ptr(&self) -> *const NemoRelayNativeString {
        self.ptr
    }
}

impl Drop for HostString {
    fn drop(&mut self) {
        unsafe { (self.host.string_free)(self.ptr) };
    }
}

pub(crate) fn read_string(
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

pub(crate) fn read_json(
    host: &NemoRelayNativeHostApiV1,
    value: *const NemoRelayNativeString,
) -> Result<Json, String> {
    serde_json::from_str(&read_string(host, value)?).map_err(|error| error.to_string())
}

/// Captures the current Relay scope as an explicit event parent.
///
/// Async plugin work runs on a plugin-owned thread, so relying on thread-local
/// scope state would orphan its marks. The host handle is a cloned scope handle
/// and remains valid until this guard is dropped.
pub(crate) struct ParentScope {
    host: NemoRelayNativeHostApiV1,
    ptr: *mut NemoRelayNativeScopeHandle,
}

unsafe impl Send for ParentScope {}
unsafe impl Sync for ParentScope {}

impl ParentScope {
    pub(crate) fn capture(host: &NemoRelayNativeHostApiV1) -> Option<Self> {
        let mut ptr = ptr::null_mut();
        let status = unsafe { (host.scope_get_current)(&mut ptr) };
        (status == NemoRelayStatus::Ok && !ptr.is_null()).then_some(Self { host: *host, ptr })
    }

    pub(crate) fn emit_mark(&self, name: &str, data: &Json, metadata: &Json) -> Result<(), String> {
        let name = HostString::text(&self.host, name)?;
        let data = HostString::json(&self.host, data)?;
        let metadata = HostString::json(&self.host, metadata)?;
        let status = unsafe {
            (self.host.emit_mark)(
                name.as_ptr(),
                self.ptr,
                data.as_ptr(),
                metadata.as_ptr(),
                ptr::null(),
            )
        };
        if status == NemoRelayStatus::Ok {
            Ok(())
        } else {
            Err(format!(
                "Relay rejected Switchyard routing mark: {status:?}"
            ))
        }
    }
}

impl Drop for ParentScope {
    fn drop(&mut self) {
        unsafe { (self.host.scope_handle_free)(self.ptr) };
    }
}

pub(crate) fn invoke_next_buffered(
    host: &NemoRelayNativeHostApiV3,
    next: usize,
    completion: usize,
    request: &LlmRequest,
) -> Result<(), String> {
    let request = HostString::json(&host.v1, request)?;
    let status = unsafe {
        (host.async_next_invoke)(
            next as *const NemoRelayNativeAsyncNext,
            request.as_ptr(),
            completion as *const NemoRelayNativeAsyncCompletion,
        )
    };
    if status == NemoRelayStatus::Ok {
        Ok(())
    } else {
        Err(format!("Relay rejected buffered pass-through: {status:?}"))
    }
}

enum DownstreamStreamItem {
    Chunk { value: Json, encoded_bytes: usize },
}

struct DownstreamStreamState {
    host: NemoRelayNativeHostApiV1,
    sender: mpsc::Sender<DownstreamStreamItem>,
    terminal: Option<oneshot::Sender<Result<(), String>>>,
    queued_bytes: Arc<AtomicUsize>,
}

pub(crate) async fn invoke_next_stream(
    host: &NemoRelayNativeHostApiV3,
    next: usize,
    output: usize,
    request: &LlmRequest,
) -> Result<(), String> {
    let request = HostString::json(&host.v1, request)?;
    let (sender, mut receiver) = mpsc::channel(MAX_PASSTHROUGH_BUFFER_EVENTS);
    let (terminal, terminal_result) = oneshot::channel();
    let queued_bytes = Arc::new(AtomicUsize::new(0));
    let state = Box::into_raw(Box::new(DownstreamStreamState {
        host: host.v1,
        sender,
        terminal: Some(terminal),
        queued_bytes: Arc::clone(&queued_bytes),
    }))
    .cast::<c_void>();
    let status = unsafe {
        (host.async_next_invoke_stream)(
            next as *const NemoRelayNativeAsyncNext,
            request.as_ptr(),
            output as *const NemoRelayNativeAsyncStream,
            downstream_stream_result as NemoRelayNativeAsyncNextStreamCb,
            state,
        )
    };
    if status != NemoRelayStatus::Ok {
        unsafe { drop(Box::from_raw(state.cast::<DownstreamStreamState>())) };
        return Err(format!("Relay rejected streaming pass-through: {status:?}"));
    }

    while let Some(item) = receiver.recv().await {
        match item {
            DownstreamStreamItem::Chunk {
                value,
                encoded_bytes,
            } => {
                let result = push_stream(host, output, &value).await;
                queued_bytes.fetch_sub(encoded_bytes, Ordering::AcqRel);
                result?;
            }
        }
    }
    terminal_result
        .await
        .unwrap_or_else(|_| Err("Relay dropped the streaming pass-through callback".into()))
}

unsafe extern "C" fn downstream_stream_result(
    user_data: *mut c_void,
    chunk_json: *const NemoRelayNativeString,
    error: *const NemoRelayNativeString,
    done: bool,
) -> bool {
    if !error.is_null() {
        let state = unsafe { Box::from_raw(user_data.cast::<DownstreamStreamState>()) };
        let error = read_string(&state.host, error)
            .unwrap_or_else(|_| "Relay streaming pass-through failed".into());
        settle_downstream_stream(state, Err(error));
        return false;
    }
    if done {
        let state = unsafe { Box::from_raw(user_data.cast::<DownstreamStreamState>()) };
        settle_downstream_stream(state, Ok(()));
        return false;
    }

    let state = unsafe { &*user_data.cast::<DownstreamStreamState>() };
    let parsed = read_string(&state.host, chunk_json).and_then(|encoded| {
        let encoded_bytes = encoded.len();
        let value = serde_json::from_str(&encoded).map_err(|error| error.to_string())?;
        Ok((value, encoded_bytes))
    });
    let (value, encoded_bytes) = match parsed {
        Ok(parsed) => parsed,
        Err(error) => {
            let state = unsafe { Box::from_raw(user_data.cast::<DownstreamStreamState>()) };
            settle_downstream_stream(state, Err(error));
            return false;
        }
    };
    if !reserve_buffer_bytes(&state.queued_bytes, encoded_bytes) {
        let state = unsafe { Box::from_raw(user_data.cast::<DownstreamStreamState>()) };
        settle_downstream_stream(
            state,
            Err(format!(
                "Relay streaming pass-through exceeded its {}-byte queued payload limit",
                MAX_PASSTHROUGH_BUFFER_BYTES
            )),
        );
        return false;
    }

    match state.sender.try_send(DownstreamStreamItem::Chunk {
        value,
        encoded_bytes,
    }) {
        Ok(()) => true,
        Err(error) => {
            let (item, message) = match error {
                mpsc::error::TrySendError::Full(item) => (
                    item,
                    format!(
                        "Relay streaming pass-through exceeded its {MAX_PASSTHROUGH_BUFFER_EVENTS}-event queue"
                    ),
                ),
                mpsc::error::TrySendError::Closed(item) => (
                    item,
                    "Relay dropped the streaming pass-through receiver".into(),
                ),
            };
            let encoded_bytes = item.encoded_bytes();
            state
                .queued_bytes
                .fetch_sub(encoded_bytes, Ordering::AcqRel);
            let state = unsafe { Box::from_raw(user_data.cast::<DownstreamStreamState>()) };
            settle_downstream_stream(state, Err(message));
            false
        }
    }
}

impl DownstreamStreamItem {
    fn encoded_bytes(&self) -> usize {
        match self {
            Self::Chunk { encoded_bytes, .. } => *encoded_bytes,
        }
    }
}

fn settle_downstream_stream(mut state: Box<DownstreamStreamState>, result: Result<(), String>) {
    if let Some(terminal) = state.terminal.take() {
        let _ = terminal.send(result);
    }
}

fn reserve_buffer_bytes(queued: &AtomicUsize, encoded_bytes: usize) -> bool {
    queued
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
            current
                .checked_add(encoded_bytes)
                .filter(|next| *next <= MAX_PASSTHROUGH_BUFFER_BYTES)
        })
        .is_ok()
}

pub(crate) async fn wait_for_completion_cancellation(
    host: &NemoRelayNativeHostApiV3,
    completion: usize,
) {
    while !completion_cancelled(host, completion as *const NemoRelayNativeAsyncCompletion) {
        tokio::time::sleep(CANCELLATION_POLL).await;
    }
}

pub(crate) async fn wait_for_stream_cancellation(host: &NemoRelayNativeHostApiV3, stream: usize) {
    while !unsafe { (host.async_stream_is_cancelled)(stream as *const NemoRelayNativeAsyncStream) }
    {
        tokio::time::sleep(CANCELLATION_POLL).await;
    }
}

pub(crate) fn completion_cancelled(
    host: &NemoRelayNativeHostApiV3,
    completion: *const NemoRelayNativeAsyncCompletion,
) -> bool {
    unsafe { (host.async_completion_is_cancelled)(completion) }
}

pub(crate) fn resolve_completion(
    host: &NemoRelayNativeHostApiV3,
    completion: *const NemoRelayNativeAsyncCompletion,
    value: &Json,
) -> NemoRelayStatus {
    match HostString::json(&host.v1, value) {
        Ok(value) => unsafe { (host.async_completion_resolve_json)(completion, value.as_ptr()) },
        Err(_) => NemoRelayStatus::Internal,
    }
}

pub(crate) fn reject_completion(
    host: &NemoRelayNativeHostApiV3,
    completion: *const NemoRelayNativeAsyncCompletion,
    message: &str,
) -> NemoRelayStatus {
    match HostString::text(&host.v1, message) {
        Ok(message) => unsafe { (host.async_completion_reject)(completion, message.as_ptr()) },
        Err(_) => NemoRelayStatus::Internal,
    }
}

pub(crate) async fn push_stream(
    host: &NemoRelayNativeHostApiV3,
    stream: usize,
    value: &Json,
) -> Result<(), String> {
    let value = HostString::json(&host.v1, value)?;
    loop {
        if unsafe { (host.async_stream_is_cancelled)(stream as *const NemoRelayNativeAsyncStream) }
        {
            return Err("Relay caller cancelled the output stream".into());
        }
        match unsafe {
            (host.async_stream_push_json)(
                stream as *const NemoRelayNativeAsyncStream,
                value.as_ptr(),
            )
        } {
            NemoRelayStatus::Ok => return Ok(()),
            // Native API v1 reports its bounded queue's WouldBlock state as Internal.
            NemoRelayStatus::Internal => tokio::time::sleep(BACKPRESSURE_POLL).await,
            status => return Err(format!("Relay rejected output stream event: {status:?}")),
        }
    }
}

pub(crate) fn finish_stream(
    host: &NemoRelayNativeHostApiV3,
    stream: *const NemoRelayNativeAsyncStream,
) -> NemoRelayStatus {
    unsafe { (host.async_stream_finish)(stream) }
}

pub(crate) async fn reject_stream(
    host: &NemoRelayNativeHostApiV3,
    stream: usize,
    message: &str,
) -> NemoRelayStatus {
    let Ok(message) = HostString::text(&host.v1, message) else {
        return NemoRelayStatus::Internal;
    };
    loop {
        if unsafe { (host.async_stream_is_cancelled)(stream as *const NemoRelayNativeAsyncStream) }
        {
            return NemoRelayStatus::InvalidArg;
        }
        match unsafe {
            (host.async_stream_reject)(
                stream as *const NemoRelayNativeAsyncStream,
                message.as_ptr(),
            )
        } {
            NemoRelayStatus::Internal => tokio::time::sleep(BACKPRESSURE_POLL).await,
            status => return status,
        }
    }
}

pub(crate) unsafe fn release_completion(
    host: &NemoRelayNativeHostApiV3,
    completion: *const NemoRelayNativeAsyncCompletion,
) {
    unsafe { (host.async_completion_release)(completion) };
}

pub(crate) unsafe fn release_next(
    host: &NemoRelayNativeHostApiV3,
    next: *const NemoRelayNativeAsyncNext,
) {
    unsafe { (host.async_next_release)(next) };
}

pub(crate) unsafe fn release_stream(
    host: &NemoRelayNativeHostApiV3,
    stream: *const NemoRelayNativeAsyncStream,
) {
    unsafe { (host.async_stream_release)(stream) };
}
