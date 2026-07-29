// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Private adapter for Rust-owned chat response values.

use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

use futures_util::{stream, Stream, StreamExt};
use parking_lot::Mutex;
use pyo3::exceptions::{PyRuntimeError, PyStopAsyncIteration, PyValueError};
use pyo3::prelude::*;
use switchyard_components::{BoxResponseStream, ChatResponse, ChatResponseType, StreamEvent};

use crate::errors::py_core_error;
use crate::py_serde::{value_from_python, value_to_python};

/// Converts a public Python response into the native response value.
pub(crate) fn response_from_python(value: &Bound<'_, PyAny>) -> PyResult<ChatResponse> {
    let response_type = response_type_from_python(&value.getattr("response_type")?)?;
    Ok(match response_type {
        ChatResponseType::OpenAiCompletion => {
            ChatResponse::openai_completion(value_from_python(&value.getattr("_body")?)?)
        }
        ChatResponseType::OpenAiResponsesCompletion => {
            ChatResponse::openai_responses_completion(value_from_python(&value.getattr("_body")?)?)
        }
        ChatResponseType::AnthropicCompletion => {
            ChatResponse::anthropic_completion(value_from_python(&value.getattr("_body")?)?)
        }
        ChatResponseType::OpenAiStream => ChatResponse::OpenAiStream(stream_from_python(value)?),
        ChatResponseType::OpenAiResponsesStream => {
            ChatResponse::OpenAiResponsesStream(stream_from_python(value)?)
        }
        ChatResponseType::AnthropicStream => {
            ChatResponse::AnthropicStream(stream_from_python(value)?)
        }
    })
}

/// Converts a native response into the public Python response value.
pub(crate) fn response_to_python(py: Python<'_>, response: ChatResponse) -> PyResult<Py<PyAny>> {
    let core = py.import("switchyard_rust.core")?;
    let (factory, payload) = match response {
        ChatResponse::OpenAiCompletion(response) => (
            "openai_completion",
            value_to_python(py, &response.into_body())?,
        ),
        ChatResponse::OpenAiResponsesCompletion(response) => (
            "openai_responses_completion",
            value_to_python(py, &response.into_body())?,
        ),
        ChatResponse::AnthropicCompletion(response) => (
            "anthropic_completion",
            value_to_python(py, &response.into_body())?,
        ),
        ChatResponse::OpenAiStream(stream) => {
            ("openai_stream", stream_to_python(py, &core, stream)?)
        }
        ChatResponse::OpenAiResponsesStream(stream) => (
            "openai_responses_stream",
            stream_to_python(py, &core, stream)?,
        ),
        ChatResponse::AnthropicStream(stream) => {
            ("anthropic_stream", stream_to_python(py, &core, stream)?)
        }
    };
    core.getattr("ChatResponse")?
        .call_method1(factory, (payload,))
        .map(Bound::unbind)
}

fn response_type_from_python(value: &Bound<'_, PyAny>) -> PyResult<ChatResponseType> {
    let raw = if let Ok(value_attr) = value.getattr("value") {
        value_attr.extract::<String>()?
    } else {
        value.extract::<String>()?
    };
    match raw.as_str() {
        "openai_completion" => Ok(ChatResponseType::OpenAiCompletion),
        "openai_stream" => Ok(ChatResponseType::OpenAiStream),
        "openai_responses_completion" => Ok(ChatResponseType::OpenAiResponsesCompletion),
        "openai_responses_stream" => Ok(ChatResponseType::OpenAiResponsesStream),
        "anthropic_completion" => Ok(ChatResponseType::AnthropicCompletion),
        "anthropic_stream" => Ok(ChatResponseType::AnthropicStream),
        _ => Err(PyValueError::new_err(format!(
            "Unknown response type: {raw:?}"
        ))),
    }
}

fn stream_from_python(value: &Bound<'_, PyAny>) -> PyResult<BoxResponseStream> {
    let native = value.getattr("stream")?.getattr("_native")?;
    native
        .extract::<PyRef<'_, PyResponseStream>>()?
        .take_core_stream()
}

fn stream_to_python(
    py: Python<'_>,
    core: &Bound<'_, PyModule>,
    stream: BoxResponseStream,
) -> PyResult<Py<PyAny>> {
    let native = Py::new(py, PyResponseStream::from_core_stream(stream))?;
    core.getattr("ChatResponseStream")?
        .call_method1("_from_native", (native,))
        .map(Bound::unbind)
}

struct PyResponseStreamSource {
    source: Py<PyAny>,
    iterator: Option<Py<PyAny>>,
    done: bool,
}

type BoxPyResponseStream = std::pin::Pin<Box<dyn Stream<Item = PyResult<Py<PyAny>>> + Send>>;

#[pyclass(name = "_NativeChatResponseStream")]
pub(crate) struct PyResponseStream {
    stream: Arc<tokio::sync::Mutex<Option<BoxPyResponseStream>>>,
    taps: Arc<Mutex<Vec<Py<PyAny>>>>,
    maps: Arc<Mutex<Vec<Py<PyAny>>>>,
    on_complete: Arc<Mutex<Vec<Py<PyAny>>>>,
    consumed: Arc<AtomicBool>,
    completed: Arc<AtomicBool>,
    // The original Python stream object (e.g. the OpenAI SDK ``AsyncStream``)
    // when this stream was built from a Python source. Retained so ``aclose``
    // can release the upstream response — and the pooled connection it holds —
    // on early termination. ``None`` for Rust-native streams, which own no
    // closable Python resource.
    source: Option<Py<PyAny>>,
}

impl PyResponseStream {
    fn new(source: Py<PyAny>) -> Self {
        let retained = Python::attach(|py| source.clone_ref(py));
        let mut stream = Self::from_stream(stream_from_python_source(source));
        stream.source = Some(retained);
        stream
    }

    fn from_stream(stream: BoxPyResponseStream) -> Self {
        Self {
            stream: Arc::new(tokio::sync::Mutex::new(Some(stream))),
            taps: Arc::new(Mutex::new(Vec::new())),
            maps: Arc::new(Mutex::new(Vec::new())),
            on_complete: Arc::new(Mutex::new(Vec::new())),
            consumed: Arc::new(AtomicBool::new(false)),
            completed: Arc::new(AtomicBool::new(false)),
            source: None,
        }
    }

    fn from_core_stream(stream: BoxResponseStream) -> Self {
        Self::from_stream(Box::pin(stream.map(|event| {
            Python::attach(|py| match event {
                Ok(StreamEvent::Json(value)) => value_to_python(py, &value),
                Ok(StreamEvent::Text(value)) => Ok(value.into_pyobject(py)?.unbind().into_any()),
                Err(error) => Err(py_core_error(error)),
            })
        })))
    }

    fn take_core_stream(&self) -> PyResult<BoxResponseStream> {
        if self.consumed.swap(true, Ordering::AcqRel) {
            return Err(PyRuntimeError::new_err(
                "ChatResponseStream has already been consumed",
            ));
        }
        let stream = Arc::clone(&self.stream);
        let taps = Arc::clone(&self.taps);
        let maps = Arc::clone(&self.maps);
        let on_complete = Arc::clone(&self.on_complete);
        let completed = Arc::clone(&self.completed);
        let core: BoxResponseStream = Box::pin(stream::unfold(
            (stream, taps, maps, on_complete, completed),
            |(stream, taps, maps, on_complete, completed)| async move {
                let event = next_stream_item_with_callbacks(
                    Arc::clone(&stream),
                    Arc::clone(&taps),
                    Arc::clone(&maps),
                    Arc::clone(&on_complete),
                    Arc::clone(&completed),
                )
                .await;
                event.map(|event| {
                    let event = match event {
                        Ok(event) => Python::attach(|py| stream_event_from_python(py, event)),
                        Err(error) => Err(error),
                    };
                    (
                        event.map_err(|error| {
                            switchyard_components::SwitchyardError::Processor(error.to_string())
                        }),
                        (stream, taps, maps, on_complete, completed),
                    )
                })
            },
        ));
        // Preserve close ownership across the Python -> core -> Python round
        // trip: ``from_core_stream`` rebuilds a ``PyResponseStream`` with no
        // ``source``, so its ``aclose`` could not release the upstream SDK
        // stream. Carrying the source on the core stream — and closing it when
        // that stream is dropped — keeps the connection releasable even after
        // response processors re-wrap the stream.
        match self.source.as_ref() {
            Some(source) => {
                let source = Python::attach(|py| source.clone_ref(py));
                Ok(Box::pin(SourceClosingStream::new(core, source)))
            }
            None => Ok(core),
        }
    }
}

#[pymethods]
impl PyResponseStream {
    #[new]
    fn py_new(source: &Bound<'_, PyAny>) -> Self {
        Self::new(source.clone().unbind())
    }

    fn tap(slf: PyRef<'_, Self>, callback: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        push_callback(&slf.taps, callback)?;
        Ok(slf.into_pyobject(callback.py())?.unbind().into_any())
    }

    fn map(slf: PyRef<'_, Self>, callback: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        push_callback(&slf.maps, callback)?;
        Ok(slf.into_pyobject(callback.py())?.unbind().into_any())
    }

    fn on_complete(slf: PyRef<'_, Self>, callback: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        push_callback(&slf.on_complete, callback)?;
        Ok(slf.into_pyobject(callback.py())?.unbind().into_any())
    }

    fn __aiter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyAny>> {
        if slf.consumed.swap(true, Ordering::AcqRel) {
            return Err(PyRuntimeError::new_err(
                "ChatResponseStream has already been consumed",
            ));
        }
        let py = slf.py();
        Ok(slf.into_pyobject(py)?.unbind().into_any())
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let stream = Arc::clone(&self.stream);
        let taps = Arc::clone(&self.taps);
        let maps = Arc::clone(&self.maps);
        let on_complete = Arc::clone(&self.on_complete);
        let completed = Arc::clone(&self.completed);

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            match next_stream_item_with_callbacks(stream, taps, maps, on_complete, completed).await
            {
                Some(event) => event,
                None => Err(PyStopAsyncIteration::new_err(())),
            }
        })
    }

    fn __repr__(&self) -> &'static str {
        "ChatResponseStream()"
    }

    /// Release the upstream stream and its underlying connection.
    ///
    /// Streaming proxies must close the upstream response when iteration ends
    /// early (client disconnect, mid-stream error) — otherwise the SDK
    /// ``AsyncStream``'s httpx response is never closed and its pooled
    /// connection leaks, exhausting the pool and pinning buffers. Marks the
    /// stream consumed and drops the inner adapter; idempotent and safe to
    /// call after completion.
    ///
    /// The upstream source is released by **two** complementary paths, because
    /// the source is reachable in only one of two stream shapes:
    /// * Built directly from a Python source (``ChatResponseStream(sdk_stream)``,
    ///   ``source`` is set): closed here, best-effort, via ``close``/``aclose``.
    /// * Rebuilt from a core stream by ``from_core_stream`` after the
    ///   ``Switchyard.call`` round trip (``source`` is ``None``): the source was
    ///   moved onto the core stream by ``take_core_stream`` as a
    ///   ``SourceClosingStream``; dropping the inner adapter here drops that
    ///   wrapper, which closes the source on drop. This is the production path
    ///   where response processors re-wrap the core stream and ``source``
    ///   cannot be recovered on this object.
    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.consumed.store(true, Ordering::Release);
        let stream = Arc::clone(&self.stream);
        let completed = Arc::clone(&self.completed);
        let source = self.source.as_ref().map(|source| source.clone_ref(py));
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            // Drop the inner adapter so it releases its borrow of the upstream
            // iterator; subsequent ``__anext__`` then yields StopAsyncIteration.
            {
                let mut guard = stream.lock().await;
                *guard = None;
            }
            completed.store(true, Ordering::Release);
            if let Some(source) = source {
                close_python_source(source).await;
            }
            Ok(())
        })
    }
}

fn push_callback(
    callbacks: &Arc<Mutex<Vec<Py<PyAny>>>>,
    callback: &Bound<'_, PyAny>,
) -> PyResult<()> {
    callbacks.lock().push(callback.clone().unbind());
    Ok(())
}

async fn next_stream_item(
    stream: Arc<tokio::sync::Mutex<Option<BoxPyResponseStream>>>,
) -> Option<PyResult<Py<PyAny>>> {
    let mut guard = stream.lock().await;
    match guard.as_mut() {
        Some(response_stream) => {
            let event = response_stream.next().await;
            if event.is_none() {
                *guard = None;
            }
            event
        }
        None => None,
    }
}

async fn next_stream_item_with_callbacks(
    stream: Arc<tokio::sync::Mutex<Option<BoxPyResponseStream>>>,
    taps: Arc<Mutex<Vec<Py<PyAny>>>>,
    maps: Arc<Mutex<Vec<Py<PyAny>>>>,
    on_complete: Arc<Mutex<Vec<Py<PyAny>>>>,
    completed: Arc<AtomicBool>,
) -> Option<PyResult<Py<PyAny>>> {
    match next_stream_item(stream).await {
        Some(Ok(event)) => {
            run_taps(taps, &event).await;
            Some(run_maps(maps, event).await)
        }
        Some(Err(error)) => Some(Err(error)),
        None => {
            run_completion_once(on_complete, completed).await;
            None
        }
    }
}

fn stream_from_python_source(source: Py<PyAny>) -> BoxPyResponseStream {
    Box::pin(stream::unfold(
        PyResponseStreamSource {
            source,
            iterator: None,
            done: false,
        },
        |mut state| async move {
            if state.done {
                return None;
            }
            let future = Python::attach(|py| {
                let iterator = match &state.iterator {
                    Some(iterator) => iterator.clone_ref(py),
                    None => {
                        let iterator = state.source.bind(py).call_method0("__aiter__")?.unbind();
                        state.iterator = Some(iterator.clone_ref(py));
                        iterator
                    }
                };
                let awaitable = iterator.bind(py).call_method0("__anext__")?;
                pyo3_async_runtimes::tokio::into_future(awaitable)
            });
            match future {
                Ok(future) => match future.await {
                    Ok(event) => Some((Ok(event), state)),
                    Err(error) if is_stop_async_iteration(&error) => None,
                    Err(error) => {
                        state.done = true;
                        Some((Err(error), state))
                    }
                },
                Err(error) => {
                    state.done = true;
                    Some((Err(error), state))
                }
            }
        },
    ))
}

/// Core stream that closes its originating Python source when dropped.
///
/// ``take_core_stream`` converts a Python-backed ``PyResponseStream`` into a
/// Rust-core ``BoxResponseStream`` for runtime processing; ``from_core_stream``
/// later rebuilds a Python wrapper that no longer references the source, so the
/// rebuilt wrapper's ``aclose`` can no longer release the upstream response.
/// Wrapping the core stream here preserves close ownership across that
/// conversion: dropping the stream — directly, or via a response processor that
/// re-wrapped it — schedules a best-effort close of the source so the SDK
/// stream and its pooled connection are released on early termination.
struct SourceClosingStream {
    inner: BoxResponseStream,
    source: Option<Py<PyAny>>,
    // The asyncio event loop + contextvars captured at construction. The
    // source's close is a Python coroutine that must be driven on *that* loop,
    // not on a bare tokio worker thread; the loop also lets the nested
    // ``aclose`` re-enter ``future_into_py`` without a "no running event loop"
    // error.
    locals: Option<pyo3_async_runtimes::TaskLocals>,
    // Runtime captured at construction. ``Drop`` is synchronous and cannot await, so the
    // close future is scheduled onto this handle as fire-and-forget.
    handle: Option<tokio::runtime::Handle>,
}

impl SourceClosingStream {
    fn new(inner: BoxResponseStream, source: Py<PyAny>) -> Self {
        Self {
            inner,
            source: Some(source),
            locals: Python::attach(|py| pyo3_async_runtimes::tokio::get_current_locals(py).ok()),
            handle: tokio::runtime::Handle::try_current().ok(),
        }
    }
}

impl Stream for SourceClosingStream {
    type Item = switchyard_components::Result<StreamEvent>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // All fields are ``Unpin``, so projecting through ``get_mut`` is sound.
        self.get_mut().inner.as_mut().poll_next(cx)
    }
}

impl Drop for SourceClosingStream {
    fn drop(&mut self) {
        let Some(source) = self.source.take() else {
            return;
        };
        let (Some(locals), Some(handle)) = (self.locals.take(), self.handle.take()) else {
            // No event loop / runtime to drive the async close (e.g. interpreter
            // teardown); dropping the source ref is the best we can do.
            tracing::warn!("ChatResponseStream: no event loop to close stream source on drop");
            return;
        };
        // Fire-and-forget on the captured runtime, scoped to the captured event
        // loop so the source's async ``aclose`` runs on asyncio. Releasing the
        // connection here is what stops the pool leak on client disconnect /
        // mid-stream error.
        handle.spawn(pyo3_async_runtimes::tokio::scope(locals, async move {
            close_python_source(source).await;
        }));
    }
}

/// Best-effort close of a Python stream source on teardown.
///
/// Async generators expose ``aclose``; SDK ``AsyncStream`` objects expose
/// ``close``. Either may return a coroutine that must be awaited. Closing must
/// never mask the teardown that triggered it, so failures are logged and
/// swallowed rather than propagated.
async fn close_python_source(source: Py<PyAny>) {
    let awaitable = match Python::attach(|py| detect_close_awaitable(source.bind(py))) {
        Ok(awaitable) => awaitable,
        Err(error) => {
            tracing::warn!(error = %error, "ChatResponseStream: failed to close stream source");
            return;
        }
    };
    let Some(awaitable) = awaitable else {
        return;
    };
    let future =
        Python::attach(|py| pyo3_async_runtimes::tokio::into_future(awaitable.bind(py).clone()));
    match future {
        Ok(future) => {
            if let Err(error) = future.await {
                tracing::warn!(error = %error, "ChatResponseStream: error closing stream source");
            }
        }
        Err(error) => {
            tracing::warn!(
                error = %error,
                "ChatResponseStream: failed to schedule stream-source close"
            );
        }
    }
}

/// Call ``aclose`` (preferred) or ``close`` on a stream source, returning the
/// coroutine to await when the method is asynchronous. Returns ``None`` when
/// the source exposes no closer or the closer is synchronous (already done).
fn detect_close_awaitable(source: &Bound<'_, PyAny>) -> PyResult<Option<Py<PyAny>>> {
    let method = if source.hasattr("aclose")? {
        "aclose"
    } else if source.hasattr("close")? {
        "close"
    } else {
        return Ok(None);
    };
    let result = source.call_method0(method)?;
    if result.hasattr("__await__")? {
        Ok(Some(result.unbind()))
    } else {
        Ok(None)
    }
}

fn stream_event_from_python(py: Python<'_>, event: Py<PyAny>) -> PyResult<StreamEvent> {
    let event = event.bind(py);
    if let Ok(value) = event.extract::<String>() {
        return Ok(StreamEvent::Text(value));
    }
    Ok(StreamEvent::Json(value_from_python(event)?))
}

async fn run_taps(callbacks: Arc<Mutex<Vec<Py<PyAny>>>>, event: &Py<PyAny>) {
    let snapshot = match clone_callbacks(&callbacks) {
        Ok(snapshot) => snapshot,
        Err(_) => return,
    };
    let mut failed = Vec::new();
    for (index, callback) in snapshot {
        if let Err(error) = call_python_callback(callback, event).await {
            tracing::warn!(
                error = %error,
                callback_index = index,
                "ChatResponseStream tap failed, quarantining"
            );
            failed.push(index);
        }
    }
    if failed.is_empty() {
        return;
    }
    let mut callbacks = callbacks.lock();
    for index in failed.into_iter().rev() {
        if index < callbacks.len() {
            callbacks.remove(index);
        }
    }
}

async fn run_maps(
    callbacks: Arc<Mutex<Vec<Py<PyAny>>>>,
    mut event: Py<PyAny>,
) -> PyResult<Py<PyAny>> {
    let callbacks = clone_callbacks(&callbacks)?;
    for (_, callback) in callbacks {
        event = call_python_callback(callback, &event).await?;
    }
    Ok(event)
}

async fn run_completion_once(callbacks: Arc<Mutex<Vec<Py<PyAny>>>>, completed: Arc<AtomicBool>) {
    if completed.swap(true, Ordering::AcqRel) {
        return;
    }
    let snapshot = match clone_callbacks(&callbacks) {
        Ok(snapshot) => snapshot,
        Err(_) => return,
    };
    for (index, callback) in snapshot {
        if let Err(error) = call_python_callback_no_args(callback).await {
            tracing::warn!(
                error = %error,
                callback_index = index,
                "ChatResponseStream completion callback failed"
            );
        }
    }
}

fn clone_callbacks(callbacks: &Arc<Mutex<Vec<Py<PyAny>>>>) -> PyResult<Vec<(usize, Py<PyAny>)>> {
    Python::attach(|py| {
        let callbacks = callbacks.lock();
        Ok(callbacks
            .iter()
            .enumerate()
            .map(|(index, callback)| (index, callback.clone_ref(py)))
            .collect::<Vec<_>>())
    })
}

async fn call_python_callback(callback: Py<PyAny>, arg: &Py<PyAny>) -> PyResult<Py<PyAny>> {
    let future = Python::attach(|py| {
        let result = callback.bind(py).call1((arg.clone_ref(py),))?;
        pyo3_async_runtimes::tokio::into_future(result)
    })?;
    future.await
}

async fn call_python_callback_no_args(callback: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let future = Python::attach(|py| {
        let result = callback.bind(py).call0()?;
        pyo3_async_runtimes::tokio::into_future(result)
    })?;
    future.await
}

fn is_stop_async_iteration(error: &PyErr) -> bool {
    Python::attach(|py| error.is_instance_of::<PyStopAsyncIteration>(py))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyResponseStream>()?;
    Ok(())
}
