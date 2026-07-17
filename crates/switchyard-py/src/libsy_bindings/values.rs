// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Neutral libsy values and response-stream adapters.

use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context as TaskContext, Poll};

use futures_util::{stream, Stream, StreamExt};
use pyo3::exceptions::{PyAttributeError, PyRuntimeError, PyStopAsyncIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
use serde::Serialize;
use serde_json::{Map, Value};
use switchyard_protocol::{
    Context, LlmResponse, LlmResponseChunk, LlmResponseStream, Metadata, Request, Response,
};

use crate::errors::py_libsy_error;
use crate::py_serde::{value_from_python, value_to_python};

use super::protocol::{PyAggLlmResponse, PyLlmRequest, PyLlmResponseChunk, PyWireFormat};

type BoxError = Box<dyn Error + Send + Sync>;

#[pyclass(
    name = "Context",
    module = "switchyard.libsy.protocol",
    frozen,
    skip_from_py_object
)]
#[derive(Clone, Default)]
pub(crate) struct PyContext {
    inner: Context,
}

impl PyContext {
    pub(crate) fn from_core(inner: Context) -> Self {
        Self { inner }
    }

    pub(crate) fn clone_core(&self) -> Context {
        self.inner.clone()
    }
}

#[pymethods]
impl PyContext {
    #[new]
    #[pyo3(signature = (*, values=None))]
    fn new(values: Option<HashMap<String, String>>) -> Self {
        Self::from_core(Context {
            values: values.unwrap_or_default(),
        })
    }

    #[getter]
    fn values(&self) -> HashMap<String, String> {
        self.inner.values.clone()
    }

    fn __repr__(&self) -> String {
        format!("Context(values={:?})", self.inner.values)
    }
}

#[pyclass(
    name = "Metadata",
    module = "switchyard.libsy.protocol",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyMetadata {
    inner: Metadata,
}

impl PyMetadata {
    fn from_core(inner: Metadata) -> Self {
        Self { inner }
    }

    fn clone_core(&self) -> Metadata {
        self.inner.clone()
    }
}

#[pymethods]
impl PyMetadata {
    #[new]
    #[pyo3(signature = (*, session_id=None, agent_id=None, task_id=None, correlation_id=None, extra_metadata=None, http_headers=None, wire_format=None))]
    fn new(
        session_id: Option<String>,
        agent_id: Option<String>,
        task_id: Option<String>,
        correlation_id: Option<String>,
        extra_metadata: Option<BTreeMap<String, String>>,
        http_headers: Option<BTreeMap<String, String>>,
        wire_format: Option<PyWireFormat>,
    ) -> Self {
        Self {
            inner: Metadata {
                session_id,
                agent_id,
                task_id,
                correlation_id,
                extra_metadata,
                http_headers,
                wire_format: wire_format.map(Into::into),
            },
        }
    }

    #[getter]
    fn session_id(&self) -> Option<String> {
        self.inner.session_id.clone()
    }

    #[getter]
    fn agent_id(&self) -> Option<String> {
        self.inner.agent_id.clone()
    }

    #[getter]
    fn task_id(&self) -> Option<String> {
        self.inner.task_id.clone()
    }

    #[getter]
    fn correlation_id(&self) -> Option<String> {
        self.inner.correlation_id.clone()
    }

    #[getter]
    fn extra_metadata(&self) -> Option<BTreeMap<String, String>> {
        self.inner.extra_metadata.clone()
    }

    #[getter]
    fn http_headers(&self) -> Option<BTreeMap<String, String>> {
        self.inner.http_headers.clone()
    }

    #[getter]
    fn wire_format(&self) -> Option<PyWireFormat> {
        self.inner.wire_format.map(Into::into)
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        value_to_python(py, &metadata_to_value(&self.inner))
    }

    fn __repr__(&self) -> String {
        format!(
            "Metadata(session_id={:?}, correlation_id={:?})",
            self.inner.session_id, self.inner.correlation_id
        )
    }
}

#[pyclass(
    name = "Request",
    module = "switchyard.libsy.protocol",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyRequest {
    inner: Request,
}

impl PyRequest {
    pub(crate) fn from_core(inner: Request) -> Self {
        Self { inner }
    }

    pub(crate) fn clone_core(&self) -> Request {
        self.inner.clone()
    }
}

#[pymethods]
impl PyRequest {
    #[new]
    #[pyo3(signature = (llm_request, *, raw_request=None, metadata=None))]
    fn new(
        llm_request: PyRef<'_, PyLlmRequest>,
        raw_request: Option<&Bound<'_, PyAny>>,
        metadata: Option<PyRef<'_, PyMetadata>>,
    ) -> PyResult<Self> {
        let raw_request = raw_request.map(value_from_python).transpose()?;
        Ok(Self {
            inner: Request {
                llm_request: llm_request.clone_core(),
                raw_request,
                metadata: metadata.map(|value| value.clone_core()),
            },
        })
    }

    #[getter]
    fn llm_request(&self, py: Python<'_>) -> PyResult<Py<PyLlmRequest>> {
        Py::new(py, PyLlmRequest::from_core(self.inner.llm_request.clone()))
    }

    #[getter]
    fn raw_request(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner.raw_request {
            Some(value) => value_to_python(py, value),
            None => Ok(py.None()),
        }
    }

    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<Option<Py<PyMetadata>>> {
        self.inner
            .metadata
            .clone()
            .map(|metadata| Py::new(py, PyMetadata::from_core(metadata)))
            .transpose()
    }

    #[getter]
    fn requested_model(&self) -> Option<String> {
        self.inner.requested_model().map(str::to_owned)
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mut object = Map::new();
        object.insert(
            "llm_request".to_string(),
            serde_json::to_value(&self.inner.llm_request)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        );
        object.insert(
            "raw_request".to_string(),
            self.inner.raw_request.clone().unwrap_or(Value::Null),
        );
        object.insert(
            "metadata".to_string(),
            self.inner
                .metadata
                .as_ref()
                .map(metadata_to_value)
                .unwrap_or(Value::Null),
        );
        value_to_python(py, &Value::Object(object))
    }

    fn __repr__(&self) -> String {
        format!("Request(model={:?})", self.inner.requested_model())
    }
}

#[pyclass(name = "Response", module = "switchyard.libsy.protocol")]
pub(crate) struct PyResponse {
    llm_response: Mutex<Option<LlmResponse>>,
    metadata: Option<Metadata>,
}

impl PyResponse {
    pub(crate) fn from_core(inner: Response) -> Self {
        Self {
            llm_response: Mutex::new(Some(inner.llm_response)),
            metadata: inner.metadata,
        }
    }

    pub(crate) fn take_core(&self) -> PyResult<Response> {
        let mut response = self
            .llm_response
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Response lock is poisoned"))?;
        let Some(llm_response) = response.take() else {
            return Err(PyRuntimeError::new_err(
                "Response has already been consumed",
            ));
        };
        Ok(Response {
            llm_response,
            metadata: self.metadata.clone(),
        })
    }
}

#[pymethods]
impl PyResponse {
    #[new]
    #[pyo3(signature = (llm_response, *, metadata=None))]
    fn new(
        llm_response: PyRef<'_, PyAggLlmResponse>,
        metadata: Option<PyRef<'_, PyMetadata>>,
    ) -> PyResult<Self> {
        Ok(Self {
            llm_response: Mutex::new(Some(LlmResponse::Agg(llm_response.clone_core()))),
            metadata: metadata.map(|value| value.clone_core()),
        })
    }

    #[classmethod]
    #[pyo3(signature = (source, *, metadata=None))]
    fn from_stream(
        _cls: &Bound<'_, PyType>,
        source: &Bound<'_, PyAny>,
        metadata: Option<PyRef<'_, PyMetadata>>,
    ) -> Self {
        Self {
            llm_response: Mutex::new(Some(LlmResponse::Stream(stream_from_python_source(
                source.clone().unbind(),
            )))),
            metadata: metadata.map(|value| value.clone_core()),
        }
    }

    #[getter]
    fn is_streaming(&self) -> PyResult<bool> {
        let response = self
            .llm_response
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Response lock is poisoned"))?;
        Ok(matches!(response.as_ref(), Some(LlmResponse::Stream(_))))
    }

    #[getter]
    fn selected_model(&self) -> PyResult<Option<String>> {
        let response = self
            .llm_response
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Response lock is poisoned"))?;
        Ok(response
            .as_ref()
            .and_then(LlmResponse::selected_model)
            .map(str::to_owned))
    }

    #[getter]
    fn aggregate(&self, py: Python<'_>) -> PyResult<Py<PyAggLlmResponse>> {
        let response = self
            .llm_response
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Response lock is poisoned"))?;
        match response.as_ref() {
            Some(LlmResponse::Agg(aggregate)) => {
                Py::new(py, PyAggLlmResponse::from_core(aggregate.clone()))
            }
            Some(LlmResponse::Stream(_)) => Err(PyAttributeError::new_err(
                "streaming Response has no aggregate value",
            )),
            None => Err(PyRuntimeError::new_err(
                "Response has already been consumed",
            )),
        }
    }

    #[getter]
    fn stream(&self, py: Python<'_>) -> PyResult<Py<PyLlmResponseStream>> {
        let mut response = self
            .llm_response
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Response lock is poisoned"))?;
        let Some(current) = response.take() else {
            return Err(PyRuntimeError::new_err(
                "Response has already been consumed",
            ));
        };
        match current {
            LlmResponse::Stream(stream) => Py::new(py, PyLlmResponseStream::new(stream)),
            aggregate @ LlmResponse::Agg(_) => {
                *response = Some(aggregate);
                Err(PyAttributeError::new_err(
                    "aggregate Response has no stream",
                ))
            }
        }
    }

    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<Option<Py<PyMetadata>>> {
        self.metadata
            .clone()
            .map(|metadata| Py::new(py, PyMetadata::from_core(metadata)))
            .transpose()
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let response = self
            .llm_response
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Response lock is poisoned"))?;
        match response.as_ref() {
            Some(LlmResponse::Agg(aggregate)) => serialized_to_python(py, aggregate),
            Some(LlmResponse::Stream(_)) => Err(PyAttributeError::new_err(
                "streaming Response has no aggregate value",
            )),
            None => Err(PyRuntimeError::new_err(
                "Response has already been consumed",
            )),
        }
    }

    fn __repr__(&self) -> PyResult<String> {
        let response = self
            .llm_response
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Response lock is poisoned"))?;
        let kind = match response.as_ref() {
            Some(LlmResponse::Agg(_)) => "aggregate",
            Some(LlmResponse::Stream(_)) => "stream",
            None => "consumed",
        };
        Ok(format!("Response(kind='{kind}')"))
    }
}

#[pyclass(name = "LlmResponseStream", module = "switchyard.libsy.protocol")]
pub(crate) struct PyLlmResponseStream {
    stream: Arc<tokio::sync::Mutex<Option<LlmResponseStream>>>,
    consumed: bool,
}

impl PyLlmResponseStream {
    fn new(stream: LlmResponseStream) -> Self {
        Self {
            stream: Arc::new(tokio::sync::Mutex::new(Some(stream))),
            consumed: false,
        }
    }
}

#[pymethods]
impl PyLlmResponseStream {
    fn __aiter__(mut slf: PyRefMut<'_, Self>) -> PyResult<Py<PyAny>> {
        if slf.consumed {
            return Err(PyRuntimeError::new_err(
                "LlmResponseStream has already been consumed",
            ));
        }
        slf.consumed = true;
        let py = slf.py();
        Ok(slf.into_pyobject(py)?.unbind().into_any())
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let stream = Arc::clone(&self.stream);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = stream.lock().await;
            let Some(inner) = guard.as_mut() else {
                return Err(PyStopAsyncIteration::new_err(()));
            };
            match inner.next().await {
                Some(Ok(chunk)) => Python::attach(|py| {
                    PyLlmResponseChunk::from_core(py, chunk)?
                        .into_pyobject(py)
                        .map(|value| value.unbind().into_any())
                }),
                Some(Err(error)) => {
                    guard.take();
                    Err(py_libsy_error(error))
                }
                None => {
                    guard.take();
                    Err(PyStopAsyncIteration::new_err(()))
                }
            }
        })
    }

    fn aclose<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.consumed = true;
        let stream = Arc::clone(&self.stream);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            stream.lock().await.take();
            Ok(())
        })
    }

    fn __repr__(&self) -> &'static str {
        "LlmResponseStream()"
    }
}

struct PythonChunkState {
    source: Py<PyAny>,
    iterator: Option<Py<PyAny>>,
    done: bool,
}

struct SourceClosingLlmStream {
    stream: LlmResponseStream,
    source: Option<Py<PyAny>>,
}

impl Stream for SourceClosingLlmStream {
    type Item = Result<LlmResponseChunk, BoxError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let result = this.stream.as_mut().poll_next(cx);
        if matches!(result, Poll::Ready(None) | Poll::Ready(Some(Err(_)))) {
            if let Some(source) = this.source.take() {
                schedule_stream_source_close(source);
            }
        }
        result
    }
}

impl Drop for SourceClosingLlmStream {
    fn drop(&mut self) {
        if let Some(source) = self.source.take() {
            schedule_stream_source_close(source);
        }
    }
}

fn stream_from_python_source(source: Py<PyAny>) -> LlmResponseStream {
    let retained = Python::attach(|py| source.clone_ref(py));
    let stream = stream::unfold(
        PythonChunkState {
            source,
            iterator: None,
            done: false,
        },
        next_python_chunk,
    );
    Box::pin(SourceClosingLlmStream {
        stream: Box::pin(stream),
        source: Some(retained),
    })
}

async fn next_python_chunk(
    mut state: PythonChunkState,
) -> Option<(Result<LlmResponseChunk, BoxError>, PythonChunkState)> {
    if state.done {
        return None;
    }

    if state.iterator.is_none() {
        match Python::attach(|py| {
            state
                .source
                .bind(py)
                .call_method0("__aiter__")
                .map(Bound::unbind)
        }) {
            Ok(iterator) => state.iterator = Some(iterator),
            Err(error) => {
                state.done = true;
                return Some((Err(python_error(error)), state));
            }
        }
    }

    let Some(iterator) = state.iterator.as_ref() else {
        state.done = true;
        return Some((Err("async iterator initialization failed".into()), state));
    };
    let future = match Python::attach(|py| {
        let awaitable = iterator.bind(py).call_method0("__anext__")?;
        pyo3_async_runtimes::tokio::into_future(awaitable)
    }) {
        Ok(future) => future,
        Err(error) => {
            state.done = true;
            return Some((Err(python_error(error)), state));
        }
    };

    match future.await {
        Ok(value) => {
            let result =
                Python::attach(|py| chunk_from_python(value.bind(py))).map_err(python_error);
            Some((result, state))
        }
        Err(error) if is_stop_async_iteration(&error) => None,
        Err(error) => {
            state.done = true;
            Some((Err(python_error(error)), state))
        }
    }
}

fn chunk_from_python(value: &Bound<'_, PyAny>) -> PyResult<LlmResponseChunk> {
    let chunk = value.extract::<PyRef<'_, PyLlmResponseChunk>>()?;
    Ok(chunk.clone_core(value.py()))
}

pub(crate) fn python_error(error: PyErr) -> BoxError {
    std::io::Error::other(error.to_string()).into()
}

fn is_stop_async_iteration(error: &PyErr) -> bool {
    Python::attach(|py| error.is_instance_of::<PyStopAsyncIteration>(py))
}

fn schedule_stream_source_close(source: Py<PyAny>) {
    let result = Python::attach(|py| {
        let source = source.bind(py);
        if source.hasattr("aclose")? {
            let awaitable = source.call_method0("aclose")?;
            pyo3_async_runtimes::tokio::into_future(awaitable).map(Some)
        } else if source.hasattr("close")? {
            source.call_method0("close")?;
            Ok(None)
        } else {
            Ok(None)
        }
    });
    match result {
        Ok(Some(future)) => {
            if let Ok(runtime) = tokio::runtime::Handle::try_current() {
                runtime.spawn(async move {
                    if let Err(error) = future.await {
                        tracing::warn!(error = %error, "libsy response stream close failed");
                    }
                });
            }
        }
        Ok(None) => {}
        Err(error) => {
            tracing::warn!(error = %error, "libsy response stream could not be closed");
        }
    }
}

fn serialized_to_python<T: Serialize>(py: Python<'_>, value: &T) -> PyResult<Py<PyAny>> {
    let value =
        serde_json::to_value(value).map_err(|error| PyValueError::new_err(error.to_string()))?;
    value_to_python(py, &value)
}

fn metadata_to_value(metadata: &Metadata) -> Value {
    let mut value = Map::new();
    value.insert(
        "session_id".to_string(),
        option_string_value(&metadata.session_id),
    );
    value.insert(
        "agent_id".to_string(),
        option_string_value(&metadata.agent_id),
    );
    value.insert(
        "task_id".to_string(),
        option_string_value(&metadata.task_id),
    );
    value.insert(
        "correlation_id".to_string(),
        option_string_value(&metadata.correlation_id),
    );
    value.insert(
        "extra_metadata".to_string(),
        string_map_value(&metadata.extra_metadata),
    );
    value.insert(
        "http_headers".to_string(),
        string_map_value(&metadata.http_headers),
    );
    value.insert(
        "wire_format".to_string(),
        metadata
            .wire_format
            .map(|format| Value::String(format.as_str().to_string()))
            .unwrap_or(Value::Null),
    );
    Value::Object(value)
}

fn option_string_value(value: &Option<String>) -> Value {
    value.clone().map(Value::String).unwrap_or(Value::Null)
}

fn string_map_value(value: &Option<BTreeMap<String, String>>) -> Value {
    value
        .as_ref()
        .map(|items| {
            Value::Object(
                items
                    .iter()
                    .map(|(key, value)| (key.clone(), Value::String(value.clone())))
                    .collect(),
            )
        })
        .unwrap_or(Value::Null)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyContext>()?;
    module.add_class::<PyMetadata>()?;
    module.add_class::<PyRequest>()?;
    module.add_class::<PyResponse>()?;
    module.add_class::<PyLlmResponseStream>()?;
    Ok(())
}
