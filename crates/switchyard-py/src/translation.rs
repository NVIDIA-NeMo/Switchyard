// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use futures::{StreamExt, stream::BoxStream};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyStopAsyncIteration};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use serde::Serialize;
use serde_json::{Value, json};
use switchyard_protocol::{LlmClientError, LlmResponseStreamEvent, LlmStreamError};
use switchyard_translation::{
    AggLlmResponse, LlmRequest, ProviderExtensions, TranslationEngine,
    TranslationError as CoreTranslationError, TranslationOutput, TranslationPolicy,
};

use tokio::sync::Mutex;

use crate::py_serde::{from_python, to_python};

create_exception!(
    switchyard.translation,
    TranslationError,
    PyException,
    "Translation failed. Attributes: kind (stable variant name), details (variant fields)."
);

/// Translate provider requests, responses, and streams.
#[pyclass(name = "TranslationEngine", module = "switchyard.translation", frozen)]
struct PyTranslationEngine {
    inner: TranslationEngine,
    default_policy: TranslationPolicy,
}

#[pymethods]
impl PyTranslationEngine {
    #[new]
    fn new() -> Self {
        Self {
            inner: TranslationEngine::default(),
            default_policy: TranslationPolicy::default(),
        }
    }

    /// Decode a provider request, returning request IR and diagnostics.
    fn decode_request(
        &self,
        py: Python<'_>,
        source: &str,
        body: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyDict>> {
        let body: Value = from_python(body.as_any())?;
        let output = py.detach(|| {
            self.inner
                .decode_request(source, &body, &self.default_policy)
        });
        let output = into_python_result(py, output)?;
        let result = PyDict::new(py);
        result.set_item("request", to_python(py, &output.request)?)?;
        result.set_item("diagnostics", to_python(py, &output.diagnostics)?)?;
        Ok(result.unbind())
    }

    /// Encode request IR, returning a provider body and diagnostics.
    fn encode_request(
        &self,
        py: Python<'_>,
        target: &str,
        request: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyDict>> {
        let request: LlmRequest = from_python(request.as_any())?;
        let output = py.detach(|| {
            self.inner
                .encode_request(target, &request, &self.default_policy)
        });
        translation_output(py, output)
    }

    /// Decode a complete provider response, returning response IR and diagnostics.
    fn decode_response(
        &self,
        py: Python<'_>,
        source: &str,
        body: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyDict>> {
        let body: Value = from_python(body.as_any())?;
        let output = py.detach(|| {
            self.inner
                .decode_response(source, &body, &self.default_policy)
        });
        let output = into_python_result(py, output)?;
        let result = PyDict::new(py);
        result.set_item("response", to_python(py, &output.response)?)?;
        result.set_item("diagnostics", to_python(py, &output.diagnostics)?)?;
        Ok(result.unbind())
    }

    /// Encode response IR, optionally restoring tool details from request extensions.
    #[pyo3(signature = (target, response, *, request_extensions=None))]
    fn encode_response(
        &self,
        py: Python<'_>,
        target: &str,
        response: &Bound<'_, PyDict>,
        request_extensions: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDict>> {
        let response: AggLlmResponse = from_python(response.as_any())?;
        let request_extensions: ProviderExtensions = request_extensions
            .map(|extensions| from_python(extensions.as_any()))
            .transpose()?
            .unwrap_or_default();
        let output = py.detach(|| {
            self.inner.encode_response_with_extensions(
                target,
                &response,
                &request_extensions,
                &self.default_policy,
            )
        });
        translation_output(py, output)
    }

    /// Translate a request dictionary, returning body and diagnostics.
    fn translate_request(
        &self,
        py: Python<'_>,
        source: &str,
        target: &str,
        body: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyDict>> {
        let body: Value = from_python(body.as_any())?;
        let output = py.detach(|| {
            self.inner
                .translate_request(source, target, &body, &self.default_policy)
        });
        translation_output(py, output)
    }

    /// Translate a complete response dictionary, returning body and diagnostics.
    fn translate_response(
        &self,
        py: Python<'_>,
        source: &str,
        target: &str,
        body: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyDict>> {
        let body: Value = from_python(body.as_any())?;
        let output = py.detach(|| {
            self.inner
                .translate_response(source, target, &body, &self.default_policy)
        });
        translation_output(py, output)
    }

    /// Decode SSE byte chunks into an async iterator of IR event dictionaries.
    fn decode_stream(
        &self,
        py: Python<'_>,
        source: &str,
        byte_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let adapter = adapter(py, byte_stream)?;
        let bytes = input_stream(&adapter, |item| {
            Ok(item.cast::<PyBytes>()?.as_bytes().to_vec())
        })?;
        let stream = self
            .inner
            .decode_stream(source, bytes)
            .map_err(|error| client_error(py, error))?;
        output_stream(
            py,
            adapter,
            stream
                .map(|item| item.map_err(LlmStreamError::Client))
                .boxed(),
        )
    }

    /// Encode IR events into an async iterator of provider event dictionaries.
    #[pyo3(signature = (target, chunks, *, served_model=None, request_extensions=None))]
    fn encode_stream(
        &self,
        py: Python<'_>,
        target: &str,
        chunks: &Bound<'_, PyAny>,
        served_model: Option<String>,
        request_extensions: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let extensions = request_extensions
            .map(|extensions| from_python(extensions.as_any()))
            .transpose()?
            .unwrap_or_default();
        let adapter = adapter(py, chunks)?;
        let chunks = input_stream(&adapter, |item| {
            from_python::<LlmResponseStreamEvent>(item.cast::<PyDict>()?.as_any())
        })?;
        let stream = self
            .inner
            .encode_stream_with_extensions(target, chunks, served_model, &extensions)
            .map_err(|error| client_error(py, error))?;
        output_stream(py, adapter, stream)
    }

    /// Convenience method for calling decode_stream and encode_stream.
    #[pyo3(signature = (source, target, byte_stream, *, served_model=None, request_extensions=None))]
    fn translate_stream(
        slf: PyRef<'_, Self>,
        source: &str,
        target: &str,
        byte_stream: &Bound<'_, PyAny>,
        served_model: Option<String>,
        request_extensions: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let kwargs = PyDict::new(py);
        kwargs.set_item("served_model", served_model)?;
        kwargs.set_item("request_extensions", request_extensions)?;
        py.import("switchyard_rust._translation_stream")?
            .getattr("_TranslatedStream")?
            .call((slf, source, target, byte_stream), Some(&kwargs))
            .map(Bound::unbind)
    }
}

// Each Python awaitable owns a handle while it polls; the Python adapter permits one poll at a time.
#[pyclass(frozen)]
struct NativeStream {
    inner: Arc<Mutex<BoxStream<'static, PyResult<Py<PyAny>>>>>,
}

#[pymethods]
impl NativeStream {
    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner
                .lock()
                .await
                .next()
                .await
                .unwrap_or_else(|| Err(PyStopAsyncIteration::new_err(())))
        })
    }
}

fn adapter<'py>(py: Python<'py>, input: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    py.import("switchyard_rust._translation_stream")?
        .getattr("_TranslationStream")?
        .call1((input,))
}

fn input_stream<T: Send + 'static>(
    adapter: &Bound<'_, PyAny>,
    convert: fn(&Bound<'_, PyAny>) -> PyResult<T>,
) -> PyResult<BoxStream<'static, Result<T, LlmClientError>>> {
    let adapter = adapter.getattr("_input")?.unbind();
    Ok(
        futures::stream::unfold(Some(adapter), move |state| async move {
            let adapter = state?;
            let next = Python::attach(|py| {
                let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
                pyo3_async_runtimes::into_future_with_locals(
                    &locals,
                    adapter.bind(py).call_method0("_read")?,
                )
            });
            let result = match next {
                Ok(next) => next.await,
                Err(error) => Err(error),
            };
            match result {
                Ok(item) => {
                    let item = Python::attach(|py| convert(item.bind(py))).map_err(|error| {
                        LlmClientError::Ffi {
                            source: Box::new(error),
                        }
                    });
                    Some((item, Some(adapter)))
                }
                Err(error)
                    if Python::attach(|py| error.is_instance_of::<PyStopAsyncIteration>(py)) =>
                {
                    None
                }
                Err(error) => Some((
                    Err(LlmClientError::Ffi {
                        source: Box::new(error),
                    }),
                    None,
                )),
            }
        })
        .boxed(),
    )
}

fn output_stream<T: Serialize + Send + 'static>(
    py: Python<'_>,
    adapter: Bound<'_, PyAny>,
    stream: BoxStream<'static, Result<T, LlmStreamError>>,
) -> PyResult<Py<PyAny>> {
    let stream = stream.map(|item| {
        Python::attach(|py| match item {
            Ok(event) => to_python(py, &event),
            Err(LlmStreamError::Client(error)) => Err(client_error(py, error)),
            Err(LlmStreamError::Upstream(error)) => Err(exception(
                py,
                "UpstreamFailure",
                format!("upstream stream error: {error}"),
                json!({"error": error}),
            )),
        })
    });
    adapter.setattr(
        "_native",
        Py::new(
            py,
            NativeStream {
                inner: Arc::new(Mutex::new(stream.boxed())),
            },
        )?,
    )?;
    Ok(adapter.unbind())
}

fn exception(py: Python<'_>, kind: &str, message: String, details: Value) -> PyErr {
    let error = TranslationError::new_err(message);
    let attach = || -> PyResult<()> {
        let value = error.value(py);
        value.setattr("kind", kind)?;
        value.setattr("details", to_python(py, &details)?)
    };
    match attach() {
        Ok(()) => error,
        Err(error) => error,
    }
}

fn client_error(py: Python<'_>, error: LlmClientError) -> PyErr {
    if let LlmClientError::Ffi { source } = &error
        && let Some(error) = source.downcast_ref::<PyErr>()
    {
        return error.clone_ref(py);
    }
    let (kind, details) = match &error {
        LlmClientError::ResponseTranslation(message) => {
            ("ResponseTranslation", json!({"message": message}))
        }
        LlmClientError::Configuration { message } => ("Configuration", json!({"message": message})),
        LlmClientError::InvalidRequest { message } => {
            ("InvalidRequest", json!({"message": message}))
        }
        LlmClientError::RequestTranslation(message) => {
            ("RequestTranslation", json!({"message": message}))
        }
        LlmClientError::RequestEncoding(message) => {
            ("RequestEncoding", json!({"message": message}))
        }
        LlmClientError::ResponseStateLimitExceeded { limit } => {
            ("ResponseStateLimitExceeded", json!({"limit": limit}))
        }
        LlmClientError::ResponseStateConflict => ("ResponseStateConflict", json!({})),
        LlmClientError::TemporarilyUnavailable => ("TemporarilyUnavailable", json!({})),
        LlmClientError::Transport { source } => {
            ("Transport", json!({"message": source.to_string()}))
        }
        LlmClientError::Timeout { source } => ("Timeout", json!({"message": source.to_string()})),
        LlmClientError::ContextWindowExceeded { model, message } => (
            "ContextWindowExceeded",
            json!({"model": model, "message": message}),
        ),
        LlmClientError::UpstreamHttp { status, body } => (
            "UpstreamHttp",
            json!({"status": status.as_u16(), "body": body}),
        ),
        LlmClientError::InvalidResponse { source } => {
            ("InvalidResponse", json!({"message": source.to_string()}))
        }
        LlmClientError::Host { source } => ("Host", json!({"message": source.to_string()})),
        LlmClientError::Ffi { source } => ("Ffi", json!({"message": source.to_string()})),
        _ => ("Other", json!({"message": error.to_string()})),
    };
    exception(py, kind, error.to_string(), details)
}

fn translation_output(
    py: Python<'_>,
    result: Result<TranslationOutput, CoreTranslationError>,
) -> PyResult<Py<PyDict>> {
    let output = into_python_result(py, result)?;
    let result = PyDict::new(py);
    result.set_item("body", to_python(py, &output.body)?)?;
    result.set_item("diagnostics", to_python(py, &output.diagnostics)?)?;
    Ok(result.unbind())
}

fn into_python_result<T>(py: Python<'_>, result: Result<T, CoreTranslationError>) -> PyResult<T> {
    match result {
        Ok(output) => Ok(output),
        Err(error) => {
            let exception = TranslationError::new_err(error.to_string());
            exception.value(py).setattr("kind", error.kind())?;
            exception
                .value(py)
                .setattr("details", to_python(py, &error_details(&error))?)?;
            Err(exception)
        }
    }
}

fn error_details(error: &CoreTranslationError) -> Value {
    match error {
        CoreTranslationError::InvalidJson(error) => json!({
            "message": error.to_string(),
            "line": error.line(),
            "column": error.column(),
        }),
        CoreTranslationError::InvalidType { path, expected } => {
            json!({"path": path, "expected": expected})
        }
        CoreTranslationError::UnsupportedTranslation { from, to } => {
            json!({"from": from, "to": to})
        }
        CoreTranslationError::LossyConversion(message) | CoreTranslationError::Other(message) => {
            json!({"message": message})
        }
        CoreTranslationError::UnknownField { path } => json!({"path": path}),
        CoreTranslationError::InvalidValue { path, message } => {
            json!({"path": path, "message": message})
        }
        CoreTranslationError::UpstreamFailure { error } => json!({"error": error}),
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let translation = PyModule::new(module.py(), "translation")?;
    translation.add_class::<PyTranslationEngine>()?;
    translation.add(
        "TranslationError",
        module.py().get_type::<TranslationError>(),
    )?;
    module.add_submodule(&translation)
}
