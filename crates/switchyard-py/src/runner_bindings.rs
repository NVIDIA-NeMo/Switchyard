// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configured Rust routes served by host-owned Python model callbacks.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use async_trait::async_trait;
use futures::{StreamExt, TryStreamExt, future::ready, stream};
use http::{HeaderMap, StatusCode};
use pyo3::exceptions::{PyStopAsyncIteration, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3_async_runtimes::TaskLocals;
use serde_json::Value;
use switchyard_llm_client::RequestSettings;
use switchyard_protocol::{
    LlmClientError, LlmResponse, LlmResponseChunk, LlmResponseStream, LlmStreamError, Metadata,
    ModelId, Request, Response, RoutedLlmClient, WireFormat,
};
use switchyard_runner::{Runner, RunnerError};
use switchyard_translation::util::SWITCHYARD_METADATA_KEY;
use switchyard_translation::{
    RawEventStream, TranslationError, decode_aggregated_response, decode_request, decode_stream,
    encode_aggregated_response_with_extensions, encode_stream_with_extensions,
};
use tokio::sync::Mutex;

use crate::errors::{ContextWindowExceededError, ServerConfigError, py_libsy_error};
use crate::libsy_bindings::{ffi_error, header_map_from_python};
use crate::py_serde::{from_python, to_python};

struct PythonClient {
    callback: Py<PyAny>,
    settings: BTreeMap<ModelId, RequestSettings>,
}

// Only explicit host errors participate in provider fallback; SDK exceptions stay terminal.
fn callback_error(py: Python<'_>, error: PyErr, model: &ModelId) -> LlmClientError {
    if error.is_instance_of::<ContextWindowExceededError>(py) {
        return LlmClientError::ContextWindowExceeded {
            model: model.clone(),
            message: error.value(py).to_string(),
        };
    }
    let upstream = (|| -> PyResult<Option<LlmClientError>> {
        let kind = py
            .import("switchyard_rust.runner")?
            .getattr("UpstreamError")?;
        if !error.value(py).is_instance(&kind)? {
            return Ok(None);
        }
        let status = error.value(py).getattr("status_code")?.extract::<u16>()?;
        let status = StatusCode::from_u16(status)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let body: Value = from_python(&error.value(py).getattr("body")?)?;
        let body = match body {
            Value::String(body) => body,
            body => body.to_string(),
        };
        Ok(Some(LlmClientError::UpstreamHttp { status, body }))
    })();
    match upstream {
        Ok(Some(error)) => error,
        _ => ffi_error(error),
    }
}

// Poll raw SSE bytes in the original request context; Python owns provider cleanup.
fn python_bytes(
    iterator: Py<PyAny>,
    locals: TaskLocals,
    model: ModelId,
) -> impl futures::Stream<Item = Result<Vec<u8>, LlmClientError>> + Send {
    stream::unfold(Some((iterator, locals, model)), |state| async move {
        let (iterator, locals, model) = state?;
        let work = Python::attach(|py| {
            let work = iterator.bind(py).call_method0("__anext__")?;
            pyo3_async_runtimes::into_future_with_locals(&locals, work)
        });
        let result = match work {
            Ok(work) => work.await,
            Err(error) => Err(error),
        };
        match result {
            Ok(bytes) => {
                let bytes = Python::attach(|py| {
                    bytes
                        .bind(py)
                        .cast::<PyBytes>()
                        .map(|bytes| bytes.as_bytes().to_vec())
                        .map_err(|error| ffi_error(error.into()))
                });
                let next = bytes.is_ok().then_some((iterator, locals, model));
                Some((bytes, next))
            }
            Err(error) => {
                if Python::attach(|py| error.is_instance_of::<PyStopAsyncIteration>(py)) {
                    None
                } else {
                    Some((
                        Err(Python::attach(|py| callback_error(py, error, &model))),
                        None,
                    ))
                }
            }
        }
    })
}

// Release consumed answer streams before a judge uses the same host connection pool.
fn python_response_stream(
    iterator: Py<PyAny>,
    locals: TaskLocals,
    model: ModelId,
    format: WireFormat,
) -> Result<LlmResponseStream, LlmClientError> {
    let bytes = python_bytes(
        Python::attach(|py| iterator.clone_ref(py)),
        locals.clone(),
        model,
    );
    let chunks = decode_stream(bytes, format)?;
    Ok(
        stream::unfold(Some((chunks, iterator, locals)), |state| async move {
            let (mut chunks, iterator, locals) = state?;
            let next = chunks.next().await;
            let finished = match &next {
                None | Some(Err(_)) => true,
                Some(Ok(event)) => event.normalized().iter().any(|chunk| {
                    matches!(
                        chunk,
                        LlmResponseChunk::StreamError { .. } | LlmResponseChunk::DecodeError { .. }
                    )
                }),
            };
            if finished {
                let work = Python::attach(|py| {
                    let work = iterator.bind(py).call_method0("aclose")?;
                    pyo3_async_runtimes::into_future_with_locals(&locals, work)
                });
                let closed = match work {
                    Ok(work) => work.await,
                    Err(error) => Err(error),
                };
                if let Err(error) = closed {
                    return Some((Err(ffi_error(error)), None));
                }
            }
            next.map(|event| (event, (!finished).then_some((chunks, iterator, locals))))
        })
        .boxed(),
    )
}

#[async_trait]
impl RoutedLlmClient for PythonClient {
    async fn call(&self, request: Request) -> Result<Response, LlmClientError> {
        let model = request
            .model_id()
            .ok_or_else(|| LlmClientError::InvalidRequest {
                message: "model is missing".to_string(),
            })?;
        let settings = self
            .settings
            .get(&model)
            .ok_or_else(|| LlmClientError::Configuration {
                message: format!("no request settings configured for {model}"),
            })?;
        let body = settings.prepare_request_body(&request.llm_request, &model)?;
        // Callbacks inherit this request's asyncio loop and context variables.
        let (locals, work) = Python::attach(|py| {
            let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
            let work = self.callback.bind(py).call1((to_python(py, &body)?,))?;
            let work = pyo3_async_runtimes::into_future_with_locals(&locals, work)?;
            Ok::<_, PyErr>((locals, work))
        })
        .map_err(ffi_error)?;
        let body = work
            .await
            .map_err(|error| Python::attach(|py| callback_error(py, error, &model)))?;
        let response = if Python::attach(|py| body.bind(py).hasattr("__aiter__"))
            .map_err(ffi_error)?
        {
            let mut chunks = python_response_stream(body, locals, model, settings.wire_format)?;
            // Lazy HTTP failures can fall back before the caller receives a stream.
            let first = chunks.try_next().await?;
            LlmResponse::Stream(match first {
                Some(first) => stream::once(ready(Ok(first))).chain(chunks).boxed(),
                None => stream::empty().boxed(),
            })
        } else {
            let body: Value = Python::attach(|py| from_python(body.bind(py))).map_err(ffi_error)?;
            LlmResponse::Agg(
                decode_aggregated_response(&body, settings.wire_format).map_err(
                    |error| match error {
                        TranslationError::UpstreamFailure { error } => {
                            LlmClientError::UpstreamHttp {
                                status: StatusCode::BAD_GATEWAY,
                                body: serde_json::json!({"error": error}).to_string(),
                            }
                        }
                        error => LlmClientError::ResponseTranslation(error.to_string()),
                    },
                )?,
            )
        };
        Ok(Response {
            llm_response: response,
            metadata: request.metadata,
            upstream_headers: HeaderMap::new(),
        })
    }
}

/// Caller-format events; closing drops the native decoding and routing stream.
#[pyclass(name = "_ResponseStream", module = "switchyard.runner", frozen)]
struct PythonResponseStream {
    inner: Arc<Mutex<Option<RawEventStream>>>,
}

#[pymethods]
impl PythonResponseStream {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let stream = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut stream = stream.lock().await;
            let event = match stream.as_mut() {
                Some(stream) => stream.next().await,
                None => None,
            };
            match event {
                Some(Ok(event)) => Python::attach(|py| to_python(py, &event)),
                // The encoder supplies an already translated terminal provider error.
                Some(Err(LlmStreamError::Upstream(event))) => {
                    stream.take();
                    Python::attach(|py| to_python(py, &event))
                }
                Some(Err(error)) => Err(py_libsy_error(error)),
                None => Err(PyStopAsyncIteration::new_err(())),
            }
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let stream = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            stream.lock().await.take();
            Ok(())
        })
    }
}

/// A persistent route table retaining Rust session and provider continuation state.
#[pyclass(name = "Runner", module = "switchyard.runner", frozen)]
struct PythonRunner {
    inner: Arc<Runner>,
}

#[pymethods]
impl PythonRunner {
    /// Load TOML with one Python callback per configured model client.
    #[staticmethod]
    #[pyo3(signature = (source, *, clients))]
    fn from_toml(
        py: Python<'_>,
        source: &str,
        clients: HashMap<String, Py<PyAny>>,
    ) -> PyResult<Self> {
        for (name, callback) in &clients {
            if !callback.bind(py).is_callable() {
                return Err(PyTypeError::new_err(format!(
                    "client {name} must be callable"
                )));
            }
        }
        let runner = py
            .detach(|| {
                Runner::from_toml_with_clients(source, |name, settings| {
                    let callback = clients
                        .get(name)
                        .ok_or_else(|| RunnerError::Configuration {
                            message: format!("unregistered host client {name}"),
                            source: None,
                        })?;
                    Ok(Arc::new(PythonClient {
                        callback: Python::attach(|py| callback.clone_ref(py)),
                        settings,
                    }))
                })
            })
            .map_err(|error| ServerConfigError::new_err(error.to_string()))?;
        Ok(Self {
            inner: Arc::new(runner),
        })
    }

    /// Execute one provider-format request through its configured route.
    #[pyo3(signature = (request, format, headers=None))]
    fn _execute<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'py, PyAny>,
        format: &str,
        headers: Option<HashMap<String, String>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let format: WireFormat = serde_json::from_value(Value::String(format.to_string()))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let mut body: Value = from_python(request)?;
        // Exact preservation state belongs to trusted translation hops, not callers.
        if let Some(metadata) = body.get_mut("metadata").and_then(Value::as_object_mut) {
            metadata.remove(SWITCHYARD_METADATA_KEY);
        }
        let llm_request = py
            .detach(|| decode_request(format, &body))
            .map_err(py_libsy_error)?;
        let model = llm_request
            .model
            .clone()
            .ok_or_else(|| PyValueError::new_err("model is missing"))?;
        let metadata = headers
            .as_ref()
            .map(header_map_from_python)
            .transpose()?
            .map(|headers| {
                let mut metadata = Metadata::from_headers(&headers);
                metadata.http_headers = Some(headers);
                metadata
            });
        let extensions = llm_request.extensions.clone();
        let request = Request {
            llm_request,
            raw_request: None,
            metadata,
        };
        let runner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let route = runner
                .route(&model)
                .ok_or_else(|| py_libsy_error(RunnerError::UnknownRouteModel(model)))?;
            let output = route.execute(request, None).await.map_err(py_libsy_error)?;
            let served = output.response.served_model().map(ModelId::to_string);
            match output.response.llm_response {
                LlmResponse::Agg(response) => {
                    let body = encode_aggregated_response_with_extensions(
                        &response,
                        format,
                        served.as_deref(),
                        &extensions,
                    )
                    .map_err(py_libsy_error)?;
                    Python::attach(|py| to_python(py, &body))
                }
                LlmResponse::Stream(chunks) => {
                    let events = encode_stream_with_extensions(chunks, format, served, &extensions)
                        .map_err(py_libsy_error)?;
                    Python::attach(|py| {
                        Py::new(
                            py,
                            PythonResponseStream {
                                inner: Arc::new(Mutex::new(Some(events))),
                            },
                        )
                        .map(Py::into_any)
                    })
                }
            }
        })
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let runner = PyModule::new(module.py(), "runner")?;
    runner.add_class::<PythonRunner>()?;
    runner.add_class::<PythonResponseStream>()?;
    module.add_submodule(&runner)
}
