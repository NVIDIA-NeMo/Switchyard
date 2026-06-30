// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for concrete LLM backends.

use std::sync::Arc;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyIterator, PyTuple};
use switchyard_components::{
    AnthropicNativeBackend, LlmTargetBackend, MultiLlmBackend, OpenAiNativeBackend,
    OpenAiPassthroughBackend,
};
use switchyard_core::{ChatRequestType, EndpointConfig, LlmBackend, LlmTargetId};

use super::config::{endpoint_config_from_python, PyEndpointConfig, PyLlmTarget};
use crate::core_bindings::request::request_type_from_python;
use crate::core_bindings::roles::PyLlmBackend;
use crate::errors::py_core_error;

#[pyclass(name = "LlmTargetBackend", frozen, skip_from_py_object)]
#[derive(Clone, Debug)]
pub(crate) struct PyLlmTargetBackend {
    inner: LlmTargetBackend,
}

impl PyLlmTargetBackend {
    fn clone_core(&self) -> LlmTargetBackend {
        self.inner.clone()
    }
}

#[pymethods]
impl PyLlmTargetBackend {
    #[new]
    fn py_new(target: PyRef<'_, PyLlmTarget>, backend: PyRef<'_, PyLlmBackend>) -> PyResult<Self> {
        Ok(Self {
            inner: LlmTargetBackend::new(
                target.clone_core(),
                native_backend(&backend, "LlmTargetBackend")?,
            ),
        })
    }

    #[getter]
    fn target(&self) -> PyLlmTarget {
        PyLlmTarget::from_core(self.inner.target().clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "LlmTargetBackend(target_id={:?}, model={:?})",
            self.inner.target().id.as_str(),
            self.inner.target().model.as_str(),
        )
    }
}

#[pyclass(
    name = "OpenAiNativeBackend",
    extends = PyLlmBackend,
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub(crate) struct PyOpenAiNativeBackend {
    inner: Arc<OpenAiNativeBackend>,
}

#[pymethods]
impl PyOpenAiNativeBackend {
    #[new]
    fn py_new(target: PyRef<'_, PyLlmTarget>) -> PyResult<PyClassInitializer<Self>> {
        let backend =
            Arc::new(OpenAiNativeBackend::new(target.clone_core()).map_err(py_core_error)?);
        let base: Arc<dyn LlmBackend> = backend.clone();
        Ok(PyClassInitializer::from(PyLlmBackend::from_native(base))
            .add_subclass(Self { inner: backend }))
    }

    #[getter]
    fn target(&self) -> PyLlmTarget {
        PyLlmTarget::from_core(self.inner.target().clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "OpenAiNativeBackend(target_id={:?}, model={:?})",
            self.inner.target().id.as_str(),
            self.inner.target().model.as_str(),
        )
    }
}

#[pyclass(
    name = "OpenAiPassthroughBackend",
    extends = PyLlmBackend,
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub(crate) struct PyOpenAiPassthroughBackend {
    inner: Arc<OpenAiPassthroughBackend>,
}

#[pymethods]
impl PyOpenAiPassthroughBackend {
    #[new]
    #[pyo3(signature = (endpoint=None, api_key=None, base_url=None, timeout_secs=None, timeout=None))]
    fn py_new(
        endpoint: Option<&Bound<'_, PyAny>>,
        api_key: Option<String>,
        base_url: Option<String>,
        timeout_secs: Option<f64>,
        timeout: Option<f64>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let mut endpoint_config = endpoint_config_from_python(endpoint)?;
        if api_key.is_some() {
            endpoint_config.api_key = api_key;
        }
        if base_url.is_some() {
            endpoint_config.base_url = base_url;
        }
        if timeout_secs.is_some() {
            endpoint_config.timeout_secs = timeout_secs;
        }
        if timeout.is_some() {
            endpoint_config.timeout_secs = timeout;
        }

        let backend =
            Arc::new(OpenAiPassthroughBackend::new(endpoint_config).map_err(py_core_error)?);
        let base: Arc<dyn LlmBackend> = backend.clone();
        Ok(PyClassInitializer::from(PyLlmBackend::from_native(base))
            .add_subclass(Self { inner: backend }))
    }

    #[getter]
    fn endpoint(&self) -> PyEndpointConfig {
        PyEndpointConfig::from_core(self.inner.endpoint().clone())
    }

    fn __repr__(&self) -> String {
        let endpoint: &EndpointConfig = self.inner.endpoint();
        format!(
            "OpenAiPassthroughBackend(base_url={:?}, timeout_secs={:?})",
            endpoint.base_url, endpoint.timeout_secs,
        )
    }
}

#[pyclass(
    name = "AnthropicNativeBackend",
    extends = PyLlmBackend,
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub(crate) struct PyAnthropicNativeBackend {
    inner: Arc<AnthropicNativeBackend>,
}

#[pymethods]
impl PyAnthropicNativeBackend {
    #[new]
    fn py_new(target: PyRef<'_, PyLlmTarget>) -> PyResult<PyClassInitializer<Self>> {
        let backend = AnthropicNativeBackend::new(target.clone_core()).map_err(py_core_error)?;
        let backend = Arc::new(backend);
        let base: Arc<dyn LlmBackend> = backend.clone();
        Ok(PyClassInitializer::from(PyLlmBackend::from_native(base))
            .add_subclass(Self { inner: backend }))
    }

    #[getter]
    fn target(&self) -> PyLlmTarget {
        PyLlmTarget::from_core(self.inner.target().clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "AnthropicNativeBackend(target_id={:?}, model={:?})",
            self.inner.target().id.as_str(),
            self.inner.target().model.as_str(),
        )
    }
}

#[pyclass(
    name = "MultiLlmBackend",
    extends = PyLlmBackend,
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub(crate) struct PyMultiLlmBackend {
    inner: Arc<MultiLlmBackend>,
}

#[pymethods]
impl PyMultiLlmBackend {
    #[new]
    #[pyo3(signature = (targets, supported_request_types=None, default_target_id=None))]
    fn py_new(
        targets: &Bound<'_, PyAny>,
        supported_request_types: Option<&Bound<'_, PyAny>>,
        default_target_id: Option<String>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let targets = target_backends_from_python(targets)?;
        let mut backend = MultiLlmBackend::new(targets).map_err(py_core_error)?;
        if let Some(supported_request_types) = request_types_from_python(supported_request_types)? {
            backend = backend
                .with_supported_request_types(supported_request_types)
                .map_err(py_core_error)?;
        }
        if let Some(default_target_id) = default_target_id {
            let default_target_id = LlmTargetId::new(default_target_id).map_err(|error| {
                PyValueError::new_err(format!("invalid default target id: {error}"))
            })?;
            backend = backend
                .with_default_target(default_target_id)
                .map_err(py_core_error)?;
        }
        let backend = Arc::new(backend);
        let base: Arc<dyn LlmBackend> = backend.clone();
        Ok(PyClassInitializer::from(PyLlmBackend::from_native(base))
            .add_subclass(Self { inner: backend }))
    }

    fn target_ids(&self) -> Vec<String> {
        self.inner
            .targets()
            .iter()
            .map(|target| target.target().id.as_str().to_string())
            .collect()
    }

    fn default_target_id(&self) -> Option<String> {
        self.inner
            .default_target_id()
            .map(|target_id| target_id.as_str().to_string())
    }

    fn __repr__(&self) -> String {
        format!(
            "MultiLlmBackend(target_ids={:?}, default_target_id={:?})",
            self.target_ids(),
            self.default_target_id(),
        )
    }
}

fn native_backend(backend: &PyRef<'_, PyLlmBackend>, owner: &str) -> PyResult<Arc<dyn LlmBackend>> {
    backend.native().ok_or_else(|| {
        PyTypeError::new_err(format!(
            "{owner} requires a Rust-native LLMBackend binding, not a Python-only subclass"
        ))
    })
}

fn target_backends_from_python(value: &Bound<'_, PyAny>) -> PyResult<Vec<LlmTargetBackend>> {
    let iterator = PyIterator::from_object(value)?;
    let mut targets = Vec::new();
    for item in iterator {
        let item = item?;
        targets.push(target_backend_from_python(&item)?);
    }
    Ok(targets)
}

fn target_backend_from_python(value: &Bound<'_, PyAny>) -> PyResult<LlmTargetBackend> {
    if let Ok(target_backend) = value.extract::<PyRef<'_, PyLlmTargetBackend>>() {
        return Ok(target_backend.clone_core());
    }

    let tuple = value.cast::<PyTuple>().map_err(|_| {
        PyTypeError::new_err(
            "MultiLlmBackend targets must be LlmTargetBackend objects or (target, backend) tuples",
        )
    })?;
    if tuple.len() != 2 {
        return Err(PyValueError::new_err(
            "MultiLlmBackend target tuples must contain exactly (target, backend)",
        ));
    }

    let target = tuple.get_item(0)?.extract::<PyRef<'_, PyLlmTarget>>()?;
    let backend = tuple.get_item(1)?.extract::<PyRef<'_, PyLlmBackend>>()?;
    Ok(LlmTargetBackend::new(
        target.clone_core(),
        native_backend(&backend, "MultiLlmBackend")?,
    ))
}

fn request_types_from_python(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Vec<ChatRequestType>>> {
    let Some(value) = value.filter(|value| !value.is_none()) else {
        return Ok(None);
    };
    let iterator = PyIterator::from_object(value)?;
    let mut request_types = Vec::new();
    for item in iterator {
        request_types.push(request_type_from_python(&item?)?);
    }
    Ok(Some(request_types))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyLlmTargetBackend>()?;
    module.add_class::<PyOpenAiNativeBackend>()?;
    module.add_class::<PyOpenAiPassthroughBackend>()?;
    module.add_class::<PyAnthropicNativeBackend>()?;
    module.add_class::<PyMultiLlmBackend>()?;
    Ok(())
}
