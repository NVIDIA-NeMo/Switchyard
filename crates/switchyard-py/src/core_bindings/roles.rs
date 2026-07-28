// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for Rust-owned backend role abstractions.

use std::sync::Arc;

use pyo3::exceptions::{PyNotImplementedError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple, PyType};
use pyo3::PyTypeInfo;
use switchyard_core::LlmBackend;

use super::context::PyProxyContext;
use super::request::{request_type_object, PyChatRequest};
use super::response::PyChatResponse;
use crate::errors::py_core_error;

#[pyclass(name = "LLMBackend", subclass)]
pub(crate) struct PyLlmBackend {
    inner: Option<Arc<dyn LlmBackend>>,
}

impl PyLlmBackend {
    pub(crate) fn from_native(inner: Arc<dyn LlmBackend>) -> Self {
        Self { inner: Some(inner) }
    }
}

#[pymethods]
impl PyLlmBackend {
    #[new]
    #[classmethod]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn py_new(
        cls: &Bound<'_, PyType>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        reject_base_instantiation::<Self>(cls, "LLMBackend")?;
        Ok(Self { inner: None })
    }

    #[getter]
    fn supported_request_types(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let backend = self.inner.as_ref().ok_or_else(|| {
            PyNotImplementedError::new_err("LLMBackend.supported_request_types must be implemented")
        })?;
        backend
            .supported_request_types()
            .iter()
            .map(|request_type| request_type_object(py, *request_type))
            .collect()
    }

    fn startup<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.inner.clone() {
            Some(backend) => pyo3_async_runtimes::tokio::future_into_py(py, async move {
                backend.startup().await.map_err(py_core_error)
            }),
            None => pyo3_async_runtimes::tokio::future_into_py(py, async { Ok(()) }),
        }
    }

    fn shutdown<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.inner.clone() {
            Some(backend) => pyo3_async_runtimes::tokio::future_into_py(py, async move {
                backend.shutdown().await.map_err(py_core_error)
            }),
            None => pyo3_async_runtimes::tokio::future_into_py(py, async { Ok(()) }),
        }
    }

    fn call<'py>(
        &self,
        py: Python<'py>,
        ctx: PyRef<'_, PyProxyContext>,
        request: PyRef<'_, PyChatRequest>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let backend = self
            .inner
            .clone()
            .ok_or_else(|| PyNotImplementedError::new_err("LLMBackend.call must be implemented"))?;
        let mut lease = ctx.lease()?;
        let request = request.clone_core();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = backend.call(lease.context_mut()?, &request).await;
            let restore_result = lease.restore();
            let response = result.map_err(py_core_error)?;
            restore_result?;
            Python::attach(|py| {
                Py::new(py, PyChatResponse::from_core(py, response)?)
                    .map(|response| response.into_any())
            })
        })
    }

    fn __repr__(&self) -> &'static str {
        "LLMBackend()"
    }
}

fn reject_base_instantiation<T: PyTypeInfo>(
    cls: &Bound<'_, PyType>,
    name: &'static str,
) -> PyResult<()> {
    if cls.is(cls.py().get_type::<T>()) {
        return Err(PyTypeError::new_err(format!(
            "can't instantiate abstract role {name}"
        )));
    }
    Ok(())
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyLlmBackend>()?;
    Ok(())
}
