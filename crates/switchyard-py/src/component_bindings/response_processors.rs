// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for concrete response-side components.

use pyo3::prelude::*;
use switchyard_components::IntakeResponseProcessor;

use super::config::PyIntakeSinkConfig;
use crate::core_bindings::context::PyProxyContext;
use crate::core_bindings::response::PyChatResponse;
use crate::errors::py_core_error;

#[pyclass(name = "IntakeResponseProcessor", skip_from_py_object)]
#[derive(Clone, Debug)]
pub(crate) struct PyIntakeResponseProcessor {
    inner: IntakeResponseProcessor,
    config: PyIntakeSinkConfig,
}

#[pymethods]
impl PyIntakeResponseProcessor {
    #[new]
    fn py_new(config: PyRef<'_, PyIntakeSinkConfig>) -> PyResult<Self> {
        let config = PyIntakeSinkConfig::from_core(config.clone_core());
        let inner =
            IntakeResponseProcessor::with_http_sink(config.clone_core()).map_err(py_core_error)?;
        Ok(Self { inner, config })
    }

    #[getter]
    fn config(&self) -> PyIntakeSinkConfig {
        self.config.clone()
    }

    fn startup<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, async { Ok(()) })
    }

    fn shutdown<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let processor = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            processor.shutdown().await.map_err(py_core_error)
        })
    }

    fn process<'py>(
        &self,
        py: Python<'py>,
        ctx: PyRef<'_, PyProxyContext>,
        mut response: PyRefMut<'_, PyChatResponse>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let processor = self.inner.clone();
        let mut lease = ctx.lease()?;
        let response = response.take_core(py)?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = processor.process(lease.context_mut()?, response).await;
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
        "IntakeResponseProcessor()"
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyIntakeResponseProcessor>()?;
    Ok(())
}
