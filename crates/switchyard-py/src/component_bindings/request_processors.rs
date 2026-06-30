// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for concrete request-side compatibility components.

use pyo3::prelude::*;
use switchyard_components::IntakeRequestProcessor;

use crate::core_bindings::context::PyProxyContext;
use crate::core_bindings::request::PyChatRequest;
use crate::errors::py_core_error;

#[pyclass(name = "IntakeRequestProcessor", skip_from_py_object)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PyIntakeRequestProcessor;

#[pymethods]
impl PyIntakeRequestProcessor {
    #[new]
    fn py_new() -> Self {
        Self
    }

    fn startup<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, async { Ok(()) })
    }

    fn shutdown<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, async { Ok(()) })
    }

    fn process<'py>(
        &self,
        py: Python<'py>,
        ctx: PyRef<'_, PyProxyContext>,
        request: PyRef<'_, PyChatRequest>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut lease = ctx.lease()?;
        let request = request.clone_core();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = IntakeRequestProcessor
                .process(lease.context_mut()?, request)
                .await;
            let restore_result = lease.restore();
            let request = result.map_err(py_core_error)?;
            restore_result?;
            Python::attach(|py| {
                Py::new(py, PyChatRequest::from_core(request)).map(|request| request.into_any())
            })
        })
    }

    fn __repr__(&self) -> &'static str {
        "IntakeRequestProcessor()"
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyIntakeRequestProcessor>()?;
    Ok(())
}
