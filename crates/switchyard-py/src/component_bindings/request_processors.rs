// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for concrete request-side compatibility components.

use pyo3::prelude::*;
use switchyard_components::{
    IntakeRequestProcessor, StatsRequestProcessor, tracking_enabled_from_env,
};

use crate::errors::py_core_error;
use crate::interop::context::lease_from_python;
use crate::interop::request::{request_from_python, request_to_python};

#[pyclass(name = "StatsRequestProcessor", skip_from_py_object)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct PyStatsRequestProcessor {
    inner: StatsRequestProcessor,
}

#[pymethods]
impl PyStatsRequestProcessor {
    #[new]
    fn py_new() -> Self {
        Self {
            inner: StatsRequestProcessor::new(tracking_enabled_from_env()),
        }
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
        ctx: &Bound<'_, PyAny>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let processor = self.inner;
        let mut lease = lease_from_python(ctx)?;
        let request = request_from_python(request)?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = processor.process(lease.context_mut()?, request).await;
            let restore_result = lease.restore();
            let request = result.map_err(py_core_error)?;
            restore_result?;
            Python::attach(|py| request_to_python(py, request))
        })
    }

    fn __repr__(&self) -> &'static str {
        "StatsRequestProcessor()"
    }
}

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
        ctx: &Bound<'_, PyAny>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut lease = lease_from_python(ctx)?;
        let request = request_from_python(request)?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = IntakeRequestProcessor
                .process(lease.context_mut()?, request)
                .await;
            let restore_result = lease.restore();
            let request = result.map_err(py_core_error)?;
            restore_result?;
            Python::attach(|py| request_to_python(py, request))
        })
    }

    fn __repr__(&self) -> &'static str {
        "IntakeRequestProcessor()"
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyStatsRequestProcessor>()?;
    module.add_class::<PyIntakeRequestProcessor>()?;
    Ok(())
}
