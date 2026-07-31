// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for concrete response-side components.

use pyo3::prelude::*;
use switchyard_components::StatsResponseProcessor;

use super::stats::PyStatsAccumulator;
use crate::errors::py_core_error;
use crate::interop::context::lease_from_python;
use crate::interop::response::{response_from_python, response_to_python};

#[pyclass(name = "StatsResponseProcessor", skip_from_py_object)]
#[derive(Clone, Debug)]
pub(crate) struct PyStatsResponseProcessor {
    inner: StatsResponseProcessor,
    accumulator: PyStatsAccumulator,
}

#[pymethods]
impl PyStatsResponseProcessor {
    #[new]
    fn py_new(accumulator: PyRef<'_, PyStatsAccumulator>) -> Self {
        let accumulator = PyStatsAccumulator::from_core(accumulator.clone_core());
        Self {
            inner: StatsResponseProcessor::new(accumulator.clone_core()),
            accumulator,
        }
    }

    #[getter]
    fn accumulator(&self) -> PyStatsAccumulator {
        self.accumulator.clone()
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
        response: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let processor = self.inner.clone();
        let mut lease = lease_from_python(ctx)?;
        let response = response_from_python(response)?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = processor.process(lease.context_mut()?, response).await;
            let restore_result = lease.restore();
            let response = result.map_err(py_core_error)?;
            restore_result?;
            Python::attach(|py| response_to_python(py, response))
        })
    }

    fn get_endpoint(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let endpoint = py
            .import("switchyard.lib.endpoints.stats_endpoint")?
            .getattr("StatsEndpoint")?
            .call1((self.accumulator.clone(),))?;
        Ok(endpoint.unbind())
    }

    fn __repr__(&self) -> &'static str {
        "StatsResponseProcessor()"
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyStatsResponseProcessor>()?;
    Ok(())
}
