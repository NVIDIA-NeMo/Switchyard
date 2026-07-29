// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for shared stats state.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::Serialize;
use switchyard_components::{StatsAccumulator, StatsRouteLabel, TokenUsage};

use crate::errors::py_core_error;
use crate::interop::context::insert_into_python;
use crate::py_serde::value_to_python;

#[pyclass(name = "StatsAccumulator", skip_from_py_object)]
#[derive(Clone, Debug)]
pub(crate) struct PyStatsAccumulator {
    inner: StatsAccumulator,
}

impl PyStatsAccumulator {
    pub(crate) fn from_core(inner: StatsAccumulator) -> Self {
        Self { inner }
    }

    pub(crate) fn clone_core(&self) -> StatsAccumulator {
        self.inner.clone()
    }
}

#[pymethods]
impl PyStatsAccumulator {
    #[new]
    fn py_new() -> Self {
        Self {
            inner: StatsAccumulator::new(),
        }
    }

    #[pyo3(signature = (model, backend_latency_ms=None, tier=None))]
    fn record_success<'py>(
        &self,
        py: Python<'py>,
        model: String,
        backend_latency_ms: Option<f64>,
        tier: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let accumulator = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            accumulator
                .record_success(model, backend_latency_ms, tier.as_deref())
                .map_err(py_core_error)
        })
    }

    #[pyo3(signature = (model, tier=None))]
    fn record_error<'py>(
        &self,
        py: Python<'py>,
        model: String,
        tier: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let accumulator = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            accumulator
                .record_error(model, tier.as_deref())
                .map_err(py_core_error)
        })
    }

    #[pyo3(signature = (
        model,
        prompt_tokens=0,
        completion_tokens=0,
        cached_tokens=0,
        cache_creation_tokens=0,
        reasoning_tokens=0,
        total_latency_ms=None,
        routing_overhead_ms=None,
        tier=None,
        success_was_untiered=false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn record_usage<'py>(
        &self,
        py: Python<'py>,
        model: String,
        prompt_tokens: u64,
        completion_tokens: u64,
        cached_tokens: u64,
        cache_creation_tokens: u64,
        reasoning_tokens: u64,
        total_latency_ms: Option<f64>,
        routing_overhead_ms: Option<f64>,
        tier: Option<String>,
        success_was_untiered: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let accumulator = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let usage = TokenUsage {
                prompt_tokens,
                completion_tokens,
                cached_tokens,
                cache_creation_tokens,
                reasoning_tokens,
                cacheable_prompt_tokens: 0,
            };
            let result = if success_was_untiered {
                accumulator.record_usage_with_success_was_untiered(
                    model,
                    usage,
                    total_latency_ms,
                    routing_overhead_ms,
                    tier.as_deref(),
                )
            } else {
                accumulator.record_usage(
                    model,
                    usage,
                    total_latency_ms,
                    routing_overhead_ms,
                    tier.as_deref(),
                )
            };
            result.map_err(py_core_error)
        })
    }

    #[pyo3(signature = (
        model,
        prompt_tokens=0,
        completion_tokens=0,
        cached_tokens=0,
        cache_creation_tokens=0,
        reasoning_tokens=0,
        latency_ms=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn record_classifier_usage<'py>(
        &self,
        py: Python<'py>,
        model: String,
        prompt_tokens: u64,
        completion_tokens: u64,
        cached_tokens: u64,
        cache_creation_tokens: u64,
        reasoning_tokens: u64,
        latency_ms: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let accumulator = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let usage = TokenUsage {
                prompt_tokens,
                completion_tokens,
                cached_tokens,
                cache_creation_tokens,
                reasoning_tokens,
                cacheable_prompt_tokens: 0,
            };
            accumulator
                .record_classifier_usage(model, usage, latency_ms)
                .map_err(py_core_error)
        })
    }

    fn record_classifier_error<'py>(
        &self,
        py: Python<'py>,
        model: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let accumulator = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            accumulator
                .record_classifier_error(model)
                .map_err(py_core_error)
        })
    }

    fn record_routing_decision<'py>(
        &self,
        py: Python<'py>,
        profile_type: String,
        source: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let accumulator = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            accumulator
                .record_routing_decision(profile_type, source)
                .map_err(py_core_error)
        })
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let accumulator = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let snapshot = accumulator.snapshot().map_err(py_core_error)?;
            Python::attach(|py| to_python(py, &snapshot))
        })
    }

    fn snapshot_sync(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let snapshot = self.inner.snapshot().map_err(py_core_error)?;
        to_python(py, &snapshot)
    }

    fn reset<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let accumulator = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            accumulator.reset().map_err(py_core_error)
        })
    }

    fn reset_sync(&self) -> PyResult<()> {
        self.inner.reset().map_err(py_core_error)
    }

    fn __repr__(&self) -> &'static str {
        "StatsAccumulator()"
    }
}

fn to_python(py: Python<'_>, value: &impl Serialize) -> PyResult<Py<PyAny>> {
    let value =
        serde_json::to_value(value).map_err(|error| PyValueError::new_err(error.to_string()))?;
    value_to_python(py, &value)
}

#[pyfunction]
fn set_stats_route_label(ctx: &Bound<'_, PyAny>, label: &str) -> PyResult<()> {
    let label = label.trim();
    if label.is_empty() {
        return Err(PyValueError::new_err("stats route label must not be empty"));
    }
    insert_into_python(ctx, StatsRouteLabel::new(label))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyStatsAccumulator>()?;
    module.add_function(wrap_pyfunction!(set_stats_route_label, module)?)?;
    Ok(())
}
