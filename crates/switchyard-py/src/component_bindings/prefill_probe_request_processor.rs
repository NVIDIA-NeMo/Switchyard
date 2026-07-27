// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python binding for the learned prefill-probe request processor.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use switchyard_components::{PrefillProbeProcessorConfig, PrefillProbeRequestProcessor};
use switchyard_core::LlmTargetId;

use crate::core_bindings::context::PyProxyContext;
use crate::core_bindings::request::PyChatRequest;
use crate::errors::py_core_error;

#[pyclass(name = "PrefillProbeRequestProcessor", skip_from_py_object)]
#[derive(Clone, Debug)]
pub(crate) struct PyPrefillProbeRequestProcessor {
    inner: PrefillProbeRequestProcessor,
}

#[pymethods]
impl PyPrefillProbeRequestProcessor {
    #[new]
    #[pyo3(signature = (
        *,
        probe_base_url,
        probe_model,
        hidden_states_dir,
        checkpoint_dir,
        strong_checkpoint_head,
        weak_checkpoint_head,
        strong_target_id,
        weak_target_id,
        routing_lambda,
        weak_cost,
        strong_cost,
    ))]
    #[expect(
        clippy::too_many_arguments,
        reason = "the binding mirrors the explicit profile fields passed by Python"
    )]
    fn py_new(
        probe_base_url: String,
        probe_model: String,
        hidden_states_dir: String,
        checkpoint_dir: String,
        strong_checkpoint_head: String,
        weak_checkpoint_head: String,
        strong_target_id: String,
        weak_target_id: String,
        routing_lambda: f64,
        weak_cost: f64,
        strong_cost: f64,
    ) -> PyResult<Self> {
        let strong_target_id = target_id(strong_target_id, "strong_target_id")?;
        let weak_target_id = target_id(weak_target_id, "weak_target_id")?;
        let inner = PrefillProbeRequestProcessor::new(PrefillProbeProcessorConfig {
            probe_base_url,
            probe_model,
            hidden_states_dir: hidden_states_dir.into(),
            checkpoint_dir: checkpoint_dir.into(),
            strong_checkpoint_head,
            weak_checkpoint_head,
            strong_target_id,
            weak_target_id,
            lambda: routing_lambda,
            weak_cost,
            strong_cost,
        })
        .map_err(py_core_error)?;
        Ok(Self { inner })
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
        let processor = self.inner.clone();
        let mut lease = ctx.lease()?;
        let request = request.clone_core();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = processor.process(lease.context_mut()?, request).await;
            let restore_result = lease.restore();
            let request = result.map_err(py_core_error)?;
            restore_result?;
            Python::attach(|py| {
                Py::new(py, PyChatRequest::from_core(request)).map(|request| request.into_any())
            })
        })
    }

    fn __repr__(&self) -> &'static str {
        "PrefillProbeRequestProcessor()"
    }
}

fn target_id(value: String, field: &str) -> PyResult<LlmTargetId> {
    LlmTargetId::new(value)
        .map_err(|error| PyValueError::new_err(format!("invalid {field}: {error}")))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyPrefillProbeRequestProcessor>()?;
    Ok(())
}
