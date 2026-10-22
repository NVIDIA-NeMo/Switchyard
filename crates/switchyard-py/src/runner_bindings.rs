// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TOML-configured routing decisions for Python hosts that serve the final completion.

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use switchyard_protocol::{LlmRequest, Metadata, Request};
use switchyard_runner::{Runner, RunnerError};

use crate::errors::{ServerConfigError, py_libsy_error};
use crate::libsy_bindings::{header_map_from_python, routing_outcome_to_python};
use crate::py_serde::from_python;

/// Persistent configured routes; native clients serve routing-time calls only.
#[pyclass(name = "Runner", module = "switchyard.runner", frozen)]
struct PyRunner {
    inner: Arc<Runner>,
}

#[pymethods]
impl PyRunner {
    /// Loads a deployment using the existing native TOML parser and clients.
    #[staticmethod]
    fn from_toml(py: Python<'_>, source: &str) -> PyResult<Self> {
        let runner = py
            .detach(|| Runner::from_toml(source))
            .map_err(|error| ServerConfigError::new_err(error.to_string()))?;
        Ok(Self {
            inner: Arc::new(runner),
        })
    }

    /// Routes a normalized Switchyard request, not a raw provider body.
    ///
    /// Native TOML clients serve routing-time calls. The outcome preserves the selected
    /// request and any answer already produced; otherwise the host serves the completion.
    /// Only the selected target's prompt is applied. The host owns provider body settings
    /// and fallback preparation. Reuse this runner and session headers for session routing.
    /// Invalid requests raise ValueError; routing failures raise LibsyError.
    #[pyo3(signature = (request, *, headers=None))]
    fn decide<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'py, PyAny>,
        headers: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let headers = headers.map(from_python).transpose()?;
        let llm_request: LlmRequest = from_python(request)?;
        let metadata = headers
            .as_ref()
            .map(header_map_from_python)
            .transpose()?
            .map(|headers| {
                let mut metadata = Metadata::from_headers(&headers);
                metadata.http_headers = Some(headers);
                metadata
            });
        let request = Request {
            llm_request,
            raw_request: None,
            metadata,
        };
        let runner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let model = request
                .llm_request
                .model
                .as_deref()
                .ok_or_else(|| PyValueError::new_err("request.model is required"))?;
            let route = runner
                .route(model)
                .ok_or_else(|| py_libsy_error(RunnerError::UnknownRouteModel(model.to_string())))?;
            let outcome = route.decide(request).await.map_err(py_libsy_error)?;
            Python::attach(|py| routing_outcome_to_python(py, outcome))
        })
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let runner_module = PyModule::new(module.py(), "runner")?;
    runner_module.add_class::<PyRunner>()?;
    module.add_submodule(&runner_module)
}
