// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configured routing decisions with trusted requirements supplied by a Python host.

use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use switchyard_protocol::{LlmClientError, Request};
use switchyard_runner::{Runner, RunnerError, require_target_labels};

use crate::errors::{ServerConfigError, py_libsy_error};
use crate::libsy_bindings::{header_map_from_python, routing_outcome_to_python};
use crate::py_serde::from_python;

/// Persistent routes using native clients for routing-time model calls.
#[pyclass(name = "Runner", module = "switchyard.runner", frozen)]
struct PythonRunner {
    inner: Arc<Runner>,
}

#[pymethods]
impl PythonRunner {
    /// Loads deployment TOML; invalid configuration raises ServerConfigError.
    #[staticmethod]
    fn from_toml(py: Python<'_>, source: &str) -> PyResult<Self> {
        let runner = py
            .detach(|| Runner::from_toml(source))
            .map_err(|error| ServerConfigError::new_err(error.to_string()))?;
        Ok(Self {
            inner: Arc::new(runner),
        })
    }

    /// Routes normalized input after checking the host's requirements and task state.
    /// Unspecified requirements differ from an explicitly empty requirement set.
    #[pyo3(signature = (request, *, headers=None, required_labels=None, task_id=None))]
    fn decide<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'py, PyAny>,
        headers: Option<&Bound<'py, PyAny>>,
        required_labels: Option<Vec<String>>,
        task_id: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let headers = headers
            .map(from_python::<HashMap<String, String>>)
            .transpose()?;
        let headers = headers.as_ref().map(header_map_from_python).transpose()?;
        let mut request = Request {
            llm_request: from_python(request)?,
            metadata: headers.map(|headers| {
                let mut metadata = switchyard_protocol::Metadata::from_headers(&headers);
                metadata.http_headers = Some(headers);
                metadata
            }),
            ..Request::default()
        };
        if let Some(task_id) = task_id {
            request.metadata.get_or_insert_default().task_id = Some(task_id);
        }
        if let Some(labels) = required_labels {
            require_target_labels(&mut request, labels)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
        }
        let runner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let model = request
                .llm_request
                .model
                .as_deref()
                .ok_or_else(|| PyValueError::new_err("request.model is required"))?;
            let route = runner
                .route(model)
                .ok_or_else(|| runner_error(RunnerError::UnknownRouteModel(model.to_owned())))?;
            let outcome = route.decide(request).await.map_err(runner_error)?;
            Python::attach(|py| routing_outcome_to_python(py, outcome))
        })
    }
}

fn runner_error(error: RunnerError) -> PyErr {
    match error {
        RunnerError::Client(LlmClientError::InvalidRequest { .. })
        | RunnerError::UnknownRouteModel(_) => PyValueError::new_err(error.to_string()),
        _ => py_libsy_error(error),
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let runner_module = PyModule::new(module.py(), "runner")?;
    runner_module.add_class::<PythonRunner>()?;
    module.add_submodule(&runner_module)
}
