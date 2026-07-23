// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal Python API for running Rust-owned libsy algorithms.

use std::error::Error;
use std::sync::Arc;

use async_trait::async_trait;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use serde_json::{json, Value};
use switchyard_libsy::algorithms::{Noop, Random, SubagentOverride};
use switchyard_libsy::{
    AggLlmResponse, Algorithm, Context, Decision, LlmResponse, LlmTarget, LlmTargetSet, Metadata,
    Request, Response, RoutedLlmClient,
};

use crate::errors::py_libsy_error;
use crate::py_serde::{from_python, to_python};

type BoxError = Box<dyn Error + Send + Sync>;

/// Adapts a Python object with `async call(request)` to libsy.
struct PythonLlmClient {
    inner: Py<PyAny>,
}

#[async_trait]
impl RoutedLlmClient for PythonLlmClient {
    async fn call(
        &self,
        _ctx: Context,
        request: Request,
        _decision: Arc<dyn Decision>,
    ) -> Result<Response, BoxError> {
        let metadata = request.metadata;
        let future = Python::attach(|py| {
            let request = to_python(py, &request.llm_request)?;
            let awaitable = self.inner.bind(py).call_method1("call", (request,))?;
            pyo3_async_runtimes::tokio::into_future(awaitable)
        })
        .map_err(boxed_python_error)?;

        let response = future.await.map_err(boxed_python_error)?;
        let aggregate = Python::attach(|py| from_python::<AggLlmResponse>(response.bind(py)))
            .map_err(boxed_python_error)?;
        Ok(Response {
            llm_response: LlmResponse::Agg(aggregate),
            metadata,
        })
    }
}

/// A required-client routing target used by Python-created algorithms.
#[pyclass(name = "LlmTarget", module = "switchyard.libsy", frozen)]
struct PyLlmTarget {
    name: String,
    client: Py<PyAny>,
}

impl PyLlmTarget {
    fn clone_core(&self, py: Python<'_>) -> LlmTarget {
        LlmTarget {
            semantic_name: self.name.clone(),
            llm_client: Some(Arc::new(PythonLlmClient {
                inner: self.client.clone_ref(py),
            })),
        }
    }
}

#[pymethods]
impl PyLlmTarget {
    #[new]
    fn new(py: Python<'_>, name: String, client: Py<PyAny>) -> PyResult<Self> {
        let call = client
            .bind(py)
            .getattr("call")
            .map_err(|_| PyTypeError::new_err("client must define async call(request)"))?;
        if !call.is_callable() {
            return Err(PyTypeError::new_err(
                "client.call must be callable as async call(request)",
            ));
        }
        Ok(Self { name, client })
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    fn __repr__(&self) -> String {
        format!("LlmTarget(name={:?})", self.name)
    }
}

/// Opaque handle shared by every Rust-owned algorithm exposed to Python.
#[pyclass(name = "Algorithm", module = "switchyard.libsy", frozen)]
struct PyAlgorithm {
    inner: Arc<dyn Algorithm>,
}

impl PyAlgorithm {
    fn new(inner: Arc<dyn Algorithm>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyAlgorithm {
    /// Run to completion using the clients configured on the algorithm's targets.
    ///
    /// `headers`, when given, is normalized into the request's correlation
    /// [`Metadata`] exactly as an HTTP host would (`Metadata::from_headers`),
    /// so metadata-driven algorithms such as `subagent_override` see the same
    /// signals in Python as when served over HTTP.
    #[pyo3(signature = (request, headers=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
        headers: Option<std::collections::BTreeMap<String, String>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let algorithm = Arc::clone(&self.inner);
        let request = Request {
            llm_request: from_python(request)?,
            raw_request: None,
            metadata: headers.map(|headers| Metadata::from_headers(&headers)),
        };
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (decisions, response) = algorithm
                .run(Context::default(), request)
                .await
                .map_err(py_libsy_error)?;
            let response = response
                .llm_response
                .into_agg()
                .await
                .map_err(py_libsy_error)?;
            let decisions = decisions
                .iter()
                .map(|decision| {
                    json!({
                        "selected_model": decision.selected_model(),
                        "reasoning": decision.reasoning(),
                    })
                })
                .collect::<Vec<Value>>();
            Python::attach(|py| Ok((to_python(py, &decisions)?, to_python(py, &response)?)))
        })
    }

    fn __repr__(&self) -> &'static str {
        "Algorithm()"
    }
}

/// Construct the no-op reference algorithm.
#[pyfunction(name = "noop")]
fn noop_algorithm() -> PyAlgorithm {
    PyAlgorithm::new(Arc::new(Noop {}))
}

/// Construct uniform random routing over targets with Python clients.
#[pyfunction(name = "random")]
fn random_algorithm(py: Python<'_>, targets: Vec<Py<PyLlmTarget>>) -> PyResult<PyAlgorithm> {
    if targets.is_empty() {
        return Err(PyValueError::new_err("random requires at least one target"));
    }
    let targets = targets
        .iter()
        .map(|target| Ok(target.bind(py).try_borrow()?.clone_core(py)))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyAlgorithm::new(Arc::new(Random::new(LlmTargetSet::new(
        targets,
    )))))
}

/// Wrap `inner`, routing delegated sub-agent work to a fixed worker target.
///
/// Detection and the work-vs-maintenance policy are the protocol crate's
/// `Metadata::from_headers` / `Metadata::is_subagent_work`, driven by the
/// `headers` passed to `Algorithm.run`.
#[pyfunction(name = "subagent_override")]
fn subagent_override_algorithm(
    py: Python<'_>,
    inner: Py<PyAlgorithm>,
    worker: Py<PyLlmTarget>,
) -> PyResult<PyAlgorithm> {
    let inner = Arc::clone(&inner.bind(py).get().inner);
    let worker = worker.bind(py).try_borrow()?.clone_core(py);
    Ok(PyAlgorithm::new(Arc::new(SubagentOverride::new(
        inner, worker,
    ))))
}

fn boxed_python_error(error: PyErr) -> BoxError {
    std::io::Error::other(error.to_string()).into()
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let libsy_module = PyModule::new(module.py(), "libsy")?;
    libsy_module.add_class::<PyAlgorithm>()?;
    libsy_module.add_class::<PyLlmTarget>()?;
    libsy_module.add_function(wrap_pyfunction!(noop_algorithm, &libsy_module)?)?;
    libsy_module.add_function(wrap_pyfunction!(random_algorithm, &libsy_module)?)?;
    libsy_module.add_function(wrap_pyfunction!(
        subagent_override_algorithm,
        &libsy_module
    )?)?;
    libsy_module.add("LibsyError", module.getattr("LibsyError")?)?;
    module.add_submodule(&libsy_module)?;
    Ok(())
}
