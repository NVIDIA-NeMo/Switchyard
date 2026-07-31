// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal Python API for running Rust-owned libsy algorithms.

use std::sync::Arc;

use async_trait::async_trait;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use serde_json::{json, Value};
use switchyard_libsy::{
    Algorithm, HandoffNoteConfig, LibsyError as RustLibsyError, LlmFallback, LlmTarget,
    LlmTargetSet, LlmTaskClassifier, Noop, PickerMode, Random, StageRouter, StageRouterConfig,
    TaskClassifierConfig,
};
use switchyard_protocol::{
    AggLlmResponse, Context, Decision, LlmClientError, LlmResponse, Metadata, Request, Response,
    RoutedLlmClient,
};

use crate::errors::py_libsy_error;
use crate::py_serde::{from_python, to_python};

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
    ) -> Result<Response, LlmClientError> {
        let metadata = request.metadata;
        let future = Python::attach(|py| {
            let request = to_python(py, &request.llm_request)?;
            let awaitable = self.inner.bind(py).call_method1("call", (request,))?;
            pyo3_async_runtimes::tokio::into_future(awaitable)
        })
        .map_err(other_python_error)?;

        let response = future.await.map_err(other_python_error)?;
        let aggregate = Python::attach(|py| from_python::<AggLlmResponse>(response.bind(py)))
            .map_err(invalid_python_response)?;
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

/// Classifier thresholds shared by standalone and stage-router classifiers.
#[pyclass(
    name = "TaskClassifierConfig",
    module = "switchyard.libsy",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
struct PyTaskClassifierConfig {
    inner: TaskClassifierConfig,
}

impl PyTaskClassifierConfig {
    fn clone_core(&self) -> TaskClassifierConfig {
        self.inner.clone()
    }
}

#[pymethods]
impl PyTaskClassifierConfig {
    #[new]
    #[pyo3(signature = (
        base_threshold,
        *,
        min_confidence=0.0,
        capability_elevated_floor=None,
        session_affinity=false,
        message_hash_fallback=false,
        recent_turn_window=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        base_threshold: f64,
        min_confidence: f64,
        capability_elevated_floor: Option<f64>,
        session_affinity: bool,
        message_hash_fallback: bool,
        recent_turn_window: Option<usize>,
    ) -> Self {
        Self {
            inner: TaskClassifierConfig {
                base_threshold,
                min_confidence,
                capability_elevated_floor,
                session_affinity,
                message_hash_fallback,
                recent_turn_window,
            },
        }
    }
}

/// Judge target and policy used when stage-router signals are inconclusive.
#[pyclass(
    name = "LlmFallback",
    module = "switchyard.libsy",
    frozen,
    skip_from_py_object
)]
struct PyLlmFallback {
    judge_target: Py<PyLlmTarget>,
    config: Py<PyTaskClassifierConfig>,
}

impl PyLlmFallback {
    fn clone_core(&self, py: Python<'_>) -> PyResult<LlmFallback> {
        Ok(LlmFallback {
            judge_target: self.judge_target.bind(py).try_borrow()?.clone_core(py),
            config: self.config.bind(py).try_borrow()?.clone_core(),
        })
    }
}

#[pymethods]
impl PyLlmFallback {
    #[new]
    #[pyo3(signature = (judge_target, *, config))]
    fn new(judge_target: Py<PyLlmTarget>, config: Py<PyTaskClassifierConfig>) -> Self {
        Self {
            judge_target,
            config,
        }
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
    /// so metadata-driven algorithms see the same signals in Python as when
    /// served over HTTP.
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

/// Construct random routing over targets with optional relative weights and seed.
#[pyfunction(name = "random")]
#[pyo3(signature = (targets, *, weights=None, seed=None))]
fn random_algorithm(
    py: Python<'_>,
    targets: Vec<Py<PyLlmTarget>>,
    weights: Option<Vec<f64>>,
    seed: Option<u64>,
) -> PyResult<PyAlgorithm> {
    let targets = targets
        .iter()
        .map(|target| Ok(target.bind(py).try_borrow()?.clone_core(py)))
        .collect::<PyResult<Vec<_>>>()?;
    let algorithm =
        Random::new(LlmTargetSet::new(targets), weights, seed).map_err(|error| match error {
            RustLibsyError::NoTargets => {
                PyValueError::new_err("random requires at least one target")
            }
            other => PyValueError::new_err(other.to_string()),
        })?;
    Ok(PyAlgorithm::new(Arc::new(algorithm)))
}

/// Construct task-level LLM classifier routing.
#[pyfunction(name = "llm_task_classifier")]
#[pyo3(signature = (
    judge_target,
    efficient_target,
    capable_target,
    *,
    config
))]
fn llm_task_classifier_algorithm(
    py: Python<'_>,
    judge_target: Py<PyLlmTarget>,
    efficient_target: Py<PyLlmTarget>,
    capable_target: Py<PyLlmTarget>,
    config: Py<PyTaskClassifierConfig>,
) -> PyResult<PyAlgorithm> {
    let algorithm = LlmTaskClassifier::new(
        judge_target.bind(py).try_borrow()?.clone_core(py),
        efficient_target.bind(py).try_borrow()?.clone_core(py),
        capable_target.bind(py).try_borrow()?.clone_core(py),
        config.bind(py).try_borrow()?.clone_core(),
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyAlgorithm::new(Arc::new(algorithm)))
}

/// Construct signal-driven stage routing with an optional LLM classifier fallback.
#[pyfunction(name = "stage_router")]
#[pyo3(signature = (
    capable_target,
    efficient_target,
    *,
    picker,
    confidence_threshold,
    recent_window=None,
    escalation_note=None,
    deescalation_note=None,
    only_on_wrong_signal_escalation=true,
    capable_system_prompt=None,
    efficient_system_prompt=None,
    classifier=None
))]
#[allow(clippy::too_many_arguments)]
fn stage_router_algorithm(
    py: Python<'_>,
    capable_target: Py<PyLlmTarget>,
    efficient_target: Py<PyLlmTarget>,
    picker: &str,
    confidence_threshold: f64,
    recent_window: Option<usize>,
    escalation_note: Option<String>,
    deescalation_note: Option<String>,
    only_on_wrong_signal_escalation: bool,
    capable_system_prompt: Option<String>,
    efficient_system_prompt: Option<String>,
    classifier: Option<Py<PyLlmFallback>>,
) -> PyResult<PyAlgorithm> {
    let mode = match picker {
        "capable_first" => PickerMode::CapableFirst,
        "efficient_first" => PickerMode::EfficientFirst,
        other => {
            return Err(PyValueError::new_err(format!(
                "picker must be 'capable_first' or 'efficient_first', got {other:?}"
            )));
        }
    };
    let capable = capable_target.bind(py).try_borrow()?.clone_core(py);
    let efficient = efficient_target.bind(py).try_borrow()?.clone_core(py);
    let mut config = StageRouterConfig::new(mode, confidence_threshold);
    config.recent_window = recent_window;
    config.handoff_notes = match (escalation_note, deescalation_note) {
        (Some(escalation), deescalation) => Some(HandoffNoteConfig::new(
            escalation,
            deescalation,
            only_on_wrong_signal_escalation,
        )),
        (None, Some(_)) => {
            return Err(PyValueError::new_err(
                "deescalation_note requires escalation_note",
            ));
        }
        (None, None) => None,
    };
    if let Some(prompt) = capable_system_prompt {
        config.tier_prompts = config
            .tier_prompts
            .with(capable.semantic_name.clone(), prompt);
    }
    if let Some(prompt) = efficient_system_prompt {
        config.tier_prompts = config
            .tier_prompts
            .with(efficient.semantic_name.clone(), prompt);
    }
    config.llm_fallback = classifier
        .map(|classifier| classifier.bind(py).try_borrow()?.clone_core(py))
        .transpose()?;

    let algorithm = StageRouter::new(capable, efficient, config)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyAlgorithm::new(Arc::new(algorithm)))
}

fn other_python_error(error: PyErr) -> LlmClientError {
    LlmClientError::Ffi {
        source: Box::new(error),
    }
}

fn invalid_python_response(error: PyErr) -> LlmClientError {
    LlmClientError::InvalidResponse {
        source: Box::new(error),
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let libsy_module = PyModule::new(module.py(), "libsy")?;
    libsy_module.add_class::<PyAlgorithm>()?;
    libsy_module.add_class::<PyLlmFallback>()?;
    libsy_module.add_class::<PyLlmTarget>()?;
    libsy_module.add_class::<PyTaskClassifierConfig>()?;
    libsy_module.add_function(wrap_pyfunction!(noop_algorithm, &libsy_module)?)?;
    libsy_module.add_function(wrap_pyfunction!(random_algorithm, &libsy_module)?)?;
    libsy_module.add_function(wrap_pyfunction!(
        llm_task_classifier_algorithm,
        &libsy_module
    )?)?;
    libsy_module.add_function(wrap_pyfunction!(stage_router_algorithm, &libsy_module)?)?;
    libsy_module.add("LibsyError", module.getattr("LibsyError")?)?;
    module.add_submodule(&libsy_module)?;
    Ok(())
}
