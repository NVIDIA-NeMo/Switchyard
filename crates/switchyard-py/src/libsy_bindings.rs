// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal Python API for running Rust-owned libsy algorithms.

use std::sync::Arc;

use async_trait::async_trait;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use serde_json::{json, Value};
use switchyard_libsy::algorithms::{
    LlmFallback, LlmTaskClassifier, Noop, Passthrough, Random, StageRouter, StageRouterConfig,
    TargetPrompts, TaskClassifierConfig,
};
use switchyard_libsy::stage_router::{HandoffNoteConfig, PickerMode};
use switchyard_libsy::{
    AggLlmResponse, Algorithm, Context, Decision, LibsyError as RustLibsyError, LlmClientError,
    LlmResponse, LlmTarget, LlmTargetSet, Metadata, Request, Response, RoutedLlmClient,
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

/// A named routing target with an optional Python-hosted client.
#[pyclass(name = "LlmTarget", module = "switchyard.libsy", frozen)]
struct PyLlmTarget {
    name: String,
    client: Option<Py<PyAny>>,
}

impl PyLlmTarget {
    fn clone_core(&self, py: Python<'_>) -> LlmTarget {
        LlmTarget {
            semantic_name: self.name.clone(),
            llm_client: self.client.as_ref().map(|client| {
                Arc::new(PythonLlmClient {
                    inner: client.clone_ref(py),
                }) as Arc<dyn RoutedLlmClient>
            }),
        }
    }
}

#[pymethods]
impl PyLlmTarget {
    #[new]
    #[pyo3(signature = (name, client=None))]
    fn new(py: Python<'_>, name: String, client: Option<Py<PyAny>>) -> PyResult<Self> {
        if let Some(client) = &client {
            let call = client
                .bind(py)
                .getattr("call")
                .map_err(|_| PyTypeError::new_err("client must define async call(request)"))?;
            if !call.is_callable() {
                return Err(PyTypeError::new_err(
                    "client.call must be callable as async call(request)",
                ));
            }
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

/// Serialize the stable metadata shared by run traces and decision-only results.
fn serialize_decision(decision: &dyn Decision) -> Value {
    json!({
        "selected_model": decision.selected_model(),
        "reasoning": decision.reasoning(),
        "routing_tier": decision.routing_tier(),
    })
}

/// Convert a normalized Python request and optional headers into libsy input.
fn request_from_python(
    request: &Bound<'_, PyAny>,
    headers: Option<std::collections::BTreeMap<String, String>>,
) -> PyResult<Request> {
    Ok(Request {
        llm_request: from_python(request)?,
        raw_request: None,
        metadata: headers.map(|headers| Metadata::from_headers(&headers)),
    })
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
        let request = request_from_python(request, headers)?;
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
                .map(|decision| serialize_decision(decision.as_ref()))
                .collect::<Vec<Value>>();
            Python::attach(|py| Ok((to_python(py, &decisions)?, to_python(py, &response)?)))
        })
    }

    /// Select one configured target without executing the selected model.
    #[pyo3(signature = (request, headers=None))]
    fn decide<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
        headers: Option<std::collections::BTreeMap<String, String>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let algorithm = Arc::clone(&self.inner);
        let request = request_from_python(request, headers)?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let decision = algorithm
                .decide(Context::default(), request)
                .await
                .map_err(py_libsy_error)?;
            Python::attach(|py| to_python(py, &serialize_decision(decision.as_ref())))
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

/// Construct fixed passthrough routing to one target.
#[pyfunction(name = "passthrough")]
fn passthrough_algorithm(py: Python<'_>, target: Py<PyLlmTarget>) -> PyResult<PyAlgorithm> {
    let target = target.bind(py).try_borrow()?.clone_core(py);
    Ok(PyAlgorithm::new(Arc::new(Passthrough::new(target))))
}

/// Construct judge-backed efficient/capable task routing.
#[pyfunction(name = "llm_classifier")]
#[pyo3(signature = (
    *,
    judge,
    efficient,
    capable,
    base_threshold,
    min_confidence=0.0,
    capability_elevated_floor=None,
    session_affinity=false,
    message_hash_fallback=false,
    recent_turn_window=None,
))]
#[allow(clippy::too_many_arguments)]
fn llm_classifier_algorithm(
    py: Python<'_>,
    judge: Py<PyLlmTarget>,
    efficient: Py<PyLlmTarget>,
    capable: Py<PyLlmTarget>,
    base_threshold: f64,
    min_confidence: f64,
    capability_elevated_floor: Option<f64>,
    session_affinity: bool,
    message_hash_fallback: bool,
    recent_turn_window: Option<usize>,
) -> PyResult<PyAlgorithm> {
    let judge = judge.bind(py).try_borrow()?.clone_core(py);
    let efficient = efficient.bind(py).try_borrow()?.clone_core(py);
    let capable = capable.bind(py).try_borrow()?.clone_core(py);
    let algorithm = LlmTaskClassifier::new(
        judge,
        efficient,
        capable,
        TaskClassifierConfig {
            base_threshold,
            min_confidence,
            capability_elevated_floor,
            session_affinity,
            message_hash_fallback,
            recent_turn_window,
        },
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyAlgorithm::new(Arc::new(algorithm)))
}

/// Construct signal-driven capable/efficient routing with an optional judge.
#[pyfunction(name = "stage_router")]
#[pyo3(signature = (
    *,
    capable,
    efficient,
    picker,
    confidence_threshold,
    recent_turn_window=None,
    handoff_escalation_note=None,
    handoff_deescalation_note=None,
    handoff_only_on_wrong_signal_escalation=true,
    capable_system_prompt=None,
    efficient_system_prompt=None,
    judge=None,
    classifier_base_threshold=None,
    classifier_min_confidence=0.0,
    classifier_capability_elevated_floor=None,
    classifier_recent_turn_window=None,
))]
#[allow(clippy::too_many_arguments)]
fn stage_router_algorithm(
    py: Python<'_>,
    capable: Py<PyLlmTarget>,
    efficient: Py<PyLlmTarget>,
    picker: &str,
    confidence_threshold: f64,
    recent_turn_window: Option<usize>,
    handoff_escalation_note: Option<String>,
    handoff_deescalation_note: Option<String>,
    handoff_only_on_wrong_signal_escalation: bool,
    capable_system_prompt: Option<String>,
    efficient_system_prompt: Option<String>,
    judge: Option<Py<PyLlmTarget>>,
    classifier_base_threshold: Option<f64>,
    classifier_min_confidence: f64,
    classifier_capability_elevated_floor: Option<f64>,
    classifier_recent_turn_window: Option<usize>,
) -> PyResult<PyAlgorithm> {
    let mode = match picker {
        "capable_first" => PickerMode::CapableFirst,
        "efficient_first" => PickerMode::EfficientFirst,
        other => {
            return Err(PyValueError::new_err(format!(
                "picker must be 'capable_first' or 'efficient_first', got {other:?}"
            )))
        }
    };
    if handoff_deescalation_note.is_some() && handoff_escalation_note.is_none() {
        return Err(PyValueError::new_err(
            "handoff_deescalation_note requires handoff_escalation_note",
        ));
    }
    if judge.is_none()
        && (classifier_min_confidence != 0.0
            || classifier_capability_elevated_floor.is_some()
            || classifier_recent_turn_window.is_some())
    {
        return Err(PyValueError::new_err(
            "classifier configuration options require judge",
        ));
    }
    if judge.is_some() != classifier_base_threshold.is_some() {
        return Err(PyValueError::new_err(
            "judge and classifier_base_threshold must be provided together",
        ));
    }

    let capable = capable.bind(py).try_borrow()?.clone_core(py);
    let efficient = efficient.bind(py).try_borrow()?.clone_core(py);
    let mut config = StageRouterConfig::new(mode, confidence_threshold);
    config.recent_window = recent_turn_window;
    config.handoff_notes = handoff_escalation_note.map(|escalation_note| {
        HandoffNoteConfig::new(
            escalation_note,
            handoff_deescalation_note,
            handoff_only_on_wrong_signal_escalation,
        )
    });
    let mut tier_prompts = TargetPrompts::default();
    if let Some(prompt) = capable_system_prompt {
        tier_prompts = tier_prompts.with(capable.semantic_name.clone(), prompt);
    }
    if let Some(prompt) = efficient_system_prompt {
        tier_prompts = tier_prompts.with(efficient.semantic_name.clone(), prompt);
    }
    config.tier_prompts = tier_prompts;
    if let (Some(judge), Some(base_threshold)) = (judge, classifier_base_threshold) {
        config.llm_fallback = Some(LlmFallback {
            judge_target: judge.bind(py).try_borrow()?.clone_core(py),
            config: TaskClassifierConfig {
                base_threshold,
                min_confidence: classifier_min_confidence,
                capability_elevated_floor: classifier_capability_elevated_floor,
                session_affinity: false,
                message_hash_fallback: false,
                recent_turn_window: classifier_recent_turn_window,
            },
        });
    }
    let algorithm = StageRouter::new(capable, efficient, config)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyAlgorithm::new(Arc::new(algorithm)))
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
    libsy_module.add_class::<PyLlmTarget>()?;
    libsy_module.add_function(wrap_pyfunction!(llm_classifier_algorithm, &libsy_module)?)?;
    libsy_module.add_function(wrap_pyfunction!(noop_algorithm, &libsy_module)?)?;
    libsy_module.add_function(wrap_pyfunction!(passthrough_algorithm, &libsy_module)?)?;
    libsy_module.add_function(wrap_pyfunction!(random_algorithm, &libsy_module)?)?;
    libsy_module.add_function(wrap_pyfunction!(stage_router_algorithm, &libsy_module)?)?;
    libsy_module.add("LibsyError", module.getattr("LibsyError")?)?;
    module.add_submodule(&libsy_module)?;
    Ok(())
}
