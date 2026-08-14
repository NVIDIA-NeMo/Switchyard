// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal Python API for running Rust-owned libsy algorithms.

use std::collections::HashMap;
use std::sync::Arc;

use futures::StreamExt;
use http::header::{HeaderName, HeaderValue};
use pyo3::exceptions::{PyBaseException, PyStopAsyncIteration, PyTypeError, PyValueError};
use pyo3::prelude::*;
use switchyard_libsy::{
    Algorithm, CallModel, ClassifierContractConfig, HandoffNoteConfig,
    LibsyError as RustLibsyError, LlmClassifierConfig, LlmFallback, LlmTaskClassifier, Noop,
    PickerMode, Random, StageRouter, StageRouterConfig, Step as RustStep, StepStream,
    TaskClassifierConfig,
};
use switchyard_protocol::{
    AggLlmResponse, Decision, LlmClientError, LlmResponse, Metadata, ModelId, Request, Response,
};
use tokio::sync::Mutex;

use crate::errors::{ContextWindowExceededError, py_libsy_error};
use crate::py_serde::{from_python, to_python};

/// Convert Python-owned headers into the request metadata expected by libsy.
fn header_map_from_python(headers: &HashMap<String, String>) -> PyResult<http::HeaderMap> {
    let mut result = http::HeaderMap::new();
    for (name, value) in headers {
        let name = HeaderName::from_bytes(name.as_bytes())
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let value = HeaderValue::from_str(value)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        result
            .try_append(name, value)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
    }
    Ok(result)
}

/// Classifier settings shared by standalone and stage-router classifiers.
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
        threshold_step=0.0,
        session_affinity=false,
        message_hash_fallback=false,
        recent_turn_window=None,
        max_output_tokens=4096,
        prompt=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        base_threshold: f64,
        threshold_step: f64,
        session_affinity: bool,
        message_hash_fallback: bool,
        recent_turn_window: Option<usize>,
        max_output_tokens: u64,
        prompt: Option<String>,
    ) -> Self {
        let mut contract = ClassifierContractConfig::default();
        if let Some(prompt) = prompt {
            contract = contract.with_prompt(prompt);
        }
        Self {
            inner: TaskClassifierConfig {
                base_threshold,
                threshold_step,
                session_affinity,
                message_hash_fallback,
                recent_turn_window,
                contract,
                max_output_tokens,
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
    judge_target: String,
    config: Py<PyTaskClassifierConfig>,
}

impl PyLlmFallback {
    fn clone_core(&self, py: Python<'_>) -> PyResult<LlmFallback> {
        Ok(LlmFallback {
            judge_target: ModelId::new(self.judge_target.clone()),
            config: self.config.bind(py).try_borrow()?.clone_core(),
        })
    }
}

#[pymethods]
impl PyLlmFallback {
    #[new]
    #[pyo3(signature = (judge_target, *, config))]
    fn new(judge_target: String, config: Py<PyTaskClassifierConfig>) -> Self {
        Self {
            judge_target,
            config,
        }
    }
}

/// A routing choice produced by an algorithm.
#[pyclass(name = "Decision", module = "switchyard.libsy", frozen)]
struct PyDecision {
    inner: Decision,
}

impl From<Decision> for PyDecision {
    fn from(inner: Decision) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyDecision {
    /// The semantic model id selected for the call.
    #[getter]
    fn selected_model_id(&self) -> &str {
        self.inner.selected_model_id().as_str()
    }

    /// Whether this call produces the answer rather than a routing verdict.
    #[getter]
    fn is_answer_call(&self) -> bool {
        self.inner.is_answer_call()
    }

    fn __repr__(&self) -> String {
        format!(
            "Decision(selected_model_id={:?}, is_answer_call={})",
            self.inner.selected_model_id(),
            self.inner.is_answer_call()
        )
    }
}

/// One model call yielded by [`PyAlgorithm::run_stream`].
#[pyclass(name = "ModelCall", module = "switchyard.libsy")]
struct PyModelCall {
    inner: Option<CallModel>,
    algorithm: String,
    request: Py<PyAny>,
    decision: Py<PyDecision>,
}

impl PyModelCall {
    fn new(py: Python<'_>, call: CallModel) -> PyResult<Self> {
        let request = to_python(py, &call.request.llm_request)?;
        let decision = Py::new(py, PyDecision::from(call.decision.clone()))?;
        Ok(Self {
            algorithm: call.algorithm.clone(),
            inner: Some(call),
            request,
            decision,
        })
    }

    fn take(&mut self) -> PyResult<CallModel> {
        self.inner
            .take()
            .ok_or_else(|| py_libsy_error("model call has already been completed"))
    }
}

#[pymethods]
impl PyModelCall {
    /// The algorithm that produced this call.
    #[getter]
    fn algorithm(&self) -> &str {
        &self.algorithm
    }

    /// The normalized LLM request to serve as a Python dictionary.
    #[getter]
    fn request(&self, py: Python<'_>) -> Py<PyAny> {
        self.request.clone_ref(py)
    }

    /// The routing decision behind this call.
    #[getter]
    fn decision(&self, py: Python<'_>) -> Py<PyDecision> {
        self.decision.clone_ref(py)
    }

    /// Consume the answer call without serving it and return its rewritten request and decision.
    #[pyo3(name = "into_parts")]
    fn take_parts(&mut self, py: Python<'_>) -> PyResult<(Py<PyAny>, Py<PyDecision>)> {
        let (request, decision) = self.take()?.into_parts();
        Ok((
            to_python(py, &request.llm_request)?,
            Py::new(py, PyDecision::from(decision))?,
        ))
    }

    /// Fulfill this call with an aggregate normalized response dictionary.
    fn respond(&mut self, response: &Bound<'_, PyAny>) -> PyResult<()> {
        let aggregate = from_python::<AggLlmResponse>(response)?;
        let call = self.take()?;
        let metadata = call.request.metadata.clone();
        call.respond(Ok(Response {
            llm_response: LlmResponse::Agg(aggregate),
            metadata,
        }))
        .map_err(py_libsy_error)
    }

    /// Fulfill this call with a Python client failure.
    fn fail(&mut self, error: &Bound<'_, PyAny>) -> PyResult<()> {
        if !error.is_instance_of::<PyBaseException>() {
            return Err(PyTypeError::new_err("error must derive from BaseException"));
        }
        let call = self.take()?;
        let target = call.decision.selected_model_id().clone();
        let source = if error.is_instance_of::<ContextWindowExceededError>() {
            LlmClientError::ContextWindowExceeded {
                model: target.clone(),
                message: error.str()?.to_string_lossy().into_owned(),
            }
        } else {
            LlmClientError::Ffi {
                source: Box::new(PyErr::from_value(error.clone())),
            }
        };
        call.respond(Err(RustLibsyError::client_call(target, source)))
            .map_err(py_libsy_error)
    }
}

/// One item yielded by a Python algorithm stream.
#[pyclass(name = "Step", module = "switchyard.libsy", frozen)]
enum PyStep {
    /// The host must serve the model call before the algorithm can continue.
    CallModel { call: Py<PyModelCall> },
    /// A routing decision emitted by the algorithm.
    Decision { decision: Py<PyDecision> },
    /// The terminal aggregate response.
    Done { response: Py<PyAny> },
}

/// Async Python iterator over one Rust algorithm run.
#[pyclass(name = "_RunStream", module = "switchyard.libsy", frozen)]
struct PyRunStream {
    inner: Arc<Mutex<StepStream>>,
}

#[pymethods]
impl PyRunStream {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let stream = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let step = stream.lock().await.next().await;
            match step {
                Some(Ok(step)) => step_to_python(step).await,
                Some(Err(error)) => Err(py_libsy_error(error)),
                None => Err(PyStopAsyncIteration::new_err(())),
            }
        })
    }
}

/// Opaque handle shared by every Rust-owned algorithm exposed to Python.
#[pyclass(name = "Algorithm", module = "switchyard.libsy", frozen)]
struct PyAlgorithm {
    inner: Arc<dyn Algorithm>,
}

#[pymethods]
impl PyAlgorithm {
    /// Run the algorithm as a stream of model calls, decisions, and one terminal response.
    ///
    /// `headers`, when given, is normalized into the request's correlation
    /// [`Metadata`] exactly as an HTTP host would (`Metadata::from_headers`),
    /// so metadata-driven algorithms see the same signals in Python as when
    /// served over HTTP.
    #[pyo3(signature = (request, headers=None))]
    fn run_stream(
        &self,
        request: &Bound<'_, PyAny>,
        headers: Option<std::collections::HashMap<String, String>>,
    ) -> PyResult<PyRunStream> {
        let headers = headers.as_ref().map(header_map_from_python).transpose()?;
        let request = Request {
            llm_request: from_python(request)?,
            raw_request: None,
            metadata: headers.map(|headers| Metadata::from_headers(&headers)),
        };
        let stream = {
            let _guard = pyo3_async_runtimes::tokio::get_runtime().enter();
            Arc::clone(&self.inner).run_stream(request)
        };
        Ok(PyRunStream {
            inner: Arc::new(Mutex::new(stream)),
        })
    }

    fn __repr__(&self) -> &'static str {
        "Algorithm()"
    }
}

async fn step_to_python(step: RustStep) -> PyResult<PyStep> {
    match step {
        RustStep::CallModel(call) => Python::attach(|py| {
            Ok(PyStep::CallModel {
                call: Py::new(py, PyModelCall::new(py, *call)?)?,
            })
        }),
        RustStep::Decision(decision) => Python::attach(|py| {
            Ok(PyStep::Decision {
                decision: Py::new(py, PyDecision::from(decision))?,
            })
        }),
        RustStep::Done(response) => {
            let response = response
                .llm_response
                .into_agg()
                .await
                .map_err(py_libsy_error)?;
            Python::attach(|py| {
                Ok(PyStep::Done {
                    response: to_python(py, &response)?,
                })
            })
        }
    }
}

/// Construct the no-op reference algorithm.
#[pyfunction(name = "noop")]
fn noop_algorithm() -> PyAlgorithm {
    PyAlgorithm {
        inner: Arc::new(Noop {}),
    }
}

/// Construct random routing over targets with optional relative weights and seed.
#[pyfunction(name = "random")]
#[pyo3(signature = (targets, *, weights=None, seed=None))]
fn random_algorithm(
    targets: Vec<String>,
    weights: Option<Vec<f64>>,
    seed: Option<u64>,
) -> PyResult<PyAlgorithm> {
    let model_ids = targets.into_iter().map(ModelId::new).collect();
    let algorithm = Random::new(model_ids, weights, seed).map_err(|error| match error {
        RustLibsyError::NoTargets => PyValueError::new_err("random requires at least one target"),
        other => PyValueError::new_err(other.to_string()),
    })?;
    Ok(PyAlgorithm {
        inner: Arc::new(algorithm),
    })
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
    judge_target: String,
    efficient_target: String,
    capable_target: String,
    config: Py<PyTaskClassifierConfig>,
) -> PyResult<PyAlgorithm> {
    let algorithm = LlmTaskClassifier::new(LlmClassifierConfig::Capability {
        judge_target: ModelId::new(judge_target),
        efficient_target: ModelId::new(efficient_target),
        capable_target: ModelId::new(capable_target),
        config: config.bind(py).try_borrow()?.clone_core(),
    })
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyAlgorithm {
        inner: Arc::new(algorithm),
    })
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
    capable_target: String,
    efficient_target: String,
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
    let capable = ModelId::new(capable_target);
    let efficient = ModelId::new(efficient_target);
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
        config.tier_prompts = config.tier_prompts.with(capable.clone(), prompt);
    }
    if let Some(prompt) = efficient_system_prompt {
        config.tier_prompts = config.tier_prompts.with(efficient.clone(), prompt);
    }
    config.llm_fallback = classifier
        .map(|classifier| classifier.bind(py).try_borrow()?.clone_core(py))
        .transpose()?;

    let algorithm = StageRouter::new(capable, efficient, config)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyAlgorithm {
        inner: Arc::new(algorithm),
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let libsy_module = PyModule::new(module.py(), "libsy")?;
    libsy_module.add_class::<PyAlgorithm>()?;
    libsy_module.add_class::<PyDecision>()?;
    libsy_module.add_class::<PyLlmFallback>()?;
    libsy_module.add_class::<PyModelCall>()?;
    libsy_module.add_class::<PyRunStream>()?;
    libsy_module.add_class::<PyStep>()?;
    libsy_module.add_class::<PyTaskClassifierConfig>()?;
    libsy_module.add_function(wrap_pyfunction!(noop_algorithm, &libsy_module)?)?;
    libsy_module.add_function(wrap_pyfunction!(random_algorithm, &libsy_module)?)?;
    libsy_module.add_function(wrap_pyfunction!(
        llm_task_classifier_algorithm,
        &libsy_module
    )?)?;
    libsy_module.add_function(wrap_pyfunction!(stage_router_algorithm, &libsy_module)?)?;
    libsy_module.add(
        "ContextWindowExceededError",
        module.getattr("ContextWindowExceededError")?,
    )?;
    libsy_module.add("LibsyError", module.getattr("LibsyError")?)?;
    module.add_submodule(&libsy_module)?;
    Ok(())
}
