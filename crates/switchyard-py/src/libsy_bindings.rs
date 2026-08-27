// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal Python API for running Rust-owned libsy algorithms.

use std::collections::HashMap;
use std::sync::Arc;

use futures::StreamExt;
use http::header::{HeaderName, HeaderValue};
use pyo3::exceptions::{PyBaseException, PyStopAsyncIteration, PyTypeError, PyValueError};
use pyo3::prelude::*;
use serde_json::Value;
use switchyard_libsy::{
    Algorithm, CallModel, ClassifierContractConfig, ClassifierResponseFormat, ClassifyTrigger,
    CustomClassifierConfig, CustomClassifierPolicy, EscalationJudgeConfig, HandoffNoteConfig,
    LibsyError as RustLibsyError, LlmClassifierConfig, LlmFallback, LlmTaskClassifier, Noop,
    PickerMode, Random, RoutingOutcome, StageRouter, StageRouterConfig, Step as RustStep,
    StepStream, TaskClassifierConfig,
};
use switchyard_protocol::{
    LlmClientError, LlmResponse, LlmResponseStream, LlmResponseStreamEvent, Metadata, ModelId,
    Request, Response,
};
use tokio::sync::Mutex;

use crate::errors::{ContextWindowExceededError, py_libsy_error};
use crate::py_serde::{from_python, to_python};

/// The Python API keeps its `session_affinity` flag, which selects the per-session trigger.
fn classify_trigger(session_affinity: bool) -> ClassifyTrigger {
    if session_affinity {
        ClassifyTrigger::NewSession
    } else {
        ClassifyTrigger::EveryRequest
    }
}

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

/// Settings for response-based escalation classification.
#[pyclass(
    name = "EscalationClassifierConfig",
    module = "switchyard.libsy",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
struct PyEscalationClassifierConfig {
    contract: ClassifierContractConfig,
    judge: EscalationJudgeConfig,
    max_output_tokens: u64,
}

#[pymethods]
impl PyEscalationClassifierConfig {
    #[new]
    #[pyo3(signature = (
        *,
        confirmations=2,
        recent_turn_window=28,
        window_message_chars=500,
        max_output_tokens=4096,
        prompt=None,
        response_format_type="json_schema"
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        confirmations: u32,
        recent_turn_window: usize,
        window_message_chars: usize,
        max_output_tokens: u64,
        prompt: Option<String>,
        response_format_type: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            contract: classifier_contract(prompt, response_format_type)?,
            judge: EscalationJudgeConfig {
                confirmations,
                recent_turn_window,
                window_message_chars,
            },
            max_output_tokens,
        })
    }
}

/// Settings for a classifier with a user-supplied verdict schema.
#[pyclass(
    name = "CustomClassifierConfig",
    module = "switchyard.libsy",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
struct PyCustomClassifierConfig {
    inner: CustomClassifierConfig,
}

impl PyCustomClassifierConfig {
    fn clone_core(&self) -> CustomClassifierConfig {
        self.inner.clone()
    }
}

#[pymethods]
impl PyCustomClassifierConfig {
    #[new]
    #[pyo3(signature = (
        prompt,
        response_schema,
        selector,
        *,
        session_affinity=false,
        message_hash_fallback=false,
        recent_turn_window=None,
        max_output_tokens=4096
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        prompt: String,
        response_schema: &Bound<'_, PyAny>,
        selector: String,
        session_affinity: bool,
        message_hash_fallback: bool,
        recent_turn_window: Option<usize>,
        max_output_tokens: u64,
    ) -> PyResult<Self> {
        // Convert the Python schema into serde JSON and pair it with the target-selector policy;
        // conversion failures propagate to Python through `PyResult`.
        let mut inner = CustomClassifierConfig::new(
            prompt,
            from_python::<Value>(response_schema)?,
            CustomClassifierPolicy::target_selector(selector),
        );
        inner.classify_trigger = classify_trigger(session_affinity);
        inner.message_hash_fallback = message_hash_fallback;
        inner.recent_turn_window = recent_turn_window;
        inner.max_output_tokens = max_output_tokens;
        Ok(Self { inner })
    }
}

/// Construction settings for a Python-hosted LLM classifier.
#[pyclass(
    name = "LlmClassifierConfig",
    module = "switchyard.libsy",
    frozen,
    skip_from_py_object
)]
struct PyLlmClassifierConfig {
    inner: LlmClassifierConfig,
}

#[pymethods]
impl PyLlmClassifierConfig {
    /// Configure capability routing between efficient and capable targets.
    #[staticmethod]
    #[pyo3(signature = (judge_target, efficient_target, capable_target, *, config))]
    fn capability(
        py: Python<'_>,
        judge_target: String,
        efficient_target: String,
        capable_target: String,
        config: Py<PyTaskClassifierConfig>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: LlmClassifierConfig::Capability {
                judge_target: ModelId::new(judge_target),
                efficient_target: ModelId::new(efficient_target),
                capable_target: ModelId::new(capable_target),
                config: config.bind(py).try_borrow()?.clone_core(),
            },
        })
    }

    /// Configure response-based escalation between efficient and capable targets.
    #[staticmethod]
    #[pyo3(signature = (judge_target, efficient_target, capable_target, *, config))]
    fn escalation(
        py: Python<'_>,
        judge_target: String,
        efficient_target: String,
        capable_target: String,
        config: Py<PyEscalationClassifierConfig>,
    ) -> PyResult<Self> {
        let config = config.bind(py).try_borrow()?;
        Ok(Self {
            inner: LlmClassifierConfig::Escalation {
                judge_target: ModelId::new(judge_target),
                efficient_target: ModelId::new(efficient_target),
                capable_target: ModelId::new(capable_target),
                contract: config.contract.clone(),
                config: config.judge.clone(),
                max_output_tokens: config.max_output_tokens,
            },
        })
    }

    /// Configure schema-driven routing across named targets.
    #[staticmethod]
    #[pyo3(signature = (judge_target, targets, *, default_target, config))]
    fn custom(
        py: Python<'_>,
        judge_target: String,
        targets: Vec<(String, String)>,
        default_target: String,
        config: Py<PyCustomClassifierConfig>,
    ) -> PyResult<Self> {
        let config = config.bind(py).try_borrow()?.clone_core();
        Ok(Self {
            inner: LlmClassifierConfig::Custom {
                judge_target: ModelId::new(judge_target),
                targets: targets
                    .into_iter()
                    .map(|(name, target)| (name, ModelId::new(target)))
                    .collect(),
                default_target,
                config,
            },
        })
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
        prompt=None,
        response_format_type="json_schema"
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
        response_format_type: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: TaskClassifierConfig {
                base_threshold,
                threshold_step,
                classify_trigger: classify_trigger(session_affinity),
                message_hash_fallback,
                recent_turn_window,
                contract: classifier_contract(prompt, response_format_type)?,
                max_output_tokens,
            },
        })
    }
}

fn classifier_contract(
    prompt: Option<String>,
    response_format_type: &str,
) -> PyResult<ClassifierContractConfig> {
    let mut contract = ClassifierContractConfig::default();
    if let Some(prompt) = prompt {
        contract = contract.with_prompt(prompt);
    }
    let response_format_type = match response_format_type {
        "json_schema" => ClassifierResponseFormat::JsonSchema,
        "json_object" => ClassifierResponseFormat::JsonObject,
        other => {
            return Err(PyValueError::new_err(format!(
                "response_format_type must be 'json_schema' or 'json_object', got {other:?}"
            )));
        }
    };
    Ok(contract.with_response_format_type(response_format_type))
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

/// A normalized aggregate response or a live stream of normalized response events.
#[pyclass(name = "LlmResponse", module = "switchyard.libsy", frozen)]
enum PyLlmResponse {
    /// A fully buffered normalized response dictionary.
    #[pyo3(constructor = (response))]
    Agg { response: Py<PyAny> },
    /// An async iterator of normalized response-event dictionaries.
    #[pyo3(constructor = (stream))]
    Stream { stream: Py<PyAny> },
}

impl PyLlmResponse {
    fn to_core(&self, py: Python<'_>, model: ModelId) -> PyResult<LlmResponse> {
        match self {
            Self::Agg { response } => from_python(response.bind(py)).map(LlmResponse::Agg),
            Self::Stream { stream } => {
                python_response_stream(py, stream.clone_ref(py), model).map(LlmResponse::Stream)
            }
        }
    }
}

fn ffi_error(error: PyErr) -> LlmClientError {
    LlmClientError::Ffi {
        source: Box::new(error),
    }
}

fn python_client_error(py: Python<'_>, error: PyErr, model: &ModelId) -> LlmClientError {
    if error.is_instance_of::<ContextWindowExceededError>(py) {
        LlmClientError::ContextWindowExceeded {
            model: model.clone(),
            message: error.value(py).to_string(),
        }
    } else {
        ffi_error(error)
    }
}

fn python_response_stream(
    py: Python<'_>,
    stream: Py<PyAny>,
    model: ModelId,
) -> PyResult<LlmResponseStream> {
    let iterator = stream.bind(py).call_method0("__aiter__")?.unbind();
    // Rust polls on Tokio, so retain the Python task's event loop and context for every item.
    let locals = pyo3_async_runtimes::tokio::get_current_locals(py)?;
    let stream = futures::stream::unfold(Some((iterator, locals, model)), |state| async move {
        let (iterator, locals, model) = state?;
        let next = Python::attach(|py| {
            pyo3_async_runtimes::into_future_with_locals(
                &locals,
                iterator.bind(py).call_method0("__anext__")?,
            )
        });
        match next {
            Ok(next) => match next.await {
                Ok(item) => {
                    let event =
                        Python::attach(|py| from_python::<LlmResponseStreamEvent>(item.bind(py)))
                            .map_err(ffi_error);
                    Some((event, Some((iterator, locals, model))))
                }
                Err(error) => {
                    if Python::attach(|py| error.is_instance_of::<PyStopAsyncIteration>(py)) {
                        None
                    } else {
                        let error = Python::attach(|py| python_client_error(py, error, &model));
                        Some((Err(error), None))
                    }
                }
            },
            Err(error) => Some((Err(ffi_error(error)), None)),
        }
    });
    Ok(Box::pin(stream))
}

/// One model call yielded by [`PyAlgorithm::run_stream`].
#[pyclass(name = "ModelCall", module = "switchyard.libsy")]
struct PyModelCall {
    inner: Option<CallModel>,
    algorithm: String,
    request: Py<PyAny>,
    models: Vec<String>,
}

impl PyModelCall {
    fn new(py: Python<'_>, call: CallModel) -> PyResult<Self> {
        let request = to_python(py, &call.request.llm_request)?;
        Ok(Self {
            algorithm: call.algorithm.clone(),
            models: call.models.iter().map(ToString::to_string).collect(),
            inner: Some(call),
            request,
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

    /// Candidate models in the order the host should try them.
    #[getter]
    fn models(&self) -> Vec<String> {
        self.models.clone()
    }

    /// Fulfill this call with a normalized aggregate or streamed response.
    fn respond(&mut self, py: Python<'_>, response: PyRef<'_, PyLlmResponse>) -> PyResult<()> {
        let model = self
            .inner
            .as_ref()
            .ok_or_else(|| py_libsy_error("model call has already been completed"))?
            .request
            .llm_request
            .model
            .as_ref()
            .map(ModelId::new)
            .ok_or_else(|| py_libsy_error("model call request is missing its selected model"))?;
        let llm_response = response.to_core(py, model)?;
        let call = self.take()?;
        let metadata = call.request.metadata.clone();
        call.respond(Ok(Response {
            llm_response,
            metadata,
        }))
        .map_err(py_libsy_error)
    }

    /// Fulfill this call with a Python client failure.
    fn fail(&mut self, error: &Bound<'_, PyAny>) -> PyResult<()> {
        if !error.is_instance_of::<PyBaseException>() {
            return Err(PyTypeError::new_err("error must derive from BaseException"));
        }
        let target = self
            .inner
            .as_ref()
            .ok_or_else(|| py_libsy_error("model call has already been completed"))?
            .request
            .llm_request
            .model
            .as_ref()
            .map(ModelId::new)
            .ok_or_else(|| py_libsy_error("model call request is missing its selected model"))?;
        let call = self.take()?;
        let source = python_client_error(error.py(), PyErr::from_value(error.clone()), &target);
        call.respond(Err(RustLibsyError::client_call(target, source)))
            .map_err(py_libsy_error)
    }
}

/// The terminal routing selection, rewritten request, and optional existing response.
#[pyclass(name = "RoutingOutcome", module = "switchyard.libsy", frozen)]
struct PyRoutingOutcome {
    selected_model_id: String,
    fallback_models: Vec<String>,
    request: Py<PyAny>,
    response: Option<Py<PyAny>>,
}

#[pymethods]
impl PyRoutingOutcome {
    /// The model selected by the algorithm and tried first by the host.
    #[getter]
    fn selected_model_id(&self) -> &str {
        &self.selected_model_id
    }

    /// Additional models the host may try in order after an eligible failure.
    #[getter]
    fn fallback_models(&self) -> Vec<String> {
        self.fallback_models.clone()
    }

    /// The normalized request after routing-time rewrites.
    #[getter]
    fn request(&self, py: Python<'_>) -> Py<PyAny> {
        self.request.clone_ref(py)
    }

    /// An aggregate or streamed answer produced while routing, when one already exists.
    #[getter]
    fn response(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.response
            .as_ref()
            .map(|response| response.clone_ref(py))
    }
}

/// Async Python iterator over one normalized Rust response stream.
#[pyclass(name = "_LlmResponseStream", module = "switchyard.libsy", frozen)]
struct PyLlmResponseStream {
    inner: Arc<Mutex<LlmResponseStream>>,
}

#[pymethods]
impl PyLlmResponseStream {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let stream = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            match stream.lock().await.next().await {
                Some(Ok(event)) => Python::attach(|py| to_python(py, &event)),
                Some(Err(error)) => Err(py_libsy_error(error)),
                None => Err(PyStopAsyncIteration::new_err(())),
            }
        })
    }
}

fn response_to_python(py: Python<'_>, response: LlmResponse) -> PyResult<Py<PyAny>> {
    let response = match response {
        LlmResponse::Agg(response) => PyLlmResponse::Agg {
            response: to_python(py, &response)?,
        },
        LlmResponse::Stream(stream) => PyLlmResponse::Stream {
            stream: Py::new(
                py,
                PyLlmResponseStream {
                    inner: Arc::new(Mutex::new(stream)),
                },
            )?
            .into_any(),
        },
    };
    response
        .into_pyobject(py)
        .map(|response| response.unbind().into_any())
}

/// One item yielded by a Python algorithm stream.
#[pyclass(name = "Step", module = "switchyard.libsy", frozen)]
enum PyStep {
    /// The host must serve the model call before the algorithm can continue.
    CallModel { call: Py<PyModelCall> },
    /// The terminal routing outcome.
    Done { outcome: Py<PyRoutingOutcome> },
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
                Some(Ok(step)) => step_to_python(step),
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
    /// Run the algorithm as routing-time model calls followed by one terminal outcome.
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

fn step_to_python(step: RustStep) -> PyResult<PyStep> {
    match step {
        RustStep::CallModel(call) => Python::attach(|py| {
            Ok(PyStep::CallModel {
                call: Py::new(py, PyModelCall::new(py, *call)?)?,
            })
        }),
        RustStep::Done(outcome) => {
            let RoutingOutcome {
                selected_model_id,
                fallback_models,
                request,
                response,
            } = *outcome;
            Python::attach(|py| {
                Ok(PyStep::Done {
                    outcome: Py::new(
                        py,
                        PyRoutingOutcome {
                            selected_model_id: selected_model_id.to_string(),
                            fallback_models: fallback_models
                                .iter()
                                .map(ToString::to_string)
                                .collect(),
                            request: to_python(py, &request.llm_request)?,
                            response: response
                                .map(|response| response_to_python(py, response.llm_response))
                                .transpose()?,
                        },
                    )?,
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

/// Construct LLM classifier routing from a mode config.
#[pyfunction(name = "llm_classifier")]
fn llm_classifier_algorithm(
    py: Python<'_>,
    config: Py<PyLlmClassifierConfig>,
) -> PyResult<PyAlgorithm> {
    build_llm_classifier(config.bind(py).try_borrow()?.inner.clone())
}

/// Construct capability classifier routing.
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
    build_llm_classifier(LlmClassifierConfig::Capability {
        judge_target: ModelId::new(judge_target),
        efficient_target: ModelId::new(efficient_target),
        capable_target: ModelId::new(capable_target),
        config: config.bind(py).try_borrow()?.clone_core(),
    })
}

fn build_llm_classifier(config: LlmClassifierConfig) -> PyResult<PyAlgorithm> {
    let algorithm =
        LlmTaskClassifier::new(config).map_err(|error| PyValueError::new_err(error.to_string()))?;
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
    libsy_module.add_class::<PyCustomClassifierConfig>()?;
    libsy_module.add_class::<PyEscalationClassifierConfig>()?;
    libsy_module.add_class::<PyLlmClassifierConfig>()?;
    libsy_module.add_class::<PyLlmFallback>()?;
    libsy_module.add_class::<PyLlmResponse>()?;
    libsy_module.add_class::<PyLlmResponseStream>()?;
    libsy_module.add_class::<PyModelCall>()?;
    libsy_module.add_class::<PyRunStream>()?;
    libsy_module.add_class::<PyRoutingOutcome>()?;
    libsy_module.add_class::<PyStep>()?;
    libsy_module.add_class::<PyTaskClassifierConfig>()?;
    libsy_module.add_function(wrap_pyfunction!(noop_algorithm, &libsy_module)?)?;
    libsy_module.add_function(wrap_pyfunction!(random_algorithm, &libsy_module)?)?;
    libsy_module.add_function(wrap_pyfunction!(llm_classifier_algorithm, &libsy_module)?)?;
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
