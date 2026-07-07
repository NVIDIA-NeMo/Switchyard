// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for component configuration values.

use pyo3::class::basic::CompareOp;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBool;
use serde::Serialize;
use switchyard_components::{
    IntakeQueueFullPolicy, IntakeSinkConfig, RandomRoutingProcessorConfig,
};
use switchyard_core::{BackendFormat, EndpointConfig, LlmTarget, LlmTargetId, ModelId};

use crate::errors::py_core_error;
use crate::py_serde::{value_from_python, value_to_python};

#[pyclass(name = "BackendFormat", frozen, skip_from_py_object)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PyBackendFormat {
    inner: BackendFormat,
}

impl PyBackendFormat {
    const fn new_inner(inner: BackendFormat) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyBackendFormat {
    #[new]
    #[pyo3(signature = (value="auto"))]
    fn py_new(value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: backend_format_from_str(value)?,
        })
    }

    #[classattr]
    const AUTO: Self = Self::new_inner(BackendFormat::Auto);

    #[classattr]
    const OPENAI: Self = Self::new_inner(BackendFormat::OpenAi);

    #[classattr]
    const RESPONSES: Self = Self::new_inner(BackendFormat::Responses);

    #[classattr]
    const ANTHROPIC: Self = Self::new_inner(BackendFormat::Anthropic);

    #[classattr]
    const GEMINI: Self = Self::new_inner(BackendFormat::Gemini);

    #[getter]
    fn value(&self) -> &'static str {
        backend_format_name(self.inner)
    }

    fn __repr__(&self) -> String {
        format!("BackendFormat.{}", backend_format_variant_name(self.inner))
    }

    fn __str__(&self) -> &'static str {
        backend_format_name(self.inner)
    }

    fn __hash__(&self) -> isize {
        match self.inner {
            BackendFormat::Auto => 1,
            BackendFormat::OpenAi => 2,
            BackendFormat::Responses => 3,
            BackendFormat::Anthropic => 4,
            BackendFormat::Gemini => 5,
        }
    }

    fn __richcmp__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<Py<PyAny>> {
        match op {
            CompareOp::Eq | CompareOp::Ne => {
                let equals = match backend_format_from_python(Some(other)) {
                    Ok(other) => self.inner == other,
                    Err(_) => false,
                };
                let result = if matches!(op, CompareOp::Eq) {
                    equals
                } else {
                    !equals
                };
                Ok(PyBool::new(py, result).to_owned().unbind().into_any())
            }
            _ => Ok(py.NotImplemented()),
        }
    }
}

#[pyclass(name = "EndpointConfig", frozen, skip_from_py_object)]
#[derive(Clone, Debug)]
pub(crate) struct PyEndpointConfig {
    inner: EndpointConfig,
}

impl PyEndpointConfig {
    pub(crate) fn from_core(inner: EndpointConfig) -> Self {
        Self { inner }
    }

    pub(crate) fn clone_core(&self) -> EndpointConfig {
        self.inner.clone()
    }
}

#[pymethods]
impl PyEndpointConfig {
    #[new]
    #[pyo3(signature = (base_url=None, api_key=None, timeout_secs=None))]
    fn py_new(
        base_url: Option<String>,
        api_key: Option<String>,
        timeout_secs: Option<f64>,
    ) -> Self {
        Self {
            inner: EndpointConfig {
                base_url,
                api_key,
                timeout_secs,
            },
        }
    }

    #[getter]
    fn base_url(&self) -> Option<String> {
        self.inner.base_url.clone()
    }

    #[getter]
    fn api_key(&self) -> Option<String> {
        self.inner.api_key.clone()
    }

    #[getter]
    fn timeout_secs(&self) -> Option<f64> {
        self.inner.timeout_secs
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_python(py, &self.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "EndpointConfig(base_url={:?}, api_key={}, timeout_secs={:?})",
            self.inner.base_url,
            if self.inner.api_key.is_some() {
                "'<redacted>'"
            } else {
                "None"
            },
            self.inner.timeout_secs,
        )
    }
}

#[pyclass(name = "LlmTarget", frozen, skip_from_py_object)]
#[derive(Clone, Debug)]
pub(crate) struct PyLlmTarget {
    inner: LlmTarget,
}

impl PyLlmTarget {
    pub(crate) fn from_core(inner: LlmTarget) -> Self {
        Self { inner }
    }

    pub(crate) fn clone_core(&self) -> LlmTarget {
        self.inner.clone()
    }
}

#[pymethods]
impl PyLlmTarget {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        id=None,
        model=None,
        format=None,
        backend_format=None,
        endpoint=None,
        base_url=None,
        api_key=None,
        timeout_secs=None,
        timeout=None,
        extra_body=None,
        extra_headers=None,
    ))]
    fn py_new(
        id: Option<String>,
        model: Option<String>,
        format: Option<&Bound<'_, PyAny>>,
        backend_format: Option<&Bound<'_, PyAny>>,
        endpoint: Option<&Bound<'_, PyAny>>,
        base_url: Option<String>,
        api_key: Option<String>,
        timeout_secs: Option<f64>,
        timeout: Option<f64>,
        extra_body: Option<&Bound<'_, PyAny>>,
        extra_headers: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mut endpoint = endpoint_config_from_python(endpoint)?;
        if base_url.is_some() {
            endpoint.base_url = base_url;
        }
        if api_key.is_some() {
            endpoint.api_key = api_key;
        }
        if timeout_secs.is_some() {
            endpoint.timeout_secs = timeout_secs;
        }
        if timeout.is_some() {
            endpoint.timeout_secs = timeout;
        }

        let (id, model) = match (id, model) {
            (Some(id), Some(model)) => (id, model),
            (None, Some(model)) => ("default".to_string(), model),
            (Some(_), None) => {
                return Err(PyValueError::new_err(
                    "LlmTarget requires a model string when id is provided",
                ));
            }
            (None, None) => return Err(PyValueError::new_err("LlmTarget requires a model string")),
        };

        let extra_body = match extra_body {
            None => None,
            Some(value) if value.is_none() => None,
            Some(value) => Some(value_from_python(value).map_err(|error| {
                PyValueError::new_err(format!(
                    "LlmTarget.extra_body must be JSON-serialisable: {error}"
                ))
            })?),
        };

        let extra_headers = match extra_headers {
            None => std::collections::BTreeMap::new(),
            Some(value) if value.is_none() => std::collections::BTreeMap::new(),
            Some(value) => {
                let raw = value_from_python(value).map_err(|error| {
                    PyValueError::new_err(format!(
                        "LlmTarget.extra_headers must be a JSON-serialisable mapping: {error}"
                    ))
                })?;
                let serde_json::Value::Object(map) = raw else {
                    return Err(PyValueError::new_err(
                        "LlmTarget.extra_headers must be a dict of str -> str",
                    ));
                };
                let mut out = std::collections::BTreeMap::new();
                for (k, v) in map {
                    let serde_json::Value::String(s) = v else {
                        return Err(PyValueError::new_err(format!(
                            "LlmTarget.extra_headers[{:?}] must be a string, got {}",
                            k, v
                        )));
                    };
                    out.insert(k, s);
                }
                out
            }
        };

        Ok(Self {
            inner: LlmTarget {
                id: LlmTargetId::new(id).map_err(|error| {
                    PyValueError::new_err(format!("invalid LLM target id: {error}"))
                })?,
                model: ModelId::new(model)
                    .map_err(|error| PyValueError::new_err(format!("invalid model id: {error}")))?,
                format: backend_format_from_python(format.or(backend_format))?,
                endpoint,
                extra_body,
                extra_headers,
            },
        })
    }

    #[getter]
    fn id(&self) -> String {
        self.inner.id.as_str().to_string()
    }

    #[getter]
    fn model(&self) -> String {
        self.inner.model.as_str().to_string()
    }

    #[getter]
    fn format(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        backend_format_object(py, self.inner.format)
    }

    #[getter]
    fn backend_format(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        backend_format_object(py, self.inner.format)
    }

    #[getter]
    fn endpoint(&self) -> PyEndpointConfig {
        PyEndpointConfig {
            inner: self.inner.endpoint.clone(),
        }
    }

    #[getter]
    fn base_url(&self) -> Option<String> {
        self.inner.endpoint.base_url.clone()
    }

    #[getter]
    fn api_key(&self) -> Option<String> {
        self.inner.endpoint.api_key.clone()
    }

    #[getter]
    fn timeout(&self) -> Option<f64> {
        self.inner.endpoint.timeout_secs
    }

    #[getter]
    fn extra_body(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner.extra_body {
            None => Ok(py.None()),
            Some(value) => value_to_python(py, value),
        }
    }

    #[getter]
    fn extra_headers(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let map: serde_json::Map<String, serde_json::Value> = self
            .inner
            .extra_headers
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
            .collect();
        value_to_python(py, &serde_json::Value::Object(map))
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_python(py, &self.inner)
    }

    fn model_dump(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.to_dict(py)
    }

    fn __richcmp__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<Py<PyAny>> {
        match op {
            CompareOp::Eq | CompareOp::Ne => {
                let equals = if let Ok(other) = other.extract::<PyRef<'_, PyLlmTarget>>() {
                    self.inner == other.inner
                } else if let Ok(other) = value_from_python(other).and_then(|value| {
                    serde_json::from_value::<LlmTarget>(value)
                        .map_err(|error| PyValueError::new_err(error.to_string()))
                }) {
                    self.inner == other
                } else {
                    false
                };
                let result = if matches!(op, CompareOp::Eq) {
                    equals
                } else {
                    !equals
                };
                Ok(PyBool::new(py, result).to_owned().unbind().into_any())
            }
            _ => Ok(py.NotImplemented()),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "LlmTarget(id={:?}, model={:?}, format='{}')",
            self.inner.id.as_str(),
            self.inner.model.as_str(),
            backend_format_name(self.inner.format),
        )
    }
}

#[pyclass(name = "RandomRoutingProcessorConfig", frozen, skip_from_py_object)]
#[derive(Clone, Debug)]
pub(crate) struct PyRandomRoutingProcessorConfig {
    inner: RandomRoutingProcessorConfig,
}

#[pymethods]
impl PyRandomRoutingProcessorConfig {
    #[new]
    #[pyo3(signature = (strong, weak, strong_probability=0.5, rng_seed=None))]
    fn py_new(
        strong: PyRef<'_, PyLlmTarget>,
        weak: PyRef<'_, PyLlmTarget>,
        strong_probability: f64,
        rng_seed: Option<u64>,
    ) -> PyResult<Self> {
        let config = RandomRoutingProcessorConfig::new(strong.clone_core(), weak.clone_core())
            .with_strong_probability(strong_probability)
            .map_err(py_core_error)?
            .with_rng_seed(rng_seed);
        Ok(Self { inner: config })
    }

    #[getter]
    fn strong(&self) -> PyLlmTarget {
        PyLlmTarget::from_core(self.inner.strong.clone())
    }

    #[getter]
    fn weak(&self) -> PyLlmTarget {
        PyLlmTarget::from_core(self.inner.weak.clone())
    }

    #[getter]
    fn strong_probability(&self) -> f64 {
        self.inner.strong_probability
    }

    #[getter]
    fn rng_seed(&self) -> Option<u64> {
        self.inner.rng_seed
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_python(py, &self.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "RandomRoutingProcessorConfig(strong={}, weak={}, strong_probability={}, rng_seed={:?})",
            self.inner.strong.model,
            self.inner.weak.model,
            self.inner.strong_probability,
            self.inner.rng_seed,
        )
    }
}

#[pyclass(name = "IntakeQueueFullPolicy", frozen, eq, skip_from_py_object)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PyIntakeQueueFullPolicy {
    inner: IntakeQueueFullPolicy,
}

impl PyIntakeQueueFullPolicy {
    const fn new_inner(inner: IntakeQueueFullPolicy) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyIntakeQueueFullPolicy {
    #[new]
    #[pyo3(signature = (value="drop"))]
    fn py_new(value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: intake_queue_policy_from_str(value)?,
        })
    }

    #[classattr]
    const DROP: Self = Self::new_inner(IntakeQueueFullPolicy::Drop);

    #[classattr]
    const BLOCK: Self = Self::new_inner(IntakeQueueFullPolicy::Block);

    #[getter]
    fn value(&self) -> &'static str {
        intake_queue_policy_name(self.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "IntakeQueueFullPolicy.{}",
            intake_queue_policy_variant_name(self.inner)
        )
    }

    fn __str__(&self) -> &'static str {
        intake_queue_policy_name(self.inner)
    }

    fn __hash__(&self) -> isize {
        match self.inner {
            IntakeQueueFullPolicy::Drop => 1,
            IntakeQueueFullPolicy::Block => 2,
        }
    }
}

#[pyclass(name = "IntakeSinkConfig", frozen, skip_from_py_object)]
#[derive(Clone, Debug)]
pub(crate) struct PyIntakeSinkConfig {
    inner: IntakeSinkConfig,
}

impl PyIntakeSinkConfig {
    pub(crate) fn from_core(inner: IntakeSinkConfig) -> Self {
        Self { inner }
    }

    pub(crate) fn clone_core(&self) -> IntakeSinkConfig {
        self.inner.clone()
    }
}

#[pymethods]
impl PyIntakeSinkConfig {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        intake_base_url=None,
        workspace=None,
        user_id=None,
        api_key=None,
        nvdataflow_project=None,
        max_queue_size=None,
        request_timeout_s=None,
        max_retries=None,
        on_queue_full=None,
        capture_content=None
    ))]
    fn py_new(
        intake_base_url: Option<String>,
        workspace: Option<String>,
        user_id: Option<String>,
        api_key: Option<String>,
        nvdataflow_project: Option<String>,
        max_queue_size: Option<usize>,
        request_timeout_s: Option<f64>,
        max_retries: Option<u32>,
        on_queue_full: Option<&Bound<'_, PyAny>>,
        capture_content: Option<bool>,
    ) -> PyResult<Self> {
        let mut config = IntakeSinkConfig::default();
        if intake_base_url.is_some() {
            config.intake_base_url = intake_base_url;
        }
        if workspace.is_some() {
            config.workspace = workspace;
        }
        if let Some(user_id) = user_id {
            config.user_id = user_id;
        }
        if api_key.is_some() {
            config.api_key = api_key;
        }
        if nvdataflow_project.is_some() {
            config.nvdataflow_project = nvdataflow_project;
        }
        if let Some(max_queue_size) = max_queue_size {
            config.max_queue_size = max_queue_size;
        }
        if let Some(request_timeout_s) = request_timeout_s {
            config.request_timeout_s = request_timeout_s;
        }
        if let Some(max_retries) = max_retries {
            config.max_retries = max_retries;
        }
        if on_queue_full.is_some() {
            config.on_queue_full = intake_queue_policy_from_python(on_queue_full)?;
        }
        config.capture_content = capture_content.unwrap_or_else(intake_capture_content_from_env);
        Ok(Self { inner: config })
    }

    #[getter]
    fn intake_base_url(&self) -> Option<String> {
        self.inner.intake_base_url.clone()
    }

    #[getter]
    fn workspace(&self) -> Option<String> {
        self.inner.workspace.clone()
    }

    #[getter]
    fn user_id(&self) -> String {
        self.inner.user_id.clone()
    }

    #[getter]
    fn nvdataflow_project(&self) -> Option<String> {
        self.inner.nvdataflow_project.clone()
    }

    #[getter]
    fn api_key(&self) -> Option<String> {
        self.inner.api_key.clone()
    }

    #[getter]
    fn max_queue_size(&self) -> usize {
        self.inner.max_queue_size
    }

    #[getter]
    fn request_timeout_s(&self) -> f64 {
        self.inner.request_timeout_s
    }

    #[getter]
    fn max_retries(&self) -> u32 {
        self.inner.max_retries
    }

    #[getter]
    fn on_queue_full(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        intake_queue_policy_object(py, self.inner.on_queue_full)
    }

    #[getter]
    fn capture_content(&self) -> bool {
        self.inner.capture_content
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_python(py, &self.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "IntakeSinkConfig(intake_base_url={:?}, workspace={:?}, user_id={:?}, api_key={}, nvdataflow_project={:?}, max_queue_size={}, request_timeout_s={}, max_retries={}, on_queue_full='{}', capture_content={})",
            self.inner.intake_base_url,
            self.inner.workspace,
            self.inner.user_id,
            if self.inner.api_key.is_some() {
                "'<redacted>'"
            } else {
                "None"
            },
            self.inner.nvdataflow_project,
            self.inner.max_queue_size,
            self.inner.request_timeout_s,
            self.inner.max_retries,
            intake_queue_policy_name(self.inner.on_queue_full),
            self.inner.capture_content,
        )
    }
}

pub(crate) fn backend_format_from_python(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<BackendFormat> {
    let Some(value) = value.filter(|value| !value.is_none()) else {
        return Ok(BackendFormat::Auto);
    };
    if let Ok(format) = value.extract::<PyRef<'_, PyBackendFormat>>() {
        return Ok(format.inner);
    }
    let raw = if let Ok(value_attr) = value.getattr("value") {
        value_attr.extract::<String>()?
    } else {
        value.extract::<String>()?
    };
    backend_format_from_str(&raw)
}

fn backend_format_from_str(value: &str) -> PyResult<BackendFormat> {
    match value {
        "auto" => Ok(BackendFormat::Auto),
        "openai" => Ok(BackendFormat::OpenAi),
        "responses" => Ok(BackendFormat::Responses),
        "anthropic" => Ok(BackendFormat::Anthropic),
        "gemini" => Ok(BackendFormat::Gemini),
        _ => Err(PyValueError::new_err(format!(
            "Unknown backend format: {value:?}"
        ))),
    }
}

fn backend_format_name(format: BackendFormat) -> &'static str {
    match format {
        BackendFormat::Auto => "auto",
        BackendFormat::OpenAi => "openai",
        BackendFormat::Responses => "responses",
        BackendFormat::Anthropic => "anthropic",
        BackendFormat::Gemini => "gemini",
    }
}

fn backend_format_variant_name(format: BackendFormat) -> &'static str {
    match format {
        BackendFormat::Auto => "AUTO",
        BackendFormat::OpenAi => "OPENAI",
        BackendFormat::Responses => "RESPONSES",
        BackendFormat::Anthropic => "ANTHROPIC",
        BackendFormat::Gemini => "GEMINI",
    }
}

fn backend_format_object(py: Python<'_>, format: BackendFormat) -> PyResult<Py<PyAny>> {
    py.get_type::<PyBackendFormat>()
        .getattr(backend_format_variant_name(format))
        .map(Bound::unbind)
}

pub(crate) fn endpoint_config_from_python(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<EndpointConfig> {
    let Some(value) = value.filter(|value| !value.is_none()) else {
        return Ok(EndpointConfig::default());
    };
    if let Ok(endpoint) = value.extract::<PyRef<'_, PyEndpointConfig>>() {
        return Ok(endpoint.clone_core());
    }
    serde_json::from_value(value_from_python(value)?)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Reads `SWITCHYARD_INTAKE_CAPTURE_CONTENT`; false (metadata-only) unless truthy.
fn intake_capture_content_from_env() -> bool {
    std::env::var("SWITCHYARD_INTAKE_CAPTURE_CONTENT")
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn intake_queue_policy_from_python(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<IntakeQueueFullPolicy> {
    let Some(value) = value.filter(|value| !value.is_none()) else {
        return Ok(IntakeQueueFullPolicy::default());
    };
    if let Ok(policy) = value.extract::<PyRef<'_, PyIntakeQueueFullPolicy>>() {
        return Ok(policy.inner);
    }
    let raw = if let Ok(value_attr) = value.getattr("value") {
        value_attr.extract::<String>()?
    } else {
        value.extract::<String>()?
    };
    intake_queue_policy_from_str(&raw)
}

fn intake_queue_policy_from_str(value: &str) -> PyResult<IntakeQueueFullPolicy> {
    match value {
        "drop" => Ok(IntakeQueueFullPolicy::Drop),
        "block" => Ok(IntakeQueueFullPolicy::Block),
        _ => Err(PyValueError::new_err(format!(
            "Unknown intake queue-full policy: {value:?}"
        ))),
    }
}

fn intake_queue_policy_name(policy: IntakeQueueFullPolicy) -> &'static str {
    match policy {
        IntakeQueueFullPolicy::Drop => "drop",
        IntakeQueueFullPolicy::Block => "block",
    }
}

fn intake_queue_policy_variant_name(policy: IntakeQueueFullPolicy) -> &'static str {
    match policy {
        IntakeQueueFullPolicy::Drop => "DROP",
        IntakeQueueFullPolicy::Block => "BLOCK",
    }
}

fn intake_queue_policy_object(
    py: Python<'_>,
    policy: IntakeQueueFullPolicy,
) -> PyResult<Py<PyAny>> {
    py.get_type::<PyIntakeQueueFullPolicy>()
        .getattr(intake_queue_policy_variant_name(policy))
        .map(Bound::unbind)
}

fn to_python(py: Python<'_>, value: &impl Serialize) -> PyResult<Py<PyAny>> {
    let value =
        serde_json::to_value(value).map_err(|error| PyValueError::new_err(error.to_string()))?;
    value_to_python(py, &value)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyBackendFormat>()?;
    module.add_class::<PyEndpointConfig>()?;
    module.add_class::<PyLlmTarget>()?;
    module.add_class::<PyRandomRoutingProcessorConfig>()?;
    module.add_class::<PyIntakeQueueFullPolicy>()?;
    module.add_class::<PyIntakeSinkConfig>()?;
    Ok(())
}
