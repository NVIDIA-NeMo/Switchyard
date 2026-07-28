// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python host adapter for `switchyard-llm-client`.

use std::collections::{BTreeMap, HashSet};
use std::env;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures_util::StreamExt;
use parking_lot::RwLock;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyIterator;
use serde_json::Value;
use switchyard_components::stats::{selected_stats_model, selected_stats_tier};
use switchyard_components::{
    BackendSelection, BackendSelectionReason, StatsAccumulator, StatsBackendLatency,
};
use switchyard_core::{
    BackendFormat, BoxResponseStream, ChatRequest, ChatRequestType, ChatResponse, EndpointConfig,
    LlmBackend, LlmTarget, LlmTargetId, ModelId, ProxyContext, Result, StreamEvent,
    SwitchyardError,
};
use switchyard_llm_client::{
    Backend, HttpBackendConfig, LlmClientError, ModelConfig, RawResponse, TranslatingLlmClient,
};
use switchyard_protocol::{Context, WireFormat};

use crate::component_bindings::config::{
    backend_format_from_python, endpoint_config_from_python, PyLlmTarget,
};
use crate::component_bindings::stats::PyStatsAccumulator;
use crate::core_bindings::roles::PyLlmBackend;
use crate::errors::py_core_error;

const SUPPORTED_REQUEST_TYPES: [ChatRequestType; 3] = [
    ChatRequestType::OpenAiChat,
    ChatRequestType::OpenAiResponses,
    ChatRequestType::Anthropic,
];
const SWITCHYARD_VERSION_HEADER: &str = "X-Switchyard-Version";
const SWITCHYARD_VERSION_ENV: &str = "SWITCHYARD_VERSION";
const SWITCHYARD_TELEMETRY_OPT_OUT_ENV: &str = "SWITCHYARD_TELEMETRY_OPT_OUT";
const NEMO_SWITCHYARD_TELEMETRY_OPT_OUT_ENV: &str = "NEMO_SWITCHYARD_TELEMETRY_OPT_OUT";

struct TargetClient {
    target: LlmTarget,
    client: TranslatingLlmClient,
}

struct LlmClientHost {
    targets: Vec<TargetClient>,
    default_target_id: Option<LlmTargetId>,
    passthrough: Option<TranslatingLlmClient>,
    stats: RwLock<Option<StatsAccumulator>>,
}

impl LlmClientHost {
    fn new(
        targets: Vec<LlmTarget>,
        default_target_id: Option<LlmTargetId>,
        passthrough: Option<(EndpointConfig, BackendFormat)>,
    ) -> Result<Self> {
        if targets.is_empty() {
            let Some((endpoint, format)) = passthrough else {
                return Err(SwitchyardError::InvalidConfig(
                    "LlmClient requires at least one target or a passthrough endpoint".to_string(),
                ));
            };
            if default_target_id.is_some() {
                return Err(SwitchyardError::InvalidConfig(
                    "a passthrough LlmClient cannot set default_target_id".to_string(),
                ));
            }
            let backend = backend_from_config(format, endpoint, None, BTreeMap::new())?;
            let client = TranslatingLlmClient::with_default_backend(&[], Some(backend))
                .map_err(map_configuration_error)?;
            return Ok(Self {
                targets: Vec::new(),
                default_target_id: None,
                passthrough: Some(client),
                stats: RwLock::new(None),
            });
        }

        let mut seen = HashSet::new();
        let mut clients = Vec::with_capacity(targets.len());
        for target in targets {
            if !seen.insert(target.id.clone()) {
                return Err(SwitchyardError::InvalidConfig(format!(
                    "duplicate LLM target id: {}",
                    target.id
                )));
            }
            let backend = backend_from_target(&target)?;
            let model_config = ModelConfig::new(target.model.as_str(), backend, None);
            let client =
                TranslatingLlmClient::new(&[model_config]).map_err(map_configuration_error)?;
            clients.push(TargetClient { target, client });
        }

        if let Some(default_target_id) = &default_target_id {
            if !clients
                .iter()
                .any(|entry| &entry.target.id == default_target_id)
            {
                return Err(SwitchyardError::InvalidConfig(format!(
                    "default target {default_target_id} is not configured"
                )));
            }
        }

        Ok(Self {
            targets: clients,
            default_target_id,
            passthrough: None,
            stats: RwLock::new(None),
        })
    }

    fn target(&self, target_id: &LlmTargetId) -> Option<&TargetClient> {
        self.targets
            .iter()
            .find(|entry| &entry.target.id == target_id)
    }

    fn select_target<'a>(
        &'a self,
        ctx: &ProxyContext,
        request: &ChatRequest,
    ) -> Result<(&'a TargetClient, BackendSelectionReason)> {
        if let Some(target_id) = ctx.selected_target() {
            return self
                .target(target_id)
                .map(|target| (target, BackendSelectionReason::ContextTarget))
                .ok_or_else(|| self.unknown_target_error("selected", target_id));
        }
        if let Some(target_id) = &self.default_target_id {
            return self
                .target(target_id)
                .map(|target| (target, BackendSelectionReason::DefaultTarget))
                .ok_or_else(|| self.unknown_target_error("default", target_id));
        }
        if let [target] = self.targets.as_slice() {
            return Ok((target, BackendSelectionReason::SingleTarget));
        }

        let Some(model) = request.model() else {
            return Err(self.missing_selection_error(None));
        };
        let matches = self
            .targets
            .iter()
            .filter(|entry| entry.target.model.as_str() == model)
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [target] => Ok((*target, BackendSelectionReason::RequestModel)),
            [] => Err(self.missing_selection_error(Some(model))),
            _ => Err(SwitchyardError::InvalidConfig(format!(
                "request model {model:?} matches multiple targets; set selected_target explicitly"
            ))),
        }
    }

    fn unknown_target_error(&self, kind: &str, target_id: &LlmTargetId) -> SwitchyardError {
        SwitchyardError::InvalidConfig(format!(
            "{kind} target {target_id} is not configured; known targets: {}",
            self.known_target_ids()
        ))
    }

    fn missing_selection_error(&self, model: Option<&str>) -> SwitchyardError {
        let model = model
            .map(|model| format!(" and request model {model:?} did not match a configured target"))
            .unwrap_or_default();
        SwitchyardError::InvalidConfig(format!(
            "LlmClient has multiple targets but no selected target{model}; known targets: {}",
            self.known_target_ids()
        ))
    }

    fn known_target_ids(&self) -> String {
        self.targets
            .iter()
            .map(|entry| entry.target.id.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn attach_stats(&self, stats: StatsAccumulator) {
        *self.stats.write() = Some(stats);
    }

    fn stats(&self) -> Option<StatsAccumulator> {
        self.stats.read().clone()
    }
}

#[async_trait]
impl LlmBackend for LlmClientHost {
    fn supported_request_types(&self) -> &[ChatRequestType] {
        &SUPPORTED_REQUEST_TYPES
    }

    async fn call(&self, ctx: &mut ProxyContext, request: &ChatRequest) -> Result<ChatResponse> {
        ctx.inbound_format = ctx.inbound_format.or(Some(request.request_type()));
        let original_model = request.model().map(str::to_string);
        let (client, model, target_id) = if let Some(client) = &self.passthrough {
            let model = request
                .model()
                .ok_or_else(|| SwitchyardError::InvalidRequest("no model given".to_string()))?;
            let model_id = ModelId::new(model.to_string())?;
            ctx.insert(BackendSelection::for_model(
                model_id,
                original_model.clone(),
                BackendSelectionReason::PassthroughModel,
            ));
            (client, model.to_string(), "passthrough".to_string())
        } else {
            let (target, reason) = self.select_target(ctx, request)?;
            ctx.insert(BackendSelection::for_target(
                target.target.id.clone(),
                target.target.model.clone(),
                original_model.clone(),
                reason,
            ));
            (
                &target.client,
                target.target.model.as_str().to_string(),
                target.target.id.as_str().to_string(),
            )
        };

        let started_at = Instant::now();
        let response = client
            .call_rewrite_model_raw(
                Context::default(),
                request.body().clone(),
                None,
                Some(&model),
                wire_format(request.request_type()),
            )
            .await;
        let latency = started_at.elapsed();
        let stats = self.stats();

        match response {
            Ok(response) => {
                ctx.insert(StatsBackendLatency(latency));
                if let Some(stats) = stats {
                    stats.record_success(
                        selected_stats_model(ctx, Some(&model)),
                        Some(latency.as_secs_f64() * 1000.0),
                        selected_stats_tier(ctx).as_deref(),
                    )?;
                }
                raw_response(response, request.request_type())
            }
            Err(error) => {
                if let Some(stats) = stats {
                    stats.record_error(
                        selected_stats_model(ctx, Some(&model)),
                        selected_stats_tier(ctx).as_deref(),
                    )?;
                }
                Err(map_client_error(error, &target_id))
            }
        }
    }
}

#[pyclass(name = "LlmClient", extends = PyLlmBackend, skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct PyLlmClient {
    inner: Arc<LlmClientHost>,
}

#[pymethods]
impl PyLlmClient {
    #[new]
    #[pyo3(signature = (
        targets=None,
        *,
        default_target_id=None,
        endpoint=None,
        format=None,
    ))]
    fn py_new(
        targets: Option<&Bound<'_, PyAny>>,
        default_target_id: Option<String>,
        endpoint: Option<&Bound<'_, PyAny>>,
        format: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let targets = targets_from_python(targets)?;
        let default_target_id = default_target_id
            .map(LlmTargetId::new)
            .transpose()
            .map_err(|error| {
                PyValueError::new_err(format!("invalid default target id: {error}"))
            })?;
        let passthrough = if targets.is_empty() {
            let format = format
                .map(|value| backend_format_from_python(Some(value)))
                .transpose()?
                .unwrap_or(BackendFormat::OpenAi);
            Some((endpoint_config_from_python(endpoint)?, format))
        } else {
            if endpoint.is_some_and(|value| !value.is_none()) || format.is_some() {
                return Err(PyValueError::new_err(
                    "endpoint and format are only valid for a target-less passthrough LlmClient",
                ));
            }
            None
        };
        let client = Arc::new(
            LlmClientHost::new(targets, default_target_id, passthrough).map_err(py_core_error)?,
        );
        let base: Arc<dyn LlmBackend> = client.clone();
        Ok(PyClassInitializer::from(PyLlmBackend::from_native(base))
            .add_subclass(Self { inner: client }))
    }

    fn target_ids(&self) -> Vec<String> {
        self.inner
            .targets
            .iter()
            .map(|entry| entry.target.id.as_str().to_string())
            .collect()
    }

    #[getter]
    fn default_target_id(&self) -> Option<String> {
        self.inner
            .default_target_id
            .as_ref()
            .map(|target_id| target_id.as_str().to_string())
    }

    fn attach_stats(&self, stats: PyRef<'_, PyStatsAccumulator>) {
        self.inner.attach_stats(stats.clone_core());
    }

    fn __repr__(&self) -> String {
        if self.inner.passthrough.is_some() {
            return "LlmClient(passthrough=True)".to_string();
        }
        format!(
            "LlmClient(target_ids={:?}, default_target_id={:?})",
            self.target_ids(),
            self.default_target_id()
        )
    }
}

fn targets_from_python(value: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<LlmTarget>> {
    let Some(value) = value.filter(|value| !value.is_none()) else {
        return Ok(Vec::new());
    };
    let mut targets = Vec::new();
    for item in PyIterator::from_object(value)? {
        let item = item?;
        let target = item
            .extract::<PyRef<'_, PyLlmTarget>>()
            .map_err(PyErr::from)?;
        targets.push(target.clone_core());
    }
    Ok(targets)
}

fn backend_from_target(target: &LlmTarget) -> Result<Backend> {
    backend_from_config(
        target.format,
        target.endpoint.clone(),
        target.extra_body.clone(),
        target.extra_headers.clone(),
    )
}

fn backend_from_config(
    format: BackendFormat,
    endpoint: EndpointConfig,
    extra_body: Option<Value>,
    mut extra_headers: BTreeMap<String, String>,
) -> Result<Backend> {
    let (default_url, api_key_env) = match format {
        BackendFormat::OpenAi | BackendFormat::Responses => {
            ("https://api.openai.com/v1", "OPENAI_API_KEY")
        }
        BackendFormat::Anthropic => ("https://api.anthropic.com", "ANTHROPIC_API_KEY"),
        BackendFormat::Auto => {
            return Err(SwitchyardError::InvalidConfig(
                "LlmClient requires a resolved target format".to_string(),
            ))
        }
    };
    let timeout = endpoint
        .timeout_secs
        .map(|seconds| {
            if !seconds.is_finite() || seconds <= 0.0 {
                return Err(SwitchyardError::InvalidConfig(format!(
                    "target timeout_secs must be finite and positive, got {seconds:?}"
                )));
            }
            Ok(Duration::from_secs_f64(seconds))
        })
        .transpose()?;
    if !extra_headers
        .keys()
        .any(|name| name.eq_ignore_ascii_case(SWITCHYARD_VERSION_HEADER))
    {
        if let Some(version) = telemetry_header_value() {
            extra_headers.insert(SWITCHYARD_VERSION_HEADER.to_string(), version);
        }
    }
    let config = HttpBackendConfig {
        base_url: endpoint.base_url.unwrap_or_else(|| default_url.to_string()),
        api_key: endpoint.api_key.or_else(|| env::var(api_key_env).ok()),
        timeout,
        extra_body,
        extra_headers,
    };
    match format {
        BackendFormat::OpenAi => Ok(Backend::OpenAiChat(config)),
        BackendFormat::Responses => Ok(Backend::OpenAiResponses(config)),
        BackendFormat::Anthropic => Ok(Backend::Anthropic(config)),
        BackendFormat::Auto => Err(SwitchyardError::InvalidConfig(
            "LlmClient requires a resolved target format".to_string(),
        )),
    }
}

fn telemetry_header_value() -> Option<String> {
    if env_value_opts_out(SWITCHYARD_TELEMETRY_OPT_OUT_ENV)
        || env_value_opts_out(NEMO_SWITCHYARD_TELEMETRY_OPT_OUT_ENV)
    {
        return None;
    }
    Some(
        env::var(SWITCHYARD_VERSION_ENV)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string()),
    )
}

fn env_value_opts_out(name: &str) -> bool {
    let Ok(value) = env::var(name) else {
        return false;
    };
    !matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "" | "0" | "false" | "no"
    )
}

fn wire_format(request_type: ChatRequestType) -> WireFormat {
    match request_type {
        ChatRequestType::OpenAiChat => WireFormat::OpenAiChat,
        ChatRequestType::OpenAiResponses => WireFormat::OpenAiResponses,
        ChatRequestType::Anthropic => WireFormat::AnthropicMessages,
    }
}

fn raw_response(response: RawResponse, request_type: ChatRequestType) -> Result<ChatResponse> {
    match response {
        RawResponse::Buffered(body) => Ok(match request_type {
            ChatRequestType::OpenAiChat => ChatResponse::openai_completion(body),
            ChatRequestType::OpenAiResponses => ChatResponse::openai_responses_completion(body),
            ChatRequestType::Anthropic => ChatResponse::anthropic_completion(body),
        }),
        RawResponse::Stream(stream) => {
            let stream: BoxResponseStream = Box::pin(stream.map(|event| {
                event
                    .map(StreamEvent::Json)
                    .map_err(|error| SwitchyardError::Upstream(error.to_string()))
            }));
            Ok(match request_type {
                ChatRequestType::OpenAiChat => ChatResponse::OpenAiStream(stream),
                ChatRequestType::OpenAiResponses => ChatResponse::OpenAiResponsesStream(stream),
                ChatRequestType::Anthropic => ChatResponse::AnthropicStream(stream),
            })
        }
    }
}

fn map_configuration_error(error: LlmClientError) -> SwitchyardError {
    SwitchyardError::InvalidConfig(error.to_string())
}

fn map_client_error(error: LlmClientError, target_id: &str) -> SwitchyardError {
    match error {
        LlmClientError::InvalidRequest { message } => SwitchyardError::InvalidRequest(message),
        LlmClientError::Configuration { message } => SwitchyardError::InvalidConfig(message),
        LlmClientError::ContextWindowExceeded { model, message } => {
            SwitchyardError::ContextWindowExceeded {
                target_id: target_id.to_string(),
                model,
                message,
            }
        }
        LlmClientError::UpstreamHttp { status, body } => SwitchyardError::UpstreamHttp {
            provider: "switchyard-llm-client".to_string(),
            status_code: status,
            body,
        },
        other => SwitchyardError::Upstream(other.to_string()),
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyLlmClient>()?;
    Ok(())
}
