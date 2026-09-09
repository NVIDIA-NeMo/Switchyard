// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;
use std::sync::{Arc, Mutex};

use futures_util::{Stream, StreamExt};
use http::StatusCode;
use nemo_relay_plugin::{
    DataSchema, Json, LlmRequest as RelayRequest, LogSeverity, MetricKind, MetricMeasurement,
    MetricValueType, PluginRuntime,
};
use serde_json::{Map, json};
use switchyard_llm_client::{LlmCallObservation, RunObservation, RunObserver};
use switchyard_protocol::{
    LlmClientError, LlmResponse, LlmResponseChunk, LlmStreamError, Metadata, ProviderExtensions,
    Request, Response, Usage, WireFormat,
};
use switchyard_runner::{Route, RouteErrorSummary, Runner, stream_error_summary};
use switchyard_translation::{TranslationEngine, encode_stream_with_extensions};

use crate::config::SwitchyardConfig;
use crate::translation;

const ROUTING_MARK_SCHEMA_VERSION: &str = "1";

#[derive(Debug)]
pub(crate) struct RoutingMark {
    pub(crate) name: String,
    pub(crate) data: Json,
    pub(crate) metadata: Json,
    pub(crate) severity: Option<LogSeverity>,
}

impl RoutingMark {
    fn data_schema(&self) -> DataSchema {
        DataSchema {
            name: self.name.clone(),
            version: ROUTING_MARK_SCHEMA_VERSION.into(),
        }
    }
}

#[derive(Debug)]
pub(crate) struct RoutingMetric {
    pub(crate) name: String,
    pub(crate) measurements: Vec<MetricMeasurement>,
    pub(crate) metadata: Json,
}

#[derive(Debug)]
pub(crate) enum RoutingEvent {
    Mark(RoutingMark),
    Metric(RoutingMetric),
}

struct MetricDescriptor<'a> {
    name: &'a str,
    kind: MetricKind,
    value_type: MetricValueType,
    unit: Option<&'a str>,
    description: &'a str,
}

pub(crate) type ReturnedEventStream = Pin<Box<dyn Stream<Item = Result<Json, String>> + Send>>;
pub(crate) type RoutingEventEmitter = Arc<dyn Fn(RoutingEvent) + Send + Sync>;

pub(crate) struct Execution<T> {
    pub(crate) result: Result<T, String>,
    pub(crate) events: Vec<RoutingEvent>,
}

pub(crate) struct SwitchyardRuntime {
    runner: Runner,
    translation: TranslationEngine,
}

impl SwitchyardRuntime {
    pub(crate) fn new(config: SwitchyardConfig) -> Result<Self, String> {
        Ok(Self {
            runner: config.load_runner()?,
            translation: TranslationEngine::default(),
        })
    }

    pub(crate) fn manages_model(&self, model: &str) -> bool {
        self.runner.route(model).is_some()
    }

    pub(crate) fn decode_request(
        &self,
        inbound: WireFormat,
        request: RelayRequest,
        streaming: bool,
    ) -> Result<Request, String> {
        let mut llm_request = translation::decode_request(&self.translation, inbound, &request)?;
        llm_request.stream = streaming;
        let headers = string_headers(&request.headers);
        let mut metadata = Metadata::from_headers(&headers);
        let relay_gateway_placeholder = !headers.contains_key("x-switchyard-session-id")
            && headers
                .get("x-nemo-relay-source")
                .and_then(|value| value.to_str().ok())
                == Some("gateway")
            && metadata.session_id.as_deref() == Some("gateway-gateway");
        if relay_gateway_placeholder {
            metadata.session_id = None;
        }
        metadata.http_headers = Some(headers);
        Ok(Request {
            llm_request,
            raw_request: Some(request.content),
            metadata: Some(metadata),
        })
    }

    pub(crate) async fn execute_buffered(
        &self,
        inbound: WireFormat,
        request: Request,
    ) -> Execution<Json> {
        let request_extensions = request.llm_request.extensions.clone();
        let Execution { result, mut events } = self.execute(inbound, request).await;
        let (result, finalization_failed) = match result {
            Ok(response) => {
                let result = finalize_buffered_response(
                    &self.translation,
                    inbound,
                    response,
                    &request_extensions,
                );
                let failed = result.is_err();
                (result, failed)
            }
            Err(error) => (Err(error), false),
        };
        if finalization_failed {
            self.error_mark(&mut events, "response_finalization", None);
        }
        Execution { result, events }
    }

    pub(crate) async fn execute_stream(
        &self,
        inbound: WireFormat,
        request: Request,
        emit_event: RoutingEventEmitter,
    ) -> Execution<ReturnedEventStream> {
        let request_extensions = request.llm_request.extensions.clone();
        let Execution { result, mut events } = self.execute(inbound, request).await;
        let (result, finalization_failed) = match result {
            Ok(response) => {
                let metadata = events
                    .iter()
                    .find_map(|event| match event {
                        RoutingEvent::Mark(mark) => Some(mark.metadata.clone()),
                        RoutingEvent::Metric(_) => None,
                    })
                    .unwrap_or_else(|| Json::Object(Map::new()));
                let result =
                    returned_events(response, inbound, &request_extensions, metadata, emit_event);
                let failed = result.is_err();
                (result, failed)
            }
            Err(error) => (Err(error), false),
        };
        if finalization_failed {
            self.error_mark(&mut events, "response_finalization", None);
        }
        Execution { result, events }
    }

    async fn execute(&self, inbound: WireFormat, request: Request) -> Execution<Response> {
        let Some(route) = self.route(&request) else {
            return Execution {
                result: Err("Switchyard has no route for this request model".into()),
                events: Vec::new(),
            };
        };
        let metadata = identity_metadata(request.metadata.as_ref());
        let mut events = vec![RoutingEvent::Mark(RoutingMark {
            name: "switchyard.routing.requested".into(),
            data: json!({"algorithm": route.algorithm_name()}),
            metadata: metadata.clone(),
            severity: Some(LogSeverity::Info),
        })];
        events.push(request_metric(route.algorithm_name(), metadata.clone()));
        if let Err(error) = route.check_caller_format(inbound) {
            self.error_mark(&mut events, "caller_format", None);
            return Execution {
                result: Err(format!("Switchyard caller format is incompatible: {error}")),
                events,
            };
        }
        let observations = Arc::new(Mutex::new(Vec::new()));
        let observed = Arc::clone(&observations);
        let observer: RunObserver = Arc::new(move |observation| {
            observed
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(observation);
        });
        match route.execute(request, Some(observer)).await {
            Ok(output) => {
                self.emit_observations(&mut events, take_observations(&observations), &metadata);
                events.push(RoutingEvent::Mark(RoutingMark {
                    name: "switchyard.routing.decision".into(),
                    data: json!({
                        "algorithm": route.algorithm_name(),
                        "selected_model": output.selected_model,
                    }),
                    metadata,
                    severity: Some(LogSeverity::Info),
                }));
                Execution {
                    result: Ok(output.response),
                    events,
                }
            }
            Err(error) => {
                self.emit_observations(&mut events, take_observations(&observations), &metadata);
                self.route_execution_error_mark(
                    &mut events,
                    &error.execution_error_summary(),
                    None,
                );
                Execution {
                    result: Err("Switchyard route execution failed".into()),
                    events,
                }
            }
        }
    }

    fn route(&self, request: &Request) -> Option<&Route> {
        request
            .llm_request
            .model
            .as_deref()
            .and_then(|model| self.runner.route(model))
    }

    fn emit_observations(
        &self,
        events: &mut Vec<RoutingEvent>,
        observations: Vec<RunObservation>,
        metadata: &Json,
    ) {
        let mut call_index = 0;
        for observation in observations {
            match observation {
                RunObservation::LlmCall(call) => {
                    call_index += 1;
                    self.routing_call_events(events, call, call_index, metadata);
                }
                RunObservation::RoutingOverhead(duration) => {
                    let latency_ms = duration.as_secs_f64() * 1_000.0;
                    events.push(RoutingEvent::Mark(RoutingMark {
                        name: "switchyard.routing.overhead".into(),
                        data: json!({"latency_ms": latency_ms}),
                        metadata: metadata.clone(),
                        severity: Some(LogSeverity::Info),
                    }));
                    events.push(routing_overhead_metric(latency_ms, metadata.clone()));
                }
                RunObservation::AnswerCall(call) => {
                    events.extend(token_usage_metrics("answer", &call, metadata));
                }
            }
        }
    }

    fn routing_call_events(
        &self,
        events: &mut Vec<RoutingEvent>,
        call: LlmCallObservation,
        call_index: usize,
        metadata: &Json,
    ) {
        let outcome = if call.is_success { "ok" } else { "error" };
        let latency_ms = call.duration.as_secs_f64() * 1_000.0;
        let token_metrics = token_usage_metrics("routing", &call, metadata);
        events.push(RoutingEvent::Mark(RoutingMark {
            name: "switchyard.routing.llm_call".into(),
            data: json!({
                "call_index": call_index,
                "selected_model": call.selected_model.as_str(),
                "call_role": "routing",
                "outcome": outcome,
                "latency_ms": latency_ms,
            }),
            metadata: metadata.clone(),
            severity: Some(LogSeverity::Debug),
        }));
        events.extend(routing_call_metrics(outcome, latency_ms, metadata.clone()));
        events.extend(token_metrics);
    }

    fn error_mark(
        &self,
        events: &mut Vec<RoutingEvent>,
        failure_kind: &str,
        metadata: Option<&Json>,
    ) {
        let metadata = metadata
            .cloned()
            .unwrap_or_else(|| event_metadata(events).unwrap_or_else(|| Json::Object(Map::new())));
        events.push(RoutingEvent::Mark(RoutingMark {
            name: "switchyard.routing.error".into(),
            data: json!({"failure_kind": failure_kind}),
            metadata: metadata.clone(),
            severity: Some(LogSeverity::Error),
        }));
        events.push(failure_metric(failure_kind, None, None, None, metadata));
    }

    fn route_execution_error_mark(
        &self,
        events: &mut Vec<RoutingEvent>,
        summary: &RouteErrorSummary,
        metadata: Option<&Json>,
    ) {
        let metadata = metadata
            .cloned()
            .unwrap_or_else(|| event_metadata(events).unwrap_or_else(|| Json::Object(Map::new())));
        events.extend(route_execution_error_events(summary, metadata));
    }
}

pub(crate) fn emit_events(runtime: &PluginRuntime, events: Vec<RoutingEvent>) {
    for event in events {
        emit_event(runtime, event);
    }
}

pub(crate) fn emit_event(runtime: &PluginRuntime, event: RoutingEvent) {
    let result = match event {
        RoutingEvent::Mark(mark) => {
            let data_schema = mark.data_schema();
            runtime
                .emit_mark_with_options(
                    &mark.name,
                    Some(&mark.data),
                    Some(&mark.metadata),
                    Some(&data_schema),
                    mark.severity,
                )
                .map_err(|error| ("routing mark", mark.name, error))
        }
        RoutingEvent::Metric(metric) => runtime
            .emit_metric(&metric.name, metric.measurements, Some(&metric.metadata))
            .map_err(|error| ("routing metric", metric.name, error)),
    };
    if let Err((kind, name, error)) = result {
        eprintln!("Switchyard could not emit {kind} {name:?}: {error}");
    }
}

fn take_observations(observations: &Mutex<Vec<RunObservation>>) -> Vec<RunObservation> {
    std::mem::take(
        &mut *observations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()),
    )
}

fn finalize_buffered_response(
    translation_engine: &TranslationEngine,
    inbound: WireFormat,
    response: Response,
    request_extensions: &ProviderExtensions,
) -> Result<Json, String> {
    let LlmResponse::Agg(response) = response.llm_response else {
        return Err("Switchyard returned a stream for a buffered request".into());
    };
    translation::encode_response(translation_engine, inbound, &response, request_extensions)
}

fn returned_events(
    response: Response,
    inbound: WireFormat,
    request_extensions: &ProviderExtensions,
    metadata: Json,
    emit_event: RoutingEventEmitter,
) -> Result<ReturnedEventStream, String> {
    let served_model = response.served_model().cloned();
    let (chunks, observe_stream_usage) = match response.llm_response {
        LlmResponse::Agg(response) => (response.into_stream(), false),
        LlmResponse::Stream(chunks) => (chunks, true),
    };
    let mut latest_usage = None;
    let mut terminal_seen = false;
    let mut usage_emitted = false;
    let mut stream_failed = false;
    let mut failure_emitted = false;
    let chunks = Box::pin(chunks.map(move |item| {
        let in_band_error = item
            .as_ref()
            .ok()
            .and_then(|event| normalized_stream_error(event.normalized()));
        let stream_error = in_band_error.as_ref().or_else(|| item.as_ref().err());
        stream_failed |= stream_error.is_some();
        if let Some(error) = stream_error
            && !failure_emitted
        {
            for event in route_execution_error_events(
                &stream_error_summary(error, served_model.as_ref()),
                metadata.clone(),
            ) {
                emit_event(event);
            }
            failure_emitted = true;
        }
        if observe_stream_usage && !stream_failed && !usage_emitted {
            if let Ok(event) = &item {
                for chunk in event.normalized() {
                    match chunk {
                        LlmResponseChunk::Usage(usage) => latest_usage = Some(usage.clone()),
                        LlmResponseChunk::MessageStop { .. } => terminal_seen = true,
                        _ => {}
                    }
                }
            }
            if terminal_seen
                && let (Some(model), Some(usage)) = (served_model.as_ref(), latest_usage.as_ref())
            {
                for event in usage_metrics("answer", model.as_str(), usage, &metadata) {
                    emit_event(event);
                }
                usage_emitted = true;
            }
        }
        item
    }));
    let events = encode_stream_with_extensions(chunks, inbound, None, request_extensions)
        .map_err(|error| format!("Switchyard response stream setup failed: {error}"))?;
    Ok(Box::pin(
        events.map(|item| item.map_err(relay_stream_error)),
    ))
}

/// Produces a telemetry-only error summary source for a normalized in-band failure.
///
/// The translation layer retains the original provider error event for client delivery;
/// this conversion is only used to classify the failure without putting its message in marks.
fn normalized_stream_error(chunks: &[LlmResponseChunk]) -> Option<LlmClientError> {
    chunks.iter().find_map(|chunk| match chunk {
        LlmResponseChunk::DecodeError { message } => {
            Some(LlmClientError::ResponseTranslation(message.clone()))
        }
        LlmResponseChunk::StreamError { message } => Some(LlmClientError::UpstreamHttp {
            status: StatusCode::BAD_GATEWAY,
            body: message.clone(),
        }),
        _ => None,
    })
}

/// Converts translation errors to Relay's string-only stream-error boundary.
///
/// `LlmStreamError::Upstream` retains a provider-shaped JSON value so its caller can
/// make protocol decisions, but neither that value nor a client error's source message
/// is safe for Relay's outer observability scope. Preserve the error result while
/// exposing only a stable failure class.
fn relay_stream_error(error: LlmStreamError) -> String {
    match error {
        LlmStreamError::Upstream(_) => "Switchyard upstream stream error".into(),
        LlmStreamError::Client(error) => format!(
            "Switchyard stream error: {}",
            stream_error_summary(&error, None).kind.as_str()
        ),
    }
}

fn route_execution_error_mark(summary: &RouteErrorSummary, metadata: Json) -> RoutingMark {
    RoutingMark {
        name: "switchyard.routing.error".into(),
        data: json!({
            "failure_kind": "route_execution",
            "category": summary.kind.as_str(),
            "phase": summary.phase.as_str(),
            "upstream_status": summary.upstream_status,
            "target": summary.target.as_ref().map(|target| target.as_str()),
        }),
        metadata,
        severity: Some(LogSeverity::Error),
    }
}

fn route_execution_error_events(summary: &RouteErrorSummary, metadata: Json) -> Vec<RoutingEvent> {
    vec![
        RoutingEvent::Mark(route_execution_error_mark(summary, metadata.clone())),
        failure_metric(
            "route_execution",
            Some(summary.kind.as_str()),
            Some(summary.phase.as_str()),
            summary.upstream_status,
            metadata,
        ),
    ]
}

fn event_metadata(events: &[RoutingEvent]) -> Option<Json> {
    events.iter().find_map(|event| match event {
        RoutingEvent::Mark(mark) => Some(mark.metadata.clone()),
        RoutingEvent::Metric(_) => None,
    })
}

fn request_metric(algorithm: &str, metadata: Json) -> RoutingEvent {
    counter_metric(
        "switchyard.routing.requests",
        "Requests managed by Switchyard routing.",
        json!({"algorithm": algorithm}),
        metadata,
    )
}

fn routing_call_metrics(outcome: &str, latency_ms: f64, metadata: Json) -> [RoutingEvent; 2] {
    let attributes = json!({"outcome": outcome});
    [
        counter_metric(
            "switchyard.routing.llm_calls",
            "Switchyard model calls made while routing.",
            attributes.clone(),
            metadata.clone(),
        ),
        histogram_metric(
            "switchyard.routing.llm_call.duration",
            "Duration of Switchyard model calls made while routing.",
            latency_ms,
            attributes,
            metadata,
        ),
    ]
}

fn routing_overhead_metric(latency_ms: f64, metadata: Json) -> RoutingEvent {
    histogram_metric(
        "switchyard.routing.overhead",
        "Time needed to produce the Switchyard routing outcome, including routing model calls.",
        latency_ms,
        json!({}),
        metadata,
    )
}

fn token_usage_metrics(
    call_role: &str,
    call: &LlmCallObservation,
    metadata: &Json,
) -> Vec<RoutingEvent> {
    let Some(usage) = call.usage.as_ref() else {
        return Vec::new();
    };
    usage_metrics(call_role, call.selected_model.as_str(), usage, metadata)
}

fn usage_metrics(
    call_role: &str,
    target_model: &str,
    usage: &Usage,
    metadata: &Json,
) -> Vec<RoutingEvent> {
    [
        ("input", usage.input_tokens),
        ("cached_input", usage.cached_input_tokens()),
        ("cache_creation_input", usage.cache_creation_input_tokens()),
        ("output", usage.output_tokens),
        ("reasoning", usage.reasoning_tokens),
        ("total", usage.total_tokens),
    ]
    .into_iter()
    .filter_map(|(token_type, value)| {
        value.map(|value| {
            metric(
                MetricDescriptor {
                    name: "switchyard.routing.llm_tokens",
                    kind: MetricKind::Counter,
                    value_type: MetricValueType::U64,
                    unit: Some("{token}"),
                    description: "Normalized tokens used by Switchyard model calls.",
                },
                json!(value),
                json!({
                    "call_role": call_role,
                    "target_model": target_model,
                    "token_type": token_type,
                }),
                metadata.clone(),
            )
        })
    })
    .collect()
}

fn failure_metric(
    failure_kind: &str,
    category: Option<&str>,
    phase: Option<&str>,
    upstream_status: Option<u16>,
    metadata: Json,
) -> RoutingEvent {
    let mut attributes = Map::new();
    attributes.insert("failure_kind".into(), Json::String(failure_kind.into()));
    if let Some(category) = category {
        attributes.insert("category".into(), Json::String(category.into()));
    }
    if let Some(phase) = phase {
        attributes.insert("phase".into(), Json::String(phase.into()));
    }
    if let Some(upstream_status) = upstream_status {
        attributes.insert("upstream_status".into(), Json::from(upstream_status));
    }
    counter_metric(
        "switchyard.routing.failures",
        "Terminal Switchyard routing failures.",
        Json::Object(attributes),
        metadata,
    )
}

fn counter_metric(name: &str, description: &str, attributes: Json, metadata: Json) -> RoutingEvent {
    metric(
        MetricDescriptor {
            name,
            kind: MetricKind::Counter,
            value_type: MetricValueType::U64,
            unit: Some("{event}"),
            description,
        },
        json!(1),
        attributes,
        metadata,
    )
}

fn histogram_metric(
    name: &str,
    description: &str,
    value: f64,
    attributes: Json,
    metadata: Json,
) -> RoutingEvent {
    metric(
        MetricDescriptor {
            name,
            kind: MetricKind::Histogram,
            value_type: MetricValueType::F64,
            unit: Some("ms"),
            description,
        },
        json!(value),
        attributes,
        metadata,
    )
}

fn metric(
    descriptor: MetricDescriptor<'_>,
    value: Json,
    attributes: Json,
    metadata: Json,
) -> RoutingEvent {
    RoutingEvent::Metric(RoutingMetric {
        name: descriptor.name.into(),
        measurements: vec![MetricMeasurement {
            name: descriptor.name.into(),
            kind: descriptor.kind,
            value_type: descriptor.value_type,
            value,
            unit: descriptor.unit.map(Into::into),
            description: Some(descriptor.description.into()),
            attributes: Some(attributes),
            boundaries: None,
        }],
        metadata,
    })
}

fn string_headers(headers: &Map<String, Json>) -> http::HeaderMap {
    let mut parsed = http::HeaderMap::with_capacity(headers.len());
    for (name, value) in headers {
        let Some(value) = value.as_str() else {
            continue;
        };
        let (Ok(name), Ok(value)) = (
            http::HeaderName::from_bytes(name.as_bytes()),
            http::HeaderValue::from_str(value),
        ) else {
            continue;
        };
        parsed.insert(name, value);
    }
    parsed
}

fn identity_metadata(metadata: Option<&Metadata>) -> Json {
    json!({
        "session_id": metadata.and_then(|value| value.session_id.as_deref()),
        "agent_id": metadata.and_then(|value| value.agent_id.as_deref()),
        "parent_agent_id": metadata.and_then(|value| value.parent_agent_id.as_deref()),
        "task_id": metadata.and_then(|value| value.task_id.as_deref()),
        "turn_id": metadata.and_then(|value| value.turn_id.as_deref()),
        "correlation_id": metadata.and_then(|value| value.correlation_id.as_deref()),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};

    use switchyard_llm_client::ClientRouter;
    use switchyard_protocol::{LlmClientError, LlmResponseStreamEvent, ModelId, Usage};
    use switchyard_runner::{AlgorithmSpec, ModelCapabilities, RunnerError};
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    fn runtime_for(model: &str) -> SwitchyardRuntime {
        let algorithm = AlgorithmSpec::Noop {}
            .build("relay", &BTreeMap::new())
            .expect("noop route should build");
        let route = Route::new(
            algorithm,
            ClientRouter::new(HashMap::new()),
            None,
            ModelCapabilities::default(),
            None,
            None,
            Vec::new(),
        );
        SwitchyardRuntime {
            runner: Runner::new(vec![(ModelId::from(model), route)]),
            translation: TranslationEngine::default(),
        }
    }

    fn runtime_for_target(format: &str, base_url: &str) -> SwitchyardRuntime {
        let deployment = json!({
            "schema_version": 1,
            "llm_clients": {
                "target": {
                    "format": format,
                    "base_url": base_url,
                }
            },
            "targets": {
                "default": {
                    "id": "target/model",
                    "llm_client": "target",
                }
            },
            "routes": {
                "default": {
                    "id": "switchyard/default",
                    "type": "passthrough",
                    "target": "default",
                }
            }
        });
        SwitchyardRuntime::new(crate::config::SwitchyardConfig {
            priority: 0,
            switchyard_config_path: None,
            switchyard_config: Some(deployment.as_object().unwrap().clone()),
        })
        .expect("cross-format runtime should load")
    }

    fn namespaced_responses_request() -> RelayRequest {
        RelayRequest {
            headers: Map::new(),
            content: json!({
                "model": "switchyard/default",
                "input": "Search the docs",
                "tools": [{
                    "type": "namespace",
                    "name": "mcp__docs",
                    "tools": [{
                        "type": "function",
                        "name": "search",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    }],
                }],
            }),
        }
    }

    #[test]
    fn only_configured_route_models_are_managed() {
        let runtime = runtime_for("switchyard");
        assert!(runtime.manages_model("switchyard"));
        assert!(!runtime.manages_model("other"));
    }

    #[tokio::test]
    async fn responses_requests_use_the_configured_target_format() {
        for (format, endpoint, upstream_response) in [
            (
                "openai_chat",
                "/v1/chat/completions",
                json!({
                    "id": "chatcmpl_test",
                    "object": "chat.completion",
                    "model": "target/model",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }),
            ),
            (
                "anthropic_messages",
                "/v1/messages",
                json!({
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": "target/model",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }),
            ),
        ] {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path(endpoint))
                .and(body_partial_json(json!({"model": "target/model"})))
                .respond_with(ResponseTemplate::new(200).set_body_json(upstream_response))
                .expect(1)
                .mount(&server)
                .await;
            let runtime = runtime_for_target(format, &format!("{}/v1", server.uri()));
            let request = RelayRequest {
                headers: Map::new(),
                content: json!({
                    "model": "switchyard/default",
                    "input": "hello",
                }),
            };
            let decoded = runtime
                .decode_request(WireFormat::OpenAiResponses, request, false)
                .expect("Responses request should decode");
            assert_eq!(
                decoded
                    .metadata
                    .as_ref()
                    .and_then(|metadata| metadata.wire_format),
                None
            );

            let execution = runtime
                .execute_buffered(WireFormat::OpenAiResponses, decoded)
                .await;
            let response = execution.result.expect("target call should succeed");
            assert_eq!(response["object"], "response");
            assert_eq!(response["model"], "target/model");
        }
    }

    #[tokio::test]
    async fn buffered_responses_restore_codex_tool_namespaces() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_partial_json(json!({"model": "target/model"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "chatcmpl_tool",
                "object": "chat.completion",
                "model": "target/model",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_search",
                            "type": "function",
                            "function": {
                                "name": "mcp__docs__search",
                                "arguments": "{\"query\":\"routing\"}",
                            },
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            })))
            .expect(1)
            .mount(&server)
            .await;
        let runtime = runtime_for_target("openai_chat", &format!("{}/v1", server.uri()));
        let request = runtime
            .decode_request(
                WireFormat::OpenAiResponses,
                namespaced_responses_request(),
                false,
            )
            .expect("namespaced Responses request should decode");

        let response = runtime
            .execute_buffered(WireFormat::OpenAiResponses, request)
            .await
            .result
            .expect("buffered target call should succeed");
        let tool_call = response["output"]
            .as_array()
            .and_then(|output| output.iter().find(|item| item["type"] == "function_call"))
            .expect("Responses output should contain a function call");
        assert_eq!(tool_call["name"], "search");
        assert_eq!(tool_call["namespace"], "mcp__docs");
    }

    #[tokio::test]
    async fn streaming_responses_restore_codex_tool_namespaces() {
        let server = MockServer::start().await;
        let sse = concat!(
            "data: {\"id\":\"chatcmpl_tool\",\"object\":\"chat.completion.chunk\",\"model\":\"target/model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_search\",\"type\":\"function\",\"function\":{\"name\":\"mcp__docs__search\",\"arguments\":\"{\\\"query\\\":\\\"routing\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl_tool\",\"object\":\"chat.completion.chunk\",\"model\":\"target/model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_partial_json(
                json!({"model": "target/model", "stream": true}),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"))
            .expect(1)
            .mount(&server)
            .await;
        let runtime = runtime_for_target("openai_chat", &format!("{}/v1", server.uri()));
        let request = runtime
            .decode_request(
                WireFormat::OpenAiResponses,
                namespaced_responses_request(),
                true,
            )
            .expect("namespaced streaming Responses request should decode");

        let stream = runtime
            .execute_stream(WireFormat::OpenAiResponses, request, Arc::new(|_| {}))
            .await
            .result
            .expect("streaming target call should succeed");
        let events = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("response stream should encode");

        for event_type in ["response.output_item.added", "response.output_item.done"] {
            let item = events
                .iter()
                .find(|event| event["type"] == event_type)
                .map(|event| &event["item"])
                .unwrap_or_else(|| panic!("stream produced no {event_type}"));
            assert_eq!(item["name"], "search", "{event_type}");
            assert_eq!(item["namespace"], "mcp__docs", "{event_type}");
        }
        let completed = events
            .iter()
            .find(|event| event["type"] == "response.completed")
            .expect("stream produced no response.completed event");
        assert_eq!(completed["response"]["output"][0]["name"], "search");
        assert_eq!(completed["response"]["output"][0]["namespace"], "mcp__docs");
    }

    #[test]
    fn execution_failure_mark_uses_the_safe_runner_summary() {
        let secret = "provider response body";
        let error = RunnerError::Client(LlmClientError::ContextWindowExceeded {
            model: ModelId::from("weak"),
            message: secret.into(),
        });

        let mark = route_execution_error_mark(
            &error.execution_error_summary(),
            json!({"session_id": "session"}),
        );

        assert_eq!(mark.name, "switchyard.routing.error");
        assert_eq!(mark.data["failure_kind"], "route_execution");
        assert_eq!(mark.data["category"], "context_window_exceeded");
        assert_eq!(mark.data["phase"], "before_response");
        assert_eq!(mark.data["target"], "weak");
        assert_eq!(mark.data["upstream_status"], Json::Null);
        assert_eq!(mark.severity, Some(LogSeverity::Error));
        assert_eq!(mark.data_schema().name, "switchyard.routing.error");
        assert_eq!(mark.data_schema().version, "1");
        assert!(!mark.data.to_string().contains(secret));
    }

    #[test]
    fn routing_observations_emit_debug_marks_and_metrics() {
        let runtime = runtime_for("switchyard");
        let mut events = Vec::new();
        runtime.emit_observations(
            &mut events,
            vec![
                RunObservation::LlmCall(LlmCallObservation {
                    selected_model: ModelId::from("routing-model"),
                    is_success: false,
                    duration: std::time::Duration::from_millis(12),
                    usage: Some(Usage {
                        input_tokens: Some(4),
                        ..Usage::default()
                    }),
                }),
                RunObservation::RoutingOverhead(std::time::Duration::from_millis(3)),
            ],
            &json!({"session_id": "session"}),
        );

        assert_eq!(events.len(), 6);
        let RoutingEvent::Mark(call_mark) = &events[0] else {
            panic!("first event should be the routing call mark");
        };
        assert_eq!(call_mark.name, "switchyard.routing.llm_call");
        assert_eq!(call_mark.severity, Some(LogSeverity::Debug));
        assert_eq!(call_mark.data["outcome"], "error");
        assert!(call_mark.data.get("usage").is_none());

        let RoutingEvent::Metric(call_count) = &events[1] else {
            panic!("second event should be the routing call counter");
        };
        assert_eq!(call_count.name, "switchyard.routing.llm_calls");
        assert_eq!(call_count.measurements[0].kind, MetricKind::Counter);
        assert_eq!(
            call_count.measurements[0].attributes,
            Some(json!({"outcome": "error"}))
        );

        let RoutingEvent::Metric(call_duration) = &events[2] else {
            panic!("third event should be the routing call histogram");
        };
        assert_eq!(call_duration.name, "switchyard.routing.llm_call.duration");
        assert_eq!(call_duration.measurements[0].kind, MetricKind::Histogram);
        assert_eq!(call_duration.measurements[0].value, json!(12.0));

        let RoutingEvent::Metric(tokens) = &events[3] else {
            panic!("fourth event should be the routing token counter");
        };
        assert_eq!(tokens.name, "switchyard.routing.llm_tokens");

        let RoutingEvent::Mark(overhead_mark) = &events[4] else {
            panic!("fifth event should be the routing overhead mark");
        };
        assert_eq!(overhead_mark.severity, Some(LogSeverity::Info));

        let RoutingEvent::Metric(overhead) = &events[5] else {
            panic!("sixth event should be the routing overhead histogram");
        };
        assert_eq!(overhead.name, "switchyard.routing.overhead");
        assert_eq!(overhead.measurements[0].attributes, Some(json!({})));
    }

    #[test]
    fn request_and_failure_metrics_use_bounded_attributes() {
        let RoutingEvent::Metric(request) = request_metric("stage_router", json!({})) else {
            panic!("request should be a metric");
        };
        assert_eq!(request.name, "switchyard.routing.requests");
        assert_eq!(
            request.measurements[0].attributes,
            Some(json!({"algorithm": "stage_router"}))
        );

        let RoutingEvent::Metric(failure) = failure_metric(
            "route_execution",
            Some("upstream_http"),
            Some("before_response"),
            Some(503),
            json!({}),
        ) else {
            panic!("failure should be a metric");
        };
        assert_eq!(failure.name, "switchyard.routing.failures");
        assert_eq!(
            failure.measurements[0].attributes,
            Some(json!({
                "failure_kind": "route_execution",
                "category": "upstream_http",
                "phase": "before_response",
                "upstream_status": 503,
            }))
        );
    }

    #[test]
    fn token_usage_metrics_distinguish_routing_and_answer_targets() {
        let call = LlmCallObservation {
            selected_model: ModelId::from("judge-model"),
            is_success: true,
            duration: std::time::Duration::from_millis(1),
            usage: Some(Usage {
                input_tokens: Some(11),
                cache: Usage::cache_details(Some(3), Some(2)),
                output_tokens: Some(7),
                total_tokens: Some(23),
                reasoning_tokens: Some(5),
            }),
        };

        let routing = token_usage_metrics("routing", &call, &json!({"session_id": "session"}));
        assert_eq!(routing.len(), 6);
        for event in &routing {
            let RoutingEvent::Metric(metric) = event else {
                panic!("token usage should be emitted as a metric");
            };
            assert_eq!(metric.name, "switchyard.routing.llm_tokens");
            assert_eq!(metric.measurements[0].kind, MetricKind::Counter);
            assert_eq!(metric.measurements[0].unit.as_deref(), Some("{token}"));
            assert_eq!(
                metric.measurements[0].attributes.as_ref().unwrap()["call_role"],
                "routing"
            );
            assert_eq!(
                metric.measurements[0].attributes.as_ref().unwrap()["target_model"],
                "judge-model"
            );
        }
        let token_values = routing
            .iter()
            .map(|event| {
                let RoutingEvent::Metric(metric) = event else {
                    panic!("token usage should be emitted as a metric");
                };
                metric.measurements[0].value.clone()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            token_values,
            vec![json!(11), json!(3), json!(2), json!(7), json!(5), json!(23)]
        );

        let answer = token_usage_metrics("answer", &call, &json!({}));
        let RoutingEvent::Metric(metric) = &answer[0] else {
            panic!("answer usage should be emitted as a metric");
        };
        assert_eq!(
            metric.measurements[0].attributes.as_ref().unwrap()["call_role"],
            "answer"
        );
    }

    #[test]
    fn answer_observations_emit_token_metrics_without_answer_logs() {
        let runtime = runtime_for("switchyard");
        let mut events = Vec::new();
        runtime.emit_observations(
            &mut events,
            vec![RunObservation::AnswerCall(LlmCallObservation {
                selected_model: ModelId::from("selected-target"),
                is_success: true,
                duration: std::time::Duration::from_millis(2),
                usage: Some(Usage {
                    output_tokens: Some(9),
                    ..Usage::default()
                }),
            })],
            &json!({}),
        );

        assert_eq!(events.len(), 1);
        let RoutingEvent::Metric(metric) = &events[0] else {
            panic!("answer observation should only emit a token metric");
        };
        assert_eq!(metric.name, "switchyard.routing.llm_tokens");
        assert_eq!(
            metric.measurements[0].attributes,
            Some(json!({
                "call_role": "answer",
                "target_model": "selected-target",
                "token_type": "output",
            }))
        );
    }

    #[tokio::test]
    async fn stream_usage_emits_answer_token_metrics() {
        let response = Response {
            llm_response: LlmResponse::Stream(Box::pin(futures_util::stream::iter([
                Ok(LlmResponseStreamEvent::new(vec![
                    LlmResponseChunk::MessageStop { reason: None },
                ])),
                Ok(LlmResponseStreamEvent::new(vec![LlmResponseChunk::Usage(
                    Usage {
                        input_tokens: Some(10),
                        output_tokens: Some(3),
                        total_tokens: Some(13),
                        ..Usage::default()
                    },
                )])),
            ]))),
            metadata: Some(Metadata {
                served_model: Some(ModelId::from("selected-target")),
                ..Default::default()
            }),
        };
        let captured = Arc::new(Mutex::new(Vec::new()));
        let emitted = Arc::clone(&captured);
        let mut stream = returned_events(
            response,
            WireFormat::OpenAiChat,
            &ProviderExtensions::default(),
            json!({"session_id": "session"}),
            Arc::new(move |event| emitted.lock().unwrap().push(event)),
        )
        .expect("stream setup should succeed");

        assert!(captured.lock().unwrap().is_empty());
        assert!(stream.next().await.expect("encoded stream event").is_ok());
        assert!(captured.lock().unwrap().is_empty());
        assert!(stream.next().await.expect("encoded usage event").is_ok());

        let events = captured.lock().unwrap();
        assert_eq!(events.len(), 3);
        let token_types = events
            .iter()
            .map(|event| {
                let RoutingEvent::Metric(metric) = event else {
                    panic!("stream usage should emit only metrics");
                };
                assert_eq!(metric.name, "switchyard.routing.llm_tokens");
                assert_eq!(metric.metadata, json!({"session_id": "session"}));
                let attributes = metric.measurements[0].attributes.as_ref().unwrap();
                assert_eq!(attributes["call_role"], "answer");
                assert_eq!(attributes["target_model"], "selected-target");
                attributes["token_type"].as_str().unwrap().to_string()
            })
            .collect::<Vec<_>>();
        assert_eq!(token_types, ["input", "output", "total"]);
    }

    #[tokio::test]
    async fn aggregate_response_stream_does_not_reemit_usage_metrics() {
        let mut aggregate =
            switchyard_protocol::text_response(Some("selected-target".into()), "ok");
        aggregate.usage = Usage {
            input_tokens: Some(10),
            output_tokens: Some(3),
            total_tokens: Some(13),
            ..Usage::default()
        };
        let response = Response {
            llm_response: LlmResponse::Agg(aggregate),
            metadata: Some(Metadata {
                served_model: Some(ModelId::from("selected-target")),
                ..Default::default()
            }),
        };
        let captured = Arc::new(Mutex::new(Vec::new()));
        let emitted = Arc::clone(&captured);
        let stream = returned_events(
            response,
            WireFormat::OpenAiChat,
            &ProviderExtensions::default(),
            json!({}),
            Arc::new(move |event| emitted.lock().unwrap().push(event)),
        )
        .expect("stream setup should succeed");

        let events = stream.collect::<Vec<_>>().await;
        assert!(events.iter().all(Result::is_ok));
        assert!(captured.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn stream_failure_emits_a_safe_failure_mark() {
        let secret = "provider response body";
        let response = Response {
            llm_response: LlmResponse::Stream(Box::pin(futures_util::stream::iter([Err::<
                LlmResponseStreamEvent,
                LlmClientError,
            >(
                LlmClientError::ContextWindowExceeded {
                    model: ModelId::from("weak"),
                    message: secret.into(),
                },
            )]))),
            metadata: Some(Metadata {
                served_model: Some(ModelId::from("strong")),
                ..Default::default()
            }),
        };
        let captured = Arc::new(Mutex::new(Vec::new()));
        let emitted = Arc::clone(&captured);
        let stream = returned_events(
            response,
            WireFormat::OpenAiChat,
            &ProviderExtensions::default(),
            json!({"session_id": "session"}),
            Arc::new(move |mark| emitted.lock().unwrap().push(mark)),
        )
        .expect("stream setup should succeed");

        let events = stream.collect::<Vec<_>>().await;
        assert!(events[0].is_err());
        assert!(
            !events[0]
                .as_ref()
                .expect_err("stream should fail")
                .contains(secret)
        );
        let events = captured.lock().unwrap();
        assert_eq!(events.len(), 2);
        let RoutingEvent::Mark(mark) = &events[0] else {
            panic!("first event should be the safe failure mark");
        };
        assert_eq!(mark.data["category"], "context_window_exceeded");
        assert_eq!(mark.data["phase"], "during_stream");
        assert_eq!(mark.data["target"], "strong");
        assert_eq!(mark.severity, Some(LogSeverity::Error));
        assert!(!mark.data.to_string().contains(secret));
        let RoutingEvent::Metric(metric) = &events[1] else {
            panic!("second event should be the failure counter");
        };
        assert_eq!(metric.name, "switchyard.routing.failures");
        assert_eq!(
            metric.measurements[0].attributes,
            Some(json!({
                "failure_kind": "route_execution",
                "category": "context_window_exceeded",
                "phase": "during_stream",
            }))
        );
    }

    async fn assert_normalized_stream_failure(
        chunk: LlmResponseChunk,
        expected_category: &str,
        expected_status: Option<u16>,
    ) {
        let secret = "provider response body";
        let response = Response {
            llm_response: LlmResponse::Stream(Box::pin(futures_util::stream::iter([Ok(
                LlmResponseStreamEvent::new(vec![chunk]),
            )]))),
            metadata: Some(Metadata {
                served_model: Some(ModelId::from("selected-target")),
                ..Default::default()
            }),
        };
        let captured = Arc::new(Mutex::new(Vec::new()));
        let emitted = Arc::clone(&captured);
        let stream = returned_events(
            response,
            WireFormat::OpenAiChat,
            &ProviderExtensions::default(),
            json!({"session_id": "session"}),
            Arc::new(move |event| emitted.lock().unwrap().push(event)),
        )
        .expect("stream setup should succeed");

        let client_events = stream.collect::<Vec<_>>().await;
        assert_eq!(client_events.len(), 1);
        assert!(client_events[0].is_err());
        assert!(
            !client_events[0]
                .as_ref()
                .expect_err("stream should fail")
                .contains(secret)
        );

        let events = captured.lock().unwrap();
        assert_eq!(events.len(), 2, "one mark and one counter should emit");
        let RoutingEvent::Mark(mark) = &events[0] else {
            panic!("first event should be the safe failure mark");
        };
        assert_eq!(mark.data["category"], expected_category);
        assert_eq!(mark.data["phase"], "during_stream");
        assert_eq!(
            mark.data["upstream_status"],
            expected_status.map_or(Json::Null, Json::from)
        );
        assert_eq!(mark.data["target"], "selected-target");
        assert!(!mark.data.to_string().contains(secret));
        let RoutingEvent::Metric(metric) = &events[1] else {
            panic!("second event should be the failure counter");
        };
        assert_eq!(metric.name, "switchyard.routing.failures");
        let mut expected_attributes = json!({
            "failure_kind": "route_execution",
            "category": expected_category,
            "phase": "during_stream",
        });
        if let Some(status) = expected_status {
            expected_attributes
                .as_object_mut()
                .expect("expected metric attributes should be an object")
                .insert("upstream_status".into(), json!(status));
        }
        assert_eq!(metric.measurements[0].attributes, Some(expected_attributes));
    }

    #[tokio::test]
    async fn normalized_stream_error_emits_failure_telemetry_once() {
        assert_normalized_stream_failure(
            LlmResponseChunk::StreamError {
                message: "provider response body".into(),
            },
            "upstream_http",
            Some(502),
        )
        .await;
    }

    #[tokio::test]
    async fn normalized_decode_error_emits_failure_telemetry_once() {
        assert_normalized_stream_failure(
            LlmResponseChunk::DecodeError {
                message: "provider response body".into(),
            },
            "response_translation",
            None,
        )
        .await;
    }
}
