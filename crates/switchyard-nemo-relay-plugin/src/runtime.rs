// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;
use std::sync::{Arc, Mutex};

use futures_util::{Stream, StreamExt};
use nemo_relay_plugin::{
    Json, LlmRequest as RelayRequest, LogSeverity, MetricKind, MetricMeasurement, MetricValueType,
    PluginRuntime,
};
use serde_json::{Map, json};
use switchyard_llm_client::{LlmCallObservation, RunObservation, RunObserver};
use switchyard_protocol::{LlmResponse, Metadata, Request, Response, WireFormat};
use switchyard_runner::{Route, RouteErrorSummary, Runner, stream_error_summary};
use switchyard_translation::{TranslationEngine, encode_stream};

use crate::config::SwitchyardConfig;
use crate::translation;

#[derive(Debug)]
pub(crate) struct RoutingMark {
    pub(crate) name: String,
    pub(crate) data: Json,
    pub(crate) metadata: Json,
    pub(crate) severity: Option<LogSeverity>,
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

    pub(crate) fn manages(&self, request: &Request) -> bool {
        request
            .llm_request
            .model
            .as_deref()
            .is_some_and(|model| self.runner.route(model).is_some())
    }

    pub(crate) fn decode_request(
        &self,
        inbound: WireFormat,
        request: &RelayRequest,
        streaming: bool,
    ) -> Result<Request, String> {
        let mut llm_request = translation::decode_request(&self.translation, inbound, request)?;
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
        metadata.wire_format = Some(inbound);
        Ok(Request {
            llm_request,
            raw_request: Some(request.content.clone()),
            metadata: Some(metadata),
        })
    }

    pub(crate) async fn execute_buffered(
        &self,
        inbound: WireFormat,
        request: Request,
    ) -> Execution<Json> {
        let Execution { result, mut events } = self.execute(request).await;
        let (result, finalization_failed) = match result {
            Ok(response) => {
                let result = finalize_buffered_response(&self.translation, inbound, response);
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
        let Execution { result, mut events } = self.execute(request).await;
        let (result, finalization_failed) = match result {
            Ok(response) => {
                let metadata = events
                    .iter()
                    .find_map(|event| match event {
                        RoutingEvent::Mark(mark) => Some(mark.metadata.clone()),
                        RoutingEvent::Metric(_) => None,
                    })
                    .unwrap_or_else(|| Json::Object(Map::new()));
                let result = returned_events(response, inbound, metadata, emit_event);
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

    async fn execute(&self, request: Request) -> Execution<Response> {
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
        if let Err(error) = route.check_caller_format(metadata_wire_format(&request)) {
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
        events.push(failure_metric(failure_kind, None, None, metadata));
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
        RoutingEvent::Mark(mark) => runtime
            .emit_mark_with_options(
                &mark.name,
                Some(&mark.data),
                Some(&mark.metadata),
                None,
                mark.severity,
            )
            .map_err(|error| ("routing mark", mark.name, error)),
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

fn metadata_wire_format(request: &Request) -> WireFormat {
    request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.wire_format)
        .expect("decoded Relay requests always carry a wire format")
}

fn finalize_buffered_response(
    translation_engine: &TranslationEngine,
    inbound: WireFormat,
    response: Response,
) -> Result<Json, String> {
    let LlmResponse::Agg(response) = response.llm_response else {
        return Err("Switchyard returned a stream for a buffered request".into());
    };
    translation::encode_response(translation_engine, inbound, &response)
}

fn returned_events(
    response: Response,
    inbound: WireFormat,
    metadata: Json,
    emit_event: RoutingEventEmitter,
) -> Result<ReturnedEventStream, String> {
    let served_model = response.served_model().cloned();
    let chunks = match response.llm_response {
        LlmResponse::Agg(response) => response.into_stream(),
        LlmResponse::Stream(chunks) => chunks,
    };
    let chunks = Box::pin(chunks.map(move |item| {
        if let Err(error) = &item {
            for event in route_execution_error_events(
                &stream_error_summary(error, served_model.as_ref()),
                metadata.clone(),
            ) {
                emit_event(event);
            }
        }
        item
    }));
    let events = encode_stream(chunks, inbound, None)
        .map_err(|error| format!("Switchyard response stream setup failed: {error}"))?;
    Ok(Box::pin(events.map(|item| {
        item.map_err(|error| format!("Switchyard response stream failed: {error}"))
    })))
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
                    "target_model": call.selected_model.as_str(),
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
    use switchyard_protocol::{
        LlmClientError, LlmResponseStreamEvent, ModelId, Usage, text_request,
    };
    use switchyard_runner::{AlgorithmSpec, ModelCapabilities, RunnerError};

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
            Vec::new(),
        );
        SwitchyardRuntime {
            runner: Runner::new(vec![(ModelId::from(model), route)]),
            translation: TranslationEngine::default(),
        }
    }

    #[test]
    fn only_configured_route_models_are_managed() {
        let runtime = runtime_for("switchyard");
        let configured = Request {
            llm_request: text_request(Some("switchyard".into()), "hello"),
            ..Request::default()
        };
        let other = Request {
            llm_request: text_request(Some("other".into()), "hello"),
            ..Request::default()
        };

        assert!(runtime.manages(&configured));
        assert!(!runtime.manages(&other));
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
            json!({"session_id": "session"}),
            Arc::new(move |mark| emitted.lock().unwrap().push(mark)),
        )
        .expect("stream setup should succeed");

        let events = stream.collect::<Vec<_>>().await;
        assert!(events[0].is_err());
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
}
