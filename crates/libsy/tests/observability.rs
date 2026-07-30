// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the crate's observability layer: OpenTelemetry metrics
//! recorded through the global meter provider, and `tracing` spans/logs captured
//! by a global subscriber.
//!
//! Both telemetry sinks are process-global, so they are installed exactly once
//! for this test binary and every assertion filters by a test-unique algorithm
//! and model name. Counters are cumulative across flushes; the helpers take the
//! latest (max) matching data point.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry::{Array as OtelArray, Value as OtelValue};
use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData, ResourceMetrics};
use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider, SpanData};
use parking_lot::Mutex;
use serde_json::json;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Subscriber};
use tracing_opentelemetry::OpenTelemetryLayer;
use tracing_subscriber::Layer;
use tracing_subscriber::layer::{Context as LayerContext, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;

use switchyard_libsy::{
    Algorithm, Driver, LibsyError, LlmTarget, LlmTargetSet, LlmTaskClassifier, Step,
    TaskClassifierConfig,
};
use switchyard_protocol::{
    Context, Decision, LlmResponse, Metadata, Request, Response, RoutedLlmClient, Usage,
};
use switchyard_protocol::{
    LlmClientError, LlmResponseChunk, LlmResponseStreamEvent, StopReason, text_request,
    text_response,
};

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
struct TestError(&'static str);

fn test_error(message: &'static str) -> LibsyError {
    LibsyError::external("test", TestError(message))
}

/// One captured span: its name, contextual parent span name, and fields
/// (creation-time fields merged with later `Span::record` updates).
#[derive(Clone, Debug, Default)]
struct SpanRecord {
    name: String,
    parent: Option<String>,
    fields: BTreeMap<String, String>,
}

/// One captured event (log line): its target and fields, `message` included.
#[derive(Clone, Debug, Default)]
struct EventRecord {
    target: String,
    level: String,
    fields: BTreeMap<String, String>,
}

/// Shared store the capture layer writes into and tests read from.
#[derive(Clone, Default)]
struct CaptureStore {
    spans: Arc<Mutex<BTreeMap<u64, SpanRecord>>>,
    events: Arc<Mutex<Vec<EventRecord>>>,
}

impl CaptureStore {
    fn spans(&self) -> Vec<SpanRecord> {
        self.spans.lock().values().cloned().collect()
    }

    fn events(&self) -> Vec<EventRecord> {
        self.events.lock().clone()
    }
}

/// Renders every field type into a string map so assertions can use `contains`.
struct FieldVisitor<'a>(&'a mut BTreeMap<String, String>);

impl Visit for FieldVisitor<'_> {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.0
            .insert(field.name().to_string(), format!("{value:?}"));
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.0.insert(field.name().to_string(), value.to_string());
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.0.insert(field.name().to_string(), value.to_string());
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.0.insert(field.name().to_string(), value.to_string());
    }

    fn record_f64(&mut self, field: &Field, value: f64) {
        self.0.insert(field.name().to_string(), value.to_string());
    }
}

/// Subscriber layer capturing spans (with contextual parents and recorded
/// fields) and events into a [`CaptureStore`].
struct CaptureLayer {
    store: CaptureStore,
}

impl<S> Layer<S> for CaptureLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: LayerContext<'_, S>) {
        let mut fields = BTreeMap::new();
        attrs.record(&mut FieldVisitor(&mut fields));
        // Resolve the parent the same way tracing does: an explicit parent wins,
        // otherwise the contextually current span (if any) at creation time.
        let parent = if let Some(parent_id) = attrs.parent() {
            ctx.span(parent_id).map(|span| span.name().to_string())
        } else if attrs.is_contextual() {
            ctx.lookup_current().map(|span| span.name().to_string())
        } else {
            None
        };
        self.store.spans.lock().insert(
            id.into_u64(),
            SpanRecord {
                name: attrs.metadata().name().to_string(),
                parent,
                fields,
            },
        );
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, _ctx: LayerContext<'_, S>) {
        if let Some(record) = self.store.spans.lock().get_mut(&id.into_u64()) {
            values.record(&mut FieldVisitor(&mut record.fields));
        }
    }

    fn on_event(&self, event: &Event<'_>, _ctx: LayerContext<'_, S>) {
        let mut fields = BTreeMap::new();
        event.record(&mut FieldVisitor(&mut fields));
        self.store.events.lock().push(EventRecord {
            target: event.metadata().target().to_string(),
            level: event.metadata().level().to_string(),
            fields,
        });
    }
}

/// Installs the process-global telemetry sinks once: an in-memory OTel metric
/// pipeline behind the global meter provider, and the capture layer as the
/// global tracing subscriber.
type Telemetry = (
    CaptureStore,
    InMemoryMetricExporter,
    SdkMeterProvider,
    InMemorySpanExporter,
    SdkTracerProvider,
);

fn telemetry() -> &'static Telemetry {
    static TELEMETRY: OnceLock<Telemetry> = OnceLock::new();
    TELEMETRY.get_or_init(|| {
        let exporter = InMemoryMetricExporter::default();
        let reader = PeriodicReader::builder(exporter.clone()).build();
        let provider = SdkMeterProvider::builder().with_reader(reader).build();
        opentelemetry::global::set_meter_provider(provider.clone());
        switchyard_libsy::initialize_metrics();

        let span_exporter = InMemorySpanExporter::default();
        let tracer_provider = SdkTracerProvider::builder()
            .with_simple_exporter(span_exporter.clone())
            .build();
        let tracer = tracer_provider.tracer("switchyard-observability-test");
        let store = CaptureStore::default();
        let otel_layer: OpenTelemetryLayer<_, _> =
            tracing_opentelemetry::layer().with_tracer(tracer);
        let subscriber = tracing_subscriber::registry()
            .with(CaptureLayer {
                store: store.clone(),
            })
            .with(otel_layer);
        if tracing::subscriber::set_global_default(subscriber).is_err() {
            panic!("a global tracing subscriber was already installed in this test binary");
        }
        (store, exporter, provider, span_exporter, tracer_provider)
    })
}

/// The tests in this file must not overlap because metrics are global.
/// There is no Rust/cargo-native way of saying this (people use `serial_test` crate) so use a
/// lock.
/// Each file in `tests/` (integration tests) runs as a separate test process, so we are not
/// concerned with interactions with tests in other files.
fn serialize_test() -> &'static tokio::sync::Mutex<()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

/// Flushes the metric pipeline and returns every exported snapshot.
fn flushed_metrics(
    exporter: &InMemoryMetricExporter,
    provider: &SdkMeterProvider,
) -> Vec<ResourceMetrics> {
    if let Err(error) = provider.force_flush() {
        panic!("force_flush failed: {error}");
    }
    match exporter.get_finished_metrics() {
        Ok(metrics) => metrics,
        Err(error) => panic!("get_finished_metrics failed: {error}"),
    }
}

/// True when the data point carries every wanted `key=value` attribute.
fn attributes_match<'a>(
    mut attributes: impl Iterator<Item = &'a opentelemetry::KeyValue>,
    wanted: &[(&str, &str)],
) -> bool {
    let present: Vec<(String, String)> = attributes
        .by_ref()
        .map(|kv| (kv.key.as_str().to_string(), kv.value.as_str().to_string()))
        .collect();
    wanted
        .iter()
        .all(|(key, value)| present.iter().any(|(k, v)| k == key && v == value))
}

/// Latest (max) value for a `switchyard`-scoped metric across snapshots, with
/// `extract` pulling the matching data points' values out of one metric's
/// aggregated data. Counters and histogram counts are cumulative, so the max
/// across snapshots is the most recent value.
fn latest_metric_value(
    snapshots: &[ResourceMetrics],
    name: &str,
    extract: impl Fn(&AggregatedMetrics) -> Vec<u64>,
) -> Option<u64> {
    snapshots
        .iter()
        .flat_map(|snapshot| snapshot.scope_metrics())
        .filter(|scope| scope.scope().name() == "switchyard")
        .flat_map(|scope| scope.metrics())
        .filter(|metric| metric.name() == name)
        .flat_map(|metric| extract(metric.data()))
        .max()
}

/// Latest cumulative value of a `u64` counter for the given attribute set.
fn u64_counter_value(
    snapshots: &[ResourceMetrics],
    name: &str,
    wanted: &[(&str, &str)],
) -> Option<u64> {
    latest_metric_value(snapshots, name, |data| match data {
        AggregatedMetrics::U64(MetricData::Sum(sum)) => sum
            .data_points()
            .filter(|point| attributes_match(point.attributes(), wanted))
            .map(|point| point.value())
            .collect(),
        _ => Vec::new(),
    })
}

/// Latest cumulative sample count of an `f64` histogram for the attribute set.
fn f64_histogram_count(
    snapshots: &[ResourceMetrics],
    name: &str,
    wanted: &[(&str, &str)],
) -> Option<u64> {
    latest_metric_value(snapshots, name, |data| match data {
        AggregatedMetrics::F64(MetricData::Histogram(histogram)) => histogram
            .data_points()
            .filter(|point| attributes_match(point.attributes(), wanted))
            .map(|point| point.count())
            .collect(),
        _ => Vec::new(),
    })
}

/// Latest cumulative sample sum of an `f64` histogram, in whole milliseconds.
fn f64_histogram_sum_ms(
    snapshots: &[ResourceMetrics],
    name: &str,
    wanted: &[(&str, &str)],
) -> Option<u64> {
    latest_metric_value(snapshots, name, |data| match data {
        AggregatedMetrics::F64(MetricData::Histogram(histogram)) => histogram
            .data_points()
            .filter(|point| attributes_match(point.attributes(), wanted))
            .map(|point| point.sum() as u64)
            .collect(),
        _ => Vec::new(),
    })
}

/// Latest value of a `u64` observable gauge.
fn u64_gauge_value(snapshots: &[ResourceMetrics], name: &str) -> Option<u64> {
    latest_metric_value(snapshots, name, |data| match data {
        AggregatedMetrics::U64(MetricData::Gauge(gauge)) => {
            gauge.data_points().map(|point| point.value()).collect()
        }
        _ => Vec::new(),
    })
}

/// Decision with a fixed model and reasoning string.
struct StaticDecision {
    model: String,
    reasoning: String,
}

impl Decision for StaticDecision {
    fn selected_model(&self) -> &str {
        &self.model
    }
    fn reasoning(&self) -> Option<&str> {
        Some(&self.reasoning)
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Client that answers every call with a fixed token [`Usage`].
struct UsageClient {
    usage: Usage,
}

/// Client that returns a weak classifier verdict. The delays let a test tell
/// classifier time apart from routed-call time.
struct ClassifierClient {
    classifier_delay: Duration,
    routed_delay: Duration,
}

#[async_trait]
impl RoutedLlmClient for ClassifierClient {
    async fn call(
        &self,
        _ctx: Context,
        _request: Request,
        decision: Arc<dyn Decision>,
    ) -> Result<Response, LlmClientError> {
        let model = decision.selected_model().to_string();
        let completion = if decision.is_routed_call() {
            tokio::time::sleep(self.routed_delay).await;
            "routed response"
        } else {
            tokio::time::sleep(self.classifier_delay).await;
            r#"{"recommended_route":"weak","p_solve":0.9,"confidence":0.9,"abstain":false,"capability_boundary":"supported","primary_rule":"SUP-1","crux":"bounded task"}"#
        };
        Ok(Response {
            llm_response: LlmResponse::Agg(text_response(Some(model), completion)),
            metadata: None,
        })
    }
}

#[async_trait]
impl RoutedLlmClient for UsageClient {
    async fn call(
        &self,
        _ctx: Context,
        _request: Request,
        decision: Arc<dyn Decision>,
    ) -> Result<Response, switchyard_protocol::LlmClientError> {
        let mut response = text_response(
            Some(decision.selected_model().to_string()),
            "observed response",
        );
        response.id = Some("obs-response-1".to_string());
        response.usage = self.usage.clone();
        response.outputs[0].stop_reason = Some(StopReason::EndTurn);
        Ok(Response {
            llm_response: LlmResponse::Agg(response),
            metadata: None,
        })
    }
}

/// Publishes one decision for the first target, then calls it — the smallest
/// algorithm exercising both instrumented driver paths.
struct SingleCallAlgo {
    name: String,
    target_set: LlmTargetSet,
}

#[async_trait]
impl Algorithm for SingleCallAlgo {
    fn name(&self) -> &str {
        &self.name
    }

    async fn create_run_task(
        self: Arc<Self>,
        ctx: Context,
        driver: Driver,
        request: Request,
    ) -> switchyard_libsy::Result<Response> {
        let target = self
            .target_set
            .targets()
            .first()
            .ok_or(LibsyError::NoTargets)?
            .clone();
        let decision: Arc<dyn Decision> = Arc::new(StaticDecision {
            reasoning: format!("picked '{}'", target.semantic_name),
            model: target.semantic_name.clone(),
        });
        driver.info(ctx.clone(), decision.clone()).await?;
        driver
            .call_llm_target(ctx, &target, request, decision)
            .await
    }
}

fn request_with_metadata(session_id: &str, correlation_id: &str) -> Request {
    Request {
        llm_request: text_request(Some("auto".to_string()), "hi"),
        raw_request: None,
        metadata: Some(Metadata {
            session_id: Some(session_id.to_string()),
            correlation_id: Some(correlation_id.to_string()),
            extra_metadata: Some(BTreeMap::from([(
                "tenant".to_string(),
                "obs-tenant-1".to_string(),
            )])),
            ..Metadata::default()
        }),
    }
}

fn algo(name: &str, model: &str, client: Option<Arc<dyn RoutedLlmClient>>) -> Arc<dyn Algorithm> {
    Arc::new(SingleCallAlgo {
        name: name.to_string(),
        target_set: LlmTargetSet::new(vec![LlmTarget {
            semantic_name: model.to_string(),
            llm_client: client,
        }]),
    })
}

fn find_span(spans: &[SpanRecord], name: &str, field: &str, value: &str) -> SpanRecord {
    match spans
        .iter()
        .find(|span| span.name == name && span.fields.get(field).map(String::as_str) == Some(value))
    {
        Some(span) => span.clone(),
        None => panic!("no '{name}' span with {field}={value} in {spans:?}"),
    }
}

fn find_otel_span(exporter: &InMemorySpanExporter, name: &str, model: &str) -> SpanData {
    let spans = match exporter.get_finished_spans() {
        Ok(spans) => spans,
        Err(error) => panic!("failed to read exported spans: {error}"),
    };
    match spans.iter().find(|span| {
        span.name == name
            && span.attributes.iter().any(|attribute| {
                attribute.key.as_str() == "gen_ai.request.model"
                    && attribute.value.as_str() == model
            })
    }) {
        Some(span) => span.clone(),
        None => {
            let available = spans
                .iter()
                .map(|span| {
                    let model = span
                        .attributes
                        .iter()
                        .find(|attribute| attribute.key.as_str() == "gen_ai.request.model")
                        .map(|attribute| attribute.value.as_str().into_owned());
                    (span.name.to_string(), model)
                })
                .collect::<Vec<_>>();
            panic!("no exported '{name}' span for model {model}; available: {available:?}")
        }
    }
}

fn otel_attribute<'a>(span: &'a SpanData, key: &str) -> Option<&'a OtelValue> {
    span.attributes
        .iter()
        .find(|attribute| attribute.key.as_str() == key)
        .map(|attribute| &attribute.value)
}

#[tokio::test]
async fn successful_run_records_metrics_spans_and_decision_log() -> switchyard_libsy::Result<()> {
    let _guard = serialize_test().lock().await;
    let (store, exporter, provider, span_exporter, _) = telemetry();
    const ALGO: &str = "obs-success-algo";
    const MODEL: &str = "obs-success-model";
    let before = flushed_metrics(exporter, provider);
    let total_requests_before =
        u64_gauge_value(&before, "switchyard.total_requests").unwrap_or_default();
    let total_errors_before =
        u64_gauge_value(&before, "switchyard.total_errors").unwrap_or_default();

    let client = Arc::new(UsageClient {
        usage: Usage {
            input_tokens: Some(11),
            output_tokens: Some(7),
            total_tokens: Some(25),
            reasoning_tokens: Some(2),
            cache: Usage::cache_details(Some(3), Some(4)),
        },
    }) as Arc<dyn RoutedLlmClient>;
    let mut request = request_with_metadata("obs-session-1", "obs-corr-1");
    request.llm_request.sampling.temperature = Some(0.25);
    request.llm_request.sampling.top_p = Some(0.9);
    request.llm_request.sampling.top_k = Some(40);
    request.llm_request.output.max_output_tokens = Some(512);
    request.llm_request.output.response_format = Some(json!({"type": "json_schema"}));
    request.llm_request.reasoning.effort = Some("high".to_string());
    let (trace, _response) = algo(ALGO, MODEL, Some(client))
        .run(Context::default(), request)
        .await?;
    assert_eq!(trace.len(), 1);

    // Metrics: run/call counters and latency histograms keyed by algorithm,
    // plus one published decision.
    let snapshots = flushed_metrics(exporter, provider);
    let run_attrs = [("algorithm", ALGO), ("outcome", "ok")];
    let call_attrs = [
        ("algorithm", ALGO),
        ("selected_model", MODEL),
        ("outcome", "ok"),
    ];
    let token_attrs = [("algorithm", ALGO), ("selected_model", MODEL)];
    assert_eq!(
        u64_counter_value(&snapshots, "switchyard.runs", &run_attrs),
        Some(1)
    );
    assert_eq!(
        u64_counter_value(&snapshots, "switchyard.llm_calls", &call_attrs),
        Some(1)
    );
    assert_eq!(
        f64_histogram_count(&snapshots, "switchyard.run_duration_ms", &run_attrs),
        Some(1)
    );
    assert_eq!(
        f64_histogram_count(&snapshots, "switchyard.llm_call_duration_ms", &call_attrs),
        Some(1)
    );
    assert_eq!(
        u64_counter_value(&snapshots, "switchyard.decisions", &token_attrs),
        Some(1)
    );
    let routed_attrs = [("model", MODEL)];
    assert_eq!(
        u64_counter_value(&snapshots, "switchyard.requests", &routed_attrs),
        Some(1)
    );
    assert_eq!(
        f64_histogram_count(
            &snapshots,
            "switchyard.model_call_latency_ms",
            &routed_attrs
        ),
        Some(1)
    );
    assert_eq!(
        u64_gauge_value(&snapshots, "switchyard.total_requests"),
        Some(total_requests_before + 1)
    );
    assert_eq!(
        u64_gauge_value(&snapshots, "switchyard.total_errors"),
        Some(total_errors_before)
    );
    // One overhead observation per run, keyed by algorithm alone.
    assert_eq!(
        f64_histogram_count(
            &snapshots,
            "switchyard.routing_overhead_ms",
            &[("algorithm", ALGO)]
        ),
        Some(1)
    );

    // Spans: one run span carrying the correlation ids and outcome, one child
    // llm_call span carrying the selection, outcome, and token counts.
    let spans = store.spans();
    let run_span = find_span(&spans, "libsy.run", "algorithm", ALGO);
    assert_eq!(run_span.parent, None);
    assert_eq!(
        run_span.fields.get("session_id").map(String::as_str),
        Some("obs-session-1")
    );
    assert_eq!(
        run_span.fields.get("session.id").map(String::as_str),
        Some("obs-session-1")
    );
    assert_eq!(
        run_span
            .fields
            .get("switchyard.algorithm")
            .map(String::as_str),
        Some(ALGO)
    );
    assert_eq!(
        run_span.fields.get("switchyard.route").map(String::as_str),
        Some("auto")
    );
    assert_eq!(
        run_span.fields.get("correlation_id").map(String::as_str),
        Some("obs-corr-1")
    );
    assert_eq!(
        run_span.fields.get("outcome").map(String::as_str),
        Some("ok")
    );
    // Host-defined labels ride in generically via Metadata.extra_metadata.
    assert!(
        run_span
            .fields
            .get("extra_metadata")
            .is_some_and(|extra| extra.contains("tenant") && extra.contains("obs-tenant-1"))
    );

    // The default-client serve inside `run` gets its own client-call span.
    let client_span = find_span(&spans, "libsy.client_call", "selected_model", MODEL);
    // Host-side span: `run`'s serve loop creates it outside the algorithm's
    // spans, so it has no libsy parent. This pins the instrument idiom
    // (`#[tracing::instrument]` / `Future::instrument`) — an `Entered` guard
    // held across the offload `.await` would leave `libsy.llm_call` entered
    // on the thread and leak it in as the parent.
    assert_eq!(client_span.parent.as_deref(), None);
    assert_eq!(
        client_span.fields.get("algorithm").map(String::as_str),
        Some(ALGO)
    );
    assert_eq!(
        client_span.fields.get("outcome").map(String::as_str),
        Some("ok")
    );
    for (field, value) in [
        ("otel.kind", "client"),
        ("switchyard.algorithm", ALGO),
        ("otel.name", "chat obs-success-model"),
        ("gen_ai.operation.name", "chat"),
        ("gen_ai.request.model", MODEL),
        ("gen_ai.request.temperature", "0.25"),
        ("gen_ai.request.top_p", "0.9"),
        ("gen_ai.request.top_k", "40"),
        ("gen_ai.request.max_tokens", "512"),
        ("gen_ai.request.reasoning.level", "high"),
        ("gen_ai.output.type", "json"),
        ("gen_ai.conversation.id", "obs-session-1"),
        ("gen_ai.response.id", "obs-response-1"),
        ("gen_ai.response.model", MODEL),
        ("gen_ai.usage.input_tokens", "18"),
        ("gen_ai.usage.output_tokens", "7"),
        ("gen_ai.usage.cache_read.input_tokens", "3"),
        ("gen_ai.usage.cache_creation.input_tokens", "4"),
        ("gen_ai.usage.reasoning.output_tokens", "2"),
    ] {
        assert_eq!(
            client_span.fields.get(field).map(String::as_str),
            Some(value),
            "unexpected {field}"
        );
    }
    assert_eq!(client_span.fields.get("gen_ai.request.stream"), None);

    let otel_span = find_otel_span(span_exporter, "chat obs-success-model", MODEL);
    assert!(matches!(
        otel_attribute(&otel_span, "gen_ai.response.finish_reasons"),
        Some(OtelValue::Array(OtelArray::String(reasons)))
            if reasons.len() == 1 && reasons[0].as_str() == "end_turn"
    ));
    assert_eq!(
        otel_attribute(&otel_span, "gen_ai.request.max_tokens"),
        Some(&OtelValue::I64(512))
    );
    assert_eq!(
        otel_attribute(&otel_span, "gen_ai.usage.input_tokens"),
        Some(&OtelValue::I64(18))
    );

    let call_span = find_span(&spans, "libsy.llm_call", "selected_model", MODEL);
    assert_eq!(call_span.parent.as_deref(), Some("libsy.run"));
    assert_eq!(
        call_span.fields.get("algorithm").map(String::as_str),
        Some(ALGO)
    );
    assert_eq!(
        call_span.fields.get("outcome").map(String::as_str),
        Some("ok")
    );
    assert_eq!(
        call_span.fields.get("input_tokens").map(String::as_str),
        Some("11")
    );
    assert_eq!(
        call_span.fields.get("output_tokens").map(String::as_str),
        Some("7")
    );
    assert_eq!(
        call_span.fields.get("total_tokens").map(String::as_str),
        Some("25")
    );
    assert_eq!(
        call_span.fields.get("reasoning_tokens").map(String::as_str),
        Some("2")
    );

    // Structured debug event: the published decision with its reasoning.
    let events = store.events();
    assert!(
        events.iter().any(|event| {
            event.target == "libsy"
                && event.level == "DEBUG"
                && event.fields.get("selected_model").map(String::as_str) == Some(MODEL)
                && event
                    .fields
                    .get("reasoning")
                    .is_some_and(|reasoning| reasoning.contains("picked"))
                && event
                    .fields
                    .get("message")
                    .is_some_and(|message| message.contains("routing decision"))
        }),
        "no routing-decision log event for {MODEL} in {events:?}"
    );
    Ok(())
}

/// A streamed response keeps the client span available until terminal usage arrives.
struct StreamingUsageClient;

#[async_trait]
impl RoutedLlmClient for StreamingUsageClient {
    async fn call(
        &self,
        _ctx: Context,
        _request: Request,
        decision: Arc<dyn Decision>,
    ) -> Result<Response, LlmClientError> {
        let usage = Usage {
            input_tokens: Some(13),
            output_tokens: Some(5),
            cache: Usage::cache_details(Some(8), None),
            ..Usage::default()
        };
        let chunks = vec![Ok(LlmResponseStreamEvent::new(vec![
            LlmResponseChunk::MessageStart {
                id: Some("obs-stream-response".to_string()),
                model: Some(decision.selected_model().to_string()),
            },
            LlmResponseChunk::Usage(usage),
            LlmResponseChunk::MessageStop {
                reason: Some("end_turn".to_string()),
            },
        ]))];
        Ok(Response {
            llm_response: LlmResponse::Stream(Box::pin(futures::stream::iter(chunks))),
            metadata: None,
        })
    }
}

struct TimeoutClient;

#[async_trait]
impl RoutedLlmClient for TimeoutClient {
    async fn call(
        &self,
        _ctx: Context,
        _request: Request,
        _decision: Arc<dyn Decision>,
    ) -> Result<Response, LlmClientError> {
        Err(LlmClientError::Timeout {
            source: Box::new(TestError("upstream timed out")),
        })
    }
}

#[tokio::test]
async fn streamed_usage_updates_the_client_call_span() -> switchyard_libsy::Result<()> {
    let _guard = serialize_test().lock().await;
    let (store, _, _, span_exporter, _) = telemetry();
    const ALGO: &str = "obs-stream-algo";
    const MODEL: &str = "obs-stream-model";
    let client = Arc::new(StreamingUsageClient) as Arc<dyn RoutedLlmClient>;
    let mut request = request_with_metadata("obs-stream-session", "obs-stream-corr");
    request.llm_request.stream = true;
    let (_, response) = algo(ALGO, MODEL, Some(client))
        .run(Context::default(), request)
        .await?;
    let LlmResponse::Stream(mut stream) = response.llm_response else {
        return Err(test_error("expected a streamed response"));
    };
    while let Some(item) = stream.next().await {
        if let Err(error) = item {
            panic!("unexpected stream error: {error}");
        }
    }

    let spans = store.spans();
    let client_span = find_span(&spans, "libsy.client_call", "selected_model", MODEL);
    for (field, value) in [
        ("otel.name", "chat obs-stream-model"),
        ("gen_ai.request.stream", "true"),
        ("gen_ai.response.id", "obs-stream-response"),
        ("gen_ai.response.model", MODEL),
        ("gen_ai.usage.input_tokens", "21"),
        ("gen_ai.usage.output_tokens", "5"),
        ("gen_ai.usage.cache_read.input_tokens", "8"),
    ] {
        assert_eq!(
            client_span.fields.get(field).map(String::as_str),
            Some(value),
            "unexpected {field}"
        );
    }
    let otel_span = find_otel_span(span_exporter, "chat obs-stream-model", MODEL);
    assert!(matches!(
        otel_attribute(&otel_span, "gen_ai.response.finish_reasons"),
        Some(OtelValue::Array(OtelArray::String(reasons)))
            if reasons.len() == 1 && reasons[0].as_str() == "end_turn"
    ));
    Ok(())
}

#[tokio::test]
async fn dropped_stream_records_cancelled_outcome() -> switchyard_libsy::Result<()> {
    let _guard = serialize_test().lock().await;
    let (store, _, _, _, _) = telemetry();
    const ALGO: &str = "obs-cancelled-stream-algo";
    const MODEL: &str = "obs-cancelled-stream-model";
    let client = Arc::new(StreamingUsageClient) as Arc<dyn RoutedLlmClient>;
    let mut request = request_with_metadata("obs-cancelled-session", "obs-cancelled-corr");
    request.llm_request.stream = true;
    let (_, response) = algo(ALGO, MODEL, Some(client))
        .run(Context::default(), request)
        .await?;
    let LlmResponse::Stream(stream) = response.llm_response else {
        return Err(test_error("expected a streamed response"));
    };
    drop(stream);

    let spans = store.spans();
    let client_span = find_span(&spans, "libsy.client_call", "selected_model", MODEL);
    assert_eq!(
        client_span.fields.get("outcome").map(String::as_str),
        Some("cancelled")
    );
    Ok(())
}

#[tokio::test]
async fn typed_client_failure_records_semantic_error_type() {
    let _guard = serialize_test().lock().await;
    let (store, _, _, _, _) = telemetry();
    const ALGO: &str = "obs-timeout-algo";
    const MODEL: &str = "obs-timeout-model";
    let result = algo(
        ALGO,
        MODEL,
        Some(Arc::new(TimeoutClient) as Arc<dyn RoutedLlmClient>),
    )
    .run(
        Context::default(),
        request_with_metadata("obs-timeout-session", "obs-timeout-corr"),
    )
    .await;
    assert!(matches!(
        result,
        Err(LibsyError::ClientCall {
            source: LlmClientError::Timeout { .. },
            ..
        })
    ));

    let spans = store.spans();
    let client_span = find_span(&spans, "libsy.client_call", "selected_model", MODEL);
    assert_eq!(
        client_span.fields.get("error.type").map(String::as_str),
        Some("timeout")
    );
}

#[tokio::test]
async fn failed_call_records_error_outcome_and_warn_logs() -> switchyard_libsy::Result<()> {
    let _guard = serialize_test().lock().await;
    let (store, exporter, provider, _, _) = telemetry();
    const ALGO: &str = "obs-failure-algo";
    const MODEL: &str = "obs-failure-model";
    let before = flushed_metrics(exporter, provider);
    let total_requests_before =
        u64_gauge_value(&before, "switchyard.total_requests").unwrap_or_default();
    let total_errors_before =
        u64_gauge_value(&before, "switchyard.total_errors").unwrap_or_default();

    // Client-less target: the call is offloaded and we fail it by hand.
    let stream = algo(ALGO, MODEL, None).run_stream(
        Context::default(),
        request_with_metadata("obs-session-2", "obs-corr-2"),
        None,
    );
    tokio::pin!(stream);

    let mut saw_error_step = false;
    while let Some(step) = stream.next().await {
        match step {
            Ok(Step::CallLlm(call)) => {
                call.respond(Err(test_error("synthetic upstream failure")))?;
            }
            Ok(Step::Decision(_)) => {}
            Ok(Step::ReturnToAgent(_)) => {
                return Err(test_error("expected the failed call to fail the run"));
            }
            Err(_) => saw_error_step = true,
        }
    }
    assert!(
        saw_error_step,
        "expected an error step from the failed call"
    );

    // Metrics: the call and the run both count under outcome=error.
    let snapshots = flushed_metrics(exporter, provider);
    let run_attrs = [("algorithm", ALGO), ("outcome", "error")];
    let call_attrs = [
        ("algorithm", ALGO),
        ("selected_model", MODEL),
        ("outcome", "error"),
    ];
    assert_eq!(
        u64_counter_value(&snapshots, "switchyard.runs", &run_attrs),
        Some(1)
    );
    assert_eq!(
        u64_counter_value(&snapshots, "switchyard.llm_calls", &call_attrs),
        Some(1)
    );
    assert_eq!(
        u64_counter_value(&snapshots, "switchyard.errors", &[("model", MODEL)]),
        Some(1)
    );
    assert_eq!(
        u64_gauge_value(&snapshots, "switchyard.total_requests"),
        Some(total_requests_before + 1)
    );
    assert_eq!(
        u64_gauge_value(&snapshots, "switchyard.total_errors"),
        Some(total_errors_before + 1)
    );
    // Nothing was served, so there is nothing to measure routing against.
    assert_eq!(
        f64_histogram_count(
            &snapshots,
            "switchyard.routing_overhead_ms",
            &[("algorithm", ALGO)]
        ),
        None
    );

    // Spans: both spans carry outcome=error and the propagated error text.
    let spans = store.spans();
    let run_span = find_span(&spans, "libsy.run", "algorithm", ALGO);
    assert_eq!(
        run_span.fields.get("outcome").map(String::as_str),
        Some("error")
    );
    assert!(
        run_span
            .fields
            .get("error")
            .is_some_and(|error| error.contains("synthetic upstream failure"))
    );
    let call_span = find_span(&spans, "libsy.llm_call", "selected_model", MODEL);
    assert_eq!(
        call_span.fields.get("outcome").map(String::as_str),
        Some("error")
    );

    // Structured logs warn once for the failed call and failed run.
    let events = store.events();
    assert!(
        events.iter().any(|event| {
            event.target == "libsy"
                && event.level == "WARN"
                && event.fields.get("selected_model").map(String::as_str) == Some(MODEL)
                && event
                    .fields
                    .get("message")
                    .is_some_and(|message| message.contains("model call failed"))
        }),
        "no call-failure log for {MODEL} in {events:?}"
    );
    assert!(
        events.iter().any(|event| {
            event.target == "libsy"
                && event.level == "WARN"
                && event.fields.get("algorithm").map(String::as_str) == Some(ALGO)
                && event
                    .fields
                    .get("message")
                    .is_some_and(|message| message.contains("algorithm run failed"))
        }),
        "no run-failure log for {ALGO} in {events:?}"
    );
    Ok(())
}

#[tokio::test]
async fn classifier_metrics_count_only_the_final_routed_call() -> switchyard_libsy::Result<()> {
    let _guard = serialize_test().lock().await;
    let (_store, exporter, provider, _, _) = telemetry();
    let before = flushed_metrics(exporter, provider);
    let total_requests_before =
        u64_gauge_value(&before, "switchyard.total_requests").unwrap_or_default();

    let client = Arc::new(ClassifierClient {
        classifier_delay: Duration::from_millis(60),
        routed_delay: Duration::from_millis(200),
    });
    let target = |name: &str| LlmTarget {
        semantic_name: name.to_string(),
        llm_client: Some(client.clone()),
    };
    let targets = LlmTargetSet::new(vec![target("weak"), target("strong")]);
    let weak = targets.get_target("weak")?;
    let strong = targets.get_target("strong")?;
    let router = Arc::new(LlmTaskClassifier::new(
        target("classifier"),
        weak,
        strong,
        TaskClassifierConfig {
            base_threshold: 0.5,
            ..TaskClassifierConfig::default()
        },
    )?);

    let (trace, _response) = router
        .run(
            Context::default(),
            Request {
                llm_request: text_request(Some("auto".to_string()), "classify this"),
                raw_request: None,
                metadata: None,
            },
        )
        .await?;

    assert_eq!(
        trace.last().and_then(|decision| decision.routing_tier()),
        Some("weak")
    );

    let snapshots = flushed_metrics(exporter, provider);
    assert_eq!(
        u64_counter_value(
            &snapshots,
            "switchyard.llm_calls",
            &[
                ("algorithm", ""),
                ("selected_model", "classifier"),
                ("outcome", "ok"),
            ],
        ),
        Some(1)
    );
    assert_eq!(
        u64_counter_value(
            &snapshots,
            "switchyard.requests",
            &[("model", "weak"), ("tier", "weak")],
        ),
        Some(1)
    );
    assert_eq!(
        u64_counter_value(
            &snapshots,
            "switchyard.requests",
            &[("model", "classifier")],
        ),
        None
    );
    assert_eq!(
        f64_histogram_count(
            &snapshots,
            "switchyard.model_call_latency_ms",
            &[("model", "classifier")],
        ),
        None
    );
    assert_eq!(
        u64_gauge_value(&snapshots, "switchyard.total_requests"),
        Some(total_requests_before + 1)
    );
    // The classifier call is the router's own work but the routed call is not,
    // so overhead lands near the classifier's 60ms, not their 260ms sum.
    let overhead = f64_histogram_sum_ms(
        &snapshots,
        "switchyard.routing_overhead_ms",
        &[("algorithm", "llm_task_classifier")],
    )
    .unwrap_or_default();
    assert!(
        (60..200).contains(&overhead),
        "expected roughly the classifier's 60ms, got {overhead}ms"
    );
    Ok(())
}
