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
use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use async_trait::async_trait;
use futures::StreamExt;
use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData, ResourceMetrics};
use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Subscriber};
use tracing_subscriber::layer::{Context as LayerContext, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;

use switchyard_libsy::{
    AggLlmResponse, Algorithm, Context, Decision, Driver, LlmResponse, LlmTarget, LlmTargetSet,
    Metadata, Request, Response, RoutedLlmClient, Step, Usage,
};
use switchyard_protocol::text_request;

type BoxErr = Box<dyn Error + Send + Sync>;

/// Locks a mutex, recovering the inner value if a panicking test poisoned it.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
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
        lock(&self.spans).values().cloned().collect()
    }

    fn events(&self) -> Vec<EventRecord> {
        lock(&self.events).clone()
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
        lock(&self.store.spans).insert(
            id.into_u64(),
            SpanRecord {
                name: attrs.metadata().name().to_string(),
                parent,
                fields,
            },
        );
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, _ctx: LayerContext<'_, S>) {
        if let Some(record) = lock(&self.store.spans).get_mut(&id.into_u64()) {
            values.record(&mut FieldVisitor(&mut record.fields));
        }
    }

    fn on_event(&self, event: &Event<'_>, _ctx: LayerContext<'_, S>) {
        let mut fields = BTreeMap::new();
        event.record(&mut FieldVisitor(&mut fields));
        lock(&self.store.events).push(EventRecord {
            target: event.metadata().target().to_string(),
            fields,
        });
    }
}

/// Installs the process-global telemetry sinks once: an in-memory OTel metric
/// pipeline behind the global meter provider, and the capture layer as the
/// global tracing subscriber.
fn telemetry() -> &'static (CaptureStore, InMemoryMetricExporter, SdkMeterProvider) {
    static TELEMETRY: OnceLock<(CaptureStore, InMemoryMetricExporter, SdkMeterProvider)> =
        OnceLock::new();
    TELEMETRY.get_or_init(|| {
        let exporter = InMemoryMetricExporter::default();
        let reader = PeriodicReader::builder(exporter.clone()).build();
        let provider = SdkMeterProvider::builder().with_reader(reader).build();
        opentelemetry::global::set_meter_provider(provider.clone());

        let store = CaptureStore::default();
        let subscriber = tracing_subscriber::registry().with(CaptureLayer {
            store: store.clone(),
        });
        if tracing::subscriber::set_global_default(subscriber).is_err() {
            panic!("a global tracing subscriber was already installed in this test binary");
        }
        (store, exporter, provider)
    })
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

/// Latest cumulative value of a `u64` counter for the given attribute set.
fn u64_counter_value(
    snapshots: &[ResourceMetrics],
    name: &str,
    wanted: &[(&str, &str)],
) -> Option<u64> {
    let mut latest = None;
    for snapshot in snapshots {
        for scope in snapshot.scope_metrics() {
            if scope.scope().name() != "libsy" {
                continue;
            }
            for metric in scope.metrics() {
                if metric.name() != name {
                    continue;
                }
                if let AggregatedMetrics::U64(MetricData::Sum(sum)) = metric.data() {
                    for point in sum.data_points() {
                        if attributes_match(point.attributes(), wanted) {
                            // Cumulative counters only grow; max = most recent.
                            latest =
                                Some(latest.map_or(point.value(), |v: u64| v.max(point.value())));
                        }
                    }
                }
            }
        }
    }
    latest
}

/// Latest cumulative sample count of an `f64` histogram for the attribute set.
fn f64_histogram_count(
    snapshots: &[ResourceMetrics],
    name: &str,
    wanted: &[(&str, &str)],
) -> Option<u64> {
    let mut latest = None;
    for snapshot in snapshots {
        for scope in snapshot.scope_metrics() {
            if scope.scope().name() != "libsy" {
                continue;
            }
            for metric in scope.metrics() {
                if metric.name() != name {
                    continue;
                }
                if let AggregatedMetrics::F64(MetricData::Histogram(histogram)) = metric.data() {
                    for point in histogram.data_points() {
                        if attributes_match(point.attributes(), wanted) {
                            latest =
                                Some(latest.map_or(point.count(), |v: u64| v.max(point.count())));
                        }
                    }
                }
            }
        }
    }
    latest
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

#[async_trait]
impl RoutedLlmClient for UsageClient {
    async fn call(
        &self,
        _ctx: Context,
        _request: Request,
        decision: Arc<dyn Decision>,
    ) -> Result<Response, BoxErr> {
        Ok(Response {
            llm_response: LlmResponse::Agg(AggLlmResponse {
                model: Some(decision.selected_model().to_string()),
                usage: self.usage.clone(),
                ..AggLlmResponse::default()
            }),
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
    ) -> Result<Response, BoxErr> {
        let target = self
            .target_set
            .targets()
            .first()
            .ok_or("no targets")?
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

#[tokio::test]
async fn successful_run_records_metrics_spans_and_decision_log() -> Result<(), BoxErr> {
    let (store, exporter, provider) = telemetry();
    const ALGO: &str = "obs-success-algo";
    const MODEL: &str = "obs-success-model";

    let client = Arc::new(UsageClient {
        usage: Usage {
            input_tokens: Some(11),
            output_tokens: Some(7),
            total_tokens: Some(18),
            reasoning_tokens: Some(2),
            ..Usage::default()
        },
    }) as Arc<dyn RoutedLlmClient>;
    let (trace, _response) = algo(ALGO, MODEL, Some(client))
        .run(
            Context::default(),
            request_with_metadata("obs-session-1", "obs-corr-1"),
        )
        .await?;
    assert_eq!(trace.len(), 1);

    // Metrics: run/call counters and latency histograms keyed by algorithm,
    // token counters from the response usage, one published decision.
    let snapshots = flushed_metrics(exporter, provider);
    let run_attrs = [("algorithm", ALGO), ("outcome", "ok")];
    let call_attrs = [
        ("algorithm", ALGO),
        ("selected_model", MODEL),
        ("outcome", "ok"),
    ];
    let token_attrs = [("algorithm", ALGO), ("selected_model", MODEL)];
    assert_eq!(
        u64_counter_value(&snapshots, "libsy.runs", &run_attrs),
        Some(1)
    );
    assert_eq!(
        u64_counter_value(&snapshots, "libsy.llm_calls", &call_attrs),
        Some(1)
    );
    assert_eq!(
        f64_histogram_count(&snapshots, "libsy.run_duration_ms", &run_attrs),
        Some(1)
    );
    assert_eq!(
        f64_histogram_count(&snapshots, "libsy.llm_call_duration_ms", &call_attrs),
        Some(1)
    );
    assert_eq!(
        u64_counter_value(&snapshots, "libsy.input_tokens", &token_attrs),
        Some(11)
    );
    assert_eq!(
        u64_counter_value(&snapshots, "libsy.output_tokens", &token_attrs),
        Some(7)
    );
    assert_eq!(
        u64_counter_value(&snapshots, "libsy.total_tokens", &token_attrs),
        Some(18)
    );
    assert_eq!(
        u64_counter_value(&snapshots, "libsy.reasoning_tokens", &token_attrs),
        Some(2)
    );
    assert_eq!(
        u64_counter_value(&snapshots, "libsy.decisions", &token_attrs),
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
        run_span.fields.get("correlation_id").map(String::as_str),
        Some("obs-corr-1")
    );
    assert_eq!(
        run_span.fields.get("outcome").map(String::as_str),
        Some("ok")
    );

    // The default-client serve inside `run` gets its own client-call span.
    let client_span = find_span(&spans, "libsy.client_call", "selected_model", MODEL);
    assert_eq!(
        client_span.fields.get("algorithm").map(String::as_str),
        Some(ALGO)
    );
    assert_eq!(
        client_span.fields.get("outcome").map(String::as_str),
        Some("ok")
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
        Some("18")
    );
    assert_eq!(
        call_span.fields.get("reasoning_tokens").map(String::as_str),
        Some("2")
    );

    // Structured log: the published decision with its reasoning.
    let events = store.events();
    assert!(
        events.iter().any(|event| {
            event.target == "libsy"
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

#[tokio::test]
async fn failed_call_records_error_outcome_and_warn_logs() -> Result<(), BoxErr> {
    let (store, exporter, provider) = telemetry();
    const ALGO: &str = "obs-failure-algo";
    const MODEL: &str = "obs-failure-model";

    // Client-less target: the call is offloaded and we fail it by hand.
    let stream = algo(ALGO, MODEL, None).run_stream(
        Context::default(),
        request_with_metadata("obs-session-2", "obs-corr-2"),
    );
    tokio::pin!(stream);

    let mut saw_error_step = false;
    while let Some(step) = stream.next().await {
        match step {
            Ok(Step::CallLlm(call)) => {
                call.respond(Err("synthetic upstream failure".into()))?;
            }
            Ok(Step::Decision(_)) => {}
            Ok(Step::ReturnToAgent(_)) => {
                return Err("expected the failed call to fail the run".into());
            }
            Err(_) => saw_error_step = true,
        }
    }
    assert!(
        saw_error_step,
        "expected an error step from the failed call"
    );

    // Metrics: the call and the run both count under outcome=error, and no
    // token counters exist for the failed model.
    let snapshots = flushed_metrics(exporter, provider);
    let run_attrs = [("algorithm", ALGO), ("outcome", "error")];
    let call_attrs = [
        ("algorithm", ALGO),
        ("selected_model", MODEL),
        ("outcome", "error"),
    ];
    assert_eq!(
        u64_counter_value(&snapshots, "libsy.runs", &run_attrs),
        Some(1)
    );
    assert_eq!(
        u64_counter_value(&snapshots, "libsy.llm_calls", &call_attrs),
        Some(1)
    );
    assert_eq!(
        u64_counter_value(
            &snapshots,
            "libsy.input_tokens",
            &[("algorithm", ALGO), ("selected_model", MODEL)],
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
    assert!(run_span
        .fields
        .get("error")
        .is_some_and(|error| error.contains("synthetic upstream failure")));
    let call_span = find_span(&spans, "libsy.llm_call", "selected_model", MODEL);
    assert_eq!(
        call_span.fields.get("outcome").map(String::as_str),
        Some("error")
    );

    // Structured logs: a warn per failed call and per failed run.
    let events = store.events();
    assert!(
        events.iter().any(|event| {
            event.target == "libsy"
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
