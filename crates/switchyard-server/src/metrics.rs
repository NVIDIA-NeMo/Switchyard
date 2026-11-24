// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-wide Prometheus export for Switchyard's OpenTelemetry metrics.

use std::sync::{LazyLock, OnceLock};

use opentelemetry::metrics::Counter;

use opentelemetry::{KeyValue, global};
use opentelemetry_sdk::metrics::{Aggregation, Instrument, SdkMeterProvider, Stream};
use prometheus::{Encoder, Registry, TextEncoder};
use switchyard_llm_client::metrics::{http_outcome_label, http_status_code_label};

pub(crate) const CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

/// Outcome label for a request the client abandoned. Kept separate from the
/// status-derived outcomes so cancelled work is never counted as a served response.
const CLIENT_DISCONNECTED_OUTCOME: &str = "client_disconnected";

/// Bucket boundaries for `switchyard.routing_overhead_ms`.
/// Need a broad range because some algos call an LLM (classifier), and some
/// do very little (passthrough).
const ROUTING_OVERHEAD_BUCKETS_MS: &[f64] = &[
    0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0,
];

/// Bucket boundaries for model-call and end-to-end LLM latency histograms.
/// Retains the SDK defaults through 10 seconds and extends them for long generations.
const LLM_LATENCY_BUCKETS_MS: &[f64] = &[
    0.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0, 250.0, 500.0, 750.0, 1000.0, 2500.0, 5000.0, 7500.0,
    10_000.0, 15_000.0, 30_000.0, 60_000.0, 120_000.0, 300_000.0,
];

struct Metrics {
    registry: Registry,
    provider: SdkMeterProvider,
}

static METRICS: OnceLock<Result<Metrics, String>> = OnceLock::new();

/// Returns the registry shared by every server router in this process.
pub(crate) fn registry() -> Result<Registry, String> {
    match METRICS.get_or_init(initialize) {
        Ok(metrics) => Ok(metrics.registry.clone()),
        Err(error) => Err(error.clone()),
    }
}

fn initialize() -> Result<Metrics, String> {
    let registry = Registry::new();
    let exporter = opentelemetry_prometheus::exporter()
        .with_registry(registry.clone())
        .build()
        .map_err(|error| format!("failed to initialize Prometheus metrics: {error}"))?;
    let mut builder = SdkMeterProvider::builder()
        .with_reader(exporter)
        .with_view(routing_overhead_buckets)
        .with_view(llm_latency_buckets)
        .with_resource(crate::observability::resource());
    if crate::observability::otlp_enabled("METRICS") {
        let exporter = opentelemetry_otlp::MetricExporter::builder()
            .with_http()
            .build()
            .map_err(|error| format!("failed to initialize OTLP metric exporter: {error}"))?;
        builder = builder.with_periodic_exporter(exporter);
    }
    let provider = builder.build();
    global::set_meter_provider(provider.clone());
    switchyard_llm_client::initialize_metrics();
    global::meter("switchyard")
        .u64_gauge("switchyard.build_info")
        .build()
        .record(1, &[KeyValue::new("version", env!("CARGO_PKG_VERSION"))]);
    seed_outcome_metrics();
    Ok(Metrics { registry, provider })
}

pub(crate) fn flush() {
    if let Some(Ok(metrics)) = METRICS.get()
        && let Err(error) = metrics.provider.force_flush()
    {
        tracing::warn!(error = %error, "failed to flush OpenTelemetry metrics");
    }
}

fn routing_overhead_buckets(instrument: &Instrument) -> Option<Stream> {
    if instrument.name() != "switchyard.routing_overhead_ms" {
        return None;
    }
    Stream::builder()
        .with_aggregation(Aggregation::ExplicitBucketHistogram {
            boundaries: ROUTING_OVERHEAD_BUCKETS_MS.to_vec(),
            // Cumulative min/max cover the whole process, so they aren't useful.
            record_min_max: false,
        })
        .build()
        .ok()
}

fn llm_latency_buckets(instrument: &Instrument) -> Option<Stream> {
    if !matches!(
        instrument.name(),
        "switchyard.model_call_latency_ms" | "switchyard.total_latency_ms"
    ) {
        return None;
    }
    Stream::builder()
        .with_aggregation(Aggregation::ExplicitBucketHistogram {
            boundaries: LLM_LATENCY_BUCKETS_MS.to_vec(),
            record_min_max: true,
        })
        .build()
        .ok()
}

/// Make the metrics exist before they get a hit. Nicer for dashboards but not really necessary.
/// The HTTP status codes we seed are somewhat arbitrary.
fn seed_outcome_metrics() {
    let meter = global::meter("switchyard");
    let upstream_attempts = meter.u64_counter("switchyard.upstream_attempts").build();
    for status in [Some(200), Some(404), Some(429), Some(500), Some(504), None] {
        upstream_attempts.add(
            0,
            &[
                KeyValue::new("outcome", http_outcome_label(status)),
                KeyValue::new("code", http_status_code_label(status)),
            ],
        );
    }

    let client_responses = meter.u64_counter("switchyard.client_responses").build();
    for outcome in [
        "ok",
        "retryable_error",
        "other_error",
        CLIENT_DISCONNECTED_OUTCOME,
    ] {
        client_responses.add(0, &[KeyValue::new("outcome", outcome)]);
    }
    meter
        .u64_counter("switchyard.router_retry_recovered")
        .build()
        .add(0, &[]);
}

/// Records a request whose downstream client disconnected before a response was
/// written. The run and its upstream calls were cancelled, so no status exists.
pub(crate) fn record_client_disconnect() {
    global::meter("switchyard")
        .u64_counter("switchyard.client_responses")
        .build()
        .add(1, &[KeyValue::new("outcome", CLIENT_DISCONNECTED_OUTCOME)]);
}

/// Records the final status returned by an LLM-serving route.
pub(crate) fn record_client_response(status: u16) {
    global::meter("switchyard")
        .u64_counter("switchyard.client_responses")
        .build()
        .add(
            1,
            &[KeyValue::new("outcome", http_outcome_label(Some(status)))],
        );
}

/// Encodes the current cumulative metric values in Prometheus text format.
pub(crate) fn encode(registry: &Registry) -> Result<Vec<u8>, String> {
    let mut body = Vec::new();
    TextEncoder::new()
        .encode(&registry.gather(), &mut body)
        .map_err(|error| format!("failed to encode Prometheus metrics: {error}"))?;
    Ok(body)
}

/// Export the same priced call that feeds the session report. Keep request IDs off counters.
pub(crate) fn export_cost(
    model: &str,
    routing: &switchyard_costs::api::observed_spend::CostRoute,
    kind: &str,
    session: Option<&str>,
    turn: Option<&str>,
    outcome_id: Option<&str>,
    call: switchyard_costs::api::observed_spend::CallCost,
) {
    static SPEND: LazyLock<Counter<f64>> = LazyLock::new(|| usd_counter("switchyard.spend"));
    static CACHE: LazyLock<Counter<f64>> = LazyLock::new(|| usd_counter("switchyard.cache.spend"));
    let labels = [
        KeyValue::new("gen_ai.request.model", model.to_owned()),
        KeyValue::new("switchyard.target", routing.target.clone()),
        KeyValue::new("switchyard.route", routing.route_id.clone()),
        KeyValue::new("switchyard.algorithm", routing.algorithm.clone()),
        KeyValue::new("switchyard.call.kind", kind.to_owned()),
    ];
    if let Some(cost) = call.cost {
        SPEND.add(cost.usd, &labels);
    }
    for (op, usd) in [
        ("read", call.cache_read_usd),
        ("write", call.cache_write_usd),
    ] {
        if let Some(usd) = usd {
            let mut attrs = labels.to_vec();
            attrs.push(KeyValue::new("switchyard.cache.operation", op));
            CACHE.add(usd, &attrs);
        }
    }
    let _span = tracing::info_span!(
        "switchyard.cost.observed",
        gen_ai.request.model = model,
        switchyard.target = routing.target.as_str(),
        switchyard.route = routing.route_id.as_str(),
        switchyard.algorithm = routing.algorithm.as_str(),
        gen_ai.request.reasoning.level = routing.reasoning_effort.as_deref(),
        outcome_id = outcome_id,
        switchyard.call.kind = kind,
        gen_ai.conversation.id = session,
        switchyard.turn.id = turn,
        switchyard.cost.source = call.cost.map(|c| match c.source {
            switchyard_costs::CostSource::Estimated => "estimated",
            switchyard_costs::CostSource::ProviderReported => "provider_reported",
        }),
        switchyard.cost.usd = call.cost.map(|c| c.usd),
        switchyard.cache.read_usd = call.cache_read_usd,
        switchyard.cache.write_usd = call.cache_write_usd
    );
}

fn usd_counter(name: &'static str) -> Counter<f64> {
    global::meter("switchyard")
        .f64_counter(name)
        .with_unit("USD")
        .build()
}
