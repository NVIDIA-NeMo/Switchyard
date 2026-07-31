// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-wide Prometheus export for Switchyard's OpenTelemetry metrics.

use std::sync::OnceLock;

use opentelemetry::{global, KeyValue};
use opentelemetry_sdk::metrics::{Aggregation, Instrument, SdkMeterProvider, Stream};
use prometheus::{Encoder, Registry, TextEncoder};
use switchyard_llm_client::metrics::{http_outcome_label, http_status_code_label};

pub(crate) const CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

/// Bucket boundaries for `switchyard.routing_overhead_ms`.
/// Need a broad range because some algos call an LLM (classifier), and some
/// do very little (passthrough).
const ROUTING_OVERHEAD_BUCKETS_MS: &[f64] = &[
    0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0,
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
    libsy::initialize_metrics();
    global::meter("switchyard")
        .u64_gauge("switchyard.build_info")
        .build()
        .record(1, &[KeyValue::new("version", env!("CARGO_PKG_VERSION"))]);
    seed_outcome_metrics();
    Ok(Metrics { registry, provider })
}

pub(crate) fn flush() {
    if let Some(Ok(metrics)) = METRICS.get() {
        if let Err(error) = metrics.provider.force_flush() {
            tracing::warn!(error = %error, "failed to flush OpenTelemetry metrics");
        }
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
    for outcome in ["success", "retryable_error", "other_error"] {
        client_responses.add(0, &[KeyValue::new("outcome", outcome)]);
    }
    meter
        .u64_counter("switchyard.router_retry_recovered")
        .build()
        .add(0, &[]);
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
