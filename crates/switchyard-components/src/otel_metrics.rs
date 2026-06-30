// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenTelemetry metric instruments and record helpers for the Rust serving path.
//!
//! This is the Rust counterpart of `switchyard/lib/metrics.py`. Instrument names
//! use the OTel dotted form (`switchyard.requests`); the Prometheus exporter
//! sanitizes `.` -> `_` and appends `_total` to monotonic counters, reproducing
//! the same Prometheus exposition surface the Python path serves. **No instrument
//! sets a unit:** the Prometheus exporter would otherwise append a unit suffix and
//! turn `switchyard.total_latency_ms` into `..._ms_ms`. Units are baked into the
//! name (`_ms`, `_usd`) instead.
//!
//! The process owns one [`SdkMeterProvider`] built lazily on first use. It always
//! holds a Prometheus pull reader (served at `/metrics` by `switchyard-server`)
//! and, when `OTEL_EXPORTER_OTLP_ENDPOINT` is set, a periodic OTLP push reader for
//! parity with the Python `PeriodicExportingMetricReader`.
//!
//! Label cardinality is intentionally bounded: `model`, `tier`, `router`, `role`,
//! and `kind` are small fixed sets; nothing per-request is ever an attribute.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use opentelemetry::metrics::{Counter, Histogram, Meter, MeterProvider};
use opentelemetry::KeyValue;
use opentelemetry_sdk::metrics::SdkMeterProvider;
use prometheus::{Encoder, Registry, TextEncoder};

use crate::stats::{estimate_model_cost, TokenUsage};

/// Instrumentation scope name shared with the Python meter.
const SCOPE: &str = "switchyard";

/// Running totals surfaced as observable gauges, mirroring the historical
/// `switchyard_total_requests` / `switchyard_total_errors` top-level series.
static TOTAL_REQUESTS: AtomicU64 = AtomicU64::new(0);
static TOTAL_ERRORS: AtomicU64 = AtomicU64::new(0);

/// Process-wide metrics state, built once on first use.
struct MetricsState {
    /// Held to keep the meter provider (and its readers) alive for the process.
    _provider: SdkMeterProvider,
    /// Prometheus registry the pull exporter writes into; rendered at `/metrics`.
    registry: Registry,
    requests: Counter<u64>,
    errors: Counter<u64>,
    prompt_tokens: Counter<u64>,
    completion_tokens: Counter<u64>,
    cached_tokens: Counter<u64>,
    cache_creation_tokens: Counter<u64>,
    reasoning_tokens: Counter<u64>,
    routing_decisions: Counter<u64>,
    model_call_latency_ms: Histogram<f64>,
    total_latency_ms: Histogram<f64>,
    routing_overhead_ms: Histogram<f64>,
    cost_usd: Histogram<f64>,
}

fn state() -> &'static MetricsState {
    static STATE: OnceLock<MetricsState> = OnceLock::new();
    STATE.get_or_init(build_state)
}

/// Builds the meter provider, instruments, and Prometheus registry.
///
/// The Prometheus pull reader is always installed. The OTLP push reader is added
/// only when `OTEL_EXPORTER_OTLP_ENDPOINT` is set, matching the Python bootstrap;
/// a build failure for either exporter logs and falls back to a provider with the
/// readers that did succeed so a misconfigured endpoint never blocks `/metrics`.
fn build_state() -> MetricsState {
    let registry = Registry::new();
    let mut builder = SdkMeterProvider::builder();

    match opentelemetry_prometheus::exporter()
        .with_registry(registry.clone())
        .build()
    {
        Ok(exporter) => builder = builder.with_reader(exporter),
        Err(error) => {
            tracing::warn!(error = %error, "failed to build Prometheus metrics exporter");
        }
    }

    if let Some(reader) = otlp_reader() {
        builder = builder.with_reader(reader);
    }

    let provider = builder.build();
    let meter = provider.meter(SCOPE);
    build_observable_gauges(&meter);

    MetricsState {
        // Counters -> Prometheus `_total`.
        requests: meter.u64_counter("switchyard.requests").build(),
        errors: meter.u64_counter("switchyard.errors").build(),
        prompt_tokens: meter.u64_counter("switchyard.prompt_tokens").build(),
        completion_tokens: meter.u64_counter("switchyard.completion_tokens").build(),
        cached_tokens: meter.u64_counter("switchyard.cached_tokens").build(),
        cache_creation_tokens: meter
            .u64_counter("switchyard.cache_creation_tokens")
            .build(),
        reasoning_tokens: meter.u64_counter("switchyard.reasoning_tokens").build(),
        routing_decisions: meter.u64_counter("switchyard.routing_decisions").build(),
        // Histograms -> Prometheus `_bucket`/`_sum`/`_count`.
        model_call_latency_ms: meter
            .f64_histogram("switchyard.model_call_latency_ms")
            .build(),
        total_latency_ms: meter.f64_histogram("switchyard.total_latency_ms").build(),
        routing_overhead_ms: meter
            .f64_histogram("switchyard.routing_overhead_ms")
            .build(),
        cost_usd: meter.f64_histogram("switchyard.cost_usd").build(),
        registry,
        _provider: provider,
    }
}

/// Builds the OTLP push reader when an endpoint is configured.
fn otlp_reader(
) -> Option<opentelemetry_sdk::metrics::PeriodicReader<opentelemetry_otlp::MetricExporter>> {
    std::env::var_os("OTEL_EXPORTER_OTLP_ENDPOINT")?;
    match opentelemetry_otlp::MetricExporter::builder()
        .with_tonic()
        .build()
    {
        Ok(exporter) => Some(opentelemetry_sdk::metrics::PeriodicReader::builder(exporter).build()),
        Err(error) => {
            tracing::warn!(error = %error, "failed to build OTLP metrics exporter; OTLP push disabled");
            None
        }
    }
}

/// Registers build-info and running-total observable gauges on the meter.
fn build_observable_gauges(meter: &Meter) {
    let version = env!("CARGO_PKG_VERSION").to_string();
    let _build_info = meter
        .u64_observable_gauge("switchyard.build_info")
        .with_callback(move |observer| {
            observer.observe(1, &[KeyValue::new("version", version.clone())]);
        })
        .build();
    let _total_requests = meter
        .u64_observable_gauge("switchyard.total_requests")
        .with_callback(|observer| {
            observer.observe(TOTAL_REQUESTS.load(Ordering::Relaxed), &[]);
        })
        .build();
    let _total_errors = meter
        .u64_observable_gauge("switchyard.total_errors")
        .with_callback(|observer| {
            observer.observe(TOTAL_ERRORS.load(Ordering::Relaxed), &[]);
        })
        .build();
}

/// Builds a counter/histogram attribute set, dropping `None` values.
fn attrs(pairs: &[(&'static str, Option<String>)]) -> Vec<KeyValue> {
    pairs
        .iter()
        .filter_map(|(key, value)| {
            value
                .as_ref()
                .map(|value| KeyValue::new(*key, value.clone()))
        })
        .collect()
}

/// Forces instrument and gauge creation so pull-only series (build info, running
/// totals) appear on the next `/metrics` scrape even before any request is served.
pub fn ensure_instruments() {
    let _ = state();
}

/// Renders the Prometheus exposition text for the process registry.
///
/// Returns an error string if encoding fails; callers map it to a 500.
pub fn render_prometheus() -> Result<String, String> {
    let state = state();
    let mut buffer = Vec::new();
    let families = state.registry.gather();
    TextEncoder::new()
        .encode(&families, &mut buffer)
        .map_err(|error| error.to_string())?;
    String::from_utf8(buffer).map_err(|error| error.to_string())
}

/// Counts one served request (and bumps the global request total).
pub fn record_request(model: &str, tier: Option<&str>, router: Option<&str>) {
    TOTAL_REQUESTS.fetch_add(1, Ordering::Relaxed);
    state().requests.add(
        1,
        &attrs(&[
            ("model", Some(model.to_string())),
            ("tier", tier.map(str::to_string)),
            ("router", router.map(str::to_string)),
        ]),
    );
}

/// Counts one backend error (and bumps the global error total).
pub fn record_error(model: &str, tier: Option<&str>, status: Option<&str>) {
    TOTAL_REQUESTS.fetch_add(1, Ordering::Relaxed);
    TOTAL_ERRORS.fetch_add(1, Ordering::Relaxed);
    state().errors.add(
        1,
        &attrs(&[
            ("model", Some(model.to_string())),
            ("tier", tier.map(str::to_string)),
            ("status", status.map(str::to_string)),
        ]),
    );
}

/// Records token counters for one response. Zero values are skipped.
pub fn record_tokens(model: &str, tier: Option<&str>, usage: TokenUsage) {
    let state = state();
    let attributes = attrs(&[
        ("model", Some(model.to_string())),
        ("tier", tier.map(str::to_string)),
    ]);
    for (counter, value) in [
        (&state.prompt_tokens, usage.prompt_tokens),
        (&state.completion_tokens, usage.completion_tokens),
        (&state.cached_tokens, usage.cached_tokens),
        (&state.cache_creation_tokens, usage.cache_creation_tokens),
        (&state.reasoning_tokens, usage.reasoning_tokens),
    ] {
        if value > 0 {
            counter.add(value, &attributes);
        }
    }
}

/// Records the latency histograms for one request. `None` values are skipped.
pub fn record_latencies(
    model: &str,
    tier: Option<&str>,
    router: Option<&str>,
    model_call_ms: Option<f64>,
    total_ms: Option<f64>,
    routing_overhead_ms: Option<f64>,
) {
    let state = state();
    let model_attrs = attrs(&[
        ("model", Some(model.to_string())),
        ("tier", tier.map(str::to_string)),
    ]);
    if let Some(value) = model_call_ms {
        state.model_call_latency_ms.record(value, &model_attrs);
    }
    if let Some(value) = total_ms {
        state.total_latency_ms.record(value, &model_attrs);
    }
    if let Some(value) = routing_overhead_ms {
        state
            .routing_overhead_ms
            .record(value, &attrs(&[("router", router.map(str::to_string))]));
    }
}

/// Records one dollar-cost sample. `role` is `routed`/`classifier`/`planner`.
pub fn record_cost(model: &str, tier: Option<&str>, role: &str, kind: &str, cost_usd: f64) {
    state().cost_usd.record(
        cost_usd,
        &attrs(&[
            ("model", Some(model.to_string())),
            ("tier", tier.map(str::to_string)),
            ("role", Some(role.to_string())),
            ("kind", Some(kind.to_string())),
        ]),
    );
}

/// Records token counters and the derived per-kind dollar cost for one response.
///
/// Mirrors `otel_usage._emit`: token counters plus a `cost_usd` sample per
/// non-zero cost kind (`input`/`cached`/`cache_write`/`output`). `role`
/// distinguishes routed-backend traffic from classifier/planner overhead so the
/// same model id never aliases across buckets.
pub fn record_usage_and_cost(model: &str, tier: Option<&str>, role: &str, usage: TokenUsage) {
    record_tokens(model, tier, usage);
    let cost = estimate_model_cost(
        model,
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.cached_tokens,
        usage.cache_creation_tokens,
    );
    for (kind, value) in [
        ("input", cost.base_input_cost),
        ("cached", cost.cached_input_cost),
        ("cache_write", cost.cache_write_cost),
        ("output", cost.output_cost),
    ] {
        if value != 0.0 {
            record_cost(model, tier, role, kind, value);
        }
    }
}

/// Counts one routing decision by strategy, decision source, and chosen tier.
pub fn record_routing_decision(router: &str, source: Option<&str>, tier: Option<&str>) {
    state().routing_decisions.add(
        1,
        &attrs(&[
            ("router", Some(router.to_string())),
            ("source", source.map(str::to_string)),
            ("tier", tier.map(str::to_string)),
        ]),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    // The process meter is a singleton, so this test asserts the Prometheus
    // surface shape (metric names / suffixes) rather than exact values, which
    // other tests in the same binary may also have moved.
    #[test]
    fn prometheus_surface_uses_expected_names_and_suffixes() {
        record_request("provider/model", Some("strong"), Some("random"));
        record_tokens(
            "provider/model",
            Some("strong"),
            TokenUsage {
                prompt_tokens: 5,
                completion_tokens: 3,
                ..TokenUsage::default()
            },
        );
        record_latencies(
            "provider/model",
            Some("strong"),
            Some("random"),
            Some(1.0),
            Some(2.0),
            Some(0.5),
        );
        record_usage_and_cost(
            "openai/openai/gpt-5.2",
            Some("strong"),
            "routed",
            TokenUsage {
                prompt_tokens: 1_000_000,
                completion_tokens: 1_000_000,
                ..TokenUsage::default()
            },
        );
        record_routing_decision("cascade", Some("override"), Some("strong"));

        let text = render_prometheus().expect("prometheus render should succeed");
        // Counters render with `_total`; no `_ms_ms` double suffix.
        assert!(text.contains("switchyard_requests_total"));
        assert!(text.contains("switchyard_prompt_tokens_total"));
        assert!(text.contains("switchyard_routing_decisions_total"));
        assert!(text.contains("switchyard_total_latency_ms_bucket"));
        assert!(text.contains("switchyard_cost_usd_bucket"));
        assert!(text.contains("switchyard_build_info"));
        assert!(!text.contains("_ms_ms"));
    }
}
