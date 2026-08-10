// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenTelemetry metrics plus `tracing` spans and structured logs for the
//! algorithm layer.
//!
//! [`Algorithm::run_stream`](crate::Algorithm::run_stream) and [`Driver`] call these
//! helpers around the [`Decision`] hook and the offload boundary, so every algorithm is
//! instrumented from the outside and carries no telemetry code of its own. The provider
//! call on the other side of the offload belongs to the host, and is instrumented by
//! whoever makes it. Metrics record through the
//! OpenTelemetry **global** meter provider under the `switchyard` scope — the host
//! installs an SDK provider and exporters; with none installed, recording is a
//! no-op. Spans and logs use the `tracing` facade (the async-native surface the
//! OpenTelemetry ecosystem bridges with `tracing-opentelemetry` /
//! `opentelemetry-appender-tracing`), so the host's subscriber decides where
//! they go. Method spans use `#[tracing::instrument]`; the `libsy.run` span is
//! attached to the spawned run task with [`tracing::Instrument`]. Neither holds
//! a [`Span::enter`] guard across an `.await` — a suspended task would leave
//! the span entered on its executor thread, mis-parenting every span other
//! tasks create there (see the `tracing` docs on spans in asynchronous code).
//!
//! Instrument names use the OTel dotted form with the unit baked into the name
//! (`switchyard.run_duration_ms`), matching the switchyard metric surface; a
//! Prometheus exporter sanitizes them to `switchyard_run_duration_ms`. Attribute
//! cardinality is bounded: `algorithm` and `selected_model` are small
//! configured sets and `outcome` is `ok`/`error`. Nothing per-request becomes a
//! metric attribute — correlation ids ride on the `libsy.run` span instead.
//!
//! Instruments are resolved from the global provider on every record (an
//! instrument-cache lookup inside the SDK) so recording follows a meter
//! provider installed at any point in the process lifetime; the cost is
//! negligible next to a model call.

use std::future::Future;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use opentelemetry::metrics::{Meter, ObservableGauge};
use opentelemetry::{KeyValue, global};
use tracing::Span;

use crate::{DriverError, LibsyError, Result};
use switchyard_protocol::{Decision, Request, Response};

const METRICS_SCOPE: &str = "switchyard";
const TRACING_TARGET: &str = "libsy";

static TOTAL_REQUESTS: AtomicU64 = AtomicU64::new(0);
static TOTAL_ERRORS: AtomicU64 = AtomicU64::new(0);
static TOTAL_GAUGES: OnceLock<(ObservableGauge<u64>, ObservableGauge<u64>)> = OnceLock::new();

/// The `libsy`-scoped meter from the globally installed provider.
pub(crate) fn meter() -> Meter {
    global::meter(METRICS_SCOPE)
}

/// Registers process-wide compatibility gauges with the installed global meter provider.
pub(crate) fn initialize_metrics() {
    TOTAL_GAUGES.get_or_init(|| {
        let meter = meter();
        let requests = meter
            .u64_observable_gauge("switchyard.total_requests")
            .with_callback(|observer| {
                observer.observe(TOTAL_REQUESTS.load(Ordering::Relaxed), &[]);
            })
            .build();
        let errors = meter
            .u64_observable_gauge("switchyard.total_errors")
            .with_callback(|observer| {
                observer.observe(TOTAL_ERRORS.load(Ordering::Relaxed), &[]);
            })
            .build();
        (requests, errors)
    });
}

/// Whether a result is a call the consumer deliberately abandoned rather than a failure.
pub(crate) fn is_abandoned<T>(result: &Result<T>) -> bool {
    matches!(result, Err(LibsyError::Driver(DriverError::Abandoned)))
}

/// `outcome` attribute value for a result: `ok`, `abandoned`, or `error`.
pub(crate) fn outcome_value<T>(result: &Result<T>) -> &'static str {
    match result {
        Ok(_) => "ok",
        Err(LibsyError::Driver(DriverError::Abandoned)) => "abandoned",
        Err(_) => "error",
    }
}

/// Span covering one algorithm run (the whole `create_run_task` execution).
///
/// Correlation ids from the request [`switchyard_protocol::Metadata`] are recorded as span fields
/// when present. `tracing` spans cannot grow field names at runtime, so
/// arbitrary host labels ride in via [`switchyard_protocol::Metadata::extra_metadata`], recorded
/// whole into the `extra_metadata` field. `outcome` and `error` are filled in
/// by [`record_run`] when the run ends.
pub(crate) fn run_span(algorithm: &str, request: &Request) -> Span {
    let span = tracing::info_span!(
        target: TRACING_TARGET,
        "libsy.run",
        algorithm,
        switchyard.algorithm = algorithm,
        openinference.span.kind = "CHAIN",
        switchyard.route = tracing::field::Empty,
        session_id = tracing::field::Empty,
        session.id = tracing::field::Empty,
        agent_id = tracing::field::Empty,
        task_id = tracing::field::Empty,
        correlation_id = tracing::field::Empty,
        extra_metadata = tracing::field::Empty,
        outcome = tracing::field::Empty,
        error = tracing::field::Empty,
    );
    if let Some(route) = request.requested_model() {
        span.record("switchyard.route", route);
    }
    if let Some(metadata) = &request.metadata {
        for (field, value) in [
            ("session_id", &metadata.session_id),
            ("agent_id", &metadata.agent_id),
            ("task_id", &metadata.task_id),
            ("correlation_id", &metadata.correlation_id),
        ] {
            if let Some(value) = value {
                span.record(field, value.as_str());
            }
        }
        if let Some(session_id) = &metadata.session_id {
            span.record("session.id", session_id.as_str());
        }
        if let Some(extra) = &metadata.extra_metadata {
            span.record("extra_metadata", tracing::field::debug(extra));
        }
    }
    span
}

/// Runs one algorithm task to completion, recording the run counter, duration
/// histogram, span outcome, and failure log when it resolves.
/// Executes inside the `libsy.run` span its caller instruments the task with.
pub(crate) async fn observe_run(
    algorithm: &str,
    run: impl Future<Output = Result<Response>>,
) -> Result<Response> {
    let started = Instant::now();
    let result = run.await;
    let duration = started.elapsed();
    record_run(algorithm, duration, &result, &Span::current());
    result
}

/// Records the end of one algorithm run: the run counter and duration
/// histogram, the `outcome`/`error` fields on `span`, and a warn log when the
/// run failed.
fn record_run(algorithm: &str, duration: Duration, result: &Result<Response>, span: &Span) {
    let outcome = outcome_value(result);
    span.record("outcome", outcome);
    // An abandoned run ended because the consumer took the call, not because anything went
    // wrong, so it is counted but not reported as a failure.
    if let Err(error) = result
        && !is_abandoned(result)
    {
        span.record("error", tracing::field::display(error));
        tracing::warn!(
            target: TRACING_TARGET,
            algorithm,
            error = %error,
            "algorithm run failed"
        );
    }

    let attributes = [
        KeyValue::new("algorithm", algorithm.to_string()),
        KeyValue::new("outcome", outcome),
    ];
    let meter = meter();
    meter
        .u64_counter("switchyard.runs")
        .build()
        .add(1, &attributes);
    meter
        .f64_histogram("switchyard.run_duration_ms")
        .build()
        .record(duration.as_secs_f64() * 1000.0, &attributes);
}

/// Records a judge failure that made the classifier route without a verdict.
pub(crate) fn record_classifier_fail_open(judge_model: &str, reason: &'static str) {
    meter()
        .u64_counter("switchyard.classifier_fail_open")
        .build()
        .add(
            1,
            &[
                KeyValue::new("judge_model", judge_model.to_string()),
                KeyValue::new("reason", reason),
            ],
        );
}

/// Records the resolution of one offloaded model call: the call counter and
/// latency histogram, the `outcome`/`error`/token fields on `span`, and a warn
/// log when the call failed.
pub(crate) fn record_llm_call(
    algorithm: &str,
    selected_model: &str,
    is_answer_call: bool,
    duration: Duration,
    result: &Result<Response>,
    span: &Span,
) {
    let outcome = outcome_value(result);
    span.record("outcome", outcome);

    let meter = meter();
    let call_attributes = [
        KeyValue::new("algorithm", algorithm.to_string()),
        KeyValue::new("selected_model", selected_model.to_string()),
        KeyValue::new("outcome", outcome),
    ];
    meter
        .u64_counter("switchyard.llm_calls")
        .build()
        .add(1, &call_attributes);
    meter
        .f64_histogram("switchyard.llm_call_duration_ms")
        .build()
        .record(duration.as_secs_f64() * 1000.0, &call_attributes);

    // An abandoned answer call was never served, so it counts as neither a served
    // request nor an error.
    if is_answer_call && !is_abandoned(result) {
        TOTAL_REQUESTS.fetch_add(1, Ordering::Relaxed);
        let routed_attributes = [KeyValue::new("model", selected_model.to_string())];
        if result.is_ok() {
            meter
                .u64_counter("switchyard.requests")
                .build()
                .add(1, &routed_attributes);
            meter
                .f64_histogram("switchyard.model_call_latency_ms")
                .build()
                .record(duration.as_secs_f64() * 1000.0, &routed_attributes);
        } else {
            TOTAL_ERRORS.fetch_add(1, Ordering::Relaxed);
            meter
                .u64_counter("switchyard.errors")
                .build()
                .add(1, &routed_attributes);
        }
    }

    match result {
        Ok(response) => {
            // Token usage exists only once a response is buffered; a streamed
            // response resolves before its usage is known, so none is recorded.
            let Some(usage) = response.llm_response.as_agg().map(|agg| &agg.usage) else {
                return;
            };
            for (field, value) in [
                ("input_tokens", usage.input_tokens),
                ("output_tokens", usage.output_tokens),
                ("total_tokens", usage.total_tokens),
                ("reasoning_tokens", usage.reasoning_tokens),
            ] {
                if let Some(value) = value {
                    span.record(field, value);
                }
            }
        }
        Err(error) if !is_abandoned(result) => {
            span.record("error", tracing::field::display(error));
            tracing::warn!(
                target: TRACING_TARGET,
                algorithm,
                selected_model,
                error = %error,
                "model call failed"
            );
        }
        Err(_) => {}
    }
}

/// Records one published routing decision: the decision counter plus a
/// structured debug event carrying the decision's reasoning.
pub(crate) fn record_decision(algorithm: &str, decision: &Decision) {
    let selected_model = decision.selected_model_id();
    tracing::debug!(
        target: TRACING_TARGET,
        algorithm,
        selected_model,
        reasoning = decision.reasoning().unwrap_or(""),
        "routing decision"
    );
    meter().u64_counter("switchyard.decisions").build().add(
        1,
        &[
            KeyValue::new("algorithm", algorithm.to_string()),
            KeyValue::new("selected_model", selected_model.to_string()),
        ],
    );
}
