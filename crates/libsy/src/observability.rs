// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenTelemetry metrics plus `tracing` spans and structured logs for the
//! algorithm layer.
//!
//! [`Algorithm::run_stream`](crate::Algorithm::run_stream) and [`Driver`] call these
//! helpers around the routing outcome and the offload boundary, so every algorithm is
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
use std::time::{Duration, Instant};

use opentelemetry::metrics::Meter;
use opentelemetry::{KeyValue, global};
use tracing::Span;
use tracing_opentelemetry::OpenTelemetrySpanExt;

use crate::{OutcomeMetadata, Result};
use switchyard_protocol::{ModelId, Request, Response};

const METRICS_SCOPE: &str = "switchyard";
const TRACING_TARGET: &str = "libsy";

/// The `libsy`-scoped meter from the globally installed provider.
pub(crate) fn meter() -> Meter {
    global::meter(METRICS_SCOPE)
}

/// `outcome` attribute value for a result: `ok` or `error`.
pub(crate) fn outcome_value<T>(result: &Result<T>) -> &'static str {
    match result {
        Ok(_) => "ok",
        Err(_) => "error",
    }
}

/// Span covering one algorithm run (the whole `route` execution).
///
/// Correlation ids from the request [`switchyard_protocol::Metadata`] are recorded as span fields
/// when present. Arbitrary extra metadata and error details are not exported.
/// [`record_outcome`] fills in successful outcome fields; [`record_run`] records
/// whether the run succeeded.
pub(crate) fn run_span(algorithm: &str, request: &Request) -> Span {
    let span = tracing::info_span!(
        target: TRACING_TARGET,
        "libsy.run",
        algorithm,
        outcome_id = tracing::field::Empty,
        evidence.source = tracing::field::Empty,
        evidence.score = tracing::field::Empty,
        evidence.confidence = tracing::field::Empty,
        evidence.threshold = tracing::field::Empty,
        evidence.verdict = tracing::field::Empty,
        evidence.trigger = tracing::field::Empty,
        evidence.reason_code = tracing::field::Empty,
        switchyard.algorithm = algorithm,
        openinference.span.kind = "CHAIN",
        switchyard.route = tracing::field::Empty,
        session_id = tracing::field::Empty,
        session.id = tracing::field::Empty,
        agent_id = tracing::field::Empty,
        task_id = tracing::field::Empty,
        task_kind = tracing::field::Empty,
        agent_role = tracing::field::Empty,
        correlation_id = tracing::field::Empty,
        outcome = tracing::field::Empty,
    );
    if let Some(route) = request.model_id() {
        span.record("switchyard.route", route.as_ref());
    }
    if let Some(metadata) = &request.metadata {
        for (field, value) in [
            ("session_id", &metadata.session_id),
            ("agent_id", &metadata.agent_id),
            ("task_id", &metadata.task_id),
            ("task_kind", &metadata.task_kind),
            ("agent_role", &metadata.agent_role),
            ("correlation_id", &metadata.correlation_id),
        ] {
            if let Some(value) = value {
                span.record(field, value.as_str());
            }
        }
        if let Some(session_id) = &metadata.session_id {
            span.record("session.id", session_id.as_str());
        }
    }
    span
}

/// Projects a successful outcome onto the existing run span. Model IDs are an
/// ordered OpenTelemetry string array, preserving fallback order. Evidence uses typed fields;
/// unknown keys and values of the wrong type are omitted.
pub(crate) fn record_outcome(metadata: &OutcomeMetadata, models: &[ModelId]) {
    let span = Span::current();
    span.record("outcome_id", metadata.outcome_id());
    span.set_attribute(
        "selected_model_ids",
        opentelemetry::Value::Array(opentelemetry::Array::String(
            models
                .iter()
                .map(|model| model.to_string().into())
                .collect(),
        )),
    );
    if let Some(evidence) = &metadata.evidence {
        for (key, field) in [
            ("source", "evidence.source"),
            ("verdict", "evidence.verdict"),
            ("trigger", "evidence.trigger"),
            ("reason_code", "evidence.reason_code"),
        ] {
            if let Some(value) = evidence.get(key).and_then(serde_json::Value::as_str) {
                span.record(field, value);
            }
        }
        for (key, field) in [
            ("score", "evidence.score"),
            ("confidence", "evidence.confidence"),
            ("threshold", "evidence.threshold"),
        ] {
            if let Some(value) = evidence.get(key).and_then(serde_json::Value::as_f64) {
                span.record(field, value);
            }
        }
    }
}

/// Holds `switchyard.algorithms_in_flight` up by one for as long as it lives.
struct InFlightRun {
    algorithm: String,
}

impl InFlightRun {
    fn enter(algorithm: &str) -> Self {
        record_algorithms_in_flight(algorithm, 1);
        Self {
            algorithm: algorithm.to_string(),
        }
    }
}

impl Drop for InFlightRun {
    fn drop(&mut self) {
        record_algorithms_in_flight(&self.algorithm, -1);
    }
}

/// Adds `delta` to the count of algorithm runs that have started and not yet
/// finished.
fn record_algorithms_in_flight(algorithm: &str, delta: i64) {
    meter()
        .i64_up_down_counter("switchyard.algorithms_in_flight")
        .build()
        .add(delta, &[KeyValue::new("algorithm", algorithm.to_string())]);
}

/// Runs one algorithm task to completion, recording the run counter, duration
/// histogram and span outcome when it resolves. Counts the run as
/// in flight for its whole duration.
/// Executes inside the `libsy.run` span its caller instruments the task with.
pub(crate) async fn observe_run<T>(
    algorithm: &str,
    run: impl Future<Output = Result<T>>,
) -> Result<T> {
    // Binding, not `let _ =`: the guard must live until the run resolves.
    let _in_flight = InFlightRun::enter(algorithm);
    let started = Instant::now();
    let result = run.await;
    let duration = started.elapsed();
    record_run(algorithm, duration, &result, &Span::current());
    result
}

/// Records the end of one algorithm run: the run counter and duration
/// histogram and the `outcome` field on `span`, without error details.
fn record_run<T>(algorithm: &str, duration: Duration, result: &Result<T>, span: &Span) {
    let outcome = outcome_value(result);
    span.record("outcome", outcome);

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
/// latency histogram and the outcome/token fields on `span`, without error details.
pub(crate) fn record_llm_call(
    algorithm: &str,
    selected_model: &str,
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

    if let Ok(response) = result {
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
}

/// Counts one published routing decision.
pub(crate) fn record_decision(algorithm: &str, selected_model: &ModelId) {
    meter().u64_counter("switchyard.decisions").build().add(
        1,
        &[
            KeyValue::new("algorithm", algorithm.to_string()),
            KeyValue::new("selected_model", selected_model.to_string()),
        ],
    );
}
