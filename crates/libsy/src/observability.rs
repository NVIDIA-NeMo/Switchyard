// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenTelemetry metrics plus `tracing` spans and structured logs for the
//! algorithm layer.
//!
//! The crate's provided run methods call these helpers around the [`Decision`]
//! hook and the offload boundary, so every algorithm is instrumented from the
//! outside and carries no telemetry code of its own. Metrics record through the
//! OpenTelemetry **global** meter provider under the `libsy` scope — the host
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
//! (`libsy.run_duration_ms`), matching the switchyard metric surface; a
//! Prometheus exporter sanitizes them to `libsy_run_duration_ms`. Attribute
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
use opentelemetry::{global, KeyValue};
use tracing::Span;

use crate::{Context, Decision, Metadata, Response};

/// Shorthand for the crate's boxed, thread-safe error type.
type BoxErr = Box<dyn std::error::Error + Send + Sync>;

/// Instrumentation scope for every libsy meter, span, and log line.
const SCOPE: &str = "libsy";

/// [`Context::values`] key under which `run_stream` stamps the algorithm's
/// telemetry label ([`Algorithm::name`](crate::Algorithm::name)).
pub(crate) const ALGORITHM_KEY: &str = "algorithm";

/// The algorithm label carried by a request context; empty until stamped.
pub(crate) fn algorithm_label<S>(ctx: &Context<S>) -> &str {
    ctx.values
        .get(ALGORITHM_KEY)
        .map(String::as_str)
        .unwrap_or("")
}

/// The `libsy`-scoped meter from the globally installed provider.
fn meter() -> Meter {
    global::meter(SCOPE)
}

/// `outcome` attribute value for a result: `ok` or `error`.
fn outcome_value<T>(result: &Result<T, BoxErr>) -> &'static str {
    if result.is_ok() {
        "ok"
    } else {
        "error"
    }
}

/// Span covering one algorithm run (the whole `create_run_task` execution).
///
/// Correlation ids from the request [`Metadata`] are recorded as span fields
/// when present. `tracing` spans cannot grow field names at runtime, so
/// arbitrary host labels ride in via [`Metadata::extra_metadata`], recorded
/// whole into the `extra_metadata` field. `outcome` and `error` are filled in
/// by [`record_run`] when the run ends.
pub(crate) fn run_span(algorithm: &str, metadata: Option<&Metadata>) -> Span {
    let span = tracing::info_span!(
        target: SCOPE,
        "libsy.run",
        algorithm,
        session_id = tracing::field::Empty,
        agent_id = tracing::field::Empty,
        task_id = tracing::field::Empty,
        correlation_id = tracing::field::Empty,
        extra_metadata = tracing::field::Empty,
        outcome = tracing::field::Empty,
        error = tracing::field::Empty,
    );
    if let Some(metadata) = metadata {
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
        if let Some(extra) = &metadata.extra_metadata {
            span.record("extra_metadata", tracing::field::debug(extra));
        }
    }
    span
}

/// Runs one algorithm task to completion, recording the run counter, duration
/// histogram, span outcome, and failure log when it resolves. Executes inside
/// the `libsy.run` span its caller instruments the task with.
pub(crate) async fn observe_run<S>(
    ctx: Context<S>,
    run: impl Future<Output = Result<Response, BoxErr>>,
) -> Result<Response, BoxErr> {
    let started = Instant::now();
    let result = run.await;
    record_run(
        algorithm_label(&ctx),
        started.elapsed(),
        &result,
        &Span::current(),
    );
    result
}

/// Records the outcome fields on the enclosing `libsy.client_call` span. The
/// failure itself is not logged here — it propagates to the algorithm, where
/// the `libsy.llm_call` recording logs it once.
pub(crate) fn record_client_call(result: &Result<Response, BoxErr>) {
    let span = Span::current();
    span.record("outcome", outcome_value(result));
    if let Err(error) = result {
        span.record("error", tracing::field::display(error));
    }
}

/// Records the end of one algorithm run: the run counter and duration
/// histogram, the `outcome`/`error` fields on `span`, and a warn log when the
/// run failed.
fn record_run(algorithm: &str, duration: Duration, result: &Result<Response, BoxErr>, span: &Span) {
    let outcome = outcome_value(result);
    span.record("outcome", outcome);
    if let Err(error) = result {
        span.record("error", tracing::field::display(error));
        tracing::warn!(target: SCOPE, algorithm, error = %error, "algorithm run failed");
    }

    let attributes = [
        KeyValue::new("algorithm", algorithm.to_string()),
        KeyValue::new("outcome", outcome),
    ];
    let meter = meter();
    meter.u64_counter("libsy.runs").build().add(1, &attributes);
    meter
        .f64_histogram("libsy.run_duration_ms")
        .build()
        .record(duration.as_secs_f64() * 1000.0, &attributes);
}

/// Records the resolution of one offloaded model call: the call counter and
/// latency histogram, token counters from the response usage (absent fields are
/// skipped, not recorded as zero), the `outcome`/`error`/token fields on
/// `span`, and a warn log when the call failed.
pub(crate) fn record_llm_call(
    algorithm: &str,
    selected_model: &str,
    duration: Duration,
    result: &Result<Response, BoxErr>,
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
        .u64_counter("libsy.llm_calls")
        .build()
        .add(1, &call_attributes);
    meter
        .f64_histogram("libsy.llm_call_duration_ms")
        .build()
        .record(duration.as_secs_f64() * 1000.0, &call_attributes);

    match result {
        Ok(response) => {
            // Token usage exists only once a response is buffered; a streamed
            // response resolves before its usage is known, so none is recorded.
            let Some(usage) = response.llm_response.as_agg().map(|agg| &agg.usage) else {
                return;
            };
            let token_attributes = [
                KeyValue::new("algorithm", algorithm.to_string()),
                KeyValue::new("selected_model", selected_model.to_string()),
            ];
            for (counter, field, value) in [
                ("libsy.input_tokens", "input_tokens", usage.input_tokens),
                ("libsy.output_tokens", "output_tokens", usage.output_tokens),
                ("libsy.total_tokens", "total_tokens", usage.total_tokens),
                (
                    "libsy.reasoning_tokens",
                    "reasoning_tokens",
                    usage.reasoning_tokens,
                ),
            ] {
                if let Some(value) = value {
                    span.record(field, value);
                    meter
                        .u64_counter(counter)
                        .build()
                        .add(value, &token_attributes);
                }
            }
        }
        Err(error) => {
            span.record("error", tracing::field::display(error));
            tracing::warn!(
                target: SCOPE,
                algorithm,
                selected_model,
                error = %error,
                "model call failed"
            );
        }
    }
}

/// Records one published routing decision: the decision counter plus a
/// structured info log carrying the decision's reasoning.
pub(crate) fn record_decision(ctx: &Context, decision: &dyn Decision) {
    let algorithm = algorithm_label(ctx);
    let selected_model = decision.selected_model();
    tracing::info!(
        target: SCOPE,
        algorithm,
        selected_model,
        reasoning = decision.reasoning().unwrap_or(""),
        "routing decision"
    );
    meter().u64_counter("libsy.decisions").build().add(
        1,
        &[
            KeyValue::new("algorithm", algorithm.to_string()),
            KeyValue::new("selected_model", selected_model.to_string()),
        ],
    );
}
