// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenTelemetry metrics plus `tracing` spans and structured logs for the
//! algorithm layer.
//!
//! The crate's provided run methods call these helpers around the [`Decision`]
//! hook and the offload boundary, so every algorithm is instrumented from the
//! outside and carries no telemetry code of its own. Metrics record through the
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

use std::borrow::Cow;
use std::future::Future;
use std::pin::Pin;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context as TaskContext, Poll};
use std::time::{Duration, Instant};

use futures::Stream;
use opentelemetry::metrics::{Meter, ObservableGauge};
use opentelemetry::{Array as OtelArray, KeyValue, StringValue, Value as OtelValue, global};
use switchyard_protocol::StopReason;
use tracing::Span;
use tracing_opentelemetry::OpenTelemetrySpanExt;

use crate::{Driver, LibsyError, Result};
use switchyard_protocol::{
    AggLlmResponse, Context, Decision, LlmClientError, LlmRequest, LlmResponse, LlmResponseChunk,
    LlmResponseStream, Request, Response, Usage,
};

const METRICS_SCOPE: &str = "switchyard";
const TRACING_TARGET: &str = "libsy";

static TOTAL_REQUESTS: AtomicU64 = AtomicU64::new(0);
static TOTAL_ERRORS: AtomicU64 = AtomicU64::new(0);
static TOTAL_GAUGES: OnceLock<(ObservableGauge<u64>, ObservableGauge<u64>)> = OnceLock::new();

/// [`Context::values`] key under which `run_stream` stamps the algorithm's
/// telemetry label ([`Algorithm::name`](crate::Algorithm::name)).
pub(crate) const ALGORITHM_KEY: &str = "algorithm";

/// The algorithm label carried by a request context; empty until stamped.
pub(crate) fn algorithm_label(ctx: &Context) -> &str {
    ctx.values
        .get(ALGORITHM_KEY)
        .map(String::as_str)
        .unwrap_or("")
}

/// The `libsy`-scoped meter from the globally installed provider.
fn meter() -> Meter {
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

/// `outcome` attribute value for a result: `ok` or `error`.
fn outcome_value<T>(result: &Result<T>) -> &'static str {
    if result.is_ok() { "ok" } else { "error" }
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
/// histogram, routing overhead, span outcome, and failure log when it resolves.
/// Executes inside the `libsy.run` span its caller instruments the task with.
/// `driver` is the run's own, holding the duration of the call that served it.
pub(crate) async fn observe_run(
    ctx: Context,
    driver: Driver,
    run: impl Future<Output = Result<Response>>,
) -> Result<Response> {
    let started = Instant::now();
    let result = run.await;
    let duration = started.elapsed();
    let algorithm = algorithm_label(&ctx);
    record_run(algorithm, duration, &result, &Span::current());
    if result.is_ok()
        && let Some(overhead) =
            record_routing_overhead(algorithm, duration, driver.routed_call_duration())
    {
        driver.observe_routing_overhead(overhead);
    }
    result
}

/// Records request parameters represented directly by the neutral IR.
pub(crate) fn record_gen_ai_request(span: &Span, request: &LlmRequest) {
    if request.stream {
        span.record("gen_ai.request.stream", true);
    }
    if let Some(value) = request.sampling.temperature {
        span.record("gen_ai.request.temperature", value);
    }
    if let Some(value) = request.sampling.top_p {
        span.record("gen_ai.request.top_p", value);
    }
    if let Some(value) = request.sampling.top_k {
        span.record("gen_ai.request.top_k", value);
    }
    if let Some(value) = request.output.max_output_tokens {
        span.record("gen_ai.request.max_tokens", otel_int(value));
    }
    if let Some(value) = request.reasoning.effort.as_deref() {
        span.record("gen_ai.request.reasoning.level", value);
    }
    if let Some(value) = request
        .output
        .response_format
        .as_ref()
        .and_then(gen_ai_output_type)
    {
        span.record("gen_ai.output.type", value);
    }
}

/// Adds terminal response and usage fields to the enclosing `libsy.client_call`
/// span without consuming or buffering a streaming response.
pub(crate) fn observe_client_call(result: Result<Response>) -> Result<Response> {
    let span = Span::current();
    match result {
        Ok(mut response) => {
            match response.llm_response {
                LlmResponse::Agg(agg) => {
                    span.record("outcome", "ok");
                    record_gen_ai_response(&span, &agg);
                    response.llm_response = LlmResponse::Agg(agg);
                }
                LlmResponse::Stream(stream) => {
                    response.llm_response =
                        LlmResponse::Stream(observe_client_stream(stream, span));
                }
            }
            Ok(response)
        }
        Err(error) => {
            let error_type = client_call_error_type(&error);
            record_client_error(&span, &error_type, &error);
            Err(error)
        }
    }
}

fn observe_client_stream(stream: LlmResponseStream, span: Span) -> LlmResponseStream {
    Box::pin(ObservedClientStream {
        stream,
        observer: Some(ClientStreamObserver {
            span,
            terminal: false,
        }),
    })
}

fn record_gen_ai_response(span: &Span, response: &AggLlmResponse) {
    record_optional(span, "gen_ai.response.id", response.id.as_deref());
    record_optional(span, "gen_ai.response.model", response.model.as_deref());
    record_finish_reasons(
        span,
        response
            .outputs
            .iter()
            .filter_map(|output| output.stop_reason)
            .map(stop_reason_name),
    );
    record_gen_ai_usage(span, &response.usage);
}

fn record_gen_ai_usage(span: &Span, usage: &Usage) {
    let cache_read = usage.cached_input_tokens();
    let cache_creation = usage.cache_creation_input_tokens();
    if usage.input_tokens.is_some() || cache_read.is_some() || cache_creation.is_some() {
        let input_tokens = usage
            .input_tokens
            .unwrap_or_default()
            .saturating_add(cache_read.unwrap_or_default())
            .saturating_add(cache_creation.unwrap_or_default());
        span.record("gen_ai.usage.input_tokens", otel_int(input_tokens));
    }
    for (field, value) in [
        ("gen_ai.usage.output_tokens", usage.output_tokens),
        ("gen_ai.usage.cache_read.input_tokens", cache_read),
        ("gen_ai.usage.cache_creation.input_tokens", cache_creation),
        (
            "gen_ai.usage.reasoning.output_tokens",
            usage.reasoning_tokens,
        ),
    ] {
        if let Some(value) = value {
            span.record(field, otel_int(value));
        }
    }
}

// OpenTelemetry integer attributes are signed; token counts are unsigned in the IR.
fn otel_int(value: u64) -> i64 {
    value.min(i64::MAX as u64) as i64
}

fn record_optional(span: &Span, field: &str, value: Option<&str>) {
    if let Some(value) = value {
        span.record(field, value);
    }
}

// `tracing` fields cannot preserve a typed string array, so write this attribute directly.
fn record_finish_reasons(span: &Span, reasons: impl IntoIterator<Item = impl Into<String>>) {
    let reasons = reasons
        .into_iter()
        .map(|reason| StringValue::from(reason.into()))
        .collect::<Vec<_>>();
    if !reasons.is_empty() {
        span.set_attribute(
            "gen_ai.response.finish_reasons",
            OtelValue::Array(OtelArray::String(reasons)),
        );
    }
}

fn stop_reason_name(reason: StopReason) -> &'static str {
    match reason {
        StopReason::EndTurn => "end_turn",
        StopReason::MaxTokens => "max_tokens",
        StopReason::ToolUse => "tool_use",
        StopReason::ContentFilter => "content_filter",
        StopReason::Error => "error",
        StopReason::Unknown => "unknown",
    }
}

fn gen_ai_output_type(response_format: &serde_json::Value) -> Option<&'static str> {
    match response_format
        .get("type")
        .and_then(serde_json::Value::as_str)
    {
        Some("json" | "json_object" | "json_schema") => Some("json"),
        Some("text") => Some("text"),
        _ => None,
    }
}

fn client_call_error_type(error: &LibsyError) -> Cow<'static, str> {
    match error {
        LibsyError::ClientCall { source, .. } => llm_client_error_type(source),
        LibsyError::TargetNotFound { .. } => Cow::Borrowed("target_not_found"),
        LibsyError::NoTargets => Cow::Borrowed("no_targets"),
        LibsyError::AlgorithmError { .. } => Cow::Borrowed("algorithm_error"),
        LibsyError::MissingClient { .. } => Cow::Borrowed("client_configuration_error"),
        LibsyError::Driver(_) => Cow::Borrowed("driver_error"),
        LibsyError::AlgorithmTask { .. } => Cow::Borrowed("algorithm_task_error"),
        LibsyError::MissingFinalResponse => Cow::Borrowed("missing_final_response"),
        LibsyError::AllTargetsExcluded => Cow::Borrowed("context_window_exceeded"),
        LibsyError::External { .. } => Cow::Borrowed("_OTHER"),
    }
}

fn llm_client_error_type(error: &LlmClientError) -> Cow<'static, str> {
    match error {
        LlmClientError::InvalidRequest { .. } => Cow::Borrowed("invalid_request"),
        LlmClientError::RequestTranslation(_) => Cow::Borrowed("request_translation"),
        LlmClientError::RequestEncoding(_) => Cow::Borrowed("request_encoding"),
        LlmClientError::ResponseTranslation(_) => Cow::Borrowed("response_translation"),
        LlmClientError::Configuration { .. } => Cow::Borrowed("configuration"),
        LlmClientError::Transport { .. } => Cow::Borrowed("transport"),
        LlmClientError::Timeout { .. } => Cow::Borrowed("timeout"),
        LlmClientError::ContextWindowExceeded { .. } => Cow::Borrowed("context_window_exceeded"),
        LlmClientError::UpstreamHttp { status, .. } => Cow::Owned(status.to_string()),
        LlmClientError::InvalidResponse { .. } => Cow::Borrowed("invalid_response"),
        LlmClientError::Ffi { .. } => Cow::Borrowed("ffi"),
        _ => Cow::Borrowed("_OTHER"),
    }
}

// Keep the client span alive until the response is drained, errors, or is abandoned.
struct ObservedClientStream {
    stream: LlmResponseStream,
    observer: Option<ClientStreamObserver>,
}

impl Stream for ObservedClientStream {
    type Item = std::result::Result<LlmResponseChunk, LlmClientError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        match self.stream.as_mut().poll_next(cx) {
            Poll::Ready(Some(item)) => {
                let failed = self
                    .observer
                    .as_mut()
                    .is_some_and(|observer| observer.observe(&item));
                if failed {
                    self.observer.take();
                }
                Poll::Ready(Some(item))
            }
            Poll::Ready(None) => {
                if let Some(mut observer) = self.observer.take() {
                    observer.complete();
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

struct ClientStreamObserver {
    span: Span,
    terminal: bool,
}

impl ClientStreamObserver {
    fn observe(&mut self, item: &std::result::Result<LlmResponseChunk, LlmClientError>) -> bool {
        match item {
            Ok(LlmResponseChunk::MessageStart { id, model }) => {
                record_optional(&self.span, "gen_ai.response.id", id.as_deref());
                record_optional(&self.span, "gen_ai.response.model", model.as_deref());
            }
            Ok(LlmResponseChunk::Usage(usage)) => record_gen_ai_usage(&self.span, usage),
            Ok(LlmResponseChunk::MessageStop { reason }) => {
                record_finish_reasons(&self.span, reason.iter().cloned());
            }
            Ok(LlmResponseChunk::DecodeError { message }) => {
                record_client_error(&self.span, "response_translation", message);
                self.terminal = true;
            }
            Ok(LlmResponseChunk::StreamError { message }) => {
                record_client_error(&self.span, "502", message);
                self.terminal = true;
            }
            Err(error) => {
                let error_type = llm_client_error_type(error);
                record_client_error(&self.span, &error_type, error);
                self.terminal = true;
            }
            _ => {}
        }
        self.terminal
    }

    fn complete(&mut self) {
        self.span.record("outcome", "ok");
        self.terminal = true;
    }
}

impl Drop for ClientStreamObserver {
    fn drop(&mut self) {
        if !self.terminal {
            self.span.record("outcome", "cancelled");
        }
    }
}

fn record_client_error(span: &Span, error_type: &str, error: &dyn std::fmt::Display) {
    span.record("outcome", "error");
    span.record("otel.status_code", "ERROR");
    span.record("error.type", error_type);
    span.record("error", tracing::field::display(error));
}

/// Records the end of one algorithm run: the run counter and duration
/// histogram, the `outcome`/`error` fields on `span`, and a warn log when the
/// run failed.
fn record_run(algorithm: &str, duration: Duration, result: &Result<Response>, span: &Span) {
    let outcome = outcome_value(result);
    span.record("outcome", outcome);
    if let Err(error) = result {
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

/// Records what routing cost on top of the call that served the run: classifier
/// calls, target resolution, decision publishing. A run with no routed call has
/// nothing to subtract, so it records nothing.
fn record_routing_overhead(
    algorithm: &str,
    run: Duration,
    routed_call: Option<Duration>,
) -> Option<Duration> {
    let routed_call = routed_call?;
    // Saturating: the two clocks start a moment apart, so a run that is all
    // routed call can come out fractionally negative.
    let overhead = run.saturating_sub(routed_call);
    meter()
        .f64_histogram("switchyard.routing_overhead_ms")
        .build()
        .record(
            overhead.as_secs_f64() * 1000.0,
            &[KeyValue::new("algorithm", algorithm.to_string())],
        );
    Some(overhead)
}

/// Records the resolution of one offloaded model call: the call counter and
/// latency histogram, the `outcome`/`error`/token fields on `span`, and a warn
/// log when the call failed.
pub(crate) fn record_llm_call(
    algorithm: &str,
    selected_model: &str,
    tier: Option<&str>,
    is_routed: bool,
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

    if is_routed {
        TOTAL_REQUESTS.fetch_add(1, Ordering::Relaxed);
        let mut routed_attributes = vec![KeyValue::new("model", selected_model.to_string())];
        if let Some(tier) = tier {
            routed_attributes.push(KeyValue::new("tier", tier.to_string()));
        }
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
        Err(error) => {
            span.record("error", tracing::field::display(error));
            tracing::warn!(
                target: TRACING_TARGET,
                algorithm,
                selected_model,
                error = %error,
                "model call failed"
            );
        }
    }
}

/// Records one published routing decision: the decision counter plus a
/// structured debug event carrying the decision's reasoning.
pub(crate) fn record_decision(ctx: &Context, decision: &dyn Decision) {
    let algorithm = algorithm_label(ctx);
    let selected_model = decision.selected_model();
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
