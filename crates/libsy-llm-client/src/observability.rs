// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `tracing` span fields for the provider call [`run`](crate::run) makes.
//!
//! libsy instruments the orchestration layer (`libsy.run`, `libsy.llm_call`) and owns the
//! `switchyard.*` metrics. This module owns the other half: the `libsy.client_call` span that
//! covers the one HTTP call a routed request performs, recorded as OpenTelemetry gen_ai
//! semantic-convention attributes.
//!
//! Nothing here records a metric — these are span fields only, written through the `tracing`
//! facade so the host's subscriber decides where they go. A streamed response resolves before
//! its usage is known, so [`observe_client_call`] wraps the stream and keeps the span alive
//! until it drains, errors, or is abandoned.

use std::borrow::Cow;
use std::pin::Pin;
use std::task::{Context as TaskContext, Poll};

use futures::Stream;
use opentelemetry::{Array as OtelArray, StringValue, Value as OtelValue};
use switchyard_libsy::{LibsyError, Result};
use switchyard_protocol::{
    AggLlmResponse, LlmClientError, LlmRequest, LlmResponse, LlmResponseChunk, LlmResponseStream,
    LlmResponseStreamEvent, Response, StopReason, Usage,
};
use tracing::Span;
use tracing_opentelemetry::OpenTelemetrySpanExt;

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
            outcome: Outcome::Open,
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
    type Item = std::result::Result<LlmResponseStreamEvent, LlmClientError>;

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

/// What the span has recorded for this stream so far.
#[derive(PartialEq)]
enum Outcome {
    /// Still streaming. Dropping the response now is a cancellation.
    Open,
    /// The model finished its message, so `outcome` is already `ok` and dropping the
    /// response is no longer a cancellation. Observation continues: OpenAI Chat sends usage
    /// in a chunk *after* the one carrying `finish_reason`, so stopping here would lose it.
    Completed,
    /// An error is recorded and terminal; nothing later is worth observing.
    Failed,
}

struct ClientStreamObserver {
    span: Span,
    outcome: Outcome,
}

impl ClientStreamObserver {
    /// Records what `item` says about the response. Returns whether the stream has failed —
    /// the only state in which the caller should stop observing.
    fn observe(
        &mut self,
        item: &std::result::Result<LlmResponseStreamEvent, LlmClientError>,
    ) -> bool {
        match item {
            Ok(event) => {
                for chunk in event.normalized() {
                    self.observe_chunk(chunk);
                    if self.outcome == Outcome::Failed {
                        break;
                    }
                }
            }
            Err(error) => {
                let error_type = llm_client_error_type(error);
                record_client_error(&self.span, &error_type, error);
                self.outcome = Outcome::Failed;
            }
        }
        self.outcome == Outcome::Failed
    }

    fn observe_chunk(&mut self, chunk: &LlmResponseChunk) {
        match chunk {
            LlmResponseChunk::MessageStart { id, model } => {
                record_optional(&self.span, "gen_ai.response.id", id.as_deref());
                record_optional(&self.span, "gen_ai.response.model", model.as_deref());
            }
            LlmResponseChunk::Usage(usage) => record_gen_ai_usage(&self.span, usage),
            LlmResponseChunk::MessageStop { reason } => {
                record_finish_reasons(&self.span, reason.iter().cloned());
                self.complete();
            }
            LlmResponseChunk::DecodeError { message } => {
                record_client_error(&self.span, "response_translation", message);
                self.outcome = Outcome::Failed;
            }
            LlmResponseChunk::StreamError { message } => {
                record_client_error(&self.span, "502", message);
                self.outcome = Outcome::Failed;
            }
            _ => {}
        }
    }

    /// Records the normal terminal outcome, unless an error already claimed it.
    fn complete(&mut self) {
        if self.outcome == Outcome::Open {
            self.span.record("outcome", "ok");
            self.outcome = Outcome::Completed;
        }
    }
}

impl Drop for ClientStreamObserver {
    fn drop(&mut self) {
        if self.outcome == Outcome::Open {
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
