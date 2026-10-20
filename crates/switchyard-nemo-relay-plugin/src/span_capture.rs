// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context as TaskContext, Poll};
use std::time::{Instant, SystemTime};

use nemo_relay_plugin::{Json, PluginRuntime, ScopeType};
use serde_json::{Map, Value};
use switchyard_runner::ProviderKeyRedactor;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::{Metadata, Subscriber};
use tracing_subscriber::Layer;
use tracing_subscriber::filter::{LevelFilter, filter_fn};
use tracing_subscriber::layer::Context;
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry::LookupSpan;

const CLIENT_CALL: &str = "libsy.client_call";
const UPSTREAM_ATTEMPT: &str = "libsy.upstream_attempt";

type SpanEmitter = Box<dyn Fn(CapturedSpan) + Send + Sync>;

#[derive(Debug)]
pub(crate) struct CapturedSpan {
    name: &'static str,
    fields: Map<String, Value>,
    started_at: SystemTime,
    ended_at: SystemTime,
    children: Vec<CapturedSpan>,
}

struct OpenSpan {
    name: &'static str,
    fields: Map<String, Value>,
    started_at: SystemTime,
    started: Instant,
    children: Vec<CapturedSpan>,
}

struct CaptureLayer {
    emit: SpanEmitter,
}

impl<S> Layer<S> for CaptureLayer
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let name = attrs.metadata().name();
        let mut fields = Map::new();
        attrs.record(&mut FieldVisitor(&mut fields));
        if let Some(span) = ctx.span(id) {
            span.extensions_mut().insert(OpenSpan {
                name,
                fields,
                started_at: SystemTime::now(),
                started: Instant::now(),
                children: Vec::new(),
            });
        }
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else {
            return;
        };
        let mut extensions = span.extensions_mut();
        if let Some(open) = extensions.get_mut::<OpenSpan>() {
            values.record(&mut FieldVisitor(&mut open.fields));
        }
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(&id) else {
            return;
        };
        let parent_id = span.parent().map(|parent| parent.id().clone());
        let Some(open) = span.extensions_mut().remove::<OpenSpan>() else {
            return;
        };
        let OpenSpan {
            name,
            mut fields,
            started_at,
            started,
            children,
        } = open;
        // A span that closes before its terminal bookkeeping ran was cancelled.
        fields
            .entry("outcome")
            .or_insert_with(|| Value::String("cancelled".into()));
        let captured = CapturedSpan {
            name,
            fields,
            started_at,
            ended_at: started_at
                .checked_add(started.elapsed())
                .unwrap_or_else(SystemTime::now),
            children,
        };

        if captured.name == UPSTREAM_ATTEMPT
            && let Some(parent) = parent_id.and_then(|parent| ctx.span(&parent))
            && let Some(open) = parent.extensions_mut().get_mut::<OpenSpan>()
        {
            open.children.push(captured);
            return;
        }
        (self.emit)(captured);
    }
}

impl CapturedSpan {
    pub(crate) fn sanitize(&mut self, redactor: &ProviderKeyRedactor) {
        if let Json::Object(fields) = redactor.value(Json::Object(std::mem::take(&mut self.fields)))
        {
            self.fields = fields;
        }
        for child in &mut self.children {
            child.sanitize(redactor);
        }
    }

    pub(crate) fn emit(self, runtime: &PluginRuntime) -> Result<(), String> {
        let mut metadata = Json::Object(self.fields);
        let output = if self.name == CLIENT_CALL {
            response_data(&metadata)
        } else {
            None
        };
        if let Some(session_id) = metadata.get("gen_ai.conversation.id").cloned() {
            metadata["session_id"] = session_id;
        }
        if let Some(status) = span_status(&metadata) {
            metadata["otel.status_code"] = status["otel.status_code"].clone();
        }
        let input = metadata
            .get("gen_ai.request.model")
            .map(|model| serde_json::json!({"model": model}));
        let mut scope = runtime.scope_at(
            self.name,
            relay_scope_type(self.name),
            None,
            Some(&metadata),
            input.as_ref(),
            self.started_at,
        )?;
        let mut child_result = Ok(());
        for child in self.children {
            child_result = child_result.and(child.emit(runtime));
        }
        let close_result = scope.close_at(output.as_ref(), Some(&metadata), self.ended_at);
        child_result.and(close_result)
    }
}

fn response_data(fields: &Json) -> Option<Json> {
    let mut output = Map::new();
    for (source, target) in [
        ("gen_ai.response.id", "id"),
        ("gen_ai.response.model", "model"),
    ] {
        if let Some(value) = fields.get(source) {
            output.insert(target.into(), value.clone());
        }
    }
    let mut usage = Map::new();
    for (source, target) in [
        ("gen_ai.usage.output_tokens", "output_tokens"),
        (
            "gen_ai.usage.cache_read.input_tokens",
            "cache_read_input_tokens",
        ),
        (
            "gen_ai.usage.cache_creation.input_tokens",
            "cache_creation_input_tokens",
        ),
    ] {
        if let Some(value) = fields.get(source) {
            usage.insert(target.into(), value.clone());
        }
    }
    if let Some(input) = fields
        .get("gen_ai.usage.input_tokens")
        .and_then(Json::as_u64)
    {
        let cached = usage
            .get("cache_read_input_tokens")
            .and_then(Json::as_u64)
            .unwrap_or(0);
        let created = usage
            .get("cache_creation_input_tokens")
            .and_then(Json::as_u64)
            .unwrap_or(0);
        usage.insert(
            "input_tokens".into(),
            input.saturating_sub(cached).saturating_sub(created).into(),
        );
    }
    if !usage.is_empty() {
        output.insert("usage".into(), Json::Object(usage));
    }
    if output.is_empty() {
        return None;
    }
    // Relay decodes scope output with provider codecs. A content-free Messages
    // response preserves separate cache-read and cache-creation token counts.
    output.insert("type".into(), Json::String("message".into()));
    output.insert("role".into(), Json::String("assistant".into()));
    output.insert("content".into(), Json::Array(Vec::new()));
    Some(Json::Object(output))
}

/// Captures Switchyard client spans for one plugin registration.
#[derive(Clone)]
pub(crate) struct SpanCapture {
    dispatch: tracing::Dispatch,
}

impl SpanCapture {
    pub(crate) fn new(emit: impl Fn(CapturedSpan) + Send + Sync + 'static) -> Self {
        let layer = CaptureLayer {
            emit: Box::new(emit),
        }
        .with_filter(filter_fn(captures).with_max_level_hint(LevelFilter::DEBUG));
        Self {
            dispatch: tracing::Dispatch::new(tracing_subscriber::registry().with(layer)),
        }
    }

    /// Polls a future with the capture subscriber as its default.
    pub(crate) async fn capture<T>(&self, future: impl Future<Output = T>) -> T {
        CaptureFuture {
            inner: Some(Box::pin(future)),
            dispatch: self.dispatch.clone(),
        }
        .await
    }
}

// Standard subscriber wrappers restore the ambient dispatch before drop. This
// wrapper also drops the inner future under capture so cancelled spans close here.
struct CaptureFuture<F> {
    inner: Option<Pin<Box<F>>>,
    dispatch: tracing::Dispatch,
}

impl<F: Future> Future for CaptureFuture<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let Some(inner) = this.inner.as_mut() else {
            return Poll::Pending;
        };
        tracing::dispatcher::with_default(&this.dispatch, || inner.as_mut().poll(cx))
    }
}

impl<F> Drop for CaptureFuture<F> {
    fn drop(&mut self) {
        let Some(inner) = self.inner.take() else {
            return;
        };
        tracing::dispatcher::with_default(&self.dispatch, || drop(inner));
    }
}

struct FieldVisitor<'a>(&'a mut Map<String, Value>);

impl FieldVisitor<'_> {
    fn insert(&mut self, field: &Field, value: impl FnOnce() -> Value) {
        if is_safe_field(field.name()) {
            self.0.insert(field.name().into(), value());
        }
    }
}

impl Visit for FieldVisitor<'_> {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.insert(field, || Value::String(format!("{value:?}")));
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.insert(field, || Value::String(value.into()));
    }

    fn record_bool(&mut self, field: &Field, value: bool) {
        self.insert(field, || Value::Bool(value));
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.insert(field, || Value::from(value));
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.insert(field, || Value::from(value));
    }

    fn record_f64(&mut self, field: &Field, value: f64) {
        self.insert(field, || Value::from(value));
    }
}

fn captures(metadata: &Metadata<'_>) -> bool {
    metadata.target() == "libsy" && matches!(metadata.name(), CLIENT_CALL | UPSTREAM_ATTEMPT)
}

fn relay_scope_type(name: &str) -> ScopeType {
    if name == CLIENT_CALL {
        ScopeType::Llm
    } else {
        ScopeType::Custom
    }
}

fn is_safe_field(name: &str) -> bool {
    matches!(
        name,
        "algorithm"
            | "switchyard.candidate"
            | "switchyard.candidate_count"
            | "switchyard.call_phase"
            | "selected_model"
            | "gen_ai.request.model"
            | "gen_ai.conversation.id"
            | "gen_ai.response.id"
            | "gen_ai.response.model"
            | "gen_ai.usage.input_tokens"
            | "gen_ai.usage.output_tokens"
            | "gen_ai.usage.cache_read.input_tokens"
            | "gen_ai.usage.cache_creation.input_tokens"
            | "gen_ai.usage.reasoning.output_tokens"
            | "model"
            | "wire_format"
            | "attempt"
            | "max_attempts"
            | "retry"
            | "outcome"
            | "status_code"
            | "will_retry"
            | "retry_delay_ms"
            | "otel.status_code"
            | "error.type"
    )
}

fn span_status(metadata: &Json) -> Option<Json> {
    let outcome = metadata.get("outcome").and_then(Json::as_str);
    let status = metadata.get("otel.status_code").and_then(Json::as_str);
    match (status, outcome) {
        (Some("ERROR"), _) | (_, Some("error" | "cancelled")) => {
            Some(serde_json::json!({"otel.status_code": "ERROR"}))
        }
        (Some("OK"), _) | (_, Some("ok")) => Some(serde_json::json!({"otel.status_code": "OK"})),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::fmt;
    use std::sync::mpsc;

    use tracing::Instrument;

    use super::*;

    struct PanicOnDebug;

    impl fmt::Debug for PanicOnDebug {
        fn fmt(&self, _formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            panic!("disallowed fields must not be formatted");
        }
    }

    fn recorder() -> (SpanCapture, mpsc::Receiver<CapturedSpan>) {
        let (sender, receiver) = mpsc::channel();
        let capture = SpanCapture::new(move |span| sender.send(span).unwrap());
        (capture, receiver)
    }

    #[tokio::test]
    async fn captures_only_allowlisted_client_and_attempt_fields() {
        let (capture, captured) = recorder();
        capture
            .capture(async {
                drop(tracing::info_span!(target: "other", CLIENT_CALL));
                let call = tracing::info_span!(
                    target: "libsy",
                    CLIENT_CALL,
                    selected_model = "target/model",
                    switchyard.call_phase = "routing",
                    outcome = tracing::field::Empty,
                    error = ?PanicOnDebug,
                );
                {
                    let _entered = call.enter();
                    drop(tracing::debug_span!(
                        target: "libsy",
                        UPSTREAM_ATTEMPT,
                        model = "target/model",
                        attempt = 1_u64,
                        outcome = "error",
                        error = "provider body",
                    ));
                    call.record("outcome", "ok");
                }
                drop(call);
            })
            .await;

        let mut call = captured.try_recv().expect("expected one client call");
        assert!(captured.try_recv().is_err());
        assert_eq!(call.name, CLIENT_CALL);
        assert!(matches!(relay_scope_type(call.name), ScopeType::Llm));
        assert_eq!(call.fields["selected_model"], "target/model");
        assert_eq!(call.fields["switchyard.call_phase"], "routing");
        assert_eq!(call.fields["outcome"], "ok");
        assert!(!call.fields.contains_key("error"));
        let [attempt] = call.children.as_slice() else {
            panic!("expected one upstream attempt: {call:?}");
        };
        assert_eq!(attempt.name, UPSTREAM_ATTEMPT);
        assert!(matches!(relay_scope_type(attempt.name), ScopeType::Custom));
        assert_eq!(attempt.fields["model"], "target/model");
        assert_eq!(attempt.fields["attempt"], 1);
        assert_eq!(attempt.fields["outcome"], "error");
        assert!(!attempt.fields.contains_key("error"));
        assert!(call.started_at <= attempt.started_at);
        assert!(attempt.ended_at <= call.ended_at);

        call.sanitize(&ProviderKeyRedactor::new(&["target/model".into()]));
        assert_eq!(call.fields["selected_model"], "[REDACTED]");
        assert_eq!(call.children[0].fields["model"], "[REDACTED]");
    }

    #[tokio::test]
    async fn captures_spans_that_close_after_the_scoped_future_returns() {
        let (capture, captured) = recorder();
        let call = capture
            .capture(async {
                let call = tracing::info_span!(target: "libsy", CLIENT_CALL);
                drop(tracing::debug_span!(target: "libsy", parent: &call, UPSTREAM_ATTEMPT));
                call
            })
            .await;

        assert!(captured.try_recv().is_err());
        drop(capture);
        drop(call);

        let call = captured
            .try_recv()
            .expect("expected one deferred client call");
        assert_eq!(call.children.len(), 1);
    }

    #[tokio::test]
    async fn one_capture_handles_concurrent_futures() {
        let (capture, captured) = recorder();

        tokio::join!(
            capture.capture(async {
                tokio::task::yield_now().await;
                drop(tracing::info_span!(target: "libsy", CLIENT_CALL, selected_model = "first"));
            }),
            capture.capture(async {
                tokio::task::yield_now().await;
                drop(tracing::info_span!(target: "libsy", CLIENT_CALL, selected_model = "second"));
            }),
        );

        let mut models = captured
            .try_iter()
            .map(|span| span.fields["selected_model"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        models.sort_unstable();
        assert_eq!(models, ["first", "second"]);
    }

    #[tokio::test]
    async fn dropped_in_flight_spans_are_cancelled() {
        let (capture, captured) = recorder();

        let mut future = Box::pin(capture.capture(async move {
            let call = tracing::info_span!(target: "libsy", CLIENT_CALL);
            let attempt = tracing::debug_span!(target: "libsy", parent: &call, UPSTREAM_ATTEMPT);
            std::future::pending::<()>()
                .instrument(attempt)
                .instrument(call)
                .await;
        }));

        assert!(futures_util::poll!(&mut future).is_pending());
        drop(future);

        let call = captured
            .try_recv()
            .expect("expected one cancelled client call");
        assert_eq!(call.fields["outcome"], "cancelled");
        assert_eq!(
            span_status(&Json::Object(call.fields.clone())),
            Some(serde_json::json!({"otel.status_code": "ERROR"}))
        );
        let [attempt] = call.children.as_slice() else {
            panic!("expected one cancelled upstream attempt: {call:?}");
        };
        assert_eq!(attempt.fields["outcome"], "cancelled");
    }
}
