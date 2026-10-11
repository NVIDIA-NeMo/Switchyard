// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use std::task::{Context as TaskContext, Poll};
use std::time::{Instant, SystemTime};

use nemo_relay_plugin::{Json, PluginRuntime, ScopeType};
use serde_json::{Map, Value};
use switchyard_runner::ProviderKeyRedactor;
use tracing::field::{Field, Visit};
use tracing::metadata::LevelFilter;
use tracing::span::{Attributes, Id, Record};
use tracing::subscriber::Interest;
use tracing::{Metadata, Subscriber};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::Context;
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry::LookupSpan;

const CLIENT_CALL: &str = "libsy.client_call";
const UPSTREAM_ATTEMPT: &str = "libsy.upstream_attempt";

type SpanEmitter = Arc<dyn Fn(CapturedSpan) + Send + Sync>;

// One shared dispatch avoids global callsite registration on every request. Each
// open span keeps its request emitter so a stream can close after this scope ends.
tokio::task_local! {
    static SPAN_EMITTER: SpanEmitter;
}

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
    emit: SpanEmitter,
}

struct CaptureLayer;

impl<S> Layer<S> for CaptureLayer
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn register_callsite(&self, metadata: &'static Metadata<'static>) -> Interest {
        if captures(metadata) {
            Interest::always()
        } else {
            Interest::never()
        }
    }

    fn enabled(&self, metadata: &Metadata<'_>, _ctx: Context<'_, S>) -> bool {
        captures(metadata)
    }

    fn max_level_hint(&self) -> Option<LevelFilter> {
        Some(LevelFilter::DEBUG)
    }

    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let Ok(emit) = SPAN_EMITTER.try_with(Arc::clone) else {
            return;
        };
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
                emit,
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
            emit,
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
        emit(captured);
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
        let metadata = Json::Object(self.fields);
        let status = span_status(&metadata);
        let mut scope = runtime.scope_at(
            self.name,
            relay_scope_type(self.name),
            None,
            Some(&metadata),
            None,
            self.started_at,
        )?;
        let mut child_result = Ok(());
        for child in self.children {
            child_result = child_result.and(child.emit(runtime));
        }
        let close_result = scope.close_at(None, status.as_ref(), self.ended_at);
        child_result.and(close_result)
    }
}

/// Captures Switchyard client spans while polling one route execution.
///
/// This capture dispatch temporarily replaces any subscriber configured inside the plugin cdylib.
pub(crate) async fn capture_client_spans<T>(
    future: impl Future<Output = T>,
    emit: impl Fn(CapturedSpan) + Send + Sync + 'static,
) -> T {
    SPAN_EMITTER
        .scope(Arc::new(emit), CaptureFuture::new(future))
        .await
}

struct CaptureFuture<F> {
    inner: Option<Pin<Box<F>>>,
    dispatch: &'static tracing::Dispatch,
}

impl<F> CaptureFuture<F> {
    fn new(inner: F) -> Self {
        Self {
            inner: Some(Box::pin(inner)),
            dispatch: capture_dispatch(),
        }
    }
}

impl<F: Future> Future for CaptureFuture<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let Some(inner) = this.inner.as_mut() else {
            return Poll::Pending;
        };
        tracing::dispatcher::with_default(this.dispatch, || inner.as_mut().poll(cx))
    }
}

impl<F> Drop for CaptureFuture<F> {
    fn drop(&mut self) {
        let Some(inner) = self.inner.take() else {
            return;
        };
        // Releasing a child-held parent reference uses the current dispatch.
        tracing::dispatcher::with_default(self.dispatch, || drop(inner));
    }
}

fn capture_dispatch() -> &'static tracing::Dispatch {
    static DISPATCH: OnceLock<tracing::Dispatch> = OnceLock::new();
    DISPATCH
        .get_or_init(|| tracing::Dispatch::new(tracing_subscriber::registry().with(CaptureLayer)))
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
    use std::sync::Mutex;

    use tracing::Instrument;

    use super::*;

    struct PanicOnDebug;

    impl fmt::Debug for PanicOnDebug {
        fn fmt(&self, _formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            panic!("disallowed fields must not be formatted");
        }
    }

    #[tokio::test]
    async fn captures_only_allowlisted_client_and_attempt_fields() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let observed = Arc::clone(&captured);
        let result = capture_client_spans(
            async {
                async {}
                    .instrument(tracing::info_span!(
                        target: "other",
                        CLIENT_CALL,
                        outcome = "ignored",
                    ))
                    .await;
                async {
                    async {}
                        .instrument(tracing::debug_span!(
                            target: "libsy",
                            UPSTREAM_ATTEMPT,
                            model = "target/model",
                            attempt = 1_u64,
                            outcome = "error",
                            error = "provider body",
                        ))
                        .await;
                    tracing::Span::current().record("outcome", "ok");
                }
                .instrument(tracing::info_span!(
                    target: "libsy",
                    CLIENT_CALL,
                    selected_model = "target/model",
                    switchyard.call_phase = "routing",
                    outcome = tracing::field::Empty,
                    error = ?PanicOnDebug,
                ))
                .await;
                7
            },
            move |span| {
                observed
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .push(span);
            },
        )
        .await;

        assert_eq!(result, 7);
        let mut spans = captured
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let [call] = spans.as_mut_slice() else {
            panic!("expected one client call: {spans:?}");
        };
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
        let captured = Arc::new(Mutex::new(Vec::new()));
        let observed = Arc::clone(&captured);
        let call = capture_client_spans(
            async {
                let call = tracing::info_span!(
                    target: "libsy",
                    CLIENT_CALL,
                    selected_model = "target/model",
                    outcome = "cancelled",
                );
                let attempt = tracing::debug_span!(
                    target: "libsy",
                    parent: &call,
                    UPSTREAM_ATTEMPT,
                    attempt = 1_u64,
                    outcome = "cancelled",
                );
                drop(attempt);
                call
            },
            move |span| {
                observed
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .push(span);
            },
        )
        .await;

        assert!(captured.lock().unwrap().is_empty());
        drop(call);

        let spans = captured.lock().unwrap();
        let [call] = spans.as_slice() else {
            panic!("expected one deferred client call: {spans:?}");
        };
        assert_eq!(call.fields["outcome"], "cancelled");
        assert_eq!(call.children.len(), 1);
        assert_eq!(call.children[0].fields["outcome"], "cancelled");
    }

    #[tokio::test]
    async fn concurrent_captures_keep_their_emitters_isolated() {
        let first = Arc::new(Mutex::new(Vec::new()));
        let first_observed = Arc::clone(&first);
        let second = Arc::new(Mutex::new(Vec::new()));
        let second_observed = Arc::clone(&second);

        tokio::join!(
            capture_client_spans(
                async {
                    tokio::task::yield_now().await;
                    async {}
                        .instrument(tracing::info_span!(
                            target: "libsy",
                            CLIENT_CALL,
                            selected_model = "first",
                        ))
                        .await;
                },
                move |span| first_observed.lock().unwrap().push(span),
            ),
            capture_client_spans(
                async {
                    tokio::task::yield_now().await;
                    async {}
                        .instrument(tracing::info_span!(
                            target: "libsy",
                            CLIENT_CALL,
                            selected_model = "second",
                        ))
                        .await;
                },
                move |span| second_observed.lock().unwrap().push(span),
            ),
        );

        let first = first.lock().unwrap();
        let second = second.lock().unwrap();
        let [first] = first.as_slice() else {
            panic!("expected one first client call: {first:?}");
        };
        let [second] = second.as_slice() else {
            panic!("expected one second client call: {second:?}");
        };
        assert_eq!(first.fields["selected_model"], "first");
        assert_eq!(second.fields["selected_model"], "second");
    }

    #[tokio::test]
    async fn dropped_in_flight_spans_are_cancelled() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let observed = Arc::clone(&captured);

        let mut future = Box::pin(capture_client_spans(
            async move {
                let call = tracing::info_span!(
                    target: "libsy",
                    CLIENT_CALL,
                    outcome = tracing::field::Empty,
                );
                async move {
                    let attempt = tracing::debug_span!(
                        target: "libsy",
                        UPSTREAM_ATTEMPT,
                        outcome = tracing::field::Empty,
                    );
                    async move {
                        std::future::pending::<()>().await;
                    }
                    .instrument(attempt)
                    .await;
                }
                .instrument(call)
                .await;
            },
            move |span| observed.lock().unwrap().push(span),
        ));

        assert!(futures_util::poll!(&mut future).is_pending());
        drop(future);

        let spans = captured.lock().unwrap();
        let [call] = spans.as_slice() else {
            panic!("expected one cancelled client call: {spans:?}");
        };
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
