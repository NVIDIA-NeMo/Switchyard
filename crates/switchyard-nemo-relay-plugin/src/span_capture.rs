// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context as TaskContext, Poll};
use std::time::{Instant, SystemTime};

use nemo_relay_plugin::{Json, PluginRuntime, ScopeType};
use serde_json::{Map, Value};
use switchyard_llm_client::with_payload_capture;
use switchyard_runner::ProviderKeyRedactor;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Metadata, Subscriber};
use tracing_subscriber::Layer;
use tracing_subscriber::filter::{LevelFilter, filter_fn};
use tracing_subscriber::layer::Context;
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry::LookupSpan;

const CLIENT_CALL: &str = "libsy.client_call";
const UPSTREAM_ATTEMPT: &str = "libsy.upstream_attempt";
const PAYLOAD_TARGET: &str = "switchyard.payload";

type SpanEmitter = Box<dyn Fn(CapturedSpan) + Send + Sync>;

#[derive(Debug, Default)]
struct Payloads {
    request: Option<Json>,
    response: Option<Json>,
}

#[derive(Debug)]
pub(crate) struct CapturedSpan {
    name: &'static str,
    fields: Map<String, Value>,
    payloads: Payloads,
    started_at: SystemTime,
    ended_at: SystemTime,
    children: Vec<CapturedSpan>,
}

struct OpenSpan {
    name: &'static str,
    fields: Map<String, Value>,
    payloads: Payloads,
    started_at: SystemTime,
    started: Instant,
    children: Vec<CapturedSpan>,
}

struct CaptureLayer {
    emit: SpanEmitter,
    payload_limit: Option<usize>,
}

impl<S> Layer<S> for CaptureLayer
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        if event.metadata().target() != PAYLOAD_TARGET {
            return;
        }
        let Some(limit) = self.payload_limit else {
            return;
        };
        let Some(span) = ctx.event_span(event) else {
            return;
        };
        let is_routing = span.scope().any(|ancestor| {
            ancestor.extensions().get::<OpenSpan>().is_some_and(|open| {
                open.name == CLIENT_CALL
                    && open
                        .fields
                        .get("switchyard.call_phase")
                        .and_then(Value::as_str)
                        == Some("routing")
            })
        });
        if !is_routing || span.name() != UPSTREAM_ATTEMPT {
            return;
        }
        let mut payload = PayloadVisitor {
            phase: None,
            body: BoundedText {
                text: String::new(),
                truncated: false,
                limit,
            },
        };
        event.record(&mut payload);
        if let Some(open) = span.extensions_mut().get_mut::<OpenSpan>() {
            let data = payload.body.into_payload();
            match payload.phase {
                Some(PayloadPhase::Request) => open.payloads.request = Some(data),
                Some(PayloadPhase::Response) => open.payloads.response = Some(data),
                None => (),
            }
        }
    }

    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let name = attrs.metadata().name();
        let mut fields = Map::new();
        attrs.record(&mut FieldVisitor(&mut fields));
        if let Some(span) = ctx.span(id) {
            span.extensions_mut().insert(OpenSpan {
                name,
                fields,
                payloads: Payloads::default(),
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
            payloads,
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
            payloads,
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
        for payload in [&mut self.payloads.request, &mut self.payloads.response]
            .into_iter()
            .flatten()
        {
            *payload = redactor.value(std::mem::take(payload));
        }
        for child in &mut self.children {
            child.sanitize(redactor);
        }
    }

    pub(crate) fn emit(self, runtime: &PluginRuntime) -> Result<(), String> {
        let mut metadata = Json::Object(self.fields);
        let output = self.payloads.response.or_else(|| {
            if self.name == CLIENT_CALL {
                response_data(&metadata)
            } else {
                None
            }
        });
        if let Some(session_id) = metadata.get("gen_ai.conversation.id").cloned() {
            metadata["session_id"] = session_id;
        }
        if let Some(status) = span_status(&metadata) {
            metadata["otel.status_code"] = status["otel.status_code"].clone();
        }
        let input = self.payloads.request.or_else(|| {
            metadata
                .get("gen_ai.request.model")
                .map(|model| serde_json::json!({"model": model}))
        });
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
    pub(crate) fn new(
        payload_limit: Option<usize>,
        emit: impl Fn(CapturedSpan) + Send + Sync + 'static,
    ) -> Self {
        let layer = CaptureLayer {
            emit: Box::new(emit),
            payload_limit,
        }
        .with_filter(
            filter_fn(move |metadata| {
                captures(metadata)
                    || (payload_limit.is_some() && metadata.target() == PAYLOAD_TARGET)
            })
            .with_max_level_hint(if payload_limit.is_some() {
                LevelFilter::TRACE
            } else {
                LevelFilter::DEBUG
            }),
        );
        Self {
            dispatch: tracing::Dispatch::new(tracing_subscriber::registry().with(layer)),
        }
    }

    /// Polls a future with the capture subscriber as its default.
    pub(crate) async fn capture<T>(&self, future: impl Future<Output = T>) -> T {
        with_payload_capture(CaptureFuture {
            inner: Some(Box::pin(future)),
            dispatch: self.dispatch.clone(),
        })
        .await
    }
}

struct PayloadVisitor {
    phase: Option<PayloadPhase>,
    body: BoundedText,
}

enum PayloadPhase {
    Request,
    Response,
}

impl Visit for PayloadVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "phase" {
            self.phase = match value {
                "request" => Some(PayloadPhase::Request),
                "response" => Some(PayloadPhase::Response),
                _ => None,
            };
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        if field.name() == "body" {
            let _ = fmt::write(&mut self.body, format_args!("{value:?}"));
        }
    }
}

struct BoundedText {
    text: String,
    truncated: bool,
    limit: usize,
}

impl BoundedText {
    fn into_payload(self) -> Json {
        // A retained prefix could end inside a credential before redaction runs.
        let mut payload = serde_json::json!({"body": null, "truncated": self.truncated});
        if !self.truncated {
            payload["body"] = self.text.into();
        }
        payload
    }
}

impl fmt::Write for BoundedText {
    fn write_str(&mut self, value: &str) -> fmt::Result {
        if value.len() > self.limit - self.text.len() {
            self.truncated = true;
            return Err(fmt::Error);
        }
        self.text.push_str(value);
        Ok(())
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
            | "gen_ai.request.stream"
            | "gen_ai.request.temperature"
            | "gen_ai.request.top_p"
            | "gen_ai.request.top_k"
            | "gen_ai.request.max_tokens"
            | "gen_ai.request.reasoning.level"
            | "gen_ai.output.type"
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

    // Real loopback HTTP proves capture sees the encoded judge body, not the outer task.
    #[tokio::test]
    async fn captures_final_judge_payload_only_when_enabled() {
        use crate::{config::SwitchyardConfig, runtime::SwitchyardRuntime};
        use switchyard_protocol::{Request, WireFormat, text_request};
        use wiremock::matchers::{body_partial_json, method};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        for (model, content) in [
            ("judge/model", r#"{"target":"efficient"}"#),
            ("answer/model", "answer"),
        ] {
            Mock::given(method("POST"))
                .and(body_partial_json(serde_json::json!({"model": model})))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "id": "smoke", "model": model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]
                })))
                .mount(&server).await;
        }
        let config: SwitchyardConfig = serde_json::from_value(serde_json::json!({
            "switchyard_config": {
                "schema_version": 1,
                "llm_clients": {"local": {"format": "openai_chat", "base_url": format!("{}/v1", server.uri())}},
                "targets": {
                    "judge": {"id": "judge/model", "llm_client": "local", "extra_body": {"debug_canary": "final-body-only"}},
                    "answer": {"id": "answer/model", "llm_client": "local"}
                },
                "routes": {"test": {
                    "id": "switchyard/test", "type": "llm_classifier", "mode": "custom",
                    "models": {"judge": ["judge"], "efficient": ["answer"], "any": ["answer"]},
                    "default_target": "efficient", "prompt": "Judge task complexity.",
                    "response_schema": r#"{"type":"object","properties":{"target":{"type":"string","enum":["efficient"]}},"required":["target"],"additionalProperties":false}"#,
                    "policy": {"type": "target_selector", "selector": "/target"}
                }}
            }
        })).unwrap();
        let runtime = SwitchyardRuntime::new(config).unwrap();
        for enabled in [false, true] {
            let (sender, spans) = mpsc::channel();
            let capture = SpanCapture::new(
                enabled.then_some(crate::config::DEFAULT_ROUTING_PAYLOAD_BYTES),
                move |span| sender.send(span).unwrap(),
            );
            let execution = capture
                .capture(runtime.execute_buffered(
                    WireFormat::OpenAiChat,
                    Request {
                        llm_request: text_request(
                            Some("switchyard/test".into()),
                            "Fix a failing Rust test.",
                        ),
                        raw_request: None,
                        metadata: None,
                    },
                ))
                .await;
            assert!(execution.result.is_ok(), "{:?}", execution.result);
            let received = server.received_requests().await.unwrap();
            let mut spans: Vec<_> = spans.try_iter().collect();
            let judge = spans
                .iter_mut()
                .find(|span| span.fields["selected_model"] == "judge/model")
                .unwrap();
            assert_eq!(judge.fields["switchyard.call_phase"], "routing");
            assert_eq!(judge.children.len(), 1);
            let attempt = &mut judge.children[0];
            assert_eq!(attempt.payloads.request.is_some(), enabled);
            assert_eq!(attempt.payloads.response.is_some(), enabled);
            if enabled {
                let actual = received
                    .into_iter()
                    .map(|r| r.body_json::<Value>().unwrap())
                    .find(|body| body["model"] == "judge/model")
                    .unwrap();
                let captured: Value = serde_json::from_str(
                    attempt.payloads.request.as_ref().unwrap()["body"]
                        .as_str()
                        .unwrap(),
                )
                .unwrap();
                assert_eq!(captured, actual);
                assert_eq!(captured["debug_canary"], "final-body-only");
                assert_eq!(captured["messages"][0]["content"], "Judge task complexity.");
                assert_eq!(
                    attempt.payloads.request.as_ref().unwrap()["truncated"],
                    false
                );
                assert!(
                    attempt.payloads.response.as_ref().unwrap()["body"]
                        .as_str()
                        .unwrap()
                        .contains("efficient")
                );
                judge.sanitize(&ProviderKeyRedactor::new(&["final-body-only".into()]));
                assert!(!format!("{judge:?}").contains("final-body-only"));
            }
            let answer = spans
                .iter()
                .find(|span| span.fields["selected_model"] == "answer/model")
                .unwrap();
            for attempt in &answer.children {
                assert!(attempt.payloads.request.is_none());
                assert!(attempt.payloads.response.is_none());
            }
        }
    }

    #[test]
    fn payload_limit_preserves_complete_bodies_and_omits_oversized_ones() {
        use std::fmt::Write;
        for (parts, limit, expected) in [
            (["plain", " text"], 10, Some("plain text")),
            (["caf", "é"], 5, Some("café")),
            (["caf", "é"], 4, None),
            (["prefix-", "synthetic-secret"], 16, None),
        ] {
            let mut body = BoundedText {
                text: String::new(),
                truncated: false,
                limit,
            };
            let result = write!(&mut body, "{}{}", parts[0], parts[1]);
            assert_eq!(result.is_err(), expected.is_none());
            assert!(body.text.len() <= limit);
            assert_eq!(
                body.into_payload(),
                serde_json::json!({"body": expected, "truncated": expected.is_none()})
            );
        }
    }

    struct PanicOnDebug;

    impl fmt::Debug for PanicOnDebug {
        fn fmt(&self, _formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            panic!("disallowed fields must not be formatted");
        }
    }

    fn recorder() -> (SpanCapture, mpsc::Receiver<CapturedSpan>) {
        let (sender, receiver) = mpsc::channel();
        let capture = SpanCapture::new(None, move |span| sender.send(span).unwrap());
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
