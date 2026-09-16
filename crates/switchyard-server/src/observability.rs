// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-wide tracing and OpenTelemetry setup for server hosts.

use std::env;
use std::sync::OnceLock;

use axum::http::HeaderMap;
use opentelemetry::propagation::{Extractor, TextMapPropagator};
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::SdkTracerProvider;
use switchyard_protocol::{
    GenericKeys, LangfuseKeys, LlmRequest, LlmResponse, Response, ResponseOrigin,
};
use tracing_opentelemetry::OpenTelemetrySpanExt;
use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::util::SubscriberInitExt as _;
use tracing_subscriber::{EnvFilter, Layer as _};

use crate::{ServerError, ServerResult, metrics};

const DEFAULT_LOG_FILTER: &str = "info,opentelemetry=warn";
const DEFAULT_SERVICE_NAME: &str = "switchyard-server";

struct Observability {
    tracer_provider: Option<SdkTracerProvider>,
}

static OBSERVABILITY: OnceLock<Result<Observability, String>> = OnceLock::new();

/// Installs metrics and tracing once for either the binary or an embedded host.
pub fn initialize_observability() -> ServerResult<()> {
    match OBSERVABILITY.get_or_init(initialize) {
        Ok(_) => Ok(()),
        Err(error) => Err(ServerError::new(error.clone())),
    }
}

/// Flushes pending OTLP telemetry without shutting down process-wide providers.
pub fn flush_observability() {
    if let Some(Ok(observability)) = OBSERVABILITY.get()
        && let Some(provider) = &observability.tracer_provider
        && let Err(error) = provider.force_flush()
    {
        tracing::warn!(error = %error, "failed to flush OpenTelemetry traces");
    }
    metrics::flush();
}

/// Creates the server request span with any incoming W3C trace context as its parent.
pub(crate) fn request_span(headers: &HeaderMap) -> tracing::Span {
    let parent = TraceContextPropagator::new().extract(&HeaderExtractor(headers));
    let span = tracing::info_span!(
        target: "switchyard_server",
        "switchyard.request",
        otel.kind = "server",
        openinference.span.kind = "CHAIN",
        // Langfuse derives trace-level input/output from the root observation's
        // input/output, which map from these attributes.
        gen_ai.prompt = tracing::field::Empty,
        gen_ai.completion = tracing::field::Empty,
        // Shared route-selection vocabulary. Names must match GenericKeys / LangfuseKeys.
        switchyard.route.id = tracing::field::Empty,
        switchyard.routing.algorithm = tracing::field::Empty,
        switchyard.routing.selected_target = tracing::field::Empty,
        switchyard.response.served_target = tracing::field::Empty,
        switchyard.response.origin = tracing::field::Empty,
        langfuse.session.id = tracing::field::Empty,
        langfuse.user.id = tracing::field::Empty,
        langfuse.observation.metadata.switchyard.route_id = tracing::field::Empty,
        langfuse.observation.metadata.switchyard.algorithm = tracing::field::Empty,
        langfuse.observation.metadata.switchyard.selected_target = tracing::field::Empty,
        langfuse.observation.metadata.switchyard.served_target = tracing::field::Empty,
        langfuse.observation.metadata.switchyard.response_origin = tracing::field::Empty,
    );
    let _ = span.set_parent(parent);
    span
}

/// Records the request's messages on the root span so Langfuse populates trace-level input.
///
/// Mirrors the `gen_ai.prompt` attribute on the nested `libsy.client_call` span; Langfuse
/// maps both to observation input, and trace-level input derives from the root observation.
pub(crate) fn record_root_input(span: &tracing::Span, request: &LlmRequest) {
    if request.messages.is_empty() {
        return;
    }
    if let Ok(json) = serde_json::to_string(&request.messages) {
        span.record("gen_ai.prompt", json);
    }
}

/// Records inbound route, algorithm, non-empty session, and non-empty user id on the root span.
///
/// Empty sessions stay off `langfuse.session.id` and empty user ids off
/// `langfuse.user.id`. Call this after the route resolves and before execute, so
/// failures still identify the route without a fabricated terminal outcome.
pub(crate) fn record_root_route_context(
    span: &tracing::Span,
    route_id: Option<&str>,
    algorithm: &str,
    session_id: Option<&str>,
    user_id: Option<&str>,
) {
    record_pair(
        span,
        GenericKeys::ALGORITHM,
        LangfuseKeys::ALGORITHM,
        algorithm,
    );
    if let Some(route_id) = route_id {
        record_pair(
            span,
            GenericKeys::ROUTE_ID,
            LangfuseKeys::ROUTE_ID,
            route_id,
        );
    }
    if let Some(session_id) = session_id.filter(|session| !session.is_empty()) {
        span.record(LangfuseKeys::SESSION_ID, session_id);
    }
    if let Some(user_id) = user_id.filter(|user| !user.is_empty()) {
        span.record(LangfuseKeys::USER_ID, user_id);
    }
}

/// Records the successful terminal selection on the root span.
///
/// `served_target` is the target that actually served the answer. `None` means
/// Switchyard generated the response, so origin is `routing_generated` and
/// served target is left unset.
pub(crate) fn record_root_outcome(
    span: &tracing::Span,
    selected_target: &str,
    served_target: Option<&str>,
) {
    record_pair(
        span,
        GenericKeys::SELECTED_TARGET,
        LangfuseKeys::SELECTED_TARGET,
        selected_target,
    );
    match served_target {
        Some(served_target) => {
            record_pair(
                span,
                GenericKeys::SERVED_TARGET,
                LangfuseKeys::SERVED_TARGET,
                served_target,
            );
            record_pair(
                span,
                GenericKeys::RESPONSE_ORIGIN,
                LangfuseKeys::RESPONSE_ORIGIN,
                ResponseOrigin::Target.as_str(),
            );
        }
        None => record_pair(
            span,
            GenericKeys::RESPONSE_ORIGIN,
            LangfuseKeys::RESPONSE_ORIGIN,
            ResponseOrigin::RoutingGenerated.as_str(),
        ),
    }
}

/// Writes the same value to a generic key and its Langfuse observation metadata key.
fn record_pair(
    span: &tracing::Span,
    generic_key: &'static str,
    langfuse_key: &'static str,
    value: &str,
) {
    span.record(generic_key, value);
    span.record(langfuse_key, value);
}

/// Records a response's output on the root span so Langfuse populates trace-level output.
///
/// Only a buffered (`Agg`) response has its output here: a streamed response's body is drained
/// by the host after this span closes, so its output is intentionally left to the nested
/// `libsy.client_call` generation observation rather than extending this span. The streamed
/// output is visible as a child observation, but a streamed request's trace-level output is
/// not captured.
pub(crate) fn record_root_output(span: &tracing::Span, response: &Response) {
    if let LlmResponse::Agg(agg) = &response.llm_response {
        if agg.outputs.is_empty() {
            return;
        }
        if let Ok(json) = serde_json::to_string(&agg.outputs) {
            span.record("gen_ai.completion", json);
        }
    }
}

struct HeaderExtractor<'a>(&'a HeaderMap);

impl Extractor for HeaderExtractor<'_> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|value| value.to_str().ok())
    }

    fn keys(&self) -> Vec<&str> {
        self.0.keys().map(|name| name.as_str()).collect()
    }
}

pub(crate) fn otlp_enabled(signal: &str) -> bool {
    if env_var_is_true("OTEL_SDK_DISABLED") {
        return false;
    }
    if env::var(format!("OTEL_{signal}_EXPORTER"))
        .ok()
        .filter(|value| !value.trim().is_empty())
        .is_some_and(|value| {
            !value
                .split(',')
                .any(|exporter| exporter.trim().eq_ignore_ascii_case("otlp"))
        })
    {
        return false;
    }
    [
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        &format!("OTEL_EXPORTER_OTLP_{signal}_ENDPOINT"),
    ]
    .into_iter()
    .any(|name| env::var(name).is_ok_and(|value| !value.trim().is_empty()))
}

pub(crate) fn resource() -> Resource {
    let service_name = env::var("OTEL_SERVICE_NAME")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_SERVICE_NAME.to_string());
    Resource::builder().with_service_name(service_name).build()
}

fn initialize() -> Result<Observability, String> {
    metrics::registry()?;

    let tracer_provider = otlp_enabled("TRACES")
        .then(build_tracer_provider)
        .transpose()?;
    let filter = log_filter()?;
    let format = tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .with_writer(std::io::stderr)
        .with_filter(filter);

    if let Some(provider) = &tracer_provider {
        let tracer = provider.tracer("switchyard");
        tracing_subscriber::registry()
            .with(format)
            .with(
                tracing_opentelemetry::layer()
                    .with_tracer(tracer)
                    .with_filter(log_filter()?),
            )
            .try_init()
            .map_err(|error| format!("failed to initialize tracing: {error}"))?;
    } else {
        tracing_subscriber::registry()
            .with(format)
            .try_init()
            .map_err(|error| format!("failed to initialize tracing: {error}"))?;
    }

    Ok(Observability { tracer_provider })
}

fn log_filter() -> Result<EnvFilter, String> {
    EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(DEFAULT_LOG_FILTER))
        .map_err(|error| format!("invalid tracing filter: {error}"))
}

fn build_tracer_provider() -> Result<SdkTracerProvider, String> {
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .build()
        .map_err(|error| format!("failed to initialize OTLP trace exporter: {error}"))?;
    let provider = SdkTracerProvider::builder()
        .with_resource(resource())
        .with_batch_exporter(exporter)
        .build();
    opentelemetry::global::set_tracer_provider(provider.clone());
    Ok(provider)
}

fn env_var_is_true(name: &str) -> bool {
    env::var(name).is_ok_and(|value| matches!(value.to_ascii_lowercase().as_str(), "true" | "1"))
}

#[cfg(test)]
mod tests {
    use axum::http::{HeaderMap, HeaderValue};
    use opentelemetry::trace::{TraceContextExt, TracerProvider as _};
    use opentelemetry_sdk::trace::SdkTracerProvider;
    use tracing_opentelemetry::OpenTelemetrySpanExt as _;
    use tracing_subscriber::layer::{Context as LayerContext, SubscriberExt as _};

    use super::request_span;

    #[test]
    fn request_span_continues_incoming_w3c_trace_context() {
        let provider = SdkTracerProvider::builder().build();
        let tracer = provider.tracer("request-span-test");
        let subscriber =
            tracing_subscriber::registry().with(tracing_opentelemetry::layer().with_tracer(tracer));
        let mut headers = HeaderMap::new();
        headers.insert(
            "traceparent",
            HeaderValue::from_static("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"),
        );
        headers.insert(
            "tracestate",
            HeaderValue::from_static("vendor=opaque-value"),
        );

        tracing::subscriber::with_default(subscriber, || {
            let span = request_span(&headers);
            let context = span.context();
            let current = context.span();
            let span_context = current.span_context();
            assert_eq!(
                span_context.trace_id().to_string(),
                "4bf92f3577b34da6a3ce929d0e0e4736"
            );
            assert_eq!(span_context.trace_state().header(), "vendor=opaque-value");
        });
    }

    #[test]
    fn record_root_input_and_output_populates_gen_ai_prompt_and_completion() {
        use super::{record_root_input, record_root_output};
        use opentelemetry_sdk::trace::InMemorySpanExporter;
        use switchyard_protocol::{
            AggLlmResponse, ContentBlock, LlmRequest, LlmResponse, Message, Response,
            ResponseOutput, Role,
        };

        let exporter = InMemorySpanExporter::default();
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(exporter.clone())
            .build();
        let tracer = provider.tracer("record-root-test");
        let subscriber =
            tracing_subscriber::registry().with(tracing_opentelemetry::layer().with_tracer(tracer));

        tracing::subscriber::with_default(subscriber, || {
            let span = request_span(&HeaderMap::new());
            let request = LlmRequest {
                messages: vec![Message::text(Role::User, "hello world")],
                ..LlmRequest::default()
            };
            let response = Response {
                llm_response: LlmResponse::Agg(AggLlmResponse {
                    model: Some("test-model".to_string()),
                    outputs: vec![ResponseOutput {
                        role: Role::Assistant,
                        content: vec![ContentBlock::Text {
                            text: "the answer".into(),
                        }],
                        stop_reason: None,
                    }],
                    ..AggLlmResponse::default()
                }),
                metadata: None,
            };
            record_root_input(&span, &request);
            record_root_output(&span, &response);
        });

        let spans = exporter.get_finished_spans().expect("failed to get spans");
        let root_span = spans
            .iter()
            .find(|s| s.name == "switchyard.request")
            .expect("no switchyard.request span");
        assert!(
            root_span
                .attributes
                .iter()
                .any(|a| a.key.as_str() == "gen_ai.prompt"),
            "gen_ai.prompt should be recorded"
        );
        assert!(
            root_span
                .attributes
                .iter()
                .any(|a| a.key.as_str() == "gen_ai.completion"),
            "gen_ai.completion should be recorded for non-streamed Agg response"
        );
    }

    #[test]
    fn record_root_input_skips_empty_messages() {
        use super::record_root_input;
        use opentelemetry_sdk::trace::InMemorySpanExporter;
        use switchyard_protocol::LlmRequest;

        let exporter = InMemorySpanExporter::default();
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(exporter.clone())
            .build();
        let tracer = provider.tracer("record-root-input-test");
        let subscriber =
            tracing_subscriber::registry().with(tracing_opentelemetry::layer().with_tracer(tracer));

        tracing::subscriber::with_default(subscriber, || {
            let span = request_span(&HeaderMap::new());
            let request = LlmRequest::default();
            record_root_input(&span, &request);
        });

        let spans = exporter.get_finished_spans().expect("failed to get spans");
        let root_span = spans
            .iter()
            .find(|s| s.name == "switchyard.request")
            .expect("no switchyard.request span");
        assert!(
            !root_span
                .attributes
                .iter()
                .any(|a| a.key.as_str() == "gen_ai.prompt"),
            "gen_ai.prompt should not be recorded when messages are empty"
        );
    }

    #[test]
    fn record_root_output_skips_completion_for_agg_with_empty_outputs() {
        use super::record_root_output;
        use opentelemetry_sdk::trace::InMemorySpanExporter;
        use switchyard_protocol::{AggLlmResponse, LlmResponse, Response};

        let exporter = InMemorySpanExporter::default();
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(exporter.clone())
            .build();
        let tracer = provider.tracer("record-root-output-test");
        let subscriber =
            tracing_subscriber::registry().with(tracing_opentelemetry::layer().with_tracer(tracer));

        tracing::subscriber::with_default(subscriber, || {
            let span = request_span(&HeaderMap::new());
            let response = Response {
                llm_response: LlmResponse::Agg(AggLlmResponse {
                    model: Some("test-model".to_string()),
                    outputs: vec![],
                    ..AggLlmResponse::default()
                }),
                metadata: None,
            };
            record_root_output(&span, &response);
        });

        let spans = exporter.get_finished_spans().expect("failed to get spans");
        let root_span = spans
            .iter()
            .find(|s| s.name == "switchyard.request")
            .expect("no switchyard.request span");
        assert!(
            !root_span
                .attributes
                .iter()
                .any(|a| a.key.as_str() == "gen_ai.completion"),
            "gen_ai.completion should not be recorded for Agg response with empty outputs"
        );
    }

    use std::collections::BTreeMap;
    use std::fmt;
    use std::sync::Arc;

    use parking_lot::Mutex;
    use tracing::Subscriber;
    use tracing::field::{Field, Visit};
    use tracing::span::{Attributes, Id, Record};
    use tracing_subscriber::Layer;
    use tracing_subscriber::registry::LookupSpan;

    use super::{record_root_outcome, record_root_route_context};
    use switchyard_protocol::{GenericKeys, LangfuseKeys, ResponseOrigin};

    #[derive(Clone, Default)]
    struct SpanRecord {
        name: String,
        fields: BTreeMap<String, String>,
    }

    #[derive(Clone, Default)]
    struct CaptureStore {
        spans: Arc<Mutex<BTreeMap<u64, SpanRecord>>>,
    }

    impl CaptureStore {
        fn root(&self) -> SpanRecord {
            self.spans
                .lock()
                .values()
                .find(|span| span.name == "switchyard.request")
                .cloned()
                .expect("missing switchyard.request span")
        }
    }

    struct FieldVisitor<'a>(&'a mut BTreeMap<String, String>);

    impl Visit for FieldVisitor<'_> {
        fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
            self.0
                .insert(field.name().to_string(), format!("{value:?}"));
        }

        fn record_str(&mut self, field: &Field, value: &str) {
            self.0.insert(field.name().to_string(), value.to_string());
        }
    }

    struct CaptureLayer {
        store: CaptureStore,
    }

    impl<S> Layer<S> for CaptureLayer
    where
        S: Subscriber + for<'a> LookupSpan<'a>,
    {
        fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, _ctx: LayerContext<'_, S>) {
            let mut fields = BTreeMap::new();
            attrs.record(&mut FieldVisitor(&mut fields));
            self.store.spans.lock().insert(
                id.into_u64(),
                SpanRecord {
                    name: attrs.metadata().name().to_string(),
                    fields,
                },
            );
        }

        fn on_record(&self, id: &Id, values: &Record<'_>, _ctx: LayerContext<'_, S>) {
            if let Some(record) = self.store.spans.lock().get_mut(&id.into_u64()) {
                values.record(&mut FieldVisitor(&mut record.fields));
            }
        }
    }

    fn capture_root(record: impl FnOnce(&tracing::Span)) -> SpanRecord {
        let store = CaptureStore::default();
        let subscriber = tracing_subscriber::registry().with(CaptureLayer {
            store: store.clone(),
        });
        tracing::subscriber::with_default(subscriber, || {
            let span = request_span(&HeaderMap::new());
            record(&span);
        });
        store.root()
    }

    fn assert_pair(span: &SpanRecord, generic: &str, langfuse: &str, value: &str) {
        assert_eq!(
            span.fields.get(generic).map(String::as_str),
            Some(value),
            "{generic}"
        );
        assert_eq!(
            span.fields.get(langfuse).map(String::as_str),
            Some(value),
            "{langfuse}"
        );
    }

    fn assert_absent(span: &SpanRecord, key: &str) {
        assert_eq!(span.fields.get(key), None, "{key} should be absent");
    }

    fn assert_route_context(span: &SpanRecord, route: &str, algorithm: &str, session: &str) {
        assert_pair(span, GenericKeys::ROUTE_ID, LangfuseKeys::ROUTE_ID, route);
        assert_pair(
            span,
            GenericKeys::ALGORITHM,
            LangfuseKeys::ALGORITHM,
            algorithm,
        );
        assert_eq!(
            span.fields
                .get(LangfuseKeys::SESSION_ID)
                .map(String::as_str),
            Some(session)
        );
    }

    #[test]
    fn successful_request_summarizes_matching_selected_and_served_target() {
        let span = capture_root(|span| {
            record_root_route_context(span, Some("auto"), "random", Some("session-1"), Some("user-1"));
            record_root_outcome(span, "primary", Some("primary"));
        });
        assert_route_context(&span, "auto", "random", "session-1");
        assert_pair(
            &span,
            GenericKeys::SELECTED_TARGET,
            LangfuseKeys::SELECTED_TARGET,
            "primary",
        );
        assert_pair(
            &span,
            GenericKeys::SERVED_TARGET,
            LangfuseKeys::SERVED_TARGET,
            "primary",
        );
        assert_pair(
            &span,
            GenericKeys::RESPONSE_ORIGIN,
            LangfuseKeys::RESPONSE_ORIGIN,
            ResponseOrigin::Target.as_str(),
        );
        assert_absent(&span, GenericKeys::CALL_ROLE);
        assert_absent(&span, GenericKeys::CALL_TARGET);
    }

    #[test]
    fn fallback_success_keeps_selected_target_distinct_from_served() {
        let span = capture_root(|span| {
            record_root_route_context(span, Some("auto"), "llm_task_classifier", Some("session-2"), Some("user-2"));
            record_root_outcome(span, "weak", Some("strong"));
        });
        assert_route_context(&span, "auto", "llm_task_classifier", "session-2");
        assert_pair(
            &span,
            GenericKeys::SELECTED_TARGET,
            LangfuseKeys::SELECTED_TARGET,
            "weak",
        );
        assert_pair(
            &span,
            GenericKeys::SERVED_TARGET,
            LangfuseKeys::SERVED_TARGET,
            "strong",
        );
        assert_pair(
            &span,
            GenericKeys::RESPONSE_ORIGIN,
            LangfuseKeys::RESPONSE_ORIGIN,
            ResponseOrigin::Target.as_str(),
        );
    }

    #[test]
    fn routing_generated_response_does_not_claim_a_target_served_it() {
        let span = capture_root(|span| {
            record_root_route_context(span, Some("auto"), "noop", Some("session-3"), Some("user-3"));
            record_root_outcome(span, "auto", None);
        });
        assert_route_context(&span, "auto", "noop", "session-3");
        assert_pair(
            &span,
            GenericKeys::SELECTED_TARGET,
            LangfuseKeys::SELECTED_TARGET,
            "auto",
        );
        assert_pair(
            &span,
            GenericKeys::RESPONSE_ORIGIN,
            LangfuseKeys::RESPONSE_ORIGIN,
            ResponseOrigin::RoutingGenerated.as_str(),
        );
        assert_absent(&span, GenericKeys::SERVED_TARGET);
        assert_absent(&span, LangfuseKeys::SERVED_TARGET);
    }

    #[test]
    fn execute_failure_does_not_fabricate_a_terminal_outcome() {
        let span = capture_root(|span| {
            record_root_route_context(span, Some("auto"), "random", Some("session-4"), Some("user-4"));
        });
        assert_route_context(&span, "auto", "random", "session-4");
        assert_absent(&span, GenericKeys::SELECTED_TARGET);
        assert_absent(&span, LangfuseKeys::SELECTED_TARGET);
        assert_absent(&span, GenericKeys::SERVED_TARGET);
        assert_absent(&span, LangfuseKeys::SERVED_TARGET);
        assert_absent(&span, GenericKeys::RESPONSE_ORIGIN);
        assert_absent(&span, LangfuseKeys::RESPONSE_ORIGIN);
    }

    #[test]
    fn present_user_id_records_langfuse_user_id_on_root_span() {
        let span = capture_root(|span| {
            record_root_route_context(span, Some("auto"), "random", Some("session-1"), Some("kcasamento"));
        });
        assert_eq!(
            span.fields.get(LangfuseKeys::USER_ID).map(String::as_str),
            Some("kcasamento")
        );
    }

    #[test]
    fn absent_user_id_records_no_langfuse_user_id_on_root_span() {
        let span = capture_root(|span| {
            record_root_route_context(span, Some("auto"), "random", Some("session-1"), None);
        });
        assert_absent(&span, LangfuseKeys::USER_ID);
    }
}
