// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-wide tracing and OpenTelemetry setup for server hosts.

use std::env;
use std::sync::OnceLock;

use axum::http::HeaderMap;
use opentelemetry::propagation::{Extractor, TextMapPropagator};
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::SdkTracerProvider;
use opentelemetry_sdk::Resource;
use tracing_opentelemetry::OpenTelemetrySpanExt;
use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::util::SubscriberInitExt as _;
use tracing_subscriber::{EnvFilter, Layer as _};

use crate::{metrics, ServerError, ServerResult};

const DEFAULT_LOG_FILTER: &str = "switchyard_server=info,libsy=info,opentelemetry=warn";
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
    if let Some(Ok(observability)) = OBSERVABILITY.get() {
        if let Some(provider) = &observability.tracer_provider {
            if let Err(error) = provider.force_flush() {
                tracing::warn!(error = %error, "failed to flush OpenTelemetry traces");
            }
        }
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
    );
    let _ = span.set_parent(parent);
    span
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
    use tracing_subscriber::layer::SubscriberExt as _;

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
}
