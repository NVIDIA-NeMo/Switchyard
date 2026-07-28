// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust HTTP server for libsy algorithms.

pub mod config;
mod metrics;
mod response;
mod sse;
mod usage_metrics;

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::{rejection::JsonRejection, DefaultBodyLimit, Request as HttpRequest, State};
use axum::http::header::CONTENT_TYPE;
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use axum_server::tls_rustls::RustlsConfig;
use libsy::{Algorithm, Context, Decision, LibsyError, LlmClientError, Metadata, Request};
use serde_json::{json, Value};
use tokio::net::{TcpListener, TcpSocket};
use tracing::Level;

use switchyard_translation::{decode_request, WireFormat};

use crate::response::into_http_response;

/// Default TCP listen backlog used by the Rust server.
pub const DEFAULT_LISTEN_BACKLOG: u32 = 65_535;

/// Maximum buffered JSON request size accepted by the LLM endpoints.
pub const DEFAULT_MAX_REQUEST_BODY_BYTES: usize = 32 * 1024 * 1024;

const HEADER_SELECTED_MODEL: &str = "x-model-router-selected-model";
const HEADER_RATIONALE: &str = "x-model-router-rationale";
const MAX_ROUTING_HEADER_VALUE_LEN: usize = 512;

/// Error returned while configuring or running the server.
#[derive(Debug)]
pub struct ServerError {
    message: String,
}

impl ServerError {
    /// Creates a server error with a user-facing message.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for ServerError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for ServerError {}

/// Result returned by server setup and lifecycle operations.
pub type ServerResult<T> = std::result::Result<T, ServerError>;

/// Shared server state used by all endpoint handlers.
#[derive(Clone)]
pub struct ServerState {
    routes: Arc<BTreeMap<String, Arc<dyn Algorithm>>>,
    metrics: prometheus::Registry,
}

impl ServerState {
    /// Creates server state from route model IDs and their libsy algorithms.
    pub fn new(
        routes: impl IntoIterator<Item = (String, Arc<dyn Algorithm>)>,
    ) -> ServerResult<Self> {
        let mut entries = BTreeMap::new();
        for (model, algorithm) in routes {
            let model = model.trim();
            if model.is_empty() {
                return Err(ServerError::new("route model must not be empty"));
            }
            if entries.insert(model.to_string(), algorithm).is_some() {
                return Err(ServerError::new(format!("duplicate route model {model}")));
            }
        }
        if entries.is_empty() {
            return Err(ServerError::new("at least one algorithm route is required"));
        }
        let metrics = metrics::registry().map_err(ServerError::new)?;
        Ok(Self {
            routes: Arc::new(entries),
            metrics,
        })
    }

    /// Returns the route model IDs served by the configured algorithms.
    pub fn models(&self) -> impl Iterator<Item = &str> {
        self.routes.keys().map(String::as_str)
    }

    fn algorithm_for_model(&self, model: &str) -> Option<Arc<dyn Algorithm>> {
        self.routes.get(model).map(Arc::clone)
    }
}

/// Runtime options shared by server entry points.
#[derive(Clone, Debug)]
pub struct ServerRunOptions {
    /// Socket address to bind.
    pub addr: SocketAddr,
    /// TCP listen backlog.
    pub backlog: u32,
    /// Validate runtime construction without binding a socket.
    pub dry_run: bool,
    /// TLS certificate configuration, when HTTPS is enabled.
    pub tls: Option<TlsOptions>,
}

/// TLS certificate paths used by the server.
#[derive(Clone, Debug)]
pub struct TlsOptions {
    /// TLS certificate path in PEM format.
    pub cert: PathBuf,
    /// TLS private-key path in PEM format.
    pub key: PathBuf,
}

impl ServerRunOptions {
    fn is_tls(&self) -> bool {
        self.tls.is_some()
    }
}

/// Validates the runtime and starts the HTTP server unless `dry_run` is set.
pub async fn run_server(state: ServerState, options: ServerRunOptions) -> ServerResult<()> {
    if options.dry_run {
        println!("{}", dry_run_summary(&state));
        return Ok(());
    }

    let listener = bind_tcp_listener(options.addr, options.backlog)?;
    let bound_addr = listener.local_addr().map_err(server_io_error)?;
    let server_options = ServerRunOptions {
        addr: bound_addr,
        ..options
    };
    eprintln!("{}", startup_banner(&server_options, &state));
    let router = build_switchyard_router(state);
    if let Some(tls) = server_options.tls {
        serve_tls(listener, router, tls).await
    } else {
        serve(listener, router).await
    }
}

async fn serve_tls(listener: TcpListener, router: Router, tls: TlsOptions) -> ServerResult<()> {
    if let Err(error) = rustls::crypto::aws_lc_rs::default_provider().install_default() {
        tracing::debug!(?error, "TLS crypto provider was already installed");
    }

    let config = RustlsConfig::from_pem_file(tls.cert, tls.key)
        .await
        .map_err(server_io_error)?;
    let handle = axum_server::Handle::new();

    let shutdown_handle = handle.clone();
    tokio::spawn(async move {
        shutdown_signal().await;
        shutdown_handle.graceful_shutdown(Some(Duration::from_secs(2)));
    });

    let std_listener = listener.into_std().map_err(server_io_error)?;
    axum_server::from_tcp_rustls(std_listener, config)
        .map_err(server_io_error)?
        .handle(handle)
        .serve(router.into_make_service())
        .await
        .map_err(server_io_error)
}

async fn serve(listener: TcpListener, router: Router) -> ServerResult<()> {
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(server_io_error)
}

/// Ingress timestamp for one request, taken before any body is read.
#[derive(Clone, Copy)]
struct RequestStart(Instant);

/// Stamps the ingress instant into request extensions. Runs as a router layer,
/// so it executes before the handlers' `Json` extractor buffers the body —
/// request-latency measurements therefore include body read and decode.
async fn stamp_request_start(mut request: HttpRequest, next: Next) -> Response {
    request
        .extensions_mut()
        .insert(RequestStart(Instant::now()));
    next.run(request).await
}

/// Builds an Axum router for the supported LLM wire formats.
pub fn build_switchyard_router(state: ServerState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(openai_chat_completions))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/responses", post(openai_responses))
        .route("/v1/messages/count_tokens", post(anthropic_count_tokens))
        .route("/v1/models", get(models))
        .route("/metrics", get(prometheus_metrics))
        .route("/health", get(health))
        .fallback(not_found)
        .layer(DefaultBodyLimit::max(DEFAULT_MAX_REQUEST_BODY_BYTES))
        // `layer` only wraps routes registered before it, so this stays last.
        .layer(axum::middleware::from_fn(stamp_request_start))
        .with_state(state)
}

fn bind_tcp_listener(addr: SocketAddr, backlog: u32) -> ServerResult<TcpListener> {
    let socket = if addr.is_ipv4() {
        TcpSocket::new_v4()
    } else {
        TcpSocket::new_v6()
    }
    .map_err(server_io_error)?;

    socket.set_reuseaddr(true).map_err(server_io_error)?;
    socket.bind(addr).map_err(server_io_error)?;
    socket.listen(backlog).map_err(server_io_error)
}

fn server_io_error(error: std::io::Error) -> ServerError {
    ServerError::new(error.to_string())
}

async fn shutdown_signal() {
    if let Err(error) = tokio::signal::ctrl_c().await {
        tracing::warn!(
            error = %error,
            "ctrl-c shutdown signal unavailable; continuing without shutdown trigger"
        );
        std::future::pending::<()>().await;
    }
}

async fn openai_chat_completions(
    State(state): State<ServerState>,
    Extension(started): Extension<RequestStart>,
    headers: HeaderMap,
    body: std::result::Result<Json<Value>, JsonRejection>,
) -> Response {
    handle_endpoint(state, started, headers, body, WireFormat::OpenAiChat).await
}

async fn anthropic_messages(
    State(state): State<ServerState>,
    Extension(started): Extension<RequestStart>,
    headers: HeaderMap,
    body: std::result::Result<Json<Value>, JsonRejection>,
) -> Response {
    handle_endpoint(state, started, headers, body, WireFormat::AnthropicMessages).await
}

async fn openai_responses(
    State(state): State<ServerState>,
    Extension(started): Extension<RequestStart>,
    headers: HeaderMap,
    body: std::result::Result<Json<Value>, JsonRejection>,
) -> Response {
    handle_endpoint(state, started, headers, body, WireFormat::OpenAiResponses).await
}

/// Anthropic token counting. Resolves the route named by `model`, then does a
/// **direct passthrough** via [`Algorithm::count_tokens`] to that route's
/// Anthropic target — it does *not* run the routing cascade (count_tokens is a
/// pre-flight estimate with no routing decision). Unknown route → 404; a route
/// with no Anthropic target → 400.
async fn anthropic_count_tokens(
    State(state): State<ServerState>,
    headers: HeaderMap,
    body: std::result::Result<Json<Value>, JsonRejection>,
) -> Response {
    let body = match llm_json_body(body) {
        Ok(body) => body,
        Err(message) => return invalid_body_error(message),
    };
    let (algorithm, request) = match resolve_route(
        &state,
        metadata_from_headers(&headers),
        body,
        WireFormat::AnthropicMessages,
    ) {
        Ok(resolved) => resolved,
        Err(response) => return response,
    };
    match algorithm.count_tokens(request).await {
        Ok(payload) => (StatusCode::OK, Json(payload)).into_response(),
        Err(error) => count_tokens_error(error),
    }
}

/// Map a [`count_tokens`](Algorithm::count_tokens) failure to an HTTP response:
/// the route has no Anthropic target → 400, an upstream HTTP error → its own
/// status, anything else → 502.
fn count_tokens_error(error: LibsyError) -> Response {
    // The one count_tokens-specific case is "no Anthropic target in the route";
    // every upstream/client failure gets the same mapping completions use.
    match &error {
        LibsyError::AlgorithmError { message } => error_response(
            StatusCode::BAD_REQUEST,
            message.clone(),
            "invalid_request_error",
            "count_tokens_unsupported",
        ),
        _ => algorithm_error(error),
    }
}

async fn handle_endpoint(
    state: ServerState,
    started: RequestStart,
    headers: HeaderMap,
    body: std::result::Result<Json<Value>, JsonRejection>,
    wire_format: WireFormat,
) -> Response {
    let metadata = metadata_from_headers(&headers);
    let request_log = RequestLogContext {
        started: started.0,
        wire_format,
        requested_model: body
            .as_ref()
            .ok()
            .and_then(|body| body.0.get("model"))
            .and_then(Value::as_str)
            .map(str::to_string),
        streaming: body
            .as_ref()
            .ok()
            .and_then(|body| body.0.get("stream"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
        session_id: metadata.session_id.clone(),
        correlation_id: metadata.correlation_id.clone(),
    };

    let response = match llm_json_body(body) {
        Ok(body) => handle_llm_request(state, started, metadata, body, wire_format).await,
        Err(message) => invalid_body_error(message),
    };
    metrics::record_client_response(response.status().as_u16());
    request_log.emit(&response);
    response
}

fn llm_json_body(
    body: std::result::Result<Json<Value>, JsonRejection>,
) -> std::result::Result<Value, String> {
    match body {
        Ok(Json(value)) if value.is_object() => Ok(value),
        Ok(_) => Err("Request body must be a JSON object".to_string()),
        Err(error) => Err(format!("Request body must be valid JSON: {error}")),
    }
}

/// Decode `body`, resolve the route named by its `model`, and build the
/// [`Request`]. Shared by the completion and `count_tokens` handlers. Returns
/// the resolved algorithm and the built request — or an error [`Response`]
/// (invalid body, empty `model` → 400, unknown route → 404).
// Both callers immediately return the `Err(Response)` as the HTTP response, so
// the large error type is intentional, not propagated up a call stack.
#[allow(clippy::type_complexity, clippy::result_large_err)]
fn resolve_route(
    state: &ServerState,
    metadata: Metadata,
    body: Value,
    wire_format: WireFormat,
) -> std::result::Result<(Arc<dyn Algorithm>, Request), Response> {
    let llm_request = decode_request(wire_format, &body)
        .map_err(|error| invalid_body_error(error.to_string()))?;
    let requested_model = llm_request
        .model
        .clone()
        .filter(|model| !model.trim().is_empty())
        .ok_or_else(|| {
            error_response(
                StatusCode::BAD_REQUEST,
                "request body must include a non-empty string `model`",
                "invalid_request_error",
                "invalid_request_error",
            )
        })?;
    let algorithm = state.algorithm_for_model(&requested_model).ok_or_else(|| {
        error_response(
            StatusCode::NOT_FOUND,
            format!("No route registered for model {requested_model}"),
            "model_not_found",
            "model_not_found",
        )
    })?;
    let request = Request {
        llm_request,
        raw_request: Some(body),
        metadata: Some(metadata),
    };
    Ok((algorithm, request))
}

async fn handle_llm_request(
    state: ServerState,
    started: RequestStart,
    metadata: Metadata,
    body: Value,
    wire_format: WireFormat,
) -> Response {
    let (algorithm, request) = match resolve_route(&state, metadata, body, wire_format) {
        Ok(resolved) => resolved,
        Err(response) => return response,
    };
    let (trace, response) = match algorithm.run(Context::default(), request).await {
        Ok(result) => result,
        Err(error) => return algorithm_error(error),
    };
    // Metrics, response body, and routing header all read the same decision, so
    // the model they name can never disagree. An empty trace leaves the body with
    // the id the upstream reported.
    let decision = trace.last();
    let response = if let Some(decision) = decision {
        usage_metrics::observe(
            response,
            decision.selected_model(),
            decision.routing_tier(),
            started.0,
        )
    } else {
        response
    };

    let served_model = decision.map(|decision| decision.selected_model().to_string());
    let mut response = match into_http_response(response, wire_format, served_model) {
        Ok(response) => response,
        Err(error) => return server_error(error.to_string()),
    };
    if let Some(decision) = decision {
        attach_routing_headers(&mut response, decision.as_ref());
    }
    response
}

// Request metadata held until the terminal response determines the event level.
struct RequestLogContext {
    started: Instant,
    wire_format: WireFormat,
    requested_model: Option<String>,
    streaming: bool,
    session_id: Option<String>,
    correlation_id: Option<String>,
}

// Error text carried separately so terminal logging never consumes an HTTP body.
#[derive(Clone)]
struct RequestLogError(String);

impl RequestLogContext {
    fn emit(self, response: &Response) {
        let selected_model = response
            .headers()
            .get(HEADER_SELECTED_MODEL)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("");
        let duration_ms = self.started.elapsed().as_secs_f64() * 1_000.0;
        let error = response
            .extensions()
            .get::<RequestLogError>()
            .map(|error| error.0.as_str())
            .unwrap_or("");

        macro_rules! emit {
            ($level:expr, $message:literal) => {
                tracing::event!(
                    target: "switchyard_server::request",
                    $level,
                    wire_format = %self.wire_format,
                    status = response.status().as_u16(),
                    requested_model = self.requested_model.as_deref().unwrap_or(""),
                    selected_model,
                    streaming = self.streaming,
                    session_id = self.session_id.as_deref().unwrap_or(""),
                    correlation_id = self.correlation_id.as_deref().unwrap_or(""),
                    handling_duration_ms = duration_ms,
                    error,
                    $message
                )
            };
        }

        match request_log_level(response.status()) {
            Level::ERROR => emit!(Level::ERROR, "LLM request failed"),
            Level::WARN => emit!(Level::WARN, "LLM request failed"),
            _ => emit!(Level::INFO, "LLM request handled"),
        }
    }
}

fn request_log_level(status: StatusCode) -> Level {
    if status.is_server_error() {
        Level::ERROR
    } else if status.is_success() {
        Level::INFO
    } else {
        Level::WARN
    }
}

fn metadata_from_headers(headers: &HeaderMap) -> Metadata {
    let headers = normalized_headers(headers);
    let mut metadata = Metadata::from_headers(&headers);
    metadata.http_headers = Some(headers);
    metadata
}

fn normalized_headers(headers: &HeaderMap) -> BTreeMap<String, String> {
    headers
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.as_str().to_ascii_lowercase(), value.to_string()))
        })
        .collect()
}

fn attach_routing_headers(response: &mut Response, decision: &dyn Decision) {
    insert_routing_header(response, HEADER_SELECTED_MODEL, decision.selected_model());
    if let Some(reasoning) = decision.reasoning() {
        insert_routing_header(response, HEADER_RATIONALE, reasoning);
    }
}

fn insert_routing_header(response: &mut Response, name: &'static str, value: &str) {
    let Some(value) = sanitize_routing_header_value(value) else {
        return;
    };
    let Ok(value) = HeaderValue::from_str(&value) else {
        return;
    };
    response
        .headers_mut()
        .insert(HeaderName::from_static(name), value);
}

fn sanitize_routing_header_value(value: &str) -> Option<String> {
    let value = value.split_whitespace().collect::<Vec<_>>().join(" ");
    (!value.is_empty()).then(|| value.chars().take(MAX_ROUTING_HEADER_VALUE_LEN).collect())
}

fn algorithm_error(error: LibsyError) -> Response {
    let LibsyError::ClientCall { source, .. } = &error else {
        return server_error(error.to_string());
    };
    match source {
        LlmClientError::InvalidRequest { message }
        | LlmClientError::RequestTranslation(message) => error_response(
            StatusCode::BAD_REQUEST,
            message,
            "invalid_request_error",
            "invalid_request_error",
        ),
        LlmClientError::Configuration { message } => error_response(
            StatusCode::BAD_GATEWAY,
            message,
            "upstream_error",
            "upstream_configuration_error",
        ),
        LlmClientError::ContextWindowExceeded { message, .. } => error_response(
            StatusCode::BAD_REQUEST,
            message,
            "invalid_request_error",
            "context_length_exceeded",
        ),
        LlmClientError::UpstreamHttp { status, body } => error_response(
            StatusCode::from_u16(*status).unwrap_or(StatusCode::BAD_GATEWAY),
            body,
            "upstream_error",
            "upstream_error",
        ),
        LlmClientError::Transport { source } | LlmClientError::InvalidResponse { source } => {
            error_response(
                StatusCode::BAD_GATEWAY,
                source.to_string(),
                "upstream_error",
                "upstream_error",
            )
        }
        LlmClientError::ResponseTranslation(message) => error_response(
            StatusCode::BAD_GATEWAY,
            message,
            "upstream_error",
            "upstream_error",
        ),
        LlmClientError::Timeout { source } => error_response(
            StatusCode::GATEWAY_TIMEOUT,
            source.to_string(),
            "upstream_error",
            "upstream_timeout",
        ),
        LlmClientError::RequestEncoding(message) => server_error(message),
        _ => server_error(error.to_string()),
    }
}

fn server_error(message: impl Into<String>) -> Response {
    error_response(
        StatusCode::INTERNAL_SERVER_ERROR,
        message,
        "server_error",
        "server_error",
    )
}

fn invalid_body_error(message: impl Into<String>) -> Response {
    error_response(
        StatusCode::BAD_REQUEST,
        message,
        "invalid_request_error",
        "invalid_body",
    )
}

fn error_response(
    status: StatusCode,
    message: impl Into<String>,
    error_type: &'static str,
    code: &'static str,
) -> Response {
    let message = message.into();
    let mut response = (
        status,
        Json(json!({
            "error": {
                "message": message.clone(),
                "type": error_type,
                "code": code,
            }
        })),
    )
        .into_response();
    response.extensions_mut().insert(RequestLogError(message));
    response
}

async fn models(State(state): State<ServerState>) -> Json<Value> {
    Json(model_list_payload(state.models()))
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

async fn prometheus_metrics(State(state): State<ServerState>) -> Response {
    match metrics::encode(&state.metrics) {
        Ok(body) => ([(CONTENT_TYPE, metrics::CONTENT_TYPE)], body).into_response(),
        Err(error) => server_error(error),
    }
}

async fn not_found() -> Response {
    error_response(
        StatusCode::NOT_FOUND,
        "Not Found",
        "not_found",
        "endpoint_not_found",
    )
}

fn model_list_payload<'a>(models: impl IntoIterator<Item = &'a str>) -> Value {
    let model_ids = models.into_iter().map(str::to_string).collect::<Vec<_>>();
    let first_id = model_ids.first().cloned();
    let last_id = model_ids.last().cloned();
    json!({
        "object": "list",
        "data": model_ids.iter().map(|model| model_entry_json(model)).collect::<Vec<_>>(),
        "first_id": first_id,
        "last_id": last_id,
        "has_more": false,
        "default_model": first_id,
        "model_pool": model_ids,
    })
}

fn model_entry_json(model: &str) -> Value {
    json!({
        "id": model,
        "object": "model",
        "type": "model",
        "created": 0,
        "owned_by": "switchyard",
        "display_name": model,
        "capabilities": {
            "streaming": true,
            "tool_calling": null,
            "context_window": null,
            "supported_inbound_formats": [
                "openai-chat-completions",
                "openai-responses",
                "anthropic-messages",
            ],
        },
    })
}

fn startup_banner(options: &ServerRunOptions, state: &ServerState) -> String {
    let scheme = if options.is_tls() { "https" } else { "http" };
    format!(
        "Switchyard libsy server\n  listening: {}\n  routes: {}",
        url_for_addr(scheme, options.addr),
        state.models().collect::<Vec<_>>().join(", ")
    )
}

fn dry_run_summary(state: &ServerState) -> String {
    format!(
        "server OK: {}",
        state.models().collect::<Vec<_>>().join(", ")
    )
}

fn url_for_addr(scheme: &'static str, addr: SocketAddr) -> String {
    format!("{scheme}://{}:{}", host_for_url(addr.ip()), addr.port())
}

fn host_for_url(ip: std::net::IpAddr) -> String {
    match ip {
        std::net::IpAddr::V4(ip) => ip.to_string(),
        std::net::IpAddr::V6(ip) => format!("[{ip}]"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Terminal request severity follows HTTP status instead of error-path bookkeeping.
    #[test]
    fn request_log_level_follows_http_status() {
        assert_eq!(request_log_level(StatusCode::OK), Level::INFO);
        assert_eq!(request_log_level(StatusCode::BAD_REQUEST), Level::WARN);
        assert_eq!(
            request_log_level(StatusCode::INTERNAL_SERVER_ERROR),
            Level::ERROR
        );
    }

    // Canonical error text remains available without consuming the response body.
    #[test]
    fn error_response_carries_request_log_error() {
        let response = error_response(
            StatusCode::BAD_REQUEST,
            "invalid request",
            "invalid_request_error",
            "invalid_request_error",
        );

        assert_eq!(
            response
                .extensions()
                .get::<RequestLogError>()
                .map(|error| error.0.as_str()),
            Some("invalid request")
        );
    }
}
