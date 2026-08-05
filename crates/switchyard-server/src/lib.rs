// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust HTTP server for libsy algorithms.

pub mod config;
mod metrics;
mod observability;
mod response;
mod routing_log;
mod shutdown;
mod sse;
mod stats;
mod usage_metrics;

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::future::Future;
use std::io::IsTerminal;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::rejection::{JsonRejection, QueryRejection};
use axum::extract::{DefaultBodyLimit, Query, Request as HttpRequest, State};
use axum::http::header::CONTENT_TYPE;
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use axum_server::tls_rustls::RustlsConfig;
use libsy::{Algorithm, LibsyError, RunObservation, RunObserver};
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{Value, json};
use switchyard_llm_client::TranslatingLlmClient;
use switchyard_protocol::{Context, Decision, LlmClientError, Metadata, Request, Usage};
use tokio::net::{TcpListener, TcpSocket};
use tokio::task;
use tracing::{Instrument, Level};

use switchyard_translation::{WireFormat, decode_request};

use crate::response::into_http_response;
use crate::stats::{StatsAccumulator, StatsSnapshot, prefix_probe, tracking_enabled_from_env};

pub use observability::{flush_observability, initialize_observability};

/// Default TCP listen backlog used by the Rust server.
pub const DEFAULT_LISTEN_BACKLOG: u32 = 65_535;

/// Default time allowed for active requests to finish during shutdown.
pub const DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);

/// Maximum buffered JSON request size accepted by the LLM endpoints.
pub const DEFAULT_MAX_REQUEST_BODY_BYTES: usize = 32 * 1024 * 1024;

const HEADER_SELECTED_MODEL: &str = "x-model-router-selected-model";
const HEADER_RATIONALE: &str = "x-model-router-rationale";
const MAX_ROUTING_HEADER_VALUE_LEN: usize = 512;
const STARTUP_BANNER_ART: &str = include_str!("../assets/startup_banner.txt");

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

/// Capabilities that one route advertises on `GET /v1/models`.
///
/// An unset capability is undeclared and serializes as `null`.
#[derive(Clone, Copy, Default)]
struct ModelCapabilities {
    context_window: Option<u32>,
    tool_calling: Option<bool>,
}

/// A registered algorithm route and its server-owned endpoint metadata.
struct RouteEntry {
    algorithm: Arc<dyn Algorithm>,
    capabilities: ModelCapabilities,
    count_tokens_target: Option<CountTokensTarget>,
}

/// Exact upstream model used by the server's Anthropic token-count endpoint.
#[derive(Clone)]
struct CountTokensTarget {
    model: String,
    client: Arc<TranslatingLlmClient>,
}

impl CountTokensTarget {
    async fn count_tokens(&self, request: Request) -> Result<Value, LlmClientError> {
        self.client.count_tokens(&self.model, request).await
    }
}

/// Shared server state used by all endpoint handlers.
#[derive(Clone)]
pub struct ServerState {
    routes: Arc<BTreeMap<String, RouteEntry>>,
    metrics: prometheus::Registry,
    stats: StatsAccumulator,
    routing_log: Option<SharedRoutingLog>,
    track_cache_eligibility: bool,
}

#[derive(Clone)]
struct SharedRoutingLog {
    writer: Arc<Mutex<routing_log::RoutingLog>>,
    path: PathBuf,
}

impl SharedRoutingLog {
    fn new(path: PathBuf) -> ServerResult<Self> {
        Ok(Self {
            writer: Arc::new(Mutex::new(routing_log::RoutingLog::new(path.clone())?)),
            path,
        })
    }

    fn append(
        &self,
        context: routing_log::RoutingLogContext,
        model: &str,
        tier: Option<&str>,
        usage: &Usage,
    ) {
        if let Err(error) = self.writer.lock().append(context, model, tier, usage) {
            tracing::warn!(path = %self.path.display(), %error, "routing log append failed");
        }
    }

    fn snapshot_session(
        &self,
        session_id: &str,
    ) -> std::io::Result<Option<routing_log::SessionStatsSnapshot>> {
        routing_log::snapshot(&self.path, session_id)
    }
}

impl ServerState {
    /// Creates server state from route model IDs and their libsy algorithms.
    pub fn new(
        routes: impl IntoIterator<Item = (String, Arc<dyn Algorithm>)>,
    ) -> ServerResult<Self> {
        Self::new_with_capabilities(
            routes
                .into_iter()
                .map(|(model, algorithm)| (model, algorithm, ModelCapabilities::default(), None)),
        )
    }

    fn new_with_capabilities(
        routes: impl IntoIterator<
            Item = (
                String,
                Arc<dyn Algorithm>,
                ModelCapabilities,
                Option<CountTokensTarget>,
            ),
        >,
    ) -> ServerResult<Self> {
        let mut entries = BTreeMap::new();
        for (model, algorithm, capabilities, count_tokens_target) in routes {
            let model = model.trim();
            if model.is_empty() {
                return Err(ServerError::new("route model must not be empty"));
            }
            let entry = RouteEntry {
                algorithm,
                capabilities,
                count_tokens_target,
            };
            if entries.insert(model.to_string(), entry).is_some() {
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
            stats: StatsAccumulator::default(),
            routing_log: None,
            track_cache_eligibility: tracking_enabled_from_env(),
        })
    }

    /// Enables durable per-request routing records at `path`.
    pub fn with_routing_log(mut self, path: impl Into<PathBuf>) -> ServerResult<Self> {
        self.routing_log = Some(SharedRoutingLog::new(path.into())?);
        Ok(self)
    }

    /// Returns the route model IDs served by the configured algorithms.
    pub fn models(&self) -> impl Iterator<Item = &str> {
        self.routes.keys().map(String::as_str)
    }

    fn route_for_model(&self, model: &str) -> Option<&RouteEntry> {
        self.routes.get(model)
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
    /// Maximum time active requests may drain after shutdown begins.
    pub shutdown_timeout: Duration,
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

    let server = BoundServer::bind(state, options)?;
    println!("{}", server.startup_banner(std::io::stdout().is_terminal()));
    server.serve(shutdown::signal()).await
}

/// A configured server with its listening socket already bound.
pub struct BoundServer {
    listener: TcpListener,
    router: Router,
    options: ServerRunOptions,
    state: ServerState,
}

impl BoundServer {
    /// Binds the configured address and prepares the HTTP router.
    pub fn bind(state: ServerState, options: ServerRunOptions) -> ServerResult<Self> {
        let listener = bind_tcp_listener(options.addr, options.backlog)?;
        let addr = listener.local_addr().map_err(server_io_error)?;
        Ok(Self {
            listener,
            router: build_switchyard_router(state.clone()),
            options: ServerRunOptions { addr, ..options },
            state,
        })
    }

    /// Returns the actual bound address, including an OS-selected port.
    pub fn local_addr(&self) -> SocketAddr {
        self.options.addr
    }

    /// Serves requests until the supplied shutdown future resolves.
    pub async fn serve(
        self,
        shutdown: impl Future<Output = ()> + Send + 'static,
    ) -> ServerResult<()> {
        let shutdown_timeout = self.options.shutdown_timeout;
        if let Some(tls) = self.options.tls {
            serve_tls(self.listener, self.router, tls, shutdown_timeout, shutdown).await
        } else {
            serve(self.listener, self.router, shutdown_timeout, shutdown).await
        }
    }

    fn startup_banner(&self, color: bool) -> String {
        startup_banner(&self.options, &self.state, color)
    }
}

async fn serve_tls(
    listener: TcpListener,
    router: Router,
    tls: TlsOptions,
    shutdown_timeout: Duration,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> ServerResult<()> {
    if let Err(error) = rustls::crypto::aws_lc_rs::default_provider().install_default() {
        tracing::debug!(?error, "TLS crypto provider was already installed");
    }

    let config = RustlsConfig::from_pem_file(tls.cert, tls.key)
        .await
        .map_err(server_io_error)?;
    let std_listener = listener.into_std().map_err(server_io_error)?;
    let server = axum_server::from_tcp_rustls(std_listener, config).map_err(server_io_error)?;
    let handle = axum_server::Handle::new();
    let server = server
        .handle(handle.clone())
        .serve(router.into_make_service());
    serve_until_shutdown(server, handle, shutdown_timeout, shutdown).await
}

async fn serve(
    listener: TcpListener,
    router: Router,
    shutdown_timeout: Duration,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> ServerResult<()> {
    let std_listener = listener.into_std().map_err(server_io_error)?;
    let server = axum_server::from_tcp(std_listener).map_err(server_io_error)?;
    let handle = axum_server::Handle::new();
    let server = server
        .handle(handle.clone())
        .serve(router.into_make_service());
    serve_until_shutdown(server, handle, shutdown_timeout, shutdown).await
}

/// Runs the server until it exits or shutdown begins, then drains active requests.
async fn serve_until_shutdown(
    server: impl Future<Output = std::io::Result<()>>,
    handle: axum_server::Handle<SocketAddr>,
    timeout: Duration,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> ServerResult<()> {
    tokio::pin!(server);
    tokio::select! {
        result = &mut server => result.map_err(server_io_error),
        _ = shutdown => {
            tracing::info!(
                ?timeout,
                "shutdown signal received; draining active requests"
            );
            handle.graceful_shutdown(Some(timeout));
            server.await.map_err(server_io_error)
        }
    }
}

/// Ingress timestamp for one request, taken before any body is read.
#[derive(Clone, Copy)]
struct RequestStart(Instant);

/// Maps routed call observations to backend stats, non-routed calls to classifier/judge
/// stats, and records routing overhead once the algorithm run completes.
fn stats_observer(stats: StatsAccumulator) -> RunObserver {
    Arc::new(move |observation| match observation {
        RunObservation::LlmCall(call) => {
            let latency_ms = call.duration.as_secs_f64() * 1_000.0;
            if call.is_routed {
                if call.is_success {
                    stats.record_success(&call.selected_model, latency_ms, call.tier.as_deref());
                } else {
                    stats.record_error(&call.selected_model, call.tier.as_deref());
                }
            } else if call.is_success {
                stats.record_classifier_success(
                    call.selected_model,
                    call.usage.as_ref().map(usage_metrics::token_usage),
                    latency_ms,
                );
            } else {
                stats.record_classifier_error(call.selected_model);
            }
        }
        RunObservation::RoutingOverhead(duration) => {
            stats.record_routing_overhead(duration.as_secs_f64() * 1_000.0);
        }
    })
}

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
    let mut router = Router::new()
        .route("/v1/chat/completions", post(openai_chat_completions))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/responses", post(openai_responses))
        .route("/v1/messages/count_tokens", post(anthropic_count_tokens))
        .route("/v1/models", get(models))
        .route("/v1/stats", get(get_stats))
        .route("/v1/stats/reset", post(reset_stats))
        .route("/metrics", get(prometheus_metrics))
        .route("/health", get(health));
    if state.routing_log.is_some() {
        router = router.route("/v1/routing/session-stats", get(get_session_stats));
    }
    router
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

/// Anthropic token counting against the route's explicitly configured target.
async fn anthropic_count_tokens(
    State(state): State<ServerState>,
    headers: HeaderMap,
    body: std::result::Result<Json<Value>, JsonRejection>,
) -> Response {
    let body = match llm_json_body(body) {
        Ok(body) => body,
        Err(message) => return anthropic_error_response(invalid_body_error(message)),
    };
    let (route, request) = match resolve_route(
        &state,
        metadata_from_headers(headers),
        body,
        WireFormat::AnthropicMessages,
    ) {
        Ok(resolved) => resolved,
        Err(response) => return anthropic_error_response(response),
    };
    let Some(target) = route.count_tokens_target.as_ref() else {
        return anthropic_error_response(error_response(
            StatusCode::BAD_REQUEST,
            "route has no Anthropic target for token counting",
            "invalid_request_error",
            "count_tokens_unsupported",
        ));
    };
    anthropic_error_response(match target.count_tokens(request).await {
        Ok(payload) => (StatusCode::OK, Json(payload)).into_response(),
        Err(error) => count_tokens_error(error),
    })
}

/// Maps a token-count client failure with the same policy as a routed client call.
fn count_tokens_error(error: LlmClientError) -> Response {
    client_error(&error)
}

async fn handle_endpoint(
    state: ServerState,
    started: RequestStart,
    headers: HeaderMap,
    body: std::result::Result<Json<Value>, JsonRejection>,
    wire_format: WireFormat,
) -> Response {
    let span = observability::request_span(&headers);
    handle_endpoint_inner(state, started, headers, body, wire_format)
        .instrument(span)
        .await
}

async fn handle_endpoint_inner(
    state: ServerState,
    started: RequestStart,
    headers: HeaderMap,
    body: std::result::Result<Json<Value>, JsonRejection>,
    wire_format: WireFormat,
) -> Response {
    let routing_log_context = state
        .routing_log
        .as_ref()
        .map(|_| routing_log::RoutingLogContext::from_headers(&headers));
    let metadata = metadata_from_headers(headers);
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
        Ok(body) => {
            handle_llm_request(
                state,
                started,
                metadata,
                body,
                wire_format,
                routing_log_context,
            )
            .await
        }
        Err(message) => invalid_body_error(message),
    };
    let response = render_error_response(response, wire_format);
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
/// the resolved route and the built request — or an error [`Response`]
/// (invalid body, empty `model` → 400, unknown route → 404).
// Both callers immediately return the `Err(Response)` as the HTTP response, so
// the large error type is intentional, not propagated up a call stack.
#[allow(clippy::type_complexity, clippy::result_large_err)]
fn resolve_route(
    state: &ServerState,
    metadata: Metadata,
    body: Value,
    wire_format: WireFormat,
) -> std::result::Result<(&RouteEntry, Request), Response> {
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
    let route = state.route_for_model(&requested_model).ok_or_else(|| {
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
    Ok((route, request))
}

async fn handle_llm_request(
    state: ServerState,
    started: RequestStart,
    metadata: Metadata,
    body: Value,
    wire_format: WireFormat,
    routing_log_context: Option<routing_log::RoutingLogContext>,
) -> Response {
    let cache_probe = state.track_cache_eligibility.then(|| prefix_probe(&body));
    let (route, request) = match resolve_route(&state, metadata, body, wire_format) {
        Ok(resolved) => resolved,
        Err(response) => return response,
    };
    let algorithm = Arc::clone(&route.algorithm);
    let observer = stats_observer(state.stats.clone());
    let (trace, response) = match algorithm
        .run_observed(Context::default(), request, Some(observer))
        .await
    {
        Ok(result) => result,
        Err(error) => return algorithm_error(error),
    };

    // Metrics, response body, and routing header all read the same decision, so
    // the model they name can never disagree. An empty trace leaves the body with
    // the id the upstream reported.
    let decision = trace.last();
    let response = if let Some(decision) = decision {
        let cache_eligible = cache_probe
            .as_ref()
            .map(|probe| {
                state
                    .stats
                    .prefix_eligibility(decision.selected_model(), probe)
            })
            .unwrap_or(0.0);
        usage_metrics::observe(
            response,
            decision.selected_model(),
            decision.routing_tier(),
            started.0,
            state.stats,
            cache_eligible,
            state.routing_log.zip(routing_log_context),
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

fn metadata_from_headers(headers: HeaderMap) -> Metadata {
    let mut metadata = Metadata::from_headers(&headers);
    metadata.http_headers = Some(headers);
    metadata
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
    client_error(source)
}

fn client_error(error: &LlmClientError) -> Response {
    match error {
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
            upstream_error_message(body),
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

// Provider errors are often JSON documents; expose their message without
// embedding the entire document as an escaped string in our error envelope.
fn upstream_error_message(body: &str) -> String {
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|body| {
            body.pointer("/error/message")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_else(|| body.to_string())
}

// Error metadata retained until the client-facing endpoint selects an envelope.
#[derive(Clone)]
struct ApiError {
    status: StatusCode,
    message: String,
    error_type: &'static str,
    code: &'static str,
}

impl ApiError {
    fn new(
        status: StatusCode,
        message: impl Into<String>,
        error_type: &'static str,
        code: &'static str,
    ) -> Self {
        Self {
            status,
            message: message.into(),
            error_type,
            code,
        }
    }

    fn into_response(self, wire_format: WireFormat) -> Response {
        let body = match wire_format {
            WireFormat::AnthropicMessages => json!({
                "type": "error",
                "error": {
                    "type": anthropic_error_type(self.status),
                    "message": self.message.clone(),
                }
            }),
            WireFormat::OpenAiChat | WireFormat::OpenAiResponses => json!({
                "error": {
                    "message": self.message.clone(),
                    "type": self.error_type,
                    "code": self.code,
                }
            }),
        };
        let mut response = (self.status, Json(body)).into_response();
        response
            .extensions_mut()
            .insert(RequestLogError(self.message.clone()));
        response.extensions_mut().insert(self);
        response
    }
}

fn render_error_response(response: Response, wire_format: WireFormat) -> Response {
    let Some(error) = response.extensions().get::<ApiError>().cloned() else {
        return response;
    };
    error.into_response(wire_format)
}

fn anthropic_error_response(response: Response) -> Response {
    render_error_response(response, WireFormat::AnthropicMessages)
}

fn anthropic_error_type(status: StatusCode) -> &'static str {
    match status {
        StatusCode::BAD_REQUEST => "invalid_request_error",
        StatusCode::UNAUTHORIZED => "authentication_error",
        StatusCode::FORBIDDEN => "permission_error",
        StatusCode::NOT_FOUND => "not_found_error",
        StatusCode::PAYLOAD_TOO_LARGE => "request_too_large",
        StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
        status if status.as_u16() == 529 => "overloaded_error",
        _ => "api_error",
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
    ApiError::new(status, message, error_type, code).into_response(WireFormat::OpenAiChat)
}

async fn models(State(state): State<ServerState>) -> Json<Value> {
    Json(model_list_payload(
        state
            .routes
            .iter()
            .map(|(model, entry)| (model.as_str(), entry.capabilities)),
    ))
}

async fn get_stats(State(state): State<ServerState>) -> Json<StatsSnapshot> {
    Json(state.stats.snapshot())
}

async fn reset_stats(State(state): State<ServerState>) -> Json<Value> {
    state.stats.reset();
    Json(json!({"status": "reset"}))
}

#[derive(Deserialize)]
struct SessionStatsQuery {
    session_id: String,
}

// TODO: This loads the entire file. It should stream the JSONL instead.
// Huge files will crash the demo server.
async fn get_session_stats(
    State(state): State<ServerState>,
    query: std::result::Result<Query<SessionStatsQuery>, QueryRejection>,
) -> Response {
    let Query(query) = match query {
        Ok(query) => query,
        Err(error) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                error.to_string(),
                "invalid_request_error",
                "invalid_query",
            );
        }
    };
    let Some(routing_log) = state.routing_log else {
        // Should be unreachable
        return not_found().await;
    };
    let session_id = query.session_id.clone();
    // Loading and de-serializing a large file is a time consuming blocking operation
    let snapshot =
        match task::spawn_blocking(move || routing_log.snapshot_session(&session_id)).await {
            Ok(s) => s,
            Err(err) => {
                return server_error(format!("failed to snapshot: {err}"));
            }
        };

    match snapshot {
        Ok(Some(snapshot)) => (StatusCode::OK, Json(snapshot)).into_response(),
        Ok(None) => error_response(
            StatusCode::NOT_FOUND,
            "Routing session not found",
            "not_found",
            "routing_session_not_found",
        ),
        Err(error) => server_error(format!("failed to read routing log: {error}")),
    }
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

fn model_list_payload<'a>(
    entries: impl IntoIterator<Item = (&'a str, ModelCapabilities)>,
) -> Value {
    let entries = entries.into_iter().collect::<Vec<_>>();
    let model_ids = entries.iter().map(|(model, _)| *model).collect::<Vec<_>>();
    let first_id = model_ids.first().copied();
    let last_id = model_ids.last().copied();
    json!({
        "object": "list",
        "data": entries.iter().map(|(model, caps)| model_entry_json(model, *caps)).collect::<Vec<_>>(),
        "first_id": first_id,
        "last_id": last_id,
        "has_more": false,
        "default_model": first_id,
        "model_pool": model_ids,
    })
}

fn model_entry_json(model: &str, capabilities: ModelCapabilities) -> Value {
    json!({
        "id": model,
        "object": "model",
        "type": "model",
        "created": 0,
        "owned_by": "switchyard",
        "display_name": model,
        "capabilities": {
            "streaming": true,
            "tool_calling": capabilities.tool_calling,
            "context_window": capabilities.context_window,
            "supported_inbound_formats": [
                "openai-chat-completions",
                "openai-responses",
                "anthropic-messages",
            ],
        },
    })
}

fn startup_banner(options: &ServerRunOptions, state: &ServerState, color: bool) -> String {
    let scheme = if options.is_tls() { "https" } else { "http" };
    let listen_url = url_for_addr(scheme, options.addr);
    let request_url = request_url_for_addr(scheme, options.addr);
    let routes = state.models().collect::<Vec<_>>();
    let route_list = routes.join(", ");
    let example_model = routes.first().copied().unwrap_or("switchyard/route");
    let example_body = json!({
        "model": example_model,
        "messages": [{"role": "user", "content": "Hello from Switchyard"}],
    });
    let example_url = shell_quote(&format!("{request_url}/v1/chat/completions"));
    let example_body = shell_quote(&example_body.to_string());
    format!(
        "{}\nSwitchyard libsy server\n  listening: {}\n  routes: {}\n\nendpoints:\n{}\n\nexample:\n  curl -s {} \\\n    -H 'Content-Type: application/json' \\\n    -d {}",
        render_startup_banner_art(color),
        listen_url,
        route_list,
        endpoint_listing(state.routing_log.is_some()),
        example_url,
        example_body,
    )
}

// Keep redirected logs plain. Terminal output applies NVIDIA-green ANSI truecolor per line.
fn render_startup_banner_art(color: bool) -> String {
    let banner = STARTUP_BANNER_ART.trim_end();
    if !color {
        return banner.to_string();
    }

    let (red, green, blue) = (118, 185, 0);
    let mut rendered = String::new();
    for line in banner.lines() {
        rendered.push_str(&format!("\x1b[38;2;{red};{green};{blue}m{line}\x1b[0m\n"));
    }
    rendered.trim_end_matches('\n').to_string()
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

// Use loopback in request examples when the listener binds all interfaces.
fn request_url_for_addr(scheme: &'static str, addr: SocketAddr) -> String {
    let host = if addr.ip().is_unspecified() {
        match addr.ip() {
            std::net::IpAddr::V4(_) => "127.0.0.1".to_string(),
            std::net::IpAddr::V6(_) => "[::1]".to_string(),
        }
    } else {
        host_for_url(addr.ip())
    };
    format!("{scheme}://{host}:{}", addr.port())
}

// Bracket IPv6 literals so they are valid inside a URL authority.
fn host_for_url(ip: std::net::IpAddr) -> String {
    match ip {
        std::net::IpAddr::V4(ip) => ip.to_string(),
        std::net::IpAddr::V6(ip) => format!("[{ip}]"),
    }
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn endpoint_listing(has_routing_log: bool) -> String {
    let mut endpoints = vec![
        "  POST /v1/chat/completions    OpenAI Chat Completions",
        "  POST /v1/messages            Anthropic Messages",
        "  POST /v1/responses           OpenAI Responses",
        "  POST /v1/messages/count_tokens",
        "  GET  /v1/models              configured routes",
        "  GET  /v1/stats               routing stats",
        "  POST /v1/stats/reset",
        "  GET  /metrics                Prometheus metrics",
        "  GET  /health",
    ];
    if has_routing_log {
        endpoints.push("  GET  /v1/routing/session-stats");
    }
    endpoints.join("\n")
}

#[cfg(test)]
mod tests {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::sync::{Notify, oneshot};

    use super::*;

    #[derive(Clone)]
    struct ShutdownTestState {
        started: Arc<Notify>,
        release: Arc<Notify>,
    }

    struct ShutdownTestServer {
        state: ShutdownTestState,
        shutdown: oneshot::Sender<()>,
        server: task::JoinHandle<ServerResult<()>>,
        request: task::JoinHandle<std::io::Result<Vec<u8>>>,
    }

    async fn blocked_request(State(state): State<ShutdownTestState>) -> &'static str {
        state.started.notify_one();
        state.release.notified().await;
        "done"
    }

    async fn raw_request(addr: SocketAddr) -> std::io::Result<Vec<u8>> {
        let mut stream = tokio::net::TcpStream::connect(addr).await?;
        stream
            .write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            .await?;
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await?;
        Ok(response)
    }

    fn shutdown_test_server(shutdown_timeout: Duration) -> ShutdownTestServer {
        let state = ShutdownTestState {
            started: Arc::new(Notify::new()),
            release: Arc::new(Notify::new()),
        };
        let router = Router::new()
            .route("/", get(blocked_request))
            .with_state(state.clone());
        let listener = bind_tcp_listener("127.0.0.1:0".parse().expect("valid address"), 16)
            .expect("listener binds");
        let addr = listener.local_addr().expect("listener has an address");
        let (shutdown, shutdown_receiver) = oneshot::channel();
        let server = tokio::spawn(serve(listener, router, shutdown_timeout, async move {
            let _ = shutdown_receiver.await;
        }));
        let request = tokio::spawn(raw_request(addr));
        ShutdownTestServer {
            state,
            shutdown,
            server,
            request,
        }
    }

    // Active requests may finish within the grace period, while stuck requests are bounded.
    #[tokio::test]
    async fn shutdown_drains_until_configured_deadline() {
        let ShutdownTestServer {
            state,
            shutdown,
            mut server,
            request,
        } = shutdown_test_server(Duration::from_secs(1));
        state.started.notified().await;
        shutdown.send(()).expect("server receives shutdown");
        assert!(
            tokio::time::timeout(Duration::from_millis(25), &mut server)
                .await
                .is_err(),
            "server must wait for the active request"
        );
        state.release.notify_one();
        tokio::time::timeout(Duration::from_secs(1), server)
            .await
            .expect("server stops after request drains")
            .expect("server task completes")
            .expect("server exits cleanly");
        let response = request
            .await
            .expect("request task completes")
            .expect("request succeeds");
        assert!(response.windows(8).any(|part| part == b"200 OK\r\n"));
        assert!(response.ends_with(b"done"));

        let ShutdownTestServer {
            state,
            shutdown,
            server,
            request,
        } = shutdown_test_server(Duration::from_millis(25));
        state.started.notified().await;
        shutdown.send(()).expect("server receives shutdown");
        tokio::time::timeout(Duration::from_secs(1), server)
            .await
            .expect("shutdown deadline is enforced")
            .expect("server task completes")
            .expect("server exits cleanly");
        state.release.notify_one();
        request.abort();
    }

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
