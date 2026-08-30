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

use async_trait::async_trait;
use axum::extract::rejection::{JsonRejection, QueryRejection};
use axum::extract::{DefaultBodyLimit, FromRequest, Query, Request as HttpRequest, State};
use axum::http::header::CONTENT_TYPE;
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use axum_server::tls_rustls::RustlsConfig;
use libsy::{Algorithm, LibsyError, RoutingOutcome};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use switchyard_llm_client::{ClientRouter, RunObservation, RunObserver};
use switchyard_protocol::{LlmClientError, Metadata, ModelId, Request, Usage};
use switchyard_runner::{
    CallerAuthKind, DecisionTarget, ModelCapabilities, Route, RunOutput, Runner, RunnerError,
};
use tokio::net::{TcpListener, TcpSocket};
use tokio::task;
use tracing::{Instrument, Level};

use switchyard_translation::{WireFormat, decode_request, encode_aggregated_response};

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
const MAX_ROUTING_HEADER_VALUE_LEN: usize = 512;
/// Non-standard status used only in logs and metrics for a request whose
/// downstream client disconnected before any response was written.
const CLIENT_CLOSED_REQUEST: u16 = 499;
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

impl From<RunnerError> for ServerError {
    fn from(error: RunnerError) -> Self {
        let mut message = error.to_string();
        let mut source = error.source();
        while let Some(error) = source {
            message.push_str(": ");
            message.push_str(&error.to_string());
            source = error.source();
        }
        Self::new(message)
    }
}

/// Result returned by server setup and lifecycle operations.
pub type ServerResult<T> = std::result::Result<T, ServerError>;

/// Decision result using the same selected/fallback order as libsy.
#[derive(Serialize)]
struct DecisionResponse {
    selected: DecisionTargetResponse,
    fallbacks: Vec<DecisionTargetResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response: Option<Value>,
}

/// Response view over one target's existing configuration.
#[derive(Serialize)]
struct DecisionTargetResponse {
    target: String,
    model: ModelId,
    llm_client: DecisionLlmClientResponse,
    extra_body: BTreeMap<String, Value>,
}

/// Non-secret client settings needed to call a selected model.
#[derive(Serialize)]
struct DecisionLlmClientResponse {
    format: WireFormat,
    base_url: String,
}

/// Host hooks around the public LLM HTTP endpoints.
///
/// The default implementation preserves the standalone server's behavior. A host may
/// authenticate before body extraction, attach trusted metadata, map an external model
/// to a route model, and present responses without copying Switchyard's handlers.
#[async_trait]
pub trait IngressHooks: Send + Sync {
    /// Inspects or rejects a request before its body is read and decoded.
    ///
    /// `metadata` has already been normalized from request headers. Implementations may
    /// add trusted host context; returning a response skips body extraction and routing.
    async fn prepare(
        &self,
        _request: &mut HttpRequest,
        _metadata: &mut Metadata,
        _wire_format: WireFormat,
    ) -> std::result::Result<(), Response> {
        Ok(())
    }

    /// Resolves the route model after wire-format decoding and before route lookup.
    ///
    /// The returned model replaces `request.llm_request.model` for route execution.
    #[allow(clippy::result_large_err)]
    fn resolve_model(
        &self,
        request: &mut Request,
        _wire_format: WireFormat,
    ) -> std::result::Result<ModelId, Response> {
        request_model(request)
    }

    /// Presents the final HTTP response after routing or rejection.
    ///
    /// The default converts Switchyard's error envelope to the caller's wire format
    /// and leaves successful responses unchanged.
    fn present_response(&self, response: Response, wire_format: WireFormat) -> Response {
        render_error_response(response, wire_format)
    }
}

struct DefaultIngressHooks;

#[async_trait]
impl IngressHooks for DefaultIngressHooks {}

/// Shared server state used by all endpoint handlers.
#[derive(Clone)]
pub struct ServerState {
    runner: Arc<Runner>,
    metrics: prometheus::Registry,
    stats: StatsAccumulator,
    routing_log: Option<SharedRoutingLog>,
    track_cache_eligibility: bool,
    ingress_hooks: Arc<dyn IngressHooks>,
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
    /// Creates server state from route model IDs, algorithms, and per-target clients.
    pub fn new(routes: Vec<(ModelId, Arc<dyn Algorithm>, ClientRouter)>) -> ServerResult<Self> {
        let routes = routes
            .into_iter()
            .map(|(model, algorithm, clients)| {
                (
                    model,
                    Route::new(
                        algorithm,
                        clients,
                        None,
                        ModelCapabilities::default(),
                        None,
                        Vec::new(),
                    ),
                )
            })
            .collect();
        let runner = Runner::new(routes);
        Self::from_runner(runner)
    }

    /// Creates HTTP-server state around an already configured runner.
    pub fn from_runner(runner: Runner) -> ServerResult<Self> {
        let metrics = metrics::registry().map_err(ServerError::new)?;
        let stats = StatsAccumulator::new(
            metrics.clone(),
            runner.models().map(|model| model.algorithm),
        );
        Ok(Self {
            runner: Arc::new(runner),
            metrics,
            stats,
            routing_log: None,
            track_cache_eligibility: tracking_enabled_from_env(),
            ingress_hooks: Arc::new(DefaultIngressHooks),
        })
    }

    /// Installs host hooks for authentication, route-model resolution, and presentation.
    pub fn with_ingress_hooks(mut self, hooks: Arc<dyn IngressHooks>) -> Self {
        self.ingress_hooks = hooks;
        self
    }

    /// Enables durable per-request routing records at `path`.
    pub fn with_routing_log(mut self, path: impl Into<PathBuf>) -> ServerResult<Self> {
        self.routing_log = Some(SharedRoutingLog::new(path.into())?);
        Ok(self)
    }

    /// Returns the route model IDs served by the configured algorithms.
    pub fn models(&self) -> impl Iterator<Item = &str> {
        self.runner.models().map(|model| model.id.as_str())
    }

    /// Returns the caller credential family used by `model`, if any.
    pub fn caller_auth_kind(&self, model: &str) -> Option<&'static str> {
        self.runner
            .route(model)
            .and_then(|route| route.caller_auth())
            .map(|kind| kind.as_str())
    }

    fn route_for_model(&self, model: &str) -> Option<&Route> {
        self.runner.route(model)
    }

    fn decision_response(
        &self,
        route_model: &ModelId,
        outcome: &RoutingOutcome,
        response: Option<Value>,
    ) -> Option<DecisionResponse> {
        let description = self.runner.describe_decision(route_model, outcome)?;
        let convert = |target: DecisionTarget| DecisionTargetResponse {
            target: target.target,
            model: target.model,
            llm_client: DecisionLlmClientResponse {
                format: target.format,
                base_url: target.base_url,
            },
            extra_body: target.extra_body,
        };
        Some(DecisionResponse {
            selected: convert(description.selected),
            fallbacks: description.fallbacks.into_iter().map(convert).collect(),
            response,
        })
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

/// Routing-log marker distinguishing classifier and judge calls from answer calls.
const CLASSIFIER_TIER: &str = "classifier";

/// Maps answer-call observations to backend stats, routing calls to classifier/judge
/// stats, and records routing overhead once the algorithm run completes.
///
/// Successful classifier/judge calls are also appended to `classifier_log` when one is
/// configured, so per-session routing snapshots account for judge token overhead. Routed
/// calls stay off this path: the served call is logged with its terminal usage in
/// [`usage_metrics::observe`], which would make a second append here a double count.
fn stats_observer(
    stats: StatsAccumulator,
    classifier_log: Option<(SharedRoutingLog, routing_log::RoutingLogContext)>,
) -> RunObserver {
    Arc::new(move |observation| match observation {
        RunObservation::AnswerCall(call) => {
            let latency_ms = call.duration.as_secs_f64() * 1_000.0;
            if call.is_success {
                stats.record_success(&call.selected_model, latency_ms);
            } else {
                stats.record_error(&call.selected_model);
            }
        }
        RunObservation::LlmCall(call) => {
            let latency_ms = call.duration.as_secs_f64() * 1_000.0;
            if call.is_success {
                if let (Some((log, context)), Some(usage)) =
                    (classifier_log.as_ref(), call.usage.as_ref())
                {
                    log.append(
                        context.clone(),
                        &call.selected_model,
                        Some(CLASSIFIER_TIER),
                        usage,
                    );
                }
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
    let mut router = llm_routes()
        .route("/v1/decision", post(decision))
        .route("/v1/messages/count_tokens", post(anthropic_count_tokens))
        .route("/v1/models", get(models))
        .route("/v1/stats", get(get_stats))
        .route("/v1/stats/reset", post(reset_stats))
        .route("/metrics", get(prometheus_metrics))
        .route("/health", get(health));
    if state.routing_log.is_some() {
        router = router.route("/v1/routing/session-stats", get(get_session_stats));
    }
    finish_router(router, state)
}

/// Builds only the public Chat Completions, Responses, and Messages routes.
///
/// Hosts can layer their own internal endpoints and middleware without exposing the
/// standalone server's metrics, stats, decision, token-counting, or health routes.
pub fn build_llm_router(state: ServerState) -> Router {
    finish_router(llm_routes(), state)
}

fn llm_routes() -> Router<ServerState> {
    Router::new()
        .route("/v1/chat/completions", post(openai_chat_completions))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/responses", post(openai_responses))
}

fn finish_router(router: Router<ServerState>, state: ServerState) -> Router {
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
    request: HttpRequest,
) -> Response {
    handle_endpoint(state, started, request, WireFormat::OpenAiChat).await
}

async fn anthropic_messages(
    State(state): State<ServerState>,
    Extension(started): Extension<RequestStart>,
    request: HttpRequest,
) -> Response {
    handle_endpoint(state, started, request, WireFormat::AnthropicMessages).await
}

async fn openai_responses(
    State(state): State<ServerState>,
    Extension(started): Extension<RequestStart>,
    request: HttpRequest,
) -> Response {
    handle_endpoint(state, started, request, WireFormat::OpenAiResponses).await
}

/// One provider request submitted for a routing decision without an answer-model call.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DecisionEndpointRequest {
    input_format: WireFormat,
    request: Value,
}

/// Selects a target while still allowing the algorithm's classifier and judge calls.
async fn decision(
    State(state): State<ServerState>,
    headers: HeaderMap,
    body: std::result::Result<Json<DecisionEndpointRequest>, JsonRejection>,
) -> Response {
    let body = match body {
        Ok(Json(body)) if body.request.is_object() => body,
        Ok(_) => {
            return invalid_body_error(StatusCode::BAD_REQUEST, "request must be a JSON object");
        }
        Err(error) => {
            return invalid_body_error(
                error.status(),
                format!("Request body must be valid JSON: {error}"),
            );
        }
    };
    let input_format = body.input_format;
    let (route, request) = match resolve_route(
        &state,
        metadata_from_headers(headers),
        body.request,
        input_format,
    ) {
        Ok(resolved) => resolved,
        Err(response) => return response,
    };
    let route_model = request
        .llm_request
        .model
        .as_deref()
        .map(ModelId::from)
        .unwrap_or_default();
    let mut outcome = match route.decide(request).await {
        Ok(outcome) => outcome,
        Err(error) => return runner_error(error),
    };
    let response = match outcome.response.take() {
        Some(response) => {
            let aggregate = match response.llm_response.into_agg().await {
                Ok(aggregate) => aggregate,
                Err(error) => return client_error(&error),
            };
            // The request moved into the decision run, so its namespace mapping
            // is gone by here. A Codex tool call in this preview keeps its
            // qualified name.
            match encode_aggregated_response(
                &aggregate,
                input_format,
                Some(outcome.selected_model_id.as_str()),
            ) {
                Ok(response) => Some(response),
                Err(error) => return server_error(error.to_string()),
            }
        }
        None => None,
    };
    match state.decision_response(&route_model, &outcome, response) {
        Some(response) => Json(response).into_response(),
        None => {
            server_error("routing outcome contains a model with no callable target configuration")
        }
    }
}

/// Anthropic token counting against the route's explicitly configured target.
async fn anthropic_count_tokens(
    State(state): State<ServerState>,
    headers: HeaderMap,
    body: std::result::Result<Json<Value>, JsonRejection>,
) -> Response {
    let body = match llm_json_body(body) {
        Ok(body) => body,
        Err((status, message)) => {
            return anthropic_error_response(invalid_body_error(status, message));
        }
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
    anthropic_error_response(match route.count_tokens(request).await {
        Ok(payload) => (StatusCode::OK, Json(payload)).into_response(),
        Err(RunnerError::CountTokensUnsupported) => error_response(
            StatusCode::BAD_REQUEST,
            "route has no Anthropic target for token counting",
            "invalid_request_error",
            "count_tokens_unsupported",
        ),
        Err(RunnerError::Client(error)) => count_tokens_error(error),
        Err(error) => server_error(error.to_string()),
    })
}

/// Maps a token-count client failure with the same policy as a routed client call.
fn count_tokens_error(error: LlmClientError) -> Response {
    client_error(&error)
}

async fn handle_endpoint(
    state: ServerState,
    started: RequestStart,
    request: HttpRequest,
    wire_format: WireFormat,
) -> Response {
    let span = observability::request_span(request.headers());
    handle_endpoint_inner(state, started, request, wire_format)
        .instrument(span)
        .await
}

async fn handle_endpoint_inner(
    state: ServerState,
    started: RequestStart,
    mut request: HttpRequest,
    wire_format: WireFormat,
) -> Response {
    let ingress_hooks = Arc::clone(&state.ingress_hooks);
    let mut metadata = metadata_from_headers(request.headers().clone());
    if let Err(response) = ingress_hooks
        .prepare(&mut request, &mut metadata, wire_format)
        .await
    {
        let mut request_log = RequestLogGuard(Some(RequestLogContext {
            started: started.0,
            wire_format,
            requested_model: None,
            streaming: false,
            session_id: metadata.session_id.clone(),
            correlation_id: metadata.correlation_id.clone(),
        }));
        let response = ingress_hooks.present_response(response, wire_format);
        metrics::record_client_response(response.status().as_u16());
        request_log.emit(&response);
        return response;
    }
    let body = Json::<Value>::from_request(request, &state).await;
    let routing_log_context = state
        .routing_log
        .as_ref()
        .map(|_| routing_log::RoutingLogContext::from_metadata(&metadata));
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

    // Axum drops this future when the downstream client disconnects before the
    // response is written, which cancels the run and its upstream calls. The guard
    // keeps that terminal state observable: the `emit` path below never runs for an
    // abandoned buffered request.
    let mut request_log = RequestLogGuard(Some(request_log));

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
        Err((status, message)) => invalid_body_error(status, message),
    };
    let response = ingress_hooks.present_response(response, wire_format);
    metrics::record_client_response(response.status().as_u16());
    request_log.emit(&response);
    response
}

fn llm_json_body(
    body: std::result::Result<Json<Value>, JsonRejection>,
) -> std::result::Result<Value, (StatusCode, String)> {
    match body {
        Ok(Json(value)) if value.is_object() => Ok(value),
        Ok(_) => Err((
            StatusCode::BAD_REQUEST,
            "Request body must be a JSON object".to_string(),
        )),
        Err(error) => Err((
            error.status(),
            format!("Request body must be valid JSON: {error}"),
        )),
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
) -> std::result::Result<(&Route, Request), Response> {
    resolve_route_inner(state, metadata, body, wire_format, None)
}

#[allow(clippy::type_complexity, clippy::result_large_err)]
fn resolve_route_with_hooks(
    state: &ServerState,
    metadata: Metadata,
    body: Value,
    wire_format: WireFormat,
) -> std::result::Result<(&Route, Request), Response> {
    resolve_route_inner(
        state,
        metadata,
        body,
        wire_format,
        Some(state.ingress_hooks.as_ref()),
    )
}

#[allow(clippy::type_complexity, clippy::result_large_err)]
fn resolve_route_inner<'state>(
    state: &'state ServerState,
    metadata: Metadata,
    body: Value,
    wire_format: WireFormat,
    hooks: Option<&dyn IngressHooks>,
) -> std::result::Result<(&'state Route, Request), Response> {
    let llm_request = decode_request(wire_format, &body)
        .map_err(|error| invalid_body_error(StatusCode::BAD_REQUEST, error.to_string()))?;
    let mut request = Request {
        llm_request,
        raw_request: Some(body),
        metadata: Some(metadata),
    };
    let route_model = match hooks {
        Some(hooks) => hooks.resolve_model(&mut request, wire_format)?,
        None => request_model(&request)?,
    };
    request.llm_request.model = Some(route_model.to_string());
    let route = state.route_for_model(route_model.as_str()).ok_or_else(|| {
        error_response(
            StatusCode::NOT_FOUND,
            format!("No route registered for model {route_model}"),
            "model_not_found",
            "model_not_found",
        )
    })?;
    if let Err(RunnerError::IncompatibleCallerFormat(caller_auth)) =
        route.check_caller_format(wire_format)
    {
        let (provider, expected_endpoint) = match caller_auth {
            CallerAuthKind::Anthropic => ("Anthropic", "/v1/messages"),
            CallerAuthKind::OpenAi => ("OpenAI", "/v1/chat/completions or /v1/responses"),
        };
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "route {requested_model} forwards an {provider} login; call it through {expected_endpoint}",
                requested_model = route_model,
            ),
            "invalid_request_error",
            "invalid_request_error",
        ));
    }
    Ok((route, request))
}

// Callers immediately return the error as the HTTP response; boxing it would
// add allocation and make the host hook API less natural.
#[allow(clippy::result_large_err)]
fn request_model(request: &Request) -> std::result::Result<ModelId, Response> {
    request
        .llm_request
        .model
        .as_deref()
        .filter(|model| !model.trim().is_empty())
        .map(ModelId::from)
        .ok_or_else(|| {
            error_response(
                StatusCode::BAD_REQUEST,
                "request body must include a non-empty string `model`",
                "invalid_request_error",
                "invalid_request_error",
            )
        })
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
    let (route, request) = match resolve_route_with_hooks(&state, metadata, body, wire_format) {
        Ok(resolved) => resolved,
        Err(response) => return response,
    };
    // Only the Codex namespace mapping is needed downstream, not the whole request.
    let request_extensions = request.llm_request.extensions.clone();
    let observer = stats_observer(
        state.stats.clone(),
        state.routing_log.clone().zip(routing_log_context.clone()),
    );
    let output = match route.execute(request, Some(observer)).await {
        Ok(output) => output,
        Err(error) => return runner_error(error),
    };
    let RunOutput {
        selected_model,
        response,
    } = output;
    // The response carries the candidate that actually served it. Fall back to the routing
    // selection for algorithms that return a response without an offloaded model call.
    let served_model = response.served_model().cloned().or(Some(selected_model));
    let response = if let Some(served_model) = served_model.as_ref() {
        let cache_eligible = cache_probe
            .as_ref()
            .map(|probe| state.stats.prefix_eligibility(served_model, probe))
            .unwrap_or(0.0);
        usage_metrics::observe(
            response,
            served_model.as_str(),
            started.0,
            state.stats,
            cache_eligible,
            state.routing_log.zip(routing_log_context),
        )
    } else {
        response
    };

    let response_model = served_model.as_ref().map(ToString::to_string);
    let mut response =
        match into_http_response(response, wire_format, response_model, request_extensions) {
            Ok(response) => response,
            Err(error) => return server_error(error.to_string()),
        };
    if let Some(served_model) = served_model.as_ref() {
        attach_routing_headers(&mut response, served_model.as_str());
    }
    response
}

/// Holds the request log context until the request reaches a terminal state.
/// Dropping the guard with the context still in place means the handler future was
/// cancelled — the client left before a response existed — so the run is reported
/// as abandoned rather than silently disappearing.
struct RequestLogGuard(Option<RequestLogContext>);

impl RequestLogGuard {
    // Emits the normal terminal event and disarms the cancellation path.
    fn emit(&mut self, response: &Response) {
        if let Some(context) = self.0.take() {
            context.emit(response);
        }
    }
}

impl Drop for RequestLogGuard {
    fn drop(&mut self) {
        if let Some(context) = self.0.take() {
            context.emit_cancelled();
        }
    }
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

    // Terminal event for a request whose client left before a response was written.
    // No model served it, so there is no selected model and no response status.
    fn emit_cancelled(self) {
        metrics::record_client_disconnect();
        tracing::event!(
            target: "switchyard_server::request",
            Level::WARN,
            wire_format = %self.wire_format,
            status = CLIENT_CLOSED_REQUEST,
            requested_model = self.requested_model.as_deref().unwrap_or(""),
            selected_model = "",
            streaming = self.streaming,
            session_id = self.session_id.as_deref().unwrap_or(""),
            correlation_id = self.correlation_id.as_deref().unwrap_or(""),
            handling_duration_ms = self.started.elapsed().as_secs_f64() * 1_000.0,
            error = "client disconnected before a response was written",
            "LLM request cancelled"
        );
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

fn attach_routing_headers(response: &mut Response, served_model: &str) {
    insert_routing_header(response, HEADER_SELECTED_MODEL, served_model);
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

fn runner_error(error: RunnerError) -> Response {
    match error {
        RunnerError::Algorithm(error) => algorithm_error(error),
        RunnerError::Client(error) => client_error(&error),
        error => server_error(error.to_string()),
    }
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
            *status,
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

fn invalid_body_error(status: StatusCode, message: impl Into<String>) -> Response {
    error_response(status, message, "invalid_request_error", "invalid_body")
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
            .runner
            .models()
            .map(|model| (model.id.as_str(), model.capabilities)),
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
    let mut entries = entries.into_iter().collect::<Vec<_>>();
    entries.sort_unstable_by_key(|(model_id, _)| *model_id);
    let model_ids = entries.iter().map(|(model, _)| *model).collect::<Vec<_>>();
    let first_id = model_ids.first().copied();
    let last_id = model_ids.last().copied();
    json!({
        "object": "list",
        "data": entries.iter().map(|(model, caps)| model_entry_json(model, *caps)).collect::<Vec<_>>(),
        "models": entries
            .iter()
            .enumerate()
            .map(|(priority, (model, caps))| codex_model_entry_json(model, *caps, priority))
            .collect::<Vec<_>>(),
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

// Builds the metadata Codex requires when it discovers models from a direct provider.
//
// This mirrors Codex's `ModelInfo` card. The benchmark harness builds the same card in
// `benchmark/codex_model_catalog_lib.py`; keep the two in sync when Codex changes
// the shape. Every field below is either derived from the route's declared capabilities or a
// required `ModelInfo` field the server has no better value for.
//
// Two kinds of fields live here. context_window, tool_calling, and reasoning are model
// facts a backend can publish; the route declares them in config today. The rest
// (shell_type, apply_patch_tool_type, base_instructions, the reasoning-level presets,
// truncation_policy) are Codex client conventions no backend returns, so they stay
// constant.
//
// TODO: source context_window, tool_calling, and reasoning from the backend, not route
// config. Switchyard is a proxy, so it should re-publish what the backend advertises
// when it can — OpenRouter's /api/v1/models exposes context_length and
// supported_parameters — and fall back to the route's declared value. Some backends
// publish nothing (the NVIDIA gateway returns id-only models and blocks /model/info),
// so keep failing closed to config.
fn codex_model_entry_json(model: &str, capabilities: ModelCapabilities, priority: usize) -> Value {
    // Codex is non-functional without shell and apply_patch, so an undeclared tool
    // capability defaults to enabled here; the OpenAI `data` entry reports the raw
    // Option separately for clients that want the undeclared state.
    let tool_calling = capabilities.tool_calling.unwrap_or(true);
    let reasoning = capabilities.reasoning.unwrap_or(false);
    json!({
        "slug": model,
        "display_name": model,
        "description": "Switchyard-routed model.",
        "default_reasoning_level": if reasoning { json!("xhigh") } else { Value::Null },
        "supported_reasoning_levels": if reasoning { reasoning_levels() } else { json!([]) },
        "shell_type": if tool_calling { "shell_command" } else { "disabled" },
        "visibility": "list",
        "supported_in_api": true,
        // Catalog list position (routes are listed in sorted id order), not a quality rank.
        "priority": priority,
        "additional_speed_tiers": [],
        "availability_nux": null,
        "upgrade": null,
        // Required `ModelInfo` string. Unlike the launcher, the server cannot read
        // Codex's bundled prompt, so it sends a minimal stub.
        "base_instructions": "You are Codex, a coding agent.",
        "supports_reasoning_summaries": reasoning,
        "default_reasoning_summary": "none",
        "support_verbosity": reasoning,
        "default_verbosity": if reasoning { json!("low") } else { Value::Null },
        "apply_patch_tool_type": if tool_calling { Some("freeform") } else { None },
        "web_search_tool_type": "text",
        "truncation_policy": {"mode": "tokens", "limit": 10_000},
        "supports_parallel_tool_calls": tool_calling,
        "supports_image_detail_original": false,
        "context_window": capabilities.context_window,
        "max_context_window": capabilities.context_window,
        "effective_context_window_percent": 95,
        "experimental_supported_tools": [],
        "input_modalities": ["text"],
        "supports_search_tool": false,
    })
}

// The reasoning-effort presets Codex offers for a reasoning-capable route. Kept in
// step with the benchmark template in `codex_model_catalog_lib.py`.
fn reasoning_levels() -> Value {
    json!([
        {"effort": "low", "description": "Fast responses with lighter reasoning"},
        {"effort": "medium", "description": "Balances speed and reasoning depth"},
        {"effort": "high", "description": "Greater reasoning depth"},
        {"effort": "xhigh", "description": "Extra high reasoning depth"},
    ])
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
    use switchyard_llm_client::LlmCallObservation;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::sync::{Notify, oneshot};

    use super::*;

    /// A successful judge call lands in the per-session routing snapshot under its
    /// model id with the classifier tier, while routed calls stay off the observer's
    /// log path — they are logged with terminal usage when the served response is
    /// observed, so an append here would double count them.
    #[test]
    fn stats_observer_logs_judge_calls_to_the_routing_log() {
        let dir = tempfile::tempdir().expect("temp dir");
        let log = SharedRoutingLog::new(dir.path().join("routing.jsonl")).expect("routing log");
        let mut headers = HeaderMap::new();
        headers.insert("proxy_x_session_id", "session-1".parse().expect("header"));
        let metadata = metadata_from_headers(headers);
        let context = routing_log::RoutingLogContext::from_metadata(&metadata);
        let observer = stats_observer(StatsAccumulator::default(), Some((log.clone(), context)));

        let call = |model: &str, answer: bool| {
            let observation = LlmCallObservation {
                selected_model: ModelId::from(model),
                is_success: true,
                duration: Duration::from_millis(3),
                usage: Some(Usage {
                    input_tokens: Some(100),
                    output_tokens: Some(7),
                    ..Usage::default()
                }),
            };
            if answer {
                RunObservation::AnswerCall(observation)
            } else {
                RunObservation::LlmCall(observation)
            }
        };
        observer(call("judge-model", false));
        observer(call("routed-model", true));

        let snapshot = log
            .snapshot_session("session-1")
            .expect("read log")
            .expect("session recorded");
        let snapshot = serde_json::to_value(&snapshot).expect("serializable snapshot");
        assert_eq!(snapshot["models"]["judge-model"]["calls"], 1);
        assert_eq!(snapshot["models"]["judge-model"]["prompt_tokens"], 100);
        assert_eq!(snapshot["models"]["judge-model"]["completion_tokens"], 7);
        assert!(snapshot["models"].get("routed-model").is_none());
    }

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

    #[derive(Clone)]
    struct CancelTestState {
        started: Arc<Notify>,
        cancelled: Arc<Notify>,
    }

    // Signals when the handler future is dropped, standing in for the in-flight
    // upstream call that a cancelled run releases.
    struct CancelSentinel(Arc<Notify>);

    impl Drop for CancelSentinel {
        fn drop(&mut self) {
            self.0.notify_one();
        }
    }

    async fn never_responds(State(state): State<CancelTestState>) -> &'static str {
        let _sentinel = CancelSentinel(state.cancelled.clone());
        state.started.notify_one();
        std::future::pending::<()>().await;
        "unreachable"
    }

    // A buffered request produces no response until the run finishes, so nothing is
    // written to the socket to reveal that the client left. The server must still drop
    // the handler future on disconnect, otherwise upstream work outlives its client.
    #[tokio::test]
    async fn client_disconnect_cancels_a_buffered_handler() {
        let state = CancelTestState {
            started: Arc::new(Notify::new()),
            cancelled: Arc::new(Notify::new()),
        };
        let router = Router::new()
            .route("/", post(never_responds))
            .with_state(state.clone());
        let listener = bind_tcp_listener("127.0.0.1:0".parse().expect("valid address"), 16)
            .expect("listener binds");
        let addr = listener.local_addr().expect("listener has an address");
        let (shutdown, shutdown_receiver) = oneshot::channel();
        let server = tokio::spawn(serve(
            listener,
            router,
            Duration::from_millis(25),
            async move {
                let _ = shutdown_receiver.await;
            },
        ));

        let mut stream = tokio::net::TcpStream::connect(addr)
            .await
            .expect("client connects");
        stream
            .write_all(b"POST / HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}")
            .await
            .expect("client sends a complete request");
        state.started.notified().await;

        // The client goes away mid-run, before any response byte exists.
        drop(stream);

        tokio::time::timeout(Duration::from_secs(5), state.cancelled.notified())
            .await
            .expect("handler future is dropped when the client disconnects");

        let _ = shutdown.send(());
        let _ = tokio::time::timeout(Duration::from_secs(5), server).await;
    }

    // Collects the terminal request events emitted while it is the active subscriber.
    #[derive(Clone, Default)]
    struct CapturedEvents(Arc<Mutex<Vec<(Level, String)>>>);

    impl tracing_subscriber::Layer<tracing_subscriber::Registry> for CapturedEvents {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, tracing_subscriber::Registry>,
        ) {
            if event.metadata().target() != "switchyard_server::request" {
                return;
            }
            let mut message = String::new();
            event.record(
                &mut |field: &tracing::field::Field, value: &dyn std::fmt::Debug| {
                    if field.name() == "message" {
                        message = format!("{value:?}");
                    }
                },
            );
            self.0.lock().push((*event.metadata().level(), message));
        }
    }

    fn request_log_context() -> RequestLogContext {
        RequestLogContext {
            started: Instant::now(),
            wire_format: WireFormat::OpenAiChat,
            requested_model: Some("switchyard/route".to_string()),
            streaming: false,
            session_id: Some("session".to_string()),
            correlation_id: None,
        }
    }

    fn captured_events(run: impl FnOnce()) -> Vec<(Level, String)> {
        use tracing_subscriber::layer::SubscriberExt;

        let captured = CapturedEvents::default();
        let subscriber = tracing_subscriber::registry().with(captured.clone());
        tracing::subscriber::with_default(subscriber, run);
        captured.0.lock().clone()
    }

    // A dropped handler future is the only trace an abandoned buffered request leaves,
    // so the guard has to turn it into a terminal event rather than logging nothing.
    #[test]
    fn dropping_the_request_log_guard_reports_a_cancelled_request() {
        let events = captured_events(|| {
            drop(RequestLogGuard(Some(request_log_context())));
        });

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, Level::WARN);
        assert!(events[0].1.contains("cancelled"), "{:?}", events[0].1);
    }

    // A request that reached a response is not a cancellation, even though the guard
    // is still dropped afterwards.
    #[test]
    fn an_answered_request_reports_only_its_response() {
        let events = captured_events(|| {
            let mut guard = RequestLogGuard(Some(request_log_context()));
            guard.emit(&StatusCode::OK.into_response());
        });

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, Level::INFO);
        assert!(events[0].1.contains("handled"), "{:?}", events[0].1);
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
