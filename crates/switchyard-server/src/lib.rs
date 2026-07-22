// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust HTTP server for libsy algorithms.

pub mod config;
mod response;
mod sse;

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{rejection::JsonRejection, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use axum_server::tls_rustls::RustlsConfig;
use libsy::{Algorithm, Context, Decision, Metadata, Request};
use serde_json::{json, Value};
use switchyard_llm_client::LlmClientError;
use tokio::net::{TcpListener, TcpSocket};

use switchyard_translation::{decode_request, WireFormat};

use crate::response::into_http_response;

/// Default TCP listen backlog used by the Rust server.
pub const DEFAULT_LISTEN_BACKLOG: u32 = 65_535;

const HEADER_SELECTED_MODEL: &str = "x-model-router-selected-model";
const HEADER_RATIONALE: &str = "x-model-router-rationale";
const MAX_ROUTING_HEADER_VALUE_LEN: usize = 512;

type BoxError = Box<dyn Error + Send + Sync>;

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
        Ok(Self {
            routes: Arc::new(entries),
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

/// Builds an Axum router for the supported LLM wire formats.
pub fn build_switchyard_router(state: ServerState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(openai_chat_completions))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/responses", post(openai_responses))
        .route("/v1/models", get(models))
        .route("/health", get(health))
        .fallback(not_found)
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
    headers: HeaderMap,
    body: std::result::Result<Json<Value>, JsonRejection>,
) -> Response {
    handle_endpoint(state, headers, body, WireFormat::OpenAiChat).await
}

async fn anthropic_messages(
    State(state): State<ServerState>,
    headers: HeaderMap,
    body: std::result::Result<Json<Value>, JsonRejection>,
) -> Response {
    handle_endpoint(state, headers, body, WireFormat::AnthropicMessages).await
}

async fn openai_responses(
    State(state): State<ServerState>,
    headers: HeaderMap,
    body: std::result::Result<Json<Value>, JsonRejection>,
) -> Response {
    handle_endpoint(state, headers, body, WireFormat::OpenAiResponses).await
}

async fn handle_endpoint(
    state: ServerState,
    headers: HeaderMap,
    body: std::result::Result<Json<Value>, JsonRejection>,
    wire_format: WireFormat,
) -> Response {
    let body = match llm_json_body(body) {
        Ok(body) => body,
        Err(message) => return invalid_body_error(message),
    };
    handle_llm_request(state, headers, body, wire_format).await
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

async fn handle_llm_request(
    state: ServerState,
    headers: HeaderMap,
    body: Value,
    wire_format: WireFormat,
) -> Response {
    let llm_request = match decode_request(wire_format, &body) {
        Ok(request) => request,
        Err(error) => return invalid_body_error(error.to_string()),
    };
    let Some(requested_model) = llm_request
        .model
        .clone()
        .filter(|model| !model.trim().is_empty())
    else {
        return error_response(
            StatusCode::BAD_REQUEST,
            "request body must include a non-empty string `model`",
            "invalid_request_error",
            "invalid_request_error",
        );
    };
    let Some(algorithm) = state.algorithm_for_model(&requested_model) else {
        return error_response(
            StatusCode::NOT_FOUND,
            format!("No route registered for model {requested_model}"),
            "model_not_found",
            "model_not_found",
        );
    };

    let request = Request {
        llm_request,
        raw_request: Some(body),
        metadata: Some(metadata_from_headers(&headers)),
    };
    let (trace, response) = match algorithm.run(Context::default(), request).await {
        Ok(result) => result,
        Err(error) => return algorithm_error(error),
    };

    let mut response = match into_http_response(response, wire_format, Some(requested_model)) {
        Ok(response) => response,
        Err(error) => return server_error(error.to_string()),
    };
    if let Some(decision) = trace.last() {
        attach_routing_headers(&mut response, decision.as_ref());
    }
    response
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

fn algorithm_error(error: BoxError) -> Response {
    let Some(error) = error.downcast_ref::<LlmClientError>() else {
        return server_error(error.to_string());
    };
    match error {
        LlmClientError::UnknownModel(model) => error_response(
            StatusCode::BAD_GATEWAY,
            format!("No upstream backend configured for model {model}"),
            "upstream_error",
            "upstream_model_not_found",
        ),
        LlmClientError::UnknownModelFormat { model, format } => error_response(
            StatusCode::BAD_GATEWAY,
            format!("No upstream backend configured for model {model} and format {format}"),
            "upstream_error",
            "upstream_format_not_found",
        ),
        LlmClientError::MissingModel => error_response(
            StatusCode::BAD_REQUEST,
            error.to_string(),
            "invalid_request_error",
            "invalid_request_error",
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
        LlmClientError::Translation(message)
        | LlmClientError::Transport(message)
        | LlmClientError::Stream(message) => error_response(
            StatusCode::BAD_GATEWAY,
            message,
            "upstream_error",
            "upstream_error",
        ),
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
    (
        status,
        Json(json!({
            "error": {
                "message": message.into(),
                "type": error_type,
                "code": code,
            }
        })),
    )
        .into_response()
}

async fn models(State(state): State<ServerState>) -> Json<Value> {
    Json(model_list_payload(state.models()))
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
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
