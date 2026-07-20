// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust HTTP server for libsy algorithms.

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

use crate::response::{translate_response, TranslatedResponse};

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

/// Public model entry advertised by `/v1/models`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ServedModel {
    /// Public model or route ID accepted in inbound request bodies.
    pub id: String,
    /// Human-readable label shown by startup logs and `/v1/models`.
    pub display_name: String,
}

/// Shared server state used by all endpoint handlers.
#[derive(Clone)]
pub struct ServerState {
    algorithm: Arc<dyn Algorithm>,
    served_model: Arc<ServedModel>,
}

impl ServerState {
    /// Creates server state for one public route backed by a libsy algorithm.
    pub fn new(
        model: impl Into<String>,
        display_name: impl Into<String>,
        algorithm: Arc<dyn Algorithm>,
    ) -> ServerResult<Self> {
        let model = model.into();
        if model.trim().is_empty() {
            return Err(ServerError::new("route model must not be empty"));
        }
        Ok(Self {
            algorithm,
            served_model: Arc::new(ServedModel {
                id: model,
                display_name: display_name.into(),
            }),
        })
    }

    /// Returns the public model served by this algorithm.
    pub fn served_model(&self) -> &ServedModel {
        self.served_model.as_ref()
    }

    fn validate_model(&self, model: Option<&str>) -> std::result::Result<(), Box<Response>> {
        let Some(model) = model.filter(|model| !model.trim().is_empty()) else {
            return Err(Box::new(error_response(
                StatusCode::BAD_REQUEST,
                "request body must include a non-empty string `model`",
                "invalid_request_error",
                "invalid_request_error",
            )));
        };
        if model != self.served_model.id {
            return Err(Box::new(error_response(
                StatusCode::NOT_FOUND,
                format!("No route registered for model {model}"),
                "model_not_found",
                "model_not_found",
            )));
        }
        Ok(())
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
        Err(response) => return *response,
    };
    handle_llm_request(state, headers, body, wire_format).await
}

fn llm_json_body(
    body: std::result::Result<Json<Value>, JsonRejection>,
) -> std::result::Result<Value, Box<Response>> {
    match body {
        Ok(Json(value)) if value.is_object() => Ok(value),
        Ok(_) => Err(Box::new(invalid_body_error(
            "Request body must be a JSON object",
        ))),
        Err(error) => Err(Box::new(invalid_body_error(format!(
            "Request body must be valid JSON: {error}"
        )))),
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
    let requested_model = llm_request.model.clone();
    if let Err(response) = state.validate_model(requested_model.as_deref()) {
        return *response;
    }

    let request = Request {
        llm_request,
        raw_request: Some(body),
        metadata: Some(metadata_from_headers(&headers)),
    };
    let (trace, response) = match Arc::clone(&state.algorithm)
        .run(Context::default(), request)
        .await
    {
        Ok(result) => result,
        Err(error) => return algorithm_error(error),
    };

    let mut response = match translate_response(response, wire_format, requested_model) {
        Ok(TranslatedResponse::Buffered(body)) => Json(body).into_response(),
        Ok(TranslatedResponse::Stream(stream)) => stream.into_response(),
        Err(error) => return server_error(error.to_string()),
    };
    if let Some(decision) = trace.last() {
        attach_routing_headers(&mut response, decision.as_ref());
    }
    response
}

fn metadata_from_headers(headers: &HeaderMap) -> Metadata {
    Metadata {
        session_id: None,
        agent_id: None,
        task_id: None,
        correlation_id: header_text(headers, "x-request-id"),
        extra_metadata: None,
        http_headers: Some(normalized_headers(headers)),
        // Inbound format must not constrain the independently configured upstream.
        wire_format: None,
    }
}

fn header_text(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
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
    Json(model_list_payload(state.served_model()))
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

fn model_list_payload(entry: &ServedModel) -> Value {
    json!({
        "object": "list",
        "data": [model_entry_json(entry)],
        "first_id": entry.id,
        "last_id": entry.id,
        "has_more": false,
        "default_model": entry.id,
        "model_pool": [entry.id],
    })
}

fn model_entry_json(entry: &ServedModel) -> Value {
    json!({
        "id": entry.id,
        "object": "model",
        "type": "model",
        "created": 0,
        "owned_by": "switchyard",
        "display_name": entry.display_name,
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
    let listen_url = url_for_addr(scheme, options.addr);
    let local_url = local_url_for_addr(scheme, options.addr);
    let mut output = String::new();

    push_line(&mut output, "Switchyard libsy server");
    push_line(&mut output, format!("  model: {}", state.served_model.id));
    push_line(
        &mut output,
        format!("  algorithm: {}", state.served_model.display_name),
    );
    push_line(&mut output, format!("  listening: {listen_url}"));
    if local_url != listen_url {
        push_line(&mut output, format!("  local: {local_url}"));
    }
    push_line(&mut output, "");
    push_line(&mut output, "  endpoints:");
    push_line(&mut output, "    GET  /health");
    push_line(&mut output, "    GET  /v1/models");
    push_line(&mut output, "    POST /v1/chat/completions");
    push_line(&mut output, "    POST /v1/messages");
    push_line(&mut output, "    POST /v1/responses");
    push_line(&mut output, "");
    push_line(&mut output, "  try:");
    push_line(&mut output, format!("    curl -s {local_url}/health"));
    let chat_payload = json!({
        "model": state.served_model.id,
        "messages": [{"role": "user", "content": "Say OK"}],
        "max_tokens": 256,
    });
    push_line(
        &mut output,
        format!("    curl -s {local_url}/v1/chat/completions -H 'content-type: application/json' -d '{chat_payload}'"),
    );
    if options.is_tls() {
        push_line(
            &mut output,
            "    For self-signed certificates use `curl --insecure`.",
        );
    }
    push_line(&mut output, "");
    push_line(&mut output, "  stop: Ctrl-C");
    output
}

fn dry_run_summary(state: &ServerState) -> String {
    format!(
        "server OK: model={}, algorithm={}\n",
        state.served_model.id, state.served_model.display_name
    )
}

fn push_line(output: &mut String, line: impl AsRef<str>) {
    output.push_str(line.as_ref());
    output.push('\n');
}

fn url_for_addr(scheme: &'static str, addr: SocketAddr) -> String {
    format!("{scheme}://{}:{}", host_for_url(addr.ip()), addr.port())
}

fn local_url_for_addr(scheme: &'static str, addr: SocketAddr) -> String {
    let host = match addr.ip() {
        std::net::IpAddr::V4(ip) if ip.is_unspecified() => "127.0.0.1".to_string(),
        std::net::IpAddr::V6(ip) if ip.is_unspecified() => "[::1]".to_string(),
        ip => host_for_url(ip),
    };
    format!("{scheme}://{host}:{}", addr.port())
}

fn host_for_url(ip: std::net::IpAddr) -> String {
    match ip {
        std::net::IpAddr::V4(ip) => ip.to_string(),
        std::net::IpAddr::V6(ip) => format!("[{ip}]"),
    }
}
