// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust HTTP server surface for components-v2 profile configs.
//!
//! The serving path is profile-native: config files load into
//! `ProfileConfigPlan`, plans build `Profile` runtimes, and HTTP requests
//! call `Profile::run()` directly.

mod registry;
mod response;
mod sse;

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{rejection::JsonRejection, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use axum_server::tls_rustls::RustlsConfig;
use serde_json::{json, Value};
use tokio::net::{TcpListener, TcpSocket};

use switchyard_components_v2::{
    parse_profile_config_path, profile_stats_accumulator, ProfileConfigPlan, ProfileInput,
    ProfileResponse, RequestMetadata, RoutingMetadata,
};
use switchyard_core::{ChatRequest, ChatRequestType, RequestId, Result, SwitchyardError};
use switchyard_translation::{TranslationEngine, TranslationPolicy, WireFormat};

pub use registry::{ProfileRegistry, ServedModel};

use crate::response::{translate_chain_response, TranslatedResponse};

/// Default TCP listen backlog used by the Rust server.
pub const DEFAULT_LISTEN_BACKLOG: u32 = 65_535;

const HEADER_SELECTED_MODEL: &str = "x-model-router-selected-model";
const HEADER_SELECTED_TIER: &str = "x-model-router-selected-tier";
const HEADER_CONFIDENCE: &str = "x-model-router-confidence";
const HEADER_ROUTER_VERSION: &str = "x-model-router-version";
const HEADER_TOLERANCE: &str = "x-model-router-tolerance";
const HEADER_RATIONALE: &str = "x-model-router-rationale";
const MAX_ROUTING_HEADER_VALUE_LEN: usize = 512;

/// Shared server state used by all endpoint handlers.
#[derive(Clone)]
pub struct ServerState {
    registry: Arc<ProfileRegistry>,
    translation: Arc<TranslationEngine>,
    translation_policy: TranslationPolicy,
}

impl ServerState {
    /// Creates server state for a profile registry.
    pub fn new(registry: ProfileRegistry) -> Self {
        Self {
            registry: Arc::new(registry),
            translation: Arc::new(TranslationEngine::default()),
            translation_policy: TranslationPolicy::default(),
        }
    }

    /// Builds server state from a resolved profile config plan.
    pub fn from_plan(plan: &ProfileConfigPlan) -> Result<Self> {
        Ok(Self::new(ProfileRegistry::from_plan(plan)?))
    }

    /// Returns the profile registry used by this server.
    pub fn registry(&self) -> &ProfileRegistry {
        self.registry.as_ref()
    }

    /// Dispatches one request to the profile selected by its `model` field.
    async fn run_profile(&self, input: ProfileInput) -> Result<ProfileResponse> {
        let profile = self.registry.lookup(input.request.model())?;
        profile.run(input).await
    }
}

/// Runtime options shared by the Rust binary and Python binding.
#[derive(Clone, Debug)]
pub struct ServerRunOptions {
    /// Path to the components-v2 profile config file.
    pub config: PathBuf,
    /// Socket address to bind.
    pub addr: SocketAddr,
    /// TCP listen backlog.
    pub backlog: u32,
    /// Validate and print public model IDs without binding a socket.
    pub dry_run: bool,
    /// If this is set server is HTTPS.
    /// Caller must validate that the files exist.
    pub tls: Option<TLSOptions>,
}

#[derive(Clone, Debug)]
pub struct TLSOptions {
    /// TLS certificate path, PEM format
    pub cert: PathBuf,

    /// TLS certificate key path, PEM format
    pub key: PathBuf,
}

impl ServerRunOptions {
    fn is_tls(&self) -> bool {
        self.tls.is_some()
    }
}

/// Builds a server state by loading and resolving a profile config path.
fn state_from_config_path(path: impl AsRef<Path>) -> Result<ServerState> {
    let document = parse_profile_config_path(path)?;
    let plan = document.resolve()?;
    ServerState::from_plan(&plan)
}

/// Entry point to this module.
/// Loads config, optionally validates it, then starts the Rust server.
pub async fn run_server(options: ServerRunOptions) -> Result<()> {
    let state = state_from_config_path(&options.config)?;
    if options.dry_run {
        println!("{}", dry_run_summary(&options.config, state.registry()));
        return Ok(());
    }

    let listener = bind_tcp_listener(options.addr, options.backlog)?;
    let bound_addr = listener.local_addr().map_err(server_io_error)?;
    let server_options = ServerRunOptions {
        addr: bound_addr,
        ..options
    };
    eprintln!("{}", startup_banner(&server_options, state.registry()));
    let router = build_switchyard_router(state);
    if let Some(tls) = server_options.tls {
        serve_tls(listener, router, tls).await
    } else {
        serve(listener, router).await
    }
}

pub async fn serve_tls(listener: TcpListener, router: Router, tls: TLSOptions) -> Result<()> {
    // aws_lc_rs is the default but other crates pull in `ring` also,
    // so rustls doesn't know which one to use. Tell it.
    if let Err(e) = rustls::crypto::aws_lc_rs::default_provider().install_default() {
        eprintln!("TLS crypto provider already installed: {e:?}");
    }

    let config = RustlsConfig::from_pem_file(tls.cert, tls.key)
        .await
        .map_err(server_io_error)?;
    let handle = axum_server::Handle::new();

    let shutdown_handle = handle.clone();
    tokio::spawn(async move {
        shutdown_signal().await;
        // Drain connections. Later should loop on connection_count() > 0.
        shutdown_handle.graceful_shutdown(Some(Duration::from_secs(2)));
    });

    let std_listener = listener.into_std().map_err(server_io_error)?;
    axum_server::from_tcp_rustls(std_listener, config)
        .map_err(server_io_error)?
        .handle(handle.clone())
        .serve(router.into_make_service())
        .await
        .map_err(server_io_error)
}

/// Serves a Switchyard router on an already-bound TCP listener.
async fn serve(listener: TcpListener, router: Router) -> Result<()> {
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(server_io_error)
}

/// Builds an Axum router with the same primary endpoint paths as the Python app.
/// Public so that integration tests can see it.
pub fn build_switchyard_router(state: ServerState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(openai_chat_completions))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/responses", post(openai_responses))
        .route("/v1/models", get(models))
        // Keep the legacy routing stats aliases wired to the same handlers.
        .route("/v1/stats", get(stats))
        .route("/v1/stats/reset", post(reset_stats))
        .route("/v1/routing/stats", get(stats))
        .route("/v1/routing/stats/reset", post(reset_stats))
        .route("/health", get(health))
        .fallback(not_found)
        .with_state(state)
}

fn bind_tcp_listener(addr: SocketAddr, backlog: u32) -> Result<TcpListener> {
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

fn server_io_error(error: std::io::Error) -> SwitchyardError {
    SwitchyardError::Other(error.to_string())
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
    let body = match llm_json_body(body) {
        Ok(body) => body,
        Err(response) => return *response,
    };
    handle_llm_request(
        state,
        ChatRequest::openai_chat(body),
        WireFormat::OpenAiChat,
        metadata_from_headers(&headers, ChatRequestType::OpenAiChat),
    )
    .await
}

async fn anthropic_messages(
    State(state): State<ServerState>,
    headers: HeaderMap,
    body: std::result::Result<Json<Value>, JsonRejection>,
) -> Response {
    let body = match llm_json_body(body) {
        Ok(body) => body,
        Err(response) => return *response,
    };
    handle_llm_request(
        state,
        ChatRequest::anthropic(body),
        WireFormat::AnthropicMessages,
        metadata_from_headers(&headers, ChatRequestType::Anthropic),
    )
    .await
}

async fn openai_responses(
    State(state): State<ServerState>,
    headers: HeaderMap,
    body: std::result::Result<Json<Value>, JsonRejection>,
) -> Response {
    let body = match llm_json_body(body) {
        Ok(body) => body,
        Err(response) => return *response,
    };
    handle_llm_request(
        state,
        ChatRequest::openai_responses(body),
        WireFormat::OpenAiResponses,
        metadata_from_headers(&headers, ChatRequestType::OpenAiResponses),
    )
    .await
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
    request: ChatRequest,
    target_format: WireFormat,
    metadata: RequestMetadata,
) -> Response {
    if let Err(error) = request.validate() {
        return llm_error(error);
    }

    let profile_response = match state.run_profile(ProfileInput { request, metadata }).await {
        Ok(response) => response,
        Err(error) => return llm_error(error),
    };
    let (response, routing_metadata) = profile_response.into_parts();

    let mut response = match translate_chain_response(
        response,
        target_format,
        Arc::clone(&state.translation),
        state.translation_policy.clone(),
    ) {
        Ok(TranslatedResponse::Buffered(body)) => Json(body).into_response(),
        Ok(TranslatedResponse::Stream(stream)) => stream.into_response(),
        Err(error) => return server_error(error.to_string()),
    };
    attach_routing_metadata_headers(&mut response, routing_metadata.as_ref());
    response
}

fn attach_routing_metadata_headers(response: &mut Response, metadata: Option<&RoutingMetadata>) {
    let Some(metadata) = metadata else {
        return;
    };
    for (name, value) in routing_metadata_headers(metadata) {
        let name = HeaderName::from_static(name);
        let Ok(value) = HeaderValue::from_str(&value) else {
            continue;
        };
        response.headers_mut().insert(name, value);
    }
}

fn routing_metadata_headers(metadata: &RoutingMetadata) -> Vec<(&'static str, String)> {
    [
        (HEADER_SELECTED_MODEL, text_header(&metadata.selected_model)),
        (HEADER_SELECTED_TIER, text_header(&metadata.selected_tier)),
        (HEADER_CONFIDENCE, number_header(metadata.confidence)),
        (HEADER_ROUTER_VERSION, text_header(&metadata.router_version)),
        (HEADER_TOLERANCE, number_header(metadata.tolerance)),
        (HEADER_RATIONALE, text_header(&metadata.rationale)),
    ]
    .into_iter()
    .filter_map(|(name, value)| value.map(|value| (name, value)))
    .collect()
}

fn text_header(value: &Option<String>) -> Option<String> {
    value.as_deref().and_then(sanitize_routing_header_value)
}

fn number_header(value: Option<f64>) -> Option<String> {
    value
        .filter(|value| value.is_finite())
        .map(|value| value.to_string())
}

fn sanitize_routing_header_value(value: &str) -> Option<String> {
    let value = value.split_whitespace().collect::<Vec<_>>().join(" ");
    (!value.is_empty()).then(|| value.chars().take(MAX_ROUTING_HEADER_VALUE_LEN).collect())
}

fn metadata_from_headers(headers: &HeaderMap, inbound_format: ChatRequestType) -> RequestMetadata {
    RequestMetadata {
        request_id: request_id_from_headers(headers),
        inbound_format: Some(inbound_format),
        headers: normalized_headers(headers),
    }
}

fn request_id_from_headers(headers: &HeaderMap) -> Option<RequestId> {
    headers
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| RequestId::new(value.to_string()).ok())
}

fn normalized_headers(headers: &HeaderMap) -> BTreeMap<String, Vec<String>> {
    let mut normalized = BTreeMap::<String, Vec<String>>::new();
    for (name, value) in headers {
        let Ok(value) = value.to_str() else {
            continue;
        };
        normalized
            .entry(name.as_str().to_ascii_lowercase())
            .or_default()
            .push(value.to_string());
    }
    normalized
}

fn llm_error(error: SwitchyardError) -> Response {
    match error {
        SwitchyardError::ModelNotFound { model } => (
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": {
                    "message": format!("No route registered for model {}", model.as_str()),
                    "type": "model_not_found",
                    "code": "model_not_found",
                }
            })),
        )
            .into_response(),
        SwitchyardError::InvalidConfig(message) | SwitchyardError::InvalidRequest(message) => (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "code": "invalid_request_error",
                }
            })),
        )
            .into_response(),
        SwitchyardError::InvalidId(error) => (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": {
                    "message": error.to_string(),
                    "type": "invalid_request_error",
                    "code": "invalid_request_error",
                }
            })),
        )
            .into_response(),
        SwitchyardError::UpstreamHttp {
            provider,
            status_code,
            body,
        } => (
            StatusCode::from_u16(status_code).unwrap_or(StatusCode::BAD_GATEWAY),
            Json(json!({
                "error": {
                    "message": body,
                    "type": "upstream_error",
                    "code": "upstream_error",
                    "provider": provider,
                }
            })),
        )
            .into_response(),
        error => server_error(error.to_string()),
    }
}

fn server_error(message: String) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({
            "error": {
                "message": message,
                "type": "server_error",
                "code": "server_error",
            }
        })),
    )
        .into_response()
}

fn invalid_body_error(message: impl Into<String>) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({
            "error": {
                "message": message.into(),
                "type": "invalid_request_error",
                "code": "invalid_body",
            }
        })),
    )
        .into_response()
}

async fn models(State(state): State<ServerState>) -> Json<Value> {
    let entries = state.registry().served_models();
    Json(model_list_payload(&entries))
}

async fn stats() -> Response {
    match profile_stats_accumulator().snapshot() {
        Ok(snapshot) => Json(json!(snapshot)).into_response(),
        Err(error) => server_error(error.to_string()),
    }
}

async fn reset_stats() -> Response {
    match profile_stats_accumulator().reset() {
        Ok(()) => Json(json!({"status": "reset"})).into_response(),
        Err(error) => server_error(error.to_string()),
    }
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

async fn not_found() -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(json!({
            "detail": "Not Found",
        })),
    )
        .into_response()
}

fn model_list_payload(entries: &[ServedModel]) -> Value {
    let data = entries.iter().map(model_entry_json).collect::<Vec<_>>();
    let model_ids = entries
        .iter()
        .map(|entry| entry.id.as_str().to_string())
        .collect::<Vec<_>>();
    let first_id = model_ids.first().cloned();
    let last_id = model_ids.last().cloned();

    json!({
        "object": "list",
        "data": data,
        "first_id": first_id,
        "last_id": last_id,
        "has_more": false,
        "default_model": first_id,
        "model_pool": model_ids,
    })
}

fn model_entry_json(entry: &ServedModel) -> Value {
    json!({
        "id": entry.id.as_str(),
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

fn startup_banner(options: &ServerRunOptions, registry: &ProfileRegistry) -> String {
    let entries = registry.served_models();
    let scheme = if options.is_tls() { "https" } else { "http" };
    let listen_url = url_for_addr(scheme, options.addr);
    let local_url = local_url_for_addr(scheme, options.addr);
    let mut output = String::new();

    push_line(&mut output, "Switchyard Rust profile server");
    push_line(
        &mut output,
        format!("  config: {}", options.config.display()),
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
    push_line(&mut output, "    GET  /v1/routing/stats");
    push_line(&mut output, "    POST /v1/routing/stats/reset");
    push_line(&mut output, "");
    push_line(&mut output, "  available models:");
    for entry in &entries {
        if entry.display_name == entry.id.as_str() {
            push_line(&mut output, format!("    - {}", entry.id.as_str()));
        } else {
            push_line(
                &mut output,
                format!("    - {} ({})", entry.id.as_str(), entry.display_name),
            );
        }
    }
    push_line(&mut output, "");
    push_line(&mut output, "  try:");
    push_line(&mut output, format!("    curl -s {local_url}/health"));
    push_line(&mut output, format!("    curl -s {local_url}/v1/models"));
    if let Some(entry) = entries.first() {
        let payload = json!({
            "model": entry.id.as_str(),
            "messages": [{"role": "user", "content": "Say ok"}],
            "max_tokens": 256, // Space for thinking tokens
        });
        push_line(
            &mut output,
            format!(
                "    curl -s {local_url}/v1/chat/completions -H 'content-type: application/json' -d '{}'",
                payload
            ),
        );
    }
    if options.is_tls() {
        push_line(
            &mut output,
            "    For self-signed certificates use `curl --insecure ..`",
        );
    }
    push_line(&mut output, "");
    push_line(&mut output, "  stop: Ctrl-C");
    output
}

fn dry_run_summary(path: &Path, registry: &ProfileRegistry) -> String {
    let entries = registry.served_models();
    let mut output = String::new();
    push_line(
        &mut output,
        format!(
            "config OK: {}, public_models={}",
            path.display(),
            entries.len()
        ),
    );
    for entry in &entries {
        push_line(&mut output, format!("  - {}", entry.id.as_str()));
    }
    output
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
