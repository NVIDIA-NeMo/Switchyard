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

use axum::body::to_bytes;
use axum::extract::{rejection::JsonRejection, Request, State};
use axum::http::header::{AUTHORIZATION, CONTENT_LENGTH, CONTENT_TYPE, WWW_AUTHENTICATE};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use switchyard_components_v2::{
    parse_profile_config_path, profile_stats_accumulator, session_id_from_normalized_headers,
    DecisionContext, ProfileConfigPlan, ProfileInput, ProfileResponse, RelayIdentityKey,
    RelaySnapshotAccumulator, RelaySnapshotLimits, RequestMetadata, RoutingMetadata,
    RoutingRequest, PROXY_SESSION_ID_HEADER, RELAY_SESSION_ID_HEADER,
};
use switchyard_core::{ChatRequest, ChatRequestType, RequestId, Result, SwitchyardError};
use switchyard_translation::{TranslationEngine, TranslationPolicy, WireFormat};
use tokio::net::{TcpListener, TcpSocket};

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
    relay_snapshots: Arc<RelaySnapshotAccumulator>,
    atof_bearer_token: Arc<Option<String>>,
}

impl ServerState {
    /// Creates server state for a profile registry.
    pub fn new(registry: ProfileRegistry) -> Self {
        Self {
            registry: Arc::new(registry),
            translation: Arc::new(TranslationEngine::default()),
            translation_policy: TranslationPolicy::default(),
            relay_snapshots: Arc::new(RelaySnapshotAccumulator::default()),
            atof_bearer_token: Arc::new(None),
        }
    }

    /// Builds server state from a resolved profile config plan.
    pub fn from_plan(plan: &ProfileConfigPlan) -> Result<Self> {
        Ok(Self::new(ProfileRegistry::from_plan(plan)?))
    }

    /// Returns a copy with optional auth; explicit blank tokens are invalid config.
    pub fn with_atof_bearer_token(mut self, token: Option<String>) -> Result<Self> {
        self.atof_bearer_token = Arc::new(validate_optional_bearer_token(token)?);
        Ok(self)
    }

    /// Returns a copy of this state with an empty accumulator using explicit limits.
    pub fn with_relay_snapshot_limits(mut self, limits: RelaySnapshotLimits) -> Result<Self> {
        self.relay_snapshots = Arc::new(RelaySnapshotAccumulator::with_limits(limits)?);
        Ok(self)
    }

    /// Returns the profile registry used by this server.
    pub fn registry(&self) -> &ProfileRegistry {
        self.registry.as_ref()
    }

    /// Returns the server-owned Relay snapshot accumulator.
    pub fn relay_snapshots(&self) -> &RelaySnapshotAccumulator {
        self.relay_snapshots.as_ref()
    }

    /// Dispatches one request to the profile selected by its `model` field.
    async fn run_profile(&self, input: ProfileInput) -> Result<ProfileResponse> {
        let profile = self.registry.lookup(input.request.model())?;
        profile.run(input).await
    }

    /// Produces one routing decision through the configured profile registry.
    async fn decide_route(
        &self,
        request: RoutingRequest,
    ) -> Result<switchyard_components_v2::RoutingDecision> {
        request.validate()?;
        let owner_id = request
            .identity
            .owner_id
            .as_deref()
            .map(str::trim)
            .filter(|owner_id| !owner_id.is_empty())
            .map(ToOwned::to_owned);
        let key = RelayIdentityKey::new(request.identity.session_id.trim(), owner_id);
        let snapshot = self.relay_snapshots.snapshot(&key);
        self.registry
            .decide(DecisionContext::new(request, snapshot))
            .await
    }

    /// Allows unauthenticated ingestion only when no bearer token was configured.
    fn authorize_atof_ingest(&self, headers: &HeaderMap) -> std::result::Result<(), Box<Response>> {
        let Some(expected) = self.atof_bearer_token.as_ref() else {
            return Ok(());
        };
        let authorized = headers
            .get(AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .and_then(parse_bearer_token)
            .is_some_and(|actual| actual == expected);
        if authorized {
            Ok(())
        } else {
            Err(Box::new(unauthorized_error(
                "ATOF ingestion authorization failed",
            )))
        }
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
    /// Optional bearer token required by `POST /v1/atof/events`.
    pub atof_bearer_token: Option<String>,
    /// Fixed memory and request-size bounds for Relay snapshot accumulation.
    pub relay_snapshot_limits: RelaySnapshotLimits,
}

/// Builds a server state by loading and resolving a profile config path.
pub fn state_from_config_path(path: impl AsRef<Path>) -> Result<ServerState> {
    let document = parse_profile_config_path(path)?;
    let plan = document.resolve()?;
    ServerState::from_plan(&plan)
}

/// Loads config, optionally validates it, then starts the Rust server.
pub async fn run_server(options: ServerRunOptions) -> Result<()> {
    let state = state_from_config_path(&options.config)?
        .with_relay_snapshot_limits(options.relay_snapshot_limits)?
        .with_atof_bearer_token(options.atof_bearer_token.clone())?;
    if options.dry_run {
        println!("{}", dry_run_summary(&options.config, state.registry()));
        return Ok(());
    }

    let listener = bind_tcp_listener(options.addr, options.backlog)?;
    let bound_addr = listener.local_addr().map_err(server_io_error)?;
    let banner_options = ServerRunOptions {
        addr: bound_addr,
        ..options
    };
    eprintln!("{}", startup_banner(&banner_options, state.registry()));
    serve(listener, state).await
}

/// Builds an Axum router with the same primary endpoint paths as the Python app.
pub fn build_switchyard_router(state: ServerState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(openai_chat_completions))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/responses", post(openai_responses))
        .route("/v1/routing/decision", post(routing_decision))
        .route("/v1/atof/events", post(atof_events))
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

/// Serves a Switchyard router on an already-bound TCP listener.
pub async fn serve(listener: TcpListener, state: ServerState) -> Result<()> {
    axum::serve(listener, build_switchyard_router(state))
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(server_io_error)
}

/// Binds and serves a Switchyard router.
pub async fn serve_addr(addr: SocketAddr, state: ServerState) -> Result<()> {
    serve_addr_with_backlog(addr, DEFAULT_LISTEN_BACKLOG, state).await
}

/// Binds with an explicit TCP listen backlog and serves a Switchyard router.
pub async fn serve_addr_with_backlog(
    addr: SocketAddr,
    backlog: u32,
    state: ServerState,
) -> Result<()> {
    let listener = bind_tcp_listener(addr, backlog)?;
    serve(listener, state).await
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
    let metadata = match metadata_from_headers(&headers, ChatRequestType::OpenAiChat) {
        Ok(metadata) => metadata,
        Err(error) => return llm_error(error),
    };
    handle_llm_request(
        state,
        ChatRequest::openai_chat(body),
        WireFormat::OpenAiChat,
        metadata,
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
    let metadata = match metadata_from_headers(&headers, ChatRequestType::Anthropic) {
        Ok(metadata) => metadata,
        Err(error) => return llm_error(error),
    };
    handle_llm_request(
        state,
        ChatRequest::anthropic(body),
        WireFormat::AnthropicMessages,
        metadata,
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
    let metadata = match metadata_from_headers(&headers, ChatRequestType::OpenAiResponses) {
        Ok(metadata) => metadata,
        Err(error) => return llm_error(error),
    };
    handle_llm_request(
        state,
        ChatRequest::openai_responses(body),
        WireFormat::OpenAiResponses,
        metadata,
    )
    .await
}

async fn routing_decision(
    State(state): State<ServerState>,
    headers: HeaderMap,
    body: std::result::Result<Json<RoutingRequest>, JsonRejection>,
) -> Response {
    let request = match routing_json_body(body) {
        Ok(request) => request,
        Err(response) => return *response,
    };
    let normalized_headers = match normalized_headers(&headers) {
        Ok(headers) => headers,
        Err(error) => return llm_error(error),
    };
    let header_session_id = match session_id_from_normalized_headers(&normalized_headers) {
        Ok(session_id) => session_id,
        Err(error) => return llm_error(error),
    };
    if let Some(header_session_id) = header_session_id {
        if header_session_id != request.identity.session_id.trim() {
            return llm_error(SwitchyardError::InvalidRequest(format!(
                "Decision session header {header_session_id:?} does not match identity.session_id {:?}",
                request.identity.session_id
            )));
        }
    }
    match state.decide_route(request).await {
        Ok(decision) => Json(decision).into_response(),
        Err(error) => llm_error(error),
    }
}

fn routing_json_body(
    body: std::result::Result<Json<RoutingRequest>, JsonRejection>,
) -> std::result::Result<RoutingRequest, Box<Response>> {
    match body {
        Ok(Json(value)) => Ok(value),
        Err(error) => Err(Box::new(invalid_body_error(format!(
            "Routing decision request body must match the JSON contract: {error}"
        )))),
    }
}

/// Accepts bounded Relay `http_post` batches encoded as one JSON object per line.
async fn atof_events(State(state): State<ServerState>, request: Request) -> Response {
    let (parts, body) = request.into_parts();
    if let Err(response) = state.authorize_atof_ingest(&parts.headers) {
        return *response;
    }
    if !is_ndjson_content_type(&parts.headers) {
        return unsupported_media_type_error(
            "ATOF ingestion requires content-type application/x-ndjson",
        );
    }

    let limits = state.relay_snapshots().limits();
    if content_length(&parts.headers).is_some_and(|length| length > limits.max_batch_bytes) {
        return payload_too_large_error(format!(
            "ATOF batch exceeds the {} byte maximum",
            limits.max_batch_bytes
        ));
    }
    let body = match to_bytes(body, limits.max_batch_bytes).await {
        Ok(body) => body,
        Err(error) => {
            return payload_too_large_error(format!(
                "ATOF batch exceeds the {} byte maximum: {error}",
                limits.max_batch_bytes
            ));
        }
    };
    let events = match parse_atof_ndjson(&body, limits.max_event_bytes) {
        Ok(events) => events,
        Err(response) => return *response,
    };
    let report = match state.relay_snapshots().ingest_batch(&events) {
        Ok(report) => report,
        Err(error) => return llm_error(error),
    };
    let counters = state.relay_snapshots().counters();

    Json(json!({
        "status": "ok",
        // Keep the original prototype names while exposing the complete report.
        "accepted_events": report.received_events,
        "accumulator_ingests": report.ingested_events,
        "batch": report,
        "cumulative": counters,
    }))
    .into_response()
}

fn parse_atof_ndjson(
    body: &[u8],
    max_event_bytes: usize,
) -> std::result::Result<Vec<Value>, Box<Response>> {
    // Parse every non-empty line before calling the accumulator so malformed
    // batches cannot partially mutate retained state.
    let mut events = Vec::new();
    for (index, raw_line) in body.split(|byte| *byte == b'\n').enumerate() {
        let line = trim_ascii_whitespace(raw_line);
        if line.is_empty() {
            continue;
        }
        if line.len() > max_event_bytes {
            return Err(Box::new(payload_too_large_error(format!(
                "ATOF NDJSON line {} is {} bytes; maximum is {max_event_bytes} bytes",
                index + 1,
                line.len()
            ))));
        }
        match serde_json::from_slice::<Value>(line) {
            Ok(event) => events.push(event),
            Err(error) => {
                return Err(Box::new(invalid_body_error(format!(
                    "ATOF NDJSON line {} must be valid JSON: {error}",
                    index + 1
                ))));
            }
        }
    }
    Ok(events)
}

fn trim_ascii_whitespace(mut value: &[u8]) -> &[u8] {
    while value.first().is_some_and(u8::is_ascii_whitespace) {
        value = &value[1..];
    }
    while value.last().is_some_and(u8::is_ascii_whitespace) {
        value = &value[..value.len() - 1];
    }
    value
}

/// Accepts the canonical NDJSON media type with optional MIME parameters.
fn is_ndjson_content_type(headers: &HeaderMap) -> bool {
    headers
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("application/x-ndjson"))
}

fn content_length(headers: &HeaderMap) -> Option<usize> {
    headers
        .get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse().ok())
}

/// Distinguishes disabled auth (`None`) from unsafe explicit blank configuration.
fn validate_optional_bearer_token(token: Option<String>) -> Result<Option<String>> {
    let Some(token) = token else {
        return Ok(None);
    };
    let token = token.trim();
    if token.is_empty() {
        return Err(SwitchyardError::InvalidConfig(
            "ATOF bearer token cannot be blank; omit it to disable authentication".to_string(),
        ));
    }
    Ok(Some(token.to_string()))
}

/// Parses exactly one non-empty case-insensitive Bearer scheme and token.
fn parse_bearer_token(value: &str) -> Option<&str> {
    let (scheme, token) = value.split_once(' ')?;
    (scheme.eq_ignore_ascii_case("bearer") && !token.is_empty()).then_some(token)
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

fn metadata_from_headers(
    headers: &HeaderMap,
    inbound_format: ChatRequestType,
) -> Result<RequestMetadata> {
    let normalized = normalized_headers(headers)?;
    Ok(RequestMetadata {
        session_id: session_id_from_normalized_headers(&normalized)?,
        request_id: request_id_from_headers(headers),
        inbound_format: Some(inbound_format),
        headers: normalized,
    })
}

fn request_id_from_headers(headers: &HeaderMap) -> Option<RequestId> {
    headers
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| RequestId::new(value.to_string()).ok())
}

fn normalized_headers(headers: &HeaderMap) -> Result<BTreeMap<String, Vec<String>>> {
    let mut normalized = BTreeMap::<String, Vec<String>>::new();
    for name in headers.keys() {
        let name = name.as_str().to_ascii_lowercase();
        for value in headers.get_all(name.as_str()) {
            let value = match value.to_str() {
                Ok(value) => value,
                Err(_)
                    if matches!(
                        name.as_str(),
                        PROXY_SESSION_ID_HEADER | RELAY_SESSION_ID_HEADER
                    ) =>
                {
                    return Err(SwitchyardError::InvalidRequest(format!(
                        "session header {name} must contain valid UTF-8"
                    )));
                }
                Err(_) => continue,
            };
            normalized
                .entry(name.clone())
                .or_default()
                .push(value.to_string());
        }
    }
    Ok(normalized)
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
        SwitchyardError::DecisionProfileNotFound { profile_id } => (
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": {
                    "message": format!("No Decision API profile registered for {profile_id}"),
                    "type": "decision_profile_not_found",
                    "code": "decision_profile_not_found",
                }
            })),
        )
            .into_response(),
        SwitchyardError::DecisionUnsupported { profile_id } => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({
                "error": {
                    "message": format!("Profile {profile_id} does not support the Decision API"),
                    "type": "unsupported_profile_error",
                    "code": "decision_not_supported",
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

fn unauthorized_error(message: impl Into<String>) -> Response {
    let mut response = (
        StatusCode::UNAUTHORIZED,
        Json(json!({
            "error": {
                "message": message.into(),
                "type": "authentication_error",
                "code": "unauthorized",
            }
        })),
    )
        .into_response();
    response
        .headers_mut()
        .insert(WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
    response
}

fn unsupported_media_type_error(message: impl Into<String>) -> Response {
    (
        StatusCode::UNSUPPORTED_MEDIA_TYPE,
        Json(json!({
            "error": {
                "message": message.into(),
                "type": "invalid_request_error",
                "code": "unsupported_media_type",
            }
        })),
    )
        .into_response()
}

fn payload_too_large_error(message: impl Into<String>) -> Response {
    (
        StatusCode::PAYLOAD_TOO_LARGE,
        Json(json!({
            "error": {
                "message": message.into(),
                "type": "invalid_request_error",
                "code": "payload_too_large",
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
    let listen_url = url_for_addr(options.addr);
    let local_url = local_url_for_addr(options.addr);
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
    push_line(&mut output, "    POST /v1/routing/decision");
    push_line(&mut output, "    POST /v1/atof/events");
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
            "max_tokens": 8,
        });
        push_line(
            &mut output,
            format!(
                "    curl -s {local_url}/v1/chat/completions -H 'content-type: application/json' -d '{}'",
                payload
            ),
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

fn url_for_addr(addr: SocketAddr) -> String {
    format!("http://{}:{}", host_for_url(addr.ip()), addr.port())
}

fn local_url_for_addr(addr: SocketAddr) -> String {
    let host = match addr.ip() {
        std::net::IpAddr::V4(ip) if ip.is_unspecified() => "127.0.0.1".to_string(),
        std::net::IpAddr::V6(ip) if ip.is_unspecified() => "[::1]".to_string(),
        ip => host_for_url(ip),
    };
    format!("http://{host}:{}", addr.port())
}

fn host_for_url(ip: std::net::IpAddr) -> String {
    match ip {
        std::net::IpAddr::V4(ip) => ip.to_string(),
        std::net::IpAddr::V6(ip) => format!("[{ip}]"),
    }
}
