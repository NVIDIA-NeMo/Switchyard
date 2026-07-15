// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal axum proxy exposing the LLM wire APIs over [`LlmModelClient`].
//!
//! Each inbound request is decoded to Switchyard's neutral IR, dispatched to the
//! configured upstream via `libsy-llm-client`, and the IR response is encoded
//! back into the same wire format the client used (buffered JSON or SSE). The
//! HTTP surface mirrors `switchyard-server` but swaps the profile-chain executor
//! for a single client. See [`build_router`].

mod encode;
mod sse;

pub mod cli;

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use axum::extract::rejection::JsonRejection;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use libsy_llm_client::{LlmClientError, LlmModelClient};
use libsy_protocol::{LlmResponse, Metadata, Request};
use serde_json::{json, Value};
use switchyard_translation::{TranslationEngine, TranslationPolicy, WireFormat};

use crate::encode::{encode_buffered, encode_stream};

/// The single model name this proxy advertises to clients.
pub const SERVED_MODEL: &str = "switchyard";

/// Shared state for all endpoint handlers.
#[derive(Clone)]
pub struct ProxyState {
    client: Arc<LlmModelClient>,
    /// Model id sent upstream and used as the client's map key.
    upstream_model: Arc<str>,
    /// Wire formats for which an upstream backend was configured.
    available: Arc<HashSet<WireFormat>>,
    /// Fallback upstream format when the inbound format has no backend.
    fallback_format: Option<WireFormat>,
    engine: Arc<TranslationEngine>,
    policy: TranslationPolicy,
}

impl ProxyState {
    /// Builds proxy state around a configured client.
    ///
    /// `available` is the set of wire formats the client has backends for;
    /// `fallback_format` (from `--upstream-format`) is used when an inbound
    /// format has no matching backend.
    pub fn new(
        client: LlmModelClient,
        upstream_model: impl Into<Arc<str>>,
        available: HashSet<WireFormat>,
        fallback_format: Option<WireFormat>,
    ) -> Self {
        Self {
            client: Arc::new(client),
            upstream_model: upstream_model.into(),
            available: Arc::new(available),
            fallback_format,
            engine: Arc::new(TranslationEngine::default()),
            policy: TranslationPolicy::default(),
        }
    }
}

/// Builds the proxy router. Public so integration tests can drive it without a socket.
pub fn build_router(state: ProxyState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(openai_chat_completions))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/responses", post(openai_responses))
        .route("/v1/models", get(models))
        .route("/health", get(health))
        .fallback(not_found)
        .with_state(state)
}

async fn openai_chat_completions(
    State(state): State<ProxyState>,
    headers: HeaderMap,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    dispatch(state, headers, body, WireFormat::OpenAiChat).await
}

async fn anthropic_messages(
    State(state): State<ProxyState>,
    headers: HeaderMap,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    dispatch(state, headers, body, WireFormat::AnthropicMessages).await
}

async fn openai_responses(
    State(state): State<ProxyState>,
    headers: HeaderMap,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    dispatch(state, headers, body, WireFormat::OpenAiResponses).await
}

// Extracts and validates the JSON body, then runs the shared request path.
async fn dispatch(
    state: ProxyState,
    headers: HeaderMap,
    body: Result<Json<Value>, JsonRejection>,
    inbound: WireFormat,
) -> Response {
    let body = match body {
        Ok(Json(value)) if value.is_object() => value,
        Ok(_) => return invalid_body("Request body must be a JSON object"),
        Err(error) => return invalid_body(format!("Request body must be valid JSON: {error}")),
    };
    handle(state, headers, body, inbound).await
}

// Decode inbound → call upstream → encode back to the inbound format.
async fn handle(
    state: ProxyState,
    headers: HeaderMap,
    body: Value,
    inbound: WireFormat,
) -> Response {
    let decoded = match state.engine.decode_request(inbound, &body, &state.policy) {
        Ok(decoded) => decoded,
        Err(error) => return bad_request(format!("failed to decode {inbound} request: {error}")),
    };
    let llm_request = decoded.request;
    let requested_model = llm_request.model.clone();

    // Prefer the inbound format; otherwise fall back to --upstream-format.
    let format = if state.available.contains(&inbound) {
        inbound
    } else if let Some(format) = state
        .fallback_format
        .filter(|format| state.available.contains(format))
    {
        format
    } else {
        return bad_request(format!(
            "no upstream backend for {inbound}; set the provider API key or --upstream-format"
        ));
    };

    let request = Request {
        llm_request,
        raw_request: None,
        metadata: Some(metadata_from_headers(&headers)),
    };

    let response = match state
        .client
        .call(request, Some(&state.upstream_model), format)
        .await
    {
        Ok(response) => response,
        Err(error) => return map_client_error(error),
    };

    match response.llm_response {
        LlmResponse::Agg(agg) => match encode_buffered(
            &state.engine,
            &state.policy,
            &agg,
            inbound,
            requested_model.as_deref(),
        ) {
            Ok(body) => Json(body).into_response(),
            Err(error) => server_error(format!("failed to encode {inbound} response: {error}")),
        },
        LlmResponse::Stream(chunks) => {
            encode_stream(chunks, inbound, requested_model).into_response()
        }
    }
}

// Carries correlation id and the full lowercased inbound header map. The client
// drops reserved/auth headers and injects the backend's real credential, so the
// sentinel key coding agents send is filtered automatically.
fn metadata_from_headers(headers: &HeaderMap) -> Metadata {
    Metadata {
        session_id: None,
        agent_id: None,
        task_id: None,
        correlation_id: header_value(headers, "x-request-id"),
        extra_metadata: None,
        http_headers: Some(normalized_headers(headers)),
    }
}

fn header_value(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
}

// Lowercases header names and keeps the first UTF-8 value for each.
fn normalized_headers(headers: &HeaderMap) -> BTreeMap<String, String> {
    let mut normalized = BTreeMap::new();
    for (name, value) in headers {
        let Ok(value) = value.to_str() else {
            continue;
        };
        normalized
            .entry(name.as_str().to_ascii_lowercase())
            .or_insert_with(|| value.to_string());
    }
    normalized
}

// Maps client errors onto OpenAI-style HTTP error envelopes.
fn map_client_error(error: LlmClientError) -> Response {
    match error {
        LlmClientError::UnknownModel(model) => error_body(
            StatusCode::NOT_FOUND,
            format!("no backend configured for model {model:?}"),
            "model_not_found",
        ),
        LlmClientError::UnknownModelFormat { .. } => server_error(error.to_string()),
        LlmClientError::MissingModel | LlmClientError::Translation(_) => error_body(
            StatusCode::BAD_REQUEST,
            error.to_string(),
            "invalid_request_error",
        ),
        LlmClientError::ContextWindowExceeded { .. } => error_body(
            StatusCode::BAD_REQUEST,
            error.to_string(),
            "context_window_exceeded",
        ),
        LlmClientError::UpstreamHttp { status, body } => error_body(
            StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY),
            body,
            "upstream_error",
        ),
        LlmClientError::Transport(message) => {
            error_body(StatusCode::BAD_GATEWAY, message, "upstream_error")
        }
        LlmClientError::Stream(message) => server_error(message),
    }
}

async fn models() -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": SERVED_MODEL,
            "object": "model",
            "created": 0,
            "owned_by": "switchyard",
            "capabilities": {
                "streaming": true,
                "supported_inbound_formats": [
                    "openai-chat-completions",
                    "openai-responses",
                    "anthropic-messages",
                ],
            },
        }],
    }))
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

async fn not_found() -> Response {
    error_body(StatusCode::NOT_FOUND, "Not Found".to_string(), "not_found")
}

fn invalid_body(message: impl Into<String>) -> Response {
    error_body(StatusCode::BAD_REQUEST, message.into(), "invalid_body")
}

fn bad_request(message: String) -> Response {
    error_body(StatusCode::BAD_REQUEST, message, "invalid_request_error")
}

fn server_error(message: String) -> Response {
    error_body(StatusCode::INTERNAL_SERVER_ERROR, message, "server_error")
}

fn error_body(status: StatusCode, message: String, code: &str) -> Response {
    (
        status,
        Json(json!({
            "error": {"message": message, "type": code, "code": code}
        })),
    )
        .into_response()
}
