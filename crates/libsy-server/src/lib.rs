// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal axum server exposing the LLM wire APIs over [`TranslatingLlmClient`].
//!
//! Each inbound request is handed to the client's raw path
//! ([`TranslatingLlmClient::call_rewrite_model_raw`]), which decodes it to
//! Switchyard's neutral IR, dispatches to the model's upstream backend, and
//! encodes the response back into the inbound wire format — buffered JSON or a
//! stream of wire events. This crate is the HTTP surface around that: routing,
//! header normalization, SSE framing, and error mapping. It mirrors
//! `switchyard-server` but swaps the profile-chain executor for a single client.
//! See [`build_router`].

mod sse;

pub mod cli;

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::extract::rejection::JsonRejection;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use switchyard_llm_client::{LlmClientError, RawResponse, TranslatingLlmClient};
use switchyard_protocol::Context;
use switchyard_translation::WireFormat;

use crate::sse::frame_stream;

/// The single model name this server advertises to clients.
pub const SERVED_MODEL: &str = "switchyard";

/// Shared state for all endpoint handlers.
#[derive(Clone)]
pub struct ProxyState {
    client: Arc<TranslatingLlmClient>,
    /// Model id sent upstream and used as the client's map key.
    upstream_model: Arc<str>,
}

impl ProxyState {
    /// Builds server state around a configured client.
    ///
    /// `upstream_model` is the model id the client resolves to its backend and
    /// sends upstream; the response is always restamped with the model the
    /// caller asked for (the served name), never this id.
    pub fn new(client: TranslatingLlmClient, upstream_model: impl Into<Arc<str>>) -> Self {
        Self {
            client: Arc::new(client),
            upstream_model: upstream_model.into(),
        }
    }
}

/// Builds the server router. Public so integration tests can drive it without a socket.
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

// The client's raw path owns decode → call upstream → encode-back-to-`inbound`;
// this only forwards headers, picks the upstream model, and frames the result.
async fn handle(
    state: ProxyState,
    headers: HeaderMap,
    body: Value,
    inbound: WireFormat,
) -> Response {
    // Caller headers ride along as metadata; the client drops reserved/auth
    // headers and injects the backend's real credential, so the sentinel key a
    // coding agent sends is filtered automatically.
    let http_headers = Some(normalized_headers(&headers));
    let response = state
        .client
        .call_rewrite_model_raw(
            Context::default(),
            body,
            http_headers,
            Some(&state.upstream_model),
            inbound,
        )
        .await;

    match response {
        Ok(RawResponse::Buffered(body)) => Json(body).into_response(),
        Ok(RawResponse::Stream(events)) => frame_stream(events, inbound).into_response(),
        Err(error) => map_client_error(error),
    }
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
        // The inbound wire format has no configured upstream backend — a client
        // problem (they hit an endpoint this proxy is not serving), not a fault.
        LlmClientError::UnknownModelFormat { format, .. } => error_body(
            StatusCode::BAD_REQUEST,
            format!("no upstream backend configured for {format} requests"),
            "unsupported_format",
        ),
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
