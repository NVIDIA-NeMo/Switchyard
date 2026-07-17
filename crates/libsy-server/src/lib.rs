// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal axum server exposing the LLM wire APIs over a libsy [`Algorithm`].
//!
//! Each inbound request is decoded to Switchyard's neutral IR and run through the
//! routing [`Algorithm`] (a [`RandomAlgo`](libsy::RandomAlgo) over weak/strong
//! targets). The algorithm picks a target and serves the call through the
//! target's client, which dispatches to the upstream backend; the neutral
//! response is then encoded back into the inbound wire format — buffered JSON or
//! a stream of wire events. This crate is the HTTP surface around that: routing,
//! header normalization, SSE framing, and error mapping. It mirrors
//! `switchyard-server` but swaps the profile-chain executor for a libsy
//! algorithm. See [`build_router`].

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
use libsy::affinity::metadata_from_headers;
use libsy::Algorithm;
use serde_json::{json, Value};
use switchyard_llm_client::LlmClientError;
use switchyard_protocol::{Context, LlmResponse, Request};
use switchyard_translation::{decode_request, encode_buffered_response, encode_stream, WireFormat};

use crate::sse::frame_stream;

/// The single model name this server advertises to clients.
pub const SERVED_MODEL: &str = "switchyard";

/// Shared state for all endpoint handlers.
#[derive(Clone)]
pub struct ProxyState {
    /// The routing algorithm run once per request; it picks a target and serves
    /// the call through the target's client.
    algorithm: Arc<dyn Algorithm>,
    /// When set, log each request's routing decision (the selected tier) to stderr.
    log_routing: bool,
}

impl ProxyState {
    /// Builds server state around a routing algorithm.
    ///
    /// Each request is decoded to the neutral IR and handed to `algorithm`, whose
    /// chosen target resolves its own upstream model id. The response is always
    /// restamped with the model the caller asked for, never the upstream id. With
    /// `log_routing`, each request's chosen tier is logged to stderr.
    pub fn new(algorithm: Arc<dyn Algorithm>, log_routing: bool) -> Self {
        Self {
            algorithm,
            log_routing,
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

// Decodes the inbound body to the neutral IR, runs the routing algorithm, then
// encodes its response back into `inbound`. The chosen target owns the upstream
// call; this frames the neutral result and restamps the caller's model.
async fn handle(
    state: ProxyState,
    headers: HeaderMap,
    body: Value,
    inbound: WireFormat,
) -> Response {
    let llm_request = match decode_request(inbound, &body) {
        Ok(request) => request,
        Err(error) => return invalid_body(format!("Request body could not be decoded: {error}")),
    };
    // The model the caller asked for; restamped onto the response so it never
    // leaks the upstream id the algorithm's target resolved.
    let requested_model = llm_request.model.clone();

    // Normalize harness headers into routing metadata (session/agent/task ids and
    // sub-agent context) so an affinity policy can key on them. The forwarded
    // headers ride along too; the client drops reserved/auth headers and injects
    // the backend's real credential, so the sentinel key a coding agent sends is
    // filtered automatically. `wire_format` stays unset so the upstream call uses
    // the chosen tier's own format — the response is translated back to `inbound`.
    let mut metadata = metadata_from_headers(&multi_headers(&headers));
    let session = metadata.session_id.clone().unwrap_or("none".to_string());
    let subagent_id = metadata
        .agent_context
        .as_deref()
        .is_some_and(|context| context.is_subagent)
        .then(|| metadata.agent_id.clone())
        .flatten();
    metadata.http_headers = Some(normalized_headers(&headers));
    let request = Request {
        llm_request,
        raw_request: Some(body),
        metadata: Some(metadata),
    };

    let (trace, response) = match state
        .algorithm
        .clone()
        .run(Context::default(), request)
        .await
    {
        Ok(result) => result,
        Err(error) => return map_run_error(error),
    };

    // The trace's first decision is the routing choice; log which tier served.
    if state.log_routing {
        if let Some(decision) = trace.first() {
            let actor = actor_label(subagent_id.as_deref());
            eprintln!(
                "[route][session={session}][actor={actor}] inbound={inbound} -> {}",
                decision.selected_model()
            );
        }
    }

    match response.llm_response {
        LlmResponse::Agg(agg) => {
            match encode_buffered_response(&agg, inbound, requested_model.as_deref()) {
                Ok(body) => Json(body).into_response(),
                Err(error) => error_body(
                    StatusCode::BAD_REQUEST,
                    error.to_string(),
                    "invalid_request_error",
                ),
            }
        }
        LlmResponse::Stream(chunks) => {
            frame_stream(encode_stream(chunks, inbound, requested_model), inbound).into_response()
        }
    }
}

// Labels the request source for the demo's routing log. Claude Code only sends a
// child-agent id for sub-agent requests, so every other request is the root path.
fn actor_label(subagent_id: Option<&str>) -> String {
    subagent_id
        .map(|id| format!("subagent:{id}"))
        .unwrap_or_else(|| "root".to_string())
}

// Maps an algorithm run failure onto an HTTP envelope. A failed upstream model
// call surfaces as a boxed [`LlmClientError`] (mapped like the direct path);
// anything else is an internal proxy fault.
fn map_run_error(error: Box<dyn std::error::Error + Send + Sync>) -> Response {
    match error.downcast::<LlmClientError>() {
        Ok(client_error) => map_client_error(*client_error),
        Err(other) => server_error(other.to_string()),
    }
}

// Collects every UTF-8 header value per lowercased name — the multi-valued shape
// `metadata_from_headers` expects for harness metadata normalization.
fn multi_headers(headers: &HeaderMap) -> BTreeMap<String, Vec<String>> {
    let mut collected: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (name, value) in headers {
        if let Ok(value) = value.to_str() {
            collected
                .entry(name.as_str().to_ascii_lowercase())
                .or_default()
                .push(value.to_string());
        }
    }
    collected
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
