// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic OpenAI and Anthropic HTTP endpoints for integration tests.

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::io;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;

use axum::extract::{OriginalUri, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use futures_util::stream::{self, StreamExt};
use rand::seq::IndexedRandom;
use serde_json::{json, Value};
use switchyard_protocol::{
    AggLlmResponse, ContentBlock, LlmResponseChunk, LlmResponseStream, ResponseOutput, Role,
    StopReason, Usage,
};
use switchyard_translation::{encode_aggregated_response, encode_stream, WireFormat};
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

/// Representative assistant text used when a test does not configure a response.
pub const DEFAULT_RESPONSE_BANK: &[&str] = &[
    "Done.",
    "The request completed successfully.",
    "Short answer: 42.",
    "Here is the result:\n\n- first item\n- second item",
    "```json\n{\"status\":\"ok\",\"items\":[]}\n```",
    "```python\nprint(\"hello from the mock server\")\n```",
    "I checked the available context and found no additional action is required.",
    "The operation succeeded. Review the returned metadata for details.",
];

/// A request received by the mock server.
#[derive(Clone, Debug, PartialEq)]
pub struct CapturedRequest {
    /// Provider endpoint path.
    pub path: String,
    /// UTF-8 request headers keyed by lowercase name.
    pub headers: BTreeMap<String, String>,
    /// Parsed JSON request body.
    pub body: Value,
}

/// Configures deterministic responses before starting a mock server.
#[derive(Clone, Debug)]
pub struct MockLlmServerBuilder {
    bind_addr: SocketAddr,
    response_bank: Vec<String>,
    model_responses: BTreeMap<String, String>,
    model_errors: BTreeMap<String, StatusCode>,
}

impl Default for MockLlmServerBuilder {
    fn default() -> Self {
        Self {
            bind_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0),
            response_bank: DEFAULT_RESPONSE_BANK
                .iter()
                .map(|response| (*response).to_string())
                .collect(),
            model_responses: BTreeMap::new(),
            model_errors: BTreeMap::new(),
        }
    }
}

impl MockLlmServerBuilder {
    /// Binds the server to `addr`; the default uses an ephemeral localhost port.
    pub fn bind_addr(mut self, addr: SocketAddr) -> Self {
        self.bind_addr = addr;
        self
    }

    /// Sets the response text used when a model has no specific response.
    pub fn default_response(mut self, text: impl Into<String>) -> Self {
        self.response_bank = vec![text.into()];
        self
    }

    /// Sets the response text returned for `model`.
    pub fn model_response(mut self, model: impl Into<String>, text: impl Into<String>) -> Self {
        self.model_responses.insert(model.into(), text.into());
        self
    }

    /// Makes requests for `model` return `status` and a standard error body.
    pub fn model_error(mut self, model: impl Into<String>, status: StatusCode) -> Self {
        self.model_errors.insert(model.into(), status);
        self
    }

    /// Starts the configured server.
    pub async fn start(self) -> io::Result<MockLlmServer> {
        let listener = TcpListener::bind(self.bind_addr).await?;
        let addr = listener.local_addr()?;
        let state = Arc::new(ServerState {
            response_bank: self.response_bank,
            model_responses: self.model_responses,
            model_errors: self.model_errors,
            requests: Mutex::new(Vec::new()),
        });
        let server_state = Arc::clone(&state);
        let task = tokio::spawn(async move {
            if let Err(error) = axum::serve(listener, router(server_state)).await {
                tracing::error!(error = %error, "test LLM server stopped");
            }
        });
        Ok(MockLlmServer { addr, state, task })
    }
}

/// A running mock LLM server that stops when dropped.
pub struct MockLlmServer {
    addr: SocketAddr,
    state: Arc<ServerState>,
    task: JoinHandle<()>,
}

impl MockLlmServer {
    /// Returns a builder for custom behavior.
    pub fn builder() -> MockLlmServerBuilder {
        MockLlmServerBuilder::default()
    }

    /// Starts a server with the default response behavior.
    pub async fn start() -> io::Result<Self> {
        Self::builder().start().await
    }

    /// Returns the server root URL.
    pub fn url(&self) -> String {
        format!("http://{}", self.addr)
    }

    /// Returns the `/v1` URL expected by Switchyard LLM client configuration.
    pub fn base_url(&self) -> String {
        format!("{}/v1", self.url())
    }

    /// Returns all requests captured so far.
    pub async fn requests(&self) -> Vec<CapturedRequest> {
        self.state.requests.lock().await.clone()
    }
}

impl Drop for MockLlmServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

struct ServerState {
    response_bank: Vec<String>,
    model_responses: BTreeMap<String, String>,
    model_errors: BTreeMap<String, StatusCode>,
    requests: Mutex<Vec<CapturedRequest>>,
}

fn router(state: Arc<ServerState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(openai_chat))
        .route("/v1/responses", post(openai_responses))
        .route("/v1/messages", post(anthropic_messages))
        .with_state(state)
}

async fn openai_chat(
    state: State<Arc<ServerState>>,
    uri: OriginalUri,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    handle_request(
        state.0,
        uri.0.path().to_string(),
        headers,
        body,
        WireFormat::OpenAiChat,
    )
    .await
}

async fn openai_responses(
    state: State<Arc<ServerState>>,
    uri: OriginalUri,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    handle_request(
        state.0,
        uri.0.path().to_string(),
        headers,
        body,
        WireFormat::OpenAiResponses,
    )
    .await
}

async fn anthropic_messages(
    state: State<Arc<ServerState>>,
    uri: OriginalUri,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    handle_request(
        state.0,
        uri.0.path().to_string(),
        headers,
        body,
        WireFormat::AnthropicMessages,
    )
    .await
}

async fn handle_request(
    state: Arc<ServerState>,
    path: String,
    headers: HeaderMap,
    body: Value,
    format: WireFormat,
) -> Response {
    state.requests.lock().await.push(CapturedRequest {
        path,
        headers: capture_headers(&headers),
        body: body.clone(),
    });

    let model = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("mock-model");
    if let Some(status) = state.model_errors.get(model).copied() {
        return mock_error(status);
    }
    if let Some(status) = requested_error_status(&headers) {
        return mock_error(status);
    }

    let text = state
        .model_responses
        .get(model)
        .map(String::as_str)
        .or_else(|| {
            state
                .response_bank
                .choose(&mut rand::rng())
                .map(String::as_str)
        })
        .unwrap_or("ok");
    if body.get("stream").and_then(Value::as_bool) == Some(true) {
        stream_response(format, model.to_string(), text.to_string())
    } else {
        aggregate_response(format, model, text)
    }
}

fn capture_headers(headers: &HeaderMap) -> BTreeMap<String, String> {
    headers
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.as_str().to_string(), value.to_string()))
        })
        .collect()
}

fn requested_error_status(headers: &HeaderMap) -> Option<StatusCode> {
    headers
        .get("x-switchyard-test-status")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u16>().ok())
        .and_then(|value| StatusCode::from_u16(value).ok())
}

fn mock_error(status: StatusCode) -> Response {
    (
        status,
        Json(json!({
            "error": {
                "message": format!("mock upstream returned {}", status.as_u16()),
                "type": "mock_error"
            }
        })),
    )
        .into_response()
}

fn aggregate_response(format: WireFormat, model: &str, text: &str) -> Response {
    let response = mock_response(model, text);
    match encode_aggregated_response(&response, format, Some(model)) {
        Ok(body) => Json(body).into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": {"message": error.to_string()}})),
        )
            .into_response(),
    }
}

fn mock_response(model: &str, text: &str) -> AggLlmResponse {
    AggLlmResponse {
        id: Some("mock-response".to_string()),
        model: Some(model.to_string()),
        outputs: vec![ResponseOutput {
            role: Role::Assistant,
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
            stop_reason: Some(StopReason::EndTurn),
        }],
        usage: mock_usage(),
        ..AggLlmResponse::default()
    }
}

fn mock_usage() -> Usage {
    Usage {
        input_tokens: Some(3),
        cache: Usage::cache_details(Some(7), None),
        output_tokens: Some(2),
        total_tokens: Some(12),
        reasoning_tokens: None,
    }
}

fn stream_response(format: WireFormat, model: String, text: String) -> Response {
    let chunks: LlmResponseStream = Box::pin(stream::iter(vec![
        Ok(LlmResponseChunk::MessageStart {
            id: Some("mock-response".to_string()),
            model: Some(model.clone()),
        }),
        Ok(LlmResponseChunk::TextDelta { index: 0, text }),
        Ok(LlmResponseChunk::Usage(mock_usage())),
        Ok(LlmResponseChunk::MessageStop {
            reason: Some("stop".to_string()),
        }),
    ]));
    let events = match encode_stream(chunks, format, Some(model)) {
        Ok(events) => events,
        Err(error) => return mock_stream_error(error.to_string()),
    };
    let framed = events.map(move |item| {
        Ok::<Event, Infallible>(match item {
            Ok(value) => frame_event(format, value),
            Err(error) => Event::default().event("error").data(error.to_string()),
        })
    });
    let terminal = match format {
        WireFormat::OpenAiChat | WireFormat::OpenAiResponses => {
            Some(Ok::<Event, Infallible>(Event::default().data("[DONE]")))
        }
        WireFormat::AnthropicMessages => None,
    };
    Sse::new(framed.chain(stream::iter(terminal))).into_response()
}

fn frame_event(format: WireFormat, value: Value) -> Event {
    match format {
        WireFormat::OpenAiChat => Event::default().data(value.to_string()),
        WireFormat::OpenAiResponses | WireFormat::AnthropicMessages => {
            let event_type = value
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("message");
            Event::default().event(event_type).data(value.to_string())
        }
    }
}

fn mock_stream_error(message: String) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({"error": {"message": message}})),
    )
        .into_response()
}
