// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]

//! Shared intake fixtures used by payload and processor tests.

use async_trait::async_trait;
use futures_util::StreamExt;
use parking_lot::Mutex;
use serde_json::{json, Value};
use switchyard_components::{
    BackendSelection, BackendSelectionReason, IntakeRequestMetadata, IntakeRequestState,
    IntakeSink, RequestMetadata,
};
use switchyard_core::{
    ChatRequest, ChatRequestType, ChatResponse, ModelId, ProxyContext, Result, StreamEvent,
    SwitchyardError,
};

/// In-memory sink that records payloads and shutdown calls.
#[derive(Default)]
pub struct RecordingSink {
    /// Payloads accepted by the fake sink.
    payloads: Mutex<Vec<Value>>,
    /// Optional one-shot error returned by `enqueue`.
    error: Mutex<Option<SwitchyardError>>,
    /// Number of times shutdown was invoked.
    shutdowns: Mutex<u64>,
}

impl RecordingSink {
    /// Creates a sink whose first enqueue returns an error.
    pub fn with_error(error: SwitchyardError) -> Self {
        Self {
            payloads: Mutex::new(Vec::new()),
            error: Mutex::new(Some(error)),
            shutdowns: Mutex::new(0),
        }
    }

    /// Returns the payloads captured so far.
    pub fn payloads(&self) -> Result<Vec<Value>> {
        Ok(self.payloads.lock().clone())
    }

    /// Returns the shutdown count captured so far.
    pub fn shutdowns(&self) -> Result<u64> {
        Ok(*self.shutdowns.lock())
    }
}

#[async_trait]
impl IntakeSink for RecordingSink {
    async fn enqueue(&self, payload: Value) -> Result<()> {
        if let Some(error) = self.error.lock().take() {
            return Err(error);
        }
        self.payloads.lock().push(payload);
        Ok(())
    }

    async fn shutdown(&self) -> Result<()> {
        let mut shutdowns = self.shutdowns.lock();
        *shutdowns = shutdowns.saturating_add(1);
        Ok(())
    }
}

/// Builds the default OpenAI Chat request used by intake tests.
pub fn request() -> ChatRequest {
    ChatRequest::openai_chat(json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}]
    }))
}

/// Builds an OpenAI Chat request with a caller-selected model.
pub fn openai_chat_request(model: &str) -> ChatRequest {
    ChatRequest::openai_chat(json!({
        "model": model,
        "messages": [{"role": "user", "content": "hi"}]
    }))
}

/// Builds a standard OpenAI completion with usage.
pub fn completion(id: &str, content: &str) -> ChatResponse {
    completion_with_usage(
        id,
        "gpt-4o",
        content,
        Some(json!({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})),
    )
}

/// Builds a standard OpenAI completion with caller-selected usage.
pub fn completion_with_usage(
    id: &str,
    model: &str,
    content: &str,
    usage: Option<Value>,
) -> ChatResponse {
    let mut body = json!({
        "id": id,
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }]
    });
    if let (Value::Object(object), Some(usage)) = (&mut body, usage) {
        object.insert("usage".to_string(), usage);
    }
    ChatResponse::openai_completion(body)
}

/// Builds a proxy context that has opted into intake capture.
pub fn opted_in_context() -> ProxyContext {
    let mut ctx = ProxyContext::new();
    record_backend_selection(&mut ctx, ModelId::from_static("gpt-4o"));
    ctx.insert(RequestMetadata {
        session_id: Some("session-123".to_string()),
        intake: IntakeRequestMetadata {
            enabled: Some(true),
            app: Some("codex".to_string()),
            task: Some("developer-session".to_string()),
        },
    });
    ctx.insert(IntakeRequestState {
        started_at_ms: 1_700_000_000_000,
        inbound_format: ChatRequestType::OpenAiChat,
        session_id: Some("session-123".to_string()),
        skip: false,
        request_snapshot: Some(request()),
    });
    ctx
}

/// Records a backend selection in the typed proxy context.
pub fn record_backend_selection(ctx: &mut ProxyContext, model: ModelId) {
    let _ = ctx.insert(BackendSelection::for_model(
        model,
        None,
        BackendSelectionReason::PassthroughModel,
    ));
}

/// Drains a streaming response into a vector of stream events.
pub async fn drain_stream(response: ChatResponse) -> Result<Vec<StreamEvent>> {
    let mut stream = match response {
        ChatResponse::OpenAiStream(stream)
        | ChatResponse::OpenAiResponsesStream(stream)
        | ChatResponse::AnthropicStream(stream) => stream,
        other => {
            return Err(SwitchyardError::Other(format!(
                "expected stream, got {:?}",
                other.response_type()
            )));
        }
    };
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event?);
    }
    Ok(events)
}
