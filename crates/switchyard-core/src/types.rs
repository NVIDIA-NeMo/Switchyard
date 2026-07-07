// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-agnostic request and response wrappers used by core chains.

use std::fmt;
use std::pin::Pin;

use futures_core::Stream;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::error::{Result, SwitchyardError};

/// Supported inbound request wire formats.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum ChatRequestType {
    #[serde(rename = "openai_chat")]
    OpenAiChat,
    #[serde(rename = "openai_responses")]
    OpenAiResponses,
    #[serde(rename = "anthropic")]
    Anthropic,
    #[serde(rename = "gemini")]
    Gemini,
}

/// JSON request body wrapper shared by all request variants.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WireRequest {
    body: Value,
}

impl WireRequest {
    /// Wraps an arbitrary JSON request body.
    pub fn new(body: Value) -> Self {
        Self { body }
    }

    /// Returns the wrapped JSON request body.
    pub fn body(&self) -> &Value {
        &self.body
    }

    /// Returns a mutable reference to the wrapped JSON request body.
    pub fn body_mut(&mut self) -> &mut Value {
        &mut self.body
    }

    /// Consumes the wrapper and returns the JSON request body.
    pub fn into_body(self) -> Value {
        self.body
    }
}

/// Request body tagged by source API format.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "request_type", content = "request", rename_all = "snake_case")]
pub enum ChatRequest {
    #[serde(rename = "openai_chat")]
    OpenAiChat(WireRequest),
    #[serde(rename = "openai_responses")]
    OpenAiResponses(WireRequest),
    #[serde(rename = "anthropic")]
    Anthropic(WireRequest),
    #[serde(rename = "gemini")]
    Gemini(WireRequest),
}

impl ChatRequest {
    /// Creates an OpenAI Chat Completions request.
    pub fn openai_chat(body: Value) -> Self {
        Self::OpenAiChat(WireRequest::new(body))
    }

    /// Creates an OpenAI Responses API request.
    pub fn openai_responses(body: Value) -> Self {
        Self::OpenAiResponses(WireRequest::new(body))
    }

    /// Creates an Anthropic Messages request.
    pub fn anthropic(body: Value) -> Self {
        Self::Anthropic(WireRequest::new(body))
    }

    /// Creates a Gemini generateContent request.
    pub fn gemini(body: Value) -> Self {
        Self::Gemini(WireRequest::new(body))
    }

    /// Validates the request body before it enters the chain.
    ///
    /// Catches structurally valid but semantically invalid input so it
    /// fails fast with a 4xx instead of reaching the backend and surfacing
    /// as an opaque upstream 5xx. Currently rejects a present-but-empty
    /// `messages` array on the message-based formats (OpenAI Chat,
    /// Anthropic) and a present-but-empty `contents` array on Gemini.
    /// Absent or non-array conversation fields are left for the backend
    /// to interpret; the Responses format carries `input`, not `messages`,
    /// and is exempt.
    pub fn validate(&self) -> Result<()> {
        let checks_messages = matches!(self, Self::OpenAiChat(_) | Self::Anthropic(_));
        if checks_messages {
            if let Some(messages) = self.body().get("messages").and_then(Value::as_array) {
                if messages.is_empty() {
                    return Err(SwitchyardError::InvalidRequest(
                        "messages must be a non-empty array".to_string(),
                    ));
                }
            }
        }
        if matches!(self, Self::Gemini(_)) {
            if let Some(contents) = self.body().get("contents").and_then(Value::as_array) {
                if contents.is_empty() {
                    return Err(SwitchyardError::InvalidRequest(
                        "contents must be a non-empty array".to_string(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Returns the request's tagged wire format.
    pub fn request_type(&self) -> ChatRequestType {
        match self {
            Self::OpenAiChat(_) => ChatRequestType::OpenAiChat,
            Self::OpenAiResponses(_) => ChatRequestType::OpenAiResponses,
            Self::Anthropic(_) => ChatRequestType::Anthropic,
            Self::Gemini(_) => ChatRequestType::Gemini,
        }
    }

    /// Returns the request body regardless of source format.
    pub fn body(&self) -> &Value {
        match self {
            Self::OpenAiChat(request)
            | Self::OpenAiResponses(request)
            | Self::Anthropic(request)
            | Self::Gemini(request) => request.body(),
        }
    }

    /// Returns the mutable request body regardless of source format.
    pub fn body_mut(&mut self) -> &mut Value {
        match self {
            Self::OpenAiChat(request)
            | Self::OpenAiResponses(request)
            | Self::Anthropic(request)
            | Self::Gemini(request) => request.body_mut(),
        }
    }

    /// Reads the request's `model` field when present and string-valued.
    pub fn model(&self) -> Option<&str> {
        self.body().get("model").and_then(Value::as_str)
    }

    /// Writes the request's `model` field, creating an object body if necessary.
    pub fn set_model(&mut self, model: impl Into<String>) {
        match self.body_mut() {
            Value::Object(body) => {
                body.insert("model".to_string(), Value::String(model.into()));
            }
            body => {
                let mut object = Map::new();
                object.insert("model".to_string(), Value::String(model.into()));
                *body = Value::Object(object);
            }
        }
    }
}

/// Supported backend response wire shapes.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum ChatResponseType {
    #[serde(rename = "openai_completion")]
    OpenAiCompletion,
    #[serde(rename = "openai_stream")]
    OpenAiStream,
    #[serde(rename = "openai_responses_completion")]
    OpenAiResponsesCompletion,
    #[serde(rename = "openai_responses_stream")]
    OpenAiResponsesStream,
    #[serde(rename = "anthropic_completion")]
    AnthropicCompletion,
    #[serde(rename = "anthropic_stream")]
    AnthropicStream,
    #[serde(rename = "gemini_completion")]
    GeminiCompletion,
    #[serde(rename = "gemini_stream")]
    GeminiStream,
}

/// JSON response body wrapper shared by buffered response variants.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WireResponse {
    body: Value,
}

impl WireResponse {
    /// Wraps an arbitrary JSON response body.
    pub fn new(body: Value) -> Self {
        Self { body }
    }

    /// Returns the wrapped JSON response body.
    pub fn body(&self) -> &Value {
        &self.body
    }

    /// Consumes the wrapper and returns the JSON response body.
    pub fn into_body(self) -> Value {
        self.body
    }
}

/// Streaming event payload carried by streaming response variants.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "data", rename_all = "snake_case")]
pub enum StreamEvent {
    Json(Value),
    Text(String),
}

/// Boxed async stream for backend response events.
pub type BoxResponseStream = Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>;

/// Backend response tagged by provider format and buffered/streaming shape.
pub enum ChatResponse {
    OpenAiCompletion(WireResponse),
    OpenAiStream(BoxResponseStream),
    OpenAiResponsesCompletion(WireResponse),
    OpenAiResponsesStream(BoxResponseStream),
    AnthropicCompletion(WireResponse),
    AnthropicStream(BoxResponseStream),
    GeminiCompletion(WireResponse),
    GeminiStream(BoxResponseStream),
}

impl ChatResponse {
    /// Creates a buffered OpenAI Chat Completions response.
    pub fn openai_completion(body: Value) -> Self {
        Self::OpenAiCompletion(WireResponse::new(body))
    }

    /// Creates a buffered OpenAI Responses API response.
    pub fn openai_responses_completion(body: Value) -> Self {
        Self::OpenAiResponsesCompletion(WireResponse::new(body))
    }

    /// Creates a buffered Anthropic Messages response.
    pub fn anthropic_completion(body: Value) -> Self {
        Self::AnthropicCompletion(WireResponse::new(body))
    }

    /// Creates a buffered Gemini generateContent response.
    pub fn gemini_completion(body: Value) -> Self {
        Self::GeminiCompletion(WireResponse::new(body))
    }

    /// Returns the response's tagged wire shape.
    pub fn response_type(&self) -> ChatResponseType {
        match self {
            Self::OpenAiCompletion(_) => ChatResponseType::OpenAiCompletion,
            Self::OpenAiStream(_) => ChatResponseType::OpenAiStream,
            Self::OpenAiResponsesCompletion(_) => ChatResponseType::OpenAiResponsesCompletion,
            Self::OpenAiResponsesStream(_) => ChatResponseType::OpenAiResponsesStream,
            Self::AnthropicCompletion(_) => ChatResponseType::AnthropicCompletion,
            Self::AnthropicStream(_) => ChatResponseType::AnthropicStream,
            Self::GeminiCompletion(_) => ChatResponseType::GeminiCompletion,
            Self::GeminiStream(_) => ChatResponseType::GeminiStream,
        }
    }

    /// Returns the JSON body for buffered responses and `None` for streams.
    pub fn body(&self) -> Option<&Value> {
        match self {
            Self::OpenAiCompletion(response)
            | Self::OpenAiResponsesCompletion(response)
            | Self::AnthropicCompletion(response)
            | Self::GeminiCompletion(response) => Some(response.body()),
            Self::OpenAiStream(_)
            | Self::OpenAiResponsesStream(_)
            | Self::AnthropicStream(_)
            | Self::GeminiStream(_) => None,
        }
    }
}

impl fmt::Debug for ChatResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OpenAiCompletion(response) => f
                .debug_tuple("OpenAiCompletion")
                .field(response.body())
                .finish(),
            Self::OpenAiStream(_) => f.debug_tuple("OpenAiStream").field(&"<stream>").finish(),
            Self::OpenAiResponsesCompletion(response) => f
                .debug_tuple("OpenAiResponsesCompletion")
                .field(response.body())
                .finish(),
            Self::OpenAiResponsesStream(_) => f
                .debug_tuple("OpenAiResponsesStream")
                .field(&"<stream>")
                .finish(),
            Self::AnthropicCompletion(response) => f
                .debug_tuple("AnthropicCompletion")
                .field(response.body())
                .finish(),
            Self::AnthropicStream(_) => {
                f.debug_tuple("AnthropicStream").field(&"<stream>").finish()
            }
            Self::GeminiCompletion(response) => f
                .debug_tuple("GeminiCompletion")
                .field(response.body())
                .finish(),
            Self::GeminiStream(_) => f.debug_tuple("GeminiStream").field(&"<stream>").finish(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn validate_rejects_empty_messages_for_openai_chat() {
        let request = ChatRequest::openai_chat(json!({"model": "m", "messages": []}));
        match request.validate() {
            Err(SwitchyardError::InvalidRequest(message)) => {
                assert!(message.contains("messages must be a non-empty array"));
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn validate_rejects_empty_messages_for_anthropic() {
        let request = ChatRequest::anthropic(json!({"model": "m", "messages": []}));
        match request.validate() {
            Err(SwitchyardError::InvalidRequest(_)) => {}
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn validate_accepts_non_empty_messages() {
        let request = ChatRequest::openai_chat(
            json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]}),
        );
        assert!(request.validate().is_ok());
    }

    #[test]
    fn validate_is_lenient_when_messages_absent() {
        // Absent or non-array `messages` is left for the backend to interpret.
        let request = ChatRequest::openai_chat(json!({"model": "m"}));
        assert!(request.validate().is_ok());
    }

    #[test]
    fn validate_exempts_responses_format() {
        // The Responses format carries `input`, not `messages`.
        let request = ChatRequest::openai_responses(json!({"model": "m", "input": "hi"}));
        assert!(request.validate().is_ok());
    }
}
