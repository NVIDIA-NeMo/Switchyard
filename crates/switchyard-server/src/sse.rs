// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SSE framing helpers for OpenAI, Anthropic, and Responses endpoints.

use axum::response::sse::{Event, Sse};
use futures_util::Stream;
use serde_json::{json, Value};
use switchyard_core::{Result, SwitchyardError};
use switchyard_translation::WireFormat;

/// Boxed stream type accepted by Axum's SSE response wrapper.
pub(crate) type SseFrameStream =
    std::pin::Pin<Box<dyn Stream<Item = std::result::Result<Event, SwitchyardError>> + Send>>;

/// Converts translated JSON events into endpoint-specific SSE frames.
pub(crate) fn frame_stream(
    stream: impl Stream<Item = Result<Value>> + Send + 'static,
    target_format: WireFormat,
) -> Sse<SseFrameStream> {
    let framed = async_stream::stream! {
        let mut stream = Box::pin(stream);
        while let Some(item) = futures_util::StreamExt::next(&mut stream).await {
            match item {
                Ok(value) => yield frame_event(target_format, value),
                Err(error) => {
                    tracing::warn!(error = %error, "stream iteration failed");
                    yield Ok(error_event(target_format, error.to_string()));
                    return;
                }
            }
        }

        if target_format == WireFormat::OpenAiChat {
            yield Ok(Event::default().data("[DONE]"));
        }
    };

    Sse::new(Box::pin(framed) as SseFrameStream)
}

fn frame_event(
    target_format: WireFormat,
    value: Value,
) -> std::result::Result<Event, SwitchyardError> {
    match target_format {
        // Gemini streams are unnamed `data:` frames like OpenAI Chat, but
        // without the terminal `[DONE]` marker (see `frame_stream`).
        WireFormat::OpenAiChat | WireFormat::GeminiGenerateContent => Event::default()
            .json_data(value)
            .map_err(|error| SwitchyardError::Other(error.to_string())),
        WireFormat::AnthropicMessages | WireFormat::OpenAiResponses => {
            let event_type = value
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("message")
                .to_string();
            Event::default()
                .event(event_type)
                .json_data(value)
                .map_err(|error| SwitchyardError::Other(error.to_string()))
        }
    }
}

fn error_event(target_format: WireFormat, message: String) -> Event {
    match target_format {
        WireFormat::OpenAiChat => Event::default().data(
            json!({
                "error": {
                    "message": message,
                    "type": "SwitchyardError",
                }
            })
            .to_string(),
        ),
        WireFormat::GeminiGenerateContent => Event::default().data(
            json!({
                "error": {
                    "code": 500,
                    "message": message,
                    "status": "INTERNAL",
                }
            })
            .to_string(),
        ),
        WireFormat::AnthropicMessages | WireFormat::OpenAiResponses => {
            Event::default().event("error").data(
                json!({
                    "type": "error",
                    "error": {
                        "message": message,
                        "type": "SwitchyardError",
                    }
                })
                .to_string(),
            )
        }
    }
}
