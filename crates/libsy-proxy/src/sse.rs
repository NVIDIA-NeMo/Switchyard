// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SSE framing for the OpenAI Chat, Anthropic Messages, and Responses formats.
//!
//! Ported from `switchyard-server/src/sse.rs`; that copy is crate-private and
//! typed against `SwitchyardError`, so this re-types the same framing against a
//! boxed error. Contract: OpenAI Chat emits `data: {json}` frames plus a
//! terminal `data: [DONE]`; Anthropic and Responses emit named
//! `event: <type>\ndata: {json}` frames with no `[DONE]` (the stream ends on
//! `message_stop` / `response.completed`).

use std::pin::Pin;

use axum::response::sse::{Event, Sse};
use futures_util::{Stream, StreamExt};
use serde_json::{json, Value};
use switchyard_translation::WireFormat;

/// Boxed, thread-safe error carried by a framed SSE item.
pub(crate) type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Boxed stream type accepted by Axum's SSE response wrapper.
pub(crate) type SseFrameStream = Pin<Box<dyn Stream<Item = Result<Event, BoxError>> + Send>>;

/// Frames a stream of target-format wire events as endpoint-specific SSE.
pub(crate) fn frame_stream(
    stream: impl Stream<Item = Result<Value, BoxError>> + Send + 'static,
    target_format: WireFormat,
) -> Sse<SseFrameStream> {
    let framed = async_stream::stream! {
        let mut stream = Box::pin(stream);
        while let Some(item) = stream.next().await {
            match item {
                Ok(value) => yield frame_event(target_format, value),
                Err(error) => {
                    // The 200 + headers are already committed, so a mid-stream
                    // failure surfaces as a final format-specific error frame.
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

// Frames one wire event `Value` for the target format.
fn frame_event(target_format: WireFormat, value: Value) -> Result<Event, BoxError> {
    match target_format {
        WireFormat::OpenAiChat => Event::default()
            .json_data(value)
            .map_err(|error| Box::new(error) as BoxError),
        WireFormat::AnthropicMessages | WireFormat::OpenAiResponses => {
            let event_type = value
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("message")
                .to_string();
            Event::default()
                .event(event_type)
                .json_data(value)
                .map_err(|error| Box::new(error) as BoxError)
        }
    }
}

// Builds a terminal error frame in the target format's shape.
fn error_event(target_format: WireFormat, message: String) -> Event {
    match target_format {
        WireFormat::OpenAiChat => Event::default()
            .data(json!({"error": {"message": message, "type": "proxy_error"}}).to_string()),
        WireFormat::AnthropicMessages | WireFormat::OpenAiResponses => {
            Event::default().event("error").data(
                json!({"type": "error", "error": {"message": message, "type": "proxy_error"}})
                    .to_string(),
            )
        }
    }
}
