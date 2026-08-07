// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Convenience wrappers over the default [`TranslationEngine`] — decode a wire
//! request/response to the neutral IR, encode the IR back, and decode/encode a
//! streamed response — so callers can translate without threading an engine and
//! policy through every call.

use std::pin::Pin;
use std::sync::LazyLock;

use async_stream::try_stream;
use futures::io::AsyncBufReadExt;
use futures::{Stream, StreamExt, TryStreamExt};
use serde_json::Value;
use switchyard_protocol::LlmClientError;

use crate::codecs::stream::encode_response_stream_event;
use crate::sse;
use crate::{
    AggLlmResponse, FormatId, LlmRequest, LlmResponseStream, LlmResponseStreamEvent, Result,
    StreamCodecRegistry, StreamTranslationState, TranslationEngine, TranslationPolicy, WireFormat,
};

static DEFAULT_TRANSLATION_POLICY: LazyLock<TranslationPolicy> =
    LazyLock::new(TranslationPolicy::default);
static DEFAULT_TRANSLATION_ENGINE: LazyLock<TranslationEngine> =
    LazyLock::new(TranslationEngine::default);

/// Decodes a `wire_format` request body into the neutral IR.
pub fn decode_request(wire_format: WireFormat, body: &Value) -> Result<LlmRequest> {
    Ok(DEFAULT_TRANSLATION_ENGINE
        .decode_request(wire_format, body, &DEFAULT_TRANSLATION_POLICY)?
        .request)
}

/// Encodes a normalized request into `wire_format`'s JSON body.
pub fn encode_request(request: &LlmRequest, wire_format: WireFormat) -> Result<Value> {
    Ok(DEFAULT_TRANSLATION_ENGINE
        .encode_request(wire_format, request, &DEFAULT_TRANSLATION_POLICY)?
        .body)
}

/// Decodes a buffered `wire_format` response body into the neutral aggregate.
pub fn decode_aggregated_response(body: &Value, wire_format: WireFormat) -> Result<AggLlmResponse> {
    Ok(DEFAULT_TRANSLATION_ENGINE
        .decode_response(wire_format, body, &DEFAULT_TRANSLATION_POLICY)?
        .response)
}

/// Encodes a buffered aggregate into `wire_format`'s JSON body, stamping
/// `served_model` over the encoded id so the caller sees which model answered.
/// Passing `None` leaves the id the upstream reported.
pub fn encode_aggregated_response(
    agg: &AggLlmResponse,
    wire_format: WireFormat,
    served_model: Option<&str>,
) -> Result<Value> {
    let mut body = DEFAULT_TRANSLATION_ENGINE
        .encode_response(wire_format, agg, &DEFAULT_TRANSLATION_POLICY)?
        .body;
    if let (Some(model), Value::Object(object)) = (served_model, &mut body) {
        object.insert("model".to_string(), Value::String(model.to_string()));
    }
    Ok(body)
}

/// A stream of wire-format event objects in one format — the unframed body of an
/// SSE response. The serving layer frames each `Value` (e.g. as an SSE
/// `data:`/`event:` block).
pub type RawEventStream = Pin<
    Box<
        dyn Stream<Item = std::result::Result<Value, Box<dyn std::error::Error + Send + Sync>>>
            + Send,
    >,
>;

/// Encodes a stream of IR chunks into a stream of target-format wire events.
///
/// `served_model` is exposed as the response model (via the stream state's
/// `target_model`); `None` falls back to the id observed on the source stream.
/// The target stream codec is resolved once and reused per chunk; terminal events
/// (`message_stop` / `response.completed`) come from `finish`.
pub fn encode_stream(
    chunks: LlmResponseStream,
    target: WireFormat,
    served_model: Option<String>,
) -> std::result::Result<RawEventStream, LlmClientError> {
    let target_format: FormatId = target.into();
    // The target is always a built-in wire format, so this lookup cannot fail; a
    // failure returns as an `Err` rather than a panic.
    let codec = StreamCodecRegistry::with_builtins()
        .codec(target_format.clone())
        // Currently the only error is that the codec is missing, which is Configuration
        .map_err(|err| LlmClientError::Configuration {
            message: err.to_string(),
        })?;

    let served_model_for_events = served_model.clone();
    let mut state = StreamTranslationState {
        target: Some(target_format.clone()),
        target_model: served_model,
        ..Default::default()
    };
    let mut chunks = chunks;

    let events = try_stream! {
        while let Some(item) = chunks.next().await {
            let event = item?;
            for mut value in
                encode_response_stream_event(&mut state, codec.as_ref(), &target_format, event)
            {
                stamp_streamed_response_model(
                    &mut value,
                    target,
                    served_model_for_events.as_deref(),
                );
                yield value;
            }
            if state.errored {
                return;
            }
        }
        for mut value in codec.finish(&mut state) {
            stamp_streamed_response_model(
                &mut value,
                target,
                served_model_for_events.as_deref(),
            );
            yield value;
        }
    };

    Ok(Box::pin(events))
}

// The raw-response helper promises that the caller sees the model that served the
// request. Same-format preservation bypasses provider codecs, so apply that
// helper-specific override after replay without disturbing any other raw fields.
fn stamp_streamed_response_model(
    event: &mut Value,
    target: WireFormat,
    served_model: Option<&str>,
) {
    let Some(served_model) = served_model else {
        return;
    };

    match target {
        WireFormat::OpenAiChat => {
            if let Some(event) = event.as_object_mut() {
                event.insert("model".to_string(), Value::String(served_model.to_string()));
            }
        }
        WireFormat::OpenAiResponses => {
            if let Some(response) = event.get_mut("response").and_then(Value::as_object_mut) {
                response.insert("model".to_string(), Value::String(served_model.to_string()));
            }
        }
        WireFormat::AnthropicMessages => {
            if let Some(message) = event.get_mut("message").and_then(Value::as_object_mut) {
                message.insert("model".to_string(), Value::String(served_model.to_string()));
            }
        }
    }
}

/// Decodes a byte stream of `source`-format SSE frames into neutral IR chunks.
///
/// Operates on raw bytes, not any HTTP client type: the caller adapts its
/// transport's body stream into `Stream<Item = Result<Vec<u8>, _>>`. Frames are
/// buffered across chunks (a partial frame waits for its boundary); the source
/// stream codec is resolved once and reused for every frame.
pub fn decode_stream<S>(
    bytes: S,
    source: WireFormat,
) -> std::result::Result<LlmResponseStream, LlmClientError>
where
    S: Stream<Item = std::result::Result<Vec<u8>, LlmClientError>> + Send + 'static,
{
    let marker = sse::done_marker(source);
    let source_format: FormatId = source.into();
    // The source is always a built-in wire format, so this lookup cannot fail; a
    // failure returns as an `Err` rather than a panic.
    let codec = StreamCodecRegistry::with_builtins()
        .codec(source_format.clone())
        .map_err(|error| LlmClientError::ResponseTranslation(error.to_string()))?;
    // Adapt the byte-chunk stream into an async line reader. The BufReader
    // reassembles data split across network chunks (including multi-byte UTF-8),
    // and `lines()` yields one SSE field line at a time. The stream is boxed to
    // an `io::Error` item so `into_async_read`'s error bound resolves cleanly. The
    // source error is boxed intact rather than stringified, so
    // `llm_client_error_from_io` can recover its original variant on the way out.
    let io_bytes: Pin<Box<dyn Stream<Item = std::io::Result<Vec<u8>>> + Send>> =
        Box::pin(bytes.map(|item| item.map_err(std::io::Error::other)));
    let lines = futures::io::BufReader::new(io_bytes.into_async_read()).lines();

    let mut state = StreamTranslationState {
        source: Some(source_format.clone()),
        ..StreamTranslationState::default()
    };
    let mut frame = String::new();
    let stream = Box::pin(try_stream! {
        futures::pin_mut!(lines);
        while let Some(line) = lines.next().await {
            let line = line.map_err(llm_client_error_from_io)?;
            // A blank line (allowing a bare CR for CRLF streams) ends the frame.
            if line.trim_end().is_empty() {
                let parsed = sse::parse_json_sse_frame(&frame, marker)
                    .map_err(|error| LlmClientError::ResponseTranslation(error.to_string()))?;
                frame.clear();
                match parsed {
                    sse::SseFrame::Empty => {}
                    sse::SseFrame::Done => break,
                    sse::SseFrame::Data(value) => {
                        let normalized = codec.decode_event(&mut state, &value);
                        yield LlmResponseStreamEvent::preserved(
                            source_format.clone(),
                            value,
                            normalized,
                        );
                    }
                }
            } else {
                frame.push_str(&line);
                frame.push('\n');
            }
        }

        // A non-standard upstream might omit the final blank line; parse a trailing
        // complete frame instead of losing its last chunk.
        #[allow(clippy::collapsible_if)]
        if !frame.trim_end().is_empty() {
            let parsed = sse::parse_json_sse_frame(&frame, marker)
                .map_err(|error| LlmClientError::ResponseTranslation(error.to_string()))?;
            if let sse::SseFrame::Data(value) = parsed {
                let normalized = codec.decode_event(&mut state, &value);
                yield LlmResponseStreamEvent::preserved(source_format, value, normalized);
            }
        }
    });
    Ok(stream)
}

// Recover transport errors wrapped for `AsyncRead`; other reader failures are
// invalid upstream responses.
fn llm_client_error_from_io(error: std::io::Error) -> LlmClientError {
    let kind = error.kind();
    let message = error.to_string();
    match error.into_inner() {
        Some(source) => match source.downcast::<LlmClientError>() {
            Ok(error) => *error,
            Err(source) => LlmClientError::InvalidResponse { source },
        },
        None => LlmClientError::InvalidResponse {
            source: Box::new(std::io::Error::new(kind, message)),
        },
    }
}

#[cfg(test)]
#[path = "helpers_tests.rs"]
mod tests;
