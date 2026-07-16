// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Convenience wrappers over the default [`TranslationEngine`] — decode a wire
//! request/response to the neutral IR, encode the IR back, and decode/encode a
//! streamed response — so callers can translate without threading an engine and
//! policy through every call.

use std::pin::Pin;
use std::sync::LazyLock;

use async_stream::try_stream;
use futures::{Stream, StreamExt};
use serde_json::Value;

use crate::sse;
use crate::{
    AggLlmResponse, LlmRequest, LlmResponseStream, Result, StreamCodecRegistry,
    StreamTranslationState, TranslationEngine, TranslationPolicy, WireFormat,
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
pub fn decode_buffered_response(body: &Value, wire_format: WireFormat) -> Result<AggLlmResponse> {
    Ok(DEFAULT_TRANSLATION_ENGINE
        .decode_response(wire_format, body, &DEFAULT_TRANSLATION_POLICY)?
        .response)
}

/// Encodes a buffered aggregate into `wire_format`'s JSON body, stamping
/// `requested_model` over the upstream id so the caller sees the model it asked for.
pub fn encode_buffered_response(
    agg: &AggLlmResponse,
    wire_format: WireFormat,
    requested_model: Option<&str>,
) -> Result<Value> {
    let mut body = DEFAULT_TRANSLATION_ENGINE
        .encode_response(wire_format, agg, &DEFAULT_TRANSLATION_POLICY)?
        .body;
    if let (Some(model), Value::Object(object)) = (requested_model, &mut body) {
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
/// `requested_model` is exposed as the response model (via the stream state's
/// `target_model`). The target stream codec is resolved once and reused per chunk;
/// terminal events (`message_stop` / `response.completed`) come from `finish`.
pub fn encode_stream(
    chunks: LlmResponseStream,
    target: WireFormat,
    requested_model: Option<String>,
) -> RawEventStream {
    // The target is always a built-in wire format, so this lookup cannot fail; a
    // failure surfaces as a single error item rather than a panic.
    let codec = match StreamCodecRegistry::with_builtins().codec(target) {
        Ok(codec) => codec,
        Err(error) => {
            let message = error.to_string();
            return Box::pin(futures::stream::once(async move {
                Err(Box::new(std::io::Error::other(message))
                    as Box<dyn std::error::Error + Send + Sync>)
            }));
        }
    };

    let events = try_stream! {
        let mut state = StreamTranslationState {
            target: Some(target.into()),
            target_model: requested_model,
            ..Default::default()
        };

        let mut chunks = chunks;
        while let Some(item) = chunks.next().await {
            let chunk = item?;
            for value in codec.encode_event(&mut state, chunk) {
                yield value;
            }
        }
        for value in codec.finish(&mut state) {
            yield value;
        }
    };

    Box::pin(events)
}

/// Decodes a byte stream of `source`-format SSE frames into neutral IR chunks.
///
/// Operates on raw bytes, not any HTTP client type: the caller adapts its
/// transport's body stream into `Stream<Item = Result<Vec<u8>, _>>`. Frames are
/// buffered across chunks (a partial frame waits for its boundary); the source
/// stream codec is resolved once and reused for every frame.
pub fn decode_stream<S>(bytes: S, source: WireFormat) -> LlmResponseStream
where
    S: Stream<Item = std::result::Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>>>
        + Send
        + 'static,
{
    let marker = sse::done_marker(source);
    // The source is always a built-in wire format, so this lookup cannot fail; a
    // failure still surfaces as a single error item rather than a panic.
    let codec = match StreamCodecRegistry::with_builtins().codec(source) {
        Ok(codec) => codec,
        Err(error) => {
            let boxed: Box<dyn std::error::Error + Send + Sync> = error.to_string().into();
            return Box::pin(futures::stream::once(async move { Err(boxed) }));
        }
    };
    Box::pin(try_stream! {
        let mut state = StreamTranslationState::default();
        let mut buffer = Vec::new();
        futures::pin_mut!(bytes);

        while let Some(chunk) = bytes.next().await {
            buffer.extend_from_slice(&chunk?);
            while let Some(frame) = sse::drain_next_sse_frame(&mut buffer)? {
                match sse::parse_json_sse_frame(&frame, marker)? {
                    sse::ParsedSseFrame::Json(value) => {
                        for event in codec.decode_event(&mut state, &value) {
                            yield event;
                        }
                    }
                    sse::ParsedSseFrame::Done => return,
                    sse::ParsedSseFrame::Empty => {}
                }
            }
        }

        // A non-standard upstream might omit the final blank line; parse a trailing
        // complete frame instead of losing its last chunk.
        if sse::has_non_whitespace_bytes(&buffer) {
            let frame = sse::decode_sse_frame(&buffer)?;
            if let sse::ParsedSseFrame::Json(value) = sse::parse_json_sse_frame(&frame, marker)? {
                for event in codec.decode_event(&mut state, &value) {
                    yield event;
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;
    use futures::{stream, StreamExt};
    use serde_json::{json, Value};
    use switchyard_protocol::{completion_text, LlmResponseChunk};

    use super::{
        decode_buffered_response, decode_request, decode_stream, encode_buffered_response,
        encode_request, encode_stream,
    };
    use crate::{LlmResponseStream, WireFormat};

    // A boxed stream item error, matching the streamed IR contract.
    type BoxError = Box<dyn std::error::Error + Send + Sync>;

    #[test]
    fn request_round_trips_through_openai_chat() {
        let body = json!({"model": "gpt", "messages": [{"role": "user", "content": "hi"}]});
        let request = decode_request(WireFormat::OpenAiChat, &body).unwrap();
        assert_eq!(request.model.as_deref(), Some("gpt"));

        let encoded = encode_request(&request, WireFormat::OpenAiChat).unwrap();
        assert_eq!(encoded["model"], "gpt");
        assert_eq!(encoded["messages"][0]["content"], "hi");
    }

    #[test]
    fn buffered_response_round_trips_and_restamps_model() {
        let body = json!({
            "id": "1",
            "model": "upstream",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hi there"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        });
        let agg = decode_buffered_response(&body, WireFormat::OpenAiChat).unwrap();
        assert_eq!(completion_text(&agg), "Hi there");

        // `requested_model` overrides the encoded model.
        let encoded =
            encode_buffered_response(&agg, WireFormat::OpenAiChat, Some("client-facing")).unwrap();
        assert_eq!(encoded["model"], "client-facing");
        assert_eq!(encoded["choices"][0]["message"]["content"], "Hi there");
    }

    #[test]
    fn encode_stream_emits_openai_chunks_whose_deltas_reassemble() {
        let chunks: LlmResponseStream = stream::iter(vec![
            Ok(LlmResponseChunk::TextDelta {
                index: 0,
                text: "Hello".to_string(),
            }),
            Ok(LlmResponseChunk::TextDelta {
                index: 0,
                text: " world".to_string(),
            }),
            Ok(LlmResponseChunk::MessageStop {
                reason: Some("stop".to_string()),
            }),
        ])
        .boxed();

        let events: Vec<Value> = block_on(
            encode_stream(chunks, WireFormat::OpenAiChat, Some("m".to_string()))
                .map(|item| item.unwrap())
                .collect(),
        );
        let content: String = events
            .iter()
            .filter_map(|event| event["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(content, "Hello world");
    }

    #[test]
    fn decode_stream_parses_sse_bytes_into_ir_chunks() {
        let sse = b"data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n\
             data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n\
             data: [DONE]\n\n"
            .to_vec();
        let bytes = stream::once(async move { Ok::<Vec<u8>, BoxError>(sse) });

        let chunks: Vec<LlmResponseChunk> = block_on(
            decode_stream(bytes, WireFormat::OpenAiChat)
                .map(|item| item.unwrap())
                .collect(),
        );
        let content: String = chunks
            .iter()
            .filter_map(|chunk| match chunk {
                LlmResponseChunk::TextDelta { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(content, "Hello world");
    }
}
