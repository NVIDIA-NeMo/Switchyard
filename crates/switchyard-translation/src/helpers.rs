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
pub fn decode_aggregated_response(body: &Value, wire_format: WireFormat) -> Result<AggLlmResponse> {
    Ok(DEFAULT_TRANSLATION_ENGINE
        .decode_response(wire_format, body, &DEFAULT_TRANSLATION_POLICY)?
        .response)
}

/// Encodes a buffered aggregate into `wire_format`'s JSON body, stamping
/// `requested_model` over the upstream id so the caller sees the model it asked for.
pub fn encode_aggregated_response(
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
) -> std::result::Result<RawEventStream, Box<dyn std::error::Error + Send + Sync>> {
    // The target is always a built-in wire format, so this lookup cannot fail; a
    // failure returns as an `Err` rather than a panic.
    let codec = StreamCodecRegistry::with_builtins().codec(target)?;

    let mut state = StreamTranslationState {
        target: Some(target.into()),
        target_model: requested_model,
        ..Default::default()
    };
    let mut chunks = chunks;

    let events = try_stream! {
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

    Ok(Box::pin(events))
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
) -> std::result::Result<LlmResponseStream, Box<dyn std::error::Error + Send + Sync>>
where
    S: Stream<Item = std::result::Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>>>
        + Send
        + 'static,
{
    let marker = sse::done_marker(source);
    // The source is always a built-in wire format, so this lookup cannot fail; a
    // failure returns as an `Err` rather than a panic.
    let codec = StreamCodecRegistry::with_builtins().codec(source)?;
    // Adapt the byte-chunk stream into an async line reader. The BufReader
    // reassembles data split across network chunks (including multi-byte UTF-8),
    // and `lines()` yields one SSE field line at a time. The stream is boxed to
    // an `io::Error` item so `into_async_read`'s error bound resolves cleanly.
    let io_bytes: Pin<Box<dyn Stream<Item = std::io::Result<Vec<u8>>> + Send>> =
        Box::pin(bytes.map(|item| item.map_err(std::io::Error::other)));
    let lines = futures::io::BufReader::new(io_bytes.into_async_read()).lines();

    let mut state = StreamTranslationState::default();
    let mut frame = String::new();
    let stream = Box::pin(try_stream! {
        futures::pin_mut!(lines);
        while let Some(line) = lines.next().await {
            let line = line?;
            // A blank line (allowing a bare CR for CRLF streams) ends the frame.
            if line.trim_end().is_empty() {
                if let Some(value) = sse::parse_json_sse_frame(&frame, marker)? {
                    for event in codec.decode_event(&mut state, &value) {
                        yield event;
                    }
                }
                frame.clear();
            } else {
                frame.push_str(&line);
                frame.push('\n');
            }
        }

        // A non-standard upstream might omit the final blank line; parse a trailing
        // complete frame instead of losing its last chunk.
        if !frame.trim_end().is_empty() {
            if let Some(value) = sse::parse_json_sse_frame(&frame, marker)? {
                for event in codec.decode_event(&mut state, &value) {
                    yield event;
                }
            }
        }
    });
    Ok(stream)
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;
    use futures::{stream, Stream, StreamExt};
    use serde_json::{json, Value};
    use switchyard_protocol::{completion_text, LlmResponseChunk};

    use super::{
        decode_aggregated_response, decode_request, decode_stream, encode_aggregated_response,
        encode_request, encode_stream,
    };
    use crate::{LlmResponseStream, WireFormat};

    // A boxed stream item error, matching the streamed IR contract.
    type BoxError = Box<dyn std::error::Error + Send + Sync>;

    // Collects a decoded IR stream, surfacing the first error instead of panicking.
    fn decode_all(
        bytes: impl Stream<Item = Result<Vec<u8>, BoxError>> + Send + 'static,
        source: WireFormat,
    ) -> Result<Vec<LlmResponseChunk>, BoxError> {
        block_on(decode_stream(bytes, source)?.collect::<Vec<_>>())
            .into_iter()
            .collect()
    }

    // Concatenates the text of every `TextDelta` chunk.
    fn text_of(chunks: &[LlmResponseChunk]) -> String {
        chunks
            .iter()
            .filter_map(|chunk| match chunk {
                LlmResponseChunk::TextDelta { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn request_round_trips_through_openai_chat() -> Result<(), BoxError> {
        let body = json!({"model": "gpt", "messages": [{"role": "user", "content": "hi"}]});
        let request = decode_request(WireFormat::OpenAiChat, &body)?;
        assert_eq!(request.model.as_deref(), Some("gpt"));

        let encoded = encode_request(&request, WireFormat::OpenAiChat)?;
        assert_eq!(encoded["model"], "gpt");
        assert_eq!(encoded["messages"][0]["content"], "hi");
        Ok(())
    }

    #[test]
    fn aggregated_response_round_trips_and_restamps_model() -> Result<(), BoxError> {
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
        let agg = decode_aggregated_response(&body, WireFormat::OpenAiChat)?;
        assert_eq!(completion_text(&agg), "Hi there");

        // `requested_model` overrides the encoded model.
        let encoded =
            encode_aggregated_response(&agg, WireFormat::OpenAiChat, Some("client-facing"))?;
        assert_eq!(encoded["model"], "client-facing");
        assert_eq!(encoded["choices"][0]["message"]["content"], "Hi there");
        Ok(())
    }

    #[test]
    fn encode_stream_reassembles_deltas_and_finishes() -> Result<(), BoxError> {
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

        let events = block_on(
            encode_stream(chunks, WireFormat::OpenAiChat, Some("m".to_string()))?
                .collect::<Vec<_>>(),
        )
        .into_iter()
        .collect::<Result<Vec<Value>, BoxError>>()?;

        let content: String = events
            .iter()
            .filter_map(|event| event["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(content, "Hello world");
        // The terminal chunk carries the stop reason through to the wire events.
        assert!(events
            .iter()
            .any(|event| event["choices"][0]["finish_reason"] == "stop"));
        Ok(())
    }

    #[test]
    fn encode_stream_propagates_chunk_errors() -> Result<(), BoxError> {
        let chunks: LlmResponseStream = stream::iter(vec![Err::<LlmResponseChunk, BoxError>(
            "chunk exploded".into(),
        )])
        .boxed();
        let results =
            block_on(encode_stream(chunks, WireFormat::OpenAiChat, None)?.collect::<Vec<_>>());
        assert!(results.iter().any(Result::is_err));
        Ok(())
    }

    #[test]
    fn decode_stream_parses_sse_bytes_into_ir_chunks() -> Result<(), BoxError> {
        let sse = b"data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n\
             data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n\
             data: [DONE]\n\n"
            .to_vec();
        let bytes = stream::once(async move { Ok::<Vec<u8>, BoxError>(sse) });
        let chunks = decode_all(bytes, WireFormat::OpenAiChat)?;
        assert_eq!(text_of(&chunks), "Hello world");
        Ok(())
    }

    #[test]
    fn decode_stream_reassembles_frames_split_across_chunks() -> Result<(), BoxError> {
        // A multi-byte codepoint and the frame boundaries are split across
        // one-byte chunks; the BufReader must reassemble them losslessly.
        let payload = json!({"choices": [{"delta": {"content": "café"}}]});
        let sse = format!("data: {payload}\n\ndata: [DONE]\n\n");
        let bytes = stream::iter(
            sse.into_bytes()
                .into_iter()
                .map(|byte| Ok::<Vec<u8>, BoxError>(vec![byte])),
        );
        let chunks = decode_all(bytes, WireFormat::OpenAiChat)?;
        assert_eq!(text_of(&chunks), "café");
        Ok(())
    }

    #[test]
    fn decode_stream_decodes_trailing_frame_without_blank_line() -> Result<(), BoxError> {
        // A non-standard upstream omits the final blank line; the last frame
        // must still be decoded rather than dropped.
        let sse = b"data: {\"choices\":[{\"delta\":{\"content\":\"tail\"}}]}".to_vec();
        let bytes = stream::once(async move { Ok::<Vec<u8>, BoxError>(sse) });
        let chunks = decode_all(bytes, WireFormat::OpenAiChat)?;
        assert_eq!(text_of(&chunks), "tail");
        Ok(())
    }

    #[test]
    fn decode_stream_decodes_crlf_delimited_frames() -> Result<(), BoxError> {
        // CRLF framing: blank lines are `\r\n\r\n` and the bare `\r` must not
        // block the frame boundary.
        let sse =
            b"data: {\"choices\":[{\"delta\":{\"content\":\"crlf\"}}]}\r\n\r\ndata: [DONE]\r\n\r\n"
                .to_vec();
        let bytes = stream::once(async move { Ok::<Vec<u8>, BoxError>(sse) });
        let chunks = decode_all(bytes, WireFormat::OpenAiChat)?;
        assert_eq!(text_of(&chunks), "crlf");
        Ok(())
    }

    #[test]
    fn decode_stream_propagates_source_errors() -> Result<(), BoxError> {
        // A transport error mid-stream surfaces as an error item, not a panic.
        let bytes = stream::iter(vec![
            Ok::<Vec<u8>, BoxError>(
                b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n".to_vec(),
            ),
            Err::<Vec<u8>, BoxError>("upstream exploded".into()),
        ]);
        let results = block_on(decode_stream(bytes, WireFormat::OpenAiChat)?.collect::<Vec<_>>());
        assert!(results.iter().any(Result::is_err));
        Ok(())
    }
}
