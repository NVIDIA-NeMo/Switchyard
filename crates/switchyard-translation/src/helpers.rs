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
    AggLlmResponse, FormatId, LlmRequest, LlmResponseChunk, LlmResponseStream,
    LlmResponseStreamEvent, LlmStreamError, Result, StreamCodecRegistry, StreamTranslationState,
    TranslationEngine, TranslationPolicy, WireFormat,
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
    encode_aggregated_response_with_extensions(
        agg,
        wire_format,
        served_model,
        &switchyard_protocol::ProviderExtensions::default(),
    )
}

/// Encodes an aggregated response, honouring request extensions.
///
/// Identical to [`encode_aggregated_response`] except that Codex tool
/// namespaces recorded on the request are restored on the encoded body.
pub fn encode_aggregated_response_with_extensions(
    agg: &AggLlmResponse,
    wire_format: WireFormat,
    served_model: Option<&str>,
    request_extensions: &switchyard_protocol::ProviderExtensions,
) -> Result<Value> {
    let mut body = DEFAULT_TRANSLATION_ENGINE
        .encode_response(wire_format, agg, &DEFAULT_TRANSLATION_POLICY)?
        .body;
    if let (Some(model), Value::Object(object)) = (served_model, &mut body) {
        object.insert("model".to_string(), Value::String(model.to_string()));
    }
    crate::codex_namespaces::restore_qualified_tool_names(
        &mut body,
        &crate::codex_namespaces::qualified_tool_origins(request_extensions),
    );
    Ok(body)
}

/// A stream of wire-format event objects in one format — the unframed body of an
/// SSE response. The serving layer frames each `Value` (e.g. as an SSE
/// `data:`/`event:` block).
pub type RawEventStream =
    Pin<Box<dyn Stream<Item = std::result::Result<Value, LlmStreamError>> + Send>>;

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
    encode_stream_with_extensions(
        chunks,
        target,
        served_model,
        &switchyard_protocol::ProviderExtensions::default(),
    )
}

/// Encodes a response stream, honouring request extensions.
///
/// Identical to [`encode_stream`] except that Codex tool namespaces recorded on
/// the request are restored on each encoded event.
pub fn encode_stream_with_extensions(
    chunks: LlmResponseStream,
    target: WireFormat,
    served_model: Option<String>,
    request_extensions: &switchyard_protocol::ProviderExtensions,
) -> std::result::Result<RawEventStream, LlmClientError> {
    let origins = crate::codex_namespaces::qualified_tool_origins(request_extensions);
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
            let event = item.map_err(LlmStreamError::Client)?;
            let mut encoded =
                encode_response_stream_event(&mut state, codec.as_ref(), &target_format, event);
            for value in &mut encoded {
                stamp_streamed_response_model(value, target, served_model_for_events.as_deref());
                crate::codex_namespaces::restore_qualified_tool_names(value, &origins);
            }
            // A codec that errored emits the error event last and nothing after it, so the
            // final value is the terminal frame.
            let terminal = state.errored.then(|| encoded.pop()).flatten();
            for value in encoded {
                yield value;
            }
            if state.errored {
                Err(terminal.map_or_else(
                    || {
                        LlmClientError::ResponseTranslation(
                            "stream ended on an in-band error without an error event".to_string(),
                        )
                        .into()
                    },
                    LlmStreamError::Stream,
                ))?;
            }
        }
        for mut value in codec.finish(&mut state) {
            stamp_streamed_response_model(
                &mut value,
                target,
                served_model_for_events.as_deref(),
            );
            crate::codex_namespaces::restore_qualified_tool_names(&mut value, &origins);
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

/// Decodes provider SSE bytes into normalized stream events.
///
/// Operates on raw bytes, not any HTTP client type: the caller adapts its
/// transport's body stream into `Stream<Item = Result<Vec<u8>, _>>`. Frames are
/// buffered across chunks (a partial frame waits for its boundary); the source
/// stream codec is resolved once and reused for every frame.
///
/// The decoder tracks the source format's protocol-specific terminal event. If
/// EOF arrives without that required event, the stream yields a deferred
/// [`LlmClientError::ResponseTranslation`] error after any valid decoded events.
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
    let mut saw_terminal = false;
    let mut saw_error = false;
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
                    sse::SseFrame::Done => {
                        saw_terminal |= source != WireFormat::AnthropicMessages;
                        break;
                    }
                    sse::SseFrame::Data(value) => {
                        saw_terminal |= sse::is_terminal_event(source, &value);
                        let normalized = codec.decode_event(&mut state, &value);
                        saw_error |= normalized.iter().any(|chunk| matches!(
                            chunk,
                            LlmResponseChunk::DecodeError { .. }
                                | LlmResponseChunk::StreamError { .. }
                        ));
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
            match parsed {
                sse::SseFrame::Done => {
                    saw_terminal |= source != WireFormat::AnthropicMessages;
                }
                sse::SseFrame::Data(value) => {
                    saw_terminal |= sse::is_terminal_event(source, &value);
                    let normalized = codec.decode_event(&mut state, &value);
                    saw_error |= normalized.iter().any(|chunk| matches!(
                        chunk,
                        LlmResponseChunk::DecodeError { .. }
                            | LlmResponseChunk::StreamError { .. }
                    ));
                    yield LlmResponseStreamEvent::preserved(source_format, value, normalized);
                }
                sse::SseFrame::Empty => {}
            }
        }

        if !saw_terminal && !saw_error {
            Err(LlmClientError::ResponseTranslation(format!(
                "{source} stream ended before a terminal event"
            )))?;
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
mod tests {
    use futures::executor::block_on;
    use futures::{Stream, StreamExt, stream};
    use serde_json::{Value, json};
    use switchyard_protocol::{
        LlmClientError, LlmResponseChunk, LlmResponseStreamEvent, completion_text,
    };

    use super::{
        decode_aggregated_response, decode_request, decode_stream, encode_aggregated_response,
        encode_request, encode_stream, stamp_streamed_response_model,
    };
    use crate::{LlmResponseStream, LlmStreamError, WireFormat};

    // A boxed stream item error, matching the streamed IR contract.
    type BoxError = Box<dyn std::error::Error + Send + Sync>;

    // Collects a decoded IR stream, surfacing the first error instead of panicking.
    fn decode_all(
        bytes: impl Stream<Item = Result<Vec<u8>, LlmClientError>> + Send + 'static,
        source: WireFormat,
    ) -> Result<Vec<LlmResponseStreamEvent>, LlmClientError> {
        block_on(decode_stream(bytes, source)?.collect::<Vec<_>>())
            .into_iter()
            .collect()
    }

    // Concatenates the text of every `TextDelta` chunk.
    fn text_of(events: &[LlmResponseStreamEvent]) -> String {
        events
            .iter()
            .flat_map(LlmResponseStreamEvent::normalized)
            .filter_map(|chunk| {
                if let LlmResponseChunk::TextDelta { text, .. } = chunk {
                    Some(text.as_str())
                } else {
                    None
                }
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
    fn aggregated_response_round_trips_and_stamps_the_served_model() -> Result<(), BoxError> {
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

        // `served_model` overrides the id the upstream reported.
        let encoded =
            encode_aggregated_response(&agg, WireFormat::OpenAiChat, Some("served/model"))?;
        assert_eq!(encoded["model"], "served/model");
        assert_eq!(encoded["choices"][0]["message"]["content"], "Hi there");
        Ok(())
    }

    #[test]
    fn encode_stream_reassembles_deltas_and_finishes() -> Result<(), BoxError> {
        let chunks: LlmResponseStream = stream::iter(vec![
            Ok(LlmResponseChunk::TextDelta {
                index: 0,
                text: "Hello".to_string(),
            }
            .into()),
            Ok(LlmResponseChunk::TextDelta {
                index: 0,
                text: " world".to_string(),
            }
            .into()),
            Ok(LlmResponseChunk::MessageStop {
                reason: Some("stop".to_string()),
            }
            .into()),
        ])
        .boxed();

        let events = block_on(
            encode_stream(chunks, WireFormat::OpenAiChat, Some("m".to_string()))?
                .collect::<Vec<_>>(),
        )
        .into_iter()
        .collect::<Result<Vec<Value>, LlmStreamError>>()?;

        let content: String = events
            .iter()
            .filter_map(|event| event["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(content, "Hello world");
        // The terminal chunk carries the stop reason through to the wire events.
        assert!(
            events
                .iter()
                .any(|event| event["choices"][0]["finish_reason"] == "stop")
        );
        Ok(())
    }

    // The served model must win over the id the source stream announced, so a
    // routed response never reports the route the caller addressed.
    #[test]
    fn encode_stream_stamps_the_served_model_on_message_start() -> Result<(), BoxError> {
        let chunks: LlmResponseStream = stream::iter(vec![
            Ok(LlmResponseChunk::MessageStart {
                id: Some("msg_1".to_string()),
                model: Some("upstream/model".to_string()),
            }
            .into()),
            Ok(LlmResponseChunk::TextDelta {
                index: 0,
                text: "hi".to_string(),
            }
            .into()),
        ])
        .boxed();

        let events = block_on(
            encode_stream(
                chunks,
                WireFormat::AnthropicMessages,
                Some("served/model".to_string()),
            )?
            .collect::<Vec<_>>(),
        )
        .into_iter()
        .collect::<Result<Vec<Value>, LlmStreamError>>()?;

        assert_eq!(events[0]["type"], "message_start");
        assert_eq!(events[0]["message"]["model"], "served/model");
        Ok(())
    }

    // Without a served model the upstream id survives, so passthrough routes keep
    // reporting whatever the provider announced.
    #[test]
    fn encode_stream_falls_back_to_the_source_model() -> Result<(), BoxError> {
        let chunks: LlmResponseStream = stream::iter(vec![
            Ok(LlmResponseChunk::MessageStart {
                id: Some("msg_1".to_string()),
                model: Some("upstream/model".to_string()),
            }
            .into()),
            Ok(LlmResponseChunk::TextDelta {
                index: 0,
                text: "hi".to_string(),
            }
            .into()),
        ])
        .boxed();

        let events = block_on(
            encode_stream(chunks, WireFormat::AnthropicMessages, None)?.collect::<Vec<_>>(),
        )
        .into_iter()
        .collect::<Result<Vec<Value>, LlmStreamError>>()?;

        assert_eq!(events[0]["message"]["model"], "upstream/model");
        Ok(())
    }

    #[test]
    fn encode_stream_propagates_chunk_errors() -> Result<(), BoxError> {
        let chunks: LlmResponseStream =
            stream::iter(vec![Err::<LlmResponseStreamEvent, LlmClientError>(
                LlmClientError::General("chunk exploded".to_string()),
            )])
            .boxed();
        let results =
            block_on(encode_stream(chunks, WireFormat::OpenAiChat, None)?.collect::<Vec<_>>());
        assert!(results.iter().any(Result::is_err));
        Ok(())
    }

    // An in-band error is terminal for every target format: the encoder emits the pre-error
    // content, surfaces the error event through the error channel, and drops any later chunk.
    // Production truncates the source before the encoder, so this contract is only observable
    // by driving encode_stream directly.
    #[test]
    fn encode_stream_stops_after_an_in_band_error() -> Result<(), BoxError> {
        for message in [
            LlmResponseChunk::StreamError {
                message: "boom".to_string(),
            },
            LlmResponseChunk::DecodeError {
                message: "boom".to_string(),
            },
        ] {
            for target in [
                WireFormat::OpenAiChat,
                WireFormat::OpenAiResponses,
                WireFormat::AnthropicMessages,
            ] {
                let chunks: LlmResponseStream = stream::iter(vec![
                    Ok(LlmResponseChunk::TextDelta {
                        index: 0,
                        text: "before".to_string(),
                    }
                    .into()),
                    Ok(message.clone().into()),
                    Ok(LlmResponseChunk::TextDelta {
                        index: 0,
                        text: "after".to_string(),
                    }
                    .into()),
                ])
                .boxed();
                let results = block_on(encode_stream(chunks, target, None)?.collect::<Vec<_>>());
                let (data, errors): (Vec<_>, Vec<_>) = results.into_iter().partition(Result::is_ok);
                let data = data.into_iter().flatten().collect::<Vec<Value>>();
                let body = serde_json::to_string(&data)?;

                assert!(
                    body.contains("before"),
                    "{target:?}: pre-error content missing:\n{body}"
                );
                assert!(
                    !body.contains("after"),
                    "{target:?}/{message:?}: content leaked after the error:\n{body}"
                );
                // The error rides the error channel, carrying the codec's own encoded event
                // so the consumer can forward the upstream payload rather than rebuild it.
                let [Err(LlmStreamError::Stream(event))] = errors.as_slice() else {
                    panic!("{target:?}/{message:?}: expected one in-band error, got {errors:?}");
                };
                assert!(
                    serde_json::to_string(event)?.contains("boom"),
                    "{target:?}: error event lost its message: {event}"
                );
            }
        }
        Ok(())
    }

    // A clean stop stays entirely on the data channel, so a consumer that appends a
    // format-level success sentinel only does so for a turn that actually completed.
    #[test]
    fn encode_stream_keeps_a_normal_stop_off_the_error_channel() -> Result<(), BoxError> {
        let chunks: LlmResponseStream = stream::iter(vec![
            Ok(LlmResponseChunk::TextDelta {
                index: 0,
                text: "hi".to_string(),
            }
            .into()),
            Ok(LlmResponseChunk::MessageStop {
                reason: Some("stop".to_string()),
            }
            .into()),
        ])
        .boxed();

        let results =
            block_on(encode_stream(chunks, WireFormat::OpenAiChat, None)?.collect::<Vec<_>>());

        assert!(
            results.iter().all(Result::is_ok),
            "a completed turn reported an error: {results:?}"
        );
        Ok(())
    }

    // A replayed provider error ends the stream before the encoder polls the source again.
    #[test]
    fn encode_stream_stops_polling_after_a_replayed_error() -> Result<(), BoxError> {
        let error = LlmResponseStreamEvent::preserved(
            WireFormat::OpenAiResponses,
            json!({"type": "error", "message": "boom"}),
            vec![LlmResponseChunk::StreamError {
                message: "boom".to_string(),
            }],
        );
        let chunks: LlmResponseStream = stream::iter([Ok(error)])
            .chain(stream::poll_fn(|_| {
                panic!("encode_stream polled the source after an in-band error")
            }))
            .boxed();

        let events =
            block_on(encode_stream(chunks, WireFormat::OpenAiResponses, None)?.collect::<Vec<_>>());

        // The replayed provider event is the terminal frame, so it arrives verbatim on the
        // error channel and nothing follows it.
        let [Err(LlmStreamError::Stream(event))] = events.as_slice() else {
            return Err(format!("expected one in-band error, got {events:?}").into());
        };
        assert_eq!(*event, json!({"type": "error", "message": "boom"}));
        Ok(())
    }

    // The guard keys on `errored`, not `finished`, so a normal completion still emits the
    // trailing usage chunk the OpenAI chat codec reports only after `finished` is set.
    #[test]
    fn encode_stream_keeps_trailing_usage_after_a_normal_stop() -> Result<(), BoxError> {
        let chunks: LlmResponseStream = stream::iter(vec![
            Ok(LlmResponseChunk::TextDelta {
                index: 0,
                text: "hi".to_string(),
            }
            .into()),
            Ok(LlmResponseChunk::MessageStop {
                reason: Some("stop".to_string()),
            }
            .into()),
            Ok(LlmResponseChunk::Usage(switchyard_protocol::llm::Usage {
                output_tokens: Some(7),
                ..Default::default()
            })
            .into()),
        ])
        .boxed();
        let events =
            block_on(encode_stream(chunks, WireFormat::OpenAiChat, None)?.collect::<Vec<_>>())
                .into_iter()
                .collect::<Result<Vec<Value>, LlmStreamError>>()?;
        let body = serde_json::to_string(&events)?;
        assert!(
            events
                .iter()
                .any(|event| event["choices"][0]["finish_reason"] == "stop"),
            "missing stop terminal:\n{body}"
        );
        assert!(
            body.contains("\"usage\""),
            "trailing usage dropped after a normal stop:\n{body}"
        );
        Ok(())
    }

    #[test]
    fn decode_stream_parses_sse_bytes_into_ir_chunks() -> Result<(), LlmClientError> {
        let sse = b"data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n\
             data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n\
             data: [DONE]\n\n\
             data: {\"choices\":[{\"delta\":{\"content\":\" ignored\"}}]}\n\n"
            .to_vec();
        let bytes = stream::once(async move { Ok::<Vec<u8>, LlmClientError>(sse) });
        let chunks = decode_all(bytes, WireFormat::OpenAiChat)?;
        assert_eq!(text_of(&chunks), "Hello world");
        Ok(())
    }

    #[test]
    fn stream_helpers_replay_same_format_provider_fields() -> Result<(), BoxError> {
        let provider_event = json!({
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_provider_specific",
            "choices": [{
                "index": 0,
                "delta": {"content": "Hello"},
                "finish_reason": "stop"
            }]
        });
        let bytes = stream::once({
            let frame = format!("data: {provider_event}\n\n").into_bytes();
            async move { Ok::<Vec<u8>, LlmClientError>(frame) }
        });
        let decoded = decode_stream(bytes, WireFormat::OpenAiChat)?;
        let replayed =
            block_on(encode_stream(decoded, WireFormat::OpenAiChat, None)?.collect::<Vec<_>>())
                .into_iter()
                .collect::<Result<Vec<Value>, LlmStreamError>>()?;

        assert_eq!(replayed, vec![provider_event]);
        Ok(())
    }

    #[test]
    fn openai_chat_replay_stamps_the_served_model_without_losing_extensions() {
        let mut event = json!({
            "choices": [{"delta": {"content": "Hello"}}],
            "system_fingerprint": "fp_provider_specific",
        });

        stamp_streamed_response_model(&mut event, WireFormat::OpenAiChat, Some("served/model"));

        assert_eq!(event["model"], "served/model");
        assert_eq!(event["system_fingerprint"], "fp_provider_specific");
    }

    #[test]
    fn responses_replay_stamps_the_served_model_inside_response() {
        let mut event = json!({
            "type": "response.created",
            "response": {
                "id": "resp_1",
                "model": "provider/model",
                "provider_extension": true,
            },
        });

        stamp_streamed_response_model(
            &mut event,
            WireFormat::OpenAiResponses,
            Some("served/model"),
        );

        assert_eq!(event["response"]["model"], "served/model");
        assert_eq!(event["response"]["provider_extension"], true);
    }

    #[test]
    fn anthropic_replay_stamps_the_served_model_inside_message() {
        let mut event = json!({
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "model": "provider/model",
                "provider_extension": true,
            },
        });

        stamp_streamed_response_model(
            &mut event,
            WireFormat::AnthropicMessages,
            Some("served/model"),
        );

        assert_eq!(event["message"]["model"], "served/model");
        assert_eq!(event["message"]["provider_extension"], true);
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
                .map(|byte| Ok::<Vec<u8>, LlmClientError>(vec![byte])),
        );
        let chunks = decode_all(bytes, WireFormat::OpenAiChat)?;
        assert_eq!(text_of(&chunks), "café");
        Ok(())
    }

    #[test]
    fn decode_stream_decodes_trailing_frame_without_blank_line() -> Result<(), BoxError> {
        // A non-standard upstream omits the final blank line; the last frame
        // must still be decoded rather than dropped.
        let sse =
            b"data: {\"choices\":[{\"delta\":{\"content\":\"tail\"}}]}\n\ndata: [DONE]".to_vec();
        let bytes = stream::once(async move { Ok::<Vec<u8>, LlmClientError>(sse) });
        let chunks = decode_all(bytes, WireFormat::OpenAiChat)?;
        assert_eq!(text_of(&chunks), "tail");
        Ok(())
    }

    #[test]
    fn decode_stream_rejects_eof_before_a_terminal_event() -> Result<(), BoxError> {
        // Preserve valid content before surfacing premature EOF as the terminal stream error.
        let sse = b"data: {\"choices\":[{\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\n".to_vec();
        let bytes = stream::once(async move { Ok::<Vec<u8>, LlmClientError>(sse) });
        let results = block_on(decode_stream(bytes, WireFormat::OpenAiChat)?.collect::<Vec<_>>());

        let Some(Ok(first)) = results.first() else {
            return Err("expected the partial event".into());
        };
        assert_eq!(text_of(std::slice::from_ref(first)), "partial");
        let Some(Err(LlmClientError::ResponseTranslation(message))) = results.last() else {
            panic!("expected incomplete OpenAI stream to fail");
        };
        assert_eq!(message, "openai_chat stream ended before a terminal event");
        Ok(())
    }

    #[test]
    fn responses_failed_before_done_remains_a_stream_error() -> Result<(), BoxError> {
        let sse = b"data: {\"type\":\"response.failed\",\"response\":{\"error\":{\"message\":\"upstream failed\"}}}\n\ndata: [DONE]\n\n".to_vec();
        let bytes = stream::once(async move { Ok::<Vec<u8>, LlmClientError>(sse) });
        let events = decode_all(bytes, WireFormat::OpenAiResponses)?;

        let Some(LlmResponseChunk::StreamError { message }) =
            events.first().and_then(|event| event.normalized().first())
        else {
            return Err("expected response.failed to decode as StreamError".into());
        };
        assert_eq!(message, "upstream failed");

        let chunks: LlmResponseStream = stream::iter(
            events
                .into_iter()
                .map(Ok::<LlmResponseStreamEvent, LlmClientError>),
        )
        .boxed();
        let replayed =
            block_on(encode_stream(chunks, WireFormat::OpenAiResponses, None)?.collect::<Vec<_>>());

        // `response.failed` stays a terminal error, replayed with its payload intact.
        let [Err(LlmStreamError::Stream(event))] = replayed.as_slice() else {
            return Err(format!("expected one in-band error, got {replayed:?}").into());
        };
        assert_eq!(
            *event,
            json!({
                "type": "response.failed",
                "response": {"error": {"message": "upstream failed"}}
            })
        );
        Ok(())
    }

    #[test]
    fn decode_stream_decodes_crlf_delimited_frames() -> Result<(), BoxError> {
        // CRLF framing: blank lines are `\r\n\r\n` and the bare `\r` must not
        // block the frame boundary.
        let sse =
            b"data: {\"choices\":[{\"delta\":{\"content\":\"crlf\"}}]}\r\n\r\ndata: [DONE]\r\n\r\n"
                .to_vec();
        let bytes = stream::once(async move { Ok::<Vec<u8>, LlmClientError>(sse) });
        let chunks = decode_all(bytes, WireFormat::OpenAiChat)?;
        assert_eq!(text_of(&chunks), "crlf");
        Ok(())
    }

    #[test]
    fn decode_stream_propagates_source_errors() -> Result<(), BoxError> {
        // A transport error mid-stream surfaces as an error item, not a panic.
        let bytes = stream::iter(vec![
            Ok::<Vec<u8>, LlmClientError>(
                b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n".to_vec(),
            ),
            Err::<Vec<u8>, LlmClientError>(LlmClientError::Transport {
                source: Box::new(std::io::Error::other("upstream exploded")),
            }),
        ]);
        let results = block_on(decode_stream(bytes, WireFormat::OpenAiChat)?.collect::<Vec<_>>());
        let Some(Err(error)) = results.last() else {
            panic!("expected the source error");
        };
        assert!(matches!(error, LlmClientError::Transport { .. }));
        Ok(())
    }

    #[test]
    fn decode_stream_classifies_invalid_sse_json() -> Result<(), BoxError> {
        let bytes =
            stream::once(async { Ok::<Vec<u8>, LlmClientError>(b"data: {invalid}\n\n".to_vec()) });
        let results = block_on(decode_stream(bytes, WireFormat::OpenAiChat)?.collect::<Vec<_>>());
        let Some(Err(error)) = results.last() else {
            panic!("expected invalid SSE JSON to fail");
        };
        assert!(matches!(error, LlmClientError::ResponseTranslation(_)));
        Ok(())
    }
}
