// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Registry-backed translation engine for buffered and streaming translation.

use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::Arc;

use async_stream::try_stream;
use futures::io::AsyncBufReadExt;
use futures::{Stream, StreamExt, TryStreamExt};
use serde_json::Value;
use switchyard_protocol::LlmClientError;

use crate::codecs::FormatCodec;
use crate::codecs::anthropic::AnthropicMessagesCodec;
use crate::codecs::openai_chat::OpenAiChatCodec;
use crate::codecs::responses::OpenAiResponsesCodec;
use crate::codecs::stream::{
    StreamCodecRegistry, StreamTranslationState, encode_response_stream_event,
};
use crate::diagnostic::TranslationDiagnostic;
use crate::error::{Result, TranslationError};
use crate::format::{FormatId, WireFormat};
use crate::llm::{AggLlmResponse, LlmRequest, ProviderExtensions};
use crate::policy::TranslationPolicy;
use crate::sse;
use crate::{LlmResponseChunk, LlmResponseStream, LlmResponseStreamEvent, LlmStreamError};

/// A stream of wire-format event objects in one format — the unframed body of an
/// SSE response. The serving layer frames each `Value` (e.g. as an SSE
/// `data:`/`event:` block).
pub type RawEventStream =
    Pin<Box<dyn Stream<Item = std::result::Result<Value, LlmStreamError>> + Send>>;

/// Encoded translation result with any diagnostics emitted along the way.
#[derive(Debug)]
pub struct TranslationOutput {
    pub body: Value,
    pub diagnostics: Vec<TranslationDiagnostic>,
}

/// Decoded request IR plus diagnostics.
#[derive(Debug)]
pub struct RequestIrOutput {
    pub request: LlmRequest,
    pub diagnostics: Vec<TranslationDiagnostic>,
}

/// Decoded response IR plus diagnostics.
#[derive(Debug)]
pub struct ResponseIrOutput {
    pub response: AggLlmResponse,
    pub diagnostics: Vec<TranslationDiagnostic>,
}

/// Registry mapping wire formats to buffered codecs.
#[derive(Default)]
pub struct FormatRegistry {
    codecs: BTreeMap<FormatId, Arc<dyn FormatCodec>>,
}

impl FormatRegistry {
    /// Creates an empty format registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a registry populated with the built-in provider codecs.
    pub fn with_builtins() -> Self {
        let mut registry = Self::new();
        registry.register(OpenAiChatCodec);
        registry.register(AnthropicMessagesCodec);
        registry.register(OpenAiResponsesCodec);
        registry
    }

    /// Registers or replaces a codec for its declared format.
    pub fn register(&mut self, codec: impl FormatCodec + 'static) {
        self.codecs.insert(codec.format(), Arc::new(codec));
    }

    /// Looks up a codec by format identifier.
    pub fn codec(&self, format: impl Into<FormatId>) -> Result<Arc<dyn FormatCodec>> {
        let format = format.into();
        self.codecs
            .get(&format)
            .cloned()
            .ok_or_else(|| TranslationError::Other(format!("no codec registered for {format}")))
    }
}

/// Stateless request/response translator that routes through the neutral IR.
pub struct TranslationEngine {
    registry: FormatRegistry,
    stream_registry: StreamCodecRegistry,
}

impl Default for TranslationEngine {
    fn default() -> Self {
        Self {
            registry: FormatRegistry::with_builtins(),
            stream_registry: StreamCodecRegistry::with_builtins(),
        }
    }
}

impl TranslationEngine {
    /// Creates an engine from an explicit buffered codec registry.
    pub fn new(registry: FormatRegistry) -> Self {
        Self {
            registry,
            stream_registry: StreamCodecRegistry::with_builtins(),
        }
    }

    /// Creates an engine from explicit buffered and streaming codec registries.
    pub fn with_registries(registry: FormatRegistry, stream_registry: StreamCodecRegistry) -> Self {
        Self {
            registry,
            stream_registry,
        }
    }

    /// Decodes a request body into the neutral request IR.
    pub fn decode_request(
        &self,
        source: impl Into<FormatId>,
        body: &Value,
        policy: &TranslationPolicy,
    ) -> Result<RequestIrOutput> {
        let source = source.into();
        let mut decoded = self.registry.codec(source)?.decode_request(body, policy)?;
        crate::normalize_compaction(&mut decoded.request, None);
        Ok(RequestIrOutput {
            request: decoded.request,
            diagnostics: decoded.diagnostics,
        })
    }

    /// Encodes a neutral request IR into a target wire format.
    ///
    /// Preserved target bodies take precedence over normalized fields, including
    /// `output.is_schema_enforced`. Clear the target's preservation entry or use
    /// [`crate::policy::PreservationPolicy::Disabled`] to encode IR edits.
    pub fn encode_request(
        &self,
        target: impl Into<FormatId>,
        request: &LlmRequest,
        policy: &TranslationPolicy,
    ) -> Result<TranslationOutput> {
        let target = target.into();
        let encoded = self
            .registry
            .codec(target)?
            .encode_request(request, policy)?;
        Ok(TranslationOutput {
            body: encoded.body,
            diagnostics: encoded.diagnostics,
        })
    }

    /// Translates a request body from source format to target format.
    pub fn translate_request(
        &self,
        source: impl Into<FormatId>,
        target: impl Into<FormatId>,
        body: &Value,
        policy: &TranslationPolicy,
    ) -> Result<TranslationOutput> {
        let source = source.into();
        let target = target.into();
        let decoded = self
            .registry
            .codec(source.clone())?
            .decode_request(body, policy)?;
        let encoded = self
            .registry
            .codec(target.clone())?
            .encode_request(&decoded.request, policy)?;
        Ok(TranslationOutput {
            body: encoded.body,
            diagnostics: with_formats(decoded.diagnostics, encoded.diagnostics, source, target),
        })
    }

    /// Decodes a response body into the neutral response IR.
    pub fn decode_response(
        &self,
        source: impl Into<FormatId>,
        body: &Value,
        policy: &TranslationPolicy,
    ) -> Result<ResponseIrOutput> {
        let source = source.into();
        let decoded = self.registry.codec(source)?.decode_response(body, policy)?;
        Ok(ResponseIrOutput {
            response: decoded.response,
            diagnostics: decoded.diagnostics,
        })
    }

    /// Encodes a neutral response IR into a target wire format.
    pub fn encode_response(
        &self,
        target: impl Into<FormatId>,
        response: &AggLlmResponse,
        policy: &TranslationPolicy,
    ) -> Result<TranslationOutput> {
        let target = target.into();
        let encoded = self
            .registry
            .codec(target)?
            .encode_response(response, policy)?;
        Ok(TranslationOutput {
            body: encoded.body,
            diagnostics: encoded.diagnostics,
        })
    }

    /// Encodes a neutral response IR while applying metadata retained from its request.
    ///
    /// Request extensions are used to restore caller-specific response details, such as
    /// Codex tool namespaces, without exposing those details to the upstream provider.
    pub fn encode_response_with_extensions(
        &self,
        target: impl Into<FormatId>,
        response: &AggLlmResponse,
        request_extensions: &ProviderExtensions,
        policy: &TranslationPolicy,
    ) -> Result<TranslationOutput> {
        let mut output = self.encode_response(target, response, policy)?;
        if !request_extensions.fields.is_empty() {
            crate::codex_namespaces::restore_qualified_tool_names(
                &mut output.body,
                &crate::codex_namespaces::qualified_tool_origins(request_extensions),
            );
            crate::codex_custom_tools::restore_custom_tool_calls(
                &mut output.body,
                &crate::codex_custom_tools::custom_tool_names(request_extensions),
            );
        }
        Ok(output)
    }

    /// Translates a response body from source format to target format.
    pub fn translate_response(
        &self,
        source: impl Into<FormatId>,
        target: impl Into<FormatId>,
        body: &Value,
        policy: &TranslationPolicy,
    ) -> Result<TranslationOutput> {
        self.translate_response_with_extensions(
            source,
            target,
            body,
            &ProviderExtensions::default(),
            policy,
        )
    }

    /// Translates a response while restoring details from the original request's extensions.
    ///
    /// Request extensions apply after encoding, including when the target body is preserved.
    pub fn translate_response_with_extensions(
        &self,
        source: impl Into<FormatId>,
        target: impl Into<FormatId>,
        body: &Value,
        request_extensions: &ProviderExtensions,
        policy: &TranslationPolicy,
    ) -> Result<TranslationOutput> {
        let source = source.into();
        let target = target.into();
        let decoded = self
            .registry
            .codec(source.clone())?
            .decode_response(body, policy)?;
        let encoded = self.encode_response_with_extensions(
            target.clone(),
            &decoded.response,
            request_extensions,
            policy,
        )?;
        Ok(TranslationOutput {
            body: encoded.body,
            diagnostics: with_formats(decoded.diagnostics, encoded.diagnostics, source, target),
        })
    }

    /// Translates one streaming source event into zero or more target events.
    pub fn translate_event(
        &self,
        state: &mut StreamTranslationState,
        source: impl Into<FormatId>,
        target: impl Into<FormatId>,
        event: &Value,
    ) -> Result<Vec<Value>> {
        let source = source.into();
        let target = target.into();
        let source_codec = self.stream_registry.codec(source.clone())?;
        let target_codec = self.stream_registry.codec(target.clone())?;
        if let Err(error) =
            crate::codecs::responses::validate_stream_output(&source, &target, event)
        {
            return Ok(target_codec.encode_event(
                state,
                crate::LlmResponseChunk::DecodeError {
                    message: error.to_string(),
                },
            ));
        }
        let canonical = source_codec.decode_event(state, event);
        state.source = Some(source);
        state.target = Some(target);
        Ok(canonical
            .into_iter()
            .flat_map(|event| target_codec.encode_event(state, event))
            .collect())
    }

    /// Decodes one provider event while retaining its parsed source JSON value.
    ///
    /// Takes ownership of `event` so preservation does not deep-copy provider JSON
    /// on the per-event streaming path.
    pub fn decode_stream_event(
        &self,
        state: &mut StreamTranslationState,
        source: impl Into<FormatId>,
        event: Value,
    ) -> Result<LlmResponseStreamEvent> {
        let source = source.into();
        let source_codec = self.stream_registry.codec(source.clone())?;
        state.source = Some(source.clone());
        let normalized = source_codec.decode_event(state, &event);
        Ok(LlmResponseStreamEvent::preserved(source, event, normalized))
    }

    /// Encodes one neutral or preserved stream event for a target provider.
    ///
    /// A preserved event replays its retained JSON value unchanged when its
    /// source and target formats match. Cross-format encoding intentionally
    /// uses only its normalized events.
    pub fn encode_stream_event(
        &self,
        state: &mut StreamTranslationState,
        target: impl Into<FormatId>,
        event: LlmResponseStreamEvent,
    ) -> Result<Vec<Value>> {
        let target = target.into();
        let target_codec = self.stream_registry.codec(target.clone())?;
        state.target = Some(target.clone());
        Ok(encode_response_stream_event(
            state,
            target_codec.as_ref(),
            &target,
            event,
        ))
    }

    /// Finishes target-provider stream emission after the source stream closes.
    pub fn finish_stream(
        &self,
        state: &mut StreamTranslationState,
        target: impl Into<FormatId>,
    ) -> Result<Vec<Value>> {
        let target = target.into();
        let target_codec = self.stream_registry.codec(target)?;
        Ok(target_codec.finish(state))
    }

    /// Encodes a stream of IR chunks into a stream of target-format wire events.
    ///
    /// `served_model` is exposed as the response model (via the stream state's
    /// `target_model`); `None` falls back to the id observed on the source stream.
    /// The target stream codec is resolved once and reused per chunk; terminal events
    /// (`message_stop` / `response.completed`) come from `finish`.
    pub fn encode_stream(
        &self,
        target: impl Into<FormatId>,
        chunks: LlmResponseStream,
        served_model: Option<String>,
    ) -> std::result::Result<RawEventStream, LlmClientError> {
        self.encode_stream_with_extensions(
            target,
            chunks,
            served_model,
            &ProviderExtensions::default(),
        )
    }

    /// Encodes a response stream, honouring request extensions.
    ///
    /// Identical to [`Self::encode_stream`] except that Codex tool namespaces recorded on
    /// the request are restored on each encoded event.
    pub fn encode_stream_with_extensions(
        &self,
        target: impl Into<FormatId>,
        chunks: LlmResponseStream,
        served_model: Option<String>,
        request_extensions: &ProviderExtensions,
    ) -> std::result::Result<RawEventStream, LlmClientError> {
        let origins = crate::codex_namespaces::qualified_tool_origins(request_extensions);
        let custom_tools = crate::codex_custom_tools::custom_tool_names(request_extensions);
        let mut custom_state = crate::codex_custom_tools::CustomToolCallStreamState::default();
        let target_format: FormatId = target.into();
        let codec = self
            .stream_registry
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
                    if let Some(model) = served_model_for_events.as_deref() {
                        codec.stamp_streamed_response_model(value, model);
                    }
                    crate::codex_namespaces::restore_qualified_tool_names(value, &origins);
                }
                // Argument deltas for a freeform tool cannot be expressed on the wire; the
                // rewritten completed item carries the input instead.
                let mut encoded: Vec<Value> = encoded
                    .into_iter()
                    .filter_map(|mut value| {
                        crate::codex_custom_tools::restore_custom_tool_calls_in_event(
                            &mut value,
                            &custom_tools,
                            &mut custom_state,
                        )
                        .then_some(value)
                    })
                    .collect();
                let terminal = encoded.pop();
                for value in encoded {
                    yield value;
                }
                match (state.errored, terminal) {
                    (false, Some(value)) => yield value,
                    (true, Some(value)) => Err(LlmStreamError::Upstream(value))?,
                    (true, None) => Err(LlmStreamError::Client(LlmClientError::ResponseTranslation(
                        "stream ended after an error without a terminal event".to_string(),
                    )))?,
                    _ => ()
                }
            }
            for mut value in codec.finish(&mut state) {
                if let Some(model) = served_model_for_events.as_deref() {
                    codec.stamp_streamed_response_model(&mut value, model);
                }
                crate::codex_namespaces::restore_qualified_tool_names(&mut value, &origins);
                if crate::codex_custom_tools::restore_custom_tool_calls_in_event(
                    &mut value,
                    &custom_tools,
                    &mut custom_state,
                ) {
                    yield value;
                }
            }
        };

        Ok(Box::pin(events))
    }

    /// Decodes provider SSE bytes into normalized stream events.
    ///
    /// Operates on raw bytes, not any HTTP client type: the caller adapts its
    /// transport's body stream into `Stream<Item = Result<Vec<u8>, _>>`. Frames are
    /// buffered across chunks (a partial frame waits for its boundary); the source
    /// stream codec is resolved once and reused for every frame.
    ///
    /// The decoder yields the terminal JSON event before it stops reading. Chat Completions
    /// keeps reading after `finish_reason` to include usage before `[DONE]` or EOF.
    /// If EOF arrives without a required terminal event, the stream yields a deferred
    /// [`LlmClientError::ResponseTranslation`] error after any valid decoded events.
    pub fn decode_stream<S>(
        &self,
        source: impl Into<FormatId>,
        bytes: S,
    ) -> std::result::Result<LlmResponseStream, LlmClientError>
    where
        S: Stream<Item = std::result::Result<Vec<u8>, LlmClientError>> + Send + 'static,
    {
        let marker = sse::done_marker();
        let source_format: FormatId = source.into();
        let codec = self
            .stream_registry
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
                            saw_terminal |= codec.sse_done_is_terminal();
                            break;
                        }
                        sse::SseFrame::Data(value) => {
                            saw_terminal |= codec.is_terminal_event(&value);
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
                            // Chat usage can follow finish_reason, so the decoder keeps reading.
                            if saw_terminal && source_format != WireFormat::OpenAiChat.into() {
                                break;
                            }
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
                        saw_terminal |= codec.sse_done_is_terminal();
                    }
                    sse::SseFrame::Data(value) => {
                        saw_terminal |= codec.is_terminal_event(&value);
                        let normalized = codec.decode_event(&mut state, &value);
                        saw_error |= normalized.iter().any(|chunk| matches!(
                            chunk,
                            LlmResponseChunk::DecodeError { .. }
                                | LlmResponseChunk::StreamError { .. }
                        ));
                        yield LlmResponseStreamEvent::preserved(source_format.clone(), value, normalized);
                    }
                    sse::SseFrame::Empty => {}
                }
            }

            if !saw_terminal && !saw_error {
                Err(LlmClientError::ResponseTranslation(format!(
                    "{source_format} stream ended before a terminal event"
                )))?;
            }
        });
        Ok(stream)
    }
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

// Attaches source and target formats to every diagnostic emitted across both passes.
fn with_formats(
    decoded: Vec<TranslationDiagnostic>,
    encoded: Vec<TranslationDiagnostic>,
    source: FormatId,
    target: FormatId,
) -> Vec<TranslationDiagnostic> {
    decoded
        .into_iter()
        .chain(encoded)
        .map(|diagnostic| diagnostic.with_formats(source.clone(), target.clone()))
        .collect()
}
