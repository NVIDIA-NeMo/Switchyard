// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared streaming codec contracts, registry, and stream state.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::codecs::anthropic::AnthropicMessagesStreamCodec;
use crate::codecs::openai_chat::OpenAiChatStreamCodec;
use crate::codecs::responses::OpenAiResponsesStreamCodec;
use crate::engine::{FormatRegistry, TranslationEngine};
use crate::error::{Result, TranslationError};
use crate::format::{FormatId, WireFormat};
use crate::llm::Usage;
use crate::LlmResponseChunk;

/// Mutable state accumulated while translating one streaming response.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct StreamTranslationState {
    pub source: Option<FormatId>,
    pub target: Option<FormatId>,
    /// Model name observed on the source provider stream.
    pub model: Option<String>,
    /// Message/response ID observed on the source provider stream.
    pub message_id: Option<String>,
    /// Optional model name the target stream should expose to the client.
    pub target_model: Option<String>,
    /// Optional message/response ID the target stream should expose to the client.
    pub target_message_id: Option<String>,
    pub saw_message_start: bool,
    pub emitted_message_start: bool,
    pub finished: bool,
    pub usage: Usage,

    pub(crate) output_tokens_seen: u64,
    pub(crate) saw_backend_usage: bool,
    pub(crate) usage_extras: BTreeMap<String, u64>,
    pub(crate) stop_reason: Option<String>,

    pub(crate) next_content_index: usize,
    pub(crate) text_block_index: Option<usize>,
    pub(crate) text_block_started: bool,
    pub(crate) emitted_content_block: bool,
    pub(crate) tool_states: BTreeMap<usize, StreamToolState>,

    pub(crate) response_created: bool,
    pub(crate) response_text_started: bool,
    pub(crate) response_text_output_index: Option<usize>,
    pub(crate) response_text: String,
    pub(crate) response_reasoning_started: bool,
    pub(crate) response_reasoning_output_index: Option<usize>,
    pub(crate) response_reasoning_text: String,
    pub(crate) next_response_output_index: usize,

    pub(crate) reasoning_block_index: Option<usize>,
    pub(crate) reasoning_block_started: bool,
}

// Tracks an in-progress streamed tool call across provider-specific deltas.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct StreamToolState {
    pub(crate) id: Option<String>,
    pub(crate) name: Option<String>,
    pub(crate) arguments: String,
    pub(crate) pending_arguments: String,
    pub(crate) started: bool,
    pub(crate) content_index: Option<usize>,
    pub(crate) response_output_index: Option<usize>,
    pub(crate) response_item_id: Option<String>,
}

impl StreamTranslationState {
    /// Creates stream state with source and target formats already attached.
    pub fn new(source: impl Into<FormatId>, target: impl Into<FormatId>) -> Self {
        Self {
            source: Some(source.into()),
            target: Some(target.into()),
            ..Self::default()
        }
    }
}

/// Registry-backed streaming translator.
#[derive(Default)]
pub struct StreamTranslationEngine {
    engine: TranslationEngine,
}

/// Codec contract for one provider streaming event format.
pub trait StreamCodec: Send + Sync {
    /// Returns the stream format handled by this codec.
    fn format(&self) -> FormatId;

    /// Decodes one provider event into zero or more neutral events.
    fn decode_event(
        &self,
        state: &mut StreamTranslationState,
        event: &Value,
    ) -> Vec<LlmResponseChunk>;

    /// Encodes one neutral event into zero or more provider events.
    fn encode_event(
        &self,
        state: &mut StreamTranslationState,
        event: LlmResponseChunk,
    ) -> Vec<Value>;

    /// Emits any terminal provider events needed after the source stream ends.
    ///
    /// This is intentionally required on every codec. Some target formats
    /// need explicit terminal events after the source closes (for example,
    /// Anthropic ``message_delta``/``message_stop`` or Responses
    /// ``response.completed``). Formats that have no source-close work should
    /// return an empty vector explicitly so the no-op behavior is a conscious
    /// codec-level choice.
    fn finish(&self, state: &mut StreamTranslationState) -> Vec<Value>;
}

/// Registry mapping stream wire formats to stream codecs.
#[derive(Default)]
pub struct StreamCodecRegistry {
    codecs: BTreeMap<FormatId, Arc<dyn StreamCodec>>,
}

impl StreamCodecRegistry {
    /// Creates an empty stream codec registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a registry populated with built-in stream codecs.
    pub fn with_builtins() -> Self {
        let mut registry = Self::new();
        registry.register(OpenAiChatStreamCodec);
        registry.register(AnthropicMessagesStreamCodec);
        registry.register(OpenAiResponsesStreamCodec);
        registry
    }

    /// Registers or replaces a stream codec for its declared format.
    pub fn register(&mut self, codec: impl StreamCodec + 'static) {
        self.codecs.insert(codec.format(), Arc::new(codec));
    }

    /// Looks up a stream codec by format identifier.
    pub fn codec(&self, format: impl Into<FormatId>) -> Result<Arc<dyn StreamCodec>> {
        let format = format.into();
        self.codecs.get(&format).cloned().ok_or_else(|| {
            TranslationError::Other(format!("no stream codec registered for {format}"))
        })
    }
}

impl StreamTranslationEngine {
    /// Creates a streaming engine from an explicit codec registry.
    pub fn new(registry: StreamCodecRegistry) -> Self {
        Self {
            engine: TranslationEngine::with_registries(FormatRegistry::with_builtins(), registry),
        }
    }

    /// Translates one source provider event into target provider events.
    pub fn translate_event(
        &self,
        state: &mut StreamTranslationState,
        source: impl Into<FormatId>,
        target: impl Into<FormatId>,
        event: &Value,
    ) -> Result<Vec<Value>> {
        self.engine.translate_event(state, source, target, event)
    }

    /// Finishes target-provider stream emission after the source stream closes.
    pub fn finish(
        &self,
        state: &mut StreamTranslationState,
        target: impl Into<FormatId>,
    ) -> Result<Vec<Value>> {
        self.engine.finish_stream(state, target)
    }

    /// Convenience helper using built-in codecs and error events instead of `Result`.
    pub fn translate_event_with_builtins(
        state: &mut StreamTranslationState,
        source: impl Into<FormatId>,
        target: impl Into<FormatId>,
        event: &Value,
    ) -> Vec<Value> {
        Self::default()
            .translate_event(state, source, target, event)
            .unwrap_or_else(|error| vec![json!({"error": {"message": error.to_string()}})])
    }
}

/// Decodes one provider stream event with the built-in codec registry.
pub fn decode_stream_event(
    state: &mut StreamTranslationState,
    source: impl Into<FormatId>,
    event: &Value,
) -> Vec<LlmResponseChunk> {
    let source = source.into();
    StreamCodecRegistry::with_builtins()
        .codec(source)
        .map(|codec| codec.decode_event(state, event))
        .unwrap_or_else(|error| {
            vec![LlmResponseChunk::Error {
                message: error.to_string(),
            }]
        })
}

/// Encodes one neutral stream event with the built-in codec registry.
pub fn encode_stream_event(
    state: &mut StreamTranslationState,
    target: impl Into<FormatId>,
    event: LlmResponseChunk,
) -> Vec<Value> {
    StreamCodecRegistry::with_builtins()
        .codec(target)
        .map(|codec| codec.encode_event(state, event))
        .unwrap_or_else(|error| vec![json!({"error": {"message": error.to_string()}})])
}

// Records source-provider identity carried by decoded stream events.
pub(crate) fn record_source_identity(
    state: &mut StreamTranslationState,
    id: Option<String>,
    model: Option<String>,
) {
    if id.is_some() {
        state.message_id = id;
    }
    if model.is_some() {
        state.model = model;
    }
}

// Returns the source model observed from the upstream stream.
pub(crate) fn source_model_or_unknown(state: &StreamTranslationState) -> String {
    state.model.clone().unwrap_or_else(|| "unknown".to_string())
}

// Returns the target/client model when supplied, otherwise the upstream model.
pub(crate) fn target_model_or_source_model(state: &StreamTranslationState) -> String {
    state
        .target_model
        .clone()
        .or_else(|| state.model.clone())
        .unwrap_or_else(|| "unknown".to_string())
}

// Returns the target/client ID when supplied, otherwise the upstream ID.
pub(crate) fn target_message_id_or_source_message_id(
    state: &StreamTranslationState,
) -> Option<&str> {
    state
        .target_message_id
        .as_deref()
        .or(state.message_id.as_deref())
}

// Checks whether the current source format matches a built-in format.
pub(crate) fn state_source_is(state: &StreamTranslationState, format: WireFormat) -> bool {
    let format_id: FormatId = format.into();
    match &state.source {
        Some(source) => source == &format_id,
        None => false,
    }
}

// Reads a non-empty string field from an event object.
pub(crate) fn string_field(object: &Map<String, Value>, key: &str) -> Option<String> {
    object
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}
