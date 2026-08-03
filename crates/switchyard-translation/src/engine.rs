// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Registry-backed translation engine for buffered requests and responses.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::Value;

use crate::LlmResponseStreamEvent;
use crate::codecs::FormatCodec;
use crate::codecs::anthropic::AnthropicMessagesCodec;
use crate::codecs::openai_chat::OpenAiChatCodec;
use crate::codecs::responses::OpenAiResponsesCodec;
use crate::codecs::stream::{
    StreamCodecRegistry, StreamTranslationState, encode_response_stream_event,
};
use crate::diagnostic::TranslationDiagnostic;
use crate::error::{Result, TranslationError};
use crate::format::FormatId;
use crate::llm::{AggLlmResponse, LlmRequest};
use crate::policy::TranslationPolicy;

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
        let decoded = self.registry.codec(source)?.decode_request(body, policy)?;
        Ok(RequestIrOutput {
            request: decoded.request,
            diagnostics: decoded.diagnostics,
        })
    }

    /// Encodes a neutral request IR into a target wire format.
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

    /// Translates a response body from source format to target format.
    pub fn translate_response(
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
            .decode_response(body, policy)?;
        let encoded = self
            .registry
            .codec(target.clone())?
            .encode_response(&decoded.response, policy)?;
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
