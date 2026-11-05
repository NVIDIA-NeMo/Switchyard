// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Buffered wire-format codecs that translate between provider JSON and IR.

pub mod anthropic;
pub(crate) mod common;
pub mod openai_chat;
mod openai_media;
pub mod responses;
pub mod stream;
mod structured_output;

use serde_json::Value;

use crate::diagnostic::TranslationDiagnostic;
use crate::error::Result;
use crate::format::FormatId;
use crate::llm::{AggLlmResponse, LlmRequest};
use crate::policy::TranslationPolicy;

/// Result of decoding a request into neutral IR.
pub struct DecodedRequest {
    pub request: LlmRequest,
    pub diagnostics: Vec<TranslationDiagnostic>,
}

/// Result of encoding a neutral request into provider JSON.
pub struct EncodedRequest {
    pub body: Value,
    pub diagnostics: Vec<TranslationDiagnostic>,
}

/// Result of decoding a response into neutral IR.
pub struct DecodedResponse {
    pub response: AggLlmResponse,
    pub diagnostics: Vec<TranslationDiagnostic>,
}

/// Result of encoding a neutral response into provider JSON.
pub struct EncodedResponse {
    pub body: Value,
    pub diagnostics: Vec<TranslationDiagnostic>,
}

/// Codec contract for one buffered provider wire format.
pub trait FormatCodec: Send + Sync {
    /// Returns the format handled by this codec.
    fn format(&self) -> FormatId;

    /// Decodes a request body into the neutral IR.
    fn decode_request(&self, body: &Value, policy: &TranslationPolicy) -> Result<DecodedRequest>;

    /// Encodes a neutral request into provider JSON.
    fn encode_request(
        &self,
        request: &LlmRequest,
        policy: &TranslationPolicy,
    ) -> Result<EncodedRequest>;

    /// Decodes a response body into the neutral IR.
    fn decode_response(&self, body: &Value, policy: &TranslationPolicy) -> Result<DecodedResponse>;

    /// Encodes a neutral response into provider JSON.
    fn encode_response(
        &self,
        response: &AggLlmResponse,
        policy: &TranslationPolicy,
    ) -> Result<EncodedResponse>;
}

/// Extracts reported usage independently of generation success or content validity.
///
/// Only the provider's top-level usage object is inspected. Missing or invalid
/// counters remain unknown; the response is not accepted as a successful result.
pub fn response_usage(body: &Value, format: crate::WireFormat) -> crate::Usage {
    let usage = body.get("usage");
    match format {
        crate::WireFormat::OpenAiChat => openai_chat::decode_openai_usage(usage),
        crate::WireFormat::OpenAiResponses => responses::decode_responses_usage(usage),
        crate::WireFormat::AnthropicMessages => anthropic::decode_anthropic_usage(usage),
    }
}
