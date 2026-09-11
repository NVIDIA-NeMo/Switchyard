// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Buffered translation with runtime diagnostics for server HTTP boundaries.

use std::sync::LazyLock;

use serde_json::Value;
use switchyard_protocol::{AggLlmResponse, LlmRequest, ProviderExtensions};
use switchyard_translation::{Result, TranslationEngine, TranslationPolicy, WireFormat};

use crate::metrics;

static ENGINE: LazyLock<TranslationEngine> = LazyLock::new(TranslationEngine::default);
static POLICY: LazyLock<TranslationPolicy> = LazyLock::new(TranslationPolicy::default);

pub(crate) fn decode_request(format: WireFormat, body: &Value) -> Result<LlmRequest> {
    let decoded = ENGINE.decode_request(format, body, &POLICY)?;
    metrics::record_translation_diagnostics(&decoded.diagnostics, "request_decode", format);
    Ok(decoded.request)
}

pub(crate) fn encode_response(
    response: &AggLlmResponse,
    format: WireFormat,
    served_model: Option<&str>,
    request_extensions: &ProviderExtensions,
) -> Result<Value> {
    let mut encoded =
        ENGINE.encode_response_with_extensions(format, response, request_extensions, &POLICY)?;
    metrics::record_translation_diagnostics(&encoded.diagnostics, "response_encode", format);
    if let (Some(model), Value::Object(body)) = (served_model, &mut encoded.body) {
        body.insert("model".to_string(), Value::String(model.to_string()));
    }
    Ok(encoded.body)
}
