// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Buffered HTTP-client translation with diagnostic metrics and logs.

use std::sync::LazyLock;

use serde_json::Value;
use switchyard_protocol::{AggLlmResponse, LlmRequest};
use switchyard_translation::{Result, TranslationEngine, TranslationPolicy, WireFormat};

use crate::metrics;

static ENGINE: LazyLock<TranslationEngine> = LazyLock::new(TranslationEngine::default);
static POLICY: LazyLock<TranslationPolicy> = LazyLock::new(TranslationPolicy::default);

pub(crate) fn encode_request(request: &LlmRequest, format: WireFormat) -> Result<Value> {
    let encoded = ENGINE.encode_request(format, request, &POLICY)?;
    metrics::record_translation_diagnostics(&encoded.diagnostics, "request_encode", format);
    Ok(encoded.body)
}

pub(crate) fn decode_response(body: &Value, format: WireFormat) -> Result<AggLlmResponse> {
    let decoded = ENGINE.decode_response(format, body, &POLICY)?;
    metrics::record_translation_diagnostics(&decoded.diagnostics, "response_decode", format);
    Ok(decoded.response)
}
