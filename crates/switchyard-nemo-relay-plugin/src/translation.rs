// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay_plugin::LlmRequest as RelayRequest;
use serde_json::Value as Json;
use switchyard_protocol::{AggLlmResponse, LlmRequest, WireFormat};
use switchyard_translation::{
    DeterministicIdPolicy, DiagnosticSeverity, LossyConversionPolicy, PreservationPolicy,
    TargetCapabilities, TranslationDiagnostic, TranslationEngine, TranslationPolicy,
    UnknownFieldPolicy,
};

pub(crate) fn decode_request(
    engine: &TranslationEngine,
    protocol: WireFormat,
    request: &RelayRequest,
) -> Result<LlmRequest, String> {
    let output = engine
        .decode_request(protocol, &request.content, &policy())
        .map_err(error)?;
    safe(&output.diagnostics)?;
    Ok(output.request)
}

pub(crate) fn validate_target_request(
    engine: &TranslationEngine,
    protocol: WireFormat,
    request: &LlmRequest,
) -> Result<(), String> {
    let output = engine
        .encode_request(protocol, request, &request_policy(protocol))
        .map_err(error)?;
    safe(&output.diagnostics)
}

pub(crate) fn encode_response(
    engine: &TranslationEngine,
    protocol: WireFormat,
    response: &AggLlmResponse,
) -> Result<Json, String> {
    let output = engine
        .encode_response(protocol, response, &policy())
        .map_err(error)?;
    safe(&output.diagnostics)?;
    Ok(output.body)
}

fn policy() -> TranslationPolicy {
    TranslationPolicy {
        unknown_field_policy: UnknownFieldPolicy::Preserve,
        lossy_conversion_policy: LossyConversionPolicy::Reject,
        deterministic_ids: DeterministicIdPolicy::GenerateStable {
            prefix: "relay".into(),
        },
        preservation: PreservationPolicy::InMemory,
        target_capabilities: TargetCapabilities::default(),
    }
}

fn request_policy(protocol: WireFormat) -> TranslationPolicy {
    let mut policy = policy();
    if protocol == WireFormat::AnthropicMessages {
        policy
            .target_capabilities
            .supports_json_schema_response_format = Some(false);
    }
    policy
}

fn safe(diagnostics: &[TranslationDiagnostic]) -> Result<(), String> {
    let unsafe_diagnostics = diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.severity != DiagnosticSeverity::Info)
        .collect::<Vec<_>>();
    if unsafe_diagnostics.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "Switchyard translation was not lossless: {unsafe_diagnostics:?}"
        ))
    }
}

fn error(error: switchyard_translation::TranslationError) -> String {
    format!("Switchyard translation failed: {error}")
}

#[cfg(test)]
mod tests {
    use serde_json::{Map, json};

    use super::*;

    #[test]
    fn same_protocol_request_preserves_unknown_fields() {
        let request = RelayRequest {
            headers: Map::new(),
            content: json!({
                "model": "route",
                "messages": [{"role": "user", "content": "hello"}],
                "provider_extension": {"exact": true}
            }),
        };
        let engine = TranslationEngine::default();
        let decoded = decode_request(&engine, WireFormat::OpenAiChat, &request).unwrap();
        validate_target_request(&engine, WireFormat::OpenAiChat, &decoded).unwrap();
        assert_eq!(
            decoded
                .preservation
                .requests
                .get(&WireFormat::OpenAiChat.into())
                .and_then(|body| body.get("provider_extension")),
            Some(&json!({"exact": true}))
        );
    }
}
