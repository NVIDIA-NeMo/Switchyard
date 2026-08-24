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
