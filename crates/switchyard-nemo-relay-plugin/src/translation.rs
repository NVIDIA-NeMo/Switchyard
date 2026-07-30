// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay_plugin::LlmRequest as RelayRequest;
use serde_json::{Map, Value as Json};
use switchyard_protocol::{AggLlmResponse, LlmRequest};
use switchyard_translation::{
    DeterministicIdPolicy, DiagnosticSeverity, LossyConversionPolicy, PreservationPolicy,
    StreamTranslationState, TargetCapabilities, TranslationDiagnostic, TranslationEngine,
    TranslationPolicy, UnknownFieldPolicy,
};

use crate::config::WireProtocol;

pub fn decode_request(
    engine: &TranslationEngine,
    protocol: WireProtocol,
    request: &RelayRequest,
) -> Result<LlmRequest, String> {
    let output = engine
        .decode_request(protocol.wire_format(), &request.content, &policy())
        .map_err(error)?;
    safe(&output.diagnostics)?;
    Ok(output.request)
}

pub fn encode_request(
    engine: &TranslationEngine,
    protocol: WireProtocol,
    request: &LlmRequest,
    headers: Map<String, Json>,
) -> Result<RelayRequest, String> {
    let output = engine
        .encode_request(protocol.wire_format(), request, &request_policy(protocol))
        .map_err(error)?;
    safe(&output.diagnostics)?;
    Ok(RelayRequest {
        headers,
        content: output.body,
    })
}

pub fn decode_response(
    engine: &TranslationEngine,
    protocol: WireProtocol,
    response: &Json,
) -> Result<AggLlmResponse, String> {
    let output = engine
        .decode_response(protocol.wire_format(), response, &policy())
        .map_err(error)?;
    safe(&output.diagnostics)?;
    Ok(output.response)
}

pub fn encode_response(
    engine: &TranslationEngine,
    protocol: WireProtocol,
    response: &AggLlmResponse,
) -> Result<Json, String> {
    let output = engine
        .encode_response(protocol.wire_format(), response, &policy())
        .map_err(error)?;
    safe(&output.diagnostics)?;
    Ok(output.body)
}

pub fn decode_stream_event(
    engine: &TranslationEngine,
    state: &mut StreamTranslationState,
    protocol: WireProtocol,
    event: Json,
) -> Result<switchyard_protocol::LlmResponseStreamEvent, String> {
    engine
        .decode_stream_event(state, protocol.wire_format(), event)
        .map_err(error)
}

pub fn encode_stream_event(
    engine: &TranslationEngine,
    state: &mut StreamTranslationState,
    protocol: WireProtocol,
    event: switchyard_protocol::LlmResponseStreamEvent,
) -> Result<Vec<Json>, String> {
    engine
        .encode_stream_event(state, protocol.wire_format(), event)
        .map_err(error)
}

pub fn finish_stream(
    engine: &TranslationEngine,
    state: &mut StreamTranslationState,
    protocol: WireProtocol,
) -> Result<Vec<Json>, String> {
    engine
        .finish_stream(state, protocol.wire_format())
        .map_err(error)
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

fn request_policy(protocol: WireProtocol) -> TranslationPolicy {
    let mut policy = policy();
    if protocol == WireProtocol::AnthropicMessages {
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
    use super::*;
    use serde_json::json;

    #[test]
    fn plugin_replays_same_protocol_provider_extensions_exactly() {
        let cases = [
            (
                WireProtocol::OpenaiChat,
                json!({
                    "id": "chatcmpl-test",
                    "object": "chat.completion.chunk",
                    "model": "gpt-test",
                    "system_fingerprint": "fp_provider_specific",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": "Hi"},
                        "finish_reason": null
                    }]
                }),
            ),
            (
                WireProtocol::OpenaiResponses,
                json!({
                    "type": "response.output_text.delta",
                    "item_id": "item-1",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "Hi",
                    "sequence_number": 2,
                    "provider_extension": {"exact": true}
                }),
            ),
            (
                WireProtocol::AnthropicMessages,
                json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Hi"},
                    "provider_extension": {"exact": true}
                }),
            ),
        ];

        let engine = TranslationEngine::default();
        for (protocol, raw) in cases {
            let mut state =
                StreamTranslationState::new(protocol.wire_format(), protocol.wire_format());
            let event = decode_stream_event(&engine, &mut state, protocol, raw.clone()).unwrap();
            assert_eq!(
                encode_stream_event(&engine, &mut state, protocol, event).unwrap(),
                vec![raw]
            );
        }
    }

    #[test]
    fn cross_protocol_streams_use_normalized_content_not_raw_extensions() {
        let engine = TranslationEngine::default();
        let mut state = StreamTranslationState::new(
            WireProtocol::OpenaiChat.wire_format(),
            WireProtocol::AnthropicMessages.wire_format(),
        );
        let event = decode_stream_event(
            &engine,
            &mut state,
            WireProtocol::OpenaiChat,
            json!({
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "model": "gpt-test",
                "system_fingerprint": "not-portable",
                "choices": [{
                    "index": 0,
                    "delta": {"content": "Hi"},
                    "finish_reason": null
                }]
            }),
        )
        .unwrap();
        let translated =
            encode_stream_event(&engine, &mut state, WireProtocol::AnthropicMessages, event)
                .unwrap();
        assert!(translated
            .iter()
            .any(|event| { event.pointer("/delta/text").and_then(Json::as_str) == Some("Hi") }));
        assert!(translated
            .iter()
            .all(|event| event.get("system_fingerprint").is_none()));
    }
}
