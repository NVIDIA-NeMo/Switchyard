// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for custom translation and stream codec extension points.

use pretty_assertions::assert_eq;
use serde_json::{json, Value};
use switchyard_translation::codecs::{
    DecodedRequest, DecodedResponse, EncodedRequest, EncodedResponse, FormatCodec,
};
use switchyard_translation::util::{
    capture_request_preservation, capture_response_preservation, exact_preserved_request,
    exact_preserved_response,
};
use switchyard_translation::{
    ConversationStreamEvent, FormatId, FormatRegistry, LlmRequest, LlmResponse,
    LossyConversionPolicy, Message, PreservationPolicy, ResponseOutput, Role, StreamCodec,
    StreamCodecRegistry, StreamTranslationState, TargetCapabilities, TranslationEngine,
    TranslationPolicy, Usage, WireFormat,
};

type TestResult = std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>;

// Verifies a custom buffered format can register beside built-in formats.
#[test]
fn custom_format_id_can_be_registered_without_touching_builtin_wire_formats() -> TestResult {
    let mut registry = FormatRegistry::with_builtins();
    registry.register(MinimalCustomCodec);
    let engine = TranslationEngine::new(registry);
    let policy = TranslationPolicy {
        preservation: PreservationPolicy::Embed,
        ..TranslationPolicy::default()
    };
    let custom_format = FormatId::new("custom_minimal");
    let original = json!({
        "model": "custom-model",
        "prompt": "translate me",
        "metadata": {"trace": "custom-request"},
        "vendor_only": {"must": "roundtrip"}
    });

    let chat = engine
        .translate_request(
            custom_format.clone(),
            WireFormat::OpenAiChat,
            &original,
            &policy,
        )?
        .body;
    assert_eq!(chat["messages"][0]["content"], "translate me");

    let roundtripped = engine
        .translate_request(WireFormat::OpenAiChat, custom_format, &chat, &policy)?
        .body;

    assert_eq!(roundtripped, original);
    Ok(())
}

// Verifies a custom stream codec can decode into built-in OpenAI Chat chunks.
#[test]
fn custom_stream_codec_can_participate_in_registered_stream_translation() -> TestResult {
    let mut registry = StreamCodecRegistry::with_builtins();
    registry.register(CustomStreamCodec);
    let engine = TranslationEngine::with_registries(FormatRegistry::with_builtins(), registry);
    let mut state = StreamTranslationState::new("custom_stream", WireFormat::OpenAiChat);

    let start = engine.translate_event(
        &mut state,
        "custom_stream",
        WireFormat::OpenAiChat,
        &json!({"kind": "start", "id": "custom_1", "model": "custom-model"}),
    )?;
    assert!(start.is_empty());

    let delta = engine.translate_event(
        &mut state,
        "custom_stream",
        WireFormat::OpenAiChat,
        &json!({"kind": "delta", "text": "hello"}),
    )?;

    assert_eq!(delta[0]["object"], "chat.completion.chunk");
    assert_eq!(delta[0]["model"], "custom-model");
    assert_eq!(delta[0]["choices"][0]["delta"]["content"], "hello");
    Ok(())
}

// Verifies target capability policy can reject unsupported request features.
#[test]
fn capability_profile_can_fail_fast_when_target_cannot_accept_request_features() {
    let engine = TranslationEngine::default();
    let policy = TranslationPolicy {
        lossy_conversion_policy: LossyConversionPolicy::Reject,
        target_capabilities: TargetCapabilities {
            supports_tools: Some(false),
            ..TargetCapabilities::default()
        },
        ..TranslationPolicy::default()
    };
    let body = json!({
        "model": "claude",
        "messages": [{"role": "user", "content": "look this up"}],
        "tools": [{"name": "lookup", "input_schema": {"type": "object"}}]
    });

    let error = match engine.translate_request(
        WireFormat::AnthropicMessages,
        WireFormat::OpenAiChat,
        &body,
        &policy,
    ) {
        Ok(_) => panic!("tools should be rejected by the target capability profile"),
        Err(error) => error,
    };

    assert!(error
        .to_string()
        .contains("target format/profile does not support tools"));
}

// Minimal custom buffered codec used to exercise the registry extension point.
struct MinimalCustomCodec;

impl FormatCodec for MinimalCustomCodec {
    fn format(&self) -> FormatId {
        FormatId::new("custom_minimal")
    }

    fn decode_request(
        &self,
        body: &Value,
        policy: &TranslationPolicy,
    ) -> switchyard_translation::Result<DecodedRequest> {
        Ok(DecodedRequest {
            request: LlmRequest {
                model: body
                    .get("model")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                messages: vec![Message::text(
                    Role::User,
                    body.get("prompt")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                )],
                preservation: capture_request_preservation(self.format(), body, policy),
                ..LlmRequest::default()
            },
            diagnostics: Vec::new(),
        })
    }

    fn encode_request(
        &self,
        request: &LlmRequest,
        policy: &TranslationPolicy,
    ) -> switchyard_translation::Result<EncodedRequest> {
        if let Some(body) = exact_preserved_request(&request.preservation, self.format(), policy) {
            return Ok(EncodedRequest {
                body,
                diagnostics: Vec::new(),
            });
        }
        Ok(EncodedRequest {
            body: json!({
                "model": request.model,
                "prompt": request
                    .messages
                    .first()
                    .and_then(|message| message.text_content("\n"))
                    .unwrap_or_default(),
            }),
            diagnostics: Vec::new(),
        })
    }

    fn decode_response(
        &self,
        body: &Value,
        policy: &TranslationPolicy,
    ) -> switchyard_translation::Result<DecodedResponse> {
        Ok(DecodedResponse {
            response: LlmResponse {
                id: body
                    .get("id")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                model: body
                    .get("model")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                outputs: vec![ResponseOutput {
                    role: Role::Assistant,
                    content: vec![switchyard_translation::ContentBlock::Text {
                        text: body
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                    }],
                    stop_reason: None,
                }],
                usage: Usage::default(),
                preservation: capture_response_preservation(self.format(), body, policy),
                ..LlmResponse::default()
            },
            diagnostics: Vec::new(),
        })
    }

    fn encode_response(
        &self,
        response: &LlmResponse,
        policy: &TranslationPolicy,
    ) -> switchyard_translation::Result<EncodedResponse> {
        if let Some(body) = exact_preserved_response(&response.preservation, self.format(), policy)
        {
            return Ok(EncodedResponse {
                body,
                diagnostics: Vec::new(),
            });
        }
        Ok(EncodedResponse {
            body: json!({
                "id": response.id,
                "model": response.model,
                "text": response
                    .first_output()
                    .and_then(|output| {
                        output.content.iter().find_map(|block| match block {
                            switchyard_translation::ContentBlock::Text { text } => {
                                Some(text.clone())
                            }
                            _ => None,
                        })
                    })
                    .unwrap_or_default(),
            }),
            diagnostics: Vec::new(),
        })
    }
}

// Minimal custom stream codec used to exercise streaming extension points.
struct CustomStreamCodec;

impl StreamCodec for CustomStreamCodec {
    fn format(&self) -> FormatId {
        FormatId::new("custom_stream")
    }

    fn decode_event(
        &self,
        _state: &mut StreamTranslationState,
        event: &Value,
    ) -> Vec<ConversationStreamEvent> {
        match event.get("kind").and_then(Value::as_str) {
            Some("start") => vec![ConversationStreamEvent::MessageStart {
                id: event
                    .get("id")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                model: event
                    .get("model")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
            }],
            Some("delta") => event
                .get("text")
                .and_then(Value::as_str)
                .map(|text| {
                    vec![ConversationStreamEvent::TextDelta {
                        index: 0,
                        text: text.to_string(),
                    }]
                })
                .unwrap_or_default(),
            _ => Vec::new(),
        }
    }

    fn encode_event(
        &self,
        _state: &mut StreamTranslationState,
        event: ConversationStreamEvent,
    ) -> Vec<Value> {
        match event {
            ConversationStreamEvent::TextDelta { text, .. } => {
                vec![json!({"kind": "delta", "text": text})]
            }
            _ => Vec::new(),
        }
    }

    fn finish(&self, _state: &mut StreamTranslationState) -> Vec<Value> {
        Vec::new()
    }
}
