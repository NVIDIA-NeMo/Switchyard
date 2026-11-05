// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for custom translation and stream codec extension points.

use futures::executor::block_on;
use futures::{StreamExt, TryStreamExt, stream};
use pretty_assertions::assert_eq;
use serde_json::{Value, json};
use switchyard_protocol::LlmClientError;
use switchyard_translation::codecs::{
    DecodedRequest, DecodedResponse, EncodedRequest, EncodedResponse, FormatCodec,
};
use switchyard_translation::util::{
    capture_request_preservation, capture_response_preservation, exact_preserved_request,
    exact_preserved_response,
};
use switchyard_translation::{
    AggLlmResponse, FormatId, FormatRegistry, LlmRequest, LlmResponseChunk, LossyConversionPolicy,
    Message, PreservationPolicy, ResponseOutput, Role, StreamCodec, StreamCodecRegistry,
    StreamTranslationState, TargetCapabilities, TranslationEngine, TranslationPolicy, Usage,
    WireFormat,
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

#[test]
fn custom_stream_codec_replays_preserved_same_format_event() -> TestResult {
    let mut registry = StreamCodecRegistry::new();
    registry.register(CustomStreamCodec);
    let engine = TranslationEngine::with_registries(FormatRegistry::new(), registry);
    let format = FormatId::new("custom_stream");
    let mut state = StreamTranslationState::new(format.clone(), format.clone());
    let event = json!({
        "kind": "delta",
        "text": "hello",
        "vendor_only": {"must": "survive"}
    });

    let preserved = engine.decode_stream_event(&mut state, format.clone(), event.clone())?;
    let replayed = engine.encode_stream_event(&mut state, format, preserved)?;

    assert_eq!(replayed, vec![event]);
    Ok(())
}

#[test]
fn custom_whole_streams_use_registered_codecs_and_outlive_the_engine() -> TestResult {
    let mut registry = StreamCodecRegistry::new();
    registry.register(CustomStreamCodec);
    let engine = TranslationEngine::with_registries(FormatRegistry::new(), registry);
    let format = FormatId::new("custom_stream");
    let bytes = stream::iter([Ok(
        b"data: {\"kind\":\"delta\",\"text\":\"hello\",\"vendor_only\":true}\n\ndata: {\"kind\":\"done\"}\n\n"
            .to_vec(),
    )]);
    let decoded = engine.decode_stream(format.clone(), bytes)?;
    let replayed = engine.encode_stream_with_extensions(
        format.clone(),
        decoded,
        Some("replayed-model".into()),
        &Default::default(),
    )?;
    let generated = engine.encode_stream(
        format,
        stream::iter([Ok(LlmResponseChunk::TextDelta {
            index: 0,
            text: "generated".into(),
        }
        .into())])
        .boxed(),
        Some("generated-model".into()),
    )?;
    drop(engine);

    assert_eq!(
        block_on(replayed.try_collect::<Vec<_>>())?,
        vec![
            json!({"kind": "delta", "text": "hello", "vendor_only": true, "backend": "replayed-model"}),
            json!({"kind": "done", "backend": "replayed-model"}),
        ]
    );
    assert_eq!(
        block_on(generated.try_collect::<Vec<_>>())?,
        vec![
            json!({"kind": "delta", "text": "generated", "backend": "generated-model"}),
            json!({"kind": "done", "backend": "generated-model"}),
        ]
    );
    Ok(())
}

#[test]
fn custom_stream_completion_uses_the_codec_rules() -> TestResult {
    let mut registry = StreamCodecRegistry::new();
    registry.register(CustomStreamCodec);
    let engine = TranslationEngine::with_registries(FormatRegistry::new(), registry);
    let cases = [
        ("EOF after text", "", false),
        ("EOF after an empty frame", "\n\n", false),
        ("SSE marker alone", "data: [DONE]\n\n", false),
        ("trailing SSE marker alone", "data: [DONE]", false),
        ("terminal JSON event", "data: {\"kind\":\"done\"}\n\n", true),
        (
            "trailing terminal JSON event",
            "data: {\"kind\":\"done\"}",
            true,
        ),
        (
            "terminal JSON event then SSE marker",
            "data: {\"kind\":\"done\"}\n\ndata: [DONE]\n\n",
            true,
        ),
        (
            "terminal JSON event then trailing SSE marker",
            "data: {\"kind\":\"done\"}\n\ndata: [DONE]",
            true,
        ),
    ];
    for (name, tail, is_complete) in cases {
        let bytes =
            format!("data: {{\"kind\":\"delta\",\"text\":\"hello\"}}\n\n{tail}").into_bytes();
        let decoded = engine.decode_stream("custom_stream", stream::iter([Ok(bytes)]))?;
        let mut events = block_on(decoded.collect::<Vec<_>>()).into_iter();
        let first = events.next().expect(name)?;
        assert_eq!(
            first.normalized(),
            &[LlmResponseChunk::TextDelta {
                index: 0,
                text: "hello".into(),
            }],
            "{name}"
        );
        let remaining = events.collect::<Result<Vec<_>, _>>();
        if is_complete {
            assert!(remaining.is_ok(), "{name}: {remaining:?}");
        } else {
            assert!(
                matches!(
                    &remaining,
                    Err(LlmClientError::ResponseTranslation(message))
                        if message == "custom_stream stream ended before a terminal event"
                ),
                "{name}: {remaining:?}"
            );
        }
    }
    Ok(())
}

#[test]
fn whole_stream_methods_report_missing_codecs_at_setup() {
    let engine =
        TranslationEngine::with_registries(FormatRegistry::new(), StreamCodecRegistry::new());
    assert!(matches!(
        engine.decode_stream("missing", stream::empty()),
        Err(LlmClientError::ResponseTranslation(message))
            if message == "no stream codec registered for missing"
    ));
    assert!(matches!(
        engine.encode_stream("missing", stream::empty().boxed(), None),
        Err(LlmClientError::Configuration { message })
            if message == "no stream codec registered for missing"
    ));
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

    assert!(
        error
            .to_string()
            .contains("target format/profile does not support tools")
    );
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
            response: AggLlmResponse {
                id: body
                    .get("id")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                model: body
                    .get("model")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                outputs: vec![ResponseOutput {
                    url_citations: Vec::new(),
                    role: Role::Assistant,
                    content: vec![switchyard_translation::ContentBlock::Text {
                        responses: None,
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
                ..AggLlmResponse::default()
            },
            diagnostics: Vec::new(),
        })
    }

    fn encode_response(
        &self,
        response: &AggLlmResponse,
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
                            switchyard_translation::ContentBlock::Text { text, .. } => {
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

    fn is_terminal_event(&self, event: &Value) -> bool {
        event.get("kind").and_then(Value::as_str) == Some("done")
    }

    fn sse_done_is_terminal(&self) -> bool {
        false
    }

    fn stamp_streamed_response_model(&self, event: &mut Value, served_model: &str) {
        event["backend"] = Value::String(served_model.to_string());
    }

    fn decode_event(
        &self,
        _state: &mut StreamTranslationState,
        event: &Value,
    ) -> Vec<LlmResponseChunk> {
        match event.get("kind").and_then(Value::as_str) {
            Some("start") => vec![LlmResponseChunk::MessageStart {
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
                    vec![LlmResponseChunk::TextDelta {
                        index: 0,
                        text: text.to_string(),
                    }]
                })
                .unwrap_or_default(),
            Some("done") => vec![LlmResponseChunk::MessageStop { reason: None }],
            _ => Vec::new(),
        }
    }

    fn encode_event(
        &self,
        state: &mut StreamTranslationState,
        event: LlmResponseChunk,
    ) -> Vec<Value> {
        match event {
            LlmResponseChunk::TextDelta { text, .. } => {
                vec![json!({"kind": "delta", "text": text})]
            }
            LlmResponseChunk::MessageStop { .. } => self.finish(state),
            _ => Vec::new(),
        }
    }

    fn finish(&self, state: &mut StreamTranslationState) -> Vec<Value> {
        if state.finished {
            Vec::new()
        } else {
            state.finished = true;
            vec![json!({"kind": "done"})]
        }
    }
}
