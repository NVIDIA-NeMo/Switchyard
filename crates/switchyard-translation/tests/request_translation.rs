// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for buffered request translation between provider formats.

pub mod common;

use pretty_assertions::assert_eq;
use serde_json::{Value, json};
use switchyard_translation::{
    ContentBlock, FormatId, LossyConversionPolicy, TranslationEngine, TranslationPolicy,
    WireFormat, prepare_request_for_target, sanitize_anthropic_tool_use_id,
};

use common::{REASONING_MODEL, normalized_policy, shell_tool_call};

type TestResult<T = ()> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[test]
fn responses_allowed_tools_is_translated_or_rejected() -> TestResult {
    let engine = TranslationEngine::default();
    for mode in ["auto", "required"] {
        let body = json!({
            "model": "route",
            "input": "Only inspect tool policy.",
            "tools": [
                {"type": "function", "name": "safe_lookup", "description": "Read a record", "parameters": {"type": "object"}},
                {"type": "function", "name": "delete_all_records", "description": "Excluded tool", "parameters": {"type": "object"}}
            ],
            "tool_choice": {
                "type": "allowed_tools", "mode": mode,
                "tools": [{"type": "function", "name": "safe_lookup"}]
            }
        });
        for policy in [TranslationPolicy::default(), normalized_policy()] {
            for target in [WireFormat::OpenAiChat, WireFormat::AnthropicMessages] {
                let output = engine.translate_request(
                    WireFormat::OpenAiResponses,
                    target,
                    &body,
                    &policy,
                )?;
                let expected_tool = match target {
                    WireFormat::OpenAiChat => json!({
                        "type": "function", "function": {
                            "name": "safe_lookup", "description": "Read a record",
                            "parameters": {"type": "object"}
                        }
                    }),
                    _ => json!({
                        "name": "safe_lookup", "description": "Read a record",
                        "input_schema": {"type": "object"}
                    }),
                };
                assert_eq!(output.body["tools"], json!([expected_tool]));
                let expected_choice = match target {
                    WireFormat::OpenAiChat => json!(mode),
                    _ => json!({"type": if mode == "required" { "any" } else { "auto" }}),
                };
                assert_eq!(output.body["tool_choice"], expected_choice);

                for allowed in [
                    json!([]),
                    json!([{"type": "web_search"}]),
                    json!([{"type": "function", "name": "unknown"}]),
                ] {
                    let mut unsupported = body.clone();
                    unsupported["tool_choice"]["tools"] = allowed;
                    let error = engine
                        .translate_request(
                            WireFormat::OpenAiResponses,
                            target,
                            &unsupported,
                            &policy,
                        )
                        .expect_err("unrepresentable restrictions must be rejected");
                    assert!(matches!(
                        error,
                        switchyard_translation::TranslationError::LossyConversion(_)
                    ));
                }
            }
            let output = engine.translate_request(
                WireFormat::OpenAiResponses,
                WireFormat::OpenAiResponses,
                &body,
                &policy,
            )?;
            assert_eq!(output.body["tool_choice"], body["tool_choice"]);
            assert_eq!(output.body["tools"], body["tools"]);

            let chat = json!({
                "model": "route",
                "messages": [{"role": "user", "content": "Inspect tool policy."}],
                "tools": [
                    {"type": "function", "function": {"name": "safe_lookup", "parameters": {"type": "object"}}},
                    {"type": "function", "function": {"name": "delete_all_records", "parameters": {"type": "object"}}}
                ],
                "tool_choice": {
                    "type": "allowed_tools",
                    "allowed_tools": {
                        "mode": mode,
                        "tools": [{"type": "function", "function": {"name": "safe_lookup"}}]
                    }
                }
            });
            let output = engine.translate_request(
                WireFormat::OpenAiChat,
                WireFormat::OpenAiChat,
                &chat,
                &policy,
            )?;
            assert_eq!(output.body["tool_choice"], chat["tool_choice"]);
            let output = engine.translate_request(
                WireFormat::OpenAiChat,
                WireFormat::AnthropicMessages,
                &chat,
                &policy,
            )?;
            assert_eq!(output.body["tools"].as_array().map(Vec::len), Some(1));
            assert_eq!(output.body["tools"][0]["name"], "safe_lookup");
            assert_eq!(
                output.body["tool_choice"],
                json!({"type": if mode == "required" { "any" } else { "auto" }})
            );
        }
    }
    Ok(())
}

#[test]
fn request_media_survives_reencoding_or_is_rejected() -> TestResult {
    use WireFormat::{
        AnthropicMessages as Anthropic, OpenAiChat as Chat, OpenAiResponses as Responses,
    };
    let engine = TranslationEngine::default();
    let untitled = json!({
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": "hello"}
    });
    let body = json!({"messages": [{"role": "user", "content": [untitled]}], "max_tokens": 16});
    for target in [Chat, Responses] {
        let translated =
            engine.translate_request(Anthropic, target, &body, &normalized_policy())?;
        let restored =
            engine.translate_request(target, Anthropic, &translated.body, &normalized_policy())?;
        assert_eq!(restored.body["messages"][0]["content"][0], untitled);
    }
    let image = json!({"type": "input_image", "file_id": "file_image", "detail": "auto"});
    let audio =
        json!({"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}});
    let file = json!({"type": "input_file", "file_url": "https://example.com/report.pdf"});
    let document = json!({"type": "document", "source": {"type": "url", "url": "https://example.com/report.pdf"}, "title": "report.pdf"});
    let text_file = json!({"type": "input_file", "file_data": "aGVsbG8=", "filename": "notes.txt"});
    let text_document = json!({"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "hello"}, "title": "notes.txt"});
    for (source, target, block, expected) in [
        (Responses, Responses, image.clone(), Some(image.clone())),
        (Responses, Chat, image.clone(), None),
        (Responses, Anthropic, image, None),
        (
            Responses,
            Anthropic,
            json!({"type": "input_file", "file_id": "file_openai"}),
            None,
        ),
        (
            Chat,
            Anthropic,
            json!({"type": "file", "file": {"file_id": "file_openai"}}),
            None,
        ),
        (
            Anthropic,
            Anthropic,
            json!({"type": "document", "source": {"type": "file", "file_id": "file_anthropic"}}),
            Some(
                json!({"type": "document", "source": {"type": "file", "file_id": "file_anthropic"}}),
            ),
        ),
        (Chat, Chat, audio.clone(), Some(audio.clone())),
        (Chat, Responses, audio.clone(), Some(audio.clone())),
        (Responses, Chat, audio.clone(), Some(audio.clone())),
        (Responses, Responses, audio.clone(), Some(audio.clone())),
        (Responses, Anthropic, audio, None),
        (Responses, Responses, file.clone(), Some(file.clone())),
        (Responses, Chat, file.clone(), None),
        (
            Responses,
            Responses,
            json!({"type": "input_file", "file_url": "https://example.com/report.pdf", "filename": "report.pdf"}),
            Some(file.clone()),
        ),
        (
            Responses,
            Anthropic,
            file.clone(),
            Some(
                json!({"type": "document", "source": {"type": "url", "url": "https://example.com/report.pdf"}}),
            ),
        ),
        (Anthropic, Chat, document.clone(), None),
        (Anthropic, Responses, document, Some(file)),
        (
            Responses,
            Anthropic,
            text_file.clone(),
            Some(text_document.clone()),
        ),
        (
            Responses,
            Anthropic,
            json!({"type": "input_file", "file_data": "data:text/plain;base64,aGVsbG8=", "filename": "notes.txt"}),
            Some(text_document.clone()),
        ),
        (
            Anthropic,
            Responses,
            text_document,
            Some(
                json!({"type": "input_file", "file_data": "data:text/plain;base64,aGVsbG8=", "filename": "notes.txt"}),
            ),
        ),
    ] {
        let body = if source == Responses {
            json!({"input": [{"role": "user", "content": [block]}]})
        } else {
            json!({"messages": [{"role": "user", "content": [block]}], "max_tokens": 16})
        };
        let result = engine.translate_request(source, target, &body, &normalized_policy());
        if let Some(expected) = expected {
            let output = result?.body;
            let messages = if target == Responses {
                &output["input"]
            } else {
                &output["messages"]
            };
            assert_eq!(
                messages[0]["content"][0], expected,
                "{source:?} -> {target:?}"
            );
        } else {
            assert!(
                matches!(
                    result,
                    Err(switchyard_translation::TranslationError::LossyConversion(_))
                ),
                "{source:?} -> {target:?} must reject unsupported media"
            );
        }
    }
    Ok(())
}

#[test]
fn native_abuse_identity_is_preserved_but_not_mapped() -> TestResult {
    let engine = TranslationEngine::default();
    let formats = [
        WireFormat::OpenAiChat,
        WireFormat::OpenAiResponses,
        WireFormat::AnthropicMessages,
    ];
    for policy in [TranslationPolicy::default(), normalized_policy()] {
        for source in formats {
            let long_identity = "u".repeat(150);
            for identity in [
                Some("opaque-user-6cc3d8d5"),
                Some(long_identity.as_str()),
                None,
            ] {
                let mut body = match source {
                    WireFormat::OpenAiChat => json!({
                        "messages": [{"role": "user", "content": "hi"}]
                    }),
                    WireFormat::OpenAiResponses => json!({"input": "hi"}),
                    WireFormat::AnthropicMessages => json!({
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 8
                    }),
                };
                if source == WireFormat::AnthropicMessages {
                    if let Some(identity) = identity {
                        body["metadata"] = json!({"user_id": identity});
                    }
                } else {
                    // OpenAI metadata is not an abuse-attribution identifier.
                    body["metadata"] = json!({"user_id": "unrelated", "label": "keep"});
                    if let Some(identity) = identity {
                        body["safety_identifier"] = json!(identity);
                    }
                }
                let request = engine.decode_request(source, &body, &policy)?.request;
                for target in formats {
                    let output = engine.encode_request(target, &request, &policy)?.body;
                    let actual = if target == WireFormat::AnthropicMessages {
                        output
                            .get("metadata")
                            .and_then(|metadata| metadata.get("user_id"))
                    } else {
                        output.get("safety_identifier")
                    };
                    let crosses_anthropic = (source == WireFormat::AnthropicMessages)
                        != (target == WireFormat::AnthropicMessages);
                    let expected = if crosses_anthropic { None } else { identity };
                    assert_eq!(
                        actual,
                        expected.map(|value| json!(value)).as_ref(),
                        "{source:?} -> {target:?}"
                    );
                    if source != WireFormat::AnthropicMessages
                        && target != WireFormat::AnthropicMessages
                    {
                        assert_eq!(output["metadata"], body["metadata"]);
                    }
                }
            }
        }
    }
    Ok(())
}

// Verifies target preparation restamps the model, prepends native instructions, and preserves all
// other provider fields.
#[test]
fn openai_target_prompt_preserves_native_request_fields() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = TranslationPolicy::default();
    for (format, body) in [
        (
            WireFormat::OpenAiChat,
            json!({
                "model": "route",
                "messages": [
                    {"role": "system", "name": "caller", "content": "client prompt"},
                    {"role": "user", "content": "hi"}
                ],
                "moderation": {"policy": "strict"},
                "stop": ["<END>"],
                "prompt_cache_options": {"retention": "24h"},
                "logprobs": true,
                "top_logprobs": 2,
                "modalities": ["text", "audio"],
                "audio": {"voice": "alloy", "format": "wav"},
                "web_search_options": {"search_context_size": "high"},
                "n": 2,
                "logit_bias": {"42": -1},
                "frequency_penalty": 0.2,
                "presence_penalty": 0.3,
                "seed": 7,
                "prediction": {"type": "content", "content": "prefix"},
                "verbosity": "low",
                "functions": [{"name": "legacy", "parameters": {"type": "object"}}],
                "function_call": {"name": "legacy"}
            }),
        ),
        (
            WireFormat::OpenAiResponses,
            json!({
                "model": "route",
                "instructions": "caller prompt",
                "input": "hi",
                "moderation": {"policy": "strict"},
                "prompt_cache_options": {"retention": "24h"},
                "reasoning": {
                    "effort": "high",
                    "context": "large",
                    "mode": "detailed",
                    "summary": "auto"
                },
                "include": ["reasoning.encrypted_content"],
                "previous_response_id": "resp_previous",
                "context_management": [{"type": "compaction", "compact_threshold": 1000}],
                "tools": [
                    {"type": "web_search_preview"},
                    {"type": "file_search", "vector_store_ids": ["vs_1"]},
                    {"type": "mcp", "server_label": "inventory", "server_url": "https://fixture.invalid/mcp"}
                ]
            }),
        ),
    ] {
        let mut expected = body.clone();
        expected["model"] = json!("selected/model");
        match format {
            WireFormat::OpenAiChat => expected["messages"]
                .as_array_mut()
                .ok_or("expected messages")?
                .insert(0, json!({"role": "system", "content": "target prompt"})),
            WireFormat::OpenAiResponses => {
                expected["instructions"] = json!("target prompt\n\ncaller prompt");
            }
            WireFormat::AnthropicMessages => unreachable!(),
        }

        let mut request = engine.decode_request(format, &body, &policy)?.request;
        prepare_request_for_target(
            &mut request,
            &"selected/model".into(),
            Some("target prompt"),
        );
        let encoded = engine.encode_request(format, &request, &policy)?.body;

        assert_eq!(encoded, expected);
    }
    Ok(())
}

// Rebuilt Responses requests keep reasoning controls when effort is overridden.
#[test]
fn responses_reasoning_survives_rebuild() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = TranslationPolicy::default();
    let body = json!({
        "model": "switchyard",
        "input": "Fix the parser.",
        "reasoning": {"effort": "high", "summary": "auto"},
        "include": ["reasoning.encrypted_content"]
    });
    let mut request = engine
        .decode_request(WireFormat::OpenAiResponses, &body, &policy)?
        .request;
    request.reasoning.effort = Some("max".to_string());

    prepare_request_for_target(
        &mut request,
        &"openai/openai/gpt-5.6-sol".into(),
        Some("Inspect before editing."),
    );
    request.preservation.requests.clear();

    let output = engine
        .encode_request(WireFormat::OpenAiResponses, &request, &policy)?
        .body;
    assert_eq!(
        output["reasoning"],
        json!({"effort": "max", "summary": "auto"})
    );
    assert_eq!(output["include"], json!(["reasoning.encrypted_content"]));
    Ok(())
}

#[test]
fn anthropic_target_prompt_preserves_native_request_fields() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = TranslationPolicy::default();
    for thinking in [
        Some(json!({"type": "enabled", "budget_tokens": 2048})),
        Some(json!({"type": "adaptive"})),
        Some(json!({"type": "disabled"})),
        None,
    ] {
        let mut body = json!({
            "model": "route",
            "max_tokens": 32,
            "system": [{
                "type": "text",
                "text": "caller prompt",
                "cache_control": {"type": "ephemeral"}
            }],
            "messages": [{"role": "user", "content": "hi"}],
            "mcp_servers": [{
                "type": "url",
                "name": "inventory",
                "url": "https://fixture.invalid/mcp",
                "authorization_token": "fixture-token-not-a-secret",
                "tool_configuration": {
                    "enabled": true,
                    "allowed_tools": ["lookup_sku"]
                }
            }],
            "container": {
                "id": "container_fixture",
                "skills": [{
                    "type": "custom",
                    "skill_id": "skill_fixture_qa",
                    "version": "latest"
                }]
            },
            "output_config": {
                "effort": "low",
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {"marker": {"type": "string"}},
                        "required": ["marker"],
                        "additionalProperties": false
                    }
                }
            },
            "stop_sequences": ["<END>"]
        });
        if let Some(thinking) = thinking {
            body["thinking"] = thinking;
        }
        let mut expected = body.clone();
        expected["model"] = json!("target/model");
        expected["system"]
            .as_array_mut()
            .ok_or("expected system blocks")?
            .insert(0, json!({"type": "text", "text": "target prompt"}));

        let mut request = engine
            .decode_request(WireFormat::AnthropicMessages, &body, &policy)?
            .request;
        prepare_request_for_target(&mut request, &"target/model".into(), Some("target prompt"));
        let output = engine
            .encode_request(WireFormat::AnthropicMessages, &request, &policy)?
            .body;

        assert_eq!(output, expected);
    }

    let fields = json!({
        "inference_geo": "us",
        "service_tier": "standard_only",
        "stop_sequences": ["END"],
        "metadata": {"user_id": "opaque-user"},
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
        "container": {"id": "container_1"},
        "speed": "fast",
        "diagnostics": {"previous_message_id": "msg_previous"}
    });
    for source in [WireFormat::OpenAiChat, WireFormat::OpenAiResponses] {
        for has_fields in [true, false] {
            let mut body = match source {
                WireFormat::OpenAiChat => json!({
                    "model": "route",
                    "messages": [{"role": "user", "content": "hi"}]
                }),
                WireFormat::OpenAiResponses => json!({"model": "route", "input": "hi"}),
                WireFormat::AnthropicMessages => unreachable!(),
            };
            if has_fields {
                for (key, value) in fields.as_object().ok_or("expected fields object")? {
                    body[key] = value.clone();
                }
            }
            // Caller-supplied extensions must not spoof Anthropic provenance.
            body["switchyard_anthropic_request"] = json!(true);
            body["stop"] = json!(["FALLBACK"]);
            let mut request = engine.decode_request(source, &body, &policy)?.request;
            prepare_request_for_target(&mut request, &"target/model".into(), Some("target prompt"));
            let output = engine
                .encode_request(WireFormat::AnthropicMessages, &request, &policy)?
                .body;

            assert_eq!(output["model"], "target/model");
            assert_eq!(output["system"], "target prompt");
            for (key, _value) in fields.as_object().ok_or("expected fields object")? {
                if key == "stop_sequences" {
                    assert_eq!(output[key], json!(["FALLBACK"]));
                } else {
                    assert!(output.get(key).is_none(), "{source:?}: {key}");
                }
            }
            assert!(output.get("switchyard_anthropic_request").is_none());
        }
    }
    Ok(())
}

// Rebuilding for target instructions must preserve the Responses conversation link.
#[test]
fn responses_previous_response_id_survives_target_prompt() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = TranslationPolicy::default();
    let body = json!({"model": "route", "input": "hi", "previous_response_id": "resp_1"});
    let mut request = engine
        .decode_request(WireFormat::OpenAiResponses, &body, &policy)?
        .request;

    prepare_request_for_target(&mut request, &"target".into(), Some("target prompt"));

    let output = engine
        .encode_request(WireFormat::OpenAiResponses, &request, &policy)?
        .body;

    assert_eq!(output["instructions"], "target prompt");
    assert_eq!(output["previous_response_id"], "resp_1");
    Ok(())
}

// Target preparation must preserve Anthropic's explicit tool failure signal.
#[test]
fn anthropic_tool_result_error_survives_target_prompt() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = TranslationPolicy::default();
    let body = json!({
        "model": "route",
        "max_tokens": 32,
        "messages": [
            {"role": "assistant", "content": [{
                "type": "tool_use",
                "id": "toolu_error_probe",
                "name": "opaque_probe",
                "input": {}
            }]},
            {"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_error_probe",
                "content": "opaque-result-42",
                "is_error": true
            }]}
        ]
    });

    let mut request = engine
        .decode_request(WireFormat::AnthropicMessages, &body, &policy)?
        .request;
    prepare_request_for_target(&mut request, &"target".into(), Some("target prompt"));

    let output = engine
        .encode_request(WireFormat::AnthropicMessages, &request, &policy)?
        .body;

    assert_eq!(output["messages"][1]["content"][0]["is_error"], true);
    Ok(())
}

// Target prompts force normalized re-encoding; built-in tool history must retain its wire shape.
#[test]
fn responses_builtin_tool_history_survives_target_prompt() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = TranslationPolicy::default();
    let input = json!([
        {
            "type": "apply_patch_call", "call_id": "patch_1",
            "operation": {"type": "update_file", "path": "src/lib.rs"}
        },
        {
            "type": "apply_patch_call_output", "call_id": "patch_1", "output": "done"
        },
        {
            "type": "shell_call", "call_id": "shell_1",
            "action": {"commands": ["pwd"]}, "environment": {"type": "local"}
        },
        {
            "type": "shell_call_output", "call_id": "shell_1",
            "output": [{"stdout": "/workspace", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}}]
        },
        {
            "type": "computer_call", "call_id": "computer_1",
            "action": {"type": "screenshot"}, "pending_safety_checks": []
        },
        {
            "type": "computer_call_output", "call_id": "computer_1",
            "output": {"type": "computer_screenshot", "file_id": "file_1"},
            "acknowledged_safety_checks": []
        }
    ]);
    let body = json!({
        "model": "route",
        "instructions": "caller instructions",
        "input": input
    });
    let mut request = engine
        .decode_request(WireFormat::OpenAiResponses, &body, &policy)?
        .request;

    prepare_request_for_target(&mut request, &"target".into(), Some("target prompt"));

    let output = engine
        .encode_request(WireFormat::OpenAiResponses, &request, &policy)?
        .body;
    assert_eq!(output["input"], input);
    assert_eq!(
        output["instructions"],
        "target prompt\n\ncaller instructions"
    );

    for target in [WireFormat::OpenAiChat, WireFormat::AnthropicMessages] {
        let error = engine
            .encode_request(target, &request, &policy)
            .expect_err("built-in Responses tool history must not become target text");
        assert_eq!(error.kind(), "UnsupportedTranslation");
    }
    Ok(())
}

// Built-in Responses tool history is valid only as top-level input items.
#[test]
fn responses_builtin_tool_history_must_be_top_level() {
    for (content, path) in [
        (
            json!([{"type": "shell_call", "call_id": "shell_1"}]),
            "$.input[0].content[0].type",
        ),
        (
            json!({"type": "shell_call", "call_id": "shell_1"}),
            "$.input[0].content.type",
        ),
    ] {
        let body = json!({
            "model": "route",
            "input": [{"type": "message", "role": "user", "content": content}]
        });
        let error = TranslationEngine::default()
            .decode_request(
                WireFormat::OpenAiResponses,
                &body,
                &TranslationPolicy::default(),
            )
            .expect_err("built-in tool history inside message content should be rejected");

        assert!(error.to_string().contains(path));
    }
}

// Model-only preparation retains provider fields while aligning exact replay with the target.
#[test]
fn preparing_without_a_prompt_preserves_exact_replay() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = TranslationPolicy::default();
    let body = json!({
        "model": "route",
        "messages": [{"role": "user", "content": "hi"}],
        "provider_field": true
    });
    let mut request = engine
        .decode_request(WireFormat::OpenAiChat, &body, &policy)?
        .request;

    prepare_request_for_target(&mut request, &"selected/model".into(), None);

    assert_eq!(request.model.as_deref(), Some("selected/model"));
    let preserved = &request.preservation.requests[&WireFormat::OpenAiChat.into()];
    assert_eq!(preserved["model"], "selected/model");
    assert_eq!(preserved["provider_field"], true);
    let encoded = engine
        .encode_request(WireFormat::OpenAiChat, &request, &policy)?
        .body;
    assert_eq!(encoded["model"], "selected/model");
    assert_eq!(encoded["provider_field"], true);

    let custom_format = FormatId::new("custom");
    request
        .preservation
        .requests
        .insert(custom_format.clone(), json!({"vendor_model": "route"}));
    prepare_request_for_target(&mut request, &"fallback/model".into(), None);
    assert_eq!(
        request.preservation.requests[&WireFormat::OpenAiChat.into()]["model"],
        "fallback/model"
    );
    assert!(!request.preservation.requests.contains_key(&custom_format));
    Ok(())
}

// Verifies Anthropic-only request fields are dropped or mapped for OpenAI Chat.
#[test]
fn anthropic_request_translates_to_openai_chat_without_anthropic_only_fields() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "claude-sonnet-4-20250514",
        "system": [{"type": "text", "text": "Be helpful."}],
        "messages": [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "lookup",
                        "input": {"query": "weather"}
                    }
                ]
            },
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": [
                        {"type": "text", "text": "72F"},
                        {"type": "image", "source": {"type": "base64", "data": "abc"}}
                    ]
                }]
            }
        ],
        "tools": [{
            "name": "lookup",
            "description": "Lookup data",
            "input_schema": {"type": "object"}
        }],
        "tool_choice": {"type": "tool", "name": "lookup"},
        "max_tokens": 100,
        "thinking": {"type": "adaptive"},
        "container": "claude-container"
    });

    let output = engine
        .translate_request(
            WireFormat::AnthropicMessages,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(
        output["messages"][0],
        json!({"role": "system", "content": "Be helpful."})
    );
    assert_eq!(
        output["messages"][1],
        json!({"role": "user", "content": "Hello"})
    );
    assert_eq!(output["messages"][2]["role"], "assistant");
    assert_eq!(output["messages"][2]["content"], "Let me check.");
    assert_eq!(
        output["messages"][2]["tool_calls"][0]["function"]["name"],
        "lookup"
    );
    assert_eq!(output["messages"][3]["role"], "tool");
    assert_eq!(output["messages"][3]["tool_call_id"], "toolu_1");
    assert!(
        output["messages"][3]["content"]
            .as_str()
            .unwrap()
            .contains("72F")
    );
    assert_eq!(output["tools"][0]["function"]["name"], "lookup");
    assert_eq!(
        output["tool_choice"],
        json!({"type": "function", "function": {"name": "lookup"}})
    );
    assert_eq!(output["max_completion_tokens"], 100);
    assert!(output.get("thinking").is_none());
    assert!(output.get("container").is_none());
    Ok(())
}

// Verifies Anthropic thinking blocks stay preserved but never leak into OpenAI Chat content.
#[test]
fn anthropic_thinking_blocks_do_not_leak_into_openai_chat_messages() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "claude-opus-4-20250514",
        "messages": [
            {"role": "user", "content": "Use the tool."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "I should call the tool.",
                        "signature": "sig-abc"
                    },
                    {"type": "redacted_thinking", "data": "encrypted"},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "lookup",
                        "input": {"query": "status"}
                    }
                ]
            }
        ],
        "tools": [{
            "name": "lookup",
            "input_schema": {"type": "object"}
        }],
        "max_tokens": 2048
    });

    let output = engine
        .translate_request(
            WireFormat::AnthropicMessages,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(output["messages"][1]["role"], "assistant");
    assert_eq!(output["messages"][1]["content"], Value::Null);
    assert!(output["messages"][1].get("reasoning_content").is_none());
    assert_eq!(
        output["messages"][1]["tool_calls"][0]["function"]["name"],
        "lookup"
    );
    assert!(!json_contains_content_type(&output, "thinking"));
    assert!(!json_contains_content_type(&output, "redacted_thinking"));

    let decoded = engine.decode_request(
        WireFormat::AnthropicMessages,
        &body,
        &TranslationPolicy::default(),
    )?;
    let replayed = engine.encode_request(
        WireFormat::AnthropicMessages,
        &decoded.request,
        &TranslationPolicy::default(),
    )?;
    assert_eq!(replayed.body, body);

    Ok(())
}

#[test]
fn anthropic_visible_text_survives_tool_history_to_responses() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "claude-opus-4-20250514",
        "messages": [
            {"role": "user", "content": "inspect"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "visible preface"},
                    {
                        "type": "tool_use",
                        "id": "toolu_status",
                        "name": "lookup",
                        "input": {"key": "status"}
                    }
                ]
            },
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_status",
                    "content": "ready"
                }]
            }
        ],
        "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
        "max_tokens": 64
    });

    let output = engine
        .translate_request(
            WireFormat::AnthropicMessages,
            WireFormat::OpenAiResponses,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(
        output["input"],
        json!([
            {"type": "message", "role": "user", "content": "inspect"},
            {"type": "message", "role": "assistant", "content": "visible preface"},
            {
                "type": "function_call",
                "call_id": "toolu_status",
                "name": "lookup",
                "arguments": "{\"key\":\"status\"}"
            },
            {
                "type": "function_call_output",
                "call_id": "toolu_status",
                "name": "lookup",
                "output": "ready"
            }
        ])
    );
    Ok(())
}

#[test]
fn anthropic_visible_text_around_tool_call_survives_when_thinking_is_filtered() -> TestResult {
    let output = TranslationEngine::default()
        .translate_request(
            WireFormat::AnthropicMessages,
            WireFormat::OpenAiResponses,
            &json!({
                "model": "claude-opus-4-20250514",
                "messages": [
                    {"role": "user", "content": "inspect"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "private", "signature": "sig"},
                            {"type": "text", "text": "before"},
                            {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}},
                            {"type": "text", "text": "after"}
                        ]
                    }
                ],
                "max_tokens": 64
            }),
            &TranslationPolicy::default(),
        )?
        .body;

    let input = output["input"]
        .as_array()
        .ok_or("Responses input is not an array")?;
    assert!(input.iter().any(|item| item["content"] == "before"));
    assert!(input.iter().any(|item| item["content"] == "after"));
    assert!(input.iter().any(|item| item["type"] == "function_call"));
    assert!(!input.iter().any(|item| item["type"] == "reasoning"));
    Ok(())
}

#[test]
fn responses_pairing_does_not_move_tool_output_across_user_text() -> TestResult {
    let output = TranslationEngine::default()
        .translate_request(
            WireFormat::AnthropicMessages,
            WireFormat::OpenAiResponses,
            &json!({
                "model": "claude-opus-4-20250514",
                "messages": [
                    {"role": "user", "content": "inspect"},
                    {"role": "assistant", "content": [{
                        "type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}
                    }]},
                    {"role": "user", "content": [
                        {"type": "text", "text": "context before result"},
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ready"}
                    ]}
                ],
                "max_tokens": 64
            }),
            &TranslationPolicy::default(),
        )?
        .body;

    let input = output["input"]
        .as_array()
        .ok_or("Responses input is not an array")?;
    let call = input
        .iter()
        .position(|item| item["type"] == "function_call")
        .ok_or("missing function call")?;
    let text = input
        .iter()
        .position(|item| item["content"] == "context before result")
        .ok_or("missing user text")?;
    let result = input
        .iter()
        .position(|item| item["type"] == "function_call_output")
        .ok_or("missing function output")?;
    assert!(call < text && text < result);
    Ok(())
}

// Verifies unsigned OpenAI-compatible reasoning is not forged as Anthropic thinking.
#[test]
fn openai_reasoning_content_does_not_forge_anthropic_thinking_block() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "gpt-reasoning",
        "messages": [
            {"role": "user", "content": "Use private reasoning."},
            {
                "role": "assistant",
                "reasoning_content": "private chain of thought",
                "content": "Visible answer"
            }
        ]
    });

    let output = engine
        .translate_request(
            WireFormat::OpenAiChat,
            WireFormat::AnthropicMessages,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(output["messages"][1]["role"], "assistant");
    assert_eq!(output["messages"][1]["content"], "Visible answer");
    assert!(!json_contains_content_type(&output, "thinking"));
    Ok(())
}

// Verifies unknown OpenAI content becomes Anthropic text, not raw provider blocks.
#[test]
fn openai_unknown_content_does_not_leak_into_anthropic_request_blocks() -> TestResult {
    let engine = TranslationEngine::default();
    let unknown_item = json!({"type": "future_openai_part", "payload": {"keep": true}});
    let body = json!({
        "model": "gpt-4o",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                unknown_item
            ]
        }]
    });

    let output = engine
        .translate_request(
            WireFormat::OpenAiChat,
            WireFormat::AnthropicMessages,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    let content = output["messages"][0]["content"]
        .as_array()
        .ok_or("Anthropic content should be an array")?;
    assert_eq!(content[1]["type"], "text");
    let recovered: Value = serde_json::from_str(
        content[1]["text"]
            .as_str()
            .ok_or("unknown block fallback should be text")?,
    )?;
    assert_eq!(recovered, unknown_item);
    assert!(!json_contains_content_type(&output, "future_openai_part"));
    Ok(())
}

// Verifies unknown Anthropic content becomes Responses text, not raw provider blocks.
#[test]
fn anthropic_unknown_content_does_not_leak_into_responses_request_blocks() -> TestResult {
    let engine = TranslationEngine::default();
    let unknown_item = json!({"type": "future_anthropic_part", "payload": {"keep": true}});
    let body = json!({
        "model": "claude-sonnet-4-20250514",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                unknown_item
            ]
        }],
        "max_tokens": 1024
    });

    let output = engine
        .translate_request(
            WireFormat::AnthropicMessages,
            WireFormat::OpenAiResponses,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    let content = output["input"][0]["content"]
        .as_array()
        .ok_or("Responses content should be an array")?;
    assert_eq!(content[1]["type"], "input_text");
    let recovered: Value = serde_json::from_str(
        content[1]["text"]
            .as_str()
            .ok_or("unknown block fallback should be text")?,
    )?;
    assert_eq!(recovered, unknown_item);
    assert!(!json_contains_content_type(
        &output,
        "future_anthropic_part"
    ));
    Ok(())
}

// Verifies Anthropic mixed tool-result and text content splits into valid OpenAI messages.
#[test]
fn anthropic_tool_result_followup_text_splits_to_openai_messages() -> TestResult {
    let engine = TranslationEngine::default();
    let raw_id = "functions.list_skills:0";
    let body = json!({
        "model": "claude-sonnet-4-20250514",
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": sanitize_anthropic_tool_use_id(raw_id),
                    "content": "72F"
                },
                {"type": "text", "text": "Now summarize it."}
            ]
        }],
        "max_tokens": 1024
    });

    let output = engine
        .translate_request(
            WireFormat::AnthropicMessages,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(
        output["messages"],
        json!([
            {"role": "tool", "tool_call_id": raw_id, "content": "72F"},
            {"role": "user", "content": "Now summarize it."}
        ])
    );
    Ok(())
}

// Verifies Anthropic multimodal blocks retain provider fields inside tool results.
#[test]
fn anthropic_tool_result_multimodal_blocks_round_trip_complete() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = TranslationPolicy {
        preservation: switchyard_translation::PreservationPolicy::Disabled,
        ..TranslationPolicy::default()
    };
    let document = json!({
        "type": "document",
        "title": "report.pdf",
        "context": "Quarterly results",
        "citations": {"enabled": true},
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": "ZG9jdW1lbnQ="
        }
    });
    let image = json!({
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "aW1hZ2U="
        }
    });
    let body = json!({
        "model": "claude-sonnet-4-20250514",
        "messages": [{
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_document",
                "content": [
                    {"type": "text", "text": "content ready"},
                    image.clone(),
                    document.clone()
                ]
            }]
        }],
        "max_tokens": 1024
    });

    let output = engine
        .translate_request(
            WireFormat::AnthropicMessages,
            WireFormat::AnthropicMessages,
            &body,
            &policy,
        )?
        .body;

    assert_eq!(
        output["messages"][0]["content"][0]["content"],
        json!([
            {"type": "text", "text": "content ready"},
            image,
            document
        ])
    );
    Ok(())
}

// Verifies provider-managed Anthropic file IDs are not reused as OpenAI file IDs.
#[test]
fn anthropic_tool_result_file_id_does_not_become_openai_file_id() -> TestResult {
    let engine = TranslationEngine::default();
    let document = json!({
        "type": "document",
        "source": {
            "type": "file",
            "file_id": "file_anthropic_123"
        }
    });
    let body = json!({
        "model": "claude-sonnet-4-20250514",
        "messages": [{
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_document",
                "content": [document.clone()]
            }]
        }],
        "max_tokens": 1024
    });

    let translated = engine.translate_request(
        WireFormat::AnthropicMessages,
        WireFormat::OpenAiChat,
        &body,
        &TranslationPolicy::default(),
    )?;

    assert_eq!(translated.body["messages"][1]["content"][0]["type"], "text");
    let recovered: Value = serde_json::from_str(
        translated.body["messages"][1]["content"][0]["text"]
            .as_str()
            .ok_or("file fallback should be text")?,
    )?;
    assert_eq!(recovered, document);
    assert!(translated.diagnostics.iter().any(|diagnostic| {
        diagnostic.message == "OpenAI Chat codec could not map file content"
    }));
    Ok(())
}

// Verifies parallel tool results preserve ordering and obey strict conversion policy.
#[test]
fn anthropic_parallel_multimodal_tool_results_preserve_order_and_policy() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "claude-sonnet-4-20250514",
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_image",
                    "content": [
                        {"type": "text", "text": "image ready"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "aW1hZ2U="
                            }
                        }
                    ]
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_document",
                    "content": [
                        {"type": "text", "text": "document ready"},
                        {
                            "type": "document",
                            "title": "report.pdf",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "ZG9jdW1lbnQ="
                            }
                        }
                    ]
                }
            ]
        }],
        "max_tokens": 1024
    });

    let output = engine
        .translate_request(
            WireFormat::AnthropicMessages,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(
        output["messages"],
        json!([
            {"role": "tool", "tool_call_id": "toolu_image", "content": "image ready"},
            {
                "role": "tool",
                "tool_call_id": "toolu_document",
                "content": "document ready"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,aW1hZ2U="}
                    },
                    {
                        "type": "file",
                        "file": {"file_data": "data:application/pdf;base64,ZG9jdW1lbnQ=", "filename": "report.pdf"}
                    }
                ]
            }
        ])
    );
    let mut policy = TranslationPolicy {
        lossy_conversion_policy: LossyConversionPolicy::Reject,
        ..TranslationPolicy::default()
    };

    let error = match engine.translate_request(
        WireFormat::AnthropicMessages,
        WireFormat::OpenAiChat,
        &body,
        &policy,
    ) {
        Ok(_) => panic!("multimodal tool result should be rejected by strict policy"),
        Err(error) => error,
    };

    assert_eq!(error.kind(), "LossyConversion");

    // Capability checks must reach media nested inside tool results.
    policy.lossy_conversion_policy = LossyConversionPolicy::AllowWithDiagnostics;
    policy.target_capabilities.supports_images = Some(false);
    policy.target_capabilities.supports_files = Some(false);
    let output = engine.translate_request(
        WireFormat::AnthropicMessages,
        WireFormat::OpenAiResponses,
        &body,
        &policy,
    )?;
    assert_eq!(output.diagnostics.len(), 2);

    policy.lossy_conversion_policy = LossyConversionPolicy::Reject;
    let error = engine
        .translate_request(
            WireFormat::AnthropicMessages,
            WireFormat::OpenAiResponses,
            &body,
            &policy,
        )
        .expect_err("nested media should be rejected by the target capability profile");
    assert_eq!(error.kind(), "LossyConversion");
    Ok(())
}

// Verifies structured Anthropic system blocks remain separated in OpenAI system text.
#[test]
fn anthropic_structured_system_blocks_preserve_boundaries_for_openai_chat() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "claude-sonnet-4-20250514",
        "system": [
            {"type": "text", "text": "You are helpful."},
            {"type": "text", "text": "Be concise."}
        ],
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100
    });

    let output = engine
        .translate_request(
            WireFormat::AnthropicMessages,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(
        output["messages"][0]["content"],
        "You are helpful.\n\nBe concise."
    );
    Ok(())
}

// Verifies invalid anonymous Anthropic tools are dropped before OpenAI encoding.
#[test]
fn anthropic_tool_without_name_is_dropped_before_openai_chat() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "claude-sonnet-4-20250514",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
        "tools": [{"description": "mystery tool", "input_schema": {}}]
    });

    let output = engine
        .translate_request(
            WireFormat::AnthropicMessages,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert!(output.get("tools").is_none());
    Ok(())
}

// Verifies Anthropic tool strictness survives translation into both OpenAI formats.
#[test]
fn anthropic_tool_strictness_is_preserved_for_openai_formats() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "claude-sonnet-4-20250514",
        "messages": [{"role": "user", "content": "Use a tool."}],
        "max_tokens": 100,
        "tools": [
            {"name": "strict_tool", "input_schema": {}, "strict": true},
            {"name": "non_strict_tool", "input_schema": {}, "strict": false},
            {"name": "unspecified_tool", "input_schema": {}}
        ]
    });

    let chat = engine
        .translate_request(
            WireFormat::AnthropicMessages,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;
    let responses = engine
        .translate_request(
            WireFormat::AnthropicMessages,
            WireFormat::OpenAiResponses,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(chat["tools"][0]["function"]["strict"], true);
    assert_eq!(chat["tools"][1]["function"]["strict"], false);
    assert!(chat["tools"][2]["function"].get("strict").is_none());
    assert_eq!(responses["tools"][0]["strict"], true);
    assert_eq!(responses["tools"][1]["strict"], false);
    assert!(responses["tools"][2].get("strict").is_none());
    Ok(())
}

// Verifies OpenAI-compatible Anthropic extension fields are preserved.
#[test]
fn anthropic_openai_compatible_extensions_are_preserved_for_openai_chat() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "claude-sonnet-4-20250514",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
        "metadata": {"user_id": "u123"},
        "stop_sequences": ["END"],
        "thinking": {"type": "enabled", "budget_tokens": 5000}
    });

    let output = engine
        .translate_request(
            WireFormat::AnthropicMessages,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(output["metadata"], json!({"user_id": "u123"}));
    assert_eq!(output["stop"], json!(["END"]));
    assert!(output.get("thinking").is_none());
    Ok(())
}

// Verifies Codex-style Responses tools translate into OpenAI Chat tool definitions.
#[test]
fn responses_request_translates_codex_tool_shape_to_openai_chat() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "gpt-4",
        "instructions": "Be brief.",
        "input": "List files",
        "max_output_tokens": 1024,
        "reasoning": {"effort": "high"},
        "tools": [
            {
                "type": "function",
                "id": "exec_command",
                "description": "Runs a command in a PTY.",
                "inputSchema": {
                    "jsonSchema": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"]
                    }
                }
            },
            {"id": "", "description": "", "inputSchema": {"jsonSchema": {}}}
        ],
        "tool_choice": "required"
    });

    let output = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(
        output["messages"][0],
        json!({"role": "system", "content": "Be brief."})
    );
    assert_eq!(
        output["messages"][1],
        json!({"role": "user", "content": "List files"})
    );
    assert_eq!(output["max_completion_tokens"], 1024);
    assert_eq!(output["reasoning_effort"], "high");
    assert_eq!(output["tool_choice"], "required");
    assert_eq!(output["tools"].as_array().unwrap().len(), 1);
    assert_eq!(output["tools"][0]["function"]["name"], "exec_command");
    assert_eq!(
        output["tools"][0]["function"]["parameters"]["required"],
        json!(["cmd"])
    );
    Ok(())
}

#[test]
fn responses_prompt_injection_preserves_namespace_description() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = TranslationPolicy::default();
    let body = json!({
        "model": "switchyard",
        "input": "Fix the parser",
        "tools": [{
            "type": "namespace",
            "name": "multi_agent_v1",
            "description": "Tools for managing sub-agents.",
            "tools": [{
                "type": "function",
                "name": "spawn_agent",
                "description": "Start one agent.",
                "parameters": {"type": "object"}
            }]
        }]
    });
    let mut request = engine
        .decode_request(WireFormat::OpenAiResponses, &body, &policy)?
        .request;

    prepare_request_for_target(&mut request, &"gpt-5.6-sol".into(), Some("Plan first."));
    let output = engine
        .encode_request(WireFormat::OpenAiResponses, &request, &policy)?
        .body;

    assert_eq!(
        output["tools"][0]["description"],
        "Tools for managing sub-agents."
    );
    Ok(())
}

// Verifies Python-style Responses tool definitions translate into OpenAI Chat tools.
#[test]
fn responses_request_translates_python_compatible_tool_shape_to_openai_chat() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "gpt-4",
        "input": "Get weather",
        "tools": [{
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"loc": {"type": "string"}}
            }
        }]
    });

    let output = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(output["tools"][0]["function"]["name"], "get_weather");
    assert_eq!(
        output["tools"][0]["function"]["parameters"]["properties"]["loc"]["type"],
        "string"
    );
    Ok(())
}

// Verifies unknown Responses input items are preserved as valid OpenAI text content.
#[test]
fn responses_unknown_input_item_is_preserved_for_openai_chat() -> TestResult {
    let engine = TranslationEngine::default();
    let unknown_item = json!({"type": "audio_clip", "data": "base64..."});
    let body = json!({
        "model": "gpt-4",
        "input": [
            {"type": "message", "role": "user", "content": "hi"},
            unknown_item,
            {"type": "message", "role": "assistant", "content": "hello"}
        ]
    });

    let output = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(output["messages"][1]["role"], "user");
    let content = output["messages"][1]["content"]
        .as_array()
        .ok_or("unknown item should encode as content array")?;
    assert_eq!(content.len(), 1);
    assert_eq!(content[0]["type"], "text");
    let text = content[0]["text"]
        .as_str()
        .ok_or("unknown item fallback should be text")?;
    let recovered: Value = serde_json::from_str(text)?;
    assert_eq!(recovered, unknown_item);
    assert_eq!(output["messages"][2]["role"], "assistant");
    Ok(())
}

// Responses accepts message-shaped input items without an explicit discriminator, and inline
// system and developer items keep their roles instead of being demoted to user.
#[test]
fn responses_input_messages_translate_with_instruction_roles_intact() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "gpt-4",
        "input": [
            {"type": "message", "role": "system", "content": "Be terse."},
            {"type": "message", "role": "developer", "content": "Return JSON only."},
            {"role": "user", "content": "hello"}
        ]
    });

    let output = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(
        output["messages"],
        json!([
            {"role": "system", "content": "Be terse."},
            {"role": "developer", "content": "Return JSON only."},
            {"role": "user", "content": "hello"}
        ])
    );
    Ok(())
}

// Inline instruction items must not detach pending reasoning from the assistant
// turn that produced it.
#[test]
fn responses_inline_instruction_does_not_detach_reasoning() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "gpt-4",
        "input": [
            {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "thinking..."}], "summary": []},
            {"type": "message", "role": "system", "content": "Be terse."},
            {"type": "message", "role": "assistant", "content": "hello"}
        ]
    });

    let output = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(output["messages"].as_array().map(Vec::len), Some(2));
    assert_eq!(output["messages"][0]["role"], "system");
    assert_eq!(output["messages"][1]["role"], "assistant");
    assert_eq!(output["messages"][1]["content"], "hello");
    assert_eq!(output["messages"][1]["reasoning"], "thinking...");
    Ok(())
}

// A discriminator-less object that is not message-shaped must not silently become prompt text.
#[test]
fn responses_input_without_type_or_message_shape_is_rejected() {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "gpt-4",
        "input": [{"payload": "ambiguous"}]
    });

    let error = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )
        .expect_err("ambiguous input item should be rejected");

    assert!(error.to_string().contains("$.input[0].type"));
}

// Verifies orphan Responses tool outputs degrade to readable user text.
#[test]
fn responses_orphan_function_call_output_degrades_to_user_text_for_openai_chat() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "gpt-4",
        "input": [{
            "type": "function_call_output",
            "call_id": "call_orphan",
            "output": "result"
        }]
    });

    let output = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(
        output["messages"],
        json!([{"role": "user", "content": "Tool result call_orphan: result"}])
    );
    Ok(())
}

// Verifies adjacent Responses function calls stay adjacent for OpenAI tool-result rules.
#[test]
fn responses_consecutive_function_calls_merge_for_openai_chat() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "gpt-4",
        "input": [
            {"type": "message", "role": "user", "content": "Do two things"},
            {"type": "function_call", "name": "tool_a", "call_id": "call_a", "arguments": "{}"},
            {"type": "function_call", "name": "tool_b", "call_id": "call_b", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_a", "output": "A done"},
            {"type": "function_call_output", "call_id": "call_b", "output": "B done"}
        ]
    });

    let output = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(output["messages"][1]["role"], "assistant");
    assert_eq!(
        output["messages"][1]["tool_calls"].as_array().map(Vec::len),
        Some(2)
    );
    assert_eq!(output["messages"][2]["role"], "tool");
    assert_eq!(output["messages"][3]["role"], "tool");
    Ok(())
}

// Verifies Responses function-call argument strings become Anthropic tool dictionaries.
#[test]
fn responses_function_call_arguments_parse_for_anthropic_tool_use() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "gpt-4",
        "input": [
            {"type": "message", "role": "user", "content": "List files"},
            {
                "type": "function_call",
                "name": "exec_command",
                "call_id": "call_1",
                "arguments": "{\"cmd\":\"ls -la\",\"limit\":2}"
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "README.md"}
        ]
    });

    let output = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            WireFormat::AnthropicMessages,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(output["messages"][1]["content"][0]["type"], "tool_use");
    assert_eq!(
        output["messages"][1]["content"][0]["input"],
        json!({"cmd": "ls -la", "limit": 2})
    );
    assert_eq!(
        output["messages"][2]["content"],
        json!([{"type": "tool_result", "tool_use_id": "call_1", "content": "README.md"}])
    );
    Ok(())
}

// Verifies malformed Responses arguments still produce object-shaped Anthropic input.
#[test]
fn responses_function_call_arguments_wrap_non_object_values_for_anthropic() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "gpt-4",
        "input": [
            {"type": "message", "role": "user", "content": "Call tools"},
            {
                "type": "function_call",
                "name": "bad_json",
                "call_id": "call_bad",
                "arguments": "not-json"
            },
            {
                "type": "function_call",
                "name": "array_json",
                "call_id": "call_array",
                "arguments": "[1,2]"
            },
            {
                "type": "function_call",
                "name": "object_value",
                "call_id": "call_object",
                "arguments": {"already": "object"}
            }
        ]
    });

    let output = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            WireFormat::AnthropicMessages,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    let content = &output["messages"][1]["content"];
    assert_eq!(content[0]["input"], json!({"raw": "not-json"}));
    assert_eq!(content[1]["input"], json!({"value": [1, 2]}));
    assert_eq!(content[2]["input"], json!({"already": "object"}));
    Ok(())
}

// Verifies deferred Responses messages remain after matching tool results.
#[test]
fn responses_deferred_message_stays_after_matching_tool_result_for_openai_chat() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "gpt-4",
        "input": [
            {"type": "message", "role": "user", "content": "Search for X"},
            {"type": "function_call", "name": "search", "call_id": "call_1", "arguments": "{}"},
            {"type": "message", "role": "assistant", "content": "I will summarize after the tool."},
            {"type": "function_call_output", "call_id": "call_1", "output": "Found X"}
        ]
    });

    let output = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(output["messages"][1]["role"], "assistant");
    assert_eq!(output["messages"][1]["tool_calls"][0]["id"], "call_1");
    assert_eq!(output["messages"][2]["role"], "tool");
    assert_eq!(output["messages"][2]["tool_call_id"], "call_1");
    assert_eq!(output["messages"][3]["role"], "assistant");
    assert_eq!(
        output["messages"][3]["content"],
        "I will summarize after the tool."
    );
    Ok(())
}

#[test]
fn parallel_tool_policy_survives_anthropic_translations() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = TranslationPolicy::default();
    for format in [WireFormat::OpenAiChat, WireFormat::OpenAiResponses] {
        for choice in [Some("any"), Some("auto"), Some("tool"), None] {
            for disabled in [Some(true), Some(false), None] {
                let mut body = json!({
                    "model": "test-model",
                    "max_tokens": 256,
                    "messages": [{"role": "user", "content": "Use the marker tool."}],
                    "tools": [{
                        "name": "marker",
                        "input_schema": {"type": "object", "properties": {}}
                    }]
                });
                if let Some(choice) = choice {
                    body["tool_choice"] = json!({"type": choice});
                    if choice == "tool" {
                        body["tool_choice"]["name"] = json!("marker");
                    }
                    if let Some(disabled) = disabled {
                        body["tool_choice"]["disable_parallel_tool_use"] = json!(disabled);
                    }
                }
                let mut openai = engine
                    .translate_request(WireFormat::AnthropicMessages, format, &body, &policy)?
                    .body;
                if choice.is_some() {
                    assert_eq!(
                        openai.get("parallel_tool_calls"),
                        disabled.map(|value| json!(!value)).as_ref(),
                        "{format:?}, {choice:?}, {disabled:?}"
                    );
                } else if let Some(disabled) = disabled {
                    // OpenAI allows this policy without an explicit tool choice.
                    openai["parallel_tool_calls"] = json!(!disabled);
                }
                let anthropic = engine
                    .translate_request(format, WireFormat::AnthropicMessages, &openai, &policy)?
                    .body;
                assert_eq!(
                    anthropic["tool_choice"].get("disable_parallel_tool_use"),
                    disabled.map(|value| json!(value)).as_ref(),
                    "{format:?}, {choice:?}, {disabled:?}"
                );
                if choice.is_some() || disabled.is_some() {
                    assert_eq!(anthropic["tool_choice"]["type"], choice.unwrap_or("auto"));
                    if choice == Some("tool") {
                        assert_eq!(anthropic["tool_choice"]["name"], "marker");
                    }
                } else {
                    assert!(anthropic.get("tool_choice").is_none());
                }
                if choice.is_none() {
                    for tools in [None, Some(json!([]))] {
                        openai.as_object_mut().unwrap().remove("tools");
                        if let Some(tools) = tools {
                            openai["tools"] = tools;
                        }
                        let anthropic = engine
                            .translate_request(
                                format,
                                WireFormat::AnthropicMessages,
                                &openai,
                                &policy,
                            )?
                            .body;
                        assert!(anthropic.get("tools").is_none());
                        assert!(anthropic.get("tool_choice").is_none());
                    }
                }
            }
        }
    }
    Ok(())
}

// Verifies Responses-compatible extension fields survive a Chat-to-Responses
// hop, and that Chat-only fields are excluded rather than passed through.
#[test]
fn chat_compatible_extensions_survive_to_responses() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"trace": "abc"},
        "parallel_tool_calls": false,
        "prompt_cache_key": "session-1",
        "prompt_cache_retention": "24h",
        "safety_identifier": "safe-1",
        "service_tier": "flex",
        "store": false,
        "stream_options": {"include_usage": true},
        "top_logprobs": 2,
        "user": "u-123"
    });

    let output = engine
        .translate_request(
            WireFormat::OpenAiChat,
            WireFormat::OpenAiResponses,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(output["metadata"], json!({"trace": "abc"}));
    assert_eq!(output["parallel_tool_calls"], false);
    assert_eq!(output["prompt_cache_key"], "session-1");
    assert_eq!(output["prompt_cache_retention"], "24h");
    assert_eq!(output["safety_identifier"], "safe-1");
    assert_eq!(output["service_tier"], "flex");
    assert_eq!(output["store"], false);
    assert_eq!(output["user"], "u-123");
    // Chat-only fields are not in the Responses allowlist, so they stay dropped.
    assert!(output.get("stream_options").is_none());
    assert!(output.get("top_logprobs").is_none());
    Ok(())
}

#[test]
fn responses_chat_compatible_extensions_survive_to_openai_chat() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "gpt-4",
        "input": "hi",
        "metadata": {"trace": "abc"},
        "parallel_tool_calls": false,
        "prompt_cache_key": "session-1",
        "prompt_cache_retention": "24h",
        "safety_identifier": "safe-1",
        "service_tier": "flex",
        "store": false,
        "stream_options": {"include_usage": true},
        "top_logprobs": 2,
        "user": "u-123"
    });

    let output = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(output["metadata"], json!({"trace": "abc"}));
    assert_eq!(output["parallel_tool_calls"], false);
    assert_eq!(output["prompt_cache_key"], "session-1");
    assert_eq!(output["prompt_cache_retention"], "24h");
    assert_eq!(output["safety_identifier"], "safe-1");
    assert_eq!(output["service_tier"], "flex");
    assert_eq!(output["store"], false);
    assert_eq!(output["stream_options"], json!({"include_usage": true}));
    assert_eq!(output["top_logprobs"], 2);
    assert_eq!(output["user"], "u-123");
    Ok(())
}

// Verifies Codex-style reasoning items attach to the turn's tool-call message
// instead of surfacing as fabricated empty assistant chat messages.
#[test]
fn responses_reasoning_items_attach_to_tool_call_turn_for_openai_chat() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "big-reasoner",
        "input": [
            {"type": "message", "role": "user", "content": "Fix pip"},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "\n\n"}]
            },
            {
                "type": "reasoning",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": "Check the python setup."}]
            },
            {
                "type": "function_call",
                "name": "shell",
                "call_id": "call-1",
                "arguments": "{\"command\":\"pip3 --version\"}"
            },
            {"type": "function_call_output", "call_id": "call-1", "output": "no pip"}
        ]
    });

    let output = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    let messages = output["messages"]
        .as_array()
        .ok_or("messages is not an array")?;
    assert!(
        !messages
            .iter()
            .any(|message| message["role"] == "assistant" && message["content"] == ""),
        "reasoning item leaked an empty assistant message: {messages:?}"
    );
    assert_eq!(messages.len(), 4);
    assert_eq!(messages[1], json!({"role": "assistant", "content": "\n\n"}));
    assert_eq!(messages[2]["reasoning"], "Check the python setup.");
    assert_eq!(messages[2]["tool_calls"][0]["id"], "call-1");
    assert_eq!(messages[3]["role"], "tool");
    Ok(())
}

// Verifies a reasoning item merges into the assistant message that follows it.
#[test]
fn responses_reasoning_item_merges_into_next_assistant_message_for_openai_chat() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "gpt-5",
        "input": [
            {"type": "message", "role": "user", "content": "Check the file"},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Reading."}]},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Let me check."}]
            }
        ]
    });

    let output = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(
        output["messages"],
        json!([
            {"role": "user", "content": "Check the file"},
            {
                "role": "assistant",
                "content": "Let me check.",
                "reasoning": "Reading."
            }
        ])
    );
    Ok(())
}

#[test]
fn openai_chat_reasoning_details_round_trip_in_assistant_history() -> TestResult {
    let engine = TranslationEngine::default();
    // Exercise the normalized IR path instead of replaying the original JSON.
    let policy = normalized_policy();
    let details = json!([
        {
            "type": "reasoning.summary",
            "summary": "Inspect the environment.",
            "id": "reasoning-1",
            "format": "openai-responses-v1",
            "index": 0
        },
        {
            "type": "reasoning.encrypted",
            "data": "opaque-encrypted-reasoning",
            "id": "reasoning-1",
            "format": "openai-responses-v1",
            "index": 1
        }
    ]);
    let body = json!({
        "model": REASONING_MODEL,
        "messages": [
            {"role": "user", "content": "Inspect the environment"},
            {
                "role": "assistant",
                "content": null,
                "reasoning": "fallback text",
                "reasoning_details": details,
                "tool_calls": [shell_tool_call()]
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "/workspace"}
        ]
    });

    let output = engine
        .translate_request(
            WireFormat::OpenAiChat,
            WireFormat::OpenAiChat,
            &body,
            &policy,
        )?
        .body;

    assert_eq!(output["messages"][1]["reasoning_details"], details);
    assert!(output["messages"][1].get("reasoning").is_none());
    assert_eq!(output["messages"][1]["tool_calls"][0]["id"], "call-1");
    Ok(())
}

#[test]
fn openai_chat_encrypted_reasoning_details_retain_fallback() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = normalized_policy();
    let details = json!([{
        "type": "reasoning.encrypted",
        "data": "opaque-encrypted-reasoning",
        "id": "reasoning-1",
        "format": "openai-responses-v1",
        "index": 0
    }]);
    let body = json!({
        "model": REASONING_MODEL,
        "messages": [{
            "role": "assistant",
            "content": null,
            "reasoning": "fallback text",
            "reasoning_details": details
        }]
    });

    let output = engine
        .translate_request(
            WireFormat::OpenAiChat,
            WireFormat::OpenAiChat,
            &body,
            &policy,
        )?
        .body;

    assert_eq!(output["messages"][0]["reasoning_details"], details);
    assert_eq!(output["messages"][0]["reasoning"], "fallback text");
    Ok(())
}

// Verifies merged reasoning re-emerges as a Responses reasoning item ahead of
// the turn's function call when encoding back to the Responses format.
#[test]
fn responses_reasoning_items_round_trip_through_decode_and_encode() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = TranslationPolicy {
        preservation: switchyard_translation::PreservationPolicy::Disabled,
        ..TranslationPolicy::default()
    };
    let body = json!({
        "model": "gpt-5",
        "input": [
            {"type": "message", "role": "user", "content": "List files"},
            {
                "type": "reasoning",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": "Simple ls."}]
            },
            {
                "type": "function_call",
                "name": "shell",
                "call_id": "call-ls",
                "arguments": "{\"command\":\"ls\"}"
            },
            {"type": "function_call_output", "call_id": "call-ls", "output": "a.py"}
        ]
    });

    let output = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiResponses,
            &body,
            &policy,
        )?
        .body;

    let input = output["input"].as_array().ok_or("input is not an array")?;
    let item_types = input
        .iter()
        .map(|item| item["type"].as_str().unwrap_or_default())
        .collect::<Vec<_>>();
    assert_eq!(
        item_types,
        vec![
            "message",
            "reasoning",
            "function_call",
            "function_call_output"
        ]
    );
    assert_eq!(
        input[1]["summary"],
        json!([{"type": "summary_text", "text": "Simple ls."}])
    );
    assert!(input[1].get("content").is_none());
    assert_eq!(input[2]["call_id"], "call-ls");
    // Outputs carry the paired call's name for upstreams (Kimi K3) that
    // resolve tool results by name rather than by call order.
    assert_eq!(input[3]["name"], "shell");
    Ok(())
}

// Verifies Codex-style encrypted reasoning remains replayable after a prompt
// mutation drops exact replay, without synthesizing invalid reasoning content.
#[test]
fn responses_encrypted_reasoning_replays_without_input_content() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "switchyard",
        "input": [
            {"type": "message", "role": "user", "content": "Inspect"},
            {
                "type": "reasoning",
                "id": "rs_prior",
                "summary": [],
                "encrypted_content": "opaque-encrypted-reasoning"
            },
            {
                "type": "function_call",
                "name": "exec_command",
                "call_id": "call-1",
                "arguments": "{\"cmd\":\"pwd\"}"
            },
            {"type": "function_call_output", "call_id": "call-1", "output": "/app"}
        ]
    });

    let policy = TranslationPolicy::default();
    let mut request = engine
        .decode_request(WireFormat::OpenAiResponses, &body, &policy)?
        .request;
    prepare_request_for_target(
        &mut request,
        &"openai/openai/gpt-5.6-sol".into(),
        Some("[router-guidance] Continue from the current state."),
    );

    let output = engine
        .encode_request(WireFormat::OpenAiResponses, &request, &policy)?
        .body;

    assert_eq!(output["model"], "openai/openai/gpt-5.6-sol");
    assert_eq!(
        output["instructions"],
        "[router-guidance] Continue from the current state."
    );
    let input = output["input"].as_array().ok_or("input is not an array")?;
    let reasoning = input
        .iter()
        .find(|item| item["type"] == "reasoning")
        .ok_or("reasoning item was not replayed")?;
    assert_eq!(reasoning["id"], "rs_prior");
    assert_eq!(reasoning["summary"], json!([]));
    assert_eq!(reasoning["encrypted_content"], "opaque-encrypted-reasoning");
    assert!(reasoning.get("content").is_none());
    Ok(())
}

// Verifies an empty non-encrypted reasoning item is omitted instead of being
// replayed as an empty assistant message.
#[test]
fn responses_empty_reasoning_without_encrypted_content_is_omitted() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = TranslationPolicy {
        preservation: switchyard_translation::PreservationPolicy::Disabled,
        ..TranslationPolicy::default()
    };
    let body = json!({
        "model": "gpt-5",
        "input": [
            {"type": "message", "role": "user", "content": "Inspect"},
            {"type": "reasoning", "summary": []},
            {"type": "message", "role": "user", "content": "Continue"}
        ]
    });

    let output = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiResponses,
            &body,
            &policy,
        )?
        .body;

    let input = output["input"].as_array().ok_or("input is not an array")?;
    let item_types = input
        .iter()
        .map(|item| item["type"].as_str().unwrap_or_default())
        .collect::<Vec<_>>();
    assert_eq!(item_types, vec!["message", "message"]);
    assert_eq!(input[0]["role"], "user");
    assert_eq!(input[1]["role"], "user");
    Ok(())
}

// Verifies Responses JSON schema text format maps to Chat response_format shape.
#[test]
fn responses_json_schema_text_format_maps_to_chat_response_format() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "gpt-4",
        "input": "Return JSON",
        "text": {
            "format": {
                "type": "json_schema",
                "name": "answer",
                "schema": {"type": "object"},
                "strict": true
            }
        }
    });

    let output = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(
        output["response_format"],
        json!({
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {"type": "object"},
                "strict": true
            }
        })
    );
    Ok(())
}

// Verifies Chat response_format JSON schema fields flatten into Responses text.format.
#[test]
fn chat_json_schema_response_format_maps_to_responses_text_format() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "gpt-5",
        "messages": [{"role": "user", "content": "Return JSON"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "description": "A structured answer",
                "schema": {"type": "object"},
                "strict": true
            }
        }
    });

    let output = engine
        .translate_request(
            WireFormat::OpenAiChat,
            WireFormat::OpenAiResponses,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(
        output["text"]["format"],
        json!({
            "type": "json_schema",
            "name": "answer",
            "description": "A structured answer",
            "schema": {"type": "object"},
            "strict": true
        })
    );
    Ok(())
}

// Verifies nested fields cannot override the Responses format discriminator.
#[test]
fn chat_json_schema_response_format_ignores_nested_type_override() -> TestResult {
    let output = translate_chat_response_format_to_responses(json!({
        "type": "json_schema",
        "json_schema": {
            "type": "text",
            "name": "answer",
            "description": "A structured answer",
            "schema": {"type": "object"},
            "strict": true
        }
    }))?;

    assert_eq!(
        output,
        json!({
            "type": "json_schema",
            "name": "answer",
            "description": "A structured answer",
            "schema": {"type": "object"},
            "strict": true
        })
    );
    Ok(())
}

// Verifies incomplete JSON schema wrappers remain unchanged instead of being partially flattened.
#[test]
fn chat_empty_json_schema_wrapper_is_preserved() -> TestResult {
    let response_format = json!({"type": "json_schema", "json_schema": {}});

    assert_eq!(
        translate_chat_response_format_to_responses(response_format.clone())?,
        response_format
    );
    Ok(())
}

fn translate_chat_response_format_to_responses(
    response_format: Value,
) -> std::result::Result<Value, Box<dyn std::error::Error + Send + Sync>> {
    let body = json!({
        "model": "gpt-5",
        "messages": [{"role": "user", "content": "Return JSON"}],
        "response_format": response_format
    });
    let output = TranslationEngine::default()
        .translate_request(
            WireFormat::OpenAiChat,
            WireFormat::OpenAiResponses,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;
    Ok(output["text"]["format"].clone())
}

// Verifies OpenAI system/developer/reasoning fields map to Anthropic request fields.
#[test]
fn openai_request_translates_system_developer_and_reasoning_to_anthropic() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "gpt-5",
        "messages": [
            {"role": "system", "content": "System rules."},
            {"role": "developer", "content": "Developer rules."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe"},
                    {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}}
                ]
            }
        ],
        "max_completion_tokens": 512,
        "reasoning_effort": "high",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "strict": true,
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": false
                }
            }
        }
    });

    let output = engine
        .translate_request(
            WireFormat::OpenAiChat,
            WireFormat::AnthropicMessages,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(output["model"], "gpt-5");
    assert_eq!(output["system"], "System rules.\n\nDeveloper rules.");
    assert_eq!(output["max_tokens"], 512);
    assert_eq!(output["thinking"], json!({"type": "adaptive"}));
    assert_eq!(
        output["output_config"],
        json!({
            "effort": "high",
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": false
                }
            }
        })
    );
    assert_eq!(output["messages"][0]["role"], "user");
    assert_eq!(
        output["messages"][0]["content"][0],
        json!({"type": "text", "text": "Describe"})
    );
    Ok(())
}

// Builds an Anthropic request whose structured output uses the given field shape.
fn anthropic_structured_output_request(output: Value) -> Value {
    let mut body = json!({
        "model": "captured-model",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "ping"}]
    });
    let object = body.as_object_mut().expect("request object");
    for (key, value) in output.as_object().expect("output object") {
        object.insert(key.clone(), value.clone());
    }
    body
}

fn city_schema() -> Value {
    json!({
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": false
    })
}

// Anthropic carries structured output in `output_config.format`, with the top-level
// `output_format` as the earlier beta spelling it still accepts. The neutral contract
// is OpenAI-shaped, so an ingress schema has to survive into `response_format` or the
// upstream is never asked for structured output. A shape that cannot be mapped is
// reported rather than forwarded unconstrained, and reasoning effort shares
// `output_config`, so reading the schema must leave it alone.
#[test]
fn anthropic_structured_output_maps_to_openai_response_format() -> TestResult {
    let engine = TranslationEngine::default();
    let format = json!({"type": "json_schema", "schema": city_schema()});
    let stale = json!({"type": "json_schema", "schema": {"type": "object"}});
    let cases: Vec<(&str, Value, Option<Value>, Option<&str>)> = vec![
        (
            "current field, alongside effort",
            json!({"output_config": {"effort": "high", "format": format}}),
            Some(city_schema()),
            Some("high"),
        ),
        (
            "legacy beta field",
            json!({"output_format": format}),
            Some(city_schema()),
            None,
        ),
        (
            "current field wins over legacy",
            json!({"output_config": {"format": format}, "output_format": stale}),
            Some(city_schema()),
            None,
        ),
        ("no structured output", json!({}), None, None),
        (
            "unsupported format type",
            json!({"output_config": {"format": {"type": "json_object"}}}),
            None,
            None,
        ),
        (
            "schema is not an object",
            json!({"output_config": {"format": {"type": "json_schema", "schema": "nope"}}}),
            None,
            None,
        ),
    ];

    for (label, output, expected_schema, expected_effort) in cases {
        let translated = engine.translate_request(
            WireFormat::AnthropicMessages,
            WireFormat::OpenAiChat,
            &anthropic_structured_output_request(output.clone()),
            &TranslationPolicy::default(),
        )?;
        let response_format = translated.body.get("response_format");

        match &expected_schema {
            Some(schema) => {
                let response_format = response_format.ok_or(label)?;
                assert_eq!(response_format["json_schema"]["schema"], *schema, "{label}");
                assert!(
                    response_format["json_schema"]["name"]
                        .as_str()
                        .is_some_and(|name| !name.is_empty()),
                    "{label}"
                );
            }
            None => assert!(response_format.is_none(), "{label}"),
        }
        if let Some(effort) = expected_effort {
            assert_eq!(translated.body["reasoning_effort"], effort, "{label}");
        }

        // A format that was present but unmapped is the only case that must report.
        let unmapped = expected_schema.is_none() && output.get("output_config").is_some();
        assert_eq!(
            translated
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message.contains("Anthropic structured output")),
            unmapped,
            "{label}"
        );
    }
    Ok(())
}

// Strict callers get an error instead of an unconstrained upstream request.
#[test]
fn anthropic_unmappable_output_format_is_rejected_under_strict_policy() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = TranslationPolicy {
        lossy_conversion_policy: LossyConversionPolicy::Reject,
        ..TranslationPolicy::default()
    };
    let body = anthropic_structured_output_request(json!({
        "output_config": {"format": {"type": "json_object"}}
    }));

    match engine.translate_request(
        WireFormat::AnthropicMessages,
        WireFormat::OpenAiChat,
        &body,
        &policy,
    ) {
        Ok(_) => panic!("an unmappable output format should be rejected by strict policy"),
        Err(error) => assert_eq!(error.kind(), "LossyConversion"),
    }
    Ok(())
}

// Verifies inline `data:` images become Anthropic base64 sources, since Anthropic's URL
// source rejects anything that is not an http(s) link. A percent-encoded payload is not raw
// base64, so it keeps the URL source it had before rather than reaching Anthropic mislabelled.
#[test]
fn openai_data_uri_images_translate_to_anthropic_sources() -> TestResult {
    let engine = TranslationEngine::default();
    let cases = [
        (
            "base64 data URI",
            json!({"url": "data:image/png;base64,aW1hZ2U=", "detail": "high"}),
            json!({"type": "base64", "media_type": "image/png", "data": "aW1hZ2U="}),
        ),
        (
            "data URI carrying extra parameters",
            json!({"url": "data:image/jpeg;charset=binary;base64,aW1hZ2U="}),
            json!({"type": "base64", "media_type": "image/jpeg", "data": "aW1hZ2U="}),
        ),
        (
            "percent-encoded data URI",
            json!({"url": "data:image/png;base64,YQ%3D%3D"}),
            json!({"type": "url", "url": "data:image/png;base64,YQ%3D%3D"}),
        ),
    ];

    for (case, image_url, expected_source) in cases {
        let body = json!({
            "model": "claude-sonnet-4-20250514",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is this?"},
                    {"type": "image_url", "image_url": image_url}
                ]
            }]
        });

        let output = engine
            .translate_request(
                WireFormat::OpenAiChat,
                WireFormat::AnthropicMessages,
                &body,
                &TranslationPolicy::default(),
            )?
            .body;

        assert_eq!(
            output["messages"][0]["content"][1],
            json!({"type": "image", "source": expected_source}),
            "{case}"
        );
    }
    Ok(())
}

// Verifies Anthropic receives its supported schema subset without mutating the neutral contract.
#[test]
fn openai_schema_constraints_are_removed_from_anthropic_output_format() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "claude-opus-4-8",
        "messages": [{"role": "user", "content": "Return a probability."}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "probability",
                "strict": true,
                "schema": {
                    "type": "object",
                    "properties": {
                        "crux": {"type": "string", "minLength": 1, "maxLength": 80},
                        "p_solve": {"type": "number", "minimum": 0, "maximum": 1}
                    },
                    "required": ["crux", "p_solve"],
                    "additionalProperties": false
                }
            }
        }
    });

    let translated = engine.translate_request(
        WireFormat::OpenAiChat,
        WireFormat::AnthropicMessages,
        &body,
        &TranslationPolicy::default(),
    )?;

    assert_eq!(
        translated.body["output_config"]["format"]["schema"],
        json!({
            "type": "object",
            "properties": {
                "crux": {"type": "string"},
                "p_solve": {"type": "number"}
            },
            "required": ["crux", "p_solve"],
            "additionalProperties": false
        })
    );
    assert_eq!(translated.diagnostics.len(), 1);
    assert!(
        translated.diagnostics[0]
            .message
            .contains("unsupported JSON Schema constraints")
    );
    assert_eq!(
        body["response_format"]["json_schema"]["schema"]["properties"]["p_solve"]["minimum"],
        0
    );
    Ok(())
}

// Verifies OpenAI tool strictness survives translation to Anthropic.
#[test]
fn openai_tool_strictness_is_preserved_for_anthropic() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "gpt-5",
        "messages": [{"role": "user", "content": "Use a tool."}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "strict", "parameters": {}, "strict": true}
            },
            {
                "type": "function",
                "function": {"name": "non_strict", "parameters": {}, "strict": false}
            },
            {
                "type": "function",
                "function": {"name": "unspecified", "parameters": {}}
            }
        ]
    });

    let output = engine
        .translate_request(
            WireFormat::OpenAiChat,
            WireFormat::AnthropicMessages,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(output["tools"][0]["strict"], true);
    assert_eq!(output["tools"][1]["strict"], false);
    assert!(output["tools"][2].get("strict").is_none());
    Ok(())
}

// Verifies Anthropic-bound OpenAI requests get the required max_tokens fallback.
#[test]
fn openai_request_to_anthropic_adds_required_default_max_tokens() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "gpt-5",
        "messages": [{"role": "user", "content": "hi"}]
    });

    let output = engine
        .translate_request(
            WireFormat::OpenAiChat,
            WireFormat::AnthropicMessages,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(output["max_tokens"], 64_000);
    Ok(())
}

// Verifies OpenAI string stop values map to Anthropic stop_sequences arrays.
#[test]
fn openai_stop_string_maps_to_anthropic_stop_sequences() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "gpt-5",
        "messages": [{"role": "user", "content": "hi"}],
        "stop": "END"
    });

    let output = engine
        .translate_request(
            WireFormat::OpenAiChat,
            WireFormat::AnthropicMessages,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(output["stop_sequences"], json!(["END"]));
    Ok(())
}

// Verifies OpenAI tool results merge into Anthropic user tool-result content.
#[test]
fn openai_tool_results_are_merged_when_translating_to_anthropic() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "call tools"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [
                    {
                        "id": "call.bad:id/with space",
                        "type": "function",
                        "function": {"name": "a", "arguments": "{}"}
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "b", "arguments": "{}"}
                    }
                ]
            },
            {"role": "tool", "tool_call_id": "call.bad:id/with space", "content": "one"},
            {"role": "tool", "tool_call_id": "call_2", "content": "two"}
        ]
    });

    let output = engine
        .translate_request(
            WireFormat::OpenAiChat,
            WireFormat::AnthropicMessages,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(
        output["messages"][1]["content"][0]["id"],
        sanitize_anthropic_tool_use_id("call.bad:id/with space")
    );
    assert_eq!(
        output["messages"][2]["content"],
        json!([
            {
                "type": "tool_result",
                "tool_use_id": sanitize_anthropic_tool_use_id("call.bad:id/with space"),
                "content": "one"
            },
            {"type": "tool_result", "tool_use_id": "call_2", "content": "two"}
        ])
    );
    Ok(())
}

// Recursively checks whether a JSON tree contains a content block with the requested type.
fn json_contains_content_type(value: &Value, expected: &str) -> bool {
    match value {
        Value::Object(object) => {
            if object.get("type").and_then(Value::as_str) == Some(expected) {
                return true;
            }
            object
                .values()
                .any(|child| json_contains_content_type(child, expected))
        }
        Value::Array(items) => items
            .iter()
            .any(|child| json_contains_content_type(child, expected)),
        _ => false,
    }
}

// Malformed provider fields must fail before normalization can change their meaning.
#[test]
fn malformed_request_fields_are_rejected() {
    let engine = TranslationEngine::default();
    let cases = [
        (
            "Anthropic object system",
            WireFormat::AnthropicMessages,
            json!({"model": "claude", "max_tokens": 8, "system": {}, "messages": []}),
            "expected string or array of text blocks at $.system",
        ),
        (
            "Anthropic boolean system",
            WireFormat::AnthropicMessages,
            json!({"model": "claude", "max_tokens": 8, "system": true, "messages": []}),
            "expected string or array of text blocks at $.system",
        ),
        (
            "Anthropic negative max_tokens",
            WireFormat::AnthropicMessages,
            json!({"model": "claude", "max_tokens": -1, "messages": []}),
            "invalid value at $.max_tokens: expected a non-negative integer",
        ),
        (
            "Anthropic string max_tokens",
            WireFormat::AnthropicMessages,
            json!({"model": "claude", "max_tokens": "8", "messages": []}),
            "invalid value at $.max_tokens: expected a non-negative integer",
        ),
        (
            "OpenAI Chat string messages",
            WireFormat::OpenAiChat,
            json!({"model": "gpt", "messages": "invalid"}),
            "expected array at $.messages",
        ),
        (
            "Anthropic string messages",
            WireFormat::AnthropicMessages,
            json!({"model": "claude", "max_tokens": 8, "messages": "invalid"}),
            "expected array at $.messages",
        ),
        (
            "Responses boolean input",
            WireFormat::OpenAiResponses,
            json!({"model": "gpt", "input": true}),
            "expected string or array at $.input",
        ),
        (
            "Responses null input",
            WireFormat::OpenAiResponses,
            json!({"model": "gpt", "input": null}),
            "expected string or array at $.input",
        ),
    ];

    for (case, format, body, expected) in cases {
        match engine.decode_request(format, &body, &TranslationPolicy::default()) {
            Ok(_) => panic!("{case} should be rejected"),
            Err(error) => assert_eq!(error.to_string(), expected, "{case}"),
        }
    }

    let valid_empty_output = json!({
        "model": "claude",
        "max_tokens": 0,
        "system": null,
        "messages": []
    });
    if let Err(error) = engine.decode_request(
        WireFormat::AnthropicMessages,
        &valid_empty_output,
        &TranslationPolicy::default(),
    ) {
        panic!("Anthropic null system and zero max_tokens should be accepted: {error}");
    }
}

// --- Invalid-role rejection ----------------------------------
// A transparent router must reject the same payloads the upstream provider
// would, rather than silently coercing an unknown role (e.g. "api") to `user`
// and returning a success. Only genuinely-unknown role strings are rejected;
// missing and known-but-unmapped roles keep their historical mapping.

// Builds a single-message request body for `format` carrying `role`.
fn single_message_request(format: WireFormat, role: &str) -> Value {
    if format == WireFormat::OpenAiResponses {
        json!({
            "model": "gpt-4o",
            "input": [{"type": "message", "role": role, "content": "hi"}],
        })
    } else {
        json!({
            "model": "gpt-4o",
            "max_tokens": 16,
            "messages": [{"role": role, "content": "hi"}],
        })
    }
}

#[test]
fn openai_chat_request_rejects_unknown_role() {
    let engine = TranslationEngine::default();
    let body = single_message_request(WireFormat::OpenAiChat, "api");
    match engine.translate_request(
        WireFormat::OpenAiChat,
        WireFormat::OpenAiChat,
        &body,
        &TranslationPolicy::default(),
    ) {
        Ok(output) => panic!("unknown role must be rejected, got Ok: {output:?}"),
        Err(err) => {
            assert_eq!(err.kind(), "InvalidValue");
            assert!(
                err.to_string().contains("api"),
                "error should name the offending value: {err}"
            );
        }
    }
}

#[test]
fn openai_responses_request_rejects_unknown_role() {
    let engine = TranslationEngine::default();
    let body = single_message_request(WireFormat::OpenAiResponses, "api");
    match engine.translate_request(
        WireFormat::OpenAiResponses,
        WireFormat::OpenAiChat,
        &body,
        &TranslationPolicy::default(),
    ) {
        Ok(output) => panic!("unknown role must be rejected, got Ok: {output:?}"),
        Err(err) => assert_eq!(err.kind(), "InvalidValue"),
    }
}

#[test]
fn anthropic_request_rejects_unknown_role() {
    let engine = TranslationEngine::default();
    let body = single_message_request(WireFormat::AnthropicMessages, "api");
    match engine.translate_request(
        WireFormat::AnthropicMessages,
        WireFormat::OpenAiChat,
        &body,
        &TranslationPolicy::default(),
    ) {
        Ok(output) => panic!("unknown role must be rejected, got Ok: {output:?}"),
        Err(err) => assert_eq!(err.kind(), "InvalidValue"),
    }
}

// Codex/cross-format safety: a known (if legacy) role such as OpenAI's
// `function` must NOT be rejected — it keeps its historical coercion to
// `user`. Only genuinely-unknown strings are rejected.
#[test]
fn openai_chat_request_accepts_legacy_function_role() {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "gpt-4o",
        "messages": [{"role": "function", "name": "lookup", "content": "ok"}],
    });
    match engine.translate_request(
        WireFormat::OpenAiChat,
        WireFormat::OpenAiChat,
        &body,
        &TranslationPolicy::default(),
    ) {
        Ok(_) => {}
        Err(err) => panic!("known legacy role must be accepted, got error: {err}"),
    }
}

// When all tools are dropped during Responses→Chat translation, tool_choice must
// also be omitted — emitting tool_choice without tools causes upstream 400s.
#[test]
fn responses_to_chat_drops_tool_choice_when_all_tools_unsupported() -> TestResult {
    let engine = TranslationEngine::default();
    // Only anonymous Anthropic-style tools, which Chat Completions cannot represent.
    let body = json!({
        "model": "gpt-4",
        "input": "do something",
        "tools": [{"description": "mystery", "input_schema": {}}],
        "tool_choice": "required"
    });
    let output = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;
    assert!(
        output.get("tools").is_none(),
        "tools must be absent when all tools are dropped"
    );
    assert!(
        output.get("tool_choice").is_none(),
        "tool_choice must be absent when tools are dropped"
    );
    Ok(())
}

// When supported tools survive translation, tool_choice must be preserved alongside them.
#[test]
fn responses_to_chat_preserves_tool_choice_when_tools_survive() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "gpt-4",
        "input": "run it",
        "tools": [{
            "type": "function",
            "id": "exec",
            "description": "Run a command.",
            "inputSchema": {
                "jsonSchema": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"]
                }
            }
        }],
        "tool_choice": "required"
    });
    let output = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;
    assert!(output.get("tools").is_some(), "tools must be present");
    assert_eq!(
        output["tool_choice"], "required",
        "tool_choice must be preserved with tools"
    );
    Ok(())
}

// Responses requires function-call arguments to be a JSON-encoded string.
#[test]
fn anthropic_tool_use_encodes_responses_arguments_as_json_string() -> TestResult {
    let engine = TranslationEngine::default();
    let raw_id = "functions.list_skills:0";
    let body = json!({
        "model": "claude-sonnet",
        "messages": [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": sanitize_anthropic_tool_use_id(raw_id),
                    "name": "get_weather",
                    "input": {"city": "SF"}
                }]
            }
        ],
        "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
        "max_tokens": 64
    });

    let output = engine
        .translate_request(
            WireFormat::AnthropicMessages,
            WireFormat::OpenAiResponses,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    let call = output["input"]
        .as_array()
        .and_then(|items| items.iter().find(|item| item["type"] == "function_call"))
        .ok_or("expected a function_call input item")?;
    let arguments = call["arguments"]
        .as_str()
        .ok_or("function_call arguments must be a JSON string")?;
    assert_eq!(call["call_id"], raw_id);
    assert_eq!(
        serde_json::from_str::<Value>(arguments)?,
        json!({"city": "SF"})
    );
    Ok(())
}

// Anthropic private thinking has no valid representation in Responses input.
#[test]
fn anthropic_thinking_is_dropped_from_responses_input() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "claude-sonnet",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "read foo.py"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "private chain of thought", "signature": "sig"},
                {"type": "text", "text": "Reading it."},
                {"type": "tool_use", "id": "tu_1", "name": "read_file", "input": {"path": "foo.py"}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "print(1)"}
            ]}
        ]
    });

    let output = engine
        .translate_request(
            WireFormat::AnthropicMessages,
            WireFormat::OpenAiResponses,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    let input = output["input"]
        .as_array()
        .ok_or("Responses input should be an array")?;
    assert!(input.iter().all(|item| item["type"] != "reasoning"));
    assert!(!json_contains_content_type(&output, "reasoning_text"));
    assert!(!output.to_string().contains("private chain of thought"));
    assert!(input.iter().any(|item| item["type"] == "function_call"));
    assert!(
        input
            .iter()
            .any(|item| item["type"] == "function_call_output")
    );
    Ok(())
}

// Verifies a flat Responses `input_file` decodes instead of being dropped.
//
// The Responses wire carries `file_data`/`filename` directly on the block, while
// `decode_file_source` read a direct `file_id` but not a direct `file_data`, so a
// Responses file fell through to `FileSource::Raw`. Chat's raw file encoder maps
// only Anthropic `document` blocks and returns `None` otherwise -- so the file did
// not merely lose its filename, it vanished from the request entirely.
#[test]
fn responses_flat_file_data_survives_into_chat() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "gpt-5.4-mini",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "read this"},
                {
                    "type": "input_file",
                    "file_data": "JVBERi0xLjQK",
                    "filename": "report.pdf"
                }
            ]
        }]
    });

    let output = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    let content = output["messages"][0]["content"]
        .as_array()
        .ok_or("Chat content should be an array")?;
    let file = content
        .iter()
        .find(|block| block["type"] == "file")
        .ok_or("the file must survive into Chat, not be dropped as unmappable raw")?;
    assert_eq!(file["file"]["file_data"], "JVBERi0xLjQK");
    assert_eq!(file["file"]["filename"], "report.pdf");
    Ok(())
}

// Verifies parallel tool calls serialize as adjacent call/output pairs so
// upstreams that resolve tool outputs by adjacency match the right call.
#[test]
fn responses_parallel_tool_calls_pair_with_their_outputs() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = TranslationPolicy {
        preservation: switchyard_translation::PreservationPolicy::Disabled,
        ..TranslationPolicy::default()
    };
    let body = json!({
        "model": "gpt-5",
        "input": [
            {"type": "message", "role": "user", "content": "Inspect"},
            {"type": "function_call", "name": "shell", "call_id": "call-a", "arguments": "{}"},
            {"type": "function_call", "name": "shell", "call_id": "call-b", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call-a", "output": "a"},
            {"type": "function_call_output", "call_id": "call-b", "output": "b"}
        ]
    });

    let output = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiResponses,
            &body,
            &policy,
        )?
        .body;

    let input = output["input"].as_array().ok_or("input is not an array")?;
    let pairs = input
        .iter()
        .filter_map(|item| Some((item["type"].as_str()?, item["call_id"].as_str()?)))
        .collect::<Vec<_>>();
    assert_eq!(
        pairs,
        vec![
            ("function_call", "call-a"),
            ("function_call_output", "call-a"),
            ("function_call", "call-b"),
            ("function_call_output", "call-b"),
        ]
    );
    Ok(())
}

// Verifies Codex `compaction_trigger` marker items never reach the upstream.
#[test]
fn responses_codex_compaction_markers_are_stripped() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = TranslationPolicy {
        preservation: switchyard_translation::PreservationPolicy::Disabled,
        ..TranslationPolicy::default()
    };
    let body = json!({
        "model": "gpt-5",
        "input": [
            {"type": "message", "role": "user", "content": "Continue"},
            {"type": "compaction_trigger"},
            {"type": "function_call", "name": "shell", "call_id": "call-a", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call-a", "output": "ok"}
        ]
    });

    let output = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiResponses,
            &body,
            &policy,
        )?
        .body;

    let types = output["input"]
        .as_array()
        .ok_or("input is not an array")?
        .iter()
        .filter_map(|item| item["type"].as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        types,
        vec!["message", "function_call", "function_call_output"]
    );
    Ok(())
}

// Codex drives GPT-5 models with freeform ("custom") tools. Through a Responses upstream the
// definition must go out verbatim and the replayed history must keep `custom_tool_call` items
// with their raw `input`; through a chat upstream the tool degrades to a single-argument function.
#[test]
fn responses_request_round_trips_custom_tools_and_custom_tool_calls() -> TestResult {
    let engine = TranslationEngine::default();
    let custom_tool = json!({
        "type": "custom",
        "name": "exec",
        "description": "Runs a shell command.",
        "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.*/"}
    });
    let body = json!({
        "model": "gpt-5.6-luna",
        "input": [
            {"type": "message", "role": "user", "content": "List files"},
            {"type": "custom_tool_call", "call_id": "call_1", "name": "exec", "input": "ls -la"},
            {"type": "custom_tool_call_output", "call_id": "call_1", "output": "README.md"}
        ],
        "tools": [
            custom_tool,
            {"type": "function", "name": "update_plan", "description": "Plan", "parameters": {"type": "object"}}
        ]
    });
    let policy = TranslationPolicy {
        preservation: switchyard_translation::PreservationPolicy::Disabled,
        ..TranslationPolicy::default()
    };

    let same = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiResponses,
            &body,
            &policy,
        )?
        .body;
    let tools = same["tools"].as_array().ok_or("tools should be an array")?;
    assert!(
        tools.iter().any(|tool| tool == &custom_tool),
        "custom tool must be re-emitted verbatim: {tools:?}"
    );
    let input = same["input"].as_array().ok_or("input should be an array")?;
    let call = input
        .iter()
        .find(|item| item["type"] == "custom_tool_call")
        .ok_or("history must keep the custom_tool_call")?;
    assert_eq!(call["name"], "exec");
    assert_eq!(call["call_id"], "call_1");
    assert_eq!(call["input"], "ls -la");
    assert!(call.get("arguments").is_none(), "{call}");
    let output = input
        .iter()
        .find(|item| item["type"] == "custom_tool_call_output")
        .ok_or("history must keep the custom_tool_call_output")?;
    assert_eq!(output["call_id"], "call_1");

    let chat = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;
    let exec = chat["tools"]
        .as_array()
        .ok_or("chat tools should be an array")?
        .iter()
        .find(|tool| tool["function"]["name"] == "exec")
        .ok_or("chat upstream should still see the tool")?;
    assert_eq!(
        exec["function"]["parameters"]["required"],
        json!(["input"]),
        "{exec}"
    );
    Ok(())
}

// Codex sends GPT-5 requests in the Responses-lite shape: no top-level `tools`, empty
// `instructions`, the tool definitions inside `input[0]` as an `additional_tools` developer
// item, and the base instructions as a developer message. Those definitions are the request's
// tools, the item must not leak into the conversation, and a Responses upstream must receive
// the request in the same shape.
#[test]
fn responses_lite_additional_tools_item_is_the_tool_list() -> TestResult {
    let engine = TranslationEngine::default();
    let tools = json!([
        {"type": "custom", "name": "exec", "description": "Run JS.",
         "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.*/"}},
        {"type": "function", "name": "update_plan", "description": "Plan",
         "parameters": {"type": "object", "properties": {}}}
    ]);
    let body = json!({
        "model": "gpt-5.6-luna-switchyard",
        "instructions": "",
        "input": [
            {"type": "additional_tools", "role": "developer", "tools": tools},
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "You are Codex."}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "List files"}]},
            {"type": "custom_tool_call", "call_id": "call_1", "name": "exec", "input": "ls"},
            {"type": "custom_tool_call_output", "call_id": "call_1", "output": "README.md"}
        ],
        "stream": true
    });
    let policy = TranslationPolicy {
        preservation: switchyard_translation::PreservationPolicy::Disabled,
        ..TranslationPolicy::default()
    };

    let decoded = engine.decode_request(WireFormat::OpenAiResponses, &body, &policy)?;
    let names = decoded
        .request
        .tools
        .iter()
        .map(|tool| tool.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(names, vec!["exec", "update_plan"]);
    assert!(
        !decoded.request.messages.iter().any(|message| {
            message
                .content
                .iter()
                .any(|block| matches!(block, switchyard_protocol::ContentBlock::Unknown { .. }))
        }),
        "the additional_tools item must not become a conversation message"
    );

    let same = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiResponses,
            &body,
            &policy,
        )?
        .body;
    assert!(same.get("tools").is_none(), "{same}");
    let input = same["input"].as_array().ok_or("input should be an array")?;
    assert_eq!(input[0]["type"], "additional_tools");
    assert_eq!(input[0]["role"], "developer");
    assert_eq!(input[0]["tools"], tools);
    assert!(
        input
            .iter()
            .skip(1)
            .all(|item| item["type"] != "additional_tools"),
        "{same}"
    );
    assert!(
        input
            .iter()
            .any(|item| item["type"] == "custom_tool_call" && item["input"] == "ls"),
        "{same}"
    );

    let chat = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;
    let chat_tools = chat["tools"]
        .as_array()
        .ok_or("chat tools should be an array")?;
    let chat_names = chat_tools
        .iter()
        .map(|tool| tool["function"]["name"].as_str().unwrap_or_default())
        .collect::<Vec<_>>();
    assert_eq!(chat_names, vec!["exec", "update_plan"]);
    let messages = chat["messages"]
        .as_array()
        .ok_or("messages should be an array")?;
    assert!(
        !messages
            .iter()
            .any(|message| message["content"].to_string().contains("additional_tools")),
        "{chat}"
    );
    Ok(())
}

/// A lite request whose whole conversation is one user text still carries its tools: the
/// encoder widens the string `input` to an array so the `additional_tools` item has a place.
#[test]
fn responses_lite_additional_tools_survive_a_single_text_input() -> TestResult {
    let engine = TranslationEngine::default();
    let tools = json!([
        {"type": "custom", "name": "exec", "description": "Run a shell command.",
         "format": {"type": "text"}}
    ]);
    let body = json!({
        "model": "gpt-5.6-luna-switchyard",
        "instructions": "",
        "input": [
            {"type": "additional_tools", "role": "developer", "tools": tools},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "List files"}]}
        ]
    });
    let policy = TranslationPolicy {
        preservation: switchyard_translation::PreservationPolicy::Disabled,
        ..TranslationPolicy::default()
    };

    let same = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiResponses,
            &body,
            &policy,
        )?
        .body;
    assert!(same.get("tools").is_none(), "{same}");
    let input = same["input"]
        .as_array()
        .ok_or("input must be an array when tools ride inside it")?;
    assert_eq!(input[0]["type"], "additional_tools");
    assert_eq!(input[0]["tools"], tools);
    assert_eq!(input.len(), 2, "{same}");
    assert_eq!(input[1]["role"], "user");
    assert_eq!(input[1]["content"], "List files");
    Ok(())
}

// Parallel freeform calls must pair with their `custom_tool_call_output` items the same way
// function calls do (#664), so upstreams that resolve outputs by adjacency see call, output,
// call, output.
#[test]
fn responses_parallel_custom_tool_calls_pair_with_their_outputs() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = TranslationPolicy {
        preservation: switchyard_translation::PreservationPolicy::Disabled,
        ..TranslationPolicy::default()
    };
    let body = json!({
        "model": "gpt-5.6-luna",
        "tools": [{"type": "custom", "name": "exec", "description": "Run.", "format": {"type": "text"}}],
        "input": [
            {"type": "message", "role": "user", "content": "Inspect"},
            {"type": "custom_tool_call", "name": "exec", "call_id": "call-a", "input": "ls"},
            {"type": "custom_tool_call", "name": "exec", "call_id": "call-b", "input": "pwd"},
            {"type": "custom_tool_call_output", "call_id": "call-a", "output": "a"},
            {"type": "custom_tool_call_output", "call_id": "call-b", "output": "b"}
        ]
    });

    let output = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiResponses,
            &body,
            &policy,
        )?
        .body;

    let input = output["input"].as_array().ok_or("input is not an array")?;
    let pairs = input
        .iter()
        .filter_map(|item| Some((item["type"].as_str()?, item["call_id"].as_str()?)))
        .collect::<Vec<_>>();
    assert_eq!(
        pairs,
        vec![
            ("custom_tool_call", "call-a"),
            ("custom_tool_call_output", "call-a"),
            ("custom_tool_call", "call-b"),
            ("custom_tool_call_output", "call-b"),
        ]
    );
    Ok(())
}

// Verifies Chat image and file parts encode as wire-valid Responses input parts,
// not serialized IR enums (image_url must be a string and file fields must be flat).
#[test]
fn openai_chat_image_and_file_parts_translate_to_valid_responses_input() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "gpt-4o",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe these."},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,YQ==", "detail": "high"}
                },
                {"type": "file", "file": {"file_id": "file_123"}},
                {
                    "type": "file",
                    "file": {"file_data": "ZG9jdW1lbnQ=", "filename": "report.pdf"}
                }
            ]
        }]
    });

    let output = engine
        .translate_request(
            WireFormat::OpenAiChat,
            WireFormat::OpenAiResponses,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(
        output["input"][0]["content"],
        json!([
            {"type": "input_text", "text": "Describe these."},
            {
                "type": "input_image",
                "image_url": "data:image/png;base64,YQ==",
                "detail": "high"
            },
            {"type": "input_file", "file_id": "file_123"},
            {
                "type": "input_file",
                "file_data": "ZG9jdW1lbnQ=",
                "filename": "report.pdf"
            }
        ])
    );
    Ok(())
}

// Verifies Anthropic base64 media becomes valid OpenAI image and file parts.
#[test]
fn anthropic_base64_media_translates_to_openai_formats() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 64,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "aW1hZ2U="
                    }
                },
                {
                    "type": "document",
                    "title": "report.pdf",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": "ZG9jdW1lbnQ="
                    }
                }
            ]
        }]
    });

    for policy in [TranslationPolicy::default(), normalized_policy()] {
        let mut files = Vec::new();
        for target in [WireFormat::OpenAiChat, WireFormat::OpenAiResponses] {
            let output = engine
                .translate_request(WireFormat::AnthropicMessages, target, &body, &policy)?
                .body;
            let content = if target == WireFormat::OpenAiChat {
                &output["messages"][0]["content"]
            } else {
                &output["input"][0]["content"]
            };
            if target == WireFormat::OpenAiChat {
                assert_eq!(content[0], json!({"type": "text", "text": "What is this?"}));
                assert_eq!(
                    content[1]["image_url"]["url"],
                    "data:image/png;base64,aW1hZ2U="
                );
                assert_eq!(content[2]["type"], "file");
                files.push(content[2]["file"].clone());
            } else {
                assert_eq!(
                    content[0],
                    json!({"type": "input_text", "text": "What is this?"})
                );
                assert_eq!(content[1]["image_url"], "data:image/png;base64,aW1hZ2U=");
                assert_eq!(content[2]["type"], "input_file");
                let mut file = content[2].clone();
                file.as_object_mut()
                    .ok_or("file must be an object")?
                    .remove("type");
                files.push(file);
            }
            let restored = engine
                .translate_request(
                    target,
                    WireFormat::AnthropicMessages,
                    &output,
                    &normalized_policy(),
                )?
                .body;
            assert_eq!(
                restored["messages"][0]["content"],
                body["messages"][0]["content"]
            );
        }
        let expected_file = json!({
            "file_data": "data:application/pdf;base64,ZG9jdW1lbnQ=",
            "filename": "report.pdf"
        });
        assert_eq!(files, vec![expected_file.clone(), expected_file]);
    }
    Ok(())
}

// Verifies tool-result image and file content survives translation in both directions between
// Anthropic Messages and OpenAI Responses.
#[test]
fn tool_result_media_survives_anthropic_and_responses_translation() -> TestResult {
    let engine = TranslationEngine::default();
    let image = json!({
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "aW1hZ2U="}
    });
    let document = json!({
        "type": "document",
        "title": "report.pdf",
        "source": {"type": "base64", "media_type": "application/pdf", "data": "ZG9jdW1lbnQ="}
    });
    let anthropic_request = |content: Value| {
        json!({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 32,
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "capture", "input": {}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": content}
                ]}
            ]
        })
    };
    let to_responses = |body: &Value| -> TestResult<Value> {
        let output = engine
            .translate_request(
                WireFormat::AnthropicMessages,
                WireFormat::OpenAiResponses,
                body,
                &TranslationPolicy::default(),
            )?
            .body;
        Ok(output["input"][1]["output"].clone())
    };

    // Anthropic -> Responses: typed parts, an image-only result, and unchanged plain text.
    assert_eq!(
        to_responses(&anthropic_request(json!([
            {"type": "text", "text": "media attached"},
            image,
            document
        ])))?,
        json!([
            {"type": "input_text", "text": "media attached"},
            {"type": "input_image", "image_url": "data:image/png;base64,aW1hZ2U="},
            {"type": "input_file", "file_data": "data:application/pdf;base64,ZG9jdW1lbnQ=", "filename": "report.pdf"}
        ])
    );
    assert_eq!(
        to_responses(&anthropic_request(json!([image])))?,
        json!([{"type": "input_image", "image_url": "data:image/png;base64,aW1hZ2U="}])
    );
    assert_eq!(
        to_responses(&anthropic_request(json!("plain")))?,
        json!("plain")
    );

    // Responses -> Anthropic: typed blocks instead of one JSON string.
    let body = json!({
        "model": "gpt-5.2",
        "input": [
            {"type": "function_call", "call_id": "call_1", "name": "capture", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": [
                {"type": "input_text", "text": "media attached"},
                {"type": "input_image", "image_url": "data:image/png;base64,aW1hZ2U=", "detail": "low"},
                {
                    "type": "input_file",
                    "file_data": "data:application/pdf;base64,ZG9jdW1lbnQ=",
                    "filename": "report.pdf"
                }
            ]}
        ]
    });
    let output = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            WireFormat::AnthropicMessages,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;
    let result = &output["messages"][1]["content"][0];
    assert_eq!(result["type"], "tool_result");
    assert_eq!(result["tool_use_id"], "call_1");
    assert_eq!(
        result["content"],
        json!([
            {"type": "text", "text": "media attached"},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "aW1hZ2U="}
            },
            {
                "type": "document",
                "title": "report.pdf",
                "source": {"type": "base64", "media_type": "application/pdf", "data": "ZG9jdW1lbnQ="}
            }
        ])
    );
    Ok(())
}

// Verifies Responses tool outputs answering calls held behind `previous_response_id` stay
// tool results, so a user-turn classifier does not treat them as a new user message, and
// re-encode with their original item types.
#[test]
fn responses_stored_tool_outputs_stay_tool_results() -> TestResult {
    let engine = TranslationEngine::default();
    let outputs = json!([
        {"type": "function_call_output", "call_id": "call_weather", "output": "sunny"},
        {"type": "custom_tool_call_output", "call_id": "call_shell", "output": "README.md"}
    ]);
    let body = json!({
        "model": "gpt-5.2",
        "previous_response_id": "resp_1",
        "input": outputs
    });

    let request = engine
        .decode_request(
            WireFormat::OpenAiResponses,
            &body,
            &TranslationPolicy::default(),
        )?
        .request;
    let tool_call_ids = request
        .messages
        .iter()
        .flat_map(|message| &message.content)
        .map(|block| match block {
            ContentBlock::ToolResult(result) => Ok(result.tool_call_id.as_str()),
            other => Err(format!("expected a tool result, got {other:?}")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(tool_call_ids, ["call_weather", "call_shell"]);

    let output = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiResponses,
            &body,
            &normalized_policy(),
        )?
        .body;
    assert_eq!(output["input"], outputs);
    Ok(())
}
