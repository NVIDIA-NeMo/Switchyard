// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests that target prompts update preserved provider bodies without losing caller fields.

use serde_json::json;
use switchyard_libsy::prepend_system_prompt;
use switchyard_protocol::{ANTHROPIC_REQUEST_KEY, LlmRequest};
use switchyard_translation::{
    TranslationEngine, TranslationPolicy, WireFormat, prepare_request_for_target,
};

type TestResult<T = ()> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

// Prepares a candidate the same way the LLM client does for a target with a prompt.
fn prepare(request: &mut LlmRequest, target: &str, prompt: &str) {
    prepare_request_for_target(request, &target.into());
    prepend_system_prompt(request, prompt);
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
                "n": 1,
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
        prepare(&mut request, "selected/model", "target prompt");
        let encoded = engine.encode_request(format, &request, &policy)?.body;

        assert_eq!(encoded, expected);
    }
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
        prepare(&mut request, "target/model", "target prompt");
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
            body[ANTHROPIC_REQUEST_KEY] = json!(true);
            body["stop"] = json!(["FALLBACK"]);
            let mut request = engine.decode_request(source, &body, &policy)?.request;
            prepare(&mut request, "target/model", "target prompt");
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
            assert!(output.get(ANTHROPIC_REQUEST_KEY).is_none());
        }
    }
    Ok(())
}
