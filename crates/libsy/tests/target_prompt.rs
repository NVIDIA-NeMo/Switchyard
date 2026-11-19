// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests that target prompts survive request encoding in each provider format.

use serde_json::json;
use switchyard_libsy::prepend_system_prompt;
use switchyard_protocol::{LlmRequest, Request};
use switchyard_translation::{
    TranslationEngine, TranslationPolicy, WireFormat, prepare_request_for_target,
};

type TestResult<T = ()> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

// This helper prepares a candidate the same way the LLM client does for a target with a prompt.
fn prepare(request: &mut LlmRequest, target: &str, prompt: &str) {
    prepare_request_for_target(request, &target.into());
    let mut routed = Request {
        llm_request: std::mem::take(request),
        ..Default::default()
    };
    prepend_system_prompt(&mut routed, prompt);
    *request = routed.llm_request;
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
            body["switchyard_anthropic_request"] = json!(true);
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

    prepare(&mut request, "target", "target prompt");

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
    prepare(&mut request, "target", "target prompt");

    let output = engine
        .encode_request(WireFormat::AnthropicMessages, &request, &policy)?
        .body;

    assert_eq!(output["messages"][1]["content"][0]["is_error"], true);
    Ok(())
}

// Target prompts force normalized re-encoding; built-in tool history must retain its wire shape.
#[test]
fn responses_builtin_tool_history_survives_target_prompt() -> TestResult {
    use switchyard_translation::Role;

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

    let roles: Vec<_> = request
        .messages
        .iter()
        .map(|message| message.role)
        .collect();
    assert_eq!(roles, [Role::Assistant, Role::Tool].repeat(3));

    prepare(&mut request, "target", "target prompt");

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

    prepare(&mut request, "gpt-5.6-sol", "Plan first.");
    let output = engine
        .encode_request(WireFormat::OpenAiResponses, &request, &policy)?
        .body;

    assert_eq!(
        output["tools"][0]["description"],
        "Tools for managing sub-agents."
    );
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
    prepare(
        &mut request,
        "openai/openai/gpt-5.6-sol",
        "[router-guidance] Continue from the current state.",
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
