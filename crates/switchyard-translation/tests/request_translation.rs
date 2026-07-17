// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for buffered request translation between provider formats.

use pretty_assertions::assert_eq;
use serde_json::{json, Value};
use switchyard_translation::{TranslationEngine, TranslationPolicy, WireFormat};

type TestResult = std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>;

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
    assert!(output["messages"][3]["content"]
        .as_str()
        .unwrap()
        .contains("72F"));
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
    let body = json!({
        "model": "claude-sonnet-4-20250514",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "72F"},
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
            {"role": "tool", "tool_call_id": "toolu_1", "content": "72F"},
            {"role": "user", "content": "Now summarize it."}
        ])
    );
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

// Verifies Chat-compatible Responses extension fields survive translation.
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
        "reasoning_effort": "high"
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
    assert_eq!(output["output_config"], json!({"effort": "high"}));
    assert_eq!(output["messages"][0]["role"], "user");
    assert_eq!(
        output["messages"][0]["content"][0],
        json!({"type": "text", "text": "Describe"})
    );
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

    assert_eq!(output["max_tokens"], 128_000);
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
        "call_bad_id_with_space"
    );
    assert_eq!(
        output["messages"][2]["content"],
        json!([
            {"type": "tool_result", "tool_use_id": "call_bad_id_with_space", "content": "one"},
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

// Verifies an Anthropic tool_use encodes into a Responses function_call whose
// `arguments` is a JSON-encoded string, not an object. The Responses API rejects
// an object here ("expected a string"), which broke agent tool turns routed to a
// Responses backend.
#[test]
fn anthropic_tool_use_encodes_responses_arguments_as_json_string() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "claude-sonnet-4-20250514",
        "messages": [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": "toolu_1",
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
    assert_eq!(call["name"], "get_weather");
    // Must be a string, and that string must parse back to the original object.
    let arguments = call["arguments"]
        .as_str()
        .ok_or("function_call arguments must be a JSON string")?;
    assert_eq!(
        serde_json::from_str::<Value>(arguments)?,
        json!({"city": "SF"})
    );
    Ok(())
}

// Verifies an Anthropic thinking block is dropped when encoding a request to
// OpenAI Responses — never emitted as a reasoning input item carrying content,
// which the Responses API rejects ("input[..].content: expected max length 0").
// Mirrors the OpenAI Chat encoder, which also drops thinking.
#[test]
fn anthropic_thinking_block_is_dropped_for_openai_responses_input() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 64,
        "tools": [{"name": "read_file", "input_schema": {"type": "object"}}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "read foo.py"}]},
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
    // No reasoning input item at all, and no private reasoning text leaks anywhere.
    assert!(
        input.iter().all(|item| item["type"] != "reasoning"),
        "reasoning input item must be dropped: {input:?}"
    );
    assert!(
        !json_contains_content_type(&output, "reasoning_text"),
        "reasoning text must not leak into Responses input"
    );
    assert!(
        !output.to_string().contains("private chain of thought"),
        "private reasoning must not leak into the request"
    );
    // The rest of the turn survives: the tool call and its output are still there.
    assert!(input.iter().any(|item| item["type"] == "function_call"));
    assert!(input
        .iter()
        .any(|item| item["type"] == "function_call_output"));
    Ok(())
}
