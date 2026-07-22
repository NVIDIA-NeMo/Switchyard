// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for buffered response translation between provider formats.

use pretty_assertions::assert_eq;
use serde_json::json;
use switchyard_translation::{TranslationEngine, TranslationPolicy, WireFormat};

type TestResult = std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>;

// Verifies OpenAI Chat responses map to Anthropic message responses.
#[test]
fn openai_chat_response_translates_to_anthropic_message() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "id": "chatcmpl-test",
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello world"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    });

    let output = engine
        .translate_response(
            WireFormat::OpenAiChat,
            WireFormat::AnthropicMessages,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(output["type"], "message");
    assert_eq!(output["role"], "assistant");
    assert_eq!(output["model"], "gpt-4o");
    assert_eq!(
        output["content"],
        json!([{"type": "text", "text": "Hello world"}])
    );
    assert_eq!(output["stop_reason"], "end_turn");
    assert_eq!(
        output["usage"],
        json!({"input_tokens": 10, "output_tokens": 5})
    );
    Ok(())
}

// Verifies Anthropic message responses map to OpenAI Chat completions.
#[test]
fn anthropic_message_response_translates_to_openai_chat_completion() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet",
        "content": [{"type": "text", "text": "Hi there"}],
        "stop_reason": "max_tokens",
        "usage": {"input_tokens": 12, "output_tokens": 7}
    });

    let output = engine
        .translate_response(
            WireFormat::AnthropicMessages,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(output["object"], "chat.completion");
    assert_eq!(output["model"], "claude-sonnet");
    assert_eq!(output["choices"][0]["message"]["content"], "Hi there");
    assert_eq!(output["choices"][0]["finish_reason"], "length");
    assert_eq!(
        output["usage"],
        json!({"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19})
    );
    Ok(())
}

// Verifies Responses usage details survive when translating back to Chat Completions.
#[test]
fn responses_reasoning_usage_translates_to_openai_chat_usage_details() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "id": "resp_test",
        "object": "response",
        "model": "gpt-reasoning",
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Visible answer"}]
        }],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "output_tokens_details": {"reasoning_tokens": 3}
        }
    });

    let output = engine
        .translate_response(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(output["usage"]["prompt_tokens"], 10);
    assert_eq!(output["usage"]["completion_tokens"], 5);
    assert_eq!(
        output["usage"]["completion_tokens_details"],
        json!({"reasoning_tokens": 3})
    );
    Ok(())
}

// Verifies OpenAI cache usage survives the Chat-to-Responses translation used by Codex.
#[test]
fn openai_chat_cache_usage_translates_to_responses_usage_details() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "id": "chatcmpl-test",
        "model": "gpt-cached",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Cached answer"},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "total_tokens": 105,
            "prompt_tokens_details": {"cached_tokens": 80}
        }
    });

    let output = engine
        .translate_response(
            WireFormat::OpenAiChat,
            WireFormat::OpenAiResponses,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(output["usage"]["input_tokens"], 100);
    assert_eq!(
        output["usage"]["input_tokens_details"],
        json!({"cached_tokens": 80})
    );
    Ok(())
}

// Verifies OpenAI cache usage is expressed in Anthropic's standard sibling fields.
#[test]
fn openai_chat_cache_usage_translates_to_anthropic_usage_fields() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "id": "chatcmpl-test",
        "model": "gpt-cached",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Cached answer"},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "total_tokens": 105,
            "prompt_tokens_details": {"cached_tokens": 80}
        }
    });

    let output = engine
        .translate_response(
            WireFormat::OpenAiChat,
            WireFormat::AnthropicMessages,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(output["usage"]["input_tokens"], 20);
    assert_eq!(output["usage"]["cache_read_input_tokens"], 80);
    assert_eq!(output["usage"]["output_tokens"], 5);
    Ok(())
}

// Verifies Anthropic thinking response blocks become OpenAI reasoning_content.
#[test]
fn anthropic_thinking_response_translates_to_openai_reasoning_content() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus",
        "content": [
            {"type": "thinking", "thinking": "private reasoning"},
            {"type": "text", "text": "Visible answer"}
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 7}
    });

    let output = engine
        .translate_response(
            WireFormat::AnthropicMessages,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    let message = &output["choices"][0]["message"];
    assert_eq!(message["content"], "Visible answer");
    assert_eq!(message["reasoning_content"], "private reasoning");
    Ok(())
}

// Verifies OpenAI reasoning_content becomes a separate Responses reasoning item.
#[test]
fn openai_reasoning_response_translates_to_responses_reasoning_item() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "id": "chatcmpl-test",
        "model": "gpt-reasoning",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "reasoning_content": "private reasoning",
                "content": "Visible answer"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": 3,
            "total_tokens": 7,
            "completion_tokens_details": {"reasoning_tokens": 2}
        }
    });

    let output = engine
        .translate_response(
            WireFormat::OpenAiChat,
            WireFormat::OpenAiResponses,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(output["output"][0]["type"], "reasoning");
    assert_eq!(
        output["output"][0]["content"][0],
        json!({"type": "reasoning_text", "text": "private reasoning"})
    );
    assert_eq!(output["output"][1]["type"], "message");
    assert_eq!(output["output"][1]["content"][0]["text"], "Visible answer");
    assert_eq!(
        output["usage"]["output_tokens_details"],
        json!({"reasoning_tokens": 2})
    );
    Ok(())
}

// Verifies reasoning-only responses do not synthesize visible output text.
#[test]
fn openai_reasoning_only_response_translates_to_responses_reasoning_only() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "id": "chatcmpl-test",
        "model": "gpt-reasoning",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "reasoning_content": "private reasoning",
                "content": null
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}
    });

    let output = engine
        .translate_response(
            WireFormat::OpenAiChat,
            WireFormat::OpenAiResponses,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    let items = output["output"]
        .as_array()
        .ok_or("Responses output should be an array")?;
    assert_eq!(items.len(), 1);
    assert_eq!(items[0]["type"], "reasoning");
    assert_eq!(items[0]["content"][0]["text"], "private reasoning");
    Ok(())
}

// Verifies OpenAI tool-call responses become Responses function-call output items.
#[test]
fn openai_chat_response_with_tool_call_translates_to_responses_output_item() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "id": "chatcmpl-test",
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{\"q\":\"rust\"}"}
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}
    });

    let output = engine
        .translate_response(
            WireFormat::OpenAiChat,
            WireFormat::OpenAiResponses,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(output["object"], "response");
    assert_eq!(output["output"][0]["type"], "function_call");
    assert_eq!(output["output"][0]["call_id"], "call_1");
    assert_eq!(output["output"][0]["name"], "lookup");
    assert_eq!(output["output"][0]["arguments"], "{\"q\": \"rust\"}");
    assert_eq!(
        output["usage"],
        json!({"input_tokens": 4, "output_tokens": 3, "total_tokens": 7})
    );
    Ok(())
}

// Verifies mixed assistant text and tool calls both survive into Responses output.
#[test]
fn openai_chat_response_with_text_and_tool_call_translates_both_to_responses() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "id": "chatcmpl-test",
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{\"q\":\"rust\"}"}
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}
    });

    let output = engine
        .translate_response(
            WireFormat::OpenAiChat,
            WireFormat::OpenAiResponses,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(output["output"][0]["type"], "message");
    assert_eq!(output["output"][0]["content"][0]["text"], "Let me check.");
    assert_eq!(output["output"][1]["type"], "function_call");
    assert_eq!(output["output"][1]["call_id"], "call_1");
    Ok(())
}
