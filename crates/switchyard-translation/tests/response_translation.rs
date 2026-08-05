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

// Verifies Responses reasoning and its following message remain one semantic assistant output.
#[test]
fn responses_reasoning_and_message_preserve_the_final_answer() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "id": "resp_test",
        "object": "response",
        "model": "gpt-reasoning",
        "status": "completed",
        "output": [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "private reasoning"}]
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Visible answer"}]
            }
        ]
    });

    let anthropic = engine
        .translate_response(
            WireFormat::OpenAiResponses,
            WireFormat::AnthropicMessages,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;
    assert_eq!(
        anthropic["content"],
        json!([
            {"type": "thinking", "thinking": "private reasoning", "signature": ""},
            {"type": "text", "text": "Visible answer"}
        ])
    );

    let chat = engine
        .translate_response(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;
    assert_eq!(chat["choices"][0]["message"]["content"], "Visible answer");
    assert_eq!(
        chat["choices"][0]["message"]["reasoning_content"],
        "private reasoning"
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

// Verifies OpenRouter's cache-write field and the legacy alias normalize identically.
#[test]
fn openai_chat_cache_write_aliases_translate_to_anthropic_usage_fields() -> TestResult {
    let engine = TranslationEngine::default();
    for cache_write_field in ["cache_write_tokens", "cache_creation_tokens"] {
        let mut body = json!({
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
                "prompt_tokens_details": {"cached_tokens": 70}
            }
        });
        body["usage"]["prompt_tokens_details"][cache_write_field] = json!(10);

        let output = engine
            .translate_response(
                WireFormat::OpenAiChat,
                WireFormat::AnthropicMessages,
                &body,
                &TranslationPolicy::default(),
            )?
            .body;

        assert_eq!(output["usage"]["input_tokens"], 20);
        assert_eq!(output["usage"]["cache_read_input_tokens"], 70);
        assert_eq!(output["usage"]["cache_creation_input_tokens"], 10);
        assert_eq!(output["usage"]["output_tokens"], 5);
    }
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
        json!({
            "input_tokens": 4,
            "output_tokens": 3,
            "total_tokens": 7,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0}
        })
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

// Verifies both Responses usage detail objects are emitted even when the upstream reports no
// cache or reasoning breakdown. The Responses schema types them as required, so omitting them
// makes the payload unparseable by OpenAI-SDK clients.
#[test]
fn openai_chat_usage_without_breakdowns_still_emits_responses_usage_details() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "id": "chatcmpl-test",
        "model": "plain-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hi"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 41, "completion_tokens": 3, "total_tokens": 44}
    });

    let output = engine
        .translate_response(
            WireFormat::OpenAiChat,
            WireFormat::OpenAiResponses,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(
        output["usage"],
        json!({
            "input_tokens": 41,
            "output_tokens": 3,
            "total_tokens": 44,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0}
        })
    );
    Ok(())
}

// Verifies a partial breakdown does not suppress the other detail object: an upstream that
// reports cached tokens but no reasoning tokens must still carry both.
#[test]
fn openai_chat_cache_only_usage_still_emits_reasoning_details() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "id": "chatcmpl-test",
        "model": "cache-only-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hi"},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 41,
            "completion_tokens": 3,
            "total_tokens": 44,
            "prompt_tokens_details": {"cached_tokens": 32}
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

    assert_eq!(
        output["usage"]["input_tokens_details"],
        json!({"cached_tokens": 32})
    );
    assert_eq!(
        output["usage"]["output_tokens_details"],
        json!({"reasoning_tokens": 0})
    );
    Ok(())
}

// Verifies a token-limit stop is reported as an incomplete Responses result.
#[test]
fn openai_chat_length_finish_translates_to_incomplete_responses_status() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "id": "chatcmpl-test",
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Half an ans"},
            "finish_reason": "length"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    });

    let output = engine
        .translate_response(
            WireFormat::OpenAiChat,
            WireFormat::OpenAiResponses,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(output["status"], "incomplete");
    assert_eq!(
        output["incomplete_details"],
        json!({"reason": "max_output_tokens"})
    );
    assert_eq!(output["output"][0]["status"], "incomplete");
    Ok(())
}

// Verifies a truncated Responses source keeps its stop reason when re-encoded.
#[test]
fn incomplete_responses_source_survives_translation() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "id": "resp_1",
        "object": "response",
        "model": "gpt-4o",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Half an ans"}]
        }],
        "usage": {"input_tokens": 10, "output_tokens": 1, "total_tokens": 11}
    });

    let output = engine
        .translate_response(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(output["choices"][0]["finish_reason"], "length");
    Ok(())
}
