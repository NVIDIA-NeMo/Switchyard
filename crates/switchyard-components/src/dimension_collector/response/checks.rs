// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure response-side checks emitted by [`super::extract_response_signals`].
//!
//! Each check is a `fn(&ChatResponse) -> bool` operating on buffered JSON
//! bodies only — streams are treated as "no signal" upstream because they
//! can't be introspected without consuming the stream. Adding a streaming
//! variant is a separate effort (collect-as-you-go, emit at end-of-stream)
//! tracked outside the first-cut scope.
//!
//! Each fn is one OpenAI / Anthropic shape match. Keep them shape-aware
//! rather than running a uniform JSONPath over every wire — the wire
//! formats are different enough that uniform JSONPath would either be
//! lossy or 4x more code than direct shape access.

use crate::ChatResponse;
use serde_json::Value;

/// Returns `true` if any `tool_calls[].function.arguments` string is not
/// valid JSON. Operates on OpenAI-Chat and Anthropic responses.
///
/// Mirrors the failure mode where a weak model emits a tool call shape
/// that looks well-formed at the top level but has malformed JSON in the
/// arguments field. Common with smaller models under tight token budgets.
pub fn is_malformed_tool_call(response: &ChatResponse) -> bool {
    let Some(body) = response.body() else {
        return false;
    };
    if let Some(tool_calls) = openai_chat_tool_calls(body) {
        return tool_calls_have_invalid_args(tool_calls);
    }
    if let Some(content_blocks) = anthropic_content_blocks(body) {
        return anthropic_tool_use_blocks_have_invalid_args(content_blocks);
    }
    false
}

/// Returns `true` if the response carries no content **and** no tool calls
/// **and** `finish_reason` is not the legitimate `"tool_calls"` /
/// `"tool_use"` sentinel. Distinguishes "model said nothing" from "model
/// finished with a tool call" — the second is normal, the first is a
/// quality failure.
pub fn is_empty_response(response: &ChatResponse) -> bool {
    let Some(body) = response.body() else {
        return false;
    };
    if let Some(choice) = openai_first_choice(body) {
        let message = choice.get("message").and_then(Value::as_object);
        let content_empty = message
            .and_then(|m| m.get("content"))
            .map(content_is_empty)
            .unwrap_or(true);
        let tool_calls_empty = message
            .and_then(|m| m.get("tool_calls"))
            .and_then(Value::as_array)
            .is_none_or(|calls| calls.is_empty());
        let finish_reason = choice
            .get("finish_reason")
            .and_then(Value::as_str)
            .unwrap_or("");
        return content_empty
            && tool_calls_empty
            && finish_reason != "tool_calls"
            && finish_reason != "tool_use";
    }
    if let Some(blocks) = anthropic_content_blocks(body) {
        let any_content = blocks.iter().any(|block| {
            let kind = block.get("type").and_then(Value::as_str).unwrap_or("");
            kind == "text" || kind == "tool_use"
        });
        let stop_reason = body
            .get("stop_reason")
            .and_then(Value::as_str)
            .unwrap_or("");
        return !any_content && stop_reason != "tool_use";
    }
    false
}

/// Returns `true` if the response was truncated by the model-side
/// `max_tokens` budget — `finish_reason == "length"` on OpenAI Chat,
/// `stop_reason == "max_tokens"` on Anthropic. A common weak-model
/// failure mode under cost-pressured `max_tokens` caps.
pub fn is_truncated_completion(response: &ChatResponse) -> bool {
    let Some(body) = response.body() else {
        return false;
    };
    if let Some(choice) = openai_first_choice(body) {
        return choice
            .get("finish_reason")
            .and_then(Value::as_str)
            .map(|reason| reason == "length")
            .unwrap_or(false);
    }
    body.get("stop_reason")
        .and_then(Value::as_str)
        .map(|reason| reason == "max_tokens")
        .unwrap_or(false)
}

/// Returns `true` if any emitted tool call is missing a top-level required
/// field per the OpenAI Chat / Anthropic tool-call shape itself.
///
/// Note: this checks *shape* requirements (must have `name`, must have
/// `arguments`/`input`), **not** the per-tool argument schemas. The
/// per-tool schema check requires the request's tool declarations and is
/// future work — the shape check catches the most common malformed
/// tool-call cases.
pub fn is_missing_required_args(response: &ChatResponse) -> bool {
    let Some(body) = response.body() else {
        return false;
    };
    if let Some(tool_calls) = openai_chat_tool_calls(body) {
        return tool_calls.iter().any(|call| {
            let function = call.get("function").and_then(Value::as_object);
            let has_name = function
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
                .is_some_and(|name| !name.is_empty());
            let has_args = function
                .and_then(|f| f.get("arguments"))
                .and_then(Value::as_str)
                .is_some();
            !(has_name && has_args)
        });
    }
    if let Some(blocks) = anthropic_content_blocks(body) {
        return blocks.iter().any(|block| {
            let kind = block.get("type").and_then(Value::as_str).unwrap_or("");
            if kind != "tool_use" {
                return false;
            }
            let has_name = block
                .get("name")
                .and_then(Value::as_str)
                .is_some_and(|name| !name.is_empty());
            let has_input = block.get("input").is_some();
            !(has_name && has_input)
        });
    }
    false
}

// ─── Shape-access helpers ────────────────────────────────────────────────

fn openai_first_choice(body: &Value) -> Option<&serde_json::Map<String, Value>> {
    body.get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(Value::as_object)
}

fn openai_chat_tool_calls(body: &Value) -> Option<&Vec<Value>> {
    openai_first_choice(body)
        .and_then(|choice| choice.get("message"))
        .and_then(Value::as_object)
        .and_then(|message| message.get("tool_calls"))
        .and_then(Value::as_array)
}

fn anthropic_content_blocks(body: &Value) -> Option<&Vec<Value>> {
    body.get("content").and_then(Value::as_array)
}

fn tool_calls_have_invalid_args(tool_calls: &[Value]) -> bool {
    tool_calls.iter().any(|call| {
        let Some(args) = call
            .get("function")
            .and_then(|function| function.get("arguments"))
            .and_then(Value::as_str)
        else {
            // Missing arguments is `is_missing_required_args`'s concern;
            // here we only flag *malformed JSON* in the provided string.
            return false;
        };
        // Empty-string arguments are convention for zero-arg tool calls;
        // treat as well-formed (`{}` equivalent).
        if args.is_empty() {
            return false;
        }
        serde_json::from_str::<Value>(args).is_err()
    })
}

fn anthropic_tool_use_blocks_have_invalid_args(blocks: &[Value]) -> bool {
    // Anthropic emits `input` as already-parsed JSON, not a string, so
    // there's no malformed-string failure mode equivalent to OpenAI's.
    // Flagged only if `input` is a string that fails to parse as JSON,
    // which is non-standard but defensive.
    blocks.iter().any(|block| {
        let kind = block.get("type").and_then(Value::as_str).unwrap_or("");
        if kind != "tool_use" {
            return false;
        }
        let Some(input) = block.get("input") else {
            return false;
        };
        match input {
            Value::String(s) if !s.is_empty() => serde_json::from_str::<Value>(s).is_err(),
            _ => false,
        }
    })
}

fn content_is_empty(content: &Value) -> bool {
    match content {
        Value::Null => true,
        Value::String(s) => s.trim().is_empty(),
        Value::Array(parts) => parts.iter().all(|part| {
            part.as_object()
                .and_then(|object| object.get("text"))
                .and_then(Value::as_str)
                .is_none_or(|text| text.trim().is_empty())
        }),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn openai_chat(body: Value) -> ChatResponse {
        ChatResponse::openai_completion(body)
    }

    fn anthropic(body: Value) -> ChatResponse {
        ChatResponse::anthropic_completion(body)
    }

    #[test]
    fn malformed_tool_call_openai_chat() {
        let bad = openai_chat(json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\": \"new york\""  // missing closing brace
                        }
                    }]
                }
            }]
        }));
        assert!(is_malformed_tool_call(&bad));

        let good = openai_chat(json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\": \"new york\"}"
                        }
                    }]
                }
            }]
        }));
        assert!(!is_malformed_tool_call(&good));
    }

    #[test]
    fn empty_arguments_string_is_well_formed() {
        let resp = openai_chat(json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": { "name": "ping", "arguments": "" }
                    }]
                }
            }]
        }));
        assert!(!is_malformed_tool_call(&resp));
    }

    #[test]
    fn empty_response_distinguishes_from_tool_call_finish() {
        let empty = openai_chat(json!({
            "choices": [{
                "message": { "content": null },
                "finish_reason": "stop"
            }]
        }));
        assert!(is_empty_response(&empty));

        let tool_call_finish = openai_chat(json!({
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "function": { "name": "x", "arguments": "{}" }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }));
        assert!(!is_empty_response(&tool_call_finish));
    }

    #[test]
    fn truncated_completion_finish_reason_length() {
        let truncated = openai_chat(json!({
            "choices": [{
                "message": { "content": "this got cut off..." },
                "finish_reason": "length"
            }]
        }));
        assert!(is_truncated_completion(&truncated));

        let ok = openai_chat(json!({
            "choices": [{
                "message": { "content": "all done" },
                "finish_reason": "stop"
            }]
        }));
        assert!(!is_truncated_completion(&ok));
    }

    #[test]
    fn truncated_anthropic_stop_reason_max_tokens() {
        let truncated = anthropic(json!({
            "content": [{ "type": "text", "text": "got cut off" }],
            "stop_reason": "max_tokens"
        }));
        assert!(is_truncated_completion(&truncated));
    }

    #[test]
    fn missing_required_args_flags_nameless_or_argless_tool_calls() {
        let no_name = openai_chat(json!({
            "choices": [{
                "message": {
                    "tool_calls": [{ "function": { "arguments": "{}" } }]
                }
            }]
        }));
        assert!(is_missing_required_args(&no_name));

        let no_args = openai_chat(json!({
            "choices": [{
                "message": {
                    "tool_calls": [{ "function": { "name": "ping" } }]
                }
            }]
        }));
        assert!(is_missing_required_args(&no_args));

        let well_formed = openai_chat(json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": { "name": "ping", "arguments": "{}" }
                    }]
                }
            }]
        }));
        assert!(!is_missing_required_args(&well_formed));
    }
}
