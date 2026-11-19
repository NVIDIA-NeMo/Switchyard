// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router guidance edits requests and keeps preserved provider bodies in sync.

use serde_json::{Map, Value, json};
use switchyard_protocol::{
    ANTHROPIC_REQUEST_KEY, ContentBlock, InstructionBlock, LlmRequest, Message, Role, WireFormat,
};

/// Prepends a system prompt and updates supported preserved provider bodies.
/// It removes bodies it cannot edit so encoding rebuilds them from the updated request.
/// Call it once per candidate on a request that has not already received a target prompt.
pub fn prepend_system_prompt(request: &mut LlmRequest, prompt: &str) {
    request.instructions.insert(
        0,
        InstructionBlock {
            role: Role::System,
            content: vec![ContentBlock::Text {
                text: prompt.to_string(),
            }],
        },
    );
    request.preservation.requests.retain(|format, body| {
        let Some(body) = body.as_object_mut() else {
            return false;
        };
        let format = format.as_str();
        if format == WireFormat::OpenAiChat.as_str() {
            prepend_openai_chat_system(body, prompt)
        } else if format == WireFormat::OpenAiResponses.as_str() {
            prepend_text_field(body, "instructions", prompt)
        } else if format == WireFormat::AnthropicMessages.as_str() {
            body.remove(ANTHROPIC_REQUEST_KEY);
            prepend_anthropic_system(body, prompt)
        } else {
            false
        }
    });
}

fn prepend_openai_chat_system(body: &mut Map<String, Value>, prompt: &str) -> bool {
    let message = json!({"role": "system", "content": prompt});
    match body.get_mut("messages") {
        None | Some(Value::Null) => {
            body.insert("messages".to_string(), Value::Array(vec![message]));
        }
        Some(Value::Array(messages)) => messages.insert(0, message),
        Some(_) => return false,
    }
    true
}

// Preserved Responses instructions and Anthropic system prompts can be strings, missing, or null.
fn prepend_text_field(body: &mut Map<String, Value>, key: &str, prompt: &str) -> bool {
    match body.get_mut(key) {
        None | Some(Value::Null) => {
            body.insert(key.to_string(), Value::String(prompt.to_string()));
        }
        Some(Value::String(text)) if text.is_empty() => {
            *text = prompt.to_string();
        }
        Some(Value::String(text)) => text.insert_str(0, &format!("{prompt}\n\n")),
        Some(_) => return false,
    }
    true
}

fn prepend_anthropic_system(body: &mut Map<String, Value>, prompt: &str) -> bool {
    if let Some(Value::Array(blocks)) = body.get_mut("system") {
        blocks.insert(0, json!({"type":"text","text":prompt}));
        true
    } else {
        prepend_text_field(body, "system", prompt)
    }
}

/// Appends temporary guidance and updates preserved provider bodies.
/// It removes bodies it cannot edit so encoding rebuilds them from the updated request.
pub(crate) fn append_note(request: &mut LlmRequest, note: &str) {
    // A separate message preserves the caller's final message boundary for OpenAI caching.
    request.messages.push(Message::text(Role::User, note));
    request.temporary_note_count += 1;
    request.preservation.requests.retain(|format, body| {
        let Some(body) = body.as_object_mut() else {
            return false;
        };
        let format = format.as_str();
        let (key, content_type) = if format == WireFormat::OpenAiChat.as_str()
            || format == WireFormat::AnthropicMessages.as_str()
        {
            ("messages", "text")
        } else if format == WireFormat::OpenAiResponses.as_str() {
            ("input", "input_text")
        } else {
            return false;
        };
        // Responses also accepts a string as a single user input.
        if key == "input"
            && let Some(Value::String(text)) = body.get_mut(key)
        {
            let original = std::mem::take(text);
            body.insert(key.into(), json!([{"role":"user", "content":original}]));
        }
        if key == "input" && body.get(key).is_none_or(Value::is_null) {
            body.insert(key.into(), Value::Array(Vec::new()));
        }
        let Some(messages) = body.get_mut(key).and_then(Value::as_array_mut) else {
            return false;
        };
        if format == WireFormat::AnthropicMessages.as_str()
            && let Some(last) = messages.last_mut()
            && last.get("role").and_then(Value::as_str) == Some("user")
        {
            // Anthropic guidance follows existing content, including any tool result.
            match last.get_mut("content") {
                Some(Value::String(text)) => {
                    let original = std::mem::take(text);
                    last["content"] = json!([
                        {"type":"text", "text":original},
                        {"type":"text", "text":note}
                    ]);
                }
                Some(Value::Array(blocks)) => blocks.push(json!({"type":"text", "text":note})),
                _ => return false,
            }
        } else {
            messages.push(json!({"role":"user", "content":[{"type":content_type,"text":note}]}));
        }
        true
    });
}

/// Clears preserved bodies and the note count after other conversation edits.
/// Subsequent encoding uses the edited request instead of replaying stale provider bodies.
pub(crate) fn drop_exact_replay(request: &mut LlmRequest) {
    request.preservation.requests.clear();
    request.temporary_note_count = 0;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use switchyard_protocol::WireFormat;
    use switchyard_translation::{TranslationEngine, TranslationPolicy};

    // Native bodies retain provider fields and tool argument text that decoding may not preserve.
    #[test]
    fn edits_preserve_native_prefixes_and_provider_fields() -> Result<(), Box<dyn std::error::Error>>
    {
        let engine = TranslationEngine::default();
        let policy = TranslationPolicy::default();
        let fixtures = [
            (
                WireFormat::OpenAiChat,
                json!({
                    "model":"auto", "messages":[
                        {"role":"developer","content":[{"type":"text","text":"Keep this instruction.","vendor":"keep"}]},
                        {"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"run","arguments":"{ \"cmd\": \"test\" }"}}]},
                        {"role":"tool","tool_call_id":"call_1","content":"exit 1","vendor":"keep"}
                    ], "metadata":{"caller":"keep"}, "temperature":0.2
                }),
            ),
            (
                WireFormat::OpenAiResponses,
                json!({
                    "model":"auto", "instructions":"Keep this instruction.", "input":[
                        {"type":"message","role":"developer","content":[{"type":"input_text","text":"Keep this too."}]},
                        {"type":"function_call","call_id":"call_1","name":"run","arguments":"{ \"cmd\": \"test\" }"},
                        {"type":"function_call_output","call_id":"call_1","output":"exit 1"}
                    ], "metadata":{"caller":"keep"}, "store":false
                }),
            ),
            (
                WireFormat::AnthropicMessages,
                json!({
                    "model":"auto", "max_tokens":32,
                    "system":[{"type":"text","text":"Keep this instruction.","cache_control":{"type":"ephemeral","ttl":"1h"}}],
                    "messages":[
                        {"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"run","input":{"cmd":"test"}}]},
                        {"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"exit 1","is_error":true,"cache_control":{"type":"ephemeral"}}]}
                    ], "metadata":{"user_id":"keep"}
                }),
            ),
        ];
        for (format, original) in fixtures {
            let mut request = engine.decode_request(format, &original, &policy)?.request;
            append_note(&mut request, "Reassess the failure.");
            let encoded = engine.encode_request(format, &request, &policy)?.body;
            let mut expected = original.clone();
            match format {
                WireFormat::OpenAiChat => expected["messages"].as_array_mut().ok_or("messages")?.push(json!({"role":"user","content":[{"type":"text","text":"Reassess the failure."}]})),
                WireFormat::OpenAiResponses => expected["input"].as_array_mut().ok_or("input")?.push(json!({"role":"user","content":[{"type":"input_text","text":"Reassess the failure."}]})),
                WireFormat::AnthropicMessages => expected["messages"][1]["content"].as_array_mut().ok_or("content")?.push(json!({"type":"text","text":"Reassess the failure."})),
            }
            assert_eq!(encoded, expected, "{format}: only the note may change");
            prepend_system_prompt(&mut request, "Use the capable tier.");
            let edited = engine.encode_request(format, &request, &policy)?.body;
            // The system prompt edit must retain the earlier note and all conversation and tool items.
            let key = if format == WireFormat::OpenAiResponses {
                "input"
            } else {
                "messages"
            };
            if format == WireFormat::OpenAiChat {
                assert_eq!(
                    edited[key].as_array().ok_or("messages")?[1..],
                    encoded[key].as_array().ok_or("messages")?[..]
                );
            } else {
                assert_eq!(edited[key], encoded[key]);
            }
            assert_eq!(edited["metadata"], original["metadata"]);
        }
        Ok(())
    }

    #[test]
    fn string_inputs_and_assistant_endings_keep_the_original_message()
    -> Result<(), Box<dyn std::error::Error>> {
        let engine = TranslationEngine::default();
        let policy = TranslationPolicy::default();
        for (format, body, key) in [
            (
                WireFormat::OpenAiChat,
                json!({"model":"m","messages":[{"role":"assistant","content":"done"}]}),
                "messages",
            ),
            (
                WireFormat::AnthropicMessages,
                json!({"model":"m","max_tokens":8,"messages":[{"role":"user","content":"question"},{"role":"assistant","content":"done"}]}),
                "messages",
            ),
            (
                WireFormat::OpenAiResponses,
                json!({"model":"m","input":"question"}),
                "input",
            ),
        ] {
            let mut request = engine.decode_request(format, &body, &policy)?.request;
            append_note(&mut request, "note one");
            append_note(&mut request, "note two");
            let encoded = engine.encode_request(format, &request, &policy)?.body;
            let items = encoded[key].as_array().ok_or("items")?;
            if body[key].is_array() {
                let original = body[key].as_array().ok_or("original")?;
                assert_eq!(&items[..original.len()], original);
            } else {
                assert_eq!(items[0], json!({"role":"user","content":"question"}));
            }
        }
        Ok(())
    }
    // Unsupported preserved bodies must not cause encoding to omit the note.
    #[test]
    fn invalid_preservation_falls_back_without_losing_the_note()
    -> Result<(), Box<dyn std::error::Error>> {
        let engine = TranslationEngine::default();
        let policy = TranslationPolicy::default();
        let source = json!({"model":"m","messages":[{"role":"user","content":"original task"}]});
        let mut request = engine
            .decode_request(WireFormat::OpenAiChat, &source, &policy)?
            .request;
        request
            .preservation
            .requests
            .insert("custom".into(), json!({"messages":"unsupported"}));
        request.preservation.requests.insert(
            WireFormat::OpenAiChat.into(),
            json!({"model":"m","messages":"unsupported"}),
        );
        append_note(&mut request, "reassess the failure");
        assert!(request.preservation.requests.is_empty());
        for target in [
            WireFormat::OpenAiChat,
            WireFormat::OpenAiResponses,
            WireFormat::AnthropicMessages,
        ] {
            let body = engine.encode_request(target, &request, &policy)?.body;
            let decoded = engine.decode_request(target, &body, &policy)?.request;
            let text: Vec<_> = decoded
                .messages
                .iter()
                .filter_map(|message| message.text_content(""))
                .collect();
            assert_eq!(
                text,
                vec!["original task", "reassess the failure"],
                "{target}"
            );
        }
        Ok(())
    }
}
