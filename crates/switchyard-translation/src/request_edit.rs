// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request edits update normalized content and preserved provider bodies together.

use serde_json::{Map, Value, json};
use switchyard_protocol::{ContentBlock, InstructionBlock, LlmRequest, Message, Role, WireFormat};

/// This function appends temporary guidance and updates preserved provider bodies.
/// It removes bodies it cannot edit so encoding rebuilds them from the updated request.
pub fn append_note(request: &mut LlmRequest, note: &str) {
    // A separate message preserves the caller's final message boundary for OpenAI caching.
    request.messages.push(Message::text(Role::User, note));
    request.temporary_note_count += 1;
    request.preservation.requests.retain(|format, body| {
        let Some(body) = body.as_object_mut() else {
            return false;
        };
        let (key, content_type) = match format.as_str() {
            "openai_chat" | "anthropic_messages" => ("messages", "text"),
            "openai_responses" => ("input", "input_text"),
            _ => return false,
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
        if format.as_str() == WireFormat::AnthropicMessages.as_str()
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

/// This function prepends a system prompt and updates supported preserved provider bodies.
/// It removes bodies it cannot edit so encoding rebuilds them from the updated request.
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
        match format.as_str() {
            "openai_chat" => prepend_openai_chat_system(body, prompt),
            "openai_responses" => prepend_text_field(body, "instructions", prompt),
            "anthropic_messages" => {
                body.remove("switchyard_anthropic_request");
                prepend_anthropic_system(body, prompt)
            }
            _ => false,
        }
    });
}

// This helper inserts a system message while keeping the existing Chat messages intact.
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

// This helper retains existing Anthropic system blocks when it prepends a prompt.
fn prepend_anthropic_system(body: &mut Map<String, Value>, prompt: &str) -> bool {
    if let Some(Value::Array(blocks)) = body.get_mut("system") {
        blocks.insert(0, json!({"type":"text","text":prompt}));
        true
    } else {
        prepend_text_field(body, "system", prompt)
    }
}
