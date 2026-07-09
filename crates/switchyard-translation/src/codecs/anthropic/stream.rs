// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming codec for Anthropic Messages events.

use serde_json::{json, Map, Value};

use crate::codecs::stream::{
    record_source_identity, target_message_id_or_source_message_id, target_model_or_source_model,
    ConversationStreamEvent, StreamCodec, StreamTranslationState,
};
use crate::format::{FormatId, WireFormat};
use crate::util::sanitize_anthropic_tool_use_id;

/// Stream codec for Anthropic Messages events.
pub struct AnthropicMessagesStreamCodec;

impl StreamCodec for AnthropicMessagesStreamCodec {
    fn format(&self) -> FormatId {
        WireFormat::AnthropicMessages.into()
    }

    fn decode_event(
        &self,
        state: &mut StreamTranslationState,
        event: &Value,
    ) -> Vec<ConversationStreamEvent> {
        decode_anthropic_stream(state, event)
    }

    fn encode_event(
        &self,
        state: &mut StreamTranslationState,
        event: ConversationStreamEvent,
    ) -> Vec<Value> {
        encode_anthropic_stream(state, event)
    }

    fn finish(&self, state: &mut StreamTranslationState) -> Vec<Value> {
        finish_anthropic_stream(state)
    }
}

// Decodes one Anthropic Messages event into neutral streaming events.
fn decode_anthropic_stream(
    state: &mut StreamTranslationState,
    event: &Value,
) -> Vec<ConversationStreamEvent> {
    let Some(object) = event.as_object() else {
        return vec![ConversationStreamEvent::Error {
            message: "Anthropic stream event is not an object".to_string(),
        }];
    };
    match object.get("type").and_then(Value::as_str) {
        Some("message_start") => {
            state.saw_message_start = true;
            let message = object.get("message").and_then(Value::as_object);
            if let Some(model) = message
                .and_then(|message| message.get("model"))
                .and_then(Value::as_str)
            {
                state.model = Some(model.to_string());
            }
            if let Some(id) = message
                .and_then(|message| message.get("id"))
                .and_then(Value::as_str)
            {
                state.message_id = Some(id.to_string());
            }
            if let Some(message) = message {
                if let Some(usage) = message.get("usage") {
                    capture_anthropic_usage(state, usage);
                }
            }
            vec![ConversationStreamEvent::MessageStart {
                id: state.message_id.clone(),
                model: state.model.clone(),
            }]
        }
        Some("content_block_start") => decode_anthropic_content_block_start(object),
        Some("content_block_delta") => decode_anthropic_content_block_delta(object),
        Some("message_delta") => {
            let mut out = Vec::new();
            if let Some(usage) = object.get("usage") {
                capture_anthropic_usage(state, usage);
                out.push(ConversationStreamEvent::Usage(state.usage.clone()));
            }
            if let Some(stop_reason) = object
                .get("delta")
                .and_then(Value::as_object)
                .and_then(|delta| delta.get("stop_reason"))
                .and_then(Value::as_str)
            {
                out.push(ConversationStreamEvent::MessageStop {
                    reason: Some(stop_reason.to_string()),
                });
            }
            out
        }
        Some("message_stop") => vec![ConversationStreamEvent::MessageStop { reason: None }],
        Some("error") => vec![ConversationStreamEvent::Error {
            message: object
                .get("error")
                .and_then(Value::as_object)
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("unknown Anthropic stream error")
                .to_string(),
        }],
        _ => Vec::new(),
    }
}

// Encodes neutral streaming events into Anthropic Messages events.
fn encode_anthropic_stream(
    state: &mut StreamTranslationState,
    event: ConversationStreamEvent,
) -> Vec<Value> {
    match event {
        ConversationStreamEvent::MessageStart { id, model } => {
            record_source_identity(state, id, model);
            if state.emitted_message_start {
                Vec::new()
            } else {
                state.emitted_message_start = true;
                vec![json!({
                    "type": "message_start",
                    "message": {
                        "id": anthropic_message_id(state),
                        "type": "message",
                        "role": "assistant",
                        "model": target_model_or_source_model(state),
                        "content": [],
                        "stop_reason": Value::Null,
                        "stop_sequence": Value::Null,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                })]
            }
        }
        ConversationStreamEvent::TextDelta { text, .. } => {
            state.output_tokens_seen += 1;
            let mut out = ensure_anthropic_text_block(state);
            out.push(json!({
                "type": "content_block_delta",
                "index": state.text_block_index.unwrap_or(0),
                "delta": {"type": "text_delta", "text": text},
            }));
            out
        }
        ConversationStreamEvent::ReasoningDelta { text, .. } => {
            let mut out = ensure_anthropic_reasoning_block(state);
            out.push(json!({
                "type": "content_block_delta",
                "index": state.reasoning_block_index.unwrap_or(0),
                "delta": {"type": "thinking_delta", "thinking": text},
            }));
            out
        }
        ConversationStreamEvent::ToolCallDelta {
            index,
            id,
            name,
            arguments_delta,
        } => encode_anthropic_tool_delta(state, index, id, name, arguments_delta),
        ConversationStreamEvent::Usage(usage) => {
            state.usage = usage;
            state.saw_backend_usage = true;
            Vec::new()
        }
        ConversationStreamEvent::MessageStop { reason } => {
            state.stop_reason = reason.or_else(|| state.stop_reason.clone());
            Vec::new()
        }
        ConversationStreamEvent::Error { message } => {
            state.finished = true;
            vec![json!({"type": "error", "error": {"message": message}})]
        }
    }
}

// Emits any missing Anthropic terminal events and closes open content blocks.
fn finish_anthropic_stream(state: &mut StreamTranslationState) -> Vec<Value> {
    if state.finished {
        return Vec::new();
    }
    let mut out = Vec::new();
    if !state.emitted_message_start {
        out.extend(encode_anthropic_stream(
            state,
            ConversationStreamEvent::MessageStart {
                id: state.message_id.clone(),
                model: state.model.clone(),
            },
        ));
    }

    if state.text_block_started {
        if let Some(index) = state.text_block_index {
            out.push(json!({"type": "content_block_stop", "index": index}));
        }
        state.text_block_started = false;
    }

    out.extend(close_anthropic_reasoning_block(state));

    for tool in state.tool_states.values_mut() {
        if tool.started {
            if let Some(index) = tool.content_index {
                out.push(json!({"type": "content_block_stop", "index": index}));
            }
            tool.started = false;
        }
    }

    if !state.emitted_content_block {
        out.push(json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }));
        out.push(json!({"type": "content_block_stop", "index": 0}));
    }

    out.push(json!({
        "type": "message_delta",
        "delta": {
            "stop_reason": anthropic_stop_reason(state.stop_reason.as_deref()),
            "stop_sequence": Value::Null,
        },
        "usage": anthropic_stream_usage(state),
    }));
    out.push(json!({"type": "message_stop"}));
    state.finished = true;
    out
}

// Converts Anthropic content-block starts into text or tool-call deltas.
fn decode_anthropic_content_block_start(
    object: &Map<String, Value>,
) -> Vec<ConversationStreamEvent> {
    let index = object.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
    let block = object.get("content_block").and_then(Value::as_object);
    match block
        .and_then(|block| block.get("type"))
        .and_then(Value::as_str)
    {
        Some("text") => block
            .and_then(|block| block.get("text"))
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty())
            .map(|text| {
                vec![ConversationStreamEvent::TextDelta {
                    index,
                    text: text.to_string(),
                }]
            })
            .unwrap_or_default(),
        Some("thinking") => block
            .and_then(|block| block.get("thinking"))
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty())
            .map(|text| {
                vec![ConversationStreamEvent::ReasoningDelta {
                    index,
                    text: text.to_string(),
                }]
            })
            .unwrap_or_default(),
        Some("tool_use") => {
            let Some(block) = block else {
                return Vec::new();
            };
            vec![ConversationStreamEvent::ToolCallDelta {
                index,
                id: block
                    .get("id")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                name: block
                    .get("name")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                arguments_delta: block.get("input").and_then(tool_input_delta),
            }]
        }
        _ => Vec::new(),
    }
}

// Converts Anthropic content-block deltas into neutral text or argument deltas.
fn decode_anthropic_content_block_delta(
    object: &Map<String, Value>,
) -> Vec<ConversationStreamEvent> {
    let index = object.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
    let Some(delta) = object.get("delta").and_then(Value::as_object) else {
        return Vec::new();
    };
    match delta.get("type").and_then(Value::as_str) {
        Some("text_delta") => delta
            .get("text")
            .and_then(Value::as_str)
            .map(|text| {
                vec![ConversationStreamEvent::TextDelta {
                    index,
                    text: text.to_string(),
                }]
            })
            .unwrap_or_default(),
        Some("thinking_delta") => delta
            .get("thinking")
            .and_then(Value::as_str)
            .map(|text| {
                vec![ConversationStreamEvent::ReasoningDelta {
                    index,
                    text: text.to_string(),
                }]
            })
            .unwrap_or_default(),
        Some("signature_delta") => Vec::new(),
        Some("input_json_delta") => delta
            .get("partial_json")
            .and_then(Value::as_str)
            .map(|partial_json| {
                vec![ConversationStreamEvent::ToolCallDelta {
                    index,
                    id: None,
                    name: None,
                    arguments_delta: Some(partial_json.to_string()),
                }]
            })
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

// Starts an Anthropic text block when the target stream needs one.
fn ensure_anthropic_text_block(state: &mut StreamTranslationState) -> Vec<Value> {
    let mut out = close_anthropic_reasoning_block(state);
    if state.text_block_started {
        return out;
    }
    let index = state.next_content_index;
    state.next_content_index += 1;
    state.text_block_index = Some(index);
    state.text_block_started = true;
    state.emitted_content_block = true;
    out.push(json!({
        "type": "content_block_start",
        "index": index,
        "content_block": {"type": "text", "text": ""},
    }));
    out
}

// Starts an Anthropic thinking block when the target stream needs one.
fn ensure_anthropic_reasoning_block(state: &mut StreamTranslationState) -> Vec<Value> {
    let mut out = Vec::new();
    if state.text_block_started {
        if let Some(index) = state.text_block_index {
            out.push(json!({"type": "content_block_stop", "index": index}));
        }
        state.text_block_started = false;
    }
    if state.reasoning_block_started {
        return out;
    }
    let index = state.next_content_index;
    state.next_content_index += 1;
    state.reasoning_block_index = Some(index);
    state.reasoning_block_started = true;
    state.emitted_content_block = true;
    out.push(json!({
        "type": "content_block_start",
        "index": index,
        "content_block": {"type": "thinking", "thinking": "", "signature": ""},
    }));
    out
}

// Closes an open Anthropic thinking block before text/tool content starts.
fn close_anthropic_reasoning_block(state: &mut StreamTranslationState) -> Vec<Value> {
    if !state.reasoning_block_started {
        return Vec::new();
    }
    let mut out = Vec::new();
    if let Some(index) = state.reasoning_block_index {
        out.push(json!({
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "signature_delta", "signature": ""},
        }));
        out.push(json!({"type": "content_block_stop", "index": index}));
    }
    state.reasoning_block_started = false;
    out
}

// Accumulates tool-call state and emits Anthropic tool-use deltas.
fn encode_anthropic_tool_delta(
    state: &mut StreamTranslationState,
    index: usize,
    id: Option<String>,
    name: Option<String>,
    arguments_delta: Option<String>,
) -> Vec<Value> {
    let mut out = Vec::new();
    out.extend(close_anthropic_reasoning_block(state));
    if state.text_block_started {
        if let Some(index) = state.text_block_index {
            out.push(json!({"type": "content_block_stop", "index": index}));
        }
        state.text_block_started = false;
    }

    let tool = state.tool_states.entry(index).or_default();
    if id.is_some() {
        tool.id = id.map(|id| sanitize_anthropic_tool_use_id(&id));
    }
    if name.is_some() {
        tool.name = name;
    }
    if let Some(delta) = arguments_delta {
        tool.arguments.push_str(&delta);
        tool.pending_arguments.push_str(&delta);
    }

    if !tool.started {
        let Some(name) = tool.name.clone() else {
            return out;
        };
        let content_index = state.next_content_index;
        state.next_content_index += 1;
        tool.content_index = Some(content_index);
        tool.started = true;
        state.emitted_content_block = true;
        out.push(json!({
            "type": "content_block_start",
            "index": content_index,
            "content_block": {
                "type": "tool_use",
                "id": tool.id.clone().unwrap_or_else(|| format!("toolu_{index}")),
                "name": name,
                "input": {},
            },
        }));
        if !tool.pending_arguments.is_empty() {
            out.push(json!({
                "type": "content_block_delta",
                "index": content_index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": tool.pending_arguments,
                },
            }));
            tool.pending_arguments.clear();
        }
        return out;
    }

    if let Some(content_index) = tool.content_index {
        if !tool.pending_arguments.is_empty() {
            out.push(json!({
                "type": "content_block_delta",
                "index": content_index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": tool.pending_arguments,
                },
            }));
            tool.pending_arguments.clear();
        }
    }
    out
}

// Preserves Anthropic usage fields and updates normalized token counts.
fn capture_anthropic_usage(state: &mut StreamTranslationState, usage: &Value) {
    let Some(usage) = usage.as_object() else {
        return;
    };
    for key in [
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    ] {
        if let Some(value) = usage.get(key).and_then(Value::as_u64) {
            state.usage_extras.insert(key.to_string(), value);
        }
    }
    state.usage.input_tokens = state.usage_extras.get("input_tokens").copied();
    state.usage.output_tokens = state.usage_extras.get("output_tokens").copied();
}

// Builds Anthropic usage payloads from normalized and provider-extra state.
fn anthropic_stream_usage(state: &StreamTranslationState) -> Value {
    let mut usage = Map::new();
    if state.saw_backend_usage {
        if let Some(input_tokens) = state.usage.input_tokens {
            usage.insert("input_tokens".to_string(), json!(input_tokens));
        }
        usage.insert(
            "output_tokens".to_string(),
            json!(state.usage.output_tokens.unwrap_or(0)),
        );
    } else {
        usage.insert("output_tokens".to_string(), json!(state.output_tokens_seen));
    }
    for (key, value) in &state.usage_extras {
        if key == "input_tokens" || key == "output_tokens" {
            continue;
        }
        usage.insert(key.clone(), json!(value));
    }
    Value::Object(usage)
}

// Converts any upstream message ID into an Anthropic-looking message ID.
fn anthropic_message_id(state: &StreamTranslationState) -> String {
    let Some(id) = target_message_id_or_source_message_id(state) else {
        return "msg_switchyard".to_string();
    };
    if id.starts_with("msg_") {
        id.to_string()
    } else {
        format!("msg_{id}")
    }
}

// Maps provider stop reasons into Anthropic's stop-reason vocabulary.
fn anthropic_stop_reason(reason: Option<&str>) -> String {
    match reason {
        Some("length") => "max_tokens".to_string(),
        Some("tool_calls") | Some("function_call") => "tool_use".to_string(),
        Some("end_turn") | Some("max_tokens") | Some("tool_use") | Some("stop_sequence") => {
            reason.unwrap_or("end_turn").to_string()
        }
        _ => "end_turn".to_string(),
    }
}

// Converts a streamed tool input fragment into a string delta.
fn tool_input_delta(value: &Value) -> Option<String> {
    match value {
        Value::String(text) if !text.is_empty() => Some(text.clone()),
        Value::Object(object) if !object.is_empty() => serde_json::to_string(value).ok(),
        _ => None,
    }
}
