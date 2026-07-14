// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming codec for OpenAI Responses API events.

use serde_json::{json, Value};

use crate::codecs::stream::{
    record_source_identity, target_message_id_or_source_message_id, target_model_or_source_model,
    ConversationStreamEvent, StreamCodec, StreamTranslationState,
};
use crate::conversation::Usage;
use crate::format::{FormatId, WireFormat};

/// Stream codec for OpenAI Responses API events.
pub struct OpenAiResponsesStreamCodec;

impl StreamCodec for OpenAiResponsesStreamCodec {
    fn format(&self) -> FormatId {
        WireFormat::OpenAiResponses.into()
    }

    fn decode_event(
        &self,
        state: &mut StreamTranslationState,
        event: &Value,
    ) -> Vec<ConversationStreamEvent> {
        decode_responses_stream(state, event)
    }

    fn encode_event(
        &self,
        state: &mut StreamTranslationState,
        event: ConversationStreamEvent,
    ) -> Vec<Value> {
        encode_responses_stream(state, event)
    }

    fn finish(&self, state: &mut StreamTranslationState) -> Vec<Value> {
        finish_responses_stream(state)
    }
}

// Decodes one OpenAI Responses event into neutral streaming events.
fn decode_responses_stream(
    state: &mut StreamTranslationState,
    event: &Value,
) -> Vec<ConversationStreamEvent> {
    let event_type = event
        .get("type")
        .or_else(|| event.get("event"))
        .and_then(Value::as_str);
    match event_type {
        Some("response.created") => {
            state.saw_message_start = true;
            let response = event.get("response").and_then(Value::as_object);
            if let Some(model) = response
                .and_then(|response| response.get("model"))
                .and_then(Value::as_str)
            {
                state.model = Some(model.to_string());
            }
            if let Some(id) = response
                .and_then(|response| response.get("id"))
                .and_then(Value::as_str)
            {
                state.message_id = Some(id.to_string());
            }
            vec![ConversationStreamEvent::MessageStart {
                id: state.message_id.clone(),
                model: state.model.clone(),
            }]
        }
        Some("response.output_text.delta") => event
            .get("delta")
            .and_then(Value::as_str)
            .map(|text| {
                vec![ConversationStreamEvent::TextDelta {
                    index: event
                        .get("output_index")
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as usize,
                    text: text.to_string(),
                }]
            })
            .unwrap_or_default(),
        Some("response.reasoning_text.delta") | Some("response.reasoning_summary_text.delta") => {
            event
                .get("delta")
                .or_else(|| event.get("text"))
                .and_then(Value::as_str)
                .map(|text| {
                    vec![ConversationStreamEvent::ReasoningDelta {
                        index: event
                            .get("output_index")
                            .and_then(Value::as_u64)
                            .unwrap_or(0) as usize,
                        text: text.to_string(),
                    }]
                })
                .unwrap_or_default()
        }
        Some("response.output_item.added") => decode_responses_output_item_added(event),
        Some("response.function_call_arguments.delta") => {
            let output_index = event
                .get("output_index")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            event
                .get("delta")
                .and_then(Value::as_str)
                .map(|delta| {
                    vec![ConversationStreamEvent::ToolCallDelta {
                        index: output_index as usize,
                        id: None,
                        name: None,
                        arguments_delta: Some(delta.to_string()),
                    }]
                })
                .unwrap_or_default()
        }
        Some("response.output_item.done") => decode_responses_output_item_done(event, state),
        Some("response.completed") => {
            let mut out = Vec::new();
            if let Some(usage) = event
                .get("response")
                .and_then(Value::as_object)
                .and_then(|response| response.get("usage"))
                .and_then(Value::as_object)
            {
                let usage = responses_usage(usage);
                state.usage = usage.clone();
                state.saw_backend_usage = true;
                out.push(ConversationStreamEvent::Usage(usage));
            }
            out.push(ConversationStreamEvent::MessageStop { reason: None });
            out
        }
        Some("error") => vec![ConversationStreamEvent::Error {
            message: event
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("unknown Responses stream error")
                .to_string(),
        }],
        _ => Vec::new(),
    }
}

// Encodes neutral streaming events into OpenAI Responses events.
fn encode_responses_stream(
    state: &mut StreamTranslationState,
    event: ConversationStreamEvent,
) -> Vec<Value> {
    match event {
        ConversationStreamEvent::MessageStart { id, model } => {
            record_source_identity(state, id, model);
            ensure_responses_created(state)
        }
        ConversationStreamEvent::TextDelta { text, .. } => encode_responses_text_delta(state, text),
        ConversationStreamEvent::ReasoningDelta { text, .. } => {
            encode_responses_reasoning_delta(state, text)
        }
        ConversationStreamEvent::ToolCallDelta {
            index,
            id,
            name,
            arguments_delta,
        } => encode_responses_tool_delta(state, index, id, name, arguments_delta),
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
            vec![json!({"type": "error", "message": message})]
        }
    }
}

// Emits final OpenAI Responses completion events from accumulated state.
fn finish_responses_stream(state: &mut StreamTranslationState) -> Vec<Value> {
    let mut out = ensure_responses_created(state);
    if state.response_text_started {
        if let Some(output_index) = state.response_text_output_index {
            out.push(json!({
                "type": "response.content_part.done",
                "output_index": output_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": state.response_text},
            }));
            out.push(json!({
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": state.response_text}],
                },
            }));
        }
    }

    let mut final_items: Vec<(usize, Value)> = Vec::new();
    if state.response_reasoning_started {
        if let Some(output_index) = state.response_reasoning_output_index {
            out.push(json!({
                "type": "response.reasoning_text.done",
                "output_index": output_index,
                "content_index": 0,
                "text": state.response_reasoning_text,
            }));
            let item = json!({
                "type": "reasoning",
                "id": format!("rs_{output_index}"),
                "status": "completed",
                "content": [{
                    "type": "reasoning_text",
                    "text": state.response_reasoning_text,
                }],
                "summary": [],
            });
            out.push(json!({
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": item,
            }));
            final_items.push((output_index, item));
        }
    }
    if state.response_text_started {
        if let Some(output_index) = state.response_text_output_index {
            final_items.push((
                output_index,
                json!({
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": state.response_text}],
                }),
            ));
        }
    }

    for tool in state.tool_states.values() {
        if !tool.started {
            continue;
        }
        let output_index = tool.response_output_index.unwrap_or(0);
        out.push(json!({
            "type": "response.function_call_arguments.done",
            "output_index": output_index,
            "arguments": tool.arguments,
        }));
        let item = json!({
            "type": "function_call",
            "id": tool.response_item_id.clone().unwrap_or_else(|| format!("fc_{output_index}")),
            "call_id": tool.id.clone().unwrap_or_else(|| format!("call_{output_index}")),
            "name": tool.name.clone().unwrap_or_default(),
            "arguments": tool.arguments,
            "status": "completed",
        });
        out.push(json!({
            "type": "response.output_item.done",
            "output_index": output_index,
            "item": item,
        }));
        final_items.push((output_index, item));
    }

    final_items.sort_by_key(|(index, _)| *index);
    let output = final_items
        .into_iter()
        .map(|(_, item)| item)
        .collect::<Vec<_>>();

    out.push(json!({
        "type": "response.completed",
        "response": {
            "id": responses_id(state),
            "object": "response",
            "status": "completed",
            "model": target_model_or_source_model(state),
            "output": output,
            "usage": responses_usage_value(&state.usage),
        },
    }));
    state.finished = true;
    out
}

// Converts Responses function-call item creation into a neutral tool-call delta.
fn decode_responses_output_item_added(event: &Value) -> Vec<ConversationStreamEvent> {
    let Some(item) = event.get("item").and_then(Value::as_object) else {
        return Vec::new();
    };
    if item.get("type").and_then(Value::as_str) != Some("function_call") {
        return Vec::new();
    }
    let index = event
        .get("output_index")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    vec![ConversationStreamEvent::ToolCallDelta {
        index,
        id: item
            .get("call_id")
            .or_else(|| item.get("id"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        name: item
            .get("name")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        arguments_delta: item
            .get("arguments")
            .and_then(Value::as_str)
            .filter(|arguments| !arguments.is_empty())
            .map(ToOwned::to_owned),
    }]
}

// Emits a final tool-call argument delta when Responses only supplies arguments at item end.
fn decode_responses_output_item_done(
    event: &Value,
    state: &StreamTranslationState,
) -> Vec<ConversationStreamEvent> {
    let Some(item) = event.get("item").and_then(Value::as_object) else {
        return Vec::new();
    };
    if item.get("type").and_then(Value::as_str) != Some("function_call") {
        return Vec::new();
    }
    let index = event
        .get("output_index")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    let arguments = item.get("arguments").and_then(Value::as_str);
    if let Some(arguments) = arguments {
        let existing = state
            .tool_states
            .get(&index)
            .map(|tool| tool.arguments.as_str())
            .unwrap_or("");
        if !arguments.is_empty() && arguments != existing {
            return vec![ConversationStreamEvent::ToolCallDelta {
                index,
                id: None,
                name: None,
                arguments_delta: Some(arguments.to_string()),
            }];
        }
    }
    Vec::new()
}

// Emits the initial Responses created event once per stream.
fn ensure_responses_created(state: &mut StreamTranslationState) -> Vec<Value> {
    if state.response_created {
        return Vec::new();
    }
    state.response_created = true;
    vec![json!({
        "type": "response.created",
        "response": {
            "id": responses_id(state),
            "object": "response",
            "status": "in_progress",
            "model": target_model_or_source_model(state),
            "output": [],
            "usage": responses_usage_value(&state.usage),
        },
    })]
}

// Accumulates assistant text and emits Responses text delta events.
fn encode_responses_text_delta(state: &mut StreamTranslationState, text: String) -> Vec<Value> {
    let mut out = ensure_responses_created(state);
    if !state.response_text_started {
        state.response_text_started = true;
        let output_index = state.next_response_output_index;
        state.next_response_output_index += 1;
        state.response_text_output_index = Some(output_index);
        out.push(json!({
            "type": "response.output_item.added",
            "output_index": output_index,
            "item": {
                "type": "message",
                "id": format!("msg_{output_index}"),
                "role": "assistant",
                "status": "in_progress",
                "content": [],
            },
        }));
        out.push(json!({
            "type": "response.content_part.added",
            "output_index": output_index,
            "content_index": 0,
            "part": {"type": "output_text", "text": ""},
        }));
    }
    state.response_text.push_str(&text);
    out.push(json!({
        "type": "response.output_text.delta",
        "output_index": state.response_text_output_index.unwrap_or(0),
        "content_index": 0,
        "delta": text,
    }));
    out
}

// Accumulates reasoning text and emits Responses reasoning events.
fn encode_responses_reasoning_delta(
    state: &mut StreamTranslationState,
    text: String,
) -> Vec<Value> {
    let mut out = ensure_responses_created(state);
    if !state.response_reasoning_started {
        state.response_reasoning_started = true;
        let output_index = state.next_response_output_index;
        state.next_response_output_index += 1;
        state.response_reasoning_output_index = Some(output_index);
        out.push(json!({
            "type": "response.output_item.added",
            "output_index": output_index,
            "item": {
                "type": "reasoning",
                "id": format!("rs_{output_index}"),
                "status": "in_progress",
                "content": [],
                "summary": [],
            },
        }));
        out.push(json!({
            "type": "response.reasoning_text.added",
            "output_index": output_index,
            "content_index": 0,
            "text": "",
        }));
    }
    state.response_reasoning_text.push_str(&text);
    out.push(json!({
        "type": "response.reasoning_text.delta",
        "output_index": state.response_reasoning_output_index.unwrap_or(0),
        "content_index": 0,
        "delta": text,
    }));
    out
}

// Accumulates tool-call state and emits Responses function-call delta events.
fn encode_responses_tool_delta(
    state: &mut StreamTranslationState,
    index: usize,
    id: Option<String>,
    name: Option<String>,
    arguments_delta: Option<String>,
) -> Vec<Value> {
    let mut out = ensure_responses_created(state);
    let tool = state.tool_states.entry(index).or_default();
    if id.is_some() {
        tool.id = id;
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
        let output_index = state.next_response_output_index;
        state.next_response_output_index += 1;
        tool.response_output_index = Some(output_index);
        tool.response_item_id = Some(format!("fc_{output_index}"));
        tool.started = true;
        out.push(json!({
            "type": "response.output_item.added",
            "output_index": output_index,
            "item": {
                "type": "function_call",
                "id": tool.response_item_id.clone().unwrap_or_else(|| format!("fc_{output_index}")),
                "call_id": tool.id.clone().unwrap_or_else(|| format!("call_{index}")),
                "name": name,
                "arguments": "",
                "status": "in_progress",
            },
        }));
        if !tool.pending_arguments.is_empty() {
            out.push(json!({
                "type": "response.function_call_arguments.delta",
                "output_index": output_index,
                "delta": tool.pending_arguments,
            }));
            tool.pending_arguments.clear();
        }
        return out;
    }

    if let Some(output_index) = tool.response_output_index {
        if !tool.pending_arguments.is_empty() {
            out.push(json!({
                "type": "response.function_call_arguments.delta",
                "output_index": output_index,
                "delta": tool.pending_arguments,
            }));
            tool.pending_arguments.clear();
        }
    }
    out
}

// Normalizes OpenAI Responses token usage fields.
fn responses_usage(usage: &serde_json::Map<String, Value>) -> Usage {
    let input_tokens = usage.get("input_tokens").and_then(Value::as_u64);
    let output_tokens = usage.get("output_tokens").and_then(Value::as_u64);
    Usage {
        input_tokens,
        output_tokens,
        total_tokens: usage
            .get("total_tokens")
            .and_then(Value::as_u64)
            .or_else(|| Some(input_tokens.unwrap_or(0) + output_tokens.unwrap_or(0))),
        reasoning_tokens: usage
            .get("output_tokens_details")
            .and_then(|details| details.get("reasoning_tokens"))
            .or_else(|| {
                usage
                    .get("completion_tokens_details")
                    .and_then(|details| details.get("reasoning_tokens"))
            })
            .and_then(Value::as_u64),
    }
}

// Builds OpenAI Responses usage payloads from normalized usage.
fn responses_usage_value(usage: &Usage) -> Value {
    let mut value = json!({
        "input_tokens": usage.input_tokens.unwrap_or(0),
        "output_tokens": usage.output_tokens.unwrap_or(0),
        "total_tokens": usage.total_tokens.unwrap_or_else(|| {
            usage.input_tokens.unwrap_or(0) + usage.output_tokens.unwrap_or(0)
        }),
    });
    if let Some(reasoning_tokens) = usage.reasoning_tokens {
        value["output_tokens_details"] = json!({
            "reasoning_tokens": reasoning_tokens,
        });
    }
    value
}

// Converts any upstream message ID into a Responses-looking response ID.
fn responses_id(state: &StreamTranslationState) -> String {
    let Some(id) = target_message_id_or_source_message_id(state) else {
        return "resp_switchyard".to_string();
    };
    if id.starts_with("resp_") {
        id.to_string()
    } else {
        format!("resp_{id}")
    }
}
