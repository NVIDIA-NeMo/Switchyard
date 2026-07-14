// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming codec for OpenAI Chat Completions chunks.

use serde_json::{json, Map, Value};

use crate::codecs::stream::{
    record_source_identity, source_model_or_unknown, state_source_is, string_field, StreamCodec,
    StreamTranslationState,
};
use crate::format::{FormatId, WireFormat};
use crate::llm::{LlmStreamEvent, Usage};

/// Stream codec for OpenAI Chat Completions chunks.
pub struct OpenAiChatStreamCodec;

impl StreamCodec for OpenAiChatStreamCodec {
    fn format(&self) -> FormatId {
        WireFormat::OpenAiChat.into()
    }

    fn decode_event(
        &self,
        state: &mut StreamTranslationState,
        event: &Value,
    ) -> Vec<LlmStreamEvent> {
        decode_openai_chat_stream(state, event)
    }

    fn encode_event(
        &self,
        state: &mut StreamTranslationState,
        event: LlmStreamEvent,
    ) -> Vec<Value> {
        encode_openai_chat_stream(state, event)
    }

    fn finish(&self, state: &mut StreamTranslationState) -> Vec<Value> {
        finish_openai_chat_stream(state)
    }
}

// Decodes one OpenAI Chat chunk into neutral streaming events.
fn decode_openai_chat_stream(
    state: &mut StreamTranslationState,
    event: &Value,
) -> Vec<LlmStreamEvent> {
    let Some(object) = event.as_object() else {
        return vec![LlmStreamEvent::Error {
            message: "OpenAI stream event is not an object".to_string(),
        }];
    };

    let mut out = Vec::new();
    if !state.saw_message_start {
        state.saw_message_start = true;
        if let Some(model) = string_field(object, "model") {
            state.model = Some(model);
        }
        if let Some(id) = string_field(object, "id") {
            state.message_id = Some(id);
        }
        out.push(LlmStreamEvent::MessageStart {
            id: state.message_id.clone(),
            model: state.model.clone(),
        });
    }

    if let Some(usage) = object.get("usage").and_then(Value::as_object) {
        let usage = openai_usage(usage);
        capture_openai_usage_extras(state, object.get("usage"));
        state.usage = usage.clone();
        state.saw_backend_usage = true;
        out.push(LlmStreamEvent::Usage(usage));
    }

    for choice in object
        .get("choices")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let Some(choice) = choice.as_object() else {
            continue;
        };
        if let Some(delta) = choice.get("delta").and_then(Value::as_object) {
            if let Some(text) = delta.get("content").and_then(Value::as_str) {
                if !text.is_empty() {
                    out.push(LlmStreamEvent::TextDelta {
                        index: 0,
                        text: text.to_string(),
                    });
                }
            }
            for reasoning_key in ["reasoning_content", "reasoning"] {
                if let Some(text) = delta.get(reasoning_key).and_then(Value::as_str) {
                    if !text.is_empty() {
                        out.push(LlmStreamEvent::ReasoningDelta {
                            index: 0,
                            text: text.to_string(),
                        });
                    }
                }
            }
            if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
                for tool_call in tool_calls {
                    if let Some(tool_call) = tool_call.as_object() {
                        let function = tool_call.get("function").and_then(Value::as_object);
                        out.push(LlmStreamEvent::ToolCallDelta {
                            index: tool_call.get("index").and_then(Value::as_u64).unwrap_or(0)
                                as usize,
                            id: tool_call
                                .get("id")
                                .and_then(Value::as_str)
                                .map(ToOwned::to_owned),
                            name: function
                                .and_then(|function| function.get("name"))
                                .and_then(Value::as_str)
                                .map(ToOwned::to_owned),
                            arguments_delta: function
                                .and_then(|function| function.get("arguments"))
                                .and_then(Value::as_str)
                                .map(ToOwned::to_owned),
                        });
                    }
                }
            }
        }
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            out.push(LlmStreamEvent::MessageStop {
                reason: Some(reason.to_string()),
            });
        }
    }
    out
}

// Encodes neutral streaming events into OpenAI Chat chunks.
fn encode_openai_chat_stream(
    state: &mut StreamTranslationState,
    event: LlmStreamEvent,
) -> Vec<Value> {
    match event {
        LlmStreamEvent::MessageStart { id, model } => {
            record_source_identity(state, id, model);
            if state.emitted_message_start
                || (!state_source_is(state, WireFormat::AnthropicMessages)
                    && !state_source_is(state, WireFormat::OpenAiResponses))
            {
                Vec::new()
            } else {
                state.emitted_message_start = true;
                vec![openai_stream_chunk(
                    state,
                    json!({"role": "assistant"}),
                    None,
                    None,
                )]
            }
        }
        LlmStreamEvent::TextDelta { text, .. } => {
            vec![openai_stream_chunk(
                state,
                json!({"content": text}),
                None,
                None,
            )]
        }
        LlmStreamEvent::ReasoningDelta { text, .. } => {
            vec![openai_stream_chunk(
                state,
                json!({"reasoning_content": text}),
                None,
                None,
            )]
        }
        LlmStreamEvent::ToolCallDelta {
            index,
            id,
            name,
            arguments_delta,
        } => vec![openai_tool_call_chunk(
            state,
            index,
            id,
            name,
            arguments_delta,
        )],
        LlmStreamEvent::Usage(usage) => {
            state.usage = usage;
            state.saw_backend_usage = true;
            Vec::new()
        }
        LlmStreamEvent::MessageStop { reason } => {
            if state.finished {
                return Vec::new();
            }
            state.finished = true;
            vec![openai_stream_chunk(
                state,
                json!({}),
                Some(openai_finish_reason(reason.as_deref())),
                Some(openai_usage_value(state)),
            )]
        }
        LlmStreamEvent::Error { message } => vec![json!({"error": {"message": message}})],
    }
}

// Emits a terminal chunk if the source stream ended before a stop event arrived.
fn finish_openai_chat_stream(state: &mut StreamTranslationState) -> Vec<Value> {
    if state.finished || !state.saw_message_start {
        return Vec::new();
    }
    state.finished = true;
    vec![openai_stream_chunk(
        state,
        json!({}),
        Some(openai_finish_reason(state.stop_reason.as_deref())),
        Some(openai_usage_value(state)),
    )]
}

// Normalizes OpenAI token usage fields.
fn openai_usage(usage: &Map<String, Value>) -> Usage {
    Usage {
        input_tokens: usage.get("prompt_tokens").and_then(Value::as_u64),
        output_tokens: usage.get("completion_tokens").and_then(Value::as_u64),
        total_tokens: usage.get("total_tokens").and_then(Value::as_u64),
        reasoning_tokens: usage
            .get("completion_tokens_details")
            .and_then(|details| details.get("reasoning_tokens"))
            .or_else(|| {
                usage
                    .get("output_tokens_details")
                    .and_then(|details| details.get("reasoning_tokens"))
            })
            .and_then(Value::as_u64),
    }
}

// Preserves OpenAI cache usage fields that have Anthropic equivalents.
fn capture_openai_usage_extras(state: &mut StreamTranslationState, usage: Option<&Value>) {
    if let Some(cached_tokens) = usage
        .and_then(|usage| usage.get("prompt_tokens_details"))
        .and_then(|details| details.get("cached_tokens"))
        .and_then(Value::as_u64)
    {
        state
            .usage_extras
            .insert("cache_read_input_tokens".to_string(), cached_tokens);
    }
}

// Builds a single OpenAI Chat stream chunk payload.
fn openai_stream_chunk(
    state: &StreamTranslationState,
    delta: Value,
    finish_reason: Option<String>,
    usage: Option<Value>,
) -> Value {
    let mut payload = json!({
        "id": openai_stream_id(state),
        "object": "chat.completion.chunk",
        "created": 0,
        "model": source_model_or_unknown(state),
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    });
    if let Some(usage) = usage {
        payload["usage"] = usage;
    }
    payload
}

// Builds an OpenAI Chat tool-call delta chunk.
fn openai_tool_call_chunk(
    state: &StreamTranslationState,
    index: usize,
    id: Option<String>,
    name: Option<String>,
    arguments: Option<String>,
) -> Value {
    let mut function = Map::new();
    if let Some(name) = name {
        function.insert("name".to_string(), Value::String(name));
    }
    if let Some(arguments) = arguments {
        function.insert("arguments".to_string(), Value::String(arguments));
    }
    let mut tool_call = Map::new();
    tool_call.insert("index".to_string(), json!(index));
    tool_call.insert("type".to_string(), json!("function"));
    tool_call.insert("function".to_string(), Value::Object(function));
    if let Some(id) = id {
        tool_call.insert("id".to_string(), Value::String(id));
    }
    openai_stream_chunk(
        state,
        json!({"tool_calls": [Value::Object(tool_call)]}),
        None,
        None,
    )
}

// Builds OpenAI usage payloads from normalized and provider-extra state.
fn openai_usage_value(state: &StreamTranslationState) -> Value {
    let cache_creation_tokens = state
        .usage_extras
        .get("cache_creation_input_tokens")
        .copied()
        .unwrap_or(0);
    let cache_read_tokens = state
        .usage_extras
        .get("cache_read_input_tokens")
        .copied()
        .unwrap_or(0);
    let prompt_tokens =
        state.usage.input_tokens.unwrap_or(0) + cache_creation_tokens + cache_read_tokens;
    let completion_tokens = state.usage.output_tokens.unwrap_or(0);
    let mut usage = json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": state.usage.total_tokens.unwrap_or(prompt_tokens + completion_tokens),
    });
    if let Some(reasoning_tokens) = state.usage.reasoning_tokens {
        usage["completion_tokens_details"] = json!({
            "reasoning_tokens": reasoning_tokens,
        });
    }
    if cache_creation_tokens > 0 || cache_read_tokens > 0 {
        usage["prompt_tokens_details"] = json!({
            "cached_tokens": cache_read_tokens,
            "cache_creation_tokens": cache_creation_tokens,
        });
    }
    usage
}

// Converts any upstream message ID into an OpenAI-looking stream ID.
fn openai_stream_id(state: &StreamTranslationState) -> String {
    let Some(id) = state.message_id.as_deref() else {
        return "chatcmpl_switchyard".to_string();
    };
    if id.starts_with("chatcmpl") {
        id.to_string()
    } else if let Some(rest) = id.strip_prefix("msg_").or_else(|| id.strip_prefix("resp_")) {
        format!("chatcmpl_{rest}")
    } else {
        format!("chatcmpl_{id}")
    }
}

// Maps provider stop reasons into OpenAI's finish-reason vocabulary.
fn openai_finish_reason(reason: Option<&str>) -> String {
    match reason {
        Some("end_turn") | Some("stop_sequence") | None => "stop".to_string(),
        Some("max_tokens") => "length".to_string(),
        Some("tool_use") => "tool_calls".to_string(),
        Some(other) => other.to_string(),
    }
}
