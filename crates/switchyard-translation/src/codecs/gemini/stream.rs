// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming codec for Gemini streamGenerateContent chunks.
//!
//! Each Gemini SSE `data:` line carries a complete `GenerateContentResponse`
//! chunk: text arrives as incremental parts, function calls arrive whole in a
//! single chunk, and the terminal chunk carries `finishReason` plus the
//! authoritative `usageMetadata`. There are no named events and no `[DONE]`
//! marker.

use serde_json::{json, Map, Value};

use crate::codecs::stream::{
    record_source_identity, target_message_id_or_source_message_id, target_model_or_source_model,
    ConversationStreamEvent, StreamCodec, StreamTranslationState,
};
use crate::format::{FormatId, WireFormat};

/// Stream codec for Gemini streamGenerateContent chunks.
pub struct GeminiGenerateContentStreamCodec;

impl StreamCodec for GeminiGenerateContentStreamCodec {
    fn format(&self) -> FormatId {
        WireFormat::GeminiGenerateContent.into()
    }

    fn decode_event(
        &self,
        state: &mut StreamTranslationState,
        event: &Value,
    ) -> Vec<ConversationStreamEvent> {
        decode_gemini_stream(state, event)
    }

    fn encode_event(
        &self,
        state: &mut StreamTranslationState,
        event: ConversationStreamEvent,
    ) -> Vec<Value> {
        encode_gemini_stream(state, event)
    }

    fn finish(&self, state: &mut StreamTranslationState) -> Vec<Value> {
        finish_gemini_stream(state)
    }
}

// Decodes one Gemini chunk into neutral streaming events.
fn decode_gemini_stream(
    state: &mut StreamTranslationState,
    event: &Value,
) -> Vec<ConversationStreamEvent> {
    let Some(object) = event.as_object() else {
        return vec![ConversationStreamEvent::Error {
            message: "Gemini stream chunk is not an object".to_string(),
        }];
    };
    if let Some(error) = object.get("error") {
        return vec![ConversationStreamEvent::Error {
            message: error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("unknown Gemini stream error")
                .to_string(),
        }];
    }

    let mut out = Vec::new();
    if !state.saw_message_start {
        state.saw_message_start = true;
        state.message_id = object
            .get("responseId")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        state.model = object
            .get("modelVersion")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        out.push(ConversationStreamEvent::MessageStart {
            id: state.message_id.clone(),
            model: state.model.clone(),
        });
    }

    if let Some(usage) = object.get("usageMetadata") {
        capture_gemini_usage(state, usage);
        out.push(ConversationStreamEvent::Usage(state.usage.clone()));
    }

    let candidate = object
        .get("candidates")
        .and_then(Value::as_array)
        .and_then(|candidates| candidates.first())
        .and_then(Value::as_object);
    if let Some(parts) = candidate
        .and_then(|candidate| candidate.get("content"))
        .and_then(|content| content.get("parts"))
        .and_then(Value::as_array)
    {
        for part in parts.iter().filter_map(Value::as_object) {
            out.extend(decode_gemini_stream_part(state, part));
        }
    }

    if let Some(reason) = candidate
        .and_then(|candidate| candidate.get("finishReason"))
        .and_then(Value::as_str)
    {
        out.push(ConversationStreamEvent::MessageStop {
            reason: Some(neutral_stop_reason(
                reason,
                state.source_tool_calls_seen > 0,
            )),
        });
    }
    out
}

// Decodes one Gemini part into neutral text, reasoning, or tool-call events.
fn decode_gemini_stream_part(
    state: &mut StreamTranslationState,
    part: &Map<String, Value>,
) -> Vec<ConversationStreamEvent> {
    if let Some(call) = part.get("functionCall").and_then(Value::as_object) {
        // Gemini delivers complete function calls in one chunk; emit a single
        // delta carrying the full arguments. Wire-format calls have no index,
        // so each call gets the next per-stream slot.
        let index = state.source_tool_calls_seen;
        state.source_tool_calls_seen += 1;
        return vec![ConversationStreamEvent::ToolCallDelta {
            index,
            id: Some(format!("call_{}", index + 1)),
            name: call
                .get("name")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            arguments_delta: Some(
                call.get("args")
                    .cloned()
                    .unwrap_or_else(|| json!({}))
                    .to_string(),
            ),
        }];
    }
    if let Some(text) = part.get("text").and_then(Value::as_str) {
        if text.is_empty() {
            return Vec::new();
        }
        if part.get("thought").and_then(Value::as_bool) == Some(true) {
            return vec![ConversationStreamEvent::ReasoningDelta {
                index: 0,
                text: text.to_string(),
            }];
        }
        return vec![ConversationStreamEvent::TextDelta {
            index: 0,
            text: text.to_string(),
        }];
    }
    Vec::new()
}

// Encodes neutral streaming events into Gemini chunks.
fn encode_gemini_stream(
    state: &mut StreamTranslationState,
    event: ConversationStreamEvent,
) -> Vec<Value> {
    match event {
        ConversationStreamEvent::MessageStart { id, model } => {
            // Gemini has no dedicated start frame; identity rides on every
            // content chunk instead.
            record_source_identity(state, id, model);
            state.emitted_message_start = true;
            Vec::new()
        }
        ConversationStreamEvent::TextDelta { text, .. } => {
            state.output_tokens_seen += 1;
            state.emitted_content_block = true;
            vec![gemini_chunk(state, json!({"text": text}), None)]
        }
        ConversationStreamEvent::ReasoningDelta { text, .. } => {
            state.emitted_content_block = true;
            vec![gemini_chunk(
                state,
                json!({"text": text, "thought": true}),
                None,
            )]
        }
        ConversationStreamEvent::ToolCallDelta {
            index,
            id,
            name,
            arguments_delta,
        } => {
            // Buffer partial tool calls; Gemini emits complete functionCall
            // parts, so buffered calls flush in the terminal chunk.
            let tool = state.tool_states.entry(index).or_default();
            if id.is_some() {
                tool.id = id;
            }
            if name.is_some() {
                tool.name = name;
            }
            if let Some(delta) = arguments_delta {
                tool.arguments.push_str(&delta);
            }
            Vec::new()
        }
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
            vec![json!({
                "error": {"code": 500, "message": message, "status": "INTERNAL"},
            })]
        }
    }
}

// Emits the terminal Gemini chunk with buffered tool calls, the finish
// reason, and usage metadata.
fn finish_gemini_stream(state: &mut StreamTranslationState) -> Vec<Value> {
    let mut parts = Vec::new();
    for tool in state.tool_states.values() {
        let Some(name) = tool.name.clone() else {
            continue;
        };
        let args = serde_json::from_str::<Value>(&tool.arguments)
            .ok()
            .filter(Value::is_object)
            .unwrap_or_else(|| json!({}));
        parts.push(json!({"functionCall": {"name": name, "args": args}}));
    }
    let has_tool_calls = !parts.is_empty();
    if parts.is_empty() && !state.emitted_content_block {
        parts.push(json!({"text": ""}));
    }

    let mut candidate = Map::new();
    if !parts.is_empty() {
        candidate.insert(
            "content".to_string(),
            json!({"parts": parts, "role": "model"}),
        );
    }
    candidate.insert(
        "finishReason".to_string(),
        Value::String(gemini_stream_finish_reason(
            state.stop_reason.as_deref(),
            has_tool_calls || !state.tool_states.is_empty(),
        )),
    );
    candidate.insert("index".to_string(), json!(0));

    let mut chunk = Map::new();
    chunk.insert(
        "candidates".to_string(),
        Value::Array(vec![Value::Object(candidate)]),
    );
    chunk.insert("usageMetadata".to_string(), gemini_stream_usage(state));
    insert_gemini_identity(state, &mut chunk);
    state.finished = true;
    vec![Value::Object(chunk)]
}

// Builds one streaming content chunk around a single part.
fn gemini_chunk(state: &StreamTranslationState, part: Value, finish_reason: Option<&str>) -> Value {
    let mut candidate = Map::new();
    candidate.insert(
        "content".to_string(),
        json!({"parts": [part], "role": "model"}),
    );
    if let Some(reason) = finish_reason {
        candidate.insert(
            "finishReason".to_string(),
            Value::String(reason.to_string()),
        );
    }
    candidate.insert("index".to_string(), json!(0));
    let mut chunk = Map::new();
    chunk.insert(
        "candidates".to_string(),
        Value::Array(vec![Value::Object(candidate)]),
    );
    insert_gemini_identity(state, &mut chunk);
    Value::Object(chunk)
}

// Stamps the response identity Gemini clients expect on every chunk.
fn insert_gemini_identity(state: &StreamTranslationState, chunk: &mut Map<String, Value>) {
    chunk.insert(
        "modelVersion".to_string(),
        Value::String(target_model_or_source_model(state)),
    );
    if let Some(id) = target_message_id_or_source_message_id(state) {
        chunk.insert("responseId".to_string(), Value::String(id.to_string()));
    }
}

// Maps Gemini finish reasons into the neutral stop-reason vocabulary shared
// by the other stream encoders. Gemini reports STOP for function-call turns,
// so calls seen on this stream win over the raw reason.
fn neutral_stop_reason(reason: &str, saw_tool_calls: bool) -> String {
    match reason {
        "STOP" | "FINISH_REASON_UNSPECIFIED" if saw_tool_calls => "tool_use".to_string(),
        "STOP" | "FINISH_REASON_UNSPECIFIED" => "end_turn".to_string(),
        "MAX_TOKENS" => "max_tokens".to_string(),
        "SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" | "IMAGE_SAFETY" => {
            "content_filter".to_string()
        }
        other => other.to_string(),
    }
}

// Maps neutral stop reasons back to Gemini's finish-reason vocabulary.
fn gemini_stream_finish_reason(reason: Option<&str>, has_tool_calls: bool) -> String {
    match reason {
        Some("max_tokens") | Some("length") => "MAX_TOKENS".to_string(),
        Some("content_filter") => "SAFETY".to_string(),
        // Gemini reports STOP even for tool-call turns.
        Some("tool_use")
        | Some("tool_calls")
        | Some("function_call")
        | Some("end_turn")
        | Some("stop")
        | Some("stop_sequence")
        | None => "STOP".to_string(),
        Some(_) if has_tool_calls => "STOP".to_string(),
        Some(_) => "OTHER".to_string(),
    }
}

// Preserves Gemini usage counters and updates normalized token counts.
fn capture_gemini_usage(state: &mut StreamTranslationState, usage: &Value) {
    let Some(usage) = usage.as_object() else {
        return;
    };
    for (key, target) in [
        ("promptTokenCount", "input_tokens"),
        ("candidatesTokenCount", "candidates_tokens"),
        ("thoughtsTokenCount", "thoughts_tokens"),
        ("totalTokenCount", "total_tokens"),
    ] {
        if let Some(value) = usage.get(key).and_then(Value::as_u64) {
            state.usage_extras.insert(target.to_string(), value);
        }
    }
    let candidates = state.usage_extras.get("candidates_tokens").copied();
    let thoughts = state.usage_extras.get("thoughts_tokens").copied();
    state.usage.input_tokens = state.usage_extras.get("input_tokens").copied();
    // Thinking tokens fold into output tokens to match other providers.
    state.usage.output_tokens = match (candidates, thoughts) {
        (None, None) => None,
        (candidates, thoughts) => Some(
            candidates
                .unwrap_or(0)
                .saturating_add(thoughts.unwrap_or(0)),
        ),
    };
    state.usage.total_tokens = state.usage_extras.get("total_tokens").copied();
    state.usage.reasoning_tokens = thoughts;
}

// Builds the terminal usageMetadata payload from accumulated stream state.
fn gemini_stream_usage(state: &StreamTranslationState) -> Value {
    let mut usage = Map::new();
    if state.saw_backend_usage {
        let input = state.usage.input_tokens.unwrap_or(0);
        let output = state.usage.output_tokens.unwrap_or(0);
        let thoughts = state.usage.reasoning_tokens.unwrap_or(0);
        usage.insert("promptTokenCount".to_string(), json!(input));
        usage.insert(
            "candidatesTokenCount".to_string(),
            json!(output.saturating_sub(thoughts)),
        );
        if thoughts > 0 {
            usage.insert("thoughtsTokenCount".to_string(), json!(thoughts));
        }
        usage.insert(
            "totalTokenCount".to_string(),
            json!(state.usage.total_tokens.unwrap_or(input + output)),
        );
    } else {
        usage.insert(
            "candidatesTokenCount".to_string(),
            json!(state.output_tokens_seen),
        );
        usage.insert(
            "totalTokenCount".to_string(),
            json!(state.output_tokens_seen),
        );
    }
    Value::Object(usage)
}
