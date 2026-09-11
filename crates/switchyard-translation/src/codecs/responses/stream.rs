// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming codec for OpenAI Responses API events.

use serde::Serialize;
use serde_json::{Value, json};

use crate::LlmResponseChunk;
use crate::codecs::common::{
    collect_responses_reasoning_text, encrypted_reasoning_data, encrypted_reasoning_item_id,
};
use crate::codecs::stream::{
    StreamCodec, StreamTranslationState, record_source_identity,
    target_message_id_or_source_message_id, target_model_or_source_model,
};
use crate::format::{FormatId, WireFormat};
use crate::llm::Usage;

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
    ) -> Vec<LlmResponseChunk> {
        decode_responses_stream(state, event)
    }

    fn encode_event(
        &self,
        state: &mut StreamTranslationState,
        event: LlmResponseChunk,
    ) -> Vec<Value> {
        let events = encode_responses_stream(state, event);
        add_sequence_numbers(state, events)
    }

    fn observe_replayed_event(
        &self,
        state: &mut StreamTranslationState,
        raw: &Value,
        normalized: Vec<LlmResponseChunk>,
    ) {
        let replayed_terminal = normalized
            .iter()
            .any(|chunk| matches!(chunk, LlmResponseChunk::MessageStop { .. }));
        // Exact replay emits `raw` once. Normalized encodings only advance codec state;
        // their generated events are discarded and must not create sequence-number gaps.
        for chunk in normalized {
            drop(encode_responses_stream(state, chunk));
        }
        state.response_sequence_number = raw
            .get("sequence_number")
            .and_then(Value::as_u64)
            .map_or(state.response_sequence_number.saturating_add(1), |number| {
                number.saturating_add(1)
            });
        if replayed_terminal {
            state.finished = true;
        }
    }

    fn finish(&self, state: &mut StreamTranslationState) -> Vec<Value> {
        let events = finish_responses_stream(state);
        add_sequence_numbers(state, events)
    }
}

/// Required fields shared by Responses stream snapshots.
#[derive(Serialize)]
struct ResponsesStreamResponse {
    id: String,
    object: &'static str,
    created_at: u64,
    completed_at: Option<u64>,
    error: Option<Value>,
    incomplete_details: Option<Value>,
    instructions: Option<Value>,
    metadata: Option<Value>,
    model: String,
    output: Vec<Value>,
    parallel_tool_calls: bool,
    frequency_penalty: Option<f64>,
    presence_penalty: Option<f64>,
    status: &'static str,
    temperature: Option<f64>,
    tool_choice: &'static str,
    tools: Vec<Value>,
    top_p: Option<f64>,
    usage: Value,
}

// Decodes one OpenAI Responses event into neutral streaming events.
fn decode_responses_stream(
    state: &mut StreamTranslationState,
    event: &Value,
) -> Vec<LlmResponseChunk> {
    // Opt-in raw capture of what the upstream actually sent, for diagnosing provider-specific
    // event shapes: `RUST_LOG=switchyard_translation::responses::raw=trace`.
    tracing::trace!(target: "switchyard_translation::responses::raw", raw = %event);
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
            vec![LlmResponseChunk::MessageStart {
                id: state.message_id.clone(),
                model: state.model.clone(),
            }]
        }
        Some("response.output_text.delta") => event
            .get("delta")
            .and_then(Value::as_str)
            .map(|text| {
                vec![LlmResponseChunk::TextDelta {
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
                    let index = event
                        .get("output_index")
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as usize;
                    // Recorded so `response.output_item.done`, which repeats the full
                    // text, can tell it is a repeat.
                    state
                        .decoded_reasoning
                        .entry(index)
                        .or_default()
                        .push_str(text);
                    vec![LlmResponseChunk::ReasoningDelta {
                        index,
                        text: text.to_string(),
                    }]
                })
                .unwrap_or_default()
        }
        Some("response.output_item.added") => decode_responses_output_item_added(event, state),
        Some("response.function_call_arguments.delta") => {
            let output_index = event
                .get("output_index")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            event
                .get("delta")
                .and_then(Value::as_str)
                .map(|delta| {
                    // Recorded so `response.output_item.done`, which repeats
                    // the complete arguments, can tell it is a repeat.
                    state
                        .tool_states
                        .entry(output_index as usize)
                        .or_default()
                        .decoded_arguments
                        .push_str(delta);
                    vec![LlmResponseChunk::ToolCallDelta {
                        index: output_index as usize,
                        id: None,
                        name: None,
                        arguments_delta: Some(delta.to_string()),
                    }]
                })
                .unwrap_or_default()
        }
        Some("response.output_item.done") => decode_responses_output_item_done(event, state),
        Some("response.reasoning_text.done") | Some("response.reasoning_summary_text.done") => {
            let index = event
                .get("output_index")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize;
            event
                .get("text")
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
                .filter(|_| {
                    state
                        .decoded_reasoning
                        .get(&index)
                        .is_none_or(String::is_empty)
                })
                .map(|text| {
                    state.decoded_reasoning.insert(index, text.to_string());
                    vec![LlmResponseChunk::ReasoningDelta {
                        index,
                        text: text.to_string(),
                    }]
                })
                .unwrap_or_default()
        }
        Some("response.reasoning_summary_part.done") => {
            let index = event
                .get("output_index")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize;
            event
                .get("part")
                .and_then(|part| part.get("text"))
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
                .filter(|_| {
                    state
                        .decoded_reasoning
                        .get(&index)
                        .is_none_or(String::is_empty)
                })
                .map(|text| {
                    state.decoded_reasoning.insert(index, text.to_string());
                    vec![LlmResponseChunk::ReasoningDelta {
                        index,
                        text: text.to_string(),
                    }]
                })
                .unwrap_or_default()
        }
        Some("response.completed") => {
            let mut out = Vec::new();
            // Some providers surface reasoning only in the final output array. Position is the
            // output index; anything already decoded is skipped by the helper.
            if let Some(items) = event
                .get("response")
                .and_then(|response| response.get("output"))
                .and_then(Value::as_array)
            {
                for (position, item) in items.iter().enumerate() {
                    if let Some(item) = item.as_object()
                        && item.get("type").and_then(Value::as_str) == Some("reasoning")
                    {
                        out.extend(decode_responses_reasoning_item(item, position, state));
                    }
                }
            }
            if let Some(usage) = event
                .get("response")
                .and_then(Value::as_object)
                .and_then(|response| response.get("usage"))
                .and_then(Value::as_object)
            {
                let usage = responses_usage(usage);
                state.usage = usage.clone();
                state.saw_backend_usage = true;
                out.push(LlmResponseChunk::Usage(usage));
            }
            out.push(LlmResponseChunk::MessageStop { reason: None });
            out
        }
        // Carries the Anthropic spelling because every encoder already maps it.
        Some("response.incomplete") => vec![LlmResponseChunk::MessageStop {
            reason: Some("max_tokens".to_string()),
        }],
        Some("response.failed") => vec![LlmResponseChunk::StreamError {
            message: event
                .get("response")
                .and_then(|response| response.get("error"))
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("unknown Responses stream error")
                .to_string(),
        }],
        Some("error") => vec![LlmResponseChunk::StreamError {
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
    event: LlmResponseChunk,
) -> Vec<Value> {
    // An in-band error is terminal: once the error is emitted, drop every later chunk.
    if state.errored {
        return Vec::new();
    }
    match event {
        LlmResponseChunk::MessageStart { id, model } => {
            record_source_identity(state, id, model);
            ensure_responses_created(state)
        }
        LlmResponseChunk::TextDelta { text, .. } => encode_responses_text_delta(state, text),
        LlmResponseChunk::ReasoningDelta { index, text } => {
            encode_responses_reasoning_delta(state, index, text)
        }
        LlmResponseChunk::ReasoningDetailsDelta {
            index,
            details,
            text,
        } => {
            // Encrypted reasoning has no streamable text, but the item must still be emitted
            // so the client can replay it on the next turn.
            let data = encrypted_reasoning_data(&details);
            let item = state.response_reasoning.entry(index).or_default();
            match encrypted_reasoning_item_id(&details) {
                Some(id) if item.started && item.item_id.as_deref() != Some(id.as_str()) => {
                    // The item already opened under another id; the payload would fail
                    // verification under it, so drop the payload rather than poison the replay.
                    tracing::warn!(
                        item_id = %id,
                        "encrypted reasoning arrived after its item opened under a different id; dropping payload"
                    );
                }
                id => {
                    if id.is_some() {
                        item.item_id = id;
                    }
                    if data.is_some() {
                        item.encrypted = data;
                    }
                }
            }
            let mut out = ensure_responses_reasoning_started(state, index);
            if !text.is_empty() {
                out.extend(encode_responses_reasoning_delta(state, index, text));
            }
            out
        }
        LlmResponseChunk::ToolCallDelta {
            index,
            id,
            name,
            arguments_delta,
        } => encode_responses_tool_delta(state, index, id, name, arguments_delta),
        LlmResponseChunk::Usage(usage) => {
            state.usage = usage;
            state.saw_backend_usage = true;
            Vec::new()
        }
        LlmResponseChunk::MessageStop { reason } => {
            state.stop_reason = reason.or_else(|| state.stop_reason.clone());
            Vec::new()
        }
        LlmResponseChunk::DecodeError { message } | LlmResponseChunk::StreamError { message } => {
            // An in-band error is terminal: emit the error, then nothing further.
            state.finished = true; // finish() adds no success events
            state.errored = true; // the entry guard drops any later chunk
            vec![json!({"type": "error", "message": message})]
        }
    }
}

// Emits final OpenAI Responses completion events from accumulated state.
fn finish_responses_stream(state: &mut StreamTranslationState) -> Vec<Value> {
    if state.finished {
        return Vec::new();
    }
    let is_truncated = matches!(
        state.stop_reason.as_deref(),
        Some("length") | Some("max_tokens")
    );
    let (event_type, status) = if is_truncated {
        ("response.incomplete", "incomplete")
    } else {
        ("response.completed", "completed")
    };
    let incomplete_details = is_truncated.then(|| json!({ "reason": "max_output_tokens" }));
    let mut out = ensure_responses_created(state);
    if state.response_text_started
        && let Some(output_index) = state.response_text_output_index
    {
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
                "id": responses_item_id(state, "msg", output_index),
                "role": "assistant",
                "status": status,
                "content": [{"type": "output_text", "text": state.response_text}],
            },
        }));
    }

    let mut final_items: Vec<(usize, Value)> = Vec::new();
    let mut reasoning_items: Vec<(usize, usize)> = state
        .response_reasoning
        .iter()
        .filter(|(_, item)| item.started)
        .filter_map(|(index, item)| item.output_index.map(|output| (*index, output)))
        .collect();
    reasoning_items.sort_by_key(|(_, output_index)| *output_index);
    for (index, output_index) in reasoning_items {
        // Encrypted-only reasoning streamed no text, so it gets no summary part; the item
        // itself still closes so the client can replay its `encrypted_content`.
        let item_id = responses_reasoning_item_id(state, index);
        let reasoning = &state.response_reasoning[&index];
        let mut summary = Vec::new();
        if !reasoning.text.is_empty() {
            out.push(json!({
                "type": "response.reasoning_summary_text.done",
                "item_id": item_id,
                "output_index": output_index,
                "summary_index": 0,
                "text": reasoning.text,
            }));
            out.push(json!({
                "type": "response.reasoning_summary_part.done",
                "item_id": item_id,
                "output_index": output_index,
                "summary_index": 0,
                "part": {"type": "summary_text", "text": reasoning.text},
            }));
            summary.push(json!({
                "type": "summary_text",
                "text": reasoning.text,
            }));
        }
        let mut item = json!({
            "type": "reasoning",
            "id": item_id,
            "status": "completed",
            "summary": summary,
        });
        if let Some(encrypted) = &reasoning.encrypted {
            item["encrypted_content"] = Value::String(encrypted.clone());
        }
        out.push(json!({
            "type": "response.output_item.done",
            "output_index": output_index,
            "item": item,
        }));
        final_items.push((output_index, item));
    }
    if state.response_text_started
        && let Some(output_index) = state.response_text_output_index
    {
        final_items.push((
            output_index,
            json!({
                "type": "message",
                "id": responses_item_id(state, "msg", output_index),
                "role": "assistant",
                "status": status,
                "content": [{"type": "output_text", "text": state.response_text}],
            }),
        ));
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
            "id": tool.response_item_id.clone().unwrap_or_else(|| responses_item_id(state, "fc", output_index)),
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
        "type": event_type,
        "response": responses_stream_response(state, status, incomplete_details, output),
    }));
    state.finished = true;
    out
}

// Converts Responses function-call item creation into a neutral tool-call delta.
// Decodes the reasoning a provider put on a reasoning output item itself: plaintext in
// `content`, `summary`, or top-level `text`, and/or an opaque `encrypted_content`. Text that was
// already decoded for this output index (from delta events or an earlier item event) is not
// repeated, so `added`, `done`, and the final `response.completed` output can all be inspected
// safely.
fn decode_responses_reasoning_item(
    item: &serde_json::Map<String, Value>,
    index: usize,
    state: &mut StreamTranslationState,
) -> Vec<LlmResponseChunk> {
    let mut out = Vec::new();
    let has_encrypted = item
        .get("encrypted_content")
        .and_then(Value::as_str)
        .is_some_and(|data| !data.is_empty());
    // The provider's item id is known from the first `added` event, long before the encrypted
    // payload arrives on `done`. Announce it so the encoder opens the item under that id;
    // otherwise summary text opens it under a synthesized id the payload cannot verify against.
    if let Some(id) = item
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        && !state.decoded_reasoning_ids.contains(&index)
    {
        state.decoded_reasoning_ids.insert(index);
        if !has_encrypted {
            out.push(LlmResponseChunk::ReasoningDetailsDelta {
                index,
                details: vec![json!({"type": "reasoning.encrypted", "id": id})],
                text: String::new(),
            });
        }
    }
    let mut parts = Vec::new();
    collect_responses_reasoning_text(item.get("content"), &mut parts);
    collect_responses_reasoning_text(item.get("summary"), &mut parts);
    collect_responses_reasoning_text(item.get("text"), &mut parts);
    let text = parts.join("\n");
    let already = state
        .decoded_reasoning
        .get(&index)
        .map(String::as_str)
        .unwrap_or("");
    if !text.is_empty() && already.is_empty() {
        state.decoded_reasoning.insert(index, text.clone());
        out.push(LlmResponseChunk::ReasoningDelta { index, text });
    }
    if let Some(data) = item
        .get("encrypted_content")
        .and_then(Value::as_str)
        .filter(|data| !data.is_empty())
        && !state.decoded_reasoning_encrypted.contains(&index)
    {
        state.decoded_reasoning_encrypted.insert(index);
        // The payload only verifies under the id it was issued with, so carry that id along.
        let mut detail = json!({"type": "reasoning.encrypted", "data": data});
        if let Some(id) = item
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
        {
            detail["id"] = Value::String(id.to_string());
        }
        out.push(LlmResponseChunk::ReasoningDetailsDelta {
            index,
            details: vec![detail],
            text: String::new(),
        });
    }
    out
}

fn decode_responses_output_item_added(
    event: &Value,
    state: &mut StreamTranslationState,
) -> Vec<LlmResponseChunk> {
    let Some(item) = event.get("item").and_then(Value::as_object) else {
        return Vec::new();
    };
    let index = event
        .get("output_index")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    if item.get("type").and_then(Value::as_str) == Some("reasoning") {
        return decode_responses_reasoning_item(item, index, state);
    }
    let item_type = item.get("type").and_then(Value::as_str);
    if item_type != Some("function_call") && item_type != Some("custom_tool_call") {
        return Vec::new();
    }
    // A freeform call's `input` becomes the single `input` argument; it is only complete on
    // the done event, so nothing is emitted for it here beyond id and name.
    let arguments_delta = if item_type == Some("custom_tool_call") {
        None
    } else {
        item.get("arguments")
            .and_then(Value::as_str)
            .filter(|arguments| !arguments.is_empty())
            .map(ToOwned::to_owned)
    };
    if let Some(arguments) = arguments_delta.as_deref() {
        state
            .tool_states
            .entry(index)
            .or_default()
            .decoded_arguments
            .push_str(arguments);
    }
    vec![LlmResponseChunk::ToolCallDelta {
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
        arguments_delta,
    }]
}

// Emits a final tool-call argument delta when Responses only supplies arguments at item end.
fn decode_responses_output_item_done(
    event: &Value,
    state: &mut StreamTranslationState,
) -> Vec<LlmResponseChunk> {
    let Some(item) = event.get("item").and_then(Value::as_object) else {
        return Vec::new();
    };
    let index = event
        .get("output_index")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    if item.get("type").and_then(Value::as_str) == Some("reasoning") {
        return decode_responses_reasoning_item(item, index, state);
    }
    let item_type = item.get("type").and_then(Value::as_str);
    if item_type != Some("function_call") && item_type != Some("custom_tool_call") {
        return Vec::new();
    }
    let custom_arguments = (item_type == Some("custom_tool_call")).then(|| {
        json!({
            crate::codex_custom_tools::INPUT_ARGUMENT:
                item.get("input").and_then(Value::as_str).unwrap_or_default()
        })
        .to_string()
    });
    let arguments = custom_arguments
        .as_deref()
        .or_else(|| item.get("arguments").and_then(Value::as_str));
    if let Some(arguments) = arguments {
        // Compared against what THIS decoder has seen. Reading the encoder's
        // `arguments` instead only deduplicates when a single state performs
        // both halves of the translation, and silently duplicates when a
        // caller buffers the stream with its own state.
        let tool = state.tool_states.entry(index).or_default();
        if !arguments.is_empty() && arguments != tool.decoded_arguments {
            tool.decoded_arguments.push_str(arguments);
            return vec![LlmResponseChunk::ToolCallDelta {
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
        "response": responses_stream_response(state, "in_progress", None, Vec::new()),
    })]
}

// Builds a schema-complete Responses snapshot for strict generated clients.
fn responses_stream_response(
    state: &StreamTranslationState,
    status: &'static str,
    incomplete_details: Option<Value>,
    output: Vec<Value>,
) -> ResponsesStreamResponse {
    ResponsesStreamResponse {
        id: responses_id(state),
        object: "response",
        created_at: 0,
        completed_at: None,
        error: None,
        incomplete_details,
        instructions: None,
        metadata: None,
        model: target_model_or_source_model(state),
        output,
        parallel_tool_calls: true,
        frequency_penalty: None,
        presence_penalty: None,
        status,
        temperature: None,
        tool_choice: "auto",
        tools: Vec::new(),
        top_p: None,
        usage: responses_usage_value(&state.usage),
    }
}

// Assigns monotonically increasing sequence numbers to generated Responses events.
fn add_sequence_numbers(state: &mut StreamTranslationState, mut events: Vec<Value>) -> Vec<Value> {
    for event in &mut events {
        if let Some(object) = event.as_object_mut() {
            object.insert(
                "sequence_number".to_string(),
                Value::from(state.response_sequence_number),
            );
            state.response_sequence_number = state.response_sequence_number.saturating_add(1);
        }
    }
    events
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
                "id": responses_item_id(state, "msg", output_index),
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

// The id of the reasoning item encoded for a source index: the provider's own id when
// encrypted reasoning binds to it, else synthesized from the emitted output index.
fn responses_reasoning_item_id(state: &StreamTranslationState, index: usize) -> String {
    let item = state.response_reasoning.get(&index);
    item.and_then(|item| item.item_id.clone())
        .unwrap_or_else(|| {
            let output_index = item.and_then(|item| item.output_index).unwrap_or(0);
            responses_item_id(state, "rs", output_index)
        })
}

// Opens the Responses reasoning output item for a source index once, emitting its `added` events.
fn ensure_responses_reasoning_started(
    state: &mut StreamTranslationState,
    index: usize,
) -> Vec<Value> {
    let mut out = ensure_responses_created(state);
    if state
        .response_reasoning
        .get(&index)
        .is_some_and(|item| item.started)
    {
        return out;
    }
    let output_index = state.next_response_output_index;
    state.next_response_output_index += 1;
    let item = state.response_reasoning.entry(index).or_default();
    item.started = true;
    item.output_index = Some(output_index);
    let item_id = responses_reasoning_item_id(state, index);
    // Standard Responses shape: reasoning text lives in `summary` as `summary_text`
    // parts. Clients such as Codex record reasoning items only in this shape.
    out.push(json!({
        "type": "response.output_item.added",
        "output_index": output_index,
        "item": {
            "type": "reasoning",
            "id": item_id,
            "status": "in_progress",
            "summary": [],
        },
    }));
    out
}

// Opens the item's single summary part the first time text streams for it. Encrypted-only
// reasoning never streams text, so it never opens a part; the item closes with an empty
// `summary`, matching what `finish_responses_stream` emits for it.
fn ensure_responses_reasoning_summary_started(
    state: &mut StreamTranslationState,
    index: usize,
) -> Vec<Value> {
    let mut out = ensure_responses_reasoning_started(state, index);
    let item_id = responses_reasoning_item_id(state, index);
    let item = state.response_reasoning.entry(index).or_default();
    if item.summary_started {
        return out;
    }
    item.summary_started = true;
    out.push(json!({
        "type": "response.reasoning_summary_part.added",
        "item_id": item_id,
        "output_index": item.output_index.unwrap_or(0),
        "summary_index": 0,
        "part": {"type": "summary_text", "text": ""},
    }));
    out
}

// Accumulates reasoning text and emits Responses reasoning events.
fn encode_responses_reasoning_delta(
    state: &mut StreamTranslationState,
    index: usize,
    text: String,
) -> Vec<Value> {
    // An empty delta carries nothing to show and must not open a part that would never close.
    if text.is_empty() {
        return ensure_responses_reasoning_started(state, index);
    }
    let mut out = ensure_responses_reasoning_summary_started(state, index);
    let item = state.response_reasoning.entry(index).or_default();
    item.text.push_str(&text);
    let output_index = item.output_index.unwrap_or(0);
    out.push(json!({
        "type": "response.reasoning_summary_text.delta",
        "item_id": responses_reasoning_item_id(state, index),
        "output_index": output_index,
        "summary_index": 0,
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
    let resp_id = responses_id(state);
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
        tool.response_item_id = Some(responses_item_id_from(&resp_id, "fc", output_index));
        tool.started = true;
        out.push(json!({
            "type": "response.output_item.added",
            "output_index": output_index,
            "item": {
                "type": "function_call",
                "id": tool.response_item_id.clone().unwrap_or_else(|| responses_item_id_from(&resp_id, "fc", output_index)),
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

    if let Some(output_index) = tool.response_output_index
        && !tool.pending_arguments.is_empty()
    {
        out.push(json!({
            "type": "response.function_call_arguments.delta",
            "output_index": output_index,
            "delta": tool.pending_arguments,
        }));
        tool.pending_arguments.clear();
    }
    out
}

// Normalizes OpenAI Responses token usage fields.
fn responses_usage(usage: &serde_json::Map<String, Value>) -> Usage {
    let aggregate_input_tokens = usage.get("input_tokens").and_then(Value::as_u64);
    let cached_input_tokens = usage
        .get("input_tokens_details")
        .and_then(|details| details.get("cached_tokens"))
        .and_then(Value::as_u64);
    let input_tokens = aggregate_input_tokens
        .map(|tokens| tokens.saturating_sub(cached_input_tokens.unwrap_or(0)));
    let output_tokens = usage.get("output_tokens").and_then(Value::as_u64);
    Usage {
        input_tokens,
        cache: Usage::cache_details(cached_input_tokens, None),
        output_tokens,
        total_tokens: usage
            .get("total_tokens")
            .and_then(Value::as_u64)
            .or_else(|| Some(aggregate_input_tokens.unwrap_or(0) + output_tokens.unwrap_or(0))),
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
    let input_tokens = usage.input_tokens.unwrap_or(0)
        + usage.cached_input_tokens().unwrap_or(0)
        + usage.cache_creation_input_tokens().unwrap_or(0);
    // Both detail objects are always present, for the same reason as the buffered encoder: the
    // Responses schema types them as required, so a missing breakdown serializes as zero.
    json!({
        "input_tokens": input_tokens,
        "output_tokens": usage.output_tokens.unwrap_or(0),
        "total_tokens": usage.total_tokens.unwrap_or_else(|| {
            input_tokens + usage.output_tokens.unwrap_or(0)
        }),
        "input_tokens_details": {"cached_tokens": usage.cached_input_tokens().unwrap_or(0)},
        "output_tokens_details": {"reasoning_tokens": usage.reasoning_tokens.unwrap_or(0)},
    })
}

// Longest response-id discriminator embedded verbatim in a synthesized item id. OpenAI rejects
// item ids over 64 characters, and some upstreams issue response ids several hundred characters
// long, so anything longer is replaced by a fixed-width digest.
const ITEM_ID_DISCRIMINATOR_CHARS: usize = 40;

// Builds a synthesized output-item id that is unique across responses. Clients replay the
// whole conversation, so ids must not repeat from one turn to the next; the response id is
// unique per upstream call and is used as the discriminator.
fn responses_item_id(state: &StreamTranslationState, prefix: &str, output_index: usize) -> String {
    responses_item_id_from(&responses_id(state), prefix, output_index)
}

// Same as [`responses_item_id`], for callers that already hold the response id and cannot
// borrow `state` again (for example while a `tool_states` entry is borrowed mutably).
fn responses_item_id_from(resp_id: &str, prefix: &str, output_index: usize) -> String {
    let resp = resp_id.strip_prefix("resp_").unwrap_or(resp_id);
    if resp.chars().count() <= ITEM_ID_DISCRIMINATOR_CHARS {
        format!("{prefix}_{resp}_{output_index}")
    } else {
        format!("{prefix}_{:016x}_{output_index}", fnv1a_64(resp))
    }
}

// FNV-1a over the UTF-8 bytes: a stable, dependency-free 64-bit digest. Uniqueness across the
// handful of responses in one conversation is all that is required of it.
fn fnv1a_64(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
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
