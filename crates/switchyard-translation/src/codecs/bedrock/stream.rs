// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ConverseStream JSON events. AWS EventStream binary framing belongs to the host.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::LlmResponseChunk;
use crate::codecs::stream::{StreamCodec, StreamTranslationState, record_source_identity};
use crate::{FormatId, WireFormat};

use super::buffered::{decode_usage, encode_usage};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum BlockKind {
    Text,
    Reasoning,
    Tool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct BedrockStreamState {
    started: bool,
    stopped: bool,
    metadata: bool,
    decoded_blocks: BTreeMap<usize, BlockKind>,
    closed_blocks: BTreeSet<usize>,
    active: Option<(usize, BlockKind)>,
    active_source_index: Option<usize>,
    active_tool: Option<usize>,
    replayed_stop: bool,
    closed_tools: BTreeSet<usize>,
}

/// Stream codec for ConverseStream's de-framed JSON union events.
pub struct BedrockConverseStreamCodec;

impl StreamCodec for BedrockConverseStreamCodec {
    fn format(&self) -> FormatId {
        WireFormat::BedrockConverse.into()
    }

    fn decode_event(
        &self,
        state: &mut StreamTranslationState,
        event: &Value,
    ) -> Vec<LlmResponseChunk> {
        match decode(state, event) {
            Ok(chunks) => chunks,
            Err(message) => vec![LlmResponseChunk::DecodeError { message }],
        }
    }

    fn encode_event(
        &self,
        state: &mut StreamTranslationState,
        event: LlmResponseChunk,
    ) -> Vec<Value> {
        encode(state, event)
    }

    fn observe_replayed_event(
        &self,
        state: &mut StreamTranslationState,
        raw: &Value,
        normalized: Vec<LlmResponseChunk>,
    ) {
        for chunk in normalized {
            drop(encode(state, chunk));
        }
        let index = raw
            .get("contentBlockStart")
            .or_else(|| raw.get("contentBlockDelta"))
            .and_then(|block| block.get("contentBlockIndex"))
            .and_then(Value::as_u64)
            .and_then(|index| usize::try_from(index).ok());
        if let Some(index) = index {
            if let Some((active, _)) = state.bedrock.active.as_mut() {
                *active = index;
            }
            if let Some(tool_index) = state.bedrock.active_tool
                && let Some(tool) = state.tool_states.get_mut(&tool_index)
            {
                tool.content_index = Some(index);
            }
            state.next_content_index = state.next_content_index.max(index.saturating_add(1));
        }
        if raw.get("contentBlockStop").is_some() {
            state.bedrock.active = None;
            state.bedrock.active_tool = None;
        }
        if raw.get("messageStop").is_some() {
            state.bedrock.replayed_stop = true;
        }
        if raw.get("metadata").is_some() {
            state.finished = true;
        }
    }

    fn finish(&self, state: &mut StreamTranslationState) -> Vec<Value> {
        finish(state)
    }
}

fn decode(
    state: &mut StreamTranslationState,
    event: &Value,
) -> Result<Vec<LlmResponseChunk>, String> {
    let obj = event.as_object().ok_or("Bedrock event must be an object")?;
    if obj.len() != 1 {
        return Err("Bedrock event must contain one union member".into());
    }
    let (key, payload) = obj.iter().next().ok_or("empty Bedrock event")?;
    let payload = payload
        .as_object()
        .ok_or("Bedrock event payload must be an object")?;
    if state.bedrock.metadata {
        return Err("Bedrock event follows terminal metadata".into());
    }
    if key.ends_with("Exception") {
        return Ok(vec![LlmResponseChunk::StreamError {
            message: format!(
                "Bedrock {key}: {}",
                payload
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("upstream stream failed")
            ),
        }]);
    }
    if key == "messageStart" {
        if state.bedrock.started {
            return Err("duplicate Bedrock messageStart".into());
        }
        if payload.get("role").and_then(Value::as_str) != Some("assistant") {
            return Err("Bedrock response role must be assistant".into());
        }
        state.bedrock.started = true;
        state.saw_message_start = true;
        return Ok(vec![LlmResponseChunk::MessageStart {
            id: None,
            model: None,
        }]);
    }
    if !state.bedrock.started {
        return Err("Bedrock event precedes messageStart".into());
    }
    if key == "metadata" {
        if !state.bedrock.stopped {
            return Err("Bedrock metadata precedes messageStop".into());
        }
        let usage = payload
            .get("usage")
            .ok_or("Bedrock metadata is missing usage")?;
        state.usage = decode_usage(usage).map_err(|e| e.to_string())?;
        state.bedrock.metadata = true;
        state.saw_backend_usage = true;
        return Ok(vec![LlmResponseChunk::Usage(state.usage.clone())]);
    }
    if state.bedrock.stopped {
        return Err("Bedrock content follows messageStop".into());
    }
    if key == "messageStop" {
        if state
            .bedrock
            .decoded_blocks
            .keys()
            .any(|i| !state.bedrock.closed_blocks.contains(i))
        {
            return Err("Bedrock messageStop precedes contentBlockStop".into());
        }
        let reason = payload
            .get("stopReason")
            .and_then(Value::as_str)
            .ok_or("Bedrock messageStop is missing stopReason")?;
        state.bedrock.stopped = true;
        state.stop_reason = Some(reason.into());
        let reason = match reason {
            "content_filtered" | "guardrail_intervened" => "content_filter",
            "model_context_window_exceeded" => "max_tokens",
            other => other,
        };
        return Ok(vec![LlmResponseChunk::MessageStop {
            reason: Some(reason.into()),
        }]);
    }
    let index = payload
        .get("contentBlockIndex")
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or("Bedrock event is missing a valid contentBlockIndex")?;
    if state.bedrock.closed_blocks.contains(&index) {
        return Err("Bedrock event uses a closed content block".into());
    }
    match key.as_str() {
        "contentBlockStart" => {
            if state.bedrock.decoded_blocks.contains_key(&index) {
                return Err("duplicate Bedrock contentBlockStart".into());
            }
            let start = payload
                .get("start")
                .and_then(Value::as_object)
                .ok_or("Bedrock contentBlockStart is missing start")?;
            if start.len() != 1 {
                return Err("Bedrock start must contain one union member".into());
            }
            let tool = start
                .get("toolUse")
                .and_then(Value::as_object)
                .ok_or("unsupported Bedrock contentBlockStart")?;
            let id = tool
                .get("toolUseId")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .ok_or("Bedrock tool start is missing toolUseId")?;
            let name = tool
                .get("name")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .ok_or("Bedrock tool start is missing name")?;
            state.bedrock.decoded_blocks.insert(index, BlockKind::Tool);
            state.decoded_tool_call = true;
            Ok(vec![LlmResponseChunk::ToolCallDelta {
                index,
                id: Some(id.into()),
                name: Some(name.into()),
                arguments_delta: None,
            }])
        }
        "contentBlockDelta" => {
            let delta = payload
                .get("delta")
                .and_then(Value::as_object)
                .ok_or("Bedrock content delta is missing delta")?;
            if delta.len() != 1 {
                return Err("Bedrock delta must contain one union member".into());
            }
            let (kind, chunk) = if let Some(text) = delta.get("text") {
                (
                    BlockKind::Text,
                    LlmResponseChunk::TextDelta {
                        index,
                        text: text
                            .as_str()
                            .ok_or("Bedrock text delta must be a string")?
                            .into(),
                    },
                )
            } else if let Some(tool) = delta.get("toolUse") {
                if state.bedrock.decoded_blocks.get(&index) != Some(&BlockKind::Tool) {
                    return Err("Bedrock tool delta precedes tool start".into());
                }
                (
                    BlockKind::Tool,
                    LlmResponseChunk::ToolCallDelta {
                        index,
                        id: None,
                        name: None,
                        arguments_delta: Some(
                            tool.get("input")
                                .and_then(Value::as_str)
                                .ok_or("Bedrock tool input delta must be a string")?
                                .into(),
                        ),
                    },
                )
            } else if let Some(reasoning) = delta.get("reasoningContent") {
                let reasoning = reasoning
                    .as_object()
                    .ok_or("Bedrock reasoning delta must be an object")?;
                if reasoning.len() != 1 {
                    return Err("Bedrock reasoning delta must contain one union member".into());
                }
                let chunk = if let Some(text) = reasoning.get("text") {
                    LlmResponseChunk::ReasoningDelta {
                        index,
                        text: text
                            .as_str()
                            .ok_or("Bedrock reasoning text must be a string")?
                            .into(),
                    }
                } else if let Some(signature) = reasoning.get("signature") {
                    LlmResponseChunk::ReasoningDetailsDelta {
                        index,
                        text: String::new(),
                        details: vec![
                            json!({"type": "bedrock.signature_delta", "signature": signature.as_str().ok_or("Bedrock signature must be a string")?}),
                        ],
                    }
                } else if let Some(data) = reasoning.get("redactedContent") {
                    LlmResponseChunk::ReasoningDetailsDelta {
                        index,
                        text: String::new(),
                        details: vec![
                            json!({"type": "bedrock.redacted_content", "data": data.as_str().ok_or("Bedrock redactedContent must be base64 text")?}),
                        ],
                    }
                } else {
                    return Err("unsupported Bedrock reasoning delta".into());
                };
                (BlockKind::Reasoning, chunk)
            } else {
                return Err("unsupported Bedrock content delta".into());
            };
            if state
                .bedrock
                .decoded_blocks
                .get(&index)
                .is_some_and(|existing| existing != &kind)
            {
                return Err("Bedrock delta changes content block kind".into());
            }
            state.bedrock.decoded_blocks.insert(index, kind);
            Ok(vec![chunk])
        }
        "contentBlockStop" => {
            if !state.bedrock.decoded_blocks.contains_key(&index) {
                return Err("Bedrock stop names an unknown content block".into());
            }
            state.bedrock.closed_blocks.insert(index);
            Ok(Vec::new())
        }
        _ => Err(format!("unsupported Bedrock stream event: {key}")),
    }
}

fn message_start(state: &mut StreamTranslationState, out: &mut Vec<Value>) {
    if !state.emitted_message_start {
        state.emitted_message_start = true;
        out.push(json!({"messageStart": {"role": "assistant"}}));
    }
}

fn close_block(state: &mut StreamTranslationState, out: &mut Vec<Value>) {
    if let Some((index, _)) = state.bedrock.active.take() {
        out.push(json!({"contentBlockStop": {"contentBlockIndex": index}}));
    }
    if let Some(index) = state.bedrock.active_tool.take() {
        state.bedrock.closed_tools.insert(index);
    }
}

fn content_index(
    state: &mut StreamTranslationState,
    kind: BlockKind,
    source_index: usize,
    out: &mut Vec<Value>,
) -> usize {
    message_start(state, out);
    if let Some((index, active)) = state.bedrock.active
        && active == kind
        && state.bedrock.active_source_index == Some(source_index)
    {
        return index;
    }
    close_block(state, out);
    let index = state.next_content_index;
    state.next_content_index += 1;
    state.bedrock.active = Some((index, kind));
    state.bedrock.active_source_index = Some(source_index);
    index
}

fn encode(state: &mut StreamTranslationState, event: LlmResponseChunk) -> Vec<Value> {
    if state.errored
        || (state.finished
            && !matches!(
                event,
                LlmResponseChunk::StreamError { .. } | LlmResponseChunk::DecodeError { .. }
            ))
    {
        return Vec::new();
    }
    let mut out = Vec::new();
    match event {
        LlmResponseChunk::MessageStart { id, model } => {
            record_source_identity(state, id, model);
            message_start(state, &mut out);
        }
        LlmResponseChunk::TextDelta { text, index } => {
            let index = content_index(state, BlockKind::Text, index, &mut out);
            out.push(
                json!({"contentBlockDelta": {"contentBlockIndex": index, "delta": {"text": text}}}),
            );
        }
        LlmResponseChunk::ReasoningDelta { text, index } => {
            let index = content_index(state, BlockKind::Reasoning, index, &mut out);
            out.push(json!({"contentBlockDelta": {"contentBlockIndex": index, "delta": {"reasoningContent": {"text": text}}}}));
        }
        LlmResponseChunk::ReasoningDetailsDelta {
            text,
            details,
            index,
        } => {
            if !text.is_empty() {
                out.extend(encode(
                    state,
                    LlmResponseChunk::ReasoningDelta { index, text },
                ));
            }
            for detail in details {
                let value = match detail.get("type").and_then(Value::as_str) {
                    Some("bedrock.signature_delta") => {
                        detail.get("signature").map(|v| json!({"signature": v}))
                    }
                    Some("bedrock.redacted_content") => {
                        detail.get("data").map(|v| json!({"redactedContent": v}))
                    }
                    _ => {
                        return encode(
                            state,
                            LlmResponseChunk::DecodeError {
                                message: "opaque reasoning details have no Bedrock stream mapping"
                                    .into(),
                            },
                        );
                    }
                };
                if let Some(value) = value {
                    let index = content_index(state, BlockKind::Reasoning, index, &mut out);
                    out.push(json!({"contentBlockDelta": {"contentBlockIndex": index, "delta": {"reasoningContent": value}}}));
                }
            }
        }
        LlmResponseChunk::ToolCallDelta {
            index,
            id,
            name,
            arguments_delta,
        } => {
            if state.bedrock.closed_tools.contains(&index) {
                return encode(
                    state,
                    LlmResponseChunk::DecodeError {
                        message: "Bedrock cannot resume a tool block after other content".into(),
                    },
                );
            }
            let tool = state.tool_states.entry(index).or_default();
            if id.is_some() {
                tool.id = id;
            }
            if name.is_some() {
                tool.name = name;
            }
            if let Some(delta) = arguments_delta {
                tool.pending_arguments.push_str(&delta);
            }
            if state.bedrock.active_tool.is_none() || state.bedrock.active_tool == Some(index) {
                emit_tool(state, index, &mut out);
            }
        }
        LlmResponseChunk::MessageStop { reason } => {
            state.stop_reason = reason.or_else(|| state.stop_reason.clone());
        }
        LlmResponseChunk::Usage(usage) => {
            state.usage = usage;
            state.saw_backend_usage = true;
        }
        LlmResponseChunk::StreamError { message } | LlmResponseChunk::DecodeError { message } => {
            state.errored = true;
            state.finished = true;
            out.push(json!({"modelStreamErrorException": {"message": message}}));
        }
    }
    out
}

fn emit_tool(state: &mut StreamTranslationState, index: usize, out: &mut Vec<Value>) {
    let Some(tool) = state.tool_states.get(&index) else {
        return;
    };
    let (Some(id), Some(name)) = (tool.id.as_ref(), tool.name.as_ref()) else {
        return;
    };
    let id = id.clone();
    let name = name.clone();
    if !tool.started {
        message_start(state, out);
        close_block(state, out);
        let content_index = state.next_content_index;
        state.next_content_index += 1;
        state.bedrock.active = Some((content_index, BlockKind::Tool));
        state.bedrock.active_tool = Some(index);
        let tool = state.tool_states.entry(index).or_default();
        tool.started = true;
        tool.content_index = Some(content_index);
        out.push(json!({"contentBlockStart": {"contentBlockIndex": content_index, "start": {"toolUse": {"toolUseId": id, "name": name}}}}));
    }
    let tool = state.tool_states.entry(index).or_default();
    if !tool.pending_arguments.is_empty() {
        let delta = std::mem::take(&mut tool.pending_arguments);
        tool.arguments.push_str(&delta);
        out.push(json!({"contentBlockDelta": {"contentBlockIndex": tool.content_index, "delta": {"toolUse": {"input": delta}}}}));
    }
}

fn finish(state: &mut StreamTranslationState) -> Vec<Value> {
    if state.finished || state.errored {
        return Vec::new();
    }
    let mut out = Vec::new();
    message_start(state, &mut out);
    // Bedrock blocks are sequential. Parallel tools after the first wait until it closes.
    let pending = state.tool_states.keys().copied().collect::<Vec<_>>();
    for index in pending {
        if state.bedrock.active_tool != Some(index)
            && state
                .tool_states
                .get(&index)
                .is_some_and(|tool| tool.started)
        {
            continue;
        }
        emit_tool(state, index, &mut out);
        if state
            .tool_states
            .get(&index)
            .is_some_and(|tool| !tool.pending_arguments.is_empty() || !tool.started)
        {
            state.errored = true;
            state.finished = true;
            return vec![
                json!({"modelStreamErrorException": {"message": "tool stream ended without a tool ID and name"}}),
            ];
        }
    }
    close_block(state, &mut out);
    if !state.bedrock.replayed_stop {
        let reason = match state.stop_reason.as_deref() {
            Some("length" | "max_tokens") => "max_tokens",
            Some("model_context_window_exceeded") => "model_context_window_exceeded",
            Some("tool_calls" | "function_call" | "tool_use") => "tool_use",
            Some("content_filter" | "content_filtered") => "content_filtered",
            Some("guardrail_intervened") => "guardrail_intervened",
            Some("stop_sequence") => "stop_sequence",
            Some("malformed_model_output" | "malformed_tool_use") => "malformed_model_output",
            _ => "end_turn",
        };
        out.push(json!({"messageStop": {"stopReason": reason}}));
    }
    match encode_usage(&state.usage) {
        Ok(usage) => out.push(json!({"metadata": {"usage": usage}})),
        Err(error) => {
            state.errored = true;
            state.finished = true;
            return vec![json!({"modelStreamErrorException": {"message": error.to_string()}})];
        }
    }
    state.finished = true;
    out
}

pub(crate) fn saw_terminal(state: &StreamTranslationState) -> bool {
    state.bedrock.metadata
}
