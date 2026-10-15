// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Assemble MCP blocks and defer conversion until their required payloads arrive.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{FormatCodec, anthropic, responses, stream::StreamTranslationState};
use crate::{
    AggLlmResponse, ContentBlock, LlmResponseChunk, McpContent, PreservationPolicy, ResponseOutput,
    Role, TranslationError, TranslationPolicy, WireFormat,
};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct McpStreamState {
    pending: BTreeMap<usize, PendingBlock>,
    // Responses needs payloads for final-snapshot checks; Anthropic only needs the index.
    completed: BTreeMap<usize, Option<Value>>,
    queue: VecDeque<LlmResponseChunk>,
    replay_pending: BTreeSet<usize>,
    emitted_ids: BTreeSet<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct PendingBlock {
    raw: Value,
    arguments: String,
    arguments_done: bool,
}

fn error(message: impl Into<String>) -> Vec<LlmResponseChunk> {
    vec![LlmResponseChunk::DecodeError {
        message: message.into(),
    }]
}

fn index(event: &Value, key: &str) -> Result<usize, &'static str> {
    event[key]
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or("MCP event requires a nonnegative index")
}

fn call_delta(
    index: usize,
    raw: &Value,
    source: WireFormat,
    delta: Option<String>,
) -> LlmResponseChunk {
    LlmResponseChunk::McpCallDelta {
        index,
        id: raw["id"].as_str().map(str::to_owned),
        server_name: raw[if source == WireFormat::AnthropicMessages {
            "server_name"
        } else {
            "server_label"
        }]
        .as_str()
        .map(str::to_owned),
        name: raw["name"].as_str().map(str::to_owned),
        arguments_delta: delta,
    }
}

// Optional defaults can differ between item.done and the final response snapshot.
// Keep every other field in the comparison, including unknown provider fields.
fn mcp_snapshots_match(previous: &Value, current: &Value) -> bool {
    if previous == current {
        return true;
    }
    let (Some(left), Some(right)) = (previous.as_object(), current.as_object()) else {
        return false;
    };
    let is_call = previous["type"] == "mcp_call";
    let is_listing = previous["type"] == "mcp_list_tools";
    let is_successful_call = is_call
        && previous["error"].is_null()
        && current["error"].is_null()
        && previous["output"].is_string()
        && current["output"].is_string();
    let completed = json!("completed");
    left.keys().chain(right.keys()).all(|key| {
        let a = left.get(key);
        let b = right.get(key);
        match key.as_str() {
            "error" if is_call || is_listing => {
                a.unwrap_or(&Value::Null) == b.unwrap_or(&Value::Null)
            }
            "approval_request_id" | "output" if is_call => {
                a.unwrap_or(&Value::Null) == b.unwrap_or(&Value::Null)
            }
            "status" if is_successful_call => a.unwrap_or(&completed) == b.unwrap_or(&completed),
            _ => a == b,
        }
    })
}

fn complete(
    state: &mut StreamTranslationState,
    index: usize,
    raw: Value,
    source: WireFormat,
) -> Vec<LlmResponseChunk> {
    if source == WireFormat::OpenAiResponses
        && matches!(raw["type"].as_str(), Some("mcp_call" | "mcp_list_tools"))
    {
        for key in ["id", "server_label"] {
            if raw[key].as_str().is_none_or(str::is_empty) {
                return error(format!("MCP item requires a nonempty {key}"));
            }
        }
        if raw["type"] == "mcp_call" && raw["name"].as_str().is_none_or(str::is_empty) {
            return error("MCP call requires a nonempty name");
        }
        if raw["type"] == "mcp_call"
            && raw.get("error").is_none_or(Value::is_null)
            && raw.get("status").is_none_or(|status| status == "completed")
            && raw["arguments"]
                .as_str()
                .and_then(|arguments| serde_json::from_str::<Value>(arguments).ok())
                .is_none_or(|arguments| !arguments.is_object())
        {
            return error("MCP arguments must be a complete JSON object");
        }
    }
    if let Some(Some(previous)) = state.mcp.completed.get(&index) {
        return if mcp_snapshots_match(previous, &raw) {
            Vec::new()
        } else {
            error("MCP snapshot conflicts with completed content")
        };
    }
    if let Some(pending) = state.mcp.pending.remove(&index) {
        for key in ["id", "name", "server_label", "server_name", "type"] {
            if pending
                .raw
                .get(key)
                .is_some_and(|value| !value.is_null() && raw.get(key) != Some(value))
            {
                return error("MCP snapshot conflicts with streamed identity");
            }
        }
        if source == WireFormat::OpenAiResponses && !pending.arguments.is_empty() {
            // Final snapshots may supply an unsent suffix, but cannot replace streamed arguments.
            let matches = if pending.arguments_done {
                serde_json::from_str::<Value>(&pending.arguments).ok()
                    == raw["arguments"]
                        .as_str()
                        .and_then(|value| serde_json::from_str::<Value>(value).ok())
            } else {
                raw["arguments"]
                    .as_str()
                    .is_some_and(|value| value.starts_with(&pending.arguments))
            };
            if !matches {
                return error("MCP snapshot conflicts with streamed arguments");
            }
        }
    }
    let Some(object) = raw.as_object() else {
        return error("MCP block must be an object");
    };
    let blocks = if source == WireFormat::AnthropicMessages {
        match anthropic::decode_anthropic_content_block(
            object,
            Role::Assistant,
            index,
            &mut Vec::new(),
            &state.policy,
        ) {
            Ok(blocks) => blocks,
            Err(err) => return error(err.to_string()),
        }
    } else {
        responses::decode_mcp_item(object)
    };
    state.mcp.completed.insert(
        index,
        (source == WireFormat::OpenAiResponses).then_some(raw),
    );
    blocks
        .into_iter()
        .map(|block| LlmResponseChunk::McpContentBlockComplete {
            index,
            block: Box::new(block),
        })
        .collect()
}

pub(crate) fn decode_anthropic(
    state: &mut StreamTranslationState,
    event: &Value,
) -> Option<Vec<LlmResponseChunk>> {
    let kind = event.get("type").or_else(|| event.get("event"))?.as_str()?;
    let is_start = kind == "content_block_start"
        && event["content_block"]["type"]
            .as_str()
            .is_some_and(|kind| kind.starts_with("mcp_"));
    let parsed_index = index(event, "index");
    let existing = parsed_index
        .as_ref()
        .is_ok_and(|index| state.mcp.pending.contains_key(index));
    if matches!(
        kind,
        "content_block_start" | "content_block_delta" | "content_block_stop"
    ) && parsed_index
        .as_ref()
        .is_ok_and(|index| state.mcp.completed.contains_key(index))
    {
        return Some(error("MCP block received an event after completion"));
    }
    if !is_start && !existing {
        return None;
    }
    let index = match parsed_index {
        Ok(index) => index,
        Err(message) => return Some(error(message)),
    };
    Some(match kind {
        "content_block_start" => {
            if existing || state.mcp.completed.contains_key(&index) {
                return Some(error("duplicate MCP block index"));
            }
            let raw = event["content_block"].clone();
            let chunks = if raw["type"] == "mcp_tool_use" {
                for key in ["id", "name", "server_name"] {
                    if raw[key].as_str().is_none_or(str::is_empty) {
                        return Some(error(format!(
                            "mcp_tool_use.{key} requires a nonempty string"
                        )));
                    }
                }
                vec![call_delta(index, &raw, WireFormat::AnthropicMessages, None)]
            } else {
                vec![LlmResponseChunk::McpContentBlockStart { index }]
            };
            state.mcp.pending.insert(
                index,
                PendingBlock {
                    raw,
                    ..Default::default()
                },
            );
            chunks
        }
        "content_block_delta" => {
            let pending = state.mcp.pending.get_mut(&index)?;
            if pending.raw["type"] != "mcp_tool_use" || event["delta"]["type"] != "input_json_delta"
            {
                return Some(error("unsupported MCP content delta"));
            }
            let Some(delta) = event["delta"]["partial_json"].as_str() else {
                return Some(error("MCP arguments delta must be a string"));
            };
            pending.arguments.push_str(delta);
            vec![call_delta(
                index,
                &pending.raw,
                WireFormat::AnthropicMessages,
                Some(delta.into()),
            )]
        }
        "content_block_stop" => {
            let pending = state.mcp.pending.get(&index)?;
            let mut raw = pending.raw.clone();
            if !pending.arguments.is_empty() {
                let Ok(arguments) = serde_json::from_str::<Value>(&pending.arguments) else {
                    return Some(error("MCP arguments are not complete JSON"));
                };
                if !arguments.is_object() {
                    return Some(error("MCP arguments must be an object"));
                }
                if raw["input"]
                    .as_object()
                    .is_some_and(|input| !input.is_empty())
                    && raw["input"] != arguments
                {
                    return Some(error("MCP arguments conflict with the initial input"));
                }
                raw["input"] = arguments;
            }
            complete(state, index, raw, WireFormat::AnthropicMessages)
        }
        _ => return None,
    })
}

pub(crate) fn decode_responses(
    state: &mut StreamTranslationState,
    event: &Value,
) -> Option<Vec<LlmResponseChunk>> {
    let kind = event.get("type").or_else(|| event.get("event"))?.as_str()?;
    let is_item = matches!(
        kind,
        "response.output_item.added" | "response.output_item.done"
    ) && event["item"]["type"]
        .as_str()
        .is_some_and(responses::is_mcp_item);
    if !is_item && !kind.starts_with("response.mcp_") {
        return None;
    }
    let index = match index(event, "output_index") {
        Ok(index) => index,
        Err(message) => return Some(error(message)),
    };
    Some(match kind {
        "response.output_item.added" => {
            if state.mcp.pending.contains_key(&index) || state.mcp.completed.contains_key(&index) {
                return Some(error("duplicate MCP output index"));
            }
            let raw = event["item"].clone();
            if raw["id"].as_str().is_none_or(str::is_empty) {
                return Some(error("MCP item requires a nonempty ID"));
            }
            let chunks = if raw["type"] == "mcp_call" {
                vec![call_delta(index, &raw, WireFormat::OpenAiResponses, None)]
            } else {
                vec![LlmResponseChunk::McpContentBlockStart { index }]
            };
            state.mcp.pending.insert(
                index,
                PendingBlock {
                    raw,
                    ..Default::default()
                },
            );
            chunks
        }
        "response.output_item.done" => complete(
            state,
            index,
            event["item"].clone(),
            WireFormat::OpenAiResponses,
        ),
        "response.mcp_call_arguments.delta" | "response.mcp_call_arguments.done" => {
            let Some(pending) = state.mcp.pending.get_mut(&index) else {
                return Some(error("MCP arguments arrived before their call"));
            };
            if pending.raw["type"] != "mcp_call" || event["item_id"] != pending.raw["id"] {
                return Some(error("MCP argument event has the wrong call ID"));
            }
            if kind.ends_with(".done") {
                let Some(arguments) = event["arguments"].as_str() else {
                    return Some(error("MCP arguments must be a string"));
                };
                if !arguments.starts_with(&pending.arguments) {
                    return Some(error("MCP arguments conflict with streamed content"));
                }
                pending.arguments = arguments.into();
                pending.arguments_done = true;
                Vec::new()
            } else {
                if pending.arguments_done {
                    return Some(error("MCP arguments arrived after completion"));
                }
                let Some(delta) = event["delta"].as_str() else {
                    return Some(error("MCP argument delta must be a string"));
                };
                pending.arguments.push_str(delta);
                vec![call_delta(
                    index,
                    &pending.raw,
                    WireFormat::OpenAiResponses,
                    Some(delta.into()),
                )]
            }
        }
        // Lifecycle notifications carry no result payload. The completed item supplies it.
        "response.mcp_call.in_progress"
        | "response.mcp_call.completed"
        | "response.mcp_call.failed"
        | "response.mcp_list_tools.in_progress"
        | "response.mcp_list_tools.completed"
        | "response.mcp_list_tools.failed" => {
            let raw = state
                .mcp
                .pending
                .get(&index)
                .map(|pending| &pending.raw)
                .or_else(|| state.mcp.completed.get(&index).and_then(Option::as_ref));
            if raw.is_none_or(|raw| raw["id"] != event["item_id"]) {
                return Some(error("MCP status event has no matching item"));
            }
            Vec::new()
        }
        _ => error("unsupported MCP stream event"),
    })
}

pub(crate) fn decode_snapshot(
    state: &mut StreamTranslationState,
    index: usize,
    item: &Value,
) -> Option<Vec<LlmResponseChunk>> {
    item["type"]
        .as_str()
        .filter(|kind| responses::is_mcp_item(kind))?;
    Some(complete(
        state,
        index,
        item.clone(),
        WireFormat::OpenAiResponses,
    ))
}

pub(crate) fn incomplete(state: &StreamTranslationState) -> bool {
    !state.mcp.pending.is_empty()
}

pub(crate) fn observe_replay(state: &mut StreamTranslationState, chunk: &LlmResponseChunk) -> bool {
    match chunk {
        LlmResponseChunk::McpCallDelta { index, .. }
        | LlmResponseChunk::McpContentBlockStart { index } => {
            state.mcp.replay_pending.insert(*index);
            true
        }
        LlmResponseChunk::McpContentBlockComplete { index, .. } => {
            state.mcp.replay_pending.remove(index);
            true
        }
        _ => false,
    }
}

fn pending_index(chunk: &LlmResponseChunk) -> Option<usize> {
    match chunk {
        LlmResponseChunk::McpCallDelta { index, .. }
        | LlmResponseChunk::McpContentBlockStart { index } => Some(*index),
        _ => None,
    }
}

// Calls wait for complete arguments and, for Responses, their result. Later chunks
// stay queued so conversion or dropping a call cannot reorder the conversation.
// There is no size cap: a missing result can retain the rest of the response until EOF.
pub(crate) fn encode(
    state: &mut StreamTranslationState,
    event: LlmResponseChunk,
    target: WireFormat,
    emit: fn(&mut StreamTranslationState, LlmResponseChunk) -> Vec<Value>,
) -> Vec<Value> {
    if state.errored {
        return Vec::new();
    }
    if matches!(
        event,
        LlmResponseChunk::DecodeError { .. } | LlmResponseChunk::StreamError { .. }
    ) {
        state.mcp.queue.clear();
        return emit(state, event);
    }
    if state.mcp.queue.is_empty()
        && !matches!(
            event,
            LlmResponseChunk::McpCallDelta { .. }
                | LlmResponseChunk::McpContentBlockStart { .. }
                | LlmResponseChunk::McpContentBlockComplete { .. }
        )
    {
        return emit(state, event);
    }
    let terminal = matches!(event, LlmResponseChunk::MessageStop { .. });
    match &event {
        LlmResponseChunk::McpCallDelta { index, .. }
        | LlmResponseChunk::McpContentBlockStart { index } => {
            if !state
                .mcp
                .queue
                .iter()
                .any(|chunk| pending_index(chunk) == Some(*index))
            {
                state.mcp.queue.push_back(event);
            }
        }
        LlmResponseChunk::McpContentBlockComplete { index, .. } => {
            if let Some(position) = state
                .mcp
                .queue
                .iter()
                .position(|chunk| pending_index(chunk) == Some(*index))
            {
                state.mcp.queue[position] = event;
            } else {
                state.mcp.queue.push_back(event);
            }
        }
        _ => state.mcp.queue.push_back(event),
    }
    drain(state, target, terminal, emit)
}

pub(crate) fn finish(
    state: &mut StreamTranslationState,
    target: WireFormat,
    emit: fn(&mut StreamTranslationState, LlmResponseChunk) -> Vec<Value>,
) -> Vec<Value> {
    if state.errored {
        return Vec::new();
    }
    if !state.mcp.replay_pending.is_empty() || incomplete(state) {
        return emit(
            state,
            LlmResponseChunk::DecodeError {
                message: "MCP stream ended before block completion".into(),
            },
        );
    }
    drain(state, target, true, emit)
}

fn drain(
    state: &mut StreamTranslationState,
    target: WireFormat,
    terminal: bool,
    emit: fn(&mut StreamTranslationState, LlmResponseChunk) -> Vec<Value>,
) -> Vec<Value> {
    let mut out = Vec::new();
    while let Some(front) = state.mcp.queue.front() {
        if pending_index(front).is_some() {
            if terminal {
                out.extend(emit(
                    state,
                    LlmResponseChunk::DecodeError {
                        message: "MCP stream ended before call arguments completed".into(),
                    },
                ));
            }
            break;
        }
        let result_position = if let LlmResponseChunk::McpContentBlockComplete { block, .. } = front
            && let ContentBlock::Mcp(McpContent::Call { call, .. }) = block.as_ref()
        {
            if target == WireFormat::OpenAiResponses {
                let position = state.mcp.queue.iter().position(|chunk| matches!(chunk,
                    LlmResponseChunk::McpContentBlockComplete { block, .. }
                    if matches!(block.as_ref(), ContentBlock::Mcp(McpContent::Result(result)) if result.tool_call_id == call.id)));
                if position.is_none() && !terminal {
                    break;
                }
                position
            } else {
                None
            }
        } else {
            None
        };
        let result = result_position.and_then(|position| state.mcp.queue.remove(position));
        let Some(chunk) = state.mcp.queue.pop_front() else {
            break;
        };
        if let LlmResponseChunk::McpContentBlockComplete { index, block } = chunk {
            let mut blocks = vec![*block];
            if let Some(LlmResponseChunk::McpContentBlockComplete { block, .. }) = result {
                blocks.push(*block);
            }
            match convert(state, target, blocks) {
                Ok(items) => {
                    for raw in items {
                        out.extend(emit(
                            state,
                            LlmResponseChunk::McpContentBlockComplete {
                                index,
                                block: Box::new(ContentBlock::Unknown {
                                    provider: target.into(),
                                    raw,
                                }),
                            },
                        ));
                    }
                }
                Err(err) => {
                    out.extend(emit(
                        state,
                        LlmResponseChunk::DecodeError {
                            message: err.to_string(),
                        },
                    ));
                    break;
                }
            }
        } else {
            out.extend(emit(state, chunk));
        }
        if state.errored {
            break;
        }
    }
    out
}

fn convert(
    state: &mut StreamTranslationState,
    target: WireFormat,
    blocks: Vec<ContentBlock>,
) -> crate::Result<Vec<Value>> {
    let generated_listing = matches!(
        blocks.first(),
        Some(ContentBlock::Mcp(McpContent::Listing { id: None, .. }))
    );
    let policy = TranslationPolicy {
        preservation: PreservationPolicy::Disabled,
        ..state.policy.clone()
    };
    let response = AggLlmResponse {
        outputs: vec![ResponseOutput {
            role: Role::Assistant,
            content: blocks,
            stop_reason: None,
            url_citations: Vec::new(),
        }],
        ..Default::default()
    };
    let encoded = match target {
        WireFormat::AnthropicMessages => {
            anthropic::AnthropicMessagesCodec.encode_response(&response, &policy)
        }
        WireFormat::OpenAiResponses => {
            responses::OpenAiResponsesCodec.encode_response(&response, &policy)
        }
        WireFormat::OpenAiChat => {
            crate::util::push_lossy(
                &mut state.diagnostics,
                &policy,
                "MCP or complete provider content cannot be streamed as Chat Completions",
            )?;
            return Ok(Vec::new());
        }
    };
    let encoded = match encoded {
        Ok(encoded) => encoded,
        Err(TranslationError::LossyConversion(message)) => {
            crate::util::push_lossy(&mut state.diagnostics, &policy, message)?;
            return Ok(Vec::new());
        }
        Err(error) => return Err(error),
    };
    state.diagnostics.extend(encoded.diagnostics);
    let key = if target == WireFormat::AnthropicMessages {
        "content"
    } else {
        "output"
    };
    let mut items = encoded.body[key].as_array().cloned().unwrap_or_default();
    // Buffered encoders may use an empty text placeholder when every block was dropped.
    let placeholder = if target == WireFormat::AnthropicMessages {
        json!({"type": "text", "text": ""})
    } else {
        json!({
            "type": "message", "id": "msg_switchyard", "status": "completed", "role": "assistant",
            "content": [{"type": "output_text", "text": "", "annotations": []}],
        })
    };
    items.retain(|item| item != &placeholder);
    for item in &mut items {
        if target == WireFormat::OpenAiResponses
            && let Some(id) = item["id"].as_str()
            && !state.mcp.emitted_ids.insert(id.to_owned())
        {
            // A generated listing ID is local to each buffered conversion.
            if generated_listing {
                let crate::DeterministicIdPolicy::GenerateStable { prefix } =
                    &policy.deterministic_ids
                else {
                    return Err(TranslationError::InvalidValue {
                        path: "mcp.id".into(),
                        message: "duplicate MCP item ID".into(),
                    });
                };
                let id = crate::util::unused_stable_id(
                    &format!("{prefix}_mcpl"),
                    state.next_response_output_index,
                    |id| state.mcp.emitted_ids.contains(id),
                );
                state.mcp.emitted_ids.insert(id.clone());
                item["id"] = json!(id);
            } else {
                return Err(TranslationError::InvalidValue {
                    path: "mcp.id".into(),
                    message: "duplicate MCP item ID".into(),
                });
            }
        }
    }
    Ok(items)
}
