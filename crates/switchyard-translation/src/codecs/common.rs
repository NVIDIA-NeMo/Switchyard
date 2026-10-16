// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-agnostic helpers shared by wire-format codecs.

use std::borrow::Cow;

use serde_json::{Map, Value};

use crate::error::{Result, TranslationError};
use crate::llm::{ContentBlock, LlmRequest, McpContent, Tool, ToolChoice};
use crate::{
    AggLlmResponse, PreservationMetadata, TranslationDiagnostic, TranslationPolicy, WireFormat,
};

// Internal provenance survives mutations that invalidate exact request replay.
pub(crate) const ANTHROPIC_REQUEST_KEY: &str = "switchyard_anthropic_request";

/// Returns true when the request was decoded from an Anthropic Messages body.
pub(crate) fn is_anthropic_request(request: &LlmRequest) -> bool {
    request.extensions.fields.get(ANTHROPIC_REQUEST_KEY) == Some(&Value::Bool(true))
}

/// Converts an OpenAI allowed-tools policy to a restricted function list and mode.
pub(crate) fn allowed_function_tools(
    request: &LlmRequest,
) -> Result<Option<(Vec<Tool>, ToolChoice)>> {
    let Some(ToolChoice::Raw(choice)) = &request.tool_choice else {
        return Ok(None);
    };
    if choice.get("type").and_then(Value::as_str) != Some("allowed_tools") {
        return Ok(None);
    }
    let error = || {
        TranslationError::LossyConversion(
            "allowed_tools translation requires auto or required mode \
             and a nonempty subset of known function tools"
                .to_string(),
        )
    };
    let policy = choice.get("allowed_tools").unwrap_or(choice);
    let mode = match policy.get("mode").and_then(Value::as_str) {
        Some("auto") => ToolChoice::Auto,
        Some("required") => ToolChoice::Required,
        _ => return Err(error()),
    };
    let allowed = policy
        .get("tools")
        .and_then(Value::as_array)
        .ok_or_else(error)?;
    if allowed.is_empty() {
        return Err(error());
    }
    let mut names = Vec::with_capacity(allowed.len());
    for tool in allowed {
        if tool.get("type").and_then(Value::as_str) != Some("function") {
            return Err(error());
        }
        let function = tool.get("function").unwrap_or(tool);
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
            .ok_or_else(error)?;
        let name = match function.get("namespace").and_then(Value::as_str) {
            Some(namespace) if !namespace.is_empty() => {
                crate::codex_namespaces::qualified_tool_name(namespace, name)
            }
            _ => name.to_string(),
        };
        if !request
            .tools
            .iter()
            .filter_map(Tool::function)
            .any(|tool| tool.name == name)
        {
            return Err(error());
        }
        names.push(name);
    }
    // Filter the catalog, rather than rebuilding it from selectors, to retain definitions
    // and avoid duplicating tools when the allowed list repeats a name.
    let tools = request
        .tools
        .iter()
        .filter(|tool| {
            tool.function()
                .is_some_and(|definition| names.contains(&definition.name))
        })
        .cloned()
        .collect();
    Ok(Some((tools, mode)))
}

/// Returns whether a role name is recognized by a supported provider API.
pub(crate) fn is_known_role_name(name: &str) -> bool {
    matches!(
        name,
        "system" | "developer" | "user" | "assistant" | "tool" | "function"
    )
}

/// Extracts text-like blocks and joins them for text-only provider fields.
pub(crate) fn text_from_blocks(content: &[ContentBlock], separator: &str) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            ContentBlock::Refusal { text } => Some(text.as_str()),
            ContentBlock::Unknown { raw, .. } => raw.as_str(),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(separator)
}

/// Extracts private reasoning blocks without mixing them into visible text.
pub(crate) fn reasoning_text_from_blocks(content: &[ContentBlock], separator: &str) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Reasoning { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(separator)
}

/// Extracts displayable text from structured reasoning details.
pub(crate) fn reasoning_text_from_details(details: &[Value]) -> Option<String> {
    let parts = details
        .iter()
        .filter_map(Value::as_object)
        .filter_map(|detail| {
            detail
                .get("text")
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
                .or_else(|| {
                    detail
                        .get("summary")
                        .and_then(Value::as_str)
                        .filter(|summary| !summary.is_empty())
                })
        })
        .collect::<Vec<_>>();
    (!parts.is_empty()).then(|| parts.join("\n"))
}

/// Collects reasoning text from a Responses reasoning item's `content` or `summary`
/// array, or from a bare string, into `out`. Empty strings are skipped.
pub(crate) fn collect_responses_reasoning_text(value: Option<&Value>, out: &mut Vec<String>) {
    match value {
        Some(Value::String(text)) if !text.is_empty() => out.push(text.clone()),
        Some(Value::Array(items)) => {
            for item in items {
                match item {
                    Value::String(text) if !text.is_empty() => out.push(text.clone()),
                    Value::Object(object) => {
                        if matches!(
                            object.get("type").and_then(Value::as_str),
                            Some("reasoning_text" | "summary_text" | "text")
                        ) && let Some(text) = object.get("text").and_then(Value::as_str)
                            && !text.is_empty()
                        {
                            out.push(text.to_string());
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
}

/// Returns the opaque payload of the first encrypted reasoning detail, if any.
///
/// Two detail shapes are accepted: the documented `{"type": "reasoning.encrypted", "data"}`
/// object, and a verbatim Responses `reasoning` item carrying `encrypted_content` (the shape
/// the buffered request decoder stores when it keeps the provider item whole).
pub(crate) fn encrypted_reasoning_data(details: &[Value]) -> Option<String> {
    details
        .iter()
        .filter_map(Value::as_object)
        .find_map(|detail| match detail.get("type").and_then(Value::as_str) {
            Some("reasoning.encrypted") => detail.get("data").and_then(Value::as_str),
            Some("reasoning") => detail.get("encrypted_content").and_then(Value::as_str),
            _ => None,
        })
        .filter(|data| !data.is_empty())
        .map(ToOwned::to_owned)
}

/// Returns the provider item id recorded on the first `reasoning.encrypted` detail, if any.
/// Encrypted reasoning is bound to the item id it was issued under, so a replay must reuse it.
pub(crate) fn encrypted_reasoning_item_id(details: &[Value]) -> Option<String> {
    details
        .iter()
        .filter_map(Value::as_object)
        .find(|detail| {
            matches!(
                detail.get("type").and_then(Value::as_str),
                Some("reasoning.encrypted" | "reasoning")
            )
        })
        .and_then(|detail| detail.get("id").and_then(Value::as_str))
        .filter(|id| !id.is_empty())
        .map(ToOwned::to_owned)
}

/// Returns the first non-empty string stored under the requested keys.
pub(crate) fn first_nonempty_string<'a>(
    object: &'a Map<String, Value>,
    keys: &[&str],
) -> Option<&'a str> {
    keys.iter().find_map(|key| {
        object
            .get(*key)
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
    })
}

/// Copies unknown provider fields into the IR extension map.
pub(crate) fn provider_extensions(
    object: &Map<String, Value>,
    known: &[&str],
) -> Map<String, Value> {
    let mut extensions = Map::new();
    for (key, value) in object {
        if key != ANTHROPIC_REQUEST_KEY
            && key != super::anthropic::MCP_LAYOUTS_KEY
            && !known.contains(&key.as_str())
        {
            extensions.insert(key.clone(), value.clone());
        }
    }
    extensions
}

// Remove unsupported tool data before exact replay or preservation embedding can restore it.
pub(crate) fn prepare_request_tools<'a>(
    request: &'a LlmRequest,
    target: WireFormat,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Cow<'a, LlmRequest>> {
    let mut unsupported = has_unsupported_tool_choice(request, target)
        || request
            .tools
            .iter()
            .any(|tool| is_unsupported_tool(tool, request, target))
        || has_unsupported_preserved_tools(request, target);
    let mut has_mcp = request.tools.iter().any(|tool| {
        matches!(tool, Tool::Mcp(_))
            || matches!(tool, Tool::Provider { raw, .. } if raw["type"] == "mcp")
    });
    // Ordinary requests need only one content walk, including provider blocks
    // that may appear without a tool definition.
    for block in request_content(request) {
        has_mcp = has_mcp || is_mcp_content(block);
        unsupported = unsupported || is_unsupported_tool_content(block, target);
        if has_mcp && unsupported {
            break;
        }
    }
    has_mcp = has_mcp
        || has_mcp_tools(
            request
                .extensions
                .fields
                .get(crate::codex_custom_tools::ADDITIONAL_TOOLS_KEY)
                .unwrap_or(&Value::Null),
        )
        || request
            .preservation
            .requests
            .values()
            .any(preserved_request_has_mcp);
    if !unsupported && (!has_mcp || can_replay_mcp_request(request, target)) {
        return Ok(Cow::Borrowed(request));
    }
    if unsupported {
        crate::util::push_lossy(
            diagnostics,
            policy,
            format!("unmapped MCP or provider tools and history are dropped for {target}"),
        )?;
    }
    let mut request = drop_unsupported_request_tools(request, target);
    // Clear every format's cached body and embedded envelope: either could restore credentials.
    request.preservation = PreservationMetadata::default();
    request
        .extensions
        .fields
        .remove(crate::codex_custom_tools::ADDITIONAL_TOOLS_KEY);
    if let Some(metadata) = request
        .extensions
        .fields
        .get_mut("metadata")
        .and_then(Value::as_object_mut)
    {
        metadata.remove(crate::PRESERVATION_METADATA_KEY);
    }
    Ok(Cow::Owned(request))
}

fn request_content(request: &LlmRequest) -> impl Iterator<Item = &ContentBlock> {
    request
        .messages
        .iter()
        .flat_map(|message| &message.content)
        .chain(
            request
                .instructions
                .iter()
                .flat_map(|instruction| &instruction.content),
        )
}

fn is_unsupported_tool(tool: &Tool, request: &LlmRequest, target: WireFormat) -> bool {
    match tool {
        Tool::Function(_) => false,
        Tool::Mcp(mcp) => match target {
            WireFormat::AnthropicMessages => !super::anthropic::can_encode_mcp(mcp, request),
            WireFormat::OpenAiResponses => !super::responses::can_encode_mcp(mcp),
            WireFormat::OpenAiChat => true,
        },
        Tool::Provider { provider, .. } => provider.as_str() != target.as_str(),
    }
}

fn has_unsupported_tool_choice(request: &LlmRequest, target: WireFormat) -> bool {
    target != WireFormat::OpenAiResponses
        && matches!(&request.tool_choice, Some(ToolChoice::Raw(raw)) if raw["type"] == "mcp")
}

fn has_unsupported_preserved_tools(request: &LlmRequest, target: WireFormat) -> bool {
    if target == WireFormat::AnthropicMessages {
        return false;
    }
    if target == WireFormat::OpenAiChat
        && request
            .extensions
            .fields
            .contains_key(super::anthropic::MCP_LAYOUTS_KEY)
    {
        return true;
    }
    if request.preservation.requests.is_empty() {
        return false;
    }
    let Some(body) = request
        .preservation
        .requests
        .get(&WireFormat::AnthropicMessages.into())
    else {
        return false;
    };
    if target == WireFormat::OpenAiChat
        && body["mcp_servers"]
            .as_array()
            .is_some_and(|servers| !servers.is_empty())
    {
        return true;
    }
    body["tools"].as_array().is_some_and(|tools| {
        tools.iter().any(|tool| {
            if super::anthropic::is_function_tool(tool) {
                return false;
            }
            match target {
                WireFormat::OpenAiResponses => tool["type"] != "mcp_toolset",
                WireFormat::OpenAiChat => true,
                WireFormat::AnthropicMessages => false,
            }
        })
    })
}

fn has_mcp_tools(tools: &Value) -> bool {
    tools.as_array().is_some_and(|tools| {
        tools.iter().any(|tool| {
            matches!(tool["type"].as_str(), Some("mcp" | "mcp_toolset"))
                || (tool["type"] == "namespace" && has_mcp_tools(&tool["tools"]))
        })
    })
}

fn has_mcp_history(items: &Value) -> bool {
    items.as_array().is_some_and(|items| {
        items.iter().any(|item| {
            item["type"]
                .as_str()
                .is_some_and(|kind| kind.starts_with("mcp_"))
        })
    })
}

fn preserved_request_has_mcp(body: &Value) -> bool {
    body.get("mcp_servers").is_some()
        || has_mcp_tools(&body["tools"])
        || has_mcp_history(&body["input"])
        || body["input"].as_array().is_some_and(|items| {
            items
                .iter()
                .any(|item| item["type"] == "additional_tools" && has_mcp_tools(&item["tools"]))
        })
        || body["messages"].as_array().is_some_and(|messages| {
            messages
                .iter()
                .any(|message| has_mcp_history(&message["content"]))
        })
}

fn preserved_response_has_mcp(body: &Value) -> bool {
    has_mcp_history(&body["output"]) || has_mcp_history(&body["content"])
}

// Multiple snapshots can contain stale credentials or history. Comparing only the
// target snapshot cannot make the other snapshots safe to embed, so clear them all.
fn can_replay_mcp_request(request: &LlmRequest, target: WireFormat) -> bool {
    request.preservation.requests.len() == 1
        && request
            .preservation
            .requests
            .get(&target.into())
            .is_some_and(|body| {
                preservation_codec(target)
                    .decode_request(body, &preservation_check_policy())
                    .is_ok_and(|original| {
                        original.request.tools == request.tools
                            && original.request.messages == request.messages
                            && instructions_match(&original.request, request)
                            && original.request.tool_choice == request.tool_choice
                            && original.request.extensions.fields == request.extensions.fields
                    })
            })
}

fn instructions_match(original: &LlmRequest, request: &LlmRequest) -> bool {
    if original.instructions == request.instructions {
        return true;
    }
    // Target prompts create separate IR blocks, but the wire body may combine
    // them into one system instruction. Compare the text those encoders emit.
    let system_text = |request: &LlmRequest| {
        let mut parts = Vec::new();
        for instruction in &request.instructions {
            if instruction.role != crate::llm::Role::System {
                return None;
            }
            for block in &instruction.content {
                let ContentBlock::Text { text } = block else {
                    return None;
                };
                parts.push(text.as_str());
            }
        }
        Some(parts.join("\n\n"))
    };
    system_text(original).is_some_and(|text| Some(text) == system_text(request))
}

fn drop_unsupported_request_tools(original: &LlmRequest, target: WireFormat) -> LlmRequest {
    let removed_servers: std::collections::HashSet<&str> = original
        .tools
        .iter()
        .filter(|tool| is_unsupported_tool(tool, original, target))
        .filter_map(|tool| match tool {
            Tool::Mcp(mcp) => Some(mcp.name.as_str()),
            Tool::Provider { raw, .. } if raw["type"] == "mcp" => raw["server_label"].as_str(),
            _ => None,
        })
        .collect();
    let removed_calls: std::collections::HashSet<&str> = if removed_servers.is_empty() {
        Default::default()
    } else {
        request_content(original)
            .filter_map(|block| match block {
                ContentBlock::Mcp(McpContent::Call { server_name, call })
                    if removed_servers.contains(server_name.as_str()) =>
                {
                    Some(call.id.as_str())
                }
                _ => None,
            })
            .collect()
    };
    let orphaned = |block: &ContentBlock| match block {
        ContentBlock::Mcp(
            McpContent::Call { server_name, .. } | McpContent::Listing { server_name, .. },
        ) => removed_servers.contains(server_name.as_str()),
        ContentBlock::Mcp(McpContent::Result(result)) => {
            removed_calls.contains(result.tool_call_id.as_str())
        }
        ContentBlock::Unknown { raw, .. } => raw["server_label"]
            .as_str()
            .is_some_and(|name| removed_servers.contains(name)),
        _ => false,
    };
    let unsupported_choice = has_unsupported_tool_choice(original, target);
    let mut request = original.clone();
    request
        .tools
        .retain(|tool| !is_unsupported_tool(tool, original, target));
    if target != WireFormat::AnthropicMessages {
        request
            .extensions
            .fields
            .remove(super::anthropic::MCP_LAYOUTS_KEY);
    }
    request.messages.retain_mut(|message| {
        message.content.retain(|block| !orphaned(block));
        drop_unsupported_tool_content(&mut message.content, target);
        !message.content.is_empty()
    });
    request.instructions.retain_mut(|instruction| {
        instruction.content.retain(|block| !orphaned(block));
        drop_unsupported_tool_content(&mut instruction.content, target);
        !instruction.content.is_empty()
    });
    if unsupported_choice
        || request.tools.is_empty()
        || matches!(&request.tool_choice, Some(ToolChoice::Tool { name })
        if !request.tools.iter().any(|tool| match tool {
            Tool::Function(definition) => &definition.name == name,
            Tool::Provider { raw, .. } => raw.get("name").and_then(Value::as_str) == Some(name),
            Tool::Mcp(_) => false,
        }))
    {
        request.tool_choice = None;
    }
    request
}

fn is_mcp_content(block: &ContentBlock) -> bool {
    match block {
        ContentBlock::Mcp(_) => true,
        ContentBlock::Unknown { provider, raw } => {
            (provider.as_str() == WireFormat::OpenAiResponses.as_str()
                && raw["type"]
                    .as_str()
                    .is_some_and(super::responses::is_mcp_item))
                || (provider.as_str() == WireFormat::AnthropicMessages.as_str()
                    && raw["type"]
                        .as_str()
                        .is_some_and(|kind| kind.starts_with("mcp_")))
        }
        ContentBlock::ToolResult(result) => result.content.iter().any(is_mcp_content),
        _ => false,
    }
}

fn is_unsupported_tool_content(block: &ContentBlock, target: WireFormat) -> bool {
    match block {
        ContentBlock::Mcp(_) => target == WireFormat::OpenAiChat,
        ContentBlock::Unknown { provider, raw } => {
            provider.as_str() != target.as_str()
                && (is_mcp_content(block)
                    || (provider.as_str() == WireFormat::AnthropicMessages.as_str()
                        && super::anthropic::is_provider_tool_block(raw)))
        }
        ContentBlock::ToolResult(result) => result
            .content
            .iter()
            .any(|block| is_unsupported_tool_content(block, target)),
        _ => false,
    }
}
fn drop_unsupported_tool_content(content: &mut Vec<ContentBlock>, target: WireFormat) {
    content.retain_mut(|block| {
        match block {
            ContentBlock::ToolResult(result) | ContentBlock::Mcp(McpContent::Result(result)) => {
                drop_unsupported_tool_content(&mut result.content, target)
            }
            _ => {}
        }
        !is_unsupported_tool_content(block, target)
    });
}

pub(crate) fn prepare_response_tools<'a>(
    response: &'a AggLlmResponse,
    target: WireFormat,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Cow<'a, AggLlmResponse>> {
    let mut unsupported = false;
    let mut has_mcp = false;
    for block in response.outputs.iter().flat_map(|output| &output.content) {
        if is_mcp_content(block) {
            has_mcp = true;
            unsupported = unsupported || is_unsupported_tool_content(block, target);
            if unsupported {
                break;
            }
        }
    }
    has_mcp = has_mcp
        || response
            .preservation
            .responses
            .values()
            .any(preserved_response_has_mcp);
    if !unsupported && (!has_mcp || can_replay_mcp_response(response, target)) {
        return Ok(Cow::Borrowed(response));
    }
    if unsupported {
        crate::util::push_lossy(
            diagnostics,
            policy,
            "unmapped MCP or provider tool output is dropped",
        )?;
    }
    let mut response = response.clone();
    for output in &mut response.outputs {
        output
            .content
            .retain(|block| !(is_mcp_content(block) && is_unsupported_tool_content(block, target)));
    }
    response.preservation = PreservationMetadata::default();
    if let Some(metadata) = response
        .extensions
        .fields
        .get_mut("metadata")
        .and_then(Value::as_object_mut)
    {
        metadata.remove(crate::PRESERVATION_METADATA_KEY);
    }
    Ok(Cow::Owned(response))
}

// As with requests, checking one snapshot does not validate other cached bodies.
fn can_replay_mcp_response(response: &AggLlmResponse, target: WireFormat) -> bool {
    response.preservation.responses.len() == 1
        && response
            .preservation
            .responses
            .get(&target.into())
            .is_some_and(|body| {
                preservation_codec(target)
                    .decode_response(body, &preservation_check_policy())
                    .is_ok_and(|original| {
                        original.response.outputs == response.outputs
                            && original.response.usage == response.usage
                    })
            })
}

// Compare normalized content before permitting exact replay. This retains native
// fields on unchanged requests without restoring MCP data removed from the IR.
fn preservation_codec(format: WireFormat) -> &'static dyn super::FormatCodec {
    match format {
        WireFormat::AnthropicMessages => &super::anthropic::AnthropicMessagesCodec,
        WireFormat::OpenAiResponses => &super::responses::OpenAiResponsesCodec,
        WireFormat::OpenAiChat => &super::openai_chat::OpenAiChatCodec,
    }
}

fn preservation_check_policy() -> TranslationPolicy {
    TranslationPolicy {
        preservation: crate::PreservationPolicy::Disabled,
        ..TranslationPolicy::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn encrypted_reasoning_helpers_accept_both_detail_shapes() {
        let documented = vec![json!({"type": "reasoning.encrypted", "data": "blob", "id": "rs_1"})];
        assert_eq!(
            encrypted_reasoning_data(&documented).as_deref(),
            Some("blob")
        );
        assert_eq!(
            encrypted_reasoning_item_id(&documented).as_deref(),
            Some("rs_1")
        );
        let verbatim_item = vec![json!({
            "type": "reasoning", "id": "rs_2", "summary": [], "encrypted_content": "blob2"
        })];
        assert_eq!(
            encrypted_reasoning_data(&verbatim_item).as_deref(),
            Some("blob2")
        );
        assert_eq!(
            encrypted_reasoning_item_id(&verbatim_item).as_deref(),
            Some("rs_2")
        );
        assert_eq!(encrypted_reasoning_data(&[json!({"type": "other"})]), None);
    }

    #[test]
    fn mcp_request_replay_preserves_instruction_and_extension_edits() -> Result<()> {
        let engine = crate::TranslationEngine::default();
        let policy = TranslationPolicy::default();
        for (target, instruction_key, body) in [
            (
                WireFormat::OpenAiResponses,
                "instructions",
                json!({
                    "model": "route", "input": "Check inventory", "instructions": "Original",
                    "metadata": {"label": "original"},
                    "tools": [{"type": "mcp", "server_label": "inventory",
                        "server_url": "https://inventory.example/mcp", "require_approval": "never"}]
                }),
            ),
            (
                WireFormat::AnthropicMessages,
                "system",
                json!({
                    "model": "route", "max_tokens": 64, "system": "Original",
                    "metadata": {"label": "original"},
                    "messages": [{"role": "user", "content": "Check inventory"}],
                    "mcp_servers": [{"type": "url", "name": "inventory", "url": "https://inventory.example/mcp"}],
                    "tools": [{"type": "mcp_toolset", "mcp_server_name": "inventory"}]
                }),
            ),
        ] {
            let request = engine.decode_request(target, &body, &policy)?.request;
            assert_eq!(engine.encode_request(target, &request, &policy)?.body, body);
            for edit_instructions in [true, false] {
                let mut edited = request.clone();
                if edit_instructions {
                    edited.instructions[0].content = vec![ContentBlock::Text {
                        text: "Updated".into(),
                    }];
                } else {
                    edited
                        .extensions
                        .fields
                        .insert("metadata".into(), json!({"label": "updated"}));
                }
                let encoded = engine.encode_request(target, &edited, &policy)?.body;
                if edit_instructions {
                    assert_eq!(encoded[instruction_key], "Updated", "{target}");
                } else {
                    assert_eq!(encoded["metadata"]["label"], "updated", "{target}");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn mcp_response_replay_preserves_usage_edits() -> Result<()> {
        let engine = crate::TranslationEngine::default();
        let policy = TranslationPolicy::default();
        for (target, body) in [
            (
                WireFormat::OpenAiResponses,
                json!({
                    "id": "resp_1", "model": "route", "status": "completed",
                    "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                    "output": [{"type": "mcp_call", "id": "call_1", "server_label": "inventory",
                        "name": "lookup", "arguments": "{}", "output": "12"}]
                }),
            ),
            (
                WireFormat::AnthropicMessages,
                json!({
                    "id": "msg_1", "model": "route", "type": "message", "role": "assistant",
                    "usage": {"input_tokens": 10, "output_tokens": 2}, "stop_reason": "end_turn",
                    "content": [{"type": "mcp_tool_use", "id": "call_1", "server_name": "inventory", "name": "lookup", "input": {}},
                        {"type": "mcp_tool_result", "tool_use_id": "call_1", "content": "12"}]
                }),
            ),
        ] {
            let mut response = engine.decode_response(target, &body, &policy)?.response;
            assert_eq!(
                engine.encode_response(target, &response, &policy)?.body,
                body
            );
            response.usage.input_tokens = Some(20);
            let encoded = engine.encode_response(target, &response, &policy)?.body;
            assert_eq!(encoded["usage"]["input_tokens"], 20, "{target}");
        }
        Ok(())
    }

    #[test]
    fn mcp_replay_requires_a_single_unchanged_snapshot() -> Result<()> {
        let engine = crate::TranslationEngine::default();
        let target = WireFormat::OpenAiResponses;
        let other = WireFormat::AnthropicMessages;
        let mut diagnostics = Vec::new();
        for preservation in [
            crate::PreservationPolicy::InMemory,
            crate::PreservationPolicy::Embed,
        ] {
            let policy = TranslationPolicy {
                preservation,
                ..TranslationPolicy::default()
            };
            let mut request = engine.decode_request(target, &json!({
                "model": "route", "input": "Check inventory", "tools": [{
                    "type": "mcp", "server_label": "inventory", "server_url": "https://inventory.example/mcp", "require_approval": "never"
                }]
            }), &policy)?.request;
            let mut response = engine
                .decode_response(
                    target,
                    &json!({
                        "id": "resp_1", "model": "route", "status": "completed", "output": [{
                            "type": "mcp_call", "id": "call_1", "server_label": "inventory",
                            "name": "lookup", "arguments": "{}", "output": "12"
                        }]
                    }),
                    &policy,
                )?
                .response;
            assert!(matches!(
                prepare_request_tools(&request, target, &mut diagnostics, &policy)?,
                Cow::Borrowed(_)
            ));
            assert!(matches!(
                prepare_response_tools(&response, target, &mut diagnostics, &policy)?,
                Cow::Borrowed(_)
            ));
            request.preservation.requests.insert(
                other.into(),
                engine
                    .encode_request(other, &request, &preservation_check_policy())?
                    .body,
            );
            response.preservation.responses.insert(
                other.into(),
                engine
                    .encode_response(other, &response, &preservation_check_policy())?
                    .body,
            );
            let prepared = prepare_request_tools(&request, target, &mut diagnostics, &policy)?;
            assert!(matches!(prepared, Cow::Owned(_)));
            assert!(prepared.preservation.requests.is_empty());
            assert_eq!(prepared.tools, request.tools);
            assert_eq!(prepared.messages, request.messages);
            let prepared = prepare_response_tools(&response, target, &mut diagnostics, &policy)?;
            assert!(matches!(prepared, Cow::Owned(_)));
            assert!(prepared.preservation.responses.is_empty());
            assert_eq!(prepared.outputs, response.outputs);
        }
        assert!(diagnostics.is_empty());
        Ok(())
    }
}
