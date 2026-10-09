// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Anthropic Messages buffered and streaming codecs.

use std::borrow::Cow;

use serde_json::Value;

use crate::codecs::common::is_anthropic_request;
use crate::util::push_lossy;
use crate::{
    ContentBlock, LlmRequest, PRESERVATION_METADATA_KEY, PreservationMetadata, Result, ToolChoice,
    TranslationDiagnostic, TranslationPolicy, WireFormat,
};

mod buffered;
mod stream;

pub use buffered::AnthropicMessagesCodec;
pub use stream::AnthropicMessagesStreamCodec;

const ANTHROPIC_TOOLS_KEY: &str = "switchyard_anthropic_tools";

pub(crate) fn prepare_request_tools<'a>(
    request: &'a LlmRequest,
    target: WireFormat,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Cow<'a, LlmRequest>> {
    // Check small definition fields before walking the conversation history.
    let has_provider_tools = (is_anthropic_request(request)
        && (request.extensions.fields.get("mcp_servers")
                .and_then(Value::as_array)
                .is_some_and(|servers| !servers.is_empty())
            || request.extensions.fields.get(ANTHROPIC_TOOLS_KEY)
                .and_then(Value::as_array)
                .is_some_and(|tools| tools.iter().any(is_server_tool))))
        // Preserved source bodies can carry credentials absent from normalized fields.
        || request.preservation.requests.get(&WireFormat::AnthropicMessages.into())
            .is_some_and(|body| {
                body.get("mcp_servers").and_then(Value::as_array)
                    .is_some_and(|servers| !servers.is_empty())
                    || body.get("tools").and_then(Value::as_array)
                        .is_some_and(|tools| tools.iter().any(is_server_tool))
            })
        || request.messages.iter().flat_map(|message| &message.content)
            .chain(request.instructions.iter().flat_map(|instruction| &instruction.content))
            .any(is_provider_tool_content);
    if has_provider_tools {
        push_lossy(
            diagnostics,
            policy,
            format!(
                "Anthropic MCP servers, server tools, and their history are dropped for {target}"
            ),
        )?;
        let mut request = request.clone();
        request.extensions.fields.remove("mcp_servers");
        request.extensions.fields.remove(ANTHROPIC_TOOLS_KEY);
        // Clear exact-replay bodies for every format and the embedded preservation envelope.
        // Any of them could restore dropped tools or forward MCP credentials.
        request.preservation = PreservationMetadata::default();
        if let Some(metadata) = request
            .extensions
            .fields
            .get_mut("metadata")
            .and_then(Value::as_object_mut)
        {
            metadata.remove(PRESERVATION_METADATA_KEY);
        }
        request.instructions.retain_mut(|instruction| {
            drop_provider_tool_content(&mut instruction.content);
            !instruction.content.is_empty()
        });
        request.messages.retain_mut(|message| {
            drop_provider_tool_content(&mut message.content);
            !message.content.is_empty()
        });
        if request.tools.is_empty()
            || matches!(&request.tool_choice, Some(ToolChoice::Tool { name })
                if !request.tools.iter().any(|tool| &tool.name == name))
        {
            request.tool_choice = None;
        }
        return Ok(Cow::Owned(request));
    }
    Ok(Cow::Borrowed(request))
}

fn is_server_tool(tool: &Value) -> bool {
    // Add new server-tool families here; typed client tools must keep their function conversion.
    tool.get("type")
        .and_then(Value::as_str)
        .is_some_and(|kind| {
            kind == "mcp_toolset"
                || [
                    "web_search_",
                    "web_fetch_",
                    "code_execution_",
                    "tool_search_tool_",
                    "advisor_",
                ]
                .iter()
                .any(|prefix| kind.starts_with(prefix))
        })
}

fn is_provider_tool_block(block: &Value) -> bool {
    block
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|kind| {
            matches!(kind, "mcp_tool_use" | "server_tool_use") || kind.ends_with("_tool_result")
        })
}

fn is_provider_tool_content(block: &ContentBlock) -> bool {
    match block {
        ContentBlock::Unknown { provider, raw } => {
            provider.as_str() == WireFormat::AnthropicMessages.as_str()
                && is_provider_tool_block(raw)
        }
        ContentBlock::ToolResult(result) => result.content.iter().any(is_provider_tool_content),
        _ => false,
    }
}

fn drop_provider_tool_content(content: &mut Vec<ContentBlock>) {
    content.retain_mut(|block| {
        if let ContentBlock::ToolResult(result) = block {
            drop_provider_tool_content(&mut result.content);
        }
        !is_provider_tool_content(block)
    });
}
