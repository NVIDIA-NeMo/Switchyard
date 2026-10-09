// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Anthropic Messages buffered and streaming codecs.

mod buffered;
mod stream;

pub use buffered::AnthropicMessagesCodec;
pub use stream::AnthropicMessagesStreamCodec;

const ANTHROPIC_TOOLS_KEY: &str = "switchyard_anthropic_tools";

pub(crate) fn prepare_request_tools<'a>(
    request: &'a crate::LlmRequest,
    target: crate::WireFormat,
    diagnostics: &mut Vec<crate::TranslationDiagnostic>,
    policy: &crate::TranslationPolicy,
) -> crate::Result<std::borrow::Cow<'a, crate::LlmRequest>> {
    let has_provider_tool_fields = |body: &serde_json::Value| {
        body.get("mcp_servers")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|servers| !servers.is_empty())
            || body
                .get("tools")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|tools| tools.iter().any(is_server_tool))
    };
    let has_mcp_servers = super::common::is_anthropic_request(request)
        && request
            .extensions
            .fields
            .get("mcp_servers")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|servers| !servers.is_empty());
    let has_provider_tool_definitions = request
        .extensions
        .fields
        .get(ANTHROPIC_TOOLS_KEY)
        .and_then(serde_json::Value::as_array)
        .is_some_and(|tools| tools.iter().any(is_server_tool));
    let has_provider_tool_history = request
        .messages
        .iter()
        .flat_map(|message| &message.content)
        .chain(
            request
                .instructions
                .iter()
                .flat_map(|instruction| &instruction.content),
        )
        .any(is_provider_tool_content);
    // Embedded source bodies can carry credentials even when the normalized fields do not.
    let has_preserved_tool_fields = request
        .preservation
        .requests
        .get(&crate::WireFormat::AnthropicMessages.into())
        .is_some_and(has_provider_tool_fields);
    if has_mcp_servers
        || has_provider_tool_definitions
        || has_provider_tool_history
        || has_preserved_tool_fields
    {
        crate::util::push_lossy(
            diagnostics,
            policy,
            format!(
                "Anthropic MCP servers, server tools, and their history are dropped for {target}"
            ),
        )?;
        let mut request = request.clone();
        request.extensions.fields.remove("mcp_servers");
        request.extensions.fields.remove(ANTHROPIC_TOOLS_KEY);
        // Original bodies can restore dropped tools or embed MCP credentials.
        request.preservation = crate::PreservationMetadata::default();
        if let Some(metadata) = request
            .extensions
            .fields
            .get_mut("metadata")
            .and_then(serde_json::Value::as_object_mut)
        {
            metadata.remove(crate::PRESERVATION_METADATA_KEY);
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
            || matches!(&request.tool_choice, Some(crate::ToolChoice::Tool { name })
                if !request.tools.iter().any(|tool| &tool.name == name))
        {
            request.tool_choice = None;
        }
        return Ok(std::borrow::Cow::Owned(request));
    }
    Ok(std::borrow::Cow::Borrowed(request))
}

fn is_server_tool(tool: &serde_json::Value) -> bool {
    tool.get("type")
        .and_then(serde_json::Value::as_str)
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

fn is_provider_tool_block(block: &serde_json::Value) -> bool {
    block
        .get("type")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|kind| {
            matches!(kind, "mcp_tool_use" | "server_tool_use") || kind.ends_with("_tool_result")
        })
}

fn is_provider_tool_content(block: &crate::ContentBlock) -> bool {
    match block {
        crate::ContentBlock::Unknown { provider, raw } => {
            provider.as_str() == crate::WireFormat::AnthropicMessages.as_str()
                && is_provider_tool_block(raw)
        }
        crate::ContentBlock::ToolResult(result) => {
            result.content.iter().any(is_provider_tool_content)
        }
        _ => false,
    }
}

fn drop_provider_tool_content(content: &mut Vec<crate::ContentBlock>) {
    content.retain_mut(|block| {
        if let crate::ContentBlock::ToolResult(result) = block {
            drop_provider_tool_content(&mut result.content);
        }
        !is_provider_tool_content(block)
    });
}
