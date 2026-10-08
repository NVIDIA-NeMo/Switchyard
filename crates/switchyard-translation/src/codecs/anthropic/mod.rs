// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Anthropic Messages buffered and streaming codecs.

mod buffered;
mod stream;

pub use buffered::AnthropicMessagesCodec;
pub use stream::AnthropicMessagesStreamCodec;

const ANTHROPIC_TOOLS_KEY: &str = "switchyard_anthropic_tools";

pub(crate) fn validate_request_tools(
    request: &crate::LlmRequest,
    target: crate::WireFormat,
) -> crate::Result<()> {
    let has_provider_tool_fields = |body: &serde_json::Value| {
        body.get("mcp_servers")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|servers| !servers.is_empty())
            || body
                .get("tools")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|tools| tools.iter().any(is_provider_tool))
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
        .is_some_and(|tools| tools.iter().any(is_provider_tool));
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
        return Err(crate::TranslationError::UnsupportedTranslation {
            from: crate::WireFormat::AnthropicMessages.into(),
            to: target.into(),
        });
    }
    Ok(())
}

fn is_provider_tool(tool: &serde_json::Value) -> bool {
    tool.get("type")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|kind| kind != "custom")
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
