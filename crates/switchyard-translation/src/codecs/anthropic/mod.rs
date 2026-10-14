// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Anthropic Messages buffered and streaming codecs.

use serde_json::Value;

use crate::{LlmRequest, McpToolset};

mod buffered;
mod stream;

pub use buffered::AnthropicMessagesCodec;
pub use stream::AnthropicMessagesStreamCodec;

pub(crate) const MCP_LAYOUTS_KEY: &str = "switchyard_anthropic_mcp_layouts";

pub(crate) fn is_provider_tool_block(block: &Value) -> bool {
    block
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|kind| {
            matches!(
                kind,
                "mcp_tool_use" | "mcp_tool_listing" | "server_tool_use"
            ) || kind.ends_with("_tool_result")
        })
}

// Names identify servers; a unique endpoint can retain the layout after a rename.
// If both change, the codec cannot safely associate the entry with an original layout.
fn source_mcp_layout<'a>(mcp: &McpToolset, request: &'a LlmRequest) -> Option<&'a Value> {
    if !super::common::is_anthropic_request(request) {
        return None;
    }
    let layouts = request
        .extensions
        .fields
        .get(MCP_LAYOUTS_KEY)?
        .as_object()?;
    if let Some(layout) = layouts.get(&mcp.name) {
        return Some(layout);
    }
    let mut matches = layouts.values().filter(|layout| {
        layout.pointer("/server/url").and_then(Value::as_str) == Some(mcp.url.as_str())
    });
    let matched = matches.next()?;
    matches.next().is_none().then_some(matched)
}

pub(crate) fn can_encode_mcp(mcp: &McpToolset, request: &LlmRequest) -> bool {
    // Anthropic's MCP connector requires HTTPS; plain HTTP is not supported.
    // It also has no setting to require approval before tool calls.
    if !mcp.url.starts_with("https://") || mcp.require_approval == Some(true) {
        return false;
    }
    let Some(source) = source_mcp_layout(mcp, request) else {
        return !super::common::is_anthropic_request(request)
            && mcp.require_approval == Some(false);
    };
    let Some(server) = source.get("server").and_then(Value::as_object) else {
        return false;
    };
    let has_toolset = source.get("toolset").is_some_and(Value::is_object);
    source.get("server_index").and_then(Value::as_u64).is_some()
        && server.get("type").and_then(Value::as_str) == Some("url")
        && (has_toolset || mcp.tool_enabled.is_empty())
}

// Distinguishes tools we can normalize as functions from those that cannot be translated into other wire formats.
// Bash, text_editor, and computer are Anthropic provider tools, but they are client-executed,
// and can reasonably be translated into function tools in other wire formats.
pub(crate) fn is_function_tool(tool: &Value) -> bool {
    tool.get("type").and_then(Value::as_str).is_none_or(|kind| {
        kind == "custom"
            || ["bash_", "text_editor_", "computer_"]
                .iter()
                .any(|prefix| kind.starts_with(prefix))
    })
}
