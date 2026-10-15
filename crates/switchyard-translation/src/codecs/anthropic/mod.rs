// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Anthropic Messages buffered and streaming codecs.

use serde_json::Value;

use crate::{LlmRequest, McpToolset, Tool};

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

// A rename can reuse a layout only when one source name disappeared and its URL
// still uniquely matches. This prevents borrowing another server's retained fields.
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
    let mut unclaimed = layouts.iter().filter(|(name, _)| {
        !request
            .tools
            .iter()
            .any(|tool| matches!(tool, Tool::Mcp(current) if &current.name == *name))
    });
    let (source_name, _) = unclaimed.next()?;
    if unclaimed.next().is_some() {
        return None;
    }
    let mut matches = layouts.values().filter(|layout| {
        layout.pointer("/server/url").and_then(Value::as_str) == Some(mcp.url.as_str())
    });
    let matched = matches.next()?;
    (matches.next().is_none() && layouts.get(source_name) == Some(matched)).then_some(matched)
}

pub(crate) fn can_encode_mcp(mcp: &McpToolset, request: &LlmRequest) -> bool {
    validate_mcp(mcp, request).is_ok()
}

// Both preparation and encoding use these checks so rejected tools and their
// history are removed together, before a cached body can restore them.
fn validate_mcp<'a>(
    mcp: &McpToolset,
    request: &'a LlmRequest,
) -> Result<Option<&'a Value>, &'static str> {
    // Anthropic requires HTTPS and has no MCP approval setting.
    if !mcp.url.starts_with("https://") || mcp.require_approval == Some(true) {
        return Err("MCP settings cannot be expressed by Anthropic");
    }
    let Some(source) = source_mcp_layout(mcp, request) else {
        return if !super::common::is_anthropic_request(request)
            && mcp.require_approval == Some(false)
        {
            Ok(None)
        } else {
            Err("MCP encoding requires an unambiguous Anthropic source layout")
        };
    };
    let valid_server = source
        .get("server")
        .and_then(Value::as_object)
        .is_some_and(|server| server.get("type").and_then(Value::as_str) == Some("url"));
    if !valid_server
        || source
            .get("server_index")
            .and_then(Value::as_u64)
            .and_then(|index| usize::try_from(index).ok())
            .is_none()
        || !source.get("toolset").is_some_and(Value::is_object)
    {
        return Err("invalid Anthropic MCP source layout");
    }
    validate_mcp_configs(&source["toolset"])?;
    Ok(Some(source))
}

fn validate_mcp_configs(toolset: &Value) -> Result<(), &'static str> {
    let valid_config = |config: &Value| {
        config.is_object()
            && ["enabled", "defer_loading"]
                .iter()
                .all(|key| config.get(key).is_none_or(Value::is_boolean))
    };
    if toolset
        .get("default_config")
        .is_some_and(|config| !valid_config(config))
    {
        return Err("MCP default_config requires an object with boolean settings");
    }
    if toolset.get("configs").is_some_and(|configs| {
        configs
            .as_object()
            .is_none_or(|configs| configs.values().any(|config| !valid_config(config)))
    }) {
        return Err("MCP configs requires objects with boolean settings");
    }
    Ok(())
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
