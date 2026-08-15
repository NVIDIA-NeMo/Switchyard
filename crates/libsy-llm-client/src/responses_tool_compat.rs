// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compatibility transforms for provider-hosted tools received through Responses.

use serde_json::{Map, Value};
use switchyard_protocol::{LlmRequest, WireFormat};

/// Replaces deferred Responses tool discovery with eagerly available definitions.
///
/// Providers without `tool_search` can still call the same client tools; they receive every
/// deferred definition up front and do not see the discovery-only replay records.
pub(crate) fn eager_load_responses_tool_search(body: &mut Value) {
    let Some(object) = body.as_object_mut() else {
        return;
    };
    let has_tool_search = object
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| tools.iter().any(is_tool_search));
    let has_tool_search_replay = object
        .get("input")
        .and_then(Value::as_array)
        .is_some_and(|input| input.iter().any(is_tool_search_replay));
    if !has_tool_search && !has_tool_search_replay {
        return;
    }

    let mut eager_tools = Vec::new();
    if let Some(tools) = object.get_mut("tools").and_then(Value::as_array_mut) {
        for tool in std::mem::take(tools) {
            push_eager_tool(tool, &mut eager_tools);
        }
    }

    if let Some(input) = object.get_mut("input").and_then(Value::as_array_mut) {
        let mut retained = Vec::with_capacity(input.len());
        for mut item in std::mem::take(input) {
            match item.get("type").and_then(Value::as_str) {
                Some("tool_search_call") => {}
                Some("tool_search_output") => {
                    if let Some(tools) = item.get_mut("tools").and_then(Value::as_array_mut) {
                        for tool in std::mem::take(tools) {
                            push_eager_tool(tool, &mut eager_tools);
                        }
                    }
                }
                _ => {
                    if item.get("type").and_then(Value::as_str) == Some("function_call")
                        && let Some(item) = item.as_object_mut()
                    {
                        item.remove("namespace");
                    }
                    retained.push(item);
                }
            }
        }
        *input = retained;
    }

    if !eager_tools.is_empty() {
        object.insert("tools".to_string(), Value::Array(eager_tools));
    } else {
        object.remove("tools");
    }
    if object
        .get("tool_choice")
        .and_then(Value::as_object)
        .and_then(|choice| choice.get("type"))
        .and_then(Value::as_str)
        == Some("tool_search")
    {
        object.insert("tool_choice".to_string(), Value::String("auto".to_string()));
    }
}

fn is_tool_search(value: &Value) -> bool {
    value.get("type").and_then(Value::as_str) == Some("tool_search")
}

fn is_tool_search_replay(value: &Value) -> bool {
    matches!(
        value.get("type").and_then(Value::as_str),
        Some("tool_search_call" | "tool_search_output")
    )
}

// Flattens namespaces because providers without deferred discovery do not accept that wrapper.
fn push_eager_tool(mut tool: Value, output: &mut Vec<Value>) {
    match tool.get("type").and_then(Value::as_str) {
        Some("tool_search") => return,
        Some("namespace") => {
            if let Some(tools) = tool.get_mut("tools").and_then(Value::as_array_mut) {
                for nested in std::mem::take(tools) {
                    push_eager_tool(nested, output);
                }
            }
            return;
        }
        _ => {}
    }
    strip_defer_loading(&mut tool);
    if !output.contains(&tool) {
        output.push(tool);
    }
}

fn strip_defer_loading(value: &mut Value) {
    match value {
        Value::Object(object) => {
            object.remove("defer_loading");
            for value in object.values_mut() {
                strip_defer_loading(value);
            }
        }
        Value::Array(values) => {
            for value in values {
                strip_defer_loading(value);
            }
        }
        _ => {}
    }
}

/// Normalizes OpenAI's Responses web-search descriptor for xAI-compatible backends.
///
/// Codex annotates `web_search` with OpenAI-only controls such as
/// `external_web_access` and `search_content_types`. xAI exposes the same hosted tool,
/// but rejects those fields. A false external-access flag removes the tool instead of
/// accidentally upgrading a disabled/cached request to live search.
pub(crate) fn normalize_xai_responses_web_search(body: &mut Value) {
    let Some(object) = body.as_object_mut() else {
        return;
    };
    let Some(tools) = object.get_mut("tools").and_then(Value::as_array_mut) else {
        return;
    };

    let mut normalized = Vec::with_capacity(tools.len());
    for tool in std::mem::take(tools) {
        if !matches!(
            tool.get("type").and_then(Value::as_str),
            Some("web_search" | "web_search_preview")
        ) {
            normalized.push(tool);
            continue;
        }
        if tool.get("external_web_access").and_then(Value::as_bool) == Some(false) {
            continue;
        }

        let mut search =
            Map::from_iter([("type".to_string(), Value::String("web_search".to_string()))]);
        if let Some(filters) = normalize_xai_search_filters(&tool) {
            search.insert("filters".to_string(), filters);
        }
        copy_search_field(
            &tool,
            &mut search,
            "enable_image_understanding",
            "enable_image_understanding",
        );
        copy_search_field(
            &tool,
            &mut search,
            "enable_image_search",
            "enable_image_search",
        );
        normalized.push(Value::Object(search));
    }

    if normalized.is_empty() {
        object.remove("tools");
        object.remove("tool_choice");
    } else {
        *object.get_mut("tools").expect("tools exists") = Value::Array(normalized);
    }
}

fn normalize_xai_search_filters(tool: &Value) -> Option<Value> {
    let source = tool.get("filters").unwrap_or(tool);
    let mut filters = Map::new();
    copy_search_field(source, &mut filters, "allowed_domains", "allowed_domains");
    copy_search_field(source, &mut filters, "excluded_domains", "excluded_domains");
    copy_search_field(source, &mut filters, "blocked_domains", "excluded_domains");
    (!filters.is_empty()).then_some(Value::Object(filters))
}

/// Returns the first Responses web-search definition preserved on the inbound request.
pub(crate) fn responses_web_search_tool(request: &LlmRequest) -> Option<Value> {
    let format = WireFormat::OpenAiResponses.into();
    request
        .preservation
        .requests
        .get(&format)
        .and_then(|body| body.get("tools"))
        .and_then(Value::as_array)
        .and_then(|tools| {
            tools.iter().find(|tool| {
                matches!(
                    tool.get("type").and_then(Value::as_str),
                    Some("web_search" | "web_search_preview")
                )
            })
        })
        .cloned()
}

/// Adds an Anthropic-native web-search definition translated from Responses.
pub(crate) fn bridge_responses_web_search_to_anthropic(body: &mut Value, source: &Value) {
    let Some(body) = body.as_object_mut() else {
        return;
    };
    let tools = body
        .entry("tools".to_string())
        .or_insert_with(|| Value::Array(Vec::new()));
    let Some(tools) = tools.as_array_mut() else {
        return;
    };
    if tools.iter().any(|tool| {
        tool.get("name").and_then(Value::as_str) == Some("web_search")
            || tool
                .get("type")
                .and_then(Value::as_str)
                .is_some_and(|kind| kind.starts_with("web_search_"))
    }) {
        return;
    }

    let mut translated = Map::from_iter([
        (
            "type".to_string(),
            Value::String("web_search_20250305".to_string()),
        ),
        ("name".to_string(), Value::String("web_search".to_string())),
    ]);
    copy_search_field(source, &mut translated, "max_uses", "max_uses");
    copy_search_field(source, &mut translated, "user_location", "user_location");
    copy_search_field(
        source,
        &mut translated,
        "allowed_domains",
        "allowed_domains",
    );
    copy_search_field(
        source,
        &mut translated,
        "blocked_domains",
        "blocked_domains",
    );
    if let Some(filters) = source.get("filters") {
        copy_search_field(
            filters,
            &mut translated,
            "allowed_domains",
            "allowed_domains",
        );
        copy_search_field(
            filters,
            &mut translated,
            "blocked_domains",
            "blocked_domains",
        );
        copy_search_field(
            filters,
            &mut translated,
            "excluded_domains",
            "blocked_domains",
        );
    }
    tools.push(Value::Object(translated));
}

fn copy_search_field(source: &Value, target: &mut Map<String, Value>, from: &str, to: &str) {
    if let Some(value) = source.get(from) {
        target.insert(to.to_string(), value.clone());
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    // Eager compatibility must retain callable and hosted tools while removing discovery records.
    #[test]
    fn eager_loading_flattens_tools_and_removes_search_replay() {
        let mut body = json!({
            "tools": [
                {
                    "type": "namespace",
                    "name": "repo",
                    "tools": [{
                        "type": "function",
                        "name": "read_file",
                        "defer_loading": true,
                        "parameters": {"type": "object"}
                    }]
                },
                {"type": "mcp", "server_label": "docs", "defer_loading": true},
                {"type": "web_search"},
                {"type": "tool_search"}
            ],
            "tool_choice": {"type": "tool_search"},
            "input": [
                {"role": "user", "content": "Continue."},
                {"type": "tool_search_call", "call_id": "search_1"},
                {
                    "type": "tool_search_output",
                    "call_id": "search_1",
                    "tools": [{
                        "type": "function",
                        "name": "write_file",
                        "defer_loading": true,
                        "parameters": {"type": "object"}
                    }]
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "read_file",
                    "namespace": "repo",
                    "arguments": "{}"
                }
            ]
        });

        eager_load_responses_tool_search(&mut body);

        assert_eq!(body["tool_choice"], "auto");
        assert_eq!(body["tools"].as_array().map(Vec::len), Some(4));
        assert!(body["tools"].as_array().is_some_and(|tools| {
            tools.iter().all(|tool| {
                !matches!(
                    tool.get("type").and_then(Value::as_str),
                    Some("namespace" | "tool_search")
                ) && tool.get("defer_loading").is_none()
            })
        }));
        assert!(
            body["input"]
                .as_array()
                .is_some_and(|input| { input.iter().all(|item| !is_tool_search_replay(item)) })
        );
        assert_eq!(body["input"][1].get("namespace"), None);
    }

    // Responses web search must become an Anthropic server tool, not a client function.
    #[test]
    fn web_search_maps_to_anthropic_server_tool() {
        let mut body = json!({"messages": [{"role": "user", "content": "Search."}]});
        let source = json!({
            "type": "web_search",
            "filters": {"allowed_domains": ["example.com"]},
            "user_location": {"type": "approximate", "country": "US"}
        });

        bridge_responses_web_search_to_anthropic(&mut body, &source);

        assert_eq!(body["tools"][0]["type"], "web_search_20250305");
        assert_eq!(body["tools"][0]["name"], "web_search");
        assert_eq!(body["tools"][0]["allowed_domains"], json!(["example.com"]));
        assert_eq!(body["tools"][0]["user_location"]["country"], "US");
    }

    #[test]
    fn xai_web_search_drops_openai_only_options_but_stays_live() {
        let mut body = json!({
            "tools": [
                {
                    "type": "web_search",
                    "external_web_access": true,
                    "search_content_types": ["text", "image"],
                    "search_context_size": "high",
                    "user_location": {"type": "approximate", "country": "US"},
                    "filters": {"allowed_domains": ["example.com"]}
                },
                {"type": "function", "name": "echo", "parameters": {"type": "object"}}
            ],
            "tool_choice": "auto"
        });

        normalize_xai_responses_web_search(&mut body);

        assert_eq!(
            body["tools"][0],
            json!({
                "type": "web_search",
                "filters": {"allowed_domains": ["example.com"]}
            })
        );
        assert_eq!(body["tools"][1]["name"], "echo");
        assert_eq!(body["tool_choice"], "auto");
    }

    #[test]
    fn xai_web_search_does_not_enable_disabled_external_access() {
        let mut body = json!({
            "tools": [{"type": "web_search", "external_web_access": false}],
            "tool_choice": "auto"
        });

        normalize_xai_responses_web_search(&mut body);

        assert_eq!(body.get("tools"), None);
        assert_eq!(body.get("tool_choice"), None);
    }
}
