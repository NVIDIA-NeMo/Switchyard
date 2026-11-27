// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Codex tool namespace preservation across a Responses/Chat translation.
//!
//! Codex groups tools into non-standard ``namespace`` containers — MCP servers,
//! usually behind an `mcp__` prefix, plus builtin groups such as
//! `multi_agent_v1` — and dispatches on the pair of tool name and namespace. It
//! therefore expects the namespace back on the call it receives:
//! `{"type": "function_call", "name": "search", "namespace": "mcp__docs"}`.
//!
//! OpenAI-compatible upstreams accept only flat `function` tools, so the request
//! codec flattens the containers and names each child `<namespace>__<tool>`.
//! Two tools differing only by namespace therefore stay distinct upstream, and
//! the conversation history uses the same spelling so the transcript never
//! teaches the model a name the upstream was not offered.
//!
//! The qualified-name to namespace mapping rides in the request's
//! [`ProviderExtensions`], so no provider-neutral type grows a Codex-specific
//! field. No codec copies unknown extension keys into an outbound body, so the
//! mapping reaches neither the upstream nor the client.
//!
//! No container is filtered by its name, because Codex resolves a call with no
//! namespace against its default group: dropping a builtin group's namespace
//! would look the call up in the wrong place.

use std::collections::HashMap;

use serde_json::{Map, Value};
use switchyard_protocol::ProviderExtensions;

pub use switchyard_protocol::codex_namespaces::{
    NAMESPACE_SEPARATOR, TOOL_NAMESPACES_KEY, split_qualified_name, tool_namespaces,
};

/// Joins a namespace and a tool name into the name used on the wire.
pub fn qualified_tool_name(namespace: &str, tool: &str) -> String {
    format!("{namespace}{NAMESPACE_SEPARATOR}{tool}")
}

/// Records that `qualified` was produced by flattening a tool out of `namespace`.
pub fn record_tool_namespace(
    namespaces: &mut Map<String, Value>,
    qualified: &str,
    namespace: &str,
    description: Option<&str>,
) {
    let value = description.map_or_else(
        || Value::String(namespace.to_string()),
        |description| {
            serde_json::json!({
                "namespace": namespace,
                "description": description,
            })
        },
    );
    namespaces.insert(qualified.to_string(), value);
}

/// Stores a collected mapping on a request's extensions, when it has entries.
pub fn attach_tool_namespaces(extensions: &mut ProviderExtensions, namespaces: Map<String, Value>) {
    if !namespaces.is_empty() {
        extensions
            .fields
            .insert(TOOL_NAMESPACES_KEY.to_string(), Value::Object(namespaces));
    }
}

/// Returns a retained description for `namespace`.
pub fn namespace_description<'a>(
    namespaces: &'a Map<String, Value>,
    namespace: &str,
) -> Option<&'a str> {
    namespaces.values().find_map(|value| {
        let object = value.as_object()?;
        (object.get("namespace").and_then(Value::as_str) == Some(namespace))
            .then(|| object.get("description").and_then(Value::as_str))
            .flatten()
    })
}

/// Maps exact qualified tool names to their Codex tool names and namespaces.
pub fn qualified_tool_origins(
    extensions: &ProviderExtensions,
) -> HashMap<String, (String, String)> {
    let Some(namespaces) = tool_namespaces(extensions) else {
        return HashMap::new();
    };
    namespaces
        .keys()
        .filter_map(|qualified| {
            let (tool, namespace) = split_qualified_name(namespaces, qualified)?;
            Some((qualified.clone(), (tool.to_string(), namespace.to_string())))
        })
        .collect()
}

/// Restore the Codex tool name and namespace on function calls and argument completions.
///
/// Walks the whole value, covering `function_call` items in buffered responses
/// and streaming events, plus the top-level name in
/// `response.function_call_arguments.done`. Preserves an existing `namespace`.
pub fn restore_qualified_tool_names(body: &mut Value, origins: &HashMap<String, (String, String)>) {
    if origins.is_empty() {
        return;
    }
    match body {
        Value::Array(values) => {
            for value in values {
                restore_qualified_tool_names(value, origins);
            }
        }
        Value::Object(object) => {
            if matches!(
                object.get("type").and_then(Value::as_str),
                Some("function_call" | "response.function_call_arguments.done")
            ) && let Some(name) = object.get("name").and_then(Value::as_str)
                && let Some((tool, namespace)) = origins.get(name)
            {
                object.insert("name".to_string(), Value::String(tool.clone()));
                object
                    .entry("namespace".to_string())
                    .or_insert_with(|| Value::String(namespace.clone()));
            }
            for value in object.values_mut() {
                restore_qualified_tool_names(value, origins);
            }
        }
        _ => {}
    }
}

/// Flattens Codex namespaces in an outbound Responses body for an OpenAI-compatible provider.
///
/// The Responses encoder rebuilds `namespace` containers and `namespace` fields on
/// calls for the client, and a replayed preserved body still holds them as sent.
/// Providers accept only flat function tools, so this pass runs on every body sent
/// to a Responses upstream. It names each child `<namespace>__<tool>`, moves the
/// `namespace` field of `function_call` items and a `tool_choice` into the name,
/// and gives each function tool a string `description`.
pub fn flatten_codex_namespaces_for_provider(body: &mut Value) {
    let Value::Object(object) = body else {
        return;
    };
    if let Some(Value::Array(items)) = object.get_mut("input") {
        for item in items.iter_mut().filter_map(Value::as_object_mut) {
            if item.get("type").and_then(Value::as_str) == Some("function_call") {
                move_namespace_into_name(item);
            }
        }
    }
    if let Some(Value::Array(tools)) = object.get_mut("tools") {
        let mut flat = Vec::with_capacity(tools.len());
        for tool in std::mem::take(tools) {
            push_flat_tool(&mut flat, tool);
        }
        *tools = flat;
    }
    if let Some(Value::Object(choice)) = object.get_mut("tool_choice")
        && choice.get("type").and_then(Value::as_str) == Some("function")
    {
        move_namespace_into_name(choice);
    }
}

fn push_flat_tool(out: &mut Vec<Value>, tool: Value) {
    let Value::Object(mut object) = tool else {
        out.push(tool);
        return;
    };
    if object.get("type").and_then(Value::as_str) == Some("namespace") {
        let namespace = object
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let Some(Value::Array(children)) = object.remove("tools") else {
            out.push(Value::Object(object));
            return;
        };
        for child in children {
            let Value::Object(mut child) = child else {
                out.push(child);
                continue;
            };
            if child.get("type").and_then(Value::as_str) == Some("function") {
                qualify_name(&mut child, &namespace);
                ensure_function_description(&mut child);
            }
            out.push(Value::Object(child));
        }
        return;
    }
    ensure_function_description(&mut object);
    out.push(Value::Object(object));
}

fn move_namespace_into_name(object: &mut Map<String, Value>) {
    if let Some(Value::String(namespace)) = object.remove("namespace") {
        qualify_name(object, &namespace);
    }
}

// Leaves a name that already carries the namespace prefix unchanged.
fn qualify_name(object: &mut Map<String, Value>, namespace: &str) {
    if namespace.is_empty() {
        return;
    }
    let Some(name) = object.get("name").and_then(Value::as_str) else {
        return;
    };
    if name
        .strip_prefix(namespace)
        .is_some_and(|rest| rest.starts_with(NAMESPACE_SEPARATOR))
    {
        return;
    }
    let qualified = qualified_tool_name(namespace, name);
    object.insert("name".to_string(), Value::String(qualified));
}

fn ensure_function_description(object: &mut Map<String, Value>) {
    if object.get("type").and_then(Value::as_str) == Some("function")
        && !matches!(object.get("description"), Some(Value::String(_)))
    {
        object.insert("description".to_string(), Value::String(String::new()));
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Map, json};
    use switchyard_protocol::ProviderExtensions;

    use super::{
        attach_tool_namespaces, qualified_tool_name, qualified_tool_origins, record_tool_namespace,
        restore_qualified_tool_names, split_qualified_name, tool_namespaces,
    };

    fn extensions(pairs: &[(&str, &str)]) -> ProviderExtensions {
        let mut namespaces = Map::new();
        for (namespace, tool) in pairs {
            record_tool_namespace(
                &mut namespaces,
                &qualified_tool_name(namespace, tool),
                namespace,
                None,
            );
        }
        let mut extensions = ProviderExtensions::default();
        attach_tool_namespaces(&mut extensions, namespaces);
        extensions
    }

    // A tool name may itself contain the separator, so the namespace is matched
    // as a prefix rather than by splitting on the first `__`.
    #[test]
    fn splits_a_qualified_name_back_apart() {
        let simple = extensions(&[("mcp__docs", "search")]);
        let simple = tool_namespaces(&simple).expect("mapping present");
        assert_eq!(
            split_qualified_name(simple, "mcp__docs__search"),
            Some(("search", "mcp__docs"))
        );
        assert_eq!(split_qualified_name(simple, "unknown_tool"), None);

        let nested = extensions(&[("mcp__docs", "fetch__raw")]);
        let nested = tool_namespaces(&nested).expect("mapping present");
        assert_eq!(
            split_qualified_name(nested, "mcp__docs__fetch__raw"),
            Some(("fetch__raw", "mcp__docs"))
        );
    }

    #[test]
    fn leaves_unqualified_names_alone() {
        let origins = qualified_tool_origins(&extensions(&[("mcp__docs", "search")]));
        let mut response = json!([
            {"output": [{"type": "function_call", "name": "search"}]},
            {"type": "response.output_item.done", "item": {
                "type": "function_call", "name": "docs__search"
            }},
            {"type": "response.function_call_arguments.done", "name": "docs__search"}
        ]);
        let before = response.clone();

        restore_qualified_tool_names(&mut response, &origins);

        assert_eq!(response, before);
    }

    // Codex namespaces builtin groups too, so nothing may key on `mcp__`.
    #[test]
    fn resolves_namespaces_that_are_not_mcp_servers() {
        let origins = qualified_tool_origins(&extensions(&[("multi_agent_v1", "spawn_agent")]));
        let mut response = json!({
            "output": [{"type": "function_call", "name": "multi_agent_v1__spawn_agent"}]
        });

        restore_qualified_tool_names(&mut response, &origins);

        assert_eq!(response["output"][0]["name"], "spawn_agent");
        assert_eq!(response["output"][0]["namespace"], "multi_agent_v1");
    }
}
