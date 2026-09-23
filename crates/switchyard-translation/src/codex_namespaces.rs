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

use std::collections::{HashMap, HashSet};

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

/// Reverse map from an upstream tool name to its Codex tool name and namespace.
///
/// The exact qualified name is always registered. A model often returns a near
/// miss instead, so two fallback spellings are registered when neither can be
/// confused with another tool:
///
/// * the qualified name without the `mcp__` prefix
/// * the bare tool name, only when exactly one namespace claims it
pub fn qualified_tool_origins(
    extensions: &ProviderExtensions,
) -> HashMap<String, (String, String)> {
    let Some(namespaces) = tool_namespaces(extensions) else {
        return HashMap::new();
    };
    let split: Vec<(String, String, String)> = namespaces
        .keys()
        .filter_map(|qualified| {
            let (tool, namespace) = split_qualified_name(namespaces, qualified)?;
            Some((qualified.clone(), tool.to_string(), namespace.to_string()))
        })
        .collect();

    // A bare name claimed by more than one namespace cannot be guessed, and a
    // fallback must never shadow a name that is itself a real qualified tool.
    let mut bare_claims: HashMap<&str, usize> = HashMap::new();
    for (_, tool, _) in &split {
        *bare_claims.entry(tool.as_str()).or_default() += 1;
    }
    let qualified_names: HashSet<&str> = split
        .iter()
        .map(|(qualified, _, _)| qualified.as_str())
        .collect();

    let mut origins = HashMap::new();
    for (qualified, tool, namespace) in &split {
        let origin = (tool.clone(), namespace.clone());
        if let Some(stripped) = namespace.strip_prefix("mcp__") {
            let spelling = qualified_tool_name(stripped, tool);
            if !qualified_names.contains(spelling.as_str()) {
                origins.entry(spelling).or_insert_with(|| origin.clone());
            }
        }
        if bare_claims.get(tool.as_str()) == Some(&1) && !qualified_names.contains(tool.as_str()) {
            origins
                .entry(tool.clone())
                .or_insert_with(|| origin.clone());
        }
        // The exact spelling always wins over a fallback.
        origins.insert(qualified.clone(), origin);
    }
    origins
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

    // A bare name claimed by two namespaces must not be guessed: a wrong guess
    // dispatches the call to the wrong server.
    #[test]
    fn leaves_an_ambiguous_bare_name_alone() {
        let origins =
            qualified_tool_origins(&extensions(&[("mcp__a", "search"), ("mcp__b", "search")]));
        let mut response = json!({"output": [{"type": "function_call", "name": "search"}]});
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
