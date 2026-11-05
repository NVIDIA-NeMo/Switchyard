// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module applies explicitly configured Responses tool adaptations.

use serde::Deserialize;
use serde_json::{Map, Value};
use std::collections::HashMap;

/// This policy selects the Responses request changes required by a configured provider.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesCompatibility {
    /// This mode leaves native item order, IDs, fields, and tool namespaces unchanged.
    #[default]
    Native,
    /// This mode joins function namespaces and names with `__`.
    FlatTools,
    /// This mode flattens function namespaces, pairs outputs with calls, and adds missing names.
    Kimi,
}

impl ResponsesCompatibility {
    pub(crate) fn normalize(self, body: &mut Value) {
        if self == Self::Native {
            return;
        }
        sanitize_openai_responses_provider_body(body);
        if self == Self::Kimi {
            pair_kimi_tool_outputs(body);
        }
    }
}

// Kimi's translator needs output adjacency and names in addition to `call_id`.
// Pairing stops at intervening messages, and existing output names remain unchanged.
fn pair_kimi_tool_outputs(body: &mut Value) {
    let Some(Value::Array(items)) = body.get_mut("input") else {
        return;
    };
    let names: HashMap<String, Value> = items
        .iter()
        .filter(|item| {
            matches!(
                item.get("type").and_then(Value::as_str),
                Some("function_call" | "custom_tool_call")
            )
        })
        .filter_map(|item| {
            Some((
                item.get("call_id")?.as_str()?.to_string(),
                item.get("name")?.clone(),
            ))
        })
        .collect();
    for item in items.iter_mut() {
        if matches!(
            item.get("type").and_then(Value::as_str),
            Some("function_call_output" | "custom_tool_call_output")
        ) && let Some(name) = item
            .get("call_id")
            .and_then(Value::as_str)
            .and_then(|id| names.get(id))
            && let Some(item) = item.as_object_mut()
        {
            item.entry("name".to_string())
                .or_insert_with(|| name.clone());
        }
    }
    let mut index = 0;
    while index < items.len() {
        let output_kind = match items[index].get("type").and_then(Value::as_str) {
            Some("function_call") => "function_call_output",
            Some("custom_tool_call") => "custom_tool_call_output",
            _ => {
                index += 1;
                continue;
            }
        };
        let Some(call_id) = items[index].get("call_id").and_then(Value::as_str) else {
            index += 1;
            continue;
        };
        let output = items
            .iter()
            .enumerate()
            .skip(index + 1)
            .take_while(|(_, item)| {
                item.get("role").is_none()
                    && item.get("type").and_then(Value::as_str) != Some("message")
            })
            .find(|(_, item)| {
                item.get("type").and_then(Value::as_str) == Some(output_kind)
                    && item.get("call_id").and_then(Value::as_str) == Some(call_id)
            })
            .map(|(position, _)| position);
        if let Some(output) = output {
            let item = items.remove(output);
            items.insert(index + 1, item);
        }
        index += 1;
    }
}

const CODEX_NAMESPACE_SEPARATOR: &str = "__";

// The `flat_tools` and `kimi` modes flatten function namespaces in history, tools, and tool choice.
// These modes also supply empty descriptions when function definitions lack string descriptions.
fn sanitize_openai_responses_provider_body(body: &mut Value) {
    let Value::Object(object) = body else {
        return;
    };
    sanitize_openai_responses_input_for_provider(object.get_mut("input"));
    sanitize_openai_responses_tools_for_provider(object.get_mut("tools"));
    sanitize_openai_responses_tool_choice_for_provider(object.get_mut("tool_choice"));
}

fn sanitize_openai_responses_input_for_provider(input: Option<&mut Value>) {
    let Some(Value::Array(items)) = input else {
        return;
    };
    for item in items {
        let Some(object) = item.as_object_mut() else {
            continue;
        };
        if object.get("type").and_then(Value::as_str) == Some("function_call") {
            qualify_responses_function_name(object);
        }
    }
}

fn sanitize_openai_responses_tools_for_provider(tools: Option<&mut Value>) {
    let Some(Value::Array(tools)) = tools else {
        return;
    };
    let mut flat_tools = Vec::with_capacity(tools.len());
    for tool in std::mem::take(tools) {
        push_sanitized_openai_responses_tool(&mut flat_tools, tool);
    }
    *tools = flat_tools;
}

fn push_sanitized_openai_responses_tool(out: &mut Vec<Value>, tool: Value) {
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
            push_sanitized_namespaced_tool(out, &namespace, child);
        }
        return;
    }
    ensure_responses_function_tool_description(&mut object);
    out.push(Value::Object(object));
}

fn push_sanitized_namespaced_tool(out: &mut Vec<Value>, namespace: &str, tool: Value) {
    let Value::Object(mut object) = tool else {
        out.push(tool);
        return;
    };
    if object.get("type").and_then(Value::as_str) == Some("function") {
        qualify_responses_function_name_with_namespace(&mut object, namespace);
        ensure_responses_function_tool_description(&mut object);
    }
    out.push(Value::Object(object));
}

fn sanitize_openai_responses_tool_choice_for_provider(tool_choice: Option<&mut Value>) {
    let Some(Value::Object(object)) = tool_choice else {
        return;
    };
    if object.get("type").and_then(Value::as_str) == Some("function") {
        qualify_responses_function_name(object);
    }
}

fn qualify_responses_function_name(object: &mut Map<String, Value>) {
    let namespace = object
        .remove("namespace")
        .and_then(|value| value.as_str().map(ToOwned::to_owned));
    let Some(namespace) = namespace.as_deref() else {
        return;
    };
    qualify_responses_function_name_with_namespace(object, namespace);
}

fn qualify_responses_function_name_with_namespace(
    object: &mut Map<String, Value>,
    namespace: &str,
) {
    if namespace.is_empty() {
        return;
    }
    let Some(name) = object.get("name").and_then(Value::as_str) else {
        return;
    };
    let prefix = format!("{namespace}{CODEX_NAMESPACE_SEPARATOR}");
    if name.starts_with(&prefix) {
        return;
    }
    object.insert("name".to_string(), Value::String(format!("{prefix}{name}")));
}

fn ensure_responses_function_tool_description(object: &mut Map<String, Value>) {
    if object.get("type").and_then(Value::as_str) == Some("function")
        && !matches!(object.get("description"), Some(Value::String(_)))
    {
        object.insert("description".to_string(), Value::String(String::new()));
    }
}
