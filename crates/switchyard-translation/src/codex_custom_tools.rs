// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Round-trips OpenAI Responses freeform ("custom") tools through the neutral IR.
//!
//! Codex drives GPT-5 models with freeform tools: the definition is `{"type": "custom", "name",
//! "description", "format"}` and the model answers with `custom_tool_call` items whose `input`
//! is a raw string rather than JSON arguments. The IR only knows function-style tools, so a
//! custom tool is represented as a function whose single argument is `input`, and the verbatim
//! definitions are kept on the request extensions. When the response is encoded back to
//! Responses, calls to those tools are rewritten into `custom_tool_call` items again.

use std::collections::HashSet;

use serde_json::{Map, Value, json};
use switchyard_protocol::ProviderExtensions;

/// Request-extension key holding the verbatim custom tool definitions, keyed by tool name.
///
/// Prefixed so it cannot collide with a real provider field, and so a codec that allowlists
/// provider fields never forwards it.
pub const CUSTOM_TOOLS_KEY: &str = "switchyard_codex_custom_tools";

/// Request-extension key holding the verbatim `tools` array of a Responses-lite
/// `additional_tools` input item, so the request can be re-emitted in the same shape.
///
/// Codex sends GPT-5 requests in a "lite" shape: no top-level `tools`, empty `instructions`,
/// and the tool definitions inside `input[0]` as `{"type": "additional_tools", "role":
/// "developer", "tools": [...]}`.
pub const ADDITIONAL_TOOLS_KEY: &str = "switchyard_codex_additional_tools";

/// Stores the verbatim tools array of an `additional_tools` input item.
pub fn attach_additional_tools(extensions: &mut ProviderExtensions, tools: Vec<Value>) {
    if !tools.is_empty() {
        extensions
            .fields
            .insert(ADDITIONAL_TOOLS_KEY.to_string(), Value::Array(tools));
    }
}

/// Reads the verbatim `additional_tools` array back off a request's extensions.
pub fn additional_tools(extensions: &ProviderExtensions) -> Option<&Vec<Value>> {
    extensions
        .fields
        .get(ADDITIONAL_TOOLS_KEY)
        .and_then(Value::as_array)
}

/// Argument name used to carry a custom tool's freeform input through the IR.
pub const INPUT_ARGUMENT: &str = "input";

/// The IR parameter schema for a custom tool: one required string, `input`.
pub fn input_schema() -> Value {
    json!({
        "type": "object",
        "properties": {INPUT_ARGUMENT: {"type": "string"}},
        "required": [INPUT_ARGUMENT],
        "additionalProperties": false,
    })
}

/// Stores the collected definitions on a request's extensions, when there are any.
pub fn attach_custom_tools(extensions: &mut ProviderExtensions, tools: Map<String, Value>) {
    if !tools.is_empty() {
        extensions
            .fields
            .insert(CUSTOM_TOOLS_KEY.to_string(), Value::Object(tools));
    }
}

/// Reads the definitions back off a request's extensions.
pub fn custom_tools(extensions: &ProviderExtensions) -> Option<&Map<String, Value>> {
    extensions
        .fields
        .get(CUSTOM_TOOLS_KEY)
        .and_then(Value::as_object)
}

/// Names of the custom tools recorded on a request.
pub fn custom_tool_names(extensions: &ProviderExtensions) -> HashSet<String> {
    custom_tools(extensions)
        .map(|tools| tools.keys().cloned().collect())
        .unwrap_or_default()
}

/// Extracts the freeform input from IR tool arguments, falling back to the serialized
/// arguments when the model did not use the `input` convention.
pub fn input_from_arguments(arguments: &Value) -> String {
    match arguments {
        Value::Object(object) => match object.get(INPUT_ARGUMENT) {
            Some(Value::String(input)) => input.clone(),
            Some(other) => other.to_string(),
            None => arguments.to_string(),
        },
        Value::String(text) => match serde_json::from_str::<Value>(text) {
            Ok(Value::Object(object)) => match object.get(INPUT_ARGUMENT) {
                Some(Value::String(input)) => input.clone(),
                _ => text.clone(),
            },
            _ => text.clone(),
        },
        other => other.to_string(),
    }
}

/// Rewrites a `function_call` output item into a `custom_tool_call` when the tool is custom.
/// Returns whether the item was rewritten.
fn rewrite_item(item: &mut Value, custom: &HashSet<String>) -> bool {
    let Some(object) = item.as_object_mut() else {
        return false;
    };
    if object.get("type").and_then(Value::as_str) != Some("function_call") {
        return false;
    }
    let Some(name) = object.get("name").and_then(Value::as_str) else {
        return false;
    };
    if !custom.contains(name) {
        return false;
    }
    let input = object
        .remove("arguments")
        .map(|arguments| input_from_arguments(&arguments))
        .unwrap_or_default();
    object.insert(
        "type".to_string(),
        Value::String("custom_tool_call".to_string()),
    );
    object.insert("input".to_string(), Value::String(input));
    // OpenAI validates replayed item ids by prefix: a custom tool call must be `ctc_...`.
    if let Some(Value::String(id)) = object.get_mut("id")
        && let Some(rest) = id.strip_prefix("fc_")
    {
        *id = format!("ctc_{rest}");
    }
    true
}

/// Rewrites custom tool calls inside a buffered Responses body's `output` array.
pub fn restore_custom_tool_calls(body: &mut Value, custom: &HashSet<String>) {
    if custom.is_empty() {
        return;
    }
    if let Some(items) = body.get_mut("output").and_then(Value::as_array_mut) {
        for item in items {
            rewrite_item(item, custom);
        }
    }
}

/// Per-stream bookkeeping for [`restore_custom_tool_calls_in_event`].
#[derive(Default)]
pub struct CustomToolCallStreamState {
    /// Output indexes whose item was rewritten into a custom tool call.
    custom_indexes: HashSet<u64>,
}

/// Rewrites a streamed Responses event so a custom tool's call reaches the client in the shape
/// it expects. Item events are rewritten in place; argument delta events for a rewritten item
/// are dropped (returns `false`), because a partial JSON delta cannot be turned into a
/// freeform input delta and clients read the completed item instead.
pub fn restore_custom_tool_calls_in_event(
    event: &mut Value,
    custom: &HashSet<String>,
    state: &mut CustomToolCallStreamState,
) -> bool {
    if custom.is_empty() {
        return true;
    }
    let kind = event
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let index = event.get("output_index").and_then(Value::as_u64);
    match kind.as_str() {
        "response.output_item.added" | "response.output_item.done" => {
            if let Some(item) = event.get_mut("item")
                && rewrite_item(item, custom)
                && let Some(index) = index
            {
                state.custom_indexes.insert(index);
            }
            true
        }
        "response.function_call_arguments.delta" | "response.function_call_arguments.done" => {
            !index.is_some_and(|index| state.custom_indexes.contains(&index))
        }
        "response.completed" | "response.incomplete" | "response.failed" => {
            if let Some(response) = event.get_mut("response") {
                restore_custom_tool_calls(response, custom);
            }
            true
        }
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_is_read_from_the_input_argument_or_left_verbatim() {
        assert_eq!(input_from_arguments(&json!({"input": "ls -la"})), "ls -la");
        assert_eq!(input_from_arguments(&json!("{\"input\":\"pwd\"}")), "pwd");
        assert_eq!(input_from_arguments(&json!("raw text")), "raw text");
        assert_eq!(
            input_from_arguments(&json!({"cmd": "x"})),
            "{\"cmd\":\"x\"}"
        );
    }

    #[test]
    fn function_call_items_for_custom_tools_become_custom_tool_calls() {
        let custom: HashSet<String> = ["exec".to_string()].into_iter().collect();
        let mut body = json!({"output": [
            {"type": "function_call", "call_id": "c1", "name": "exec", "arguments": "{\"input\":\"ls\"}"},
            {"type": "function_call", "call_id": "c2", "name": "update_plan", "arguments": "{}"}
        ]});
        restore_custom_tool_calls(&mut body, &custom);
        assert_eq!(body["output"][0]["type"], "custom_tool_call");
        assert_eq!(body["output"][0]["input"], "ls");
        assert!(body["output"][0].get("arguments").is_none());
        assert_eq!(body["output"][1]["type"], "function_call");
    }

    #[test]
    fn rewritten_custom_tool_calls_take_the_ctc_id_prefix() {
        let custom: HashSet<String> = ["exec".to_string()].into_iter().collect();
        let mut body = json!({"output": [
            {"type": "function_call", "id": "fc_abc_1", "call_id": "c1", "name": "exec", "arguments": "{\"input\":\"ls\"}"}
        ]});
        restore_custom_tool_calls(&mut body, &custom);
        assert_eq!(body["output"][0]["type"], "custom_tool_call");
        assert_eq!(body["output"][0]["id"], "ctc_abc_1");
    }

    #[test]
    fn streamed_argument_deltas_for_custom_tools_are_dropped() {
        let custom: HashSet<String> = ["exec".to_string()].into_iter().collect();
        let mut state = CustomToolCallStreamState::default();
        let mut added = json!({"type": "response.output_item.added", "output_index": 1,
            "item": {"type": "function_call", "call_id": "c1", "name": "exec", "arguments": ""}});
        assert!(restore_custom_tool_calls_in_event(
            &mut added, &custom, &mut state
        ));
        assert_eq!(added["item"]["type"], "custom_tool_call");
        let mut delta = json!({"type": "response.function_call_arguments.delta", "output_index": 1, "delta": "{\"in"});
        assert!(!restore_custom_tool_calls_in_event(
            &mut delta, &custom, &mut state
        ));
        let mut other = json!({"type": "response.function_call_arguments.delta", "output_index": 2, "delta": "{}"});
        assert!(restore_custom_tool_calls_in_event(
            &mut other, &custom, &mut state
        ));
        let mut done = json!({"type": "response.output_item.done", "output_index": 1,
            "item": {"type": "function_call", "call_id": "c1", "name": "exec", "arguments": "{\"input\":\"ls -la\"}"}});
        assert!(restore_custom_tool_calls_in_event(
            &mut done, &custom, &mut state
        ));
        assert_eq!(done["item"]["input"], "ls -la");
    }
}
