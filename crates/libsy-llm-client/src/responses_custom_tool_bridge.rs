// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compatibility bridge for Responses providers that support function tools but not custom tools.

use std::collections::HashSet;

use futures_util::StreamExt;
use futures_util::future::ready;
use serde_json::{Map, Value, json};
use switchyard_protocol::{LlmRequest, LlmResponseStream, LlmResponseStreamEvent, WireFormat};

/// Returns custom tool names retained in the original Responses request.
pub(crate) fn responses_custom_tool_names(request: &LlmRequest) -> HashSet<String> {
    let format = WireFormat::OpenAiResponses.into();
    request
        .preservation
        .requests
        .get(&format)
        .and_then(|body| body.get("tools"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|tool| {
            (tool.get("type").and_then(Value::as_str) == Some("custom"))
                .then(|| tool.get("name").and_then(Value::as_str))
                .flatten()
                .filter(|name| !name.is_empty())
                .map(ToOwned::to_owned)
        })
        .collect()
}

/// Converts custom definitions and replay items to Responses function equivalents.
pub(crate) fn bridge_responses_custom_tool_request(body: &mut Value) {
    let Some(object) = body.as_object_mut() else {
        return;
    };
    let Some(tools) = object.get_mut("tools").and_then(Value::as_array_mut) else {
        return;
    };

    let mut names = HashSet::new();
    for tool in tools {
        let Some(custom) = tool.as_object() else {
            continue;
        };
        if custom.get("type").and_then(Value::as_str) != Some("custom") {
            continue;
        }
        let Some(name) = custom
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
        else {
            continue;
        };
        let name = name.to_string();
        let description = custom
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        *tool = json!({
            "type": "function",
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Raw input for the custom tool."
                    }
                },
                "required": ["input"],
                "additionalProperties": false
            }
        });
        names.insert(name);
    }
    if names.is_empty() {
        return;
    }

    if let Some(choice) = object.get_mut("tool_choice").and_then(Value::as_object_mut)
        && choice.get("type").and_then(Value::as_str) == Some("custom")
        && choice
            .get("name")
            .and_then(Value::as_str)
            .is_some_and(|name| names.contains(name))
    {
        choice.insert("type".to_string(), Value::String("function".to_string()));
    }

    let Some(input) = object.get_mut("input").and_then(Value::as_array_mut) else {
        return;
    };
    for item in input {
        bridge_responses_custom_tool_replay_item(item, &names);
    }
}

// Rewrites one prior custom call or result into an upstream function replay item.
fn bridge_responses_custom_tool_replay_item(item: &mut Value, names: &HashSet<String>) {
    let Some(object) = item.as_object() else {
        return;
    };
    match object.get("type").and_then(Value::as_str) {
        Some("custom_tool_call")
            if object
                .get("name")
                .and_then(Value::as_str)
                .is_some_and(|name| names.contains(name)) =>
        {
            let Some(name) = object.get("name").and_then(Value::as_str) else {
                return;
            };
            let mut replacement = Map::new();
            replacement.insert(
                "type".to_string(),
                Value::String("function_call".to_string()),
            );
            replacement.insert("name".to_string(), Value::String(name.to_string()));
            if let Some(call_id) = object.get("call_id").and_then(Value::as_str) {
                replacement.insert("call_id".to_string(), Value::String(call_id.to_string()));
            }
            if let Some(id) = object
                .get("id")
                .and_then(Value::as_str)
                .filter(|id| id.starts_with("fc_"))
            {
                replacement.insert("id".to_string(), Value::String(id.to_string()));
            }
            let input = object
                .get("input")
                .and_then(Value::as_str)
                .unwrap_or_default();
            replacement.insert(
                "arguments".to_string(),
                json!({"input": input}).to_string().into(),
            );
            *item = Value::Object(replacement);
        }
        Some("custom_tool_call_output") => {
            let mut replacement = Map::new();
            replacement.insert(
                "type".to_string(),
                Value::String("function_call_output".to_string()),
            );
            if let Some(call_id) = object.get("call_id").and_then(Value::as_str) {
                replacement.insert("call_id".to_string(), Value::String(call_id.to_string()));
            }
            if let Some(output) = object.get("output") {
                replacement.insert("output".to_string(), output.clone());
            }
            *item = Value::Object(replacement);
        }
        _ => {}
    }
}

/// Converts bridged function calls in a buffered Responses result back to custom calls.
pub(crate) fn bridge_responses_custom_tool_response(body: &mut Value, names: &HashSet<String>) {
    bridge_response_output(body.get_mut("output"), names);
    bridge_response_output(
        body.get_mut("response")
            .and_then(|value| value.get_mut("output")),
        names,
    );
}

// Converts function-call items in one Responses output array.
fn bridge_response_output(output: Option<&mut Value>, names: &HashSet<String>) {
    let Some(output) = output.and_then(Value::as_array_mut) else {
        return;
    };
    for item in output {
        bridge_function_call_item(item, names);
    }
}

// Converts one bridged function-call output item to the caller's custom-tool shape.
fn bridge_function_call_item(item: &mut Value, names: &HashSet<String>) -> bool {
    let Some(object) = item.as_object_mut() else {
        return false;
    };
    if object.get("type").and_then(Value::as_str) != Some("function_call")
        || !object
            .get("name")
            .and_then(Value::as_str)
            .is_some_and(|name| names.contains(name))
    {
        return false;
    }
    let input = custom_input_from_arguments(object.get("arguments"));
    object.insert(
        "type".to_string(),
        Value::String("custom_tool_call".to_string()),
    );
    object.insert("input".to_string(), Value::String(input));
    object.remove("arguments");
    object.remove("status");
    // `fc_...` identifies an upstream function-call item and is invalid for a
    // custom-tool item (`ctc...` on OpenAI). The ID is optional; `call_id` retains
    // the portable tool-call identity used to associate the result.
    object.remove("id");
    true
}

// Extracts the raw custom input from the JSON argument wrapper used upstream.
fn custom_input_from_arguments(arguments: Option<&Value>) -> String {
    let Some(arguments) = arguments else {
        return String::new();
    };
    match arguments {
        Value::String(raw) => serde_json::from_str::<Value>(raw)
            .ok()
            .and_then(|value| custom_input_from_value(&value))
            .unwrap_or_else(|| raw.clone()),
        value => custom_input_from_value(value).unwrap_or_else(|| value.to_string()),
    }
}

// Accepts the bridge's canonical key plus common coding-agent aliases.
fn custom_input_from_value(value: &Value) -> Option<String> {
    if let Some(value) = value.as_str() {
        return Some(value.to_string());
    }
    let object = value.as_object()?;
    for key in ["input", "arguments", "patch"] {
        if let Some(value) = object.get(key).and_then(Value::as_str) {
            return Some(value.to_string());
        }
    }
    if object.len() == 1 {
        return object
            .values()
            .next()
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
    }
    None
}

/// Rewrites preserved Responses stream events while retaining their normalized metadata.
pub(crate) fn bridge_responses_custom_tool_stream(
    stream: LlmResponseStream,
    wire_format: WireFormat,
    names: HashSet<String>,
) -> LlmResponseStream {
    if wire_format != WireFormat::OpenAiResponses || names.is_empty() {
        return stream;
    }
    let mut bridged_indices = HashSet::new();
    Box::pin(stream.filter_map(move |item| {
        let rewritten = match item {
            Err(error) => Some(Err(error)),
            Ok(event) => bridge_stream_event(event, &names, &mut bridged_indices).map(Ok),
        };
        ready(rewritten)
    }))
}

// Converts or suppresses one preserved upstream function-call stream event.
fn bridge_stream_event(
    event: LlmResponseStreamEvent,
    names: &HashSet<String>,
    bridged_indices: &mut HashSet<usize>,
) -> Option<LlmResponseStreamEvent> {
    let (preservation, normalized) = event.into_parts();
    let Some(preservation) = preservation else {
        return Some(LlmResponseStreamEvent::new(normalized));
    };
    let (source, mut raw) = preservation.into_parts();
    if source != WireFormat::OpenAiResponses.into() {
        return Some(LlmResponseStreamEvent::preserved(source, raw, normalized));
    }
    if !bridge_stream_raw_event(&mut raw, names, bridged_indices) {
        return None;
    }
    Some(LlmResponseStreamEvent::preserved(source, raw, normalized))
}

// Returns false for function argument deltas that cannot be exposed as raw custom input.
fn bridge_stream_raw_event(
    event: &mut Value,
    names: &HashSet<String>,
    bridged_indices: &mut HashSet<usize>,
) -> bool {
    match event.get("type").and_then(Value::as_str) {
        Some("response.output_item.added") | Some("response.output_item.done") => {
            let index = event
                .get("output_index")
                .and_then(Value::as_u64)
                .and_then(|index| usize::try_from(index).ok());
            if let Some(item) = event.get_mut("item")
                && bridge_function_call_item(item, names)
                && let Some(index) = index
            {
                bridged_indices.insert(index);
            }
        }
        Some("response.function_call_arguments.delta") => {
            return !stream_event_is_bridged(event, bridged_indices);
        }
        Some("response.function_call_arguments.done")
            if stream_event_is_bridged(event, bridged_indices) =>
        {
            let input = custom_input_from_arguments(event.get("arguments"));
            let Some(object) = event.as_object_mut() else {
                return true;
            };
            object.insert(
                "type".to_string(),
                Value::String("response.custom_tool_call_input.done".to_string()),
            );
            object.insert("input".to_string(), Value::String(input));
            object.remove("arguments");
        }
        Some("response.completed") | Some("response.incomplete") => {
            bridge_responses_custom_tool_response(event, names);
        }
        _ => {}
    }
    true
}

// Matches argument events to the custom call discovered in output_item.added.
fn stream_event_is_bridged(event: &Value, bridged_indices: &HashSet<usize>) -> bool {
    event
        .get("output_index")
        .and_then(Value::as_u64)
        .and_then(|index| usize::try_from(index).ok())
        .is_some_and(|index| bridged_indices.contains(&index))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Custom definitions and replay items must all use the same function-call protocol upstream.
    #[test]
    fn request_bridge_rewrites_custom_definitions_choice_and_replay() {
        let mut body = json!({
            "model": "grok",
            "tools": [{
                "type": "custom",
                "name": "apply_patch",
                "description": "Edit files",
                "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"}
            }],
            "tool_choice": {"type": "custom", "name": "apply_patch"},
            "input": [
                {
                    "type": "custom_tool_call",
                    "id": "ctc_1",
                    "call_id": "call_1",
                    "name": "apply_patch",
                    "input": "*** Begin Patch\n*** End Patch",
                    "status": "completed"
                },
                {
                    "type": "custom_tool_call_output",
                    "id": "cto_1",
                    "call_id": "call_1",
                    "output": "Done",
                    "status": "completed"
                }
            ]
        });

        bridge_responses_custom_tool_request(&mut body);

        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["parameters"]["required"], json!(["input"]));
        assert_eq!(
            body["tool_choice"],
            json!({"type": "function", "name": "apply_patch"})
        );
        assert_eq!(body["input"][0]["type"], "function_call");
        assert_eq!(body["input"][0].get("id"), None);
        assert_eq!(
            body["input"][0]["arguments"],
            json!("{\"input\":\"*** Begin Patch\\n*** End Patch\"}")
        );
        assert_eq!(
            body["input"][1],
            json!({
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "Done"
            })
        );
    }

    // Function argument wrappers must not leak into the raw custom input returned to the caller.
    #[test]
    fn response_bridge_unwraps_function_arguments() {
        let names = HashSet::from(["apply_patch".to_string()]);
        let mut body = json!({
            "output": [{
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "apply_patch",
                "arguments": "{\"input\":\"*** Begin Patch\\n*** End Patch\"}",
                "status": "completed"
            }]
        });

        bridge_responses_custom_tool_response(&mut body, &names);

        assert_eq!(
            body["output"][0],
            json!({
                "type": "custom_tool_call",
                "call_id": "call_1",
                "name": "apply_patch",
                "input": "*** Begin Patch\n*** End Patch"
            })
        );
    }
}
