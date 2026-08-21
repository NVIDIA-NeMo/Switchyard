// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Codex tool-declaration shapes, preserved across a Responses/Chat translation.
//!
//! Codex declares its tools in two ways a plain Responses request does not, and
//! both were lost in translation. The symptom of either is the same: the upstream
//! is offered no such tool, the model — still told about it by Codex's own prompt —
//! writes the call as prose, and the turn completes having executed nothing, which
//! reads as success to the client.
//!
//! **`additional_tools` input item.** Codex 0.146 puts its whole toolset in an
//! input item, `{"type": "additional_tools", "role": "developer", "tools": [...]}`,
//! and sends no top-level `tools` key at all. Undecoded, that item reaches the
//! input catch-all and becomes opaque user content. The item's `tools` array holds
//! ordinary Responses tool specs, so the request codec feeds it through the normal
//! tool decoder and records the names here, letting a same-format hop put the
//! declaration back where the client had it.
//!
//! **`custom` tools.** Codex declares its shell and patch tools as Responses
//! **custom** tools, whose input is freeform text rather than JSON arguments:
//! `{"type": "custom", "name": "exec", "format": {"type": "text"}}`. It then
//! expects the call back as a `custom_tool_call` carrying a raw `input` string. An
//! OpenAI-compatible chat upstream has no such tool type — it accepts only
//! `function` tools with a JSON Schema. The request codec therefore exposes each
//! custom tool as a function with a single string property, and this module records
//! which names were originally custom so the response can be turned back into a
//! `custom_tool_call`.
//!
//! The mapping rides in the request's [`ProviderExtensions`] under a prefixed key,
//! so no provider-neutral type grows a Codex-specific field and no codec forwards
//! it to an upstream — the same approach as [`crate::codex_namespaces`].
//!
//! A `grammar` format cannot be enforced through a JSON Schema, so a
//! grammar-constrained custom tool degrades to an unconstrained string. That is
//! strictly better than dropping it: the model can still call the tool, and Codex
//! validates the input on receipt.

use std::collections::{HashMap, HashSet};

use serde_json::{Map, Value, json};
use switchyard_protocol::ProviderExtensions;

/// Request extension key holding the names declared via `additional_tools`.
///
/// Codex 0.146 declares its tools in an `additional_tools` **input item** rather
/// than the request's `tools` array. Recording which names arrived that way lets a
/// Responses-to-Responses hop put them back where the client had them.
pub const ADDITIONAL_TOOLS_KEY: &str = "switchyard_codex_additional_tools";

/// Reads the tool specs out of every `additional_tools` input item.
///
/// The item's `tools` array holds ordinary Responses tool specs — `function`,
/// `namespace`, and `custom` — so the caller feeds them through the normal tool
/// decoder. Left undecoded the item reaches the input catch-all and becomes
/// opaque user content: the upstream is then offered no tools at all, while the
/// model is still told about them by Codex's prompt.
pub fn additional_tool_specs(input: Option<&Value>) -> Vec<Value> {
    let Some(items) = input.and_then(Value::as_array) else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(Value::as_object)
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("additional_tools"))
        .filter_map(|item| item.get("tools").and_then(Value::as_array))
        .flatten()
        .cloned()
        .collect()
}

/// Records that `name` was declared through an `additional_tools` item.
pub fn record_additional_tool(additional: &mut Map<String, Value>, name: &str) {
    additional.insert(name.to_string(), Value::Bool(true));
}

/// Stores the collected `additional_tools` names on a request's extensions.
pub fn attach_additional_tools(
    extensions: &mut ProviderExtensions,
    additional: Map<String, Value>,
) {
    if !additional.is_empty() {
        extensions
            .fields
            .insert(ADDITIONAL_TOOLS_KEY.to_string(), Value::Object(additional));
    }
}

/// Reads the recorded `additional_tools` names back off a request's extensions.
pub fn additional_tool_names(extensions: &ProviderExtensions) -> HashSet<String> {
    extensions
        .fields
        .get(ADDITIONAL_TOOLS_KEY)
        .and_then(Value::as_object)
        .map(|additional| additional.keys().cloned().collect())
        .unwrap_or_default()
}

/// Request extension key holding the set of tool names that were `custom`.
///
/// Prefixed so it cannot collide with a real provider field, and so a codec that
/// allowlists provider fields never forwards it.
pub const CUSTOM_TOOLS_KEY: &str = "switchyard_codex_custom_tools";

/// Property name carrying a custom tool's freeform input on the chat wire.
pub const CUSTOM_INPUT_PROPERTY: &str = "input";

/// The JSON Schema a custom tool is advertised with on a chat upstream.
///
/// One required string, because the tool's real contract is freeform text. The
/// description names the tool so a model that reads only the schema still knows
/// what the field is for.
pub fn custom_tool_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            CUSTOM_INPUT_PROPERTY: {
                "type": "string",
                "description": "The complete tool input, verbatim, as a single string.",
            }
        },
        "required": [CUSTOM_INPUT_PROPERTY],
        "additionalProperties": false,
    })
}

/// Records that `name` was declared as a Responses `custom` tool.
pub fn record_custom_tool(customs: &mut Map<String, Value>, name: &str) {
    customs.insert(name.to_string(), Value::Bool(true));
}

/// Stores a collected set on a request's extensions, when it has entries.
pub fn attach_custom_tools(extensions: &mut ProviderExtensions, customs: Map<String, Value>) {
    if !customs.is_empty() {
        extensions
            .fields
            .insert(CUSTOM_TOOLS_KEY.to_string(), Value::Object(customs));
    }
}

/// Reads the recorded set back off a request's extensions.
pub fn custom_tool_names(extensions: &ProviderExtensions) -> HashSet<String> {
    extensions
        .fields
        .get(CUSTOM_TOOLS_KEY)
        .and_then(Value::as_object)
        .map(|customs| customs.keys().cloned().collect())
        .unwrap_or_default()
}

/// Extracts a custom tool's freeform input from chat-style JSON arguments.
///
/// The advertised schema asks for `{"input": "..."}`, but a model may answer with
/// a bare string, or with the argument object it would have used for a function
/// tool. Each case yields the most faithful text available rather than an error,
/// because a dropped call costs the whole turn.
pub fn custom_input_from_arguments(arguments: &Value) -> String {
    match arguments {
        Value::String(text) => text.clone(),
        Value::Object(object) => match object.get(CUSTOM_INPUT_PROPERTY) {
            // The expected shape.
            Some(Value::String(text)) => text.clone(),
            // A single unnamed argument is unambiguous even under a wrong key.
            None if object.len() == 1 => match object.values().next() {
                Some(Value::String(text)) => text.clone(),
                Some(other) => other.to_string(),
                None => String::new(),
            },
            // Anything else is passed through as JSON: Codex can still read it,
            // and inventing a shape here would hide the model's actual output.
            Some(other) => other.to_string(),
            None => Value::Object(object.clone()).to_string(),
        },
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Item-id prefix a Responses backend requires on a custom tool call.
///
/// A strict backend validates the prefix per item type: Azure rejects a
/// `custom_tool_call` whose id starts with `fc` -- "Expected an ID that begins
/// with 'ctc'". The client persists the item and replays it next turn, so a
/// wrong prefix here fails the *following* request, not this one.
const CUSTOM_CALL_ID_PREFIX: &str = "ctc";

/// Prefix a function call id carries, and which a custom call must not keep.
const FUNCTION_CALL_ID_PREFIX: &str = "fc";

/// Re-prefixes a function-call item id for a custom tool call.
///
/// The suffix is preserved so the id stays as stable and as traceable as the one
/// the upstream or the id policy produced.
fn custom_call_item_id(id: &str) -> String {
    if id.starts_with(CUSTOM_CALL_ID_PREFIX) {
        return id.to_string();
    }
    let suffix = id
        .strip_prefix(FUNCTION_CALL_ID_PREFIX)
        .unwrap_or(id)
        .trim_start_matches('_');
    if suffix.is_empty() {
        CUSTOM_CALL_ID_PREFIX.to_string()
    } else {
        format!("{CUSTOM_CALL_ID_PREFIX}_{suffix}")
    }
}

/// Tracks the streamed items that belong to a custom tool, and their new ids.
///
/// A Responses stream announces the tool name once, on
/// `response.output_item.added`, and every later delta identifies the item only
/// by `item_id`. Rewriting those deltas therefore needs the mapping this type
/// accumulates as the stream is walked. It maps the id as the upstream sent it to
/// the re-prefixed one, so every reference to the item stays consistent.
#[derive(Debug, Default)]
pub struct CustomToolStreamState {
    custom_item_ids: HashMap<String, String>,
}

impl CustomToolStreamState {
    /// Remembers that the item once called `old_id` is now `new_id`.
    fn remember(&mut self, old_id: &str, new_id: &str) {
        if !old_id.is_empty() {
            self.custom_item_ids
                .insert(old_id.to_string(), new_id.to_string());
        }
    }

    /// The new id of an item announced as a custom tool call, if any.
    fn new_id_of(&self, item_id: &str) -> Option<&str> {
        self.custom_item_ids.get(item_id).map(String::as_str)
    }
}

/// Rewrites a `function_call` item into a `custom_tool_call`.
///
/// Returns the item's old and new id when it was rewritten, so a caller walking a
/// stream can remap the deltas that follow.
fn rewrite_call_item(
    object: &mut Map<String, Value>,
    names: &HashSet<String>,
) -> Option<(String, String)> {
    if object.get("type").and_then(Value::as_str) != Some("function_call") {
        return None;
    }
    let name = object.get("name").and_then(Value::as_str)?;
    if !names.contains(name) {
        return None;
    }

    let input = object
        .get("arguments")
        .map(custom_input_from_arguments)
        .unwrap_or_default();
    object.insert("type".to_string(), Value::String("custom_tool_call".into()));
    object.insert("input".to_string(), Value::String(input));
    // `arguments` has no meaning on a custom tool call, and leaving it would
    // present the same call twice in two different shapes.
    object.remove("arguments");

    let old_id = object.get("id").and_then(Value::as_str)?.to_string();
    let new_id = custom_call_item_id(&old_id);
    object.insert("id".to_string(), Value::String(new_id.clone()));
    Some((old_id, new_id))
}

/// Turns `function_call` items and their argument deltas back into custom tool
/// calls, for the tool names the request declared as `custom`.
///
/// Walks the whole value, covering a buffered body and each streaming event.
/// `state` carries the item ids seen so far and must be reused across the events
/// of one stream; a buffered body can pass a fresh one.
pub fn restore_custom_tool_calls(
    body: &mut Value,
    names: &HashSet<String>,
    state: &mut CustomToolStreamState,
) {
    if names.is_empty() {
        return;
    }
    restore_in_value(body, names, state);
}

fn restore_in_value(value: &mut Value, names: &HashSet<String>, state: &mut CustomToolStreamState) {
    match value {
        Value::Array(values) => {
            for value in values {
                restore_in_value(value, names, state);
            }
        }
        Value::Object(object) => {
            // Children first: an `item` inside a `response.output_item.added` event
            // must register its new id before the event's own `item_id` is remapped.
            for value in object.values_mut() {
                restore_in_value(value, names, state);
            }
            if let Some((old_id, new_id)) = rewrite_call_item(object, names) {
                state.remember(&old_id, &new_id);
            }
            rewrite_stream_event(object, state);
        }
        _ => {}
    }
}

/// Renames the argument-delta events of a custom tool call.
///
/// A Responses client reads a custom tool's input from
/// `response.custom_tool_call_input.delta` / `.done`, not from the
/// `function_call_arguments` events, so an unrenamed delta stream leaves the call
/// with empty input even once the item itself is the right type.
fn rewrite_stream_event(object: &mut Map<String, Value>, state: &CustomToolStreamState) {
    // The item is identified only by id here, so an event for a function tool must
    // be left alone. Every event that names a rewritten item carries its new id,
    // including the ones that keep their own type.
    let Some(new_id) = object
        .get("item_id")
        .and_then(Value::as_str)
        .and_then(|item_id| state.new_id_of(item_id))
        .map(ToOwned::to_owned)
    else {
        return;
    };
    object.insert("item_id".to_string(), Value::String(new_id));

    let Some(event) = object.get("type").and_then(Value::as_str) else {
        return;
    };
    let renamed = match event {
        "response.function_call_arguments.delta" => "response.custom_tool_call_input.delta",
        "response.function_call_arguments.done" => "response.custom_tool_call_input.done",
        _ => return,
    };
    object.insert("type".to_string(), Value::String(renamed.into()));
    // The payload field is named for the tool kind as well.
    if let Some(delta) = object.remove("delta") {
        object.insert("delta".to_string(), delta);
    }
    if let Some(arguments) = object.remove("arguments") {
        object.insert("input".to_string(), arguments);
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Map, json};
    use switchyard_protocol::ProviderExtensions;

    use super::{
        CustomToolStreamState, attach_custom_tools, custom_call_item_id,
        custom_input_from_arguments, custom_tool_names, record_custom_tool,
        restore_custom_tool_calls,
    };

    fn extensions(names: &[&str]) -> ProviderExtensions {
        let mut customs = Map::new();
        for name in names {
            record_custom_tool(&mut customs, name);
        }
        let mut extensions = ProviderExtensions::default();
        attach_custom_tools(&mut extensions, customs);
        extensions
    }

    #[test]
    fn records_and_reads_back_custom_tool_names() {
        let names = custom_tool_names(&extensions(&["exec", "apply_patch"]));
        assert!(names.contains("exec"));
        assert!(names.contains("apply_patch"));
        assert_eq!(names.len(), 2);
        // An absent mapping is empty rather than an error, so a non-Codex request
        // is untouched.
        assert!(custom_tool_names(&ProviderExtensions::default()).is_empty());
    }

    // The advertised schema asks for {"input": "..."}, but a model that answers
    // in another shape must still produce a usable call.
    #[test]
    fn recovers_freeform_input_from_every_argument_shape() {
        assert_eq!(
            custom_input_from_arguments(&json!({"input": "echo hi"})),
            "echo hi"
        );
        assert_eq!(custom_input_from_arguments(&json!("echo hi")), "echo hi");
        assert_eq!(
            custom_input_from_arguments(&json!({"cmd": "echo hi"})),
            "echo hi"
        );
        assert_eq!(custom_input_from_arguments(&json!(null)), "");
        // Two named arguments are ambiguous, so the JSON is preserved rather than
        // one of them being picked.
        let many = custom_input_from_arguments(&json!({"cmd": "ls", "dir": "/tmp"}));
        assert!(many.contains("\"cmd\""), "{many}");
        assert!(many.contains("\"dir\""), "{many}");
    }

    #[test]
    fn rewrites_a_buffered_function_call_into_a_custom_tool_call() {
        let names = custom_tool_names(&extensions(&["exec"]));
        let mut body = json!({
            "output": [{
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "exec",
                "arguments": {"input": "echo hi"}
            }]
        });

        restore_custom_tool_calls(&mut body, &names, &mut CustomToolStreamState::default());

        let item = &body["output"][0];
        assert_eq!(item["type"], "custom_tool_call");
        assert_eq!(item["input"], "echo hi");
        assert_eq!(item["call_id"], "call_1");
        assert!(
            item.get("arguments").is_none(),
            "arguments must not survive: {item}"
        );
    }

    // A function tool keeps its own shape, or every tool call would arrive as a
    // custom one.
    #[test]
    fn leaves_a_function_tool_alone() {
        let names = custom_tool_names(&extensions(&["exec"]));
        let mut body = json!({
            "output": [{"type": "function_call", "name": "search", "arguments": {"q": "x"}}]
        });
        let before = body.clone();

        restore_custom_tool_calls(&mut body, &names, &mut CustomToolStreamState::default());

        assert_eq!(body, before);
    }

    // Deltas name only the item id, so the rename depends on state carried from
    // the `output_item.added` event earlier in the same stream.
    #[test]
    fn renames_the_argument_deltas_of_a_custom_call_only() {
        let names = custom_tool_names(&extensions(&["exec"]));
        let mut state = CustomToolStreamState::default();

        let mut added = json!({
            "type": "response.output_item.added",
            "item": {"type": "function_call", "id": "fc_1", "name": "exec", "arguments": {}}
        });
        restore_custom_tool_calls(&mut added, &names, &mut state);
        assert_eq!(added["item"]["type"], "custom_tool_call");
        assert_eq!(added["item"]["id"], "ctc_1");

        let mut delta = json!({
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "delta": "echo hi"
        });
        restore_custom_tool_calls(&mut delta, &names, &mut state);
        assert_eq!(delta["type"], "response.custom_tool_call_input.delta");
        assert_eq!(delta["delta"], "echo hi");
        // The delta must name the item by its new id, or the client cannot attach
        // the input to the call it announced.
        assert_eq!(delta["item_id"], "ctc_1");

        // An item that was never announced as custom keeps the function events.
        let mut other = json!({
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_2",
            "delta": "{}"
        });
        restore_custom_tool_calls(&mut other, &names, &mut state);
        assert_eq!(other["type"], "response.function_call_arguments.delta");
    }

    // A strict Responses backend validates the item-id prefix per item type. Azure
    // rejected a replayed `custom_tool_call` with
    //   Invalid 'input[8].id': 'fc_2'. Expected an ID that begins with 'ctc'.
    // The client persists the item, so a wrong prefix fails the FOLLOWING request.
    #[test]
    fn re_prefixes_the_item_id_for_a_custom_call() {
        assert_eq!(custom_call_item_id("fc_2"), "ctc_2");
        assert_eq!(custom_call_item_id("fc2"), "ctc_2");
        // An id the upstream already made a custom one is left untouched.
        assert_eq!(custom_call_item_id("ctc_9"), "ctc_9");
        // An id with no recognizable prefix still comes back valid.
        assert_eq!(custom_call_item_id("abc"), "ctc_abc");
        assert_eq!(custom_call_item_id("fc"), "ctc");
    }

    #[test]
    fn does_nothing_without_recorded_custom_tools() {
        let mut body = json!({
            "output": [{"type": "function_call", "name": "exec", "arguments": {"input": "x"}}]
        });
        let before = body.clone();

        restore_custom_tool_calls(
            &mut body,
            &custom_tool_names(&ProviderExtensions::default()),
            &mut CustomToolStreamState::default(),
        );

        assert_eq!(body, before);
    }
}
