// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-agnostic helpers shared by wire-format codecs.

use serde_json::{Map, Value};

use crate::diagnostic::TranslationDiagnostic;
use crate::error::{Result, TranslationError};
use crate::format::WireFormat;
use crate::llm::{ContentBlock, LlmRequest, Role, Tool, ToolChoice};
use crate::policy::TranslationPolicy;
use crate::util::push_lossy;

/// This check reports metadata loss or rejects conversion before grouping can discard the fields.
pub(crate) fn check_responses_metadata_loss(
    request: &LlmRequest,
    target: WireFormat,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<()> {
    let mut has_conversation = false;
    for message in &request.messages {
        let Some(metadata) = &message.responses else {
            has_conversation = true;
            continue;
        };
        if metadata.fields.get("type").and_then(Value::as_str) == Some("additional_tools") {
            continue;
        }
        if metadata.fields.contains_key("phase") {
            push_lossy(
                diagnostics,
                policy,
                format!("Responses assistant phase is not supported by {target}"),
            )?;
        }
        // Reasoning item IDs live in the block's details and can survive Chat conversion.
        let loses_reasoning_id = target != WireFormat::OpenAiChat
            && message.content.iter().any(|block| {
                matches!(block, ContentBlock::Reasoning { details, .. }
                    if details.iter().any(|item| item.get("id").is_some()))
            });
        if metadata.fields.contains_key("id") || loses_reasoning_id {
            push_lossy(
                diagnostics,
                policy,
                format!(
                    "Responses item id is not supported by {target}; tool call_id remains separate"
                ),
            )?;
        }
        if matches!(message.role, Role::System | Role::Developer) {
            if has_conversation {
                push_lossy(
                    diagnostics,
                    policy,
                    format!("Responses inline instruction position changes when encoding {target}"),
                )?;
            }
            if message.role == Role::Developer && target == WireFormat::AnthropicMessages {
                push_lossy(
                    diagnostics,
                    policy,
                    "Responses developer instructions become Anthropic system instructions",
                )?;
            }
        } else {
            has_conversation = true;
        }
    }
    Ok(())
}

// Internal provenance survives mutations that invalidate exact request replay.
pub(crate) const ANTHROPIC_REQUEST_KEY: &str = "switchyard_anthropic_request";

/// Returns true when the request was decoded from an Anthropic Messages body.
pub(crate) fn is_anthropic_request(request: &LlmRequest) -> bool {
    request.extensions.fields.get(ANTHROPIC_REQUEST_KEY) == Some(&Value::Bool(true))
}

/// Converts an OpenAI allowed-tools policy to a restricted function list and mode.
pub(crate) fn allowed_function_tools(
    request: &LlmRequest,
) -> Result<Option<(Vec<Tool>, ToolChoice)>> {
    let Some(ToolChoice::Raw(choice)) = &request.tool_choice else {
        return Ok(None);
    };
    if choice.get("type").and_then(Value::as_str) != Some("allowed_tools") {
        return Ok(None);
    }
    let error = || {
        TranslationError::LossyConversion(
            "allowed_tools translation requires auto or required mode \
             and a nonempty subset of known function tools"
                .to_string(),
        )
    };
    let policy = choice.get("allowed_tools").unwrap_or(choice);
    let mode = match policy.get("mode").and_then(Value::as_str) {
        Some("auto") => ToolChoice::Auto,
        Some("required") => ToolChoice::Required,
        _ => return Err(error()),
    };
    let allowed = policy
        .get("tools")
        .and_then(Value::as_array)
        .ok_or_else(error)?;
    if allowed.is_empty() {
        return Err(error());
    }
    let mut names = Vec::with_capacity(allowed.len());
    for tool in allowed {
        if tool.get("type").and_then(Value::as_str) != Some("function") {
            return Err(error());
        }
        let function = tool.get("function").unwrap_or(tool);
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
            .ok_or_else(error)?;
        let name = match function.get("namespace").and_then(Value::as_str) {
            Some(namespace) if !namespace.is_empty() => {
                crate::codex_namespaces::qualified_tool_name(namespace, name)
            }
            _ => name.to_string(),
        };
        if !request
            .tools
            .iter()
            .filter_map(Tool::function)
            .any(|tool| tool.name == name)
        {
            return Err(error());
        }
        names.push(name);
    }
    // Filter the catalog, rather than rebuilding it from selectors, to retain definitions
    // and avoid duplicating tools when the allowed list repeats a name.
    let tools = request
        .tools
        .iter()
        .filter(|tool| {
            tool.function()
                .is_some_and(|tool| names.contains(&tool.name))
        })
        .cloned()
        .collect();
    Ok(Some((tools, mode)))
}

/// Returns whether a role name is recognized by a supported provider API.
pub(crate) fn is_known_role_name(name: &str) -> bool {
    matches!(
        name,
        "system" | "developer" | "user" | "assistant" | "tool" | "function"
    )
}

/// Extracts text-like blocks and joins them for text-only provider fields.
pub(crate) fn text_from_blocks(content: &[ContentBlock], separator: &str) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text, .. } => Some(text.as_str()),
            ContentBlock::Refusal { text } => Some(text.as_str()),
            ContentBlock::Unknown { raw, .. } => raw.as_str(),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(separator)
}

/// Extracts private reasoning blocks without mixing them into visible text.
pub(crate) fn reasoning_text_from_blocks(content: &[ContentBlock], separator: &str) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Reasoning { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(separator)
}

/// Extracts displayable text from structured reasoning details.
pub(crate) fn reasoning_text_from_details(details: &[Value]) -> Option<String> {
    let parts = details
        .iter()
        .filter_map(Value::as_object)
        .filter_map(|detail| {
            detail
                .get("text")
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
                .or_else(|| {
                    detail
                        .get("summary")
                        .and_then(Value::as_str)
                        .filter(|summary| !summary.is_empty())
                })
        })
        .collect::<Vec<_>>();
    (!parts.is_empty()).then(|| parts.join("\n"))
}

/// Collects reasoning text from a Responses reasoning item's `content` or `summary`
/// array, or from a bare string, into `out`. Empty strings are skipped.
pub(crate) fn collect_responses_reasoning_text(value: Option<&Value>, out: &mut Vec<String>) {
    match value {
        Some(Value::String(text)) if !text.is_empty() => out.push(text.clone()),
        Some(Value::Array(items)) => {
            for item in items {
                match item {
                    Value::String(text) if !text.is_empty() => out.push(text.clone()),
                    Value::Object(object) => {
                        if matches!(
                            object.get("type").and_then(Value::as_str),
                            Some("reasoning_text" | "summary_text" | "text")
                        ) && let Some(text) = object.get("text").and_then(Value::as_str)
                            && !text.is_empty()
                        {
                            out.push(text.to_string());
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
}

/// Returns the opaque payload of the first encrypted reasoning detail, if any.
///
/// Two detail shapes are accepted: the documented `{"type": "reasoning.encrypted", "data"}`
/// object, and a verbatim Responses `reasoning` item carrying `encrypted_content` (the shape
/// the buffered request decoder stores when it keeps the provider item whole).
pub(crate) fn encrypted_reasoning_data(details: &[Value]) -> Option<String> {
    details
        .iter()
        .filter_map(Value::as_object)
        .find_map(|detail| match detail.get("type").and_then(Value::as_str) {
            Some("reasoning.encrypted") => detail.get("data").and_then(Value::as_str),
            Some("reasoning") => detail.get("encrypted_content").and_then(Value::as_str),
            _ => None,
        })
        .filter(|data| !data.is_empty())
        .map(ToOwned::to_owned)
}

/// Returns the provider item id recorded on the first `reasoning.encrypted` detail, if any.
/// Encrypted reasoning is bound to the item id it was issued under, so a replay must reuse it.
pub(crate) fn encrypted_reasoning_item_id(details: &[Value]) -> Option<String> {
    details
        .iter()
        .filter_map(Value::as_object)
        .find(|detail| {
            matches!(
                detail.get("type").and_then(Value::as_str),
                Some("reasoning.encrypted" | "reasoning")
            )
        })
        .and_then(|detail| detail.get("id").and_then(Value::as_str))
        .filter(|id| !id.is_empty())
        .map(ToOwned::to_owned)
}

/// Returns the first non-empty string stored under the requested keys.
pub(crate) fn first_nonempty_string<'a>(
    object: &'a Map<String, Value>,
    keys: &[&str],
) -> Option<&'a str> {
    keys.iter().find_map(|key| {
        object
            .get(*key)
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
    })
}

/// Copies unknown provider fields into the IR extension map.
pub(crate) fn provider_extensions(
    object: &Map<String, Value>,
    known: &[&str],
) -> Map<String, Value> {
    let mut extensions = Map::new();
    for (key, value) in object {
        if key != ANTHROPIC_REQUEST_KEY && !known.contains(&key.as_str()) {
            extensions.insert(key.clone(), value.clone());
        }
    }
    extensions
}

/// Invalid charges remain unknown so accounting can use configured token rates.
pub(crate) fn provider_cost(value: &Map<String, Value>) -> Option<f64> {
    value
        .get("cost")
        .and_then(Value::as_f64)
        .filter(|cost| cost.is_finite() && *cost >= 0.0)
}

/// Cache duration must survive decoding because one-hour writes can have a different rate.
pub(crate) fn cache_usage_details(
    value: &Map<String, Value>,
    cached: Option<u64>,
    created: Option<u64>,
) -> Option<Box<crate::llm::InputCacheUsage>> {
    let one_hour = value
        .get("cache_creation")
        .and_then(|details| details.get("ephemeral_1h_input_tokens"))
        .and_then(Value::as_u64);
    let mut cache = crate::llm::Usage::cache_details(cached, created);
    if let Some(tokens) = one_hour {
        cache
            .get_or_insert_with(Default::default)
            .cache_creation_1h_input_tokens = Some(tokens);
    }
    cache
}

/// These fields preserve provider charges and cache duration across response formats.
pub(crate) fn encode_usage_metadata(mut value: Value, usage: &crate::llm::Usage) -> Value {
    if let Some(cost) = usage
        .provider_cost
        .filter(|cost| cost.is_finite() && *cost >= 0.0)
    {
        value["cost"] = serde_json::json!(cost);
    }
    if let Some(tokens) = usage
        .cache
        .as_ref()
        .and_then(|cache| cache.cache_creation_1h_input_tokens)
    {
        value["cache_creation"] = serde_json::json!({"ephemeral_1h_input_tokens": tokens});
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn encrypted_reasoning_helpers_accept_both_detail_shapes() {
        let documented = vec![json!({"type": "reasoning.encrypted", "data": "blob", "id": "rs_1"})];
        assert_eq!(
            encrypted_reasoning_data(&documented).as_deref(),
            Some("blob")
        );
        assert_eq!(
            encrypted_reasoning_item_id(&documented).as_deref(),
            Some("rs_1")
        );
        let verbatim_item = vec![json!({
            "type": "reasoning", "id": "rs_2", "summary": [], "encrypted_content": "blob2"
        })];
        assert_eq!(
            encrypted_reasoning_data(&verbatim_item).as_deref(),
            Some("blob2")
        );
        assert_eq!(
            encrypted_reasoning_item_id(&verbatim_item).as_deref(),
            Some("rs_2")
        );
        assert_eq!(encrypted_reasoning_data(&[json!({"type": "other"})]), None);
    }
}
