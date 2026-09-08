// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-agnostic helpers shared by wire-format codecs.

use serde_json::{Map, Value};

use crate::llm::ContentBlock;

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
            ContentBlock::Text { text } => Some(text.as_str()),
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
        if !known.contains(&key.as_str()) {
            extensions.insert(key.clone(), value.clone());
        }
    }
    extensions
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
