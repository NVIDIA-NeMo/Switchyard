// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for codec validation, diagnostics, and preservation metadata.

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};

use crate::diagnostic::TranslationDiagnostic;
use crate::error::{Result, TranslationError};
use crate::format::FormatId;
use crate::llm::{ContentBlock, LlmRequest, Message, PreservationMetadata};
use crate::policy::{
    LossyConversionPolicy, PreservationPolicy, TranslationPolicy, UnknownFieldPolicy,
};

/// Metadata key used to embed exact preserved payloads in provider JSON.
pub const SWITCHYARD_METADATA_KEY: &str = "_switchyard_translation";
/// Public alias for the embedded preservation metadata key.
pub const PRESERVATION_METADATA_KEY: &str = SWITCHYARD_METADATA_KEY;

/// Reads a JSON object or returns a typed translation error at the given path.
pub fn object<'a>(value: &'a Value, path: &str) -> Result<&'a Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| TranslationError::InvalidType {
            path: path.to_string(),
            expected: "object",
        })
}

/// Converts JSON scalars to Python-compatible string values where providers do so.
pub fn string_value(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Null => None,
        other => Some(match other {
            Value::Bool(value) => {
                if *value {
                    "True".to_string()
                } else {
                    "False".to_string()
                }
            }
            _ => other.to_string(),
        }),
    }
}

/// Returns a non-empty string when the value is string-like enough to preserve.
pub fn is_truthy_string(value: &Value) -> Option<String> {
    match value {
        Value::String(text) if !text.is_empty() => Some(text.clone()),
        _ => None,
    }
}

/// Applies the unknown-field policy and records diagnostics when configured.
pub fn push_unknown_field(
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
    path: impl Into<String>,
) -> Result<()> {
    let path = path.into();
    match policy.unknown_field_policy {
        UnknownFieldPolicy::Preserve => Ok(()),
        UnknownFieldPolicy::DropWithWarning => {
            diagnostics.push(
                TranslationDiagnostic::warning(
                    "unknown_field_dropped",
                    format!("unknown field at {path} was dropped"),
                )
                .at_path(path),
            );
            Ok(())
        }
        UnknownFieldPolicy::Reject => Err(TranslationError::UnknownField { path }),
    }
}

/// Applies the lossy-conversion policy and records diagnostics when configured.
pub fn push_lossy(
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
    message: impl Into<String>,
) -> Result<()> {
    let message = message.into();
    match policy.lossy_conversion_policy {
        LossyConversionPolicy::AllowWithDiagnostics => {
            diagnostics.push(TranslationDiagnostic::warning("lossy_conversion", message));
            Ok(())
        }
        LossyConversionPolicy::Reject => Err(TranslationError::LossyConversion(message)),
    }
}

/// Generates a stable, human-readable ID from a prefix and counter.
pub fn stable_id(prefix: &str, counter: usize) -> String {
    format!("{prefix}_{counter:08}")
}

/// Serializes JSON values into provider argument strings.
pub fn json_string(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => serde_json::to_string(other).unwrap_or_else(|_| other.to_string()),
    }
}

/// Joins non-empty text fragments with a caller-provided separator.
pub fn compact_text_blocks<'a>(
    blocks: impl IntoIterator<Item = &'a str>,
    separator: &str,
) -> String {
    blocks
        .into_iter()
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(separator)
}

/// Checks a request against declared target capabilities.
pub fn validate_request_capabilities(
    request: &LlmRequest,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<()> {
    if policy.target_capabilities.supports_tools == Some(false)
        && (!request.tools.is_empty() || messages_have_tools(&request.messages))
    {
        push_lossy(
            diagnostics,
            policy,
            "target format/profile does not support tools",
        )?;
    }
    if policy.target_capabilities.supports_images == Some(false)
        && messages_have_block(&request.messages, |block| {
            matches!(block, ContentBlock::Image { .. })
        })
    {
        push_lossy(
            diagnostics,
            policy,
            "target format/profile does not support images",
        )?;
    }
    if policy.target_capabilities.supports_audio == Some(false)
        && messages_have_block(&request.messages, |block| {
            matches!(block, ContentBlock::Audio { .. })
        })
    {
        push_lossy(
            diagnostics,
            policy,
            "target format/profile does not support audio",
        )?;
    }
    if policy.target_capabilities.supports_video == Some(false)
        && messages_have_block(&request.messages, |block| {
            matches!(block, ContentBlock::Video { .. })
        })
    {
        push_lossy(
            diagnostics,
            policy,
            "target format/profile does not support video",
        )?;
    }
    if policy.target_capabilities.supports_files == Some(false)
        && messages_have_block(&request.messages, |block| {
            matches!(block, ContentBlock::File { .. })
        })
    {
        push_lossy(
            diagnostics,
            policy,
            "target format/profile does not support files",
        )?;
    }
    if policy.target_capabilities.supports_reasoning_effort == Some(false)
        && request.reasoning.effort.is_some()
    {
        push_lossy(
            diagnostics,
            policy,
            "target format/profile does not support reasoning effort",
        )?;
    }
    if policy
        .target_capabilities
        .supports_json_schema_response_format
        == Some(false)
        && request.output.response_format.is_some()
    {
        push_lossy(
            diagnostics,
            policy,
            "target format/profile does not support structured response formats",
        )?;
    }
    Ok(())
}

// Detects whether any message carries tool calls or tool results.
fn messages_have_tools(messages: &[Message]) -> bool {
    messages_have_block(messages, |block| {
        matches!(
            block,
            ContentBlock::ToolCall(_) | ContentBlock::ToolResult(_)
        )
    })
}

// Scans message content for a caller-provided block predicate.
fn messages_have_block(messages: &[Message], predicate: impl FnMut(&ContentBlock) -> bool) -> bool {
    messages
        .iter()
        .flat_map(|message| message.content.iter())
        .any(predicate)
}

/// Captures an exact source request body according to preservation policy.
pub fn capture_request_preservation(
    format: impl Into<FormatId>,
    body: &Value,
    policy: &TranslationPolicy,
) -> PreservationMetadata {
    let mut preservation = extract_preservation(body);
    if policy.preservation != PreservationPolicy::Disabled {
        preservation.requests.insert(format.into(), body.clone());
    }
    preservation
}

/// Captures an exact source response body according to preservation policy.
pub fn capture_response_preservation(
    format: impl Into<FormatId>,
    body: &Value,
    policy: &TranslationPolicy,
) -> PreservationMetadata {
    let mut preservation = extract_preservation(body);
    if policy.preservation != PreservationPolicy::Disabled {
        preservation.responses.insert(format.into(), body.clone());
    }
    preservation
}

/// Returns an exact preserved request for the target format when available.
pub fn exact_preserved_request(
    preservation: &PreservationMetadata,
    format: impl Into<FormatId>,
    policy: &TranslationPolicy,
) -> Option<Value> {
    let format = format.into();
    (policy.preservation != PreservationPolicy::Disabled)
        .then(|| preservation.requests.get(&format).cloned())
        .flatten()
}

/// Returns an exact preserved response for the target format when available.
pub fn exact_preserved_response(
    preservation: &PreservationMetadata,
    format: impl Into<FormatId>,
    policy: &TranslationPolicy,
) -> Option<Value> {
    let format = format.into();
    (policy.preservation != PreservationPolicy::Disabled)
        .then(|| preservation.responses.get(&format).cloned())
        .flatten()
}

/// Embeds preservation metadata into a translated wire body when requested.
pub fn embed_preservation(
    mut body: Value,
    preservation: &PreservationMetadata,
    policy: &TranslationPolicy,
) -> Value {
    if policy.preservation != PreservationPolicy::Embed {
        return body;
    }
    let Ok(envelope) = serde_json::to_value(preservation) else {
        return body;
    };
    let metadata = json!({SWITCHYARD_METADATA_KEY: envelope});
    if let Some(object) = body.as_object_mut() {
        match object.get_mut("metadata") {
            Some(Value::Object(existing)) => {
                existing.insert(
                    SWITCHYARD_METADATA_KEY.to_string(),
                    metadata[SWITCHYARD_METADATA_KEY].clone(),
                );
            }
            _ => {
                object.insert("metadata".to_string(), metadata);
            }
        }
    }
    body
}

/// Extracts embedded preservation metadata from a provider wire body.
pub fn extract_preservation(body: &Value) -> PreservationMetadata {
    body.get("metadata")
        .and_then(Value::as_object)
        .and_then(|metadata| metadata.get(SWITCHYARD_METADATA_KEY))
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok())
        .unwrap_or_default()
}

/// Normalizes Anthropic tool-use IDs while keeping tool_use/tool_result pairs aligned.
pub fn normalize_anthropic_tool_use_ids(value: Value) -> Value {
    match value {
        Value::Array(messages) => {
            let mut id_map = BTreeMap::new();
            let mut used_ids = BTreeMap::new();
            Value::Array(
                messages
                    .into_iter()
                    .map(|message| normalize_message_tool_ids(message, &mut id_map, &mut used_ids))
                    .collect(),
            )
        }
        other => other,
    }
}

/// Converts a single ID into Anthropic-safe characters.
pub fn sanitize_anthropic_tool_use_id(raw: &str) -> String {
    let sanitized = raw
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    if sanitized.is_empty() {
        "toolu_empty".to_string()
    } else {
        sanitized
    }
}

// Normalizes every content block in one Anthropic message.
fn normalize_message_tool_ids(
    message: Value,
    id_map: &mut BTreeMap<String, String>,
    used_ids: &mut BTreeMap<String, String>,
) -> Value {
    let Value::Object(mut message) = message else {
        return message;
    };
    let Some(content_value) = message.remove("content") else {
        return Value::Object(message);
    };
    let Value::Array(content) = content_value else {
        message.insert("content".to_string(), content_value);
        return Value::Object(message);
    };
    let normalized = content
        .into_iter()
        .map(|block| normalize_tool_block(block, id_map, used_ids).unwrap_or_else(|block| block))
        .collect::<Vec<_>>();
    message.insert("content".to_string(), Value::Array(normalized));
    Value::Object(message)
}

// Rewrites tool_use/tool_result IDs and leaves unrelated blocks untouched.
fn normalize_tool_block(
    block: Value,
    id_map: &mut BTreeMap<String, String>,
    used_ids: &mut BTreeMap<String, String>,
) -> std::result::Result<Value, Value> {
    let Value::Object(mut block_map) = block else {
        return Err(block);
    };
    match block_map.get("type").and_then(Value::as_str) {
        Some("tool_use") => {
            let raw = block_map
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let normalized = mapped_tool_id(&raw, id_map, used_ids);
            if normalized != raw {
                block_map.insert("id".to_string(), Value::String(normalized));
                Ok(Value::Object(block_map))
            } else {
                Err(Value::Object(block_map))
            }
        }
        Some("tool_result") => {
            let raw = block_map
                .get("tool_use_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let normalized = mapped_tool_id(&raw, id_map, used_ids);
            if normalized != raw {
                block_map.insert("tool_use_id".to_string(), Value::String(normalized));
                Ok(Value::Object(block_map))
            } else {
                Err(Value::Object(block_map))
            }
        }
        _ => Err(Value::Object(block_map)),
    }
}

// Gives colliding raw IDs stable, deterministic suffixes.
fn mapped_tool_id(
    raw: &str,
    id_map: &mut BTreeMap<String, String>,
    used_ids: &mut BTreeMap<String, String>,
) -> String {
    if let Some(existing) = id_map.get(raw) {
        return existing.clone();
    }
    let mut candidate = sanitize_anthropic_tool_use_id(raw);
    if let Some(owner) = used_ids.get(&candidate) {
        if owner != raw {
            candidate = format!("{}_{}", candidate, stable_suffix(raw));
        }
    }
    id_map.insert(raw.to_string(), candidate.clone());
    used_ids.insert(candidate.clone(), raw.to_string());
    candidate
}

// Stable FNV-1a suffix for collision disambiguation.
fn stable_suffix(raw: &str) -> String {
    let mut hash: u64 = 1469598103934665603;
    for byte in raw.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(1099511628211);
    }
    format!("{hash:08x}")
}
