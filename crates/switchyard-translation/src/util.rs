// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for codec validation, diagnostics, and preservation metadata.

use std::collections::BTreeMap;

use serde_json::{Map, Value, json};

use crate::diagnostic::TranslationDiagnostic;
use crate::error::{Result, TranslationError};
use crate::format::{FormatId, WireFormat};
use crate::llm::{ContentBlock, InstructionBlock, LlmRequest, Message, PreservationMetadata, Role};
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

/// Prepends a system prompt while preserving unrelated fields in built-in request formats.
///
/// Exact preserved bodies are patched in place so unrelated provider fields survive. A custom
/// or malformed preserved body is discarded and will be rebuilt from the normalized request.
pub fn prepend_system_prompt(request: &mut LlmRequest, prompt: &str) {
    let is_already_first = request.instructions.first().is_some_and(|instruction| {
        instruction.role == Role::System
            && matches!(
                instruction.content.as_slice(),
                [ContentBlock::Text { text }] if text_starts_with_prompt(text, prompt)
            )
    });
    if !is_already_first {
        request.instructions.insert(
            0,
            InstructionBlock {
                role: Role::System,
                content: vec![ContentBlock::Text {
                    text: prompt.to_string(),
                }],
            },
        );
    }
    let mut embedded_formats = Vec::new();
    request.preservation.requests.retain(|format, body| {
        let patched = match format.as_str() {
            format if format == WireFormat::OpenAiChat.as_str() => {
                prepend_openai_chat_system_prompt(body, prompt)
            }
            format if format == WireFormat::OpenAiResponses.as_str() => {
                prepend_text_system_prompt(body, "instructions", prompt)
            }
            format if format == WireFormat::AnthropicMessages.as_str() => {
                prepend_anthropic_system_prompt(body, prompt)
            }
            _ => false,
        };
        if patched && take_embedded_preservation(body) {
            embedded_formats.push(format.clone());
        }
        patched
    });
    // Refresh envelopes only on bodies that arrived with one. The snapshot is serialized while
    // every body is envelope-free, so it cannot recursively nest stale preservation metadata.
    if !embedded_formats.is_empty()
        && let Ok(envelope) = serde_json::to_value(&request.preservation)
    {
        for format in embedded_formats {
            if let Some(body) = request.preservation.requests.get_mut(&format) {
                restore_embedded_preservation(body, &envelope);
            }
        }
    }
}

// Inserts a system message ahead of an exact Chat Completions message list.
fn prepend_openai_chat_system_prompt(body: &mut Value, prompt: &str) -> bool {
    let Some(body) = body.as_object_mut() else {
        return false;
    };
    let messages = body
        .entry("messages".to_string())
        .or_insert_with(|| Value::Array(Vec::new()));
    let Some(messages) = messages.as_array_mut() else {
        return false;
    };
    if messages.first().is_some_and(|message| {
        message.get("role").and_then(Value::as_str) == Some("system")
            && message
                .get("content")
                .is_some_and(|content| content_starts_with_prompt(content, prompt))
    }) {
        return true;
    }
    messages.insert(0, json!({"role": "system", "content": prompt}));
    true
}

// Recognizes the scalar and block-array forms accepted for system content.
fn content_starts_with_prompt(content: &Value, prompt: &str) -> bool {
    match content {
        Value::String(text) => text_starts_with_prompt(text, prompt),
        Value::Array(blocks) => blocks
            .first()
            .is_some_and(|block| text_block_matches_prompt(block, prompt)),
        _ => false,
    }
}

fn text_block_matches_prompt(block: &Value, prompt: &str) -> bool {
    block.get("type").and_then(Value::as_str) == Some("text")
        && block
            .get("text")
            .and_then(Value::as_str)
            .is_some_and(|text| text_starts_with_prompt(text, prompt))
}

fn text_starts_with_prompt(text: &str, prompt: &str) -> bool {
    text == prompt
        || text
            .strip_prefix(prompt)
            .is_some_and(|suffix| suffix.starts_with("\n\n"))
}

// Prepends a prompt to a provider's scalar instruction field.
fn prepend_text_system_prompt(body: &mut Value, field: &str, prompt: &str) -> bool {
    let Some(body) = body.as_object_mut() else {
        return false;
    };
    prepend_text_field(body, field, prompt)
}

fn prepend_text_field(body: &mut Map<String, Value>, field: &str, prompt: &str) -> bool {
    match body.get_mut(field) {
        Some(Value::String(existing)) if text_starts_with_prompt(existing, prompt) => {
            return true;
        }
        Some(Value::String(existing)) if existing.is_empty() => {
            *existing = prompt.to_string();
        }
        Some(Value::String(existing)) => {
            *existing = format!("{prompt}\n\n{existing}");
        }
        Some(value @ Value::Null) => {
            *value = Value::String(prompt.to_string());
        }
        None => {
            body.insert(field.to_string(), Value::String(prompt.to_string()));
        }
        Some(_) => return false,
    }
    true
}

// Preserves Anthropic's structured system blocks, including cache-control metadata.
fn prepend_anthropic_system_prompt(body: &mut Value, prompt: &str) -> bool {
    let Some(body) = body.as_object_mut() else {
        return false;
    };
    if let Some(Value::Array(blocks)) = body.get_mut("system") {
        if blocks
            .first()
            .is_some_and(|block| text_block_matches_prompt(block, prompt))
        {
            return true;
        }
        blocks.insert(0, json!({"type": "text", "text": prompt}));
        return true;
    }
    prepend_text_field(body, "system", prompt)
}

// Removes a stale embedded snapshot and reports whether the caller must refresh it.
fn take_embedded_preservation(body: &mut Value) -> bool {
    if let Some(metadata) = body.get_mut("metadata").and_then(Value::as_object_mut) {
        return metadata.remove(SWITCHYARD_METADATA_KEY).is_some();
    }
    false
}

// Attaches the refreshed snapshot to a body that originally carried one.
fn restore_embedded_preservation(body: &mut Value, envelope: &Value) {
    if let Some(metadata) = body.get_mut("metadata").and_then(Value::as_object_mut) {
        metadata.insert(SWITCHYARD_METADATA_KEY.to_string(), envelope.clone());
    }
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
    if let Some(owner) = used_ids.get(&candidate)
        && owner != raw
    {
        candidate = format!("{}_{}", candidate, stable_suffix(raw));
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
