// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for codec validation, diagnostics, and preservation metadata.

use std::collections::{BTreeMap, HashSet};

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use serde_json::{Map, Value, json};
use switchyard_protocol::ModelId;

use crate::diagnostic::TranslationDiagnostic;
use crate::error::{Result, TranslationError};
use crate::format::{FormatId, WireFormat};
use crate::llm::{
    ContentBlock, InstructionBlock, LlmRequest, Message, PreservationMetadata, ProviderExtensions,
    Role,
};
use crate::policy::{
    LossyConversionPolicy, PreservationPolicy, TranslationPolicy, UnknownFieldPolicy,
};

/// Metadata key used to embed exact preserved payloads in provider JSON.
pub const SWITCHYARD_METADATA_KEY: &str = "_switchyard_translation";
/// Public alias for the embedded preservation metadata key.
pub const PRESERVATION_METADATA_KEY: &str = SWITCHYARD_METADATA_KEY;

const ANTHROPIC_TOOL_ID_ENCODING_PREFIX: &str = "sy64_";

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

// These Responses history items are valid only as top-level `input` items.
pub(crate) fn is_responses_builtin_tool_item(item: &Value) -> bool {
    matches!(
        item.get("type").and_then(Value::as_str),
        Some(
            "apply_patch_call"
                | "apply_patch_call_output"
                | "shell_call"
                | "shell_call_output"
                | "computer_call"
                | "computer_call_output"
        )
    )
}

// Prevent provider-specific tool history from being downgraded to target-visible prose.
pub(crate) fn reject_responses_builtin_tool_item(
    provider: &FormatId,
    item: &Value,
    target: WireFormat,
) -> Result<()> {
    if provider.as_str() == WireFormat::OpenAiResponses.as_str()
        && is_responses_builtin_tool_item(item)
    {
        return Err(TranslationError::UnsupportedTranslation {
            from: WireFormat::OpenAiResponses.into(),
            to: target.into(),
        });
    }
    Ok(())
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
fn messages_have_block(
    messages: &[Message],
    mut predicate: impl FnMut(&ContentBlock) -> bool,
) -> bool {
    messages
        .iter()
        .any(|message| content_has_block(&message.content, &mut predicate))
}

// Scans content recursively because tool results can contain media blocks.
fn content_has_block(
    content: &[ContentBlock],
    predicate: &mut impl FnMut(&ContentBlock) -> bool,
) -> bool {
    content.iter().any(|block| {
        predicate(block)
            || match block {
                ContentBlock::ToolResult(result) => content_has_block(&result.content, predicate),
                _ => false,
            }
    })
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

/// Applies a selected target model and optionally prepends its system prompt.
///
/// Preserved built-in bodies are updated in their native format when possible so exact replay
/// keeps caller fields that are not represented in the normalized request.
/// Call this once per candidate using a request that has not already received a target prompt.
pub fn prepare_request_for_target(
    request: &mut LlmRequest,
    target: &ModelId,
    prompt: Option<&str>,
) {
    let target = target.to_string();
    request.model = Some(target.clone());
    if let Some(prompt) = prompt {
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
    prepare_preserved_requests(&mut request.preservation, &target, prompt);
}

// Retains exact replay only where the built-in wire body can receive the target changes safely.
fn prepare_preserved_requests(
    preservation: &mut PreservationMetadata,
    target: &str,
    prompt: Option<&str>,
) {
    preservation.requests.retain(|format, body| {
        let is_builtin = format.as_str() == WireFormat::OpenAiChat.as_str()
            || format.as_str() == WireFormat::OpenAiResponses.as_str()
            || format.as_str() == WireFormat::AnthropicMessages.as_str();
        let Some(body) = is_builtin.then_some(body).and_then(Value::as_object_mut) else {
            return false;
        };
        body.insert("model".to_string(), Value::String(target.to_string()));
        match prompt {
            None => true,
            Some(prompt) if format.as_str() == WireFormat::OpenAiChat.as_str() => {
                prepend_openai_chat_system(body, prompt)
            }
            Some(prompt) if format.as_str() == WireFormat::OpenAiResponses.as_str() => {
                prepend_openai_responses_instructions(body, prompt)
            }
            Some(prompt) if format.as_str() == WireFormat::AnthropicMessages.as_str() => {
                body.remove(crate::codecs::common::ANTHROPIC_REQUEST_KEY);
                prepend_anthropic_system(body, prompt)
            }
            Some(_) => false,
        }
    });
}

// Prepends a target prompt without rebuilding or otherwise changing a Chat request.
fn prepend_openai_chat_system(body: &mut Map<String, Value>, prompt: &str) -> bool {
    let message = json!({"role": "system", "content": prompt});
    match body.get_mut("messages") {
        None | Some(Value::Null) => {
            body.insert("messages".to_string(), Value::Array(vec![message]));
        }
        Some(Value::Array(messages)) => messages.insert(0, message),
        Some(_) => return false,
    }
    true
}

// Prepends a target prompt without rebuilding or otherwise changing a Responses request.
fn prepend_openai_responses_instructions(body: &mut Map<String, Value>, prompt: &str) -> bool {
    match body.get_mut("instructions") {
        None | Some(Value::Null) => {
            body.insert(
                "instructions".to_string(),
                Value::String(prompt.to_string()),
            );
        }
        Some(Value::String(instructions)) if instructions.is_empty() => {
            *instructions = prompt.to_string();
        }
        Some(Value::String(instructions)) => {
            instructions.insert_str(0, &format!("{prompt}\n\n"));
        }
        Some(_) => return false,
    }
    true
}

// Prepends a target prompt without rebuilding or otherwise changing an Anthropic request.
fn prepend_anthropic_system(body: &mut Map<String, Value>, prompt: &str) -> bool {
    match body.get_mut("system") {
        None | Some(Value::Null) => {
            body.insert("system".to_string(), Value::String(prompt.to_string()));
        }
        Some(Value::String(system)) if system.is_empty() => {
            *system = prompt.to_string();
        }
        Some(Value::String(system)) => {
            system.insert_str(0, &format!("{prompt}\n\n"));
        }
        Some(Value::Array(blocks)) => blocks.insert(
            0,
            json!({
                "type": "text",
                "text": prompt,
            }),
        ),
        Some(_) => return false,
    }
    true
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

/// Metadata key holding the tool IDs an inbound conversation already uses.
pub const SEEN_TOOL_IDS_KEY: &str = "switchyard_seen_tool_ids";

/// Prefix for rewritten repeats of an already-used Anthropic tool ID.
const ANTHROPIC_TOOL_ID_COLLISION_PREFIX: &str = "sydup";

/// Rewrites tool IDs that repeat within one Anthropic-facing conversation.
///
/// Anthropic rejects a conversation whose `tool_use` IDs are not unique, and
/// clients pair `tool_result` blocks to calls by ID alone, so a backend that
/// repeats an ID (`call_0`, `call_1`, `call_0`) would leave the conversation
/// unusable. An ID seen for the first time passes through unchanged; a repeat
/// mints `sydup{N}_{raw}` (N counts the occurrences, starting at 2), which
/// stays within Anthropic's allowed characters and keeps the original ID
/// readable after stripping `sydup{N}_`. The prefix is distinct from the
/// `sy64_` encoding scheme so the two never alias.
///
/// The rewriter walks encoded Anthropic JSON, so the same pass serves request
/// bodies (`messages[].content[]`), response bodies (`content[]`), and stream
/// events (`content_block_start`).
#[derive(Default)]
pub struct AnthropicToolIdRewriter {
    /// Maps a raw ID to the ID its most recent occurrence was given, so a
    /// tool result resolves to the call it answers.
    assigned: BTreeMap<String, String>,
    /// Every ID reserved so far: seeds, first occurrences, and minted IDs.
    used: HashSet<String>,
}

impl AnthropicToolIdRewriter {
    /// Creates a rewriter that treats `seen` as already-used conversation IDs.
    pub fn new<I>(seen: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<String>,
    {
        let mut rewriter = Self::default();
        for id in seen {
            let id = id.into();
            rewriter.used.insert(id.clone());
            rewriter.assigned.insert(id.clone(), id);
        }
        rewriter
    }

    /// Rewrites every `tool_use` and `tool_result` ID in `body` in place.
    pub fn rewrite_body(&mut self, body: &mut Value) {
        match body {
            Value::Array(items) => {
                for item in items {
                    self.rewrite_body(item);
                }
            }
            Value::Object(object) => {
                // Only a block's own ID field is rewritten; everything else is
                // walked so nested blocks are still found.
                let id_field = match object.get("type").and_then(Value::as_str) {
                    Some("tool_use") => Some("id"),
                    Some("tool_result") => Some("tool_use_id"),
                    _ => None,
                };
                match id_field {
                    Some(field) => {
                        if let Some(raw) = object.get(field).and_then(Value::as_str) {
                            let raw = raw.to_owned();
                            let rewritten = if field == "id" {
                                self.rewrite_call(&raw)
                            } else {
                                self.resolve_result(&raw)
                            };
                            object.insert(field.to_string(), Value::String(rewritten));
                        }
                    }
                    None => {
                        for value in object.values_mut() {
                            self.rewrite_body(value);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // Gives the next occurrence of a call ID an unused Anthropic-facing ID.
    fn rewrite_call(&mut self, raw: &str) -> String {
        if self.used.insert(raw.to_string()) {
            self.assigned.insert(raw.to_string(), raw.to_string());
            return raw.to_string();
        }
        // Occurrence 2 and later: mint sydup{N}_{raw} until it is unused. A
        // backend id that literally spells a minted candidate bumps N instead.
        let mut occurrence = 2;
        let mut candidate = self.mint(raw, occurrence);
        while self.used.contains(&candidate) {
            occurrence += 1;
            candidate = self.mint(raw, occurrence);
        }
        self.used.insert(candidate.clone());
        self.assigned.insert(raw.to_string(), candidate.clone());
        candidate
    }

    // Pairs a result with the ID its call was given; an unseen result ID is
    // dangling history and passes through unchanged.
    fn resolve_result(&mut self, raw: &str) -> String {
        match self.assigned.get(raw) {
            Some(assigned) => assigned.clone(),
            None => self.rewrite_call(raw),
        }
    }

    fn mint(&self, raw: &str, occurrence: u32) -> String {
        format!("{ANTHROPIC_TOOL_ID_COLLISION_PREFIX}{occurrence}_{raw}")
    }
}

/// Reads the tool IDs a request's extensions recorded for this conversation.
pub(crate) fn seen_tool_ids(extensions: &ProviderExtensions) -> Vec<String> {
    extensions
        .fields
        .get(SEEN_TOOL_IDS_KEY)
        .and_then(Value::as_array)
        .map(|ids| {
            ids.iter()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

/// Records the tool IDs an inbound conversation already uses, so response
/// encoders can give backend repeats of those IDs distinct replacements.
pub(crate) fn attach_seen_tool_ids(fields: &mut Map<String, Value>, messages: &[Message]) {
    let mut ids = Vec::new();
    for message in messages {
        for block in &message.content {
            match block {
                ContentBlock::ToolCall(call) => ids.push(call.id.clone()),
                ContentBlock::ToolResult(result) => ids.push(result.tool_call_id.clone()),
                _ => {}
            }
        }
    }
    if !ids.is_empty() {
        fields.insert(
            SEEN_TOOL_IDS_KEY.to_string(),
            Value::Array(ids.into_iter().map(Value::String).collect()),
        );
    }
}

/// Converts an ID into a reversible Anthropic-safe representation.
pub fn sanitize_anthropic_tool_use_id(raw: &str) -> String {
    let is_safe = !raw.is_empty()
        && raw
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-');
    if is_safe && !raw.starts_with(ANTHROPIC_TOOL_ID_ENCODING_PREFIX) {
        return raw.to_string();
    }

    format!(
        "{ANTHROPIC_TOOL_ID_ENCODING_PREFIX}{}",
        URL_SAFE_NO_PAD.encode(raw.as_bytes())
    )
}

/// Restores an ID encoded by [`sanitize_anthropic_tool_use_id`].
pub(crate) fn desanitize_anthropic_tool_use_id(encoded: &str) -> String {
    let Some(payload) = encoded.strip_prefix(ANTHROPIC_TOOL_ID_ENCODING_PREFIX) else {
        return encoded.to_string();
    };

    URL_SAFE_NO_PAD
        .decode(payload)
        .ok()
        .and_then(|bytes| String::from_utf8(bytes).ok())
        .unwrap_or_else(|| encoded.to_string())
}

#[cfg(test)]
mod tests {
    use super::{
        AnthropicToolIdRewriter, desanitize_anthropic_tool_use_id, sanitize_anthropic_tool_use_id,
    };
    use serde_json::json;

    // Keeps ordinary provider IDs unchanged while making unsafe IDs reversible.
    #[test]
    fn anthropic_tool_id_encoding_round_trips() {
        assert_eq!(
            sanitize_anthropic_tool_use_id("call_abc-123"),
            "call_abc-123"
        );

        for raw in ["", "functions.list_skills:0", "工具/lookup"] {
            let encoded = sanitize_anthropic_tool_use_id(raw);
            assert!(
                encoded
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
            );
            assert_eq!(desanitize_anthropic_tool_use_id(&encoded), raw);
        }
    }

    // Escapes the reserved prefix and leaves malformed encoded values untouched.
    #[test]
    fn anthropic_tool_id_encoding_disambiguates_its_prefix() {
        let raw = "sy64_Zm9v";
        let encoded = sanitize_anthropic_tool_use_id(raw);
        assert_ne!(encoded, raw);
        assert_eq!(desanitize_anthropic_tool_use_id(&encoded), raw);
        assert_eq!(desanitize_anthropic_tool_use_id("sy64_%%%"), "sy64_%%%");
    }

    // First occurrences and their results pass through; repeats get occurrence
    // numbered IDs and each result still pairs with its own call.
    #[test]
    fn tool_id_rewriter_passes_unique_ids_and_mints_repeats() {
        let mut body = json!([
            {"type": "tool_use", "id": "call_0"},
            {"type": "tool_result", "tool_use_id": "call_0"},
            {"type": "tool_use", "id": "call_9"},
            {"type": "tool_use", "id": "call_0"},
            {"type": "tool_result", "tool_use_id": "call_0"},
            {"type": "text", "text": "untouched"}
        ]);
        AnthropicToolIdRewriter::default().rewrite_body(&mut body);

        let string_field = |index: usize, field: &str| body[index][field].as_str().unwrap();
        assert_eq!(string_field(0, "id"), "call_0");
        assert_eq!(string_field(1, "tool_use_id"), "call_0");
        assert_eq!(string_field(2, "id"), "call_9");
        assert_eq!(string_field(3, "id"), "sydup2_call_0");
        assert_eq!(string_field(4, "tool_use_id"), "sydup2_call_0");
        assert_eq!(body[5]["text"], "untouched");
    }

    // IDs recorded from earlier turns collide on the first occurrence here, and
    // a minted ID never aliases an ID the backend spelled literally.
    #[test]
    fn tool_id_rewriter_treats_seen_ids_as_used() {
        let mut body = json!([
            {"type": "tool_use", "id": "call_0"},
            {"type": "tool_use", "id": "call_0"}
        ]);
        AnthropicToolIdRewriter::new(["call_0", "sydup2_call_0"]).rewrite_body(&mut body);

        assert_eq!(body[0]["id"], "sydup3_call_0");
        assert_eq!(body[1]["id"], "sydup4_call_0");
        // The original ID stays readable in the minted one.
        assert_eq!(
            body[1]["id"].as_str().unwrap().strip_prefix("sydup4_"),
            Some("call_0")
        );
    }
}
