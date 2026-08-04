// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-agnostic helpers shared by buffered wire-format codecs.

use serde_json::{Map, Value};

use crate::diagnostic::TranslationDiagnostic;
use crate::error::Result;
use crate::format::{FormatId, WireFormat};
use crate::llm::ContentBlock;
use crate::policy::{TranslationPolicy, UnknownFieldPolicy};
use crate::util::{push_lossy_at, push_unknown_field};

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

/// Applies policy to unknown top-level provider fields and preserves them when configured.
pub(crate) fn provider_extensions(
    object: &Map<String, Value>,
    known: &[&str],
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Map<String, Value>> {
    let mut extensions = Map::new();
    for (key, value) in object {
        if !known.contains(&key.as_str()) {
            push_unknown_field(diagnostics, policy, format!("$.{key}"))?;
            if policy.unknown_field_policy == UnknownFieldPolicy::Preserve {
                extensions.insert(key.clone(), value.clone());
            }
        }
    }
    Ok(extensions)
}

/// Returns a raw tool for its owning format or applies cross-format loss policy.
pub(crate) fn raw_tool_for_target(
    provider: &FormatId,
    raw: &Value,
    index: usize,
    target: WireFormat,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Option<Value>> {
    if provider.as_str() == target.as_str() {
        return Ok(Some(raw.clone()));
    }
    let path = format!("$.tools[{index}]");
    push_lossy_at(
        diagnostics,
        policy,
        format!("provider tool at {path} for {provider} cannot be represented in {target}"),
        path,
        provider.as_str(),
        target,
    )?;
    Ok(None)
}
