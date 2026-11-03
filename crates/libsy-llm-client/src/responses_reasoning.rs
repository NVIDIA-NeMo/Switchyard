// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Target-specific replay policy for OpenAI Responses reasoning items.

use serde::Deserialize;
use serde_json::Value;

/// Controls which Responses reasoning items are replayed to an upstream.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesReasoningPolicy {
    /// Preserve encrypted reasoning and stored IDs the provider can read.
    ///
    /// This is the default for every Responses model, regardless of its URL.
    #[default]
    PreserveEncrypted,
    /// Drop all reasoning items and replayed hosted web searches while preserving
    /// messages and tool-call history.
    ///
    /// Use this for local Responses-compatible servers that cannot consume
    /// another provider's encrypted reasoning representation.
    Drop,
}

impl ResponsesReasoningPolicy {
    /// Normalizes a Responses request body for this replay policy.
    pub(crate) fn normalize(self, body: &mut Value) {
        let has_stored_reasoning = body.get("store").and_then(Value::as_bool) != Some(false);
        let Some(Value::Array(input)) = body.get_mut("input") else {
            return;
        };
        input.retain_mut(|item| self.normalize_item(item, has_stored_reasoning));
    }

    // Stored reasoning can be replayed by ID without encrypted content.
    fn normalize_item(self, item: &mut Value, has_stored_reasoning: bool) -> bool {
        let Some(object) = item.as_object_mut() else {
            return true;
        };
        // A hosted provider's server-side web search is replayed as a `web_search_call` item,
        // which a local server cannot run or type (llama.cpp answers 400 "Cannot determine type
        // of 'item'"). Its findings are in the assistant message that follows it.
        if self == Self::Drop
            && object.get("type").and_then(Value::as_str) == Some("web_search_call")
        {
            return false;
        }
        if object.get("type").and_then(Value::as_str) != Some("reasoning") {
            return true;
        }

        let encrypted_content = object.get("encrypted_content").and_then(Value::as_str);
        if encrypted_content.is_some_and(switchyard_translation::is_anthropic_thinking_envelope) {
            return false;
        }
        let has_encrypted_content = encrypted_content.is_some_and(|content| !content.is_empty());
        let has_stored_id = object
            .get("id")
            .and_then(Value::as_str)
            .is_some_and(|id| !id.is_empty());
        if self == Self::PreserveEncrypted
            && (has_encrypted_content || (has_stored_reasoning && has_stored_id))
        {
            object.insert("content".to_string(), Value::Array(Vec::new()));
            true
        } else {
            false
        }
    }
}
