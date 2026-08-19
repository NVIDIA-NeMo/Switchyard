// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Target-specific replay policy for OpenAI Responses reasoning items.

use serde::Deserialize;
use serde_json::Value;

/// Controls which Responses reasoning items are replayed to an upstream.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesReasoningPolicy {
    /// Preserve provider-encrypted reasoning but remove plaintext reasoning.
    ///
    /// This is the safe default for strict hosted Responses providers.
    #[default]
    PreserveEncrypted,
    /// Drop all reasoning items while preserving messages and tool-call history.
    ///
    /// Use this for local Responses-compatible servers that cannot consume
    /// another provider's encrypted reasoning representation.
    Drop,
}

impl ResponsesReasoningPolicy {
    /// Normalizes a Responses request body for this replay policy.
    pub(crate) fn normalize(self, body: &mut Value) {
        let Some(Value::Array(input)) = body.get_mut("input") else {
            return;
        };
        input.retain_mut(|item| self.normalize_item(item));
    }

    fn normalize_item(self, item: &mut Value) -> bool {
        let Some(object) = item.as_object_mut() else {
            return true;
        };
        if object.get("type").and_then(Value::as_str) != Some("reasoning") {
            return true;
        }

        let signed = matches!(
            object.get("encrypted_content").and_then(Value::as_str),
            Some(encrypted_content) if !encrypted_content.is_empty()
        );
        if self == Self::PreserveEncrypted && signed {
            object.insert("content".to_string(), Value::Array(Vec::new()));
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn mixed_history() -> Value {
        json!({
            "input": [
                {"type": "message", "role": "user", "content": []},
                {
                    "type": "reasoning",
                    "content": [{"type": "reasoning_text", "text": "plaintext"}],
                    "encrypted_content": ""
                },
                {"type": "function_call", "call_id": "call_1"},
                {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
                {
                    "type": "reasoning",
                    "content": [{"type": "reasoning_text", "text": "must be removed"}],
                    "encrypted_content": "encrypted"
                }
            ]
        })
    }

    #[test]
    fn preserve_encrypted_drops_unsigned_and_clears_plaintext() {
        let mut body = mixed_history();
        ResponsesReasoningPolicy::PreserveEncrypted.normalize(&mut body);

        let input = body["input"].as_array().expect("input array");
        let reasoning: Vec<&Value> = input
            .iter()
            .filter(|item| item["type"] == "reasoning")
            .collect();
        assert_eq!(reasoning.len(), 1);
        assert_eq!(reasoning[0]["encrypted_content"], "encrypted");
        assert_eq!(reasoning[0]["content"], json!([]));
        assert!(input.iter().any(|item| item["type"] == "function_call"));
        assert!(
            input
                .iter()
                .any(|item| item["type"] == "function_call_output")
        );
    }

    #[test]
    fn drop_removes_all_reasoning_and_keeps_tool_history() {
        let mut body = mixed_history();
        ResponsesReasoningPolicy::Drop.normalize(&mut body);

        let input = body["input"].as_array().expect("input array");
        assert!(input.iter().all(|item| item["type"] != "reasoning"));
        assert!(input.iter().any(|item| item["type"] == "function_call"));
        assert!(
            input
                .iter()
                .any(|item| item["type"] == "function_call_output")
        );
    }
}
