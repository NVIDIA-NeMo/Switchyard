// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::{Value, json};
use switchyard_translation::{PreservationPolicy, TranslationPolicy};

pub const REASONING_MODEL: &str = "z-ai/glm-5.2";

pub fn normalized_policy() -> TranslationPolicy {
    TranslationPolicy {
        preservation: PreservationPolicy::Disabled,
        ..TranslationPolicy::default()
    }
}

pub fn text_and_encrypted_reasoning_details() -> Value {
    json!([
        {
            "type": "reasoning.text",
            "text": "Inspect the tool result.",
            "signature": "opaque-signature",
            "id": "reasoning-1",
            "format": "anthropic-claude-v1",
            "index": 0
        },
        {
            "type": "reasoning.encrypted",
            "data": "opaque-encrypted-reasoning",
            "id": "reasoning-1",
            "format": "anthropic-claude-v1",
            "index": 1
        }
    ])
}

pub fn shell_tool_call() -> Value {
    json!({
        "id": "call-1",
        "type": "function",
        "function": {"name": "shell", "arguments": "{\"command\":\"pwd\"}"}
    })
}
