// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::{Value, json};
use switchyard_translation::{LossyConversionPolicy, PreservationPolicy, TranslationPolicy};

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

pub fn strict_normalized_policy() -> TranslationPolicy {
    TranslationPolicy {
        preservation: PreservationPolicy::Disabled,
        lossy_conversion_policy: LossyConversionPolicy::Reject,
        ..TranslationPolicy::default()
    }
}

pub fn mcp_definition() -> Value {
    json!({"type": "mcp", "server_label": "inventory",
        "server_url": "https://inventory.example/mcp", "require_approval": "never"})
}

pub fn mcp_request(tool: Value) -> Value {
    json!({"model": "route", "input": "Check inventory", "tools": [tool]})
}

pub fn mcp_call() -> Value {
    json!({"type": "mcp_call", "id": "mcp_1", "server_label": "inventory",
        "name": "lookup", "arguments": "{\"sku\":\"widget\"}", "output": "12",
        "error": null, "approval_request_id": null})
}

pub fn mcp_listing() -> Value {
    json!({"type": "mcp_list_tools", "id": "mcpl_1", "server_label": "inventory",
        "tools": [{"name": "lookup", "description": "Look up stock",
            "input_schema": {"type": "object"}, "annotations": {"readOnlyHint": true}}]})
}

pub fn mcp_response(items: Value) -> Value {
    json!({"id": "resp_1", "model": "route", "status": "completed", "output": items})
}
