// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::{Value, json};

use super::{
    ErrorExpectation, Scenario, ScenarioGroup, ScenarioOptions, Session, chat_payload, user,
};

pub fn build(options: ScenarioOptions<'_>) -> Scenario {
    let mut messages = vec![user(
        "[scenario:tool_call_burst] Inspect eight shards, one call at a time.",
    )];
    let mut payloads = vec![chat_payload(
        options.model,
        messages.clone(),
        options.max_output_tokens.min(128),
    )];
    for index in 0..8 {
        let call_id = format!("call_{index}");
        messages.push(json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": "inspect_shard", "arguments": format!("{{\"shard\":{index}}}")}
            }]
        }));
        messages.push(json!({
            "role": "tool",
            "tool_call_id": format!("call_{index}"),
            "content": format!("shard {index}: healthy")
        }));
        payloads.push(chat_payload(
            options.model,
            messages.clone(),
            options.max_output_tokens.min(128),
        ));
    }
    for payload in &mut payloads {
        payload["tools"] = Value::Array(vec![json!({
            "type": "function",
            "function": {
                "name": "inspect_shard",
                "parameters": {
                    "type": "object",
                    "properties": {"shard": {"type": "integer"}},
                    "required": ["shard"]
                }
            }
        })]);
    }
    Scenario::chat(
        "tool-call-burst",
        ScenarioGroup::Agentic,
        "An eight-turn burst of linked assistant tool calls and tool results.",
        "The route keeps session state and forwards every call/result pair.",
        ErrorExpectation::SUCCESS,
        vec![Session {
            session_id: "tool-call-burst".to_string(),
            payloads,
        }],
    )
}
