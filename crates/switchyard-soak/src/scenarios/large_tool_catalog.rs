// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::{Value, json};

use super::{
    ErrorExpectation, Scenario, ScenarioGroup, ScenarioOptions, Session, chat_payload, user,
};

fn tools(count: usize) -> Vec<Value> {
    (0..count)
        .map(|index| {
            json!({
                "type": "function",
                "function": {
                    "name": format!("lookup_record_{index}"),
                    "description": format!("Look up record {index} in the test catalog."),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {"type": "integer", "minimum": 1, "maximum": 10}
                        },
                        "required": ["query"],
                        "additionalProperties": false
                    }
                }
            })
        })
        .collect()
}

pub fn build(options: ScenarioOptions<'_>) -> Scenario {
    let sessions = [16, 64]
        .into_iter()
        .map(|count| {
            let mut payload = chat_payload(
                options.model,
                vec![user(format!(
                    "[scenario:large_tool_catalog] Use lookup_record_{} for query green.",
                    count - 1
                ))],
                options.max_output_tokens.min(128),
            );
            payload["tools"] = Value::Array(tools(count));
            payload["tool_choice"] = json!("auto");
            Session {
                session_id: format!("large-tool-catalog-{count}"),
                payloads: vec![payload],
            }
        })
        .collect();
    Scenario::chat(
        "large-tool-catalog",
        ScenarioGroup::Agentic,
        "Requests with 16-tool and 64-tool JSON-schema catalogs.",
        "Routing overhead remains bounded while large tool schemas are forwarded intact.",
        ErrorExpectation::SUCCESS,
        sessions,
    )
}
