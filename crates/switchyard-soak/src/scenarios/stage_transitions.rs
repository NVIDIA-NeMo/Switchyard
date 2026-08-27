// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::json;

use super::{
    ErrorExpectation, Scenario, ScenarioGroup, ScenarioOptions, Session, chat_payload, user,
};

pub fn build(options: ScenarioOptions<'_>) -> Scenario {
    let mut messages = vec![user(
        "[scenario:stage_exploration] Find the failing module, then fix it.",
    )];
    let mut payloads = vec![chat_payload(
        options.model,
        messages.clone(),
        options.max_output_tokens.min(128),
    )];

    messages.extend([
        json!({"role":"assistant","tool_calls":[{"id":"stage_1","type":"function","function":{"name":"run_tests","arguments":"{}"}}]}),
        json!({"role":"tool","tool_call_id":"stage_1","content":"fatal: critical tool failure: repository index is unavailable"}),
    ]);
    payloads.push(chat_payload(
        options.model,
        messages.clone(),
        options.max_output_tokens.min(128),
    ));

    messages.extend([
        json!({"role":"assistant","tool_calls":[{"id":"stage_2","type":"function","function":{"name":"apply_patch","arguments":"{}"}}]}),
        json!({"role":"tool","tool_call_id":"stage_2","content":"patch applied"}),
        json!({"role":"assistant","tool_calls":[{"id":"stage_3","type":"function","function":{"name":"run_tests","arguments":"{}"}}]}),
        json!({"role":"tool","tool_call_id":"stage_3","content":"all tests passed"}),
    ]);
    payloads.push(chat_payload(
        options.model,
        messages,
        options.max_output_tokens.min(128),
    ));

    Scenario::chat(
        "stage-transitions",
        ScenarioGroup::Agentic,
        "One growing session that moves through exploration, critical failure, and productive work.",
        "The stage router changes tiers at the configured evidence boundaries.",
        ErrorExpectation::SUCCESS,
        vec![Session {
            session_id: "stage-transitions".to_string(),
            payloads,
        }],
    )
}
