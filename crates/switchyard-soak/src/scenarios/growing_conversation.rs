// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    ErrorExpectation, Scenario, ScenarioGroup, ScenarioOptions, Session, assistant, chat_payload,
    user,
};

pub fn build(options: ScenarioOptions<'_>) -> Scenario {
    let mut messages = Vec::new();
    let mut payloads = Vec::new();
    for turn in 0..8 {
        messages.push(user(format!(
            "[scenario:growing_conversation] Turn {turn}: remember the number {turn}."
        )));
        payloads.push(chat_payload(
            options.model,
            messages.clone(),
            options.max_output_tokens.min(64),
        ));
        messages.push(assistant(format!("I recorded {turn}.")));
    }
    Scenario::chat(
        "growing-conversation",
        ScenarioGroup::Agentic,
        "An eight-turn session whose complete message history grows each turn.",
        "Session affinity holds and latency tracks the growing history.",
        ErrorExpectation::SUCCESS,
        vec![Session {
            session_id: "growing-conversation".to_string(),
            payloads,
        }],
    )
}
