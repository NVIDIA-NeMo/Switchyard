// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    ErrorExpectation, Scenario, ScenarioGroup, ScenarioOptions, Session, chat_payload, token_text,
    user,
};

pub fn build(options: ScenarioOptions<'_>) -> Scenario {
    let lengths = [16, 24, 32, 48, 64, 96, 128, 1_024, 2_048, 8_192];
    let sessions = lengths
        .into_iter()
        .enumerate()
        .map(|(index, tokens)| Session {
            session_id: format!("mixed-{index}"),
            payloads: vec![chat_payload(
                options.model,
                vec![user(format!(
                    "{}\nReply with the request number {index}.",
                    token_text(
                        tokens.min(options.context_window_tokens / 2),
                        "mixed_traffic"
                    )
                ))],
                if index == 9 {
                    256
                } else {
                    options.max_output_tokens
                },
            )],
        })
        .collect();
    Scenario::chat(
        "mixed-traffic",
        ScenarioGroup::Core,
        "A 70/20/10 mix of short, medium, and long requests.",
        "Tail latency stays bounded when unlike request sizes share one route.",
        ErrorExpectation::SUCCESS,
        sessions,
    )
}
