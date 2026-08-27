// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    ErrorExpectation, Scenario, ScenarioGroup, ScenarioOptions, Session, chat_payload, token_text,
    user,
};

pub fn build(options: ScenarioOptions<'_>) -> Scenario {
    let tokens = (options.context_window_tokens * 9 / 10).max(1);
    Scenario::chat(
        "context-overflow",
        ScenarioGroup::Resilience,
        "A near-window request that one configured target rejects as too long.",
        "A multi-target route retries an eligible target and returns a valid response.",
        ErrorExpectation::SUCCESS,
        vec![Session {
            session_id: "context-overflow".to_string(),
            payloads: vec![chat_payload(
                options.model,
                vec![user(token_text(tokens, "context_overflow"))],
                options.max_output_tokens,
            )],
        }],
    )
}
