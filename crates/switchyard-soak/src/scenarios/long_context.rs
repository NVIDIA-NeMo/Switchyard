// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    ErrorExpectation, Scenario, ScenarioGroup, ScenarioOptions, Session, chat_payload, token_text,
    user,
};

pub fn build(options: ScenarioOptions<'_>) -> Scenario {
    let mut lengths = vec![8_192, 32_000, options.context_window_tokens * 9 / 10];
    lengths.retain(|length| *length < options.context_window_tokens);
    lengths.sort_unstable();
    lengths.dedup();
    let sessions = lengths
        .into_iter()
        .map(|tokens| Session {
            session_id: format!("long-context-{tokens}"),
            payloads: vec![chat_payload(
                options.model,
                vec![user(format!(
                    "{}\nSummarize the final marker in one sentence.",
                    token_text(tokens, "long_context")
                ))],
                options.max_output_tokens.min(64),
            )],
        })
        .collect();
    Scenario::chat(
        "long-context",
        ScenarioGroup::Core,
        "Requests at 8K, 32K, and near the configured context window.",
        "TTFT rises with input length without routing errors or memory growth.",
        ErrorExpectation::SUCCESS,
        sessions,
    )
}
