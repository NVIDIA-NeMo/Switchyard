// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    ErrorExpectation, Scenario, ScenarioGroup, ScenarioOptions, Session, chat_payload, token_text,
    user,
};

pub fn build(options: ScenarioOptions<'_>) -> Scenario {
    let prefix_tokens = (options.prompt_bytes / 2).clamp(512, 8_192);
    let shared = token_text(prefix_tokens, "prefix_reuse_shared");
    let sessions = (0..8)
        .map(|index| {
            let prefix = if index < 4 {
                shared.clone()
            } else {
                token_text(prefix_tokens, &format!("prefix_reuse_unique_{index}"))
            };
            Session {
                session_id: format!("prefix-reuse-{index}"),
                payloads: vec![chat_payload(
                    options.model,
                    vec![user(format!("{prefix}\nReturn item {index}."))],
                    options.max_output_tokens.min(32),
                )],
            }
        })
        .collect();
    Scenario::chat(
        "prefix-reuse",
        ScenarioGroup::Core,
        "Matched requests with shared and unique long prefixes.",
        "The report exposes cache-sensitive TTFT without changing route behavior.",
        ErrorExpectation::SUCCESS,
        sessions,
    )
}
