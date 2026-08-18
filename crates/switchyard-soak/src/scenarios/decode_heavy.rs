// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    ErrorExpectation, Scenario, ScenarioGroup, ScenarioOptions, Session, chat_payload, user,
};

pub fn build(options: ScenarioOptions<'_>) -> Scenario {
    let sessions = [512, 1024]
        .into_iter()
        .map(|tokens| Session {
            session_id: format!("decode-heavy-{tokens}"),
            payloads: vec![chat_payload(
                options.model,
                vec![user(format!(
                    "[scenario:decode_heavy] Write {tokens} numbered one-word items."
                ))],
                tokens,
            )],
        })
        .collect();
    Scenario::chat(
        "decode-heavy",
        ScenarioGroup::Core,
        "Short inputs with 512-token and 1,024-token output limits.",
        "Output token throughput stays stable while long streams remain valid.",
        ErrorExpectation::SUCCESS,
        sessions,
    )
}
