// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    ErrorExpectation, Scenario, ScenarioGroup, ScenarioOptions, Session, chat_payload, user,
};

pub fn build(options: ScenarioOptions<'_>) -> Scenario {
    let sessions = (0..20)
        .map(|index| {
            let hard = (8..10).contains(&index) || index >= 15;
            let marker = if hard {
                "classifier_hard"
            } else {
                "classifier_easy"
            };
            Session {
                session_id: format!("classifier-mix-{index}"),
                payloads: vec![chat_payload(
                    options.model,
                    vec![user(format!(
                        "[scenario:{marker}] Request {index}: {}",
                        if hard {
                            "reason about a distributed failure with incomplete evidence"
                        } else {
                            "return the sum of two and two"
                        }
                    ))],
                    options.max_output_tokens,
                )],
            }
        })
        .collect();
    Scenario::chat(
        "classifier-mix",
        ScenarioGroup::Agentic,
        "Two deterministic easy/hard mixes: 80/20 followed by 50/50.",
        "Selection shares match the input mix and classifier overhead is visible.",
        ErrorExpectation::SUCCESS,
        sessions,
    )
}
