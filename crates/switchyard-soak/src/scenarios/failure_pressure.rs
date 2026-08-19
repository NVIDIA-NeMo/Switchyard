// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    ErrorExpectation, Scenario, ScenarioGroup, ScenarioOptions, Session, chat_payload, user,
};

pub fn build(options: ScenarioOptions<'_>) -> Scenario {
    let cases = [
        ("truncated-stream", "truncated_stream"),
        ("upstream-429", "upstream_429"),
        ("upstream-500", "upstream_500"),
        ("classifier-invalid", "classifier_invalid"),
    ];
    let sessions = cases
        .into_iter()
        .map(|(id, marker)| Session {
            session_id: id.to_string(),
            payloads: vec![chat_payload(
                options.model,
                vec![user(format!(
                    "[scenario:{marker}] Exercise the configured recovery path."
                ))],
                options.max_output_tokens,
            )],
        })
        .collect();
    Scenario::chat(
        "failure-pressure",
        ScenarioGroup::Resilience,
        "Bounded 429, 500, malformed-classifier, and truncated-stream injections.",
        "Retries recover transient failures; terminal failures remain explicit and bounded.",
        ErrorExpectation::MIXED,
        sessions,
    )
}
