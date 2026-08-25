// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    ErrorExpectation, Scenario, ScenarioGroup, ScenarioOptions, Session, chat_payload, user,
};

pub fn build(options: ScenarioOptions<'_>) -> Scenario {
    Scenario::chat(
        "client-cancellation",
        ScenarioGroup::Resilience,
        "Slow streaming requests whose client timeout cancels work in flight.",
        "Cancellation releases connections and does not destabilize later traffic.",
        ErrorExpectation::ALL,
        vec![Session {
            session_id: "client-cancellation".to_string(),
            payloads: vec![chat_payload(
                options.model,
                vec![user(
                    "[scenario:client_cancellation] Delay this response until the client leaves.",
                )],
                options.max_output_tokens,
            )],
        }],
    )
}
