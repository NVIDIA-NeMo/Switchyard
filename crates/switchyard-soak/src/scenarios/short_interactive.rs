// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::client::{Endpoint, request_body};

use super::{
    ErrorExpectation, RequestCase, Scenario, ScenarioGroup, ScenarioOptions, Session,
    baseline_load, chat_payload, user,
};

pub fn build(options: ScenarioOptions<'_>) -> Scenario {
    let prompts = [
        "Reply with exactly OK.".to_string(),
        "Name one primary color.".to_string(),
        "Return the number after 41.".to_string(),
        format!(
            "[scenario:short_interactive] {} Reply briefly.",
            "stable prefix ".repeat(options.prompt_bytes.div_ceil(14))
        ),
    ];
    let sessions = prompts
        .iter()
        .enumerate()
        .map(|(index, prompt)| Session {
            session_id: format!("short-{index}"),
            payloads: vec![chat_payload(
                options.model,
                vec![user(prompt)],
                options.max_output_tokens,
            )],
        })
        .collect();
    let soak_requests = Endpoint::ALL
        .iter()
        .flat_map(|endpoint| {
            prompts.iter().enumerate().map(move |(index, prompt)| {
                RequestCase::new(
                    *endpoint,
                    format!("short-{index}-{}", endpoint.as_str()),
                    request_body(
                        *endpoint,
                        options.model,
                        prompt,
                        options.max_output_tokens,
                        true,
                    ),
                )
            })
        })
        .collect();
    Scenario {
        id: "short-interactive",
        group: ScenarioGroup::Core,
        description: "Short prompts that establish HTTP, routing, TTFT, and latency overhead.",
        expected: "All public endpoints succeed; oha and AIPerf establish the baseline.",
        expected_error_rate: ErrorExpectation::SUCCESS,
        sessions,
        soak_requests,
        load_profiles: baseline_load(),
    }
}
