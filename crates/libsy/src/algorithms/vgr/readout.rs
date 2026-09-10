// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Probability-scored verifier readout.

use serde_json::{Value, json};
use switchyard_protocol::{AggLlmResponse, FormatId, Request, WireFormat};

pub(super) const MAX_OUTPUT_TOKENS: u64 = 4;

pub(super) fn request_logprobs(request: &mut Request) {
    request.llm_request.reasoning.effort = Some("none".to_string());
    request
        .llm_request
        .extensions
        .fields
        .insert("logprobs".into(), json!(true));
    request
        .llm_request
        .extensions
        .fields
        .insert("top_logprobs".into(), json!(8));
}

pub(super) fn p_yes(response: &AggLlmResponse) -> Option<f64> {
    let alternatives = response
        .preservation
        .responses
        .get(&FormatId::from(WireFormat::OpenAiChat))?
        .pointer("/choices/0/logprobs/content/0/top_logprobs")?
        .as_array()?;
    let mut yes = 0.0;
    let mut no = 0.0;
    let mut found = false;
    for entry in alternatives {
        let token = normalize(entry.get("token")?.as_str()?);
        let probability = entry.get("logprob").and_then(Value::as_f64)?.exp();
        match token.as_str() {
            "yes" | "y" | "true" => {
                yes += probability;
                found = true;
            }
            "no" | "n" | "false" => {
                no += probability;
                found = true;
            }
            _ => {}
        }
    }
    (found && yes + no > 0.0).then_some(yes / (yes + no))
}

fn normalize(token: &str) -> String {
    token
        .trim()
        .trim_matches(['"', '\'', '.', ','])
        .to_lowercase()
}
