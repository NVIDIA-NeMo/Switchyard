// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The cheap readout: the probability a verifier put on "yes" at its first token.
//!
//! Normalized responses carry no logprobs, so this reads the preserved Chat
//! body. Anything that cannot be scored is `None`, which never commits.

use serde_json::{Value, json};
use switchyard_protocol::{AggLlmResponse, FormatId, Request, WireFormat};

pub(super) const MAX_OUTPUT_TOKENS: u64 = 4;

/// Asks for a direct verdict with its alternatives; reasoning stays request-local.
pub(super) fn request_logprobs(request: &mut Request) {
    request.llm_request.reasoning.effort = Some("none".to_string());
    let extensions = &mut request.llm_request.extensions.fields;
    extensions.insert("logprobs".to_string(), json!(true));
    extensions.insert("top_logprobs".to_string(), json!(8));
}

pub(super) fn p_yes(response: &AggLlmResponse) -> Option<f64> {
    let alternatives = response
        .preservation
        .responses
        .get(&FormatId::from(WireFormat::OpenAiChat))?
        .pointer("/choices/0/logprobs/content/0/top_logprobs")?
        .as_array()?;
    let (mut yes, mut no) = (None, None);
    for entry in alternatives {
        let Some(probability) = entry.get("logprob").and_then(Value::as_f64) else {
            continue;
        };
        let token = entry.get("token")?.as_str()?;
        let slot = match token
            .trim()
            .trim_matches(['"', '\'', '.', ','])
            .to_lowercase()
            .as_str()
        {
            "yes" | "y" | "true" => &mut yes,
            "no" | "n" | "false" => &mut no,
            _ => continue,
        };
        *slot.get_or_insert(0.0) += probability.exp();
    }
    match (yes, no) {
        (Some(yes), Some(no)) if yes + no > 0.0 => Some(yes / (yes + no)),
        (Some(_), None) => Some(1.0),
        (None, Some(_)) => Some(0.0),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scored(alternatives: Value) -> Option<f64> {
        let mut response = AggLlmResponse::default();
        response.preservation.responses.insert(
            FormatId::from(WireFormat::OpenAiChat),
            json!({"choices": [{"logprobs": {"content": [{"top_logprobs": alternatives}]}}]}),
        );
        p_yes(&response)
    }

    #[test]
    fn verdict_mass_is_scored_against_the_pair() {
        let half = 0.5_f64.ln();
        let score = scored(json!([
            {"token": " Yes", "logprob": half},
            {"token": "no", "logprob": half},
            {"token": ".", "logprob": -9.0}
        ]));
        assert!(score.is_some_and(|score| (score - 0.5).abs() < 1e-9));
        assert_eq!(
            scored(json!([{"token": "yes", "logprob": -3.0}])),
            Some(1.0)
        );
        assert_eq!(scored(json!([{"token": "maybe", "logprob": 0.0}])), None);
        assert_eq!(p_yes(&AggLlmResponse::default()), None);
    }
}
