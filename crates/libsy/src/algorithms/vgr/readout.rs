// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The cheap readout: a few tokens of output, scored by token probability.
//!
//! The readout asks a verifier a yes/no question and reads the *probability* it
//! assigned to "yes" rather than the word it happened to emit. That turns a
//! four-token completion into a graded score, which is what makes it cheap
//! enough to run before any of the costlier rungs.
//!
//! # Why this reads raw provider JSON
//!
//! The normalized response types carry no token probabilities: the decoder
//! keeps `choices[].logprobs` out of [`AggLlmResponse`] entirely. What it does
//! keep, under the default preservation policy, is the provider's original
//! response body — so the probabilities are recoverable from there, and only
//! from there. This module is the single place that dependency lives; if
//! typed logprob support is added later, only [`p_yes`] changes.
//!
//! # Failure is silence, never noise
//!
//! A target that does not support logprobs, a streamed response, a provider
//! whose body is shaped differently, a malformed payload — all yield `None`.
//! The decision rules already treat an absent readout as evidence that never
//! commits, so a deployment that cannot produce readouts degrades to its
//! costlier rungs rather than misreading them.

use serde_json::{Value, json};
use switchyard_protocol::{AggLlmResponse, FormatId, Request, WireFormat};

/// How many alternatives to request at the scored token position.
///
/// The verdict is one token, so a handful of alternatives is enough to find
/// both "yes" and "no" among them.
const TOP_ALTERNATIVES: u64 = 8;

/// Completion budget for a readout call.
///
/// The prompt demands a single word; the few extra tokens absorb a leading
/// space or a stray punctuation mark without paying for a sentence.
pub(super) const MAX_OUTPUT_TOKENS: u64 = 4;

/// Asks the provider for a direct verdict and its token probabilities.
///
/// Disabling reasoning is request-local: the serving model remains free to
/// think on ordinary agent calls, while the four-token verifier budget reaches
/// `yes` or `no`. Both logprob fields are required together.
pub(super) fn request_logprobs(request: &mut Request) {
    request.llm_request.reasoning.effort = Some("none".to_string());
    let extensions = &mut request.llm_request.extensions.fields;
    extensions.insert("logprobs".to_string(), json!(true));
    extensions.insert("top_logprobs".to_string(), json!(TOP_ALTERNATIVES));
}

/// The probability the verifier assigned to "yes" at its first token.
///
/// Returns `None` whenever the answer cannot be established — see the module
/// documentation on why that is deliberately indistinguishable from an
/// unavailable verifier.
pub(super) fn p_yes(response: &AggLlmResponse) -> Option<f64> {
    let body = response
        .preservation
        .responses
        .get(&FormatId::from(WireFormat::OpenAiChat))?;
    let alternatives = body
        .pointer("/choices/0/logprobs/content/0/top_logprobs")?
        .as_array()?;

    // The two verdicts are scored against each other rather than against the
    // whole distribution, so probability mass on unrelated tokens — whitespace,
    // punctuation, a stray capitalization — cannot dilute the verdict.
    let mut yes = None;
    let mut no = None;
    for entry in alternatives {
        let Some(probability) = entry.get("logprob").and_then(Value::as_f64) else {
            continue;
        };
        let probability = probability.exp();
        match normalize(entry.get("token")?.as_str()?) {
            Verdict::Yes => *yes.get_or_insert(0.0) += probability,
            Verdict::No => *no.get_or_insert(0.0) += probability,
            Verdict::Other => {}
        }
    }

    match (yes, no) {
        // Both verdicts present: the score is yes against the pair.
        (Some(yes), Some(no)) if yes + no > 0.0 => Some(yes / (yes + no)),
        // Normalize over observed verdict mass, matching the reference: one
        // observed verdict receives the whole binary mass.
        (Some(_), None) => Some(1.0),
        (None, Some(_)) => Some(0.0),
        _ => None,
    }
}

/// Which verdict, if either, a reported token stands for.
enum Verdict {
    Yes,
    No,
    Other,
}

/// Classifies a reported token, ignoring the framing a tokenizer adds.
///
/// Providers report the token as emitted, so the same verdict arrives variously
/// as `yes`, `Yes`, or a space-prefixed form depending on the tokenizer.
fn normalize(token: &str) -> Verdict {
    let token = token
        .trim()
        .trim_matches(['"', '\'', '.', ','])
        .to_lowercase();
    match token.as_str() {
        "yes" | "y" | "true" => Verdict::Yes,
        "no" | "n" | "false" => Verdict::No,
        _ => Verdict::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use switchyard_protocol::{LlmRequest, PreservationMetadata};

    /// An aggregate carrying a preserved provider body with the given alternatives.
    fn response_with(alternatives: Value) -> AggLlmResponse {
        let mut preservation = PreservationMetadata::default();
        preservation.responses.insert(
            FormatId::from(WireFormat::OpenAiChat),
            json!({"choices": [{"logprobs": {"content": [{"top_logprobs": alternatives}]}}]}),
        );
        AggLlmResponse {
            preservation,
            ..Default::default()
        }
    }

    /// A reported alternative at the given natural probability.
    fn alternative(token: &str, probability: f64) -> Value {
        json!({"token": token, "logprob": probability.ln()})
    }

    #[test]
    fn scores_yes_against_no_rather_than_the_whole_distribution() {
        // Mass on unrelated tokens must not dilute the verdict.
        let response = response_with(json!([
            alternative("yes", 0.6),
            alternative("no", 0.2),
            alternative("\n", 0.2),
        ]));
        assert!(p_yes(&response).is_some_and(|score| (score - 0.75).abs() < 1e-9));
    }

    #[test]
    fn reads_the_verdict_through_tokenizer_framing() {
        // The same verdict arrives capitalized or space-prefixed by tokenizer.
        for token in [" Yes", "YES", "yes.", "\"yes\""] {
            let response = response_with(json!([alternative(token, 0.9), alternative("no", 0.1)]));
            assert!(
                p_yes(&response).is_some_and(|score| (score - 0.9).abs() < 1e-9),
                "{token}"
            );
        }
    }

    #[test]
    fn a_lone_verdict_receives_the_whole_observed_binary_mass() {
        let only_yes = response_with(json!([alternative("yes", 0.97)]));
        assert_eq!(p_yes(&only_yes), Some(1.0));
        let only_no = response_with(json!([alternative("no", 0.95)]));
        assert_eq!(p_yes(&only_no), Some(0.0));
    }

    #[test]
    fn anything_unreadable_scores_nothing_rather_than_guessing() {
        // No preserved body at all.
        assert_eq!(p_yes(&AggLlmResponse::default()), None);
        // A preserved body with no probabilities in it.
        let mut preservation = PreservationMetadata::default();
        preservation.responses.insert(
            FormatId::from(WireFormat::OpenAiChat),
            json!({"choices": [{"message": {"content": "yes"}}]}),
        );
        let bare = AggLlmResponse {
            preservation,
            ..Default::default()
        };
        assert_eq!(p_yes(&bare), None);
        // Alternatives that contain neither verdict.
        let neither = response_with(json!([alternative("maybe", 0.9)]));
        assert_eq!(p_yes(&neither), None);
    }

    #[test]
    fn readout_requests_direct_verdict_logprobs() {
        // Thinking can consume the four-token budget before the verdict. Keep
        // ordinary model calls unchanged while making this request answer directly.
        let mut request = Request {
            llm_request: LlmRequest::default(),
            raw_request: None,
            metadata: None,
        };
        request_logprobs(&mut request);
        assert_eq!(
            request.llm_request.reasoning.effort.as_deref(),
            Some("none")
        );
        let fields = &request.llm_request.extensions.fields;
        assert_eq!(fields.get("logprobs"), Some(&json!(true)));
        assert_eq!(fields.get("top_logprobs"), Some(&json!(TOP_ALTERNATIVES)));
    }
}
