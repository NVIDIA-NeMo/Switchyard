// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared detection for upstream 4xx context-window-overflow bodies.
//!
//! Provider-specific detectors (`openai`, `anthropic`) supply a structured
//! check (against the parsed error envelope) and a list of substring phrases.
//! The common shell parses the body once, runs the structured check, then
//! falls back to matching phrases against `error.message` or — if the body
//! isn't JSON — the raw body. Centralising the shape here means each new
//! provider-wrap (e.g. NVIDIA/LiteLLM's wrapping of the canonical OpenAI
//! error) is one-line phrase entry per provider, not a duplicated rewrite.

use serde_json::Value;

/// Detect a context-overflow body using a provider-supplied structured check
/// and substring phrase list. See module docs for the matching strategy.
pub(super) fn is_overflow_body<F>(body: &str, structured_check: F, phrases: &[&str]) -> bool
where
    F: Fn(&Value) -> bool,
{
    if let Ok(value) = serde_json::from_str::<Value>(body) {
        if structured_check(&value) {
            return true;
        }
        if let Some(message) = value
            .get("error")
            .and_then(|err| err.get("message"))
            .and_then(Value::as_str)
            && contains_any(message, phrases)
        {
            return true;
        }
    }
    // Some upstream proxies return plain-text bodies; fall through to a
    // string match on the raw body.
    contains_any(body, phrases)
}

fn contains_any(message: &str, phrases: &[&str]) -> bool {
    let lower = message.to_ascii_lowercase();
    phrases.iter().any(|phrase| lower.contains(phrase))
}

#[cfg(test)]
mod tests {
    use super::*;

    const PHRASES: &[&str] = &["context window", "too long"];

    fn never(_value: &Value) -> bool {
        false
    }

    #[test]
    fn structured_check_short_circuits() {
        let body = r#"{"error":{"code":"context_length_exceeded","message":"unrelated"}}"#;
        let matched = is_overflow_body(
            body,
            |value| {
                value
                    .get("error")
                    .and_then(|err| err.get("code"))
                    .and_then(Value::as_str)
                    == Some("context_length_exceeded")
            },
            &[], // empty phrases — structured check is the only path
        );
        assert!(matched);
    }

    #[test]
    fn falls_back_to_message_phrase_match() {
        let body = r#"{"error":{"message":"prompt too long"}}"#;
        assert!(is_overflow_body(body, never, PHRASES));
    }

    #[test]
    fn matches_plain_text_body() {
        assert!(is_overflow_body(
            "plain text mentioning context window",
            never,
            PHRASES
        ));
    }

    #[test]
    fn non_match_returns_false() {
        let body = r#"{"error":{"message":"rate limit exceeded"}}"#;
        assert!(!is_overflow_body(body, never, PHRASES));
    }
}
