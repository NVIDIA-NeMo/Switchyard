// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical client error re-export and shared context-window-overflow detection.
//!
//! The overflow detection is ported from
//! `switchyard-components/src/backends/context_overflow.rs`; that helper is
//! crate-private there and this crate cannot depend on `switchyard-components`,
//! so the small, self-contained logic is vendored here.

use serde_json::Value;

pub use switchyard_protocol::LlmClientError;

/// Result alias for LLM client operations.
pub type Result<T> = std::result::Result<T, LlmClientError>;

/// Detects a context-overflow body using a provider-supplied structured check
/// and a substring phrase list.
///
/// Parses the body once, runs the structured check (e.g. against `error.code`),
/// then falls back to matching phrases against `error.message` or — when the
/// body is not JSON — the raw body. Centralizing the shape means each new
/// provider-wrap of the canonical error is a one-line phrase entry, not a fork
/// of the parsing logic.
pub(crate) fn is_overflow_body<F>(body: &str, structured_check: F, phrases: &[&str]) -> bool
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
        {
            if contains_any(message, phrases) {
                return true;
            }
        }
    }
    // Some upstream proxies return plain-text bodies; fall through to a string
    // match on the raw body.
    contains_any(body, phrases)
}

// Case-insensitive substring match of any phrase against the message.
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
            &[],
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
