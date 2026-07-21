// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Error type for the LLM client, plus shared context-window-overflow detection.
//!
//! The overflow detection is ported from
//! `switchyard-components/src/backends/context_overflow.rs`; that helper is
//! crate-private there and this crate cannot depend on `switchyard-components`,
//! so the small, self-contained logic is vendored here.

use serde_json::Value;
use switchyard_translation::WireFormat;
use thiserror::Error;

/// Result alias for LLM client operations.
pub type Result<T> = std::result::Result<T, LlmClientError>;

/// Failures surfaced while resolving, translating, sending, or decoding a call.
#[derive(Debug, Error)]
pub enum LlmClientError {
    /// The resolved model name has no entry in the client's backend map.
    #[error("no backend configured for model {0:?}")]
    UnknownModel(String),

    /// The model exists but has no backend configured for the requested format.
    #[error("model {model:?} has no backend for format {format}")]
    UnknownModelFormat {
        /// Model that was resolved.
        model: String,
        /// Wire format requested for the call.
        format: WireFormat,
    },

    /// Neither an explicit `model_name` nor a model on the request was given.
    #[error("no model given: pass model_name or set request.llm_request.model")]
    MissingModel,

    /// Request encoding or response decoding failed in the translation engine.
    #[error("translation failed: {0}")]
    Translation(String),

    /// The HTTP request could not be sent (connect/timeout/transport failure).
    #[error("upstream transport error: {0}")]
    Transport(String),

    /// An upstream 400 whose body is detected as a context-window overflow.
    ///
    /// Kept distinct from [`LlmClientError::UpstreamHttp`] and checked first so a
    /// caller can implement evict-and-retry by matching this variant instead of
    /// sniffing error strings.
    #[error("context window exceeded for model {model}: {message}")]
    ContextWindowExceeded {
        /// Model that overflowed, as sent upstream.
        model: String,
        /// Raw upstream error body.
        message: String,
    },

    /// A non-2xx upstream response that is not a context-window overflow.
    #[error("upstream returned HTTP {status}: {body}")]
    UpstreamHttp {
        /// Upstream HTTP status code.
        status: u16,
        /// Raw upstream error body.
        body: String,
    },

    /// A streamed response failed mid-flight (read error or malformed frame).
    #[error("stream error: {0}")]
    Stream(String),
}

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
