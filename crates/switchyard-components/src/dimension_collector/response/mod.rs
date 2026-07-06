// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Response-side context signals.
//!
//! Sibling module to the request-side dimension collector. Same character:
//! deterministic, no estimator needed, calibration-friendly. Each
//! [`ResponseFlag`] is the binary outcome of one pure check against a
//! `ChatResponse` body.
//!
//! [`ResponseSignalCollector`] (in `request_processors/`) wraps this as a
//! response-side adapter; the pure logic lives here so it's
//! independently unit-testable and reusable by future routers that want
//! to inspect response quality without going through the full chain.

pub mod checks;

use serde::{Deserialize, Serialize};
use switchyard_core::ChatResponse;

/// Aggregate of response-side quality flags emitted by [`extract_response_signals`].
///
/// Stamped into `ProxyContext.extensions` by the
/// `ResponseSignalCollector` adapter. Empty `flags` means all checks
/// passed — the response is considered acceptable from the stage-router
/// router's reactive-escalation viewpoint.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseSignals {
    pub flags: Vec<ResponseFlag>,
}

impl ResponseSignals {
    /// Returns `true` when at least one check flagged the response.
    ///
    /// StageRouter routers consume this as the per-attempt acceptability
    /// gate; a `true` here triggers escalation to the next tier.
    pub fn has_failures(&self) -> bool {
        !self.flags.is_empty()
    }

    /// Returns `true` if the given flag is present.
    pub fn contains(&self, flag: ResponseFlag) -> bool {
        self.flags.contains(&flag)
    }
}

/// Closed set of response-side quality failures emitted by the
/// dimension-collector response layer.
///
/// Adding a new flag is a deliberate API change: downstream estimators
/// match on this enum, and growth here is observable in their behavior.
/// Keep it small.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFlag {
    /// `tool_calls[].function.arguments` doesn't parse as JSON.
    MalformedToolCallJson,
    /// No `content` and no `tool_calls`; not a legitimate
    /// `tool_calls`-finish.
    EmptyResponse,
    /// `finish_reason == "length"` (OpenAI) or
    /// `stop_reason == "max_tokens"` (Anthropic).
    TruncatedCompletion,
    /// A tool call is missing `name` or `arguments` at the shape level
    /// (not per-tool-schema; that's future work).
    MissingRequiredArgs,
}

/// Runs all four checks against a buffered `ChatResponse` and returns
/// the set of failing flags.
///
/// Streaming responses (where [`ChatResponse::body`] returns `None`)
/// short-circuit to an empty `ResponseSignals` — streams can't be
/// introspected without consuming them. A streaming-aware checker is
/// future work outside the first cut.
pub fn extract_response_signals(response: &ChatResponse) -> ResponseSignals {
    if response.body().is_none() {
        return ResponseSignals::default();
    }
    let mut flags = Vec::new();
    if checks::is_malformed_tool_call(response) {
        flags.push(ResponseFlag::MalformedToolCallJson);
    }
    if checks::is_empty_response(response) {
        flags.push(ResponseFlag::EmptyResponse);
    }
    if checks::is_truncated_completion(response) {
        flags.push(ResponseFlag::TruncatedCompletion);
    }
    if checks::is_missing_required_args(response) {
        flags.push(ResponseFlag::MissingRequiredArgs);
    }
    ResponseSignals { flags }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extract_returns_empty_for_well_formed_response() {
        let resp = ChatResponse::openai_completion(json!({
            "choices": [{
                "message": { "content": "ok" },
                "finish_reason": "stop"
            }]
        }));
        let signals = extract_response_signals(&resp);
        assert!(signals.flags.is_empty());
        assert!(!signals.has_failures());
    }

    #[test]
    fn extract_aggregates_multiple_failures() {
        // Empty content + truncated finish at the same time.
        let resp = ChatResponse::openai_completion(json!({
            "choices": [{
                "message": { "content": "" },
                "finish_reason": "length"
            }]
        }));
        let signals = extract_response_signals(&resp);
        assert!(signals.has_failures());
        assert!(signals.contains(ResponseFlag::EmptyResponse));
        assert!(signals.contains(ResponseFlag::TruncatedCompletion));
    }
}
