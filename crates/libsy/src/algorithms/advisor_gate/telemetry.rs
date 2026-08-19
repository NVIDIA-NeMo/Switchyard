// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Gate metrics and the benchmark audit lines.

use opentelemetry::KeyValue;
use switchyard_protocol::Usage;

use crate::observability;

// ── Accounting ──────────────────────────────────────────────────────────────

/// Inclusive prompt tokens: non-cached input plus both cache buckets, the
/// same fold the routing log uses, so advisor and executor rows reconcile.
pub(super) fn inclusive_prompt_tokens(usage: &Usage) -> u64 {
    usage
        .input_tokens
        .unwrap_or(0)
        .saturating_add(usage.cached_input_tokens().unwrap_or(0))
        .saturating_add(usage.cache_creation_input_tokens().unwrap_or(0))
}

pub(super) fn record_review(verdict: &'static str, trigger: &'static str) {
    observability::meter()
        .u64_counter("switchyard.advisor_gate.reviews")
        .build()
        .add(
            1,
            &[
                KeyValue::new("verdict", verdict),
                KeyValue::new("trigger", trigger),
            ],
        );
}

pub(super) fn record_consult_failure(reason: &'static str) {
    observability::meter()
        .u64_counter("switchyard.advisor_gate.consult_failures")
        .build()
        .add(1, &[KeyValue::new("reason", reason)]);
}

/// Counts a REDO-discarded executor turn and its tokens; the client never
/// sees the turn, so the host's terminal usage accounting never prices it.
pub(super) fn record_discarded(usage: &Usage) {
    let meter = observability::meter();
    meter
        .u64_counter("switchyard.advisor_gate.discarded_turns")
        .build()
        .add(1, &[]);
    let tokens = meter
        .u64_counter("switchyard.advisor_gate.discarded_tokens")
        .build();
    for (kind, value) in [
        ("input", usage.input_tokens.unwrap_or(0)),
        ("cached", usage.cached_input_tokens().unwrap_or(0)),
        (
            "cache_creation",
            usage.cache_creation_input_tokens().unwrap_or(0),
        ),
        ("output", usage.output_tokens.unwrap_or(0)),
    ] {
        if value > 0 {
            tokens.add(value, &[KeyValue::new("kind", kind)]);
        }
    }
}

/// One review consult's audit payload.
pub(super) struct ReviewAudit<'a> {
    pub(super) verdict: &'static str,
    pub(super) error: Option<String>,
    pub(super) latency_ms: f64,
    pub(super) reply_head: Option<String>,
    pub(super) usage: Option<&'a Usage>,
}

/// Emits the one-line sorted-key JSON audit record benchmark tooling greps
/// for (`advisor_review=`).
pub(super) fn emit_review_audit(audit: ReviewAudit<'_>) {
    let mut payload = serde_json::Map::new();
    payload.insert("advisor_review".to_string(), true.into());
    payload.insert(
        "latency_ms".to_string(),
        ((audit.latency_ms * 10.0).round() / 10.0).into(),
    );
    payload.insert("verdict".to_string(), audit.verdict.into());
    if let Some(error) = audit.error {
        payload.insert("error".to_string(), error.into());
    }
    if let Some(head) = audit.reply_head
        && !head.is_empty()
    {
        payload.insert("reply_head".to_string(), head.into());
    }
    if let Some(usage) = audit.usage {
        payload.insert(
            "prompt_tokens".to_string(),
            inclusive_prompt_tokens(usage).into(),
        );
        payload.insert(
            "completion_tokens".to_string(),
            usage.output_tokens.unwrap_or(0).into(),
        );
        let cached = usage.cached_input_tokens().unwrap_or(0);
        if cached > 0 {
            payload.insert("cached_tokens".to_string(), cached.into());
        }
        let creation = usage.cache_creation_input_tokens().unwrap_or(0);
        if creation > 0 {
            payload.insert("cache_creation_tokens".to_string(), creation.into());
        }
    }
    tracing::info!(
        target: "libsy",
        "advisor_review={}",
        serde_json::Value::Object(payload)
    );
}

/// Emits the discarded-turn audit record (`advisor_discarded=`), the gate's
/// own accounting for a turn no host-side observer can price.
pub(super) fn emit_discarded_audit(model: &str, usage: &Usage) {
    let mut payload = serde_json::Map::new();
    payload.insert("advisor_discarded".to_string(), true.into());
    payload.insert("model".to_string(), model.into());
    payload.insert(
        "prompt_tokens".to_string(),
        inclusive_prompt_tokens(usage).into(),
    );
    payload.insert(
        "cached_tokens".to_string(),
        usage.cached_input_tokens().unwrap_or(0).into(),
    );
    payload.insert(
        "cache_creation_tokens".to_string(),
        usage.cache_creation_input_tokens().unwrap_or(0).into(),
    );
    payload.insert(
        "completion_tokens".to_string(),
        usage.output_tokens.unwrap_or(0).into(),
    );
    tracing::info!(
        target: "libsy",
        "advisor_discarded={}",
        serde_json::Value::Object(payload)
    );
}
