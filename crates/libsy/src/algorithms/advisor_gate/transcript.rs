// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Serializing the conversation for the advisor and parsing its verdict.

use switchyard_protocol::{AggLlmResponse, Message};

use super::turn::visible_text;

/// Splices the two surviving ends of an over-cap transcript.
pub(super) const TRUNCATION_MARKER: &str = "\n...<middle of the conversation truncated>...\n";
/// Stands in for a terminal turn with no reviewable text at all.
pub(super) const NO_TEXT_PLACEHOLDER: &str = "(no text)";
/// Anchored verdict parse: optional wrapper characters and an optional
/// "(final) verdict:" label, then APPROVE or REDO as the first real word.
/// Anchoring matters — an unanchored scan turns "I cannot approve this —
/// REDO: run the tests" into APPROVE.
pub(super) const VERDICT_PATTERN: &str =
    r#"(?i)^[\s*_#>"'(\[`]*(?:(?:final\s+)?verdict\s*:\s*[\s*_#>"'(\[`]*)?(APPROVE|REDO)\b"#;

/// Advisor verdict on one terminal turn.
pub(super) enum Verdict {
    Approve,
    Redo { plan: String },
}

// ── Transcript and verdict ──────────────────────────────────────────────────

/// Serializes the conversation for the advisor. The JSON body is capped with
/// a middle drop — the head keeps the task statement, the tail keeps the
/// recent evidence a completeness review is about — while the terminal turn
/// is appended uncapped. `cap` is the route's `transcript_max_chars`: a
/// character budget on the serialized JSON (~4 chars per token, so the 200k
/// default is ~50k tokens of advisor input).
pub(super) fn review_transcript(
    messages: &[Message],
    review_tail: Option<&str>,
    cap: usize,
) -> String {
    let text = serde_json::to_string(messages).unwrap_or_default();
    let text = middle_drop(text, cap);
    format!(
        "Conversation so far (JSON):\n\n{text}\n\nThe executor's latest turn (a plan, or its claim the task is done):\n{}",
        review_tail.unwrap_or(NO_TEXT_PLACEHOLDER)
    )
}

/// Keeps the first `cap / 4` and last `cap - cap / 4` characters of an
/// over-cap string, splicing [`TRUNCATION_MARKER`] between them. Boundaries
/// are computed per character so multi-byte text never splits a code point.
pub(super) fn middle_drop(text: String, cap: usize) -> String {
    let total = text.chars().count();
    if total <= cap {
        return text;
    }
    let head_chars = cap / 4;
    let tail_chars = cap - head_chars;
    let head_end = text
        .char_indices()
        .nth(head_chars)
        .map(|(index, _)| index)
        .unwrap_or(text.len());
    let tail_start = text
        .char_indices()
        .nth(total - tail_chars)
        .map(|(index, _)| index)
        .unwrap_or(0);
    format!(
        "{}{TRUNCATION_MARKER}{}",
        &text[..head_end],
        &text[tail_start..]
    )
}

/// Text of the advisor's reply: all text blocks across outputs, trimmed.
pub(super) fn advisor_reply_text(agg: &AggLlmResponse) -> String {
    visible_text(agg).unwrap_or_default().trim().to_string()
}

/// Parses the anchored verdict. A REDO's plan is the remainder after the
/// verdict token with leading separators stripped; an empty plan falls back
/// to the whole reply so the executor still gets actionable feedback. `None`
/// means the reply led with prose and cannot be trusted as a verdict.
pub(super) fn parse_verdict(verdict_re: &regex::Regex, reply: &str) -> Option<Verdict> {
    let reply = reply.trim();
    let captures = verdict_re.captures(reply)?;
    let token = captures.get(1)?;
    if token.as_str().eq_ignore_ascii_case("APPROVE") {
        return Some(Verdict::Approve);
    }
    let plan = reply[token.end()..]
        .trim_start_matches([' ', '*', '_', ':', '\n', '-'])
        .trim();
    let plan = if plan.is_empty() { reply } else { plan };
    Some(Verdict::Redo {
        plan: plan.to_string(),
    })
}
