// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Text primitives shared by capability derivation and the judged views.
//!
//! Everything here is pure and deterministic: content extraction from normalized
//! messages, baseline redaction, character-budgeted clipping, and the two
//! detectors that read locally produced attempt text for code or tool activity.
//!
//! Budgets and clipping are measured in **characters**, not bytes, so that a
//! multi-byte conversation is budgeted the same way an ASCII one is.

use regex::Regex;
use std::sync::LazyLock;
use switchyard_protocol::{ContentBlock, InstructionBlock, Message, Request, Role};

/// One conversation turn flattened for derivation: a role and its extracted text.
///
/// Instruction blocks and messages are flattened into a single ordered sequence
/// because the reference implementation carries system instructions in band with
/// the conversation, and both completeness gates count system turns.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Turn {
    /// Actor that produced the turn.
    pub(super) role: Role,
    /// Extracted, not-yet-redacted text of the turn.
    pub(super) text: String,
}

/// Baseline secret and PII patterns scrubbed before any text is rendered.
static REDACT_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    [
        r"\b(sk|rk|pk)-[A-Za-z0-9_\-]{16,}\b",
        r"\bAKIA[0-9A-Z]{16}\b",
        r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}",
        r"\b[\w.+-]+@[\w-]+\.[\w.]+\b",
    ]
    .iter()
    .filter_map(|pattern| Regex::new(pattern).ok())
    .collect()
});

/// Locally produced evidence that the attempt did code or test work.
///
/// Activity forms only — a line-anchored runner invocation, a fenced block, a
/// diff header, test counts, a traceback. Prose mentioning a test runner never
/// matches.
static CODE_ACTIVITY: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(concat!(
        r"(?m)```|^\$ |^diff --git|^@@ |",
        r"\bTraceback \(most recent call last\)|\b\d+ (?:passed|failed)\b|",
        r"^(?:\$ |> )?(?:pytest|unittest|npm (?:test|run)|cargo (?:test|build)|",
        r"go test|make test)\b",
    ))
    .ok()
});

/// Locally produced evidence that the attempt operated tools.
///
/// Line-anchored bracketed tool records or a quoted `tool_calls` JSON key —
/// never prose tool vocabulary.
static TOOL_ACTIVITY: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(r#"(?m)^\[tool\b|^\[[^\]\n]* calls tools?:|"tool_calls?"\s*:"#).ok()
});

/// Extracts the text of one content sequence, and whether it carried a part the
/// router cannot faithfully judge.
///
/// Text and refusal blocks yield their text explicitly joined. Media and
/// unrecognized provider blocks flag the content unsupported, which fails
/// derivation closed. Tool calls, tool results, and reasoning blocks are
/// structural rather than task-bearing: they contribute no text and do not flag
/// the content unsupported, because request-side tool structure is untrusted
/// input that derivation ignores in any case.
pub(super) fn content_text(content: &[ContentBlock]) -> (String, bool) {
    let mut parts: Vec<&str> = Vec::new();
    let mut unsupported = false;
    for block in content {
        match block {
            ContentBlock::Text { text } | ContentBlock::Refusal { text } => parts.push(text),
            ContentBlock::ToolCall(_) | ContentBlock::ToolResult(_) => {}
            ContentBlock::Reasoning { .. } => {}
            _ => unsupported = true,
        }
    }
    (parts.join("\n"), unsupported)
}

/// Flattens a request's instruction blocks and messages into one ordered turn
/// sequence, reporting whether any turn carried unsupported content.
///
/// Instructions render first, as system turns, matching the reference
/// implementation's in-band ordering.
pub(super) fn turns(request: &Request) -> (Vec<Turn>, bool) {
    let mut flattened = Vec::new();
    let mut unsupported = false;

    let mut push = |role: Role, content: &[ContentBlock]| {
        let (text, block_unsupported) = content_text(content);
        unsupported |= block_unsupported;
        flattened.push(Turn { role, text });
    };

    for InstructionBlock { role, content } in &request.llm_request.instructions {
        push(*role, content);
    }
    for Message { role, content } in &request.llm_request.messages {
        push(*role, content);
    }

    (flattened, unsupported)
}

/// The latest non-empty user turn — the only outside datum branching consults.
///
/// Returns an empty string when the request carries no user text, which fails
/// derivation closed.
pub(super) fn user_task_text(turns: &[Turn]) -> String {
    turns
        .iter()
        .rev()
        .find(|turn| turn.role == Role::User && !turn.text.trim().is_empty())
        .map(|turn| turn.text.clone())
        .unwrap_or_default()
}

/// True when the request continues a conversation rather than opening one.
///
/// The answer branch's instruments see only the latest user message, so on a
/// multi-turn request they are invalid rather than merely weaker.
pub(super) fn has_assistant_turn(turns: &[Turn]) -> bool {
    turns.iter().any(|turn| turn.role == Role::Assistant)
}

/// Whether the normalized conversation contains an executed or proposed tool step.
pub(super) fn has_tool_trajectory(request: &Request) -> bool {
    request.llm_request.messages.iter().any(|message| {
        message.role == Role::Tool
            || message.content.iter().any(|block| {
                matches!(
                    block,
                    ContentBlock::ToolCall(_) | ContentBlock::ToolResult(_)
                )
            })
    })
}

/// Baseline secret and PII scrub applied before any text leaves the device.
///
/// Idempotent: rendering applies it once per section and the gates measure
/// post-redaction lengths, so a second application must not grow the text.
pub(super) fn redact(text: &str) -> String {
    let mut out = text.to_string();
    for pattern in REDACT_PATTERNS.iter() {
        out = pattern.replace_all(&out, "[REDACTED]").into_owned();
    }
    out
}

/// Counts characters, the unit every render budget and completeness gate uses.
pub(super) fn char_len(text: &str) -> usize {
    text.chars().count()
}

/// Head-and-tail clip to `budget` characters, keeping `head_frac` of the budget
/// as the head and noting how much was dropped.
///
/// `head_frac` below `1.0` weights the surviving text toward the tail, which is
/// what an attempt's final output needs.
pub(super) fn clip_mid(text: &str, budget: usize, head_frac: f64) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= budget {
        return text.to_string();
    }
    let head = ((budget as f64 * head_frac) as usize).max(1);
    let tail = budget.saturating_sub(head);
    let omitted = chars.len() - budget;
    let head_text: String = chars[..head].iter().collect();
    let tail_text: String = chars[chars.len() - tail..].iter().collect();
    format!("{head_text}\n...[{omitted} chars omitted]...\n{tail_text}")
}

/// True when locally produced attempt text shows code or test activity.
///
/// Evidence can only harden the selected branch; it never selects a weaker one,
/// so a client gains nothing by injecting or suppressing markers in the request.
pub(super) fn observed_hardening(attempt: &str) -> bool {
    CODE_ACTIVITY
        .as_ref()
        .is_some_and(|pattern| pattern.is_match(attempt))
}

/// True when locally produced attempt text shows tool activity.
pub(super) fn observed_tool_activity(attempt: &str) -> bool {
    TOOL_ACTIVITY
        .as_ref()
        .is_some_and(|pattern| pattern.is_match(attempt))
}
