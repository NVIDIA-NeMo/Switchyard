// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure text primitives shared by derivation and the judged views.
//!
//! Budgets and clipping count characters, not bytes.

use std::{borrow::Cow, sync::LazyLock};

use regex::Regex;
use serde_json::Value;
use switchyard_protocol::{ContentBlock, InstructionBlock, Message, Request, Role, ToolResult};

use crate::algorithms::util::affinity::is_user_turn;
use crate::algorithms::util::tool_signals::classify_text;

/// One flattened turn: instructions first, then messages.
pub(super) struct Turn {
    pub(super) role: Role,
    pub(super) text: String,
}

// API keys, AWS access keys, bearer tokens, and email addresses.
// The 16-character key/token minimum avoids matching short ordinary identifiers.
static REDACT_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    [
        r"\b(sk|rk|pk)-[A-Za-z0-9_\-]{16,}\b",
        r"\bAKIA[0-9A-Z]{16}\b",
        r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}",
        r"\b[\w.+-]+@[\w-]+\.[\w.]+\b",
    ]
    .iter()
    .map(|pattern| Regex::new(pattern).expect("valid redaction pattern"))
    .collect()
});

// Heuristic code/test markers: fences, shell commands, diffs, tracebacks, and test counts.
static CODE_ACTIVITY: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(concat!(
        r"(?m)```|^\$ |^diff --git|^@@ |",
        r"\bTraceback \(most recent call last\)|\b\d+ (?:passed|failed)\b|",
        r"^(?:\$ |> )?(?:pytest|unittest|npm (?:test|run)|cargo (?:test|build)|",
        r"go test|make test)\b",
    ))
    .ok()
});

// Tool transcript labels and JSON tool-call fields.
static TOOL_ACTIVITY: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(r#"(?m)^\[tool\b|^\[[^\]\n]* calls tools?:|"tool_calls?"\s*:"#).ok()
});

/// Text of a content sequence, and whether it carried media or unknown blocks.
/// Tool and reasoning blocks are skipped; unsupported blocks make derivation fail closed.
pub(super) fn content_text(content: &[ContentBlock]) -> (String, bool) {
    let mut parts: Vec<&str> = Vec::new();
    let mut unsupported = false;
    for block in content {
        match block {
            ContentBlock::Text { text } | ContentBlock::Refusal { text } => parts.push(text),
            ContentBlock::ToolCall(_)
            | ContentBlock::ToolResult(_)
            | ContentBlock::Reasoning { .. } => {}
            _ => unsupported = true,
        }
    }
    (parts.join("\n"), unsupported)
}

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

/// The latest non-empty user turn.
pub(super) fn user_task_text(turns: &[Turn]) -> &str {
    turns
        .iter()
        .rev()
        .find(|turn| turn.role == Role::User && !turn.text.trim().is_empty())
        .map(|turn| turn.text.as_str())
        .unwrap_or_default()
}

/// Whether the newest user message is empty, ignoring whitespace and zero-width characters.
///
/// Any non-text block, such as an image or a tool result, is content.
pub(super) fn newest_prompt_is_blank(request: &Request) -> bool {
    request
        .llm_request
        .messages
        .iter()
        .rev()
        .find(|message| message.role == Role::User)
        .is_none_or(|message| {
            message.content.iter().all(|block| match block {
                ContentBlock::Text { text } => text.chars().all(|c| {
                    c.is_whitespace()
                        || matches!(c, '\u{200B}'..='\u{200D}' | '\u{2060}' | '\u{FEFF}')
                }),
                _ => false,
            })
        })
}

pub(super) fn has_assistant_turn(turns: &[Turn]) -> bool {
    turns.iter().any(|turn| turn.role == Role::Assistant)
}

/// Whether the conversation contains an executed or proposed tool step.
pub(super) fn has_tool_trajectory(request: &Request) -> bool {
    request.llm_request.messages.iter().any(is_tool_step)
}

/// Whether the user message at `index` answers a finished assistant reply.
///
/// Agent clients also inject user-role continuations (after context compaction,
/// at an iteration limit) that follow tool results, not a final reply; those
/// continue the current user turn.
pub(super) fn starts_user_turn(messages: &[Message], index: usize) -> bool {
    let (Some(message), Some(previous)) = (
        messages.get(index),
        index.checked_sub(1).and_then(|i| messages.get(i)),
    ) else {
        return false;
    };
    is_user_turn(message)
        && previous.role == Role::Assistant
        && !previous
            .content
            .iter()
            .any(|block| matches!(block, ContentBlock::ToolCall(_)))
}

/// Whether the current user turn contains an executed or proposed tool step.
pub(super) fn turn_has_tool_trajectory(request: &Request) -> bool {
    let messages = &request.llm_request.messages;
    let start = (0..messages.len())
        .rev()
        .find(|&index| starts_user_turn(messages, index))
        .unwrap_or(0);
    messages[start..].iter().any(is_tool_step)
}

fn is_tool_step(message: &Message) -> bool {
    message.role == Role::Tool
        || message.content.iter().any(|block| {
            matches!(
                block,
                ContentBlock::ToolCall(_) | ContentBlock::ToolResult(_)
            )
        })
}

/// Idempotent secret and PII scrub applied before text reaches a verifier.
pub(super) fn redact(text: &str) -> String {
    let mut out = text.to_string();
    for pattern in REDACT_PATTERNS.iter() {
        if let Cow::Owned(redacted) = pattern.replace_all(&out, "[REDACTED]") {
            out = redacted;
        }
    }
    out
}

pub(super) fn char_len(text: &str) -> usize {
    text.chars().count()
}

/// Head-and-tail clip to `budget` characters, keeping `head_frac` as head.
pub(super) fn clip_mid(text: &str, budget: usize, head_frac: f64) -> String {
    let len = char_len(text);
    if len <= budget {
        return text.to_string();
    }
    let head = ((budget as f64 * head_frac) as usize).max(1).min(budget);
    let tail = budget.saturating_sub(head);
    let omitted = len - budget;
    let head_end = text
        .char_indices()
        .nth(head)
        .map_or(text.len(), |(index, _)| index);
    let tail_start = if tail == 0 {
        text.len()
    } else {
        text.char_indices()
            .rev()
            .nth(tail - 1)
            .map_or(0, |(index, _)| index)
    };
    let head_text = &text[..head_end];
    let tail_text = &text[tail_start..];
    format!("{head_text}\n...[{omitted} chars omitted]...\n{tail_text}")
}

/// Detects heuristic code/test markers in the local attempt.
pub(super) fn observed_hardening(attempt: &str) -> bool {
    CODE_ACTIVITY
        .as_ref()
        .is_some_and(|pattern| pattern.is_match(attempt))
}

/// Detects tool transcript labels and JSON tool-call fields in the local attempt.
pub(super) fn observed_tool_activity(attempt: &str) -> bool {
    TOOL_ACTIVITY
        .as_ref()
        .is_some_and(|pattern| pattern.is_match(attempt))
}

/// The router's own summary of every tool result in the conversation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct ToolRecord {
    pub(super) errors: i32,
    pub(super) results: i32,
    /// Consecutive clean results ending the conversation.
    pub(super) clean_tail: i32,
}

/// Uses executor status and inferred failures, with a successful exit taking precedence.
pub(super) fn tool_result_failed(result: &ToolResult) -> bool {
    let text = result
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } | ContentBlock::Refusal { text } => Some(text),
            _ => None,
        })
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join("\n");
    let body = serde_json::from_str::<Value>(&text).ok();
    let object = body.as_ref().and_then(Value::as_object);
    // The executor's own `exit_code: 0` overrules every inferred failure:
    // harnesses send `"error": null` on success, and negative checks print
    // error-shaped text.
    let exit_ok = object
        .and_then(|object| object.get("exit_code"))
        .and_then(Value::as_i64)
        == Some(0);
    (result.is_error == Some(true)
        || object.is_some_and(|object| object.contains_key("error"))
        || classify_text(&text).0 > 0.0)
        && !exit_ok
}

impl ToolRecord {
    pub(super) fn from_request(request: &Request) -> Self {
        let mut record = Self::default();
        let results = request
            .llm_request
            .messages
            .iter()
            .flat_map(|message| &message.content)
            .filter_map(|block| match block {
                ContentBlock::ToolResult(result) => Some(result),
                _ => None,
            });
        for result in results {
            let failed = tool_result_failed(result);
            record.results = record.results.saturating_add(1);
            record.errors = record.errors.saturating_add(i32::from(failed));
            record.clean_tail = if failed {
                0
            } else {
                record.clean_tail.saturating_add(1)
            };
        }
        record
    }
}
