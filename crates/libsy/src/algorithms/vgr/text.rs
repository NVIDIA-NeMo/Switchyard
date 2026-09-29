// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure text primitives shared by derivation and the judged views.
//!
//! Budgets and clipping count characters, not bytes.

use std::sync::LazyLock;

use regex::Regex;
use serde_json::Value;
use switchyard_protocol::{ContentBlock, InstructionBlock, Message, Request, Role};

use crate::algorithms::util::tool_signals::classify_text;

/// One flattened turn: instructions first, then messages.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Turn {
    pub(super) role: Role,
    pub(super) text: String,
}

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

/// Code or test activity in the attempt itself; prose never matches.
static CODE_ACTIVITY: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(concat!(
        r"(?m)```|^\$ |^diff --git|^@@ |",
        r"\bTraceback \(most recent call last\)|\b\d+ (?:passed|failed)\b|",
        r"^(?:\$ |> )?(?:pytest|unittest|npm (?:test|run)|cargo (?:test|build)|",
        r"go test|make test)\b",
    ))
    .ok()
});

static TOOL_ACTIVITY: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(r#"(?m)^\[tool\b|^\[[^\]\n]* calls tools?:|"tool_calls?"\s*:"#).ok()
});

/// Text of a content sequence, and whether it carried media or unknown blocks.
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
pub(super) fn user_task_text(turns: &[Turn]) -> String {
    turns
        .iter()
        .rev()
        .find(|turn| turn.role == Role::User && !turn.text.trim().is_empty())
        .map(|turn| turn.text.clone())
        .unwrap_or_default()
}

pub(super) fn has_assistant_turn(turns: &[Turn]) -> bool {
    turns.iter().any(|turn| turn.role == Role::Assistant)
}

/// Whether the conversation contains an executed or proposed tool step.
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

/// Idempotent secret and PII scrub applied before text reaches a verifier.
pub(super) fn redact(text: &str) -> String {
    let mut out = text.to_string();
    for pattern in REDACT_PATTERNS.iter() {
        out = pattern.replace_all(&out, "[REDACTED]").into_owned();
    }
    out
}

pub(super) fn char_len(text: &str) -> usize {
    text.chars().count()
}

/// Head-and-tail clip to `budget` characters, keeping `head_frac` as head.
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

pub(super) fn observed_hardening(attempt: &str) -> bool {
    CODE_ACTIVITY
        .as_ref()
        .is_some_and(|pattern| pattern.is_match(attempt))
}

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
    pub(super) tail_clean: bool,
    /// Consecutive clean results ending the conversation.
    pub(super) clean_tail: i32,
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
            let failed = (result.is_error == Some(true)
                || object.is_some_and(|object| object.contains_key("error"))
                || classify_text(&text).0 > 0.0)
                && !exit_ok;
            record.results = record.results.saturating_add(1);
            record.errors = record.errors.saturating_add(i32::from(failed));
            record.tail_clean = !failed;
            record.clean_tail = if failed {
                0
            } else {
                record.clean_tail.saturating_add(1)
            };
        }
        record
    }
}
