// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal text extraction for verification-gated routing.

use std::sync::LazyLock;

use regex::Regex;
use switchyard_protocol::{ContentBlock, Request, Role};

const MAX_TRANSCRIPT_CHARS: usize = 24_000;

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

static CODE_ACTIVITY: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(
        r"(?m)```|^diff --git|^@@ |\bTraceback \(most recent call last\)|\b\d+ (?:passed|failed)\b|^(?:\$ )?(?:pytest|unittest|npm (?:test|run)|cargo (?:test|build)|go test|make test)\b",
    )
    .ok()
});

static TOOL_ACTIVITY: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r#"(?m)^\[tool\b|"tool_calls?"\s*:"#).ok());

pub(super) struct RequestView {
    pub(super) task_text: String,
    pub(super) transcript: String,
}

/// Extracts a complete judged view, rejecting unsupported or oversized input.
pub(super) fn request_view(request: &Request, attempt: &str) -> Option<RequestView> {
    if attempt.trim().is_empty() {
        return None;
    }

    let mut lines = Vec::new();
    let mut latest_user = None;
    for instruction in &request.llm_request.instructions {
        let text = content_text(&instruction.content)?;
        if !text.trim().is_empty() {
            lines.push(format!(
                "[{}] {}",
                role_label(instruction.role),
                redact(&text)
            ));
        }
    }
    for message in &request.llm_request.messages {
        let text = content_text(&message.content)?;
        if message.role == Role::User && !text.trim().is_empty() {
            latest_user = Some(text.clone());
        }
        if !text.trim().is_empty() {
            lines.push(format!("[{}] {}", role_label(message.role), redact(&text)));
        }
    }
    let task_text = latest_user?;
    lines.push(format!("[assistant attempt] {}", redact(attempt)));
    let transcript = lines.join("\n");
    (transcript.chars().count() <= MAX_TRANSCRIPT_CHARS).then_some(RequestView {
        task_text,
        transcript,
    })
}

fn content_text(content: &[ContentBlock]) -> Option<String> {
    let mut parts = Vec::new();
    for block in content {
        match block {
            ContentBlock::Text { text } | ContentBlock::Refusal { text } => {
                parts.push(text.clone());
            }
            ContentBlock::ToolResult(result) => parts.push(content_text(&result.content)?),
            ContentBlock::ToolCall(_) | ContentBlock::Reasoning { .. } => {}
            _ => return None,
        }
    }
    Some(parts.join("\n"))
}

fn redact(text: &str) -> String {
    REDACT_PATTERNS
        .iter()
        .fold(text.to_string(), |value, pattern| {
            pattern.replace_all(&value, "[REDACTED]").into_owned()
        })
}

fn role_label(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::Developer => "developer",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

pub(super) fn observed_code_activity(attempt: &str) -> bool {
    CODE_ACTIVITY
        .as_ref()
        .is_some_and(|pattern| pattern.is_match(attempt))
}

pub(super) fn observed_tool_activity(attempt: &str) -> bool {
    TOOL_ACTIVITY
        .as_ref()
        .is_some_and(|pattern| pattern.is_match(attempt))
}
