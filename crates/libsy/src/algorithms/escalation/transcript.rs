// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Condenses a conversation into the bounded transcript the escalation judge reads.
//!
//! The judge sees a small fraction of a long agentic run, so what survives here decides what it
//! can detect. The sizing below is fixed; the three knobs an operator can move live on
//! [`EscalationJudgeSettings`].

use switchyard_protocol::{ContentBlock, Message, Role};

use super::EscalationJudgeSettings;
use crate::Request;

/// Separator marking where [`truncate_middle`] dropped a message's interior.
const TRIM_MARKER: &str = " ...[trimmed] ";

/// Suffix marking a transcript cut off by [`MAX_REQUEST_CHARS`].
const TRUNCATION_SUFFIX: &str = "...<truncated>";

/// Per-message cap for system and developer anchors. Coding-agent harnesses inject very large
/// boilerplate system prompts carrying no trajectory signal; uncapped they crowd out the window.
const SYSTEM_CHARS: usize = 1_000;

/// Cap for the first user message — the task statement the judge needs to detect drift, so it
/// gets the most generous anchor budget.
const FIRST_USER_CHARS: usize = 2_000;

/// Backstop on the whole assembled transcript, for a pathological single message. The window
/// caps below normally bind first; when this does bind, the oldest window lines drop.
const MAX_REQUEST_CHARS: usize = 18_000;

/// The 1-indexed model invocation the transcript ends on: one per assistant reply, the newest
/// being the turn under judgement.
pub(super) fn conversation_turn(request: &Request) -> usize {
    request
        .llm_request
        .messages
        .iter()
        .filter(|message| message.role == Role::Assistant)
        .count()
}

/// Flattens a message to plain text, tool calls and tool results included.
///
/// Not [`Message::text_content`]: that keeps only text and refusal blocks, erasing exactly the
/// repeated-command signal the judge's loop detection relies on.
fn message_text(message: &Message) -> String {
    let mut parts = Vec::new();
    collect_text(&message.content, &mut parts);
    parts.join(" ")
}

/// Appends the judge-relevant text of each block, descending into tool results.
fn collect_text(content: &[ContentBlock], parts: &mut Vec<String>) {
    for block in content {
        match block {
            ContentBlock::Text { text } | ContentBlock::Refusal { text } => {
                parts.push(text.clone());
            }
            ContentBlock::ToolCall(call) => {
                parts.push(format!("tool_call {}({})", call.name, call.arguments));
            }
            ContentBlock::ToolResult(result) => collect_text(&result.content, parts),
            _ => {}
        }
    }
}

/// Keeps the head and tail of `text` within `limit` characters.
///
/// The head gets two thirds: the command or error signature opening a message carries more
/// signal than its trailing output.
fn truncate_middle(text: &str, limit: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= limit {
        return text.to_string();
    }
    let keep = limit
        .saturating_sub(TRIM_MARKER.chars().count())
        .max(20)
        .min(chars.len());
    let head = keep * 2 / 3;
    let tail = keep - head;
    let mut out: String = chars[..head].iter().collect();
    out.push_str(TRIM_MARKER);
    out.extend(chars[chars.len() - tail..].iter());
    out
}

/// Renders a compact role-labelled transcript for the judge.
///
/// The framing anchors — system/developer messages and the first user message, where harnesses
/// put the task statement — are kept unconditionally and capped individually; the trailing
/// window carries recent activity, under a coverage header so the judge can reason about pace
/// rather than assume it sees everything. Over [`MAX_REQUEST_CHARS`] the oldest window lines go
/// first: the newest evidence is strictly the most valuable.
pub(super) fn summarize_for_judge(
    messages: &[Message],
    turn: usize,
    settings: &EscalationJudgeSettings,
) -> String {
    let mut anchors: Vec<String> = Vec::new();
    let mut window: Vec<String> = Vec::new();
    let mut first_user_seen = false;

    for message in messages {
        let text = message_text(message);
        match message.role {
            Role::System | Role::Developer => anchors.push(format!(
                "[{}] {}",
                role_label(message.role),
                truncate_middle(&text, SYSTEM_CHARS)
            )),
            Role::User if !first_user_seen => {
                first_user_seen = true;
                anchors.push(format!(
                    "[user (task)] {}",
                    truncate_middle(&text, FIRST_USER_CHARS)
                ));
            }
            role => window.push(format!(
                "[{}] {}",
                role_label(role),
                truncate_middle(&text, settings.window_message_chars)
            )),
        }
    }

    if window.len() > settings.recent_turn_window {
        window.drain(..window.len() - settings.recent_turn_window);
    }

    let header = format!(
        "Conversation turn {turn}; showing the last {} of {} messages after the task framing.",
        window.len(),
        messages.len(),
    );
    let assemble = |window: &[String]| {
        std::iter::once(header.as_str())
            .chain(anchors.iter().map(String::as_str))
            .chain(window.iter().map(String::as_str))
            .collect::<Vec<_>>()
            .join("\n")
    };

    let mut text = assemble(&window);
    while text.chars().count() > MAX_REQUEST_CHARS && !window.is_empty() {
        window.remove(0);
        text = assemble(&window);
    }
    if text.chars().count() > MAX_REQUEST_CHARS {
        let keep = MAX_REQUEST_CHARS.saturating_sub(TRUNCATION_SUFFIX.chars().count() + 1);
        text = text.chars().take(keep).collect::<String>() + TRUNCATION_SUFFIX;
    }
    text
}

/// The transcript label for a role.
fn role_label(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::Developer => "developer",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use switchyard_protocol::{LlmRequest, Message, Role, ToolCall, ToolResult};

    use super::*;

    /// A conversation carrying `replies` assistant messages, each answered by a user message.
    fn with_replies(replies: usize) -> Request {
        let mut messages = vec![Message::text(Role::User, "What is 2+2?")];
        for attempt in 1..=replies {
            messages.push(Message::text(Role::Assistant, format!("attempt {attempt}")));
            messages.push(Message::text(Role::User, format!("still wrong {attempt}")));
        }
        Request {
            llm_request: LlmRequest {
                messages,
                ..LlmRequest::default()
            },
            ..Request::default()
        }
    }

    #[test]
    fn conversation_turn_counts_assistant_replies() {
        // The transcript handed to the judge ends on the reply being judged, so the count is
        // the assistant replies present — no lookahead.
        assert_eq!(conversation_turn(&with_replies(0)), 0);
        assert_eq!(conversation_turn(&with_replies(4)), 4);
    }

    #[test]
    fn message_text_keeps_tool_calls_and_results() {
        let call = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text {
                    text: "running it".to_string(),
                },
                ContentBlock::ToolCall(ToolCall {
                    id: "call-1".to_string(),
                    name: "bash".to_string(),
                    arguments: json!({"cmd": "ls"}),
                }),
            ],
        };
        let text = message_text(&call);
        assert!(text.contains("running it"), "{text}");
        assert!(text.contains(r#"tool_call bash({"cmd":"ls"})"#), "{text}");

        let result = Message {
            role: Role::Tool,
            content: vec![ContentBlock::ToolResult(ToolResult {
                tool_call_id: "call-1".to_string(),
                content: vec![ContentBlock::Text {
                    text: "no such file".to_string(),
                }],
                is_error: Some(true),
            })],
        };
        assert_eq!(message_text(&result), "no such file");
    }

    #[test]
    fn truncate_middle_keeps_head_and_tail() {
        let text = "a".repeat(40) + &"z".repeat(40);
        let trimmed = truncate_middle(&text, 50);
        assert!(trimmed.chars().count() <= 50, "{trimmed}");
        assert!(trimmed.starts_with('a'));
        assert!(trimmed.ends_with('z'));
        assert!(trimmed.contains("[trimmed]"));

        // Under the limit the text is returned untouched.
        assert_eq!(truncate_middle("short", 50), "short");
    }

    #[test]
    fn summary_keeps_anchors_and_the_recent_window() {
        let mut messages = vec![
            Message::text(Role::System, "you are a coding agent"),
            Message::text(Role::User, "fix the failing test"),
        ];
        for i in 0..10 {
            messages.push(Message::text(Role::Assistant, format!("step {i}")));
        }
        let settings = EscalationJudgeSettings {
            recent_turn_window: 3,
            ..EscalationJudgeSettings::default()
        };

        let summary = summarize_for_judge(&messages, 10, &settings);

        assert!(
            summary.contains("[system] you are a coding agent"),
            "{summary}"
        );
        assert!(
            summary.contains("[user (task)] fix the failing test"),
            "{summary}"
        );
        assert!(summary.contains("Conversation turn 10; showing the last 3 of 12 messages"));
        // Only the newest window entries survive.
        assert!(summary.contains("step 9"), "{summary}");
        assert!(summary.contains("step 7"), "{summary}");
        assert!(!summary.contains("step 6"), "{summary}");
    }

    #[test]
    fn summary_drops_oldest_window_lines_under_the_char_cap() {
        // MAX_REQUEST_CHARS is a backstop, not a dial: at default settings the window caps
        // bind first (28 x 500 plus anchors sits under it), so reaching it takes an unusually
        // wide per-message cap. That is the point — it only fires on pathological input.
        let mut messages = vec![
            Message::text(Role::System, "framing"),
            Message::text(Role::User, "task"),
        ];
        for i in 0..20 {
            messages.push(Message::text(
                Role::Assistant,
                format!("{i} {}", "x".repeat(2_000)),
            ));
        }
        let settings = EscalationJudgeSettings {
            window_message_chars: 2_000,
            ..EscalationJudgeSettings::default()
        };

        let summary = summarize_for_judge(&messages, 20, &settings);

        assert!(
            summary.chars().count() <= MAX_REQUEST_CHARS,
            "{}",
            summary.chars().count()
        );
        // Anchors are never dropped, and the newest activity outlives the oldest.
        assert!(summary.contains("[system] framing"), "{summary}");
        assert!(summary.contains("[user (task)] task"), "{summary}");
        assert!(summary.contains("19 xxx"), "{summary}");
        assert!(!summary.contains("0 xxx"), "{summary}");
    }
}
