// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The judged views a verifier reads.
//!
//! Each section is redacted before its budget applies, and user requirements
//! render in full while trajectory and attempt evidence clip.

use switchyard_protocol::{ContentBlock, Request, Role};

use super::text::{Turn, clip_mid, content_text, redact, tool_result_failed};

const ASSISTANT_TURN_BUDGET: usize = 1200;
pub(super) const ATTEMPT_BUDGET: usize = 2200;
const AGENTIC_TRAJECTORY_BUDGET: usize = 12_000;
const AGENTIC_EVENT_BUDGET: usize = 1400;

/// Earlier turns first, then the latest user instruction and the attempt.
pub(super) fn render_session(turns: &[Turn], attempt: &str) -> String {
    let latest_user = turns
        .iter()
        .rposition(|turn| turn.role == Role::User && !turn.text.trim().is_empty());
    let mut lines = Vec::new();
    for (index, turn) in turns.iter().enumerate() {
        if turn.text.trim().is_empty() || Some(index) == latest_user {
            continue;
        }
        let mut text = redact(&turn.text);
        if matches!(turn.role, Role::Assistant | Role::Tool) {
            text = clip_mid(&text, ASSISTANT_TURN_BUDGET, 0.5);
        }
        lines.push(format!("[{}] {text}", turn.role));
    }
    if let Some(index) = latest_user {
        let text = redact(&turns[index].text);
        lines.push(format!("[user (latest)] {text}"));
    }
    let attempt = redact(attempt);
    let attempt = clip_mid(&attempt, ATTEMPT_BUDGET, 1.0 / 3.0);
    lines.push(format!("[assistant attempt] {attempt}"));
    lines.join("\n")
}

/// The user task and updates, the tool trajectory, and the attempt.
///
/// Framework instructions are omitted.
pub(super) fn render_agentic_view(request: &Request, turns: &[Turn], attempt: &str) -> String {
    let user_turns = turns
        .iter()
        .filter(|turn| turn.role == Role::User && !turn.text.trim().is_empty());
    let mut requirements = Vec::new();
    for (index, turn) in user_turns.enumerate() {
        let text = redact(&turn.text);
        requirements.push(match index {
            0 => format!("[user task] {text}"),
            _ => format!("[user update {index}] {text}"),
        });
    }
    let requirements = requirements.join("\n");
    let trajectory = agentic_trajectory(request);
    let mut parts = vec![format!("USER TASK AND UPDATES:\n{requirements}")];
    if !trajectory.is_empty() {
        parts.push(format!(
            "TOOL TRAJECTORY:\n{}",
            clip_mid(&trajectory.join("\n"), AGENTIC_TRAJECTORY_BUDGET, 1.0 / 3.0)
        ));
    }
    let attempt = redact(attempt);
    let attempt = clip_mid(&attempt, ATTEMPT_BUDGET, 1.0 / 3.0);
    parts.push(format!("CURRENT ATTEMPT:\n{attempt}"));
    parts.join("\n\n")
}

/// Bounded assistant and tool events after the first user task.
fn agentic_trajectory(request: &Request) -> Vec<String> {
    let mut messages = request.llm_request.messages.iter();
    for message in messages.by_ref() {
        if message.role == Role::User && !content_text(&message.content).0.trim().is_empty() {
            break;
        }
    }
    let mut events = Vec::new();
    for message in messages {
        for block in &message.content {
            let event = match block {
                ContentBlock::Text { text } | ContentBlock::Refusal { text } => {
                    match message.role {
                        Role::Assistant => Some(("assistant", text.clone())),
                        Role::Tool => Some(("tool result", text.clone())),
                        _ => None,
                    }
                }
                ContentBlock::ToolCall(call) => {
                    Some(("tool call", format!("{}({})", call.name, call.arguments)))
                }
                ContentBlock::ToolResult(result) => Some((
                    if tool_result_failed(result) {
                        "tool error"
                    } else {
                        "tool result"
                    },
                    content_text(&result.content).0,
                )),
                _ => None,
            };
            if let Some((label, text)) = event {
                let text = redact(&text);
                let text = clip_mid(&text, AGENTIC_EVENT_BUDGET, 0.5);
                events.push(format!("[{label}] {text}"));
            }
        }
    }
    events
}
