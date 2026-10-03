// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The judged views a verifier reads.
//!
//! Each section is redacted before its budget applies, and user requirements
//! render in full while trajectory and attempt evidence clip.

use switchyard_protocol::{ContentBlock, Request, Role};

use super::text::{Turn, clip_mid, content_text, redact};

const ASSISTANT_TURN_BUDGET: usize = 1200;
pub(super) const ATTEMPT_BUDGET: usize = 2200;
const AGENTIC_TRAJECTORY_BUDGET: usize = 12_000;
const AGENTIC_EVENT_BUDGET: usize = 1400;

/// Redacted non-empty turns, and the index of the latest user turn among them.
fn rendered(turns: &[Turn]) -> (Vec<(Role, String)>, Option<usize>) {
    let rendered: Vec<(Role, String)> = turns
        .iter()
        .filter(|turn| !turn.text.trim().is_empty())
        .map(|turn| (turn.role, redact(&turn.text)))
        .collect();
    let latest_user = rendered.iter().rposition(|(role, _)| *role == Role::User);
    (rendered, latest_user)
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

fn attempt_section(attempt: &str) -> String {
    clip_mid(&redact(attempt), ATTEMPT_BUDGET, 1.0 / 3.0)
}

/// Earlier turns first, then the latest user instruction and the attempt.
pub(super) fn render_session(turns: &[Turn], attempt: &str) -> String {
    let (turns, latest_user) = rendered(turns);
    let mut lines: Vec<String> = turns
        .iter()
        .enumerate()
        .filter(|(index, _)| Some(*index) != latest_user)
        .map(|(_, (role, text))| {
            let text = if matches!(role, Role::User | Role::System | Role::Developer) {
                text.clone()
            } else {
                clip_mid(text, ASSISTANT_TURN_BUDGET, 0.5)
            };
            format!("[{}] {text}", role_label(*role))
        })
        .collect();
    if let Some(index) = latest_user {
        lines.push(format!("[user (latest)] {}", turns[index].1));
    }
    lines.push(format!("[assistant attempt] {}", attempt_section(attempt)));
    lines.join("\n")
}

/// The user task and updates, the tool trajectory, and the attempt.
///
/// Framework instructions are omitted.
pub(super) fn render_agentic_view(request: &Request, turns: &[Turn], attempt: &str) -> String {
    let (turns, _) = rendered(turns);
    let requirements = turns
        .iter()
        .filter(|(role, _)| *role == Role::User)
        .enumerate()
        .map(|(index, (_, text))| match index {
            0 => format!("[user task] {text}"),
            _ => format!("[user update {index}] {text}"),
        })
        .collect::<Vec<_>>()
        .join("\n");
    let trajectory = agentic_trajectory(request);
    let mut parts = vec![format!("USER TASK AND UPDATES:\n{requirements}")];
    if !trajectory.is_empty() {
        parts.push(format!(
            "TOOL TRAJECTORY:\n{}",
            clip_mid(&trajectory.join("\n"), AGENTIC_TRAJECTORY_BUDGET, 1.0 / 3.0)
        ));
    }
    parts.push(format!("CURRENT ATTEMPT:\n{}", attempt_section(attempt)));
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
                    if result.is_error == Some(true) {
                        "tool error"
                    } else {
                        "tool result"
                    },
                    content_text(&result.content).0,
                )),
                _ => None,
            };
            if let Some((label, text)) = event {
                events.push(format!(
                    "[{label}] {}",
                    clip_mid(&redact(&text), AGENTIC_EVENT_BUDGET, 0.5)
                ));
            }
        }
    }
    events
}
