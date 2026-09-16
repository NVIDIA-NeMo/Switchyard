// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The judged views of a request and their completeness gates.
//!
//! Three renderers produce the transcript a downstream verifier reads: a general
//! session view, a coding view that preserves independently extracted evidence,
//! and an agentic view that omits framework instructions while retaining the
//! user task and tool trajectory.
//!
//! Two invariants hold across all renderers:
//!
//! - **Redaction precedes budgeting.** Each section is redacted before its
//!   budget applies, so redaction expansion can never push a later section out
//!   of the verifier's window.
//! - **The user task and attempt stay visible.** Requirement text is represented
//!   in full, while trajectory and attempt evidence may clip to bounded windows.

use switchyard_protocol::{ContentBlock, Request, Role};

use super::text::{Turn, clip_mid, redact};

/// Evidence-only character budgets for judged views.
const ASSISTANT_TURN_BUDGET: usize = 1200;
pub(super) const ATTEMPT_BUDGET: usize = 2200;

/// Agentic-view budgets.
///
/// User requirements render in full; trajectory evidence may clip.
const AGENTIC_TRAJECTORY_BUDGET: usize = 12_000;
const AGENTIC_EVENT_BUDGET: usize = 1400;

/// Redacted, non-empty turns plus the index of the latest user turn.
///
/// Blank turns are dropped before indexing, so the returned index refers to the
/// filtered sequence.
struct RenderedTurns {
    turns: Vec<(Role, String)>,
    latest_user: Option<usize>,
}

impl RenderedTurns {
    fn new(turns: &[Turn]) -> Self {
        let rendered: Vec<(Role, String)> = turns
            .iter()
            .filter(|turn| !turn.text.trim().is_empty())
            .map(|turn| (turn.role, redact(&turn.text)))
            .collect();
        let latest_user = rendered.iter().rposition(|(role, _)| *role == Role::User);
        Self {
            turns: rendered,
            latest_user,
        }
    }

    /// Text of every instruction turn, in order.
    ///
    /// System and developer turns are one class here: both carry operator
    /// requirements the judged view must represent, and neither may be dropped.
    fn instructions(&self) -> Vec<&str> {
        self.turns
            .iter()
            .filter(|(role, _)| is_instruction(*role))
            .map(|(_, text)| text.as_str())
            .collect()
    }

    /// Text of every user turn other than the latest, in order.
    fn earlier_user(&self) -> Vec<&str> {
        self.turns
            .iter()
            .enumerate()
            .filter(|(index, (role, _))| *role == Role::User && Some(*index) != self.latest_user)
            .map(|(_, (_, text))| text.as_str())
            .collect()
    }

    fn latest_user_text(&self) -> &str {
        self.latest_user
            .map(|index| self.turns[index].1.as_str())
            .unwrap_or_default()
    }
}

/// True for roles carrying operator instructions rather than conversation.
fn is_instruction(role: Role) -> bool {
    matches!(role, Role::System | Role::Developer)
}

/// Stable role label used in rendered transcripts.
fn role_label(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::Developer => "developer",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

/// Renders the canonical judged view of a request and its local attempt.
///
/// Earlier turns render first and fill the start of the verifier's window with
/// the conversation opening; the latest user instruction and the attempt render
/// last, so both are always visible. The attempt is tail-weighted because its
/// final output carries the evidence.
pub(super) fn render_session(turns: &[Turn], attempt: &str) -> String {
    let rendered = RenderedTurns::new(turns);
    let mut lines: Vec<String> = rendered
        .turns
        .iter()
        .enumerate()
        .filter(|(index, _)| Some(*index) != rendered.latest_user)
        .map(|(_, (role, text))| {
            let text = if *role == Role::User || is_instruction(*role) {
                text.clone()
            } else {
                clip_mid(text, ASSISTANT_TURN_BUDGET, 0.5)
            };
            format!("[{}] {text}", role_label(*role))
        })
        .collect();
    if rendered.latest_user.is_some() {
        lines.push(format!("[user (latest)] {}", rendered.latest_user_text()));
    }
    lines.push(format!(
        "[assistant attempt] {}",
        clip_mid(&redact(attempt), ATTEMPT_BUDGET, 1.0 / 3.0)
    ));
    lines.join("\n")
}

/// True when the session has a user task to verify.
///
/// Requirement-bearing turns are rendered in full. The verifier model itself
/// decides whether that complete view fits its live context window.
pub(super) fn session_context_complete(turns: &[Turn], _attempt: &str) -> bool {
    let rendered = RenderedTurns::new(turns);
    rendered.latest_user.is_some()
}

/// Whether all user-authored requirements fit the agentic view without clipping.
///
/// System and developer instructions are deliberately excluded because agent
/// framework boilerplate is not part of the task-level view used by the
/// reference experiments.
pub(super) fn agentic_context_complete(turns: &[Turn]) -> bool {
    let rendered = RenderedTurns::new(turns);
    rendered.latest_user.is_some()
}

/// Renders the user task, prior tool trajectory, and current local attempt.
///
/// Agent framework instructions are omitted. User-authored text is preserved in
/// full after [`agentic_context_complete`] succeeds, while model and tool
/// evidence is bounded and tail-weighted.
pub(super) fn render_agentic_view(request: &Request, turns: &[Turn], attempt: &str) -> String {
    let rendered = RenderedTurns::new(turns);
    let requirements = rendered
        .turns
        .iter()
        .filter(|(role, _)| *role == Role::User)
        .enumerate()
        .map(|(index, (_, text))| {
            let label = if index == 0 {
                "user task".to_string()
            } else {
                format!("user update {index}")
            };
            format!("[{label}] {text}")
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
    parts.push(format!(
        "CURRENT ATTEMPT:\n{}",
        clip_mid(&redact(attempt), ATTEMPT_BUDGET, 1.0 / 3.0)
    ));
    parts.join("\n\n")
}

/// Bounded assistant and tool events after the first user task.
fn agentic_trajectory(request: &Request) -> Vec<String> {
    let mut started = false;
    let mut events = Vec::new();
    for message in &request.llm_request.messages {
        if !started {
            let (text, _) = super::text::content_text(&message.content);
            if message.role == Role::User && !text.trim().is_empty() {
                started = true;
            }
            continue;
        }

        for block in &message.content {
            let event = match block {
                ContentBlock::Text { text } | ContentBlock::Refusal { text }
                    if message.role == Role::Assistant =>
                {
                    Some(("assistant", text.clone()))
                }
                ContentBlock::Text { text } | ContentBlock::Refusal { text }
                    if message.role == Role::Tool =>
                {
                    Some(("tool result", text.clone()))
                }
                ContentBlock::ToolCall(call) => {
                    Some(("tool call", format!("{}({})", call.name, call.arguments)))
                }
                ContentBlock::ToolResult(result) => {
                    let (text, _) = super::text::content_text(&result.content);
                    let label = if result.is_error == Some(true) {
                        "tool error"
                    } else {
                        "tool result"
                    };
                    Some((label, text))
                }
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

/// True when a coding conversation has a user task to verify.
///
/// Requirement-bearing turns are rendered in full and the verifier model's
/// live context limit decides whether they are representable.
pub(super) fn coding_context_complete(turns: &[Turn]) -> bool {
    RenderedTurns::new(turns).latest_user.is_some()
}

/// Renders the coding judged view: the requirement turns in full, plus evidence
/// sections extracted from the attempt by structure alone.
///
/// Requirement turns render unclipped by contract — [`coding_context_complete`]
/// gates their post-redaction lengths before this renderer is consulted. Earlier
/// assistant turns are omitted as the model's own words. The extracted sections
/// (modified files, the final test invocation, the last traceback, the end of the
/// attempt) are benchmark-independent: they key off diff headers, line-anchored
/// runner invocations, and traceback headers, not any dataset's conventions.
///
/// Trust note: in-transcript test output remains agent-written text. It is one
/// input among several and never commits on its own.
pub(super) fn render_coding_view(turns: &[Turn], attempt: &str) -> String {
    let rendered = RenderedTurns::new(turns);
    let attempt = redact(attempt);
    let lines: Vec<&str> = attempt.lines().collect();

    let files: Vec<&str> = lines
        .iter()
        .filter_map(|line| line.strip_prefix("diff --git "))
        .take(12)
        .collect();
    let test_block = trailing_block(&lines, 25, is_test_invocation);
    let traceback_block = trailing_block(&lines, 20, |line| {
        line.starts_with("Traceback (most recent call last)")
    });

    let mut parts = vec![format!("LATEST REQUEST:\n{}", rendered.latest_user_text())];
    let system = rendered.instructions();
    if !system.is_empty() {
        parts.push(format!(
            "SYSTEM INSTRUCTIONS (in full):\n{}",
            system.join("\n")
        ));
    }
    let earlier = rendered.earlier_user();
    if !earlier.is_empty() {
        let listed = earlier
            .iter()
            .enumerate()
            .map(|(index, text)| format!("[user {}] {text}", index + 1))
            .collect::<Vec<_>>()
            .join("\n");
        parts.push(format!(
            "EARLIER REQUIREMENTS (each earlier user turn, in full):\n{listed}"
        ));
    }
    if !files.is_empty() {
        parts.push(format!(
            "FILES MODIFIED:\n{}",
            clip_mid(&files.join("\n"), 500, 0.5)
        ));
    }
    if let Some(block) = test_block {
        parts.push(format!(
            "FINAL TEST RUN:\n{}",
            clip_mid(&block, 1800, 1.0 / 3.0)
        ));
    }
    if let Some(block) = traceback_block {
        parts.push(format!("LAST TRACEBACK:\n{}", clip_mid(&block, 800, 0.5)));
    }
    parts.push(format!(
        "FINAL OUTPUT (end of attempt):\n{}",
        tail_chars(&attempt, 2500)
    ));
    parts.join("\n\n")
}

/// The last block of `span` lines starting at the final line matching `predicate`.
fn trailing_block(lines: &[&str], span: usize, predicate: impl Fn(&str) -> bool) -> Option<String> {
    let start = lines.iter().rposition(|line| predicate(line))?;
    Some(lines[start..lines.len().min(start + span)].join("\n"))
}

/// True for a line-anchored invocation of a known test runner.
///
/// Matched by hand rather than by regex so the leading-`$`/whitespace tolerance
/// stays readable; the reference implementation anchors the same runner list.
fn is_test_invocation(line: &str) -> bool {
    const RUNNERS: &[&str] = &[
        "pytest",
        "unittest",
        "npm test",
        "npm run",
        "cargo test",
        "cargo build",
        "go test",
        "make test",
    ];
    let rest = line.trim_start().trim_start_matches('$').trim_start();
    RUNNERS.iter().any(|runner| {
        rest.strip_prefix(runner).is_some_and(|tail| {
            tail.is_empty() || !tail.starts_with(|c: char| c.is_alphanumeric() || c == '_')
        })
    })
}

/// The last `budget` characters of `text`.
fn tail_chars(text: &str, budget: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    chars[chars.len().saturating_sub(budget)..].iter().collect()
}
