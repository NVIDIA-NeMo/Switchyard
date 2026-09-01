// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The judged views of a request and their completeness gates.
//!
//! Two renderers produce the transcript a downstream verifier reads: a general
//! session view and a coding view that preserves independently extracted
//! evidence sections. Each has a gate that answers whether the request can be
//! represented *in full*; a request that would need a requirement turn clipped
//! is never judged at all, because judging against partial requirements is worse
//! than not judging.
//!
//! Two invariants hold across both renderers:
//!
//! - **Redaction precedes budgeting.** Each section is redacted before its
//!   budget applies, so redaction expansion can never push a later section out
//!   of the verifier's window.
//! - **The latest instruction and the attempt render last**, inside the window a
//!   verifier is guaranteed to see, so a long earlier conversation cannot evict
//!   them.

use switchyard_protocol::Role;

use super::text::{Turn, char_len, clip_mid, redact};

/// Per-section character budgets for the session view.
pub(super) const EARLIER_TURN_BUDGET: usize = 1200;
pub(super) const LATEST_USER_BUDGET: usize = 8000;
pub(super) const ATTEMPT_BUDGET: usize = 2200;

/// Whole-session budget: a rendered session longer than this would overflow the
/// verifier window and silently drop requirement content.
const SESSION_WINDOW: usize = 24_000;

/// Coding-view completeness limits, all measured post-redaction.
const CODING_MAX_EARLIER_USER_TURNS: usize = 4;
const CODING_EARLIER_TURN_MAX: usize = 1000;
const CODING_LATEST_MAX: usize = 12_000;
const CODING_MAX_SYSTEM_TURNS: usize = 1;
const CODING_SYSTEM_TURN_MAX: usize = 2000;

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
            format!(
                "[{}] {}",
                role_label(*role),
                clip_mid(text, EARLIER_TURN_BUDGET, 0.5)
            )
        })
        .collect();
    if rendered.latest_user.is_some() {
        lines.push(format!(
            "[user (latest)] {}",
            clip_mid(rendered.latest_user_text(), LATEST_USER_BUDGET, 0.5)
        ));
    }
    lines.push(format!(
        "[assistant attempt] {}",
        clip_mid(&redact(attempt), ATTEMPT_BUDGET, 1.0 / 3.0)
    ));
    lines.join("\n")
}

/// True when every requirement-bearing turn fits its budget unclipped and the
/// whole rendered session fits the verifier window.
///
/// Assistant turns are the model's own words and may clip; the attempt is
/// evidence, tail-weighted by design, not a requirement.
pub(super) fn session_context_complete(turns: &[Turn], attempt: &str) -> bool {
    let rendered = RenderedTurns::new(turns);
    for (index, (role, text)) in rendered.turns.iter().enumerate() {
        let budget = if Some(index) == rendered.latest_user {
            LATEST_USER_BUDGET
        } else if *role == Role::User || is_instruction(*role) {
            EARLIER_TURN_BUDGET
        } else {
            continue;
        };
        if char_len(text) > budget {
            return false;
        }
    }
    char_len(&render_session(turns, attempt)) <= SESSION_WINDOW
}

/// True when every task-bearing turn of a coding conversation is representable
/// in full, the latest request included.
///
/// Lengths are measured post-redaction with the router's own redactor, so an
/// expanding redactor cannot smuggle a turn past the gate. A turn that would
/// need clipping may carry a mid-turn requirement a verifier would never see, so
/// this fails closed rather than judging against partial constraints.
pub(super) fn coding_context_complete(turns: &[Turn]) -> bool {
    let rendered = RenderedTurns::new(turns);
    if rendered.latest_user.is_none() || char_len(rendered.latest_user_text()) > CODING_LATEST_MAX {
        return false;
    }
    let earlier = rendered.earlier_user();
    let system = rendered.instructions();
    earlier.len() <= CODING_MAX_EARLIER_USER_TURNS
        && earlier
            .iter()
            .all(|text| char_len(text) <= CODING_EARLIER_TURN_MAX)
        && system.len() <= CODING_MAX_SYSTEM_TURNS
        && system
            .iter()
            .all(|text| char_len(text) <= CODING_SYSTEM_TURN_MAX)
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
    let system = &system[..system.len().min(CODING_MAX_SYSTEM_TURNS)];
    if !system.is_empty() {
        parts.push(format!(
            "SYSTEM INSTRUCTIONS (in full):\n{}",
            system.join("\n")
        ));
    }
    let earlier = rendered.earlier_user();
    let earlier = &earlier[..earlier.len().min(CODING_MAX_EARLIER_USER_TURNS)];
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
