// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Verification-gated routing: fail-closed capability derivation.
//!
//! Capabilities are never client-declared. They come from the router's own
//! typing of the request, the attempt it produced locally, and its own reading
//! of the tool results in the conversation. No client-reachable input selects
//! a weaker regime than the default one; unsupported content and missing
//! evidence select [`Branch::Unknown`], which never commits.

#![allow(dead_code)]

use switchyard_protocol::Request;

use self::text::ToolRecord;

mod decide;
mod readout;
mod render;
mod rungs;
mod text;

#[cfg(test)]
mod tests;

/// The verification regime a request's capabilities license.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum Branch {
    /// Code or test activity, with no sandboxed checker to appeal to.
    Coding,
    /// A single-turn request typed as answer-seeking.
    Answer,
    /// Conversational traffic.
    Chat,
    /// A tool-using session verified from its trajectory.
    Agentic,
    /// Anything else with an attempt to verify.
    DefaultVerified,
    /// Nothing to verify.
    #[default]
    Unknown,
}

/// A task type produced by the router's own typing call; `None` abstains.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TaskType {
    Coding,
    Agentic,
    Answer,
    Chat,
}

/// The complete input the decision core sees.
#[derive(Clone, Debug, Default, PartialEq)]
struct Capabilities {
    branch: Branch,
    /// The judged view of the request and attempt.
    transcript: Option<String>,
    /// The router's own tool-result record; unused by the answer regime.
    tools: Option<ToolRecord>,
}

/// Derives capabilities from the router's own evidence, failing closed.
fn derive_capabilities(
    request: &Request,
    attempt: &str,
    task_type: Option<TaskType>,
) -> Capabilities {
    let (turns, unsupported) = text::turns(request);
    if unsupported || text::user_task_text(&turns).trim().is_empty() || attempt.trim().is_empty() {
        return Capabilities::default();
    }
    let caps = |branch, transcript| Capabilities {
        branch,
        transcript,
        tools: Some(ToolRecord::from_request(request)),
    };

    if task_type == Some(TaskType::Answer) && !text::has_assistant_turn(&turns) {
        return Capabilities {
            tools: None,
            ..caps(
                Branch::Answer,
                Some(render::render_session(&turns, attempt)),
            )
        };
    }
    // A tool trajectory is judged as agentic work unless typed conversational.
    if task_type == Some(TaskType::Agentic)
        || (text::has_tool_trajectory(request)
            && !matches!(task_type, Some(TaskType::Answer | TaskType::Chat)))
    {
        return caps(
            Branch::Agentic,
            Some(render::render_agentic_view(request, &turns, attempt)),
        );
    }
    if text::observed_hardening(attempt) || task_type == Some(TaskType::Coding) {
        return caps(Branch::Coding, None);
    }
    let transcript = Some(render::render_session(&turns, attempt));
    if text::observed_tool_activity(attempt) {
        caps(Branch::Agentic, transcript)
    } else if matches!(task_type, Some(TaskType::Chat | TaskType::Answer)) {
        caps(Branch::Chat, transcript)
    } else {
        caps(Branch::DefaultVerified, transcript)
    }
}
