// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Verification-gated routing capability derivation.
//!
//! Capabilities come only from router-produced task typing, the normalized
//! request, and the local attempt. Missing or unrepresentable context selects
//! [`Branch::Unknown`], so later routing fails closed.

#![allow(dead_code)]

use switchyard_protocol::Request;

mod config;
mod decide;
mod readout;
mod runtime;
mod safety;
mod text;

#[cfg(test)]
mod tests;

/// Verification regime selected from trusted request and attempt evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Branch {
    /// Code-producing work without an executable checker.
    Coding,
    /// Conversational or answer-seeking work.
    Chat,
    /// Tool-operating work.
    Agentic,
    /// Untyped work with a complete judged view.
    DefaultVerified,
    /// No complete evidence is available to verify.
    Unknown,
}

/// Task type produced by the router's own typing call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TaskType {
    /// Code-producing work.
    Coding,
    /// Tool-operating work.
    Agentic,
    /// Answer-seeking work.
    Answer,
    /// Conversational work.
    Chat,
}

/// Tool-error count together with its trust provenance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ToolErrorCount {
    /// Count from host-owned execution evidence.
    Host(i32),
    /// Count derived from untrusted request history.
    Untrusted(i32),
}

/// Complete input visible to the decision policy.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Capabilities {
    /// Latest user instruction.
    pub task_text: Option<String>,
    /// Bounded, redacted request and local attempt.
    pub transcript: Option<String>,
    /// The request is coding work.
    pub is_coding: bool,
    /// The request is conversational or answer-seeking.
    pub is_chat: bool,
    /// The request is agentic.
    pub is_agentic: bool,
    /// No narrower regime was established.
    pub default_verified: bool,
    /// Locally generated attempt.
    pub attempt: Option<String>,
    /// Tool-error count supplied by the routing host or untrusted history.
    pub tool_errors: Option<ToolErrorCount>,
}

/// Selects the strongest verification regime present in `caps`.
pub fn select_branch(caps: &Capabilities) -> Branch {
    if caps.is_coding {
        Branch::Coding
    } else if caps.is_chat {
        Branch::Chat
    } else if caps.is_agentic {
        Branch::Agentic
    } else if caps.default_verified {
        Branch::DefaultVerified
    } else {
        Branch::Unknown
    }
}

/// Derives policy capabilities without accepting client-declared routing flags.
pub fn derive_capabilities(
    request: &Request,
    attempt: &str,
    task_type: Option<TaskType>,
    tool_errors: Option<ToolErrorCount>,
) -> Capabilities {
    let Some(view) = text::request_view(request, attempt) else {
        return Capabilities::default();
    };
    let mut caps = Capabilities {
        task_text: Some(view.task_text),
        transcript: Some(view.transcript),
        attempt: Some(attempt.to_string()),
        tool_errors,
        ..Default::default()
    };

    if text::observed_code_activity(attempt) || task_type == Some(TaskType::Coding) {
        caps.is_coding = true;
    } else if text::observed_tool_activity(attempt) || task_type == Some(TaskType::Agentic) {
        caps.is_agentic = true;
    } else if matches!(task_type, Some(TaskType::Answer | TaskType::Chat)) {
        caps.is_chat = true;
    } else {
        caps.default_verified = true;
    }
    caps
}
