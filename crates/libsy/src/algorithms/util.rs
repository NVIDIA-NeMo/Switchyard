// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod affinity;
pub(crate) mod classifier_contract;
pub mod escalation;
pub(crate) mod llm_judge;
pub(crate) mod prompts;
pub(crate) mod robustness;
pub(crate) mod stage;
pub mod subagent;
pub(crate) mod target_selector;
#[cfg(test)]
pub(crate) mod tier_fixtures;
pub(crate) mod tool_signals;

use switchyard_protocol::ModelId;

use crate::core::classifier::{Classification, Score};

/// A single full-confidence recommendation.
pub(crate) fn decisive(target: &ModelId) -> Classification {
    Classification::Scores(vec![Score {
        target: target.clone(),
        confidence: 1.0,
    }])
}

/// Default completion budget for internal classifier and escalation judge calls.
pub(crate) const DEFAULT_JUDGE_MAX_OUTPUT_TOKENS: u64 = 4_096;

/// Default character budget for a judge payload, shared by the escalation judge and the
/// windowed classifier judges so one turn carrying a large tool result cannot decide how
/// much a judge call costs.
pub(crate) const DEFAULT_JUDGE_CHAR_BUDGET: usize = 18_000;

/// Separator marking where [`truncate_middle`] dropped a message's interior.
pub(crate) const TRIM_MARKER: &str = " ...[trimmed] ";

/// Keeps the head and tail of `text` within `limit` characters.
///
/// The head gets two thirds of the surviving budget: for a judge reading agent activity the
/// command or error signature that opens a message carries more signal than its trailing
/// output. Clipping is marked so the judge can tell a trimmed message from a short one.
pub(crate) fn truncate_middle(text: &str, limit: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= limit {
        return text.to_string();
    }
    // Below the marker's own width there is no room to say the text was clipped, so keep
    // what fits and drop the marker. Marking anyway would push the result past `limit`,
    // which callers budgeting a payload rely on it never doing.
    let marker = TRIM_MARKER.chars().count();
    if limit <= marker {
        return chars[..limit].iter().collect();
    }
    let keep = limit - marker;
    let head = keep * 2 / 3;
    let tail = keep - head;
    let mut out: String = chars[..head].iter().collect();
    out.push_str(TRIM_MARKER);
    out.extend(chars[chars.len() - tail..].iter());
    out
}
