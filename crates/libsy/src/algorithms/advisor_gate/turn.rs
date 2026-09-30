// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Extract reply text and tool-use signals for advisor review.

use switchyard_protocol::{AggLlmResponse, ContentBlock, StopReason};

// ── Response inspection ────────────────────────────────────────────────────

/// Whether the turn carries tool use on either signal: a `ToolUse` stop
/// reason, or any tool-call block (some OSS servers mislabel tool-call turns
/// as an ordinary stop, so block presence wins).
pub(super) fn has_tool_use(agg: &AggLlmResponse) -> bool {
    agg.outputs.iter().any(|output| {
        output.stop_reason == Some(StopReason::ToolUse)
            || output
                .content
                .iter()
                .any(|block| matches!(block, ContentBlock::ToolCall(_)))
    })
}

/// The turn's visible text: all text blocks joined; empty means none.
pub(super) fn visible_text(agg: &AggLlmResponse) -> Option<String> {
    let text: Vec<&str> = agg
        .outputs
        .iter()
        .flat_map(|output| output.content.iter())
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    if text.is_empty() {
        return None;
    }
    let joined = text.join("\n");
    if joined.is_empty() {
        None
    } else {
        Some(joined)
    }
}

/// The turn's internal reasoning, the review evidence of last resort.
pub(super) fn reasoning_text(agg: &AggLlmResponse) -> Option<String> {
    let text: Vec<&str> = agg
        .outputs
        .iter()
        .flat_map(|output| output.content.iter())
        .filter_map(|block| match block {
            ContentBlock::Reasoning { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    if text.is_empty() {
        return None;
    }
    let joined = text.join("\n");
    if joined.is_empty() {
        None
    } else {
        Some(joined)
    }
}
