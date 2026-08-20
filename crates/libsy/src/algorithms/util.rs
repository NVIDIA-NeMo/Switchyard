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
