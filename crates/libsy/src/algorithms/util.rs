// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod affinity;
pub mod escalation;
pub(crate) mod llm_judge;
pub(crate) mod prompts;
pub(crate) mod stage;
pub mod subagent;
pub(crate) mod tool_signals;

/// Default completion budget for internal classifier and escalation judge calls.
pub(crate) const DEFAULT_JUDGE_MAX_OUTPUT_TOKENS: u64 = 4_096;
