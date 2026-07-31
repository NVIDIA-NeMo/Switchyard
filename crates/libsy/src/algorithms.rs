// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Concrete algorithms and the interfaces for building them.
//!
//! Reach for them by name — `use switchyard_libsy::algorithms::Random` — rather than through the
//! per-algorithm submodules.

mod fall_through;
pub mod llm_class;
pub mod noop;
pub mod passthrough;
pub mod rand;
pub mod stage;

pub(crate) use fall_through::{DefaultTarget, FallThrough};
pub use llm_class::{EscalationJudgeConfig, LlmTaskClassifier, TaskClassifierConfig};
pub use noop::{Noop, NoopDecision};
pub use passthrough::{Passthrough, PassthroughDecision};
pub use rand::{Random, RandomClassifier, RandomDecision};
pub use stage::{LlmFallback, StageRouter, StageRouterConfig};
pub use util::{
    append_note, AffinityRouter, SubagentOverride, SystemPromptProcessor, TargetPrompts,
};

pub mod util;

#[cfg(test)]
mod subagent_affinity_tests;
