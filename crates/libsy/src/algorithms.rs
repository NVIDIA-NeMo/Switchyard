// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Concrete algorithms and the interfaces for building them.
//!
//! Reach for them by name — `use switchyard_libsy::algorithms::Random` — rather than through the
//! per-algorithm submodules.

pub mod affinity;
pub mod fall_through;
pub mod llm_class;
pub mod noop;
pub mod rand;
pub mod subagent_override;

pub use affinity::AffinityRouter;
pub use fall_through::{FallThrough, FallThroughDecision};
pub use llm_class::{ClassifierDecision, ClassifierTier, LlmClassifier};
pub use noop::{Noop, NoopDecision};
pub use rand::{Random, RandomDecision};
pub use subagent_override::{SubagentDecision, SubagentOverride};
