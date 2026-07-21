// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Concrete algorithms and the interfaces for building them.
//!
//! Reach for them by name — `use libsy::algorithms::Random` — rather than through the
//! per-algorithm submodules.

pub mod core;
pub mod llm_class;
pub mod noop;
pub mod rand;

pub use core::*;
pub use llm_class::{ClassifierDecision, ClassifierTier, LlmClassifierOrch};
pub use noop::{Noop, NoopDecision};
pub use rand::{Random, RandomDecision};
