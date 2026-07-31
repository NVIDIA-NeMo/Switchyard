// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Concrete algorithms and the interfaces for building them.
//!
//! Everything public here is re-exported at the crate root; reach for it by name —
//! `use switchyard_libsy::Random`.

pub mod fall_through;
pub mod llm_class;
pub mod noop;
pub mod passthrough;
pub mod rand;
pub mod stage;

pub mod util;

#[cfg(test)]
mod subagent_affinity_tests;
