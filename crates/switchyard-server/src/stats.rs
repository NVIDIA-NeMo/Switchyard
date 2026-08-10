// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-local JSON stats accounting for the Rust server.

mod accumulator;
mod algorithms;
mod cache_eligibility;

pub(crate) use accumulator::{StatsAccumulator, StatsSnapshot, TokenUsage};
pub(crate) use cache_eligibility::{prefix_probe, tracking_enabled_from_env};
