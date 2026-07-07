// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared stats surface for profile-owned runtimes.

mod stream_tap;

use std::sync::OnceLock;

use switchyard_components::StatsAccumulator;

pub(crate) use stream_tap::{record_usage_or_tap_stream, UsageAttribution};

/// Returns the process-wide stats accumulator used by v2 profiles.
pub fn profile_stats_accumulator() -> StatsAccumulator {
    static PROFILE_STATS: OnceLock<StatsAccumulator> = OnceLock::new();
    PROFILE_STATS.get_or_init(StatsAccumulator::new).clone()
}
