// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Server-owned projections of algorithm OpenTelemetry metrics.

mod stage_router;

use serde::Serialize;

pub(super) use stage_router::StageRouterCumulative;
use stage_router::StageRouterStatsSnapshot;

/// Curated algorithm-specific data included in the JSON stats response.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub(crate) struct AlgorithmStatsSnapshot {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage_router: Option<StageRouterStatsSnapshot>,
}
