// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Server-owned projections of algorithm OpenTelemetry metrics.

mod stage_router;

use std::collections::BTreeSet;

use prometheus::Registry;
use serde::Serialize;

use stage_router::{StageRouterCumulative, StageRouterStatsSnapshot};

pub(super) const STAGE_ROUTER: &str = "stage_router";

#[derive(Clone, Debug, Default)]
pub(super) struct AlgorithmMetrics {
    stage_router: StageRouterCumulative,
}

/// Curated algorithm-specific data included in the JSON stats response.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub(crate) struct AlgorithmStatsSnapshot {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage_router: Option<StageRouterStatsSnapshot>,
}

impl AlgorithmMetrics {
    pub(super) fn collect(registry: &Registry) -> Self {
        let families = registry.gather();
        Self {
            stage_router: StageRouterCumulative::collect(&families),
        }
    }

    pub(super) fn snapshot(
        &self,
        baseline: &Self,
        configured: &BTreeSet<String>,
    ) -> AlgorithmStatsSnapshot {
        AlgorithmStatsSnapshot {
            stage_router: configured
                .contains(STAGE_ROUTER)
                .then(|| self.stage_router.delta(&baseline.stage_router)),
        }
    }
}
