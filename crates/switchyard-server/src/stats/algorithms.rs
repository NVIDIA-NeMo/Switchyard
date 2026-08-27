// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Server-owned projections of algorithm OpenTelemetry metrics.

mod advisor_gate;
mod stage_router;

use std::collections::HashSet;

use prometheus::Registry;
use serde::Serialize;

use advisor_gate::{AdvisorGateCumulative, AdvisorGateStatsSnapshot};
use stage_router::{StageRouterCumulative, StageRouterStatsSnapshot};

const ADVISOR_GATE: &str = "advisor_gate";
const STAGE_ROUTER: &str = "stage_router";

/// Owns algorithm metric baselines behind the generic server stats interface.
pub(super) struct AlgorithmStats {
    registry: Registry,
    advisor_gate_baseline: Option<AdvisorGateCumulative>,
    stage_router_baseline: Option<StageRouterCumulative>,
}

/// Curated algorithm-specific data included in the JSON stats response.
///
/// Each block is present only when the deployment contains a route running
/// the matching algorithm; deployments without one omit the key entirely.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub(crate) struct AlgorithmStatsSnapshot {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub advisor_gate: Option<AdvisorGateStatsSnapshot>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage_router: Option<StageRouterStatsSnapshot>,
}

impl AlgorithmStats {
    pub(super) fn new<'a>(
        registry: Registry,
        algorithms: impl IntoIterator<Item = &'a str>,
    ) -> Self {
        let algorithms: HashSet<_> = algorithms.into_iter().collect();
        let families = registry.gather();
        Self {
            advisor_gate_baseline: algorithms
                .contains(ADVISOR_GATE)
                .then(|| AdvisorGateCumulative::collect(&families)),
            stage_router_baseline: algorithms
                .contains(STAGE_ROUTER)
                .then(|| StageRouterCumulative::collect(&families)),
            registry,
        }
    }

    pub(super) fn snapshot(&self) -> AlgorithmStatsSnapshot {
        let families = self.registry.gather();
        AlgorithmStatsSnapshot {
            advisor_gate: self
                .advisor_gate_baseline
                .as_ref()
                .map(|baseline| AdvisorGateCumulative::collect(&families).delta(baseline)),
            stage_router: self
                .stage_router_baseline
                .as_ref()
                .map(|baseline| StageRouterCumulative::collect(&families).delta(baseline)),
        }
    }

    pub(super) fn reset(&mut self) {
        if let Some(baseline) = &mut self.advisor_gate_baseline {
            *baseline = AdvisorGateCumulative::collect(&self.registry.gather());
        }
        if let Some(baseline) = &mut self.stage_router_baseline {
            *baseline = StageRouterCumulative::collect(&self.registry.gather());
        }
    }
}
