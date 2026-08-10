// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Server-owned projections of algorithm OpenTelemetry metrics.

mod stage_router;

use std::collections::BTreeSet;
use std::sync::Arc;

use parking_lot::Mutex;
use prometheus::Registry;
use serde::Serialize;

use stage_router::{StageRouterCumulative, StageRouterStatsSnapshot};

const STAGE_ROUTER: &str = "stage_router";

/// Cumulative algorithm metrics and the baseline used by `/v1/stats/reset`.
#[derive(Clone)]
pub(crate) struct AlgorithmStats {
    inner: Arc<AlgorithmStatsInner>,
}

impl Default for AlgorithmStats {
    fn default() -> Self {
        Self::new(Registry::new(), std::iter::empty())
    }
}

struct AlgorithmStatsInner {
    registry: Registry,
    configured: BTreeSet<String>,
    baseline: Mutex<AlgorithmMetrics>,
}

#[derive(Clone, Debug, Default)]
struct AlgorithmMetrics {
    stage_router: StageRouterCumulative,
}

/// Curated algorithm-specific data included in the JSON stats response.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub(crate) struct AlgorithmStatsSnapshot {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage_router: Option<StageRouterStatsSnapshot>,
}

impl AlgorithmStats {
    /// Starts algorithm stats at the registry's current cumulative values.
    pub(crate) fn new(registry: Registry, configured: impl IntoIterator<Item = String>) -> Self {
        let baseline = collect(&registry);
        Self {
            inner: Arc::new(AlgorithmStatsInner {
                registry,
                configured: configured.into_iter().collect(),
                baseline: Mutex::new(baseline),
            }),
        }
    }

    /// Projects cumulative OpenTelemetry metrics since the last baseline.
    pub(crate) fn snapshot(&self) -> AlgorithmStatsSnapshot {
        let current = collect(&self.inner.registry);
        let baseline = self.inner.baseline.lock();
        AlgorithmStatsSnapshot {
            stage_router: self
                .inner
                .configured
                .contains(STAGE_ROUTER)
                .then(|| current.stage_router.delta(&baseline.stage_router)),
        }
    }

    /// Moves the JSON baseline without resetting process-lifetime OpenTelemetry metrics.
    pub(crate) fn reset(&self) {
        *self.inner.baseline.lock() = collect(&self.inner.registry);
    }
}

fn collect(registry: &Registry) -> AlgorithmMetrics {
    let families = registry.gather();
    AlgorithmMetrics {
        stage_router: StageRouterCumulative::collect(&families),
    }
}
