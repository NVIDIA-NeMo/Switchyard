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

#[cfg(test)]
mod tests {
    use opentelemetry::KeyValue;
    use opentelemetry::metrics::MeterProvider as _;
    use opentelemetry_sdk::metrics::SdkMeterProvider;

    use super::*;

    #[test]
    fn stage_router_projection_preserves_decisions_scores_and_reset_baseline() {
        let registry = Registry::new();
        let exporter = opentelemetry_prometheus::exporter()
            .with_registry(registry.clone())
            .build()
            .unwrap_or_else(|error| panic!("failed to build metrics exporter: {error}"));
        let provider = SdkMeterProvider::builder().with_reader(exporter).build();
        let meter = provider.meter("switchyard");
        let stats = AlgorithmStats::new(registry, [STAGE_ROUTER.to_string()]);

        meter
            .u64_counter("switchyard.stage_router.routing_decisions")
            .build()
            .add(
                2,
                &[
                    KeyValue::new("decision_source", "dimensions"),
                    KeyValue::new("target_name", "model/efficient"),
                ],
            );
        meter
            .u64_counter("switchyard.stage_router.routing_decisions")
            .build()
            .add(
                1,
                &[
                    KeyValue::new("decision_source", "dimensions"),
                    KeyValue::new("target_name", "model/capable"),
                ],
            );
        for value in [0.5, -0.25] {
            meter
                .f64_histogram("switchyard.stage_router.score")
                .build()
                .record(value, &[]);
        }
        meter
            .f64_histogram("switchyard.stage_router.confidence")
            .build()
            .record(0.75, &[]);

        let snapshot = stats.snapshot();
        let stage = snapshot
            .stage_router
            .unwrap_or_else(|| panic!("stage-router stats missing"));
        let dimensions = &stage.routing_decisions["dimensions"];
        assert_eq!(dimensions.total, 3);
        assert_eq!(dimensions.targets["model/efficient"], 2);
        assert_eq!(dimensions.targets["model/capable"], 1);
        assert_eq!(stage.scoring.score.count, 2);
        assert_eq!(stage.scoring.score.sum, 0.25);
        assert_eq!(stage.scoring.score.avg, 0.125);
        assert_eq!(stage.scoring.confidence.avg, 0.75);

        stats.reset();
        assert_eq!(
            stats.snapshot().stage_router,
            Some(StageRouterStatsSnapshot::default())
        );

        meter
            .u64_counter("switchyard.stage_router.routing_decisions")
            .build()
            .add(
                1,
                &[
                    KeyValue::new("decision_source", "override"),
                    KeyValue::new("target_name", "model/capable"),
                ],
            );
        let after_reset = stats
            .snapshot()
            .stage_router
            .unwrap_or_else(|| panic!("stage-router stats missing after reset"));
        assert_eq!(after_reset.routing_decisions["override"].total, 1);
        assert!(!after_reset.routing_decisions.contains_key("dimensions"));
    }
}
