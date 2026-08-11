// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stage-router projection from cumulative Prometheus metric families.

use std::collections::BTreeMap;

use prometheus::proto::{Metric, MetricFamily};
use serde::Serialize;

const ROUTING_DECISIONS_METRIC: &str = "switchyard_stage_router_routing_decisions_total";
const SCORE_METRIC: &str = "switchyard_stage_router_score";
const CONFIDENCE_METRIC: &str = "switchyard_stage_router_confidence";
const SEVERITY_METRIC: &str = "switchyard_stage_router_severity";
const SPINNING_METRIC: &str = "switchyard_stage_router_spinning";
const EXPLORING_METRIC: &str = "switchyard_stage_router_exploring";
const PRODUCTION_INTENSITY_METRIC: &str = "switchyard_stage_router_production_intensity";

#[derive(Clone, Debug, Default)]
pub(super) struct StageRouterCumulative {
    decisions: BTreeMap<DecisionKey, u64>,
    score: HistogramTotal,
    confidence: HistogramTotal,
    severity: HistogramTotal,
    spinning: HistogramTotal,
    exploring: HistogramTotal,
    production_intensity: HistogramTotal,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct DecisionKey {
    source: String,
    target: String,
}

#[derive(Clone, Copy, Debug, Default)]
struct HistogramTotal {
    count: u64,
    sum: f64,
}

/// Human-readable stage-router stats derived from its native metrics.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub(crate) struct StageRouterStatsSnapshot {
    pub routing_decisions: BTreeMap<String, DecisionStatsSnapshot>,
    pub scoring: ScoringStatsSnapshot,
}

/// One decision source, retaining the semantic targets it selected.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub(crate) struct DecisionStatsSnapshot {
    pub total: u64,
    pub targets: BTreeMap<String, u64>,
}

/// Stage scorer output and input-dimension summaries.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub(crate) struct ScoringStatsSnapshot {
    pub score: MetricSummary,
    pub confidence: MetricSummary,
    pub dimensions: DimensionStatsSnapshot,
}

/// Input dimensions evaluated by the stage scorer.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub(crate) struct DimensionStatsSnapshot {
    pub severity: MetricSummary,
    pub spinning: MetricSummary,
    pub exploring: MetricSummary,
    pub production_intensity: MetricSummary,
}

/// Exact count and mean since the last JSON stats reset.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub(crate) struct MetricSummary {
    pub count: u64,
    pub mean: f64,
}

impl StageRouterCumulative {
    pub(super) fn collect(families: &[MetricFamily]) -> Self {
        Self {
            decisions: collect_decisions(families),
            score: collect_histogram(families, SCORE_METRIC),
            confidence: collect_histogram(families, CONFIDENCE_METRIC),
            severity: collect_histogram(families, SEVERITY_METRIC),
            spinning: collect_histogram(families, SPINNING_METRIC),
            exploring: collect_histogram(families, EXPLORING_METRIC),
            production_intensity: collect_histogram(families, PRODUCTION_INTENSITY_METRIC),
        }
    }

    pub(super) fn delta(&self, baseline: &Self) -> StageRouterStatsSnapshot {
        let mut routing_decisions: BTreeMap<String, DecisionStatsSnapshot> = BTreeMap::new();
        for (key, current) in &self.decisions {
            let count = current.saturating_sub(*baseline.decisions.get(key).unwrap_or(&0));
            if count == 0 {
                continue;
            }
            let source = routing_decisions.entry(key.source.clone()).or_default();
            source.total = source.total.saturating_add(count);
            source.targets.insert(key.target.clone(), count);
        }
        StageRouterStatsSnapshot {
            routing_decisions,
            scoring: ScoringStatsSnapshot {
                score: self.score.delta(baseline.score),
                confidence: self.confidence.delta(baseline.confidence),
                dimensions: DimensionStatsSnapshot {
                    severity: self.severity.delta(baseline.severity),
                    spinning: self.spinning.delta(baseline.spinning),
                    exploring: self.exploring.delta(baseline.exploring),
                    production_intensity: self
                        .production_intensity
                        .delta(baseline.production_intensity),
                },
            },
        }
    }
}

impl HistogramTotal {
    fn delta(self, baseline: Self) -> MetricSummary {
        let count = self.count.saturating_sub(baseline.count);
        if count == 0 {
            return MetricSummary::default();
        }
        let sum = round4(self.sum - baseline.sum);
        MetricSummary {
            count,
            mean: round4(sum / count as f64),
        }
    }
}

fn collect_decisions(families: &[MetricFamily]) -> BTreeMap<DecisionKey, u64> {
    let mut decisions = BTreeMap::new();
    for metric in metrics(families, ROUTING_DECISIONS_METRIC) {
        let Some(source) = label(metric, "decision_source") else {
            continue;
        };
        let Some(target) = label(metric, "target_name") else {
            continue;
        };
        let Some(counter) = metric.get_counter().as_ref() else {
            continue;
        };
        let value = counter.value();
        if value.is_finite() && value > 0.0 {
            let count = decisions
                .entry(DecisionKey {
                    source: source.to_string(),
                    target: target.to_string(),
                })
                .or_insert(0u64);
            *count = count.saturating_add(value as u64);
        }
    }
    decisions
}

fn collect_histogram(families: &[MetricFamily], name: &str) -> HistogramTotal {
    metrics(families, name).fold(HistogramTotal::default(), |mut total, metric| {
        let Some(histogram) = metric.get_histogram().as_ref() else {
            return total;
        };
        total.count = total.count.saturating_add(histogram.sample_count());
        total.sum += histogram.sample_sum();
        total
    })
}

fn metrics<'a>(families: &'a [MetricFamily], name: &'a str) -> impl Iterator<Item = &'a Metric> {
    families
        .iter()
        .filter(move |family| family.name() == name)
        .flat_map(|family| family.get_metric())
}

fn label<'a>(metric: &'a Metric, name: &str) -> Option<&'a str> {
    metric
        .get_label()
        .iter()
        .find(|label| label.name() == name)
        .map(|label| label.value())
}

fn round4(value: f64) -> f64 {
    let rounded = (value * 10_000.0).round() / 10_000.0;
    if rounded == 0.0 { 0.0 } else { rounded }
}

#[cfg(test)]
mod tests {
    use opentelemetry::KeyValue;
    use opentelemetry::metrics::MeterProvider as _;
    use opentelemetry_sdk::metrics::SdkMeterProvider;
    use prometheus::Registry;

    use super::*;
    use crate::stats::StatsAccumulator;

    #[test]
    fn stage_router_projection_preserves_decisions_scores_and_reset_baseline() {
        let registry = Registry::new();
        let exporter = opentelemetry_prometheus::exporter()
            .with_registry(registry.clone())
            .build()
            .unwrap_or_else(|error| panic!("failed to build metrics exporter: {error}"));
        let provider = SdkMeterProvider::builder().with_reader(exporter).build();
        let meter = provider.meter("switchyard");
        let stats = StatsAccumulator::new(registry, ["stage_router"]);

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
            .algorithm_stats
            .stage_router
            .unwrap_or_else(|| panic!("stage-router stats missing"));
        let dimensions = &stage.routing_decisions["dimensions"];
        assert_eq!(dimensions.total, 3);
        assert_eq!(dimensions.targets["model/efficient"], 2);
        assert_eq!(dimensions.targets["model/capable"], 1);
        assert_eq!(stage.scoring.score.count, 2);
        assert_eq!(stage.scoring.score.mean, 0.125);
        assert_eq!(stage.scoring.confidence.mean, 0.75);

        stats.reset();
        assert_eq!(
            stats.snapshot().algorithm_stats.stage_router,
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
            .algorithm_stats
            .stage_router
            .unwrap_or_else(|| panic!("stage-router stats missing after reset"));
        assert_eq!(after_reset.routing_decisions["override"].total, 1);
        assert!(!after_reset.routing_decisions.contains_key("dimensions"));
    }
}
