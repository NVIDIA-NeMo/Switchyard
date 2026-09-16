// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! VGR decision projection from cumulative Prometheus counters.

use std::collections::BTreeMap;

use prometheus::proto::{Metric, MetricFamily};
use serde::Serialize;

const VGR_DECISIONS_METRIC: &str = "switchyard_vgr_decisions_total";

#[derive(Clone, Debug, Default)]
pub(super) struct VgrCumulative {
    decisions: BTreeMap<DecisionKey, u64>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct DecisionKey {
    predicted: String,
    effective: String,
    served: String,
    branch: String,
    readiness_gate: String,
    short_circuit: String,
}

/// Bounded VGR decision counts since the last stats reset.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub(crate) struct VgrStatsSnapshot {
    pub routing_decisions: Vec<VgrDecisionStatsSnapshot>,
}

/// One unique combination of policy and serving outcomes.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct VgrDecisionStatsSnapshot {
    pub predicted: String,
    pub effective: String,
    pub served: String,
    pub branch: String,
    pub readiness_gate: String,
    pub short_circuit: String,
    pub total: u64,
}

impl VgrCumulative {
    pub(super) fn collect(families: &[MetricFamily]) -> Self {
        Self {
            decisions: collect_decisions(families),
        }
    }

    pub(super) fn delta(&self, baseline: &Self) -> VgrStatsSnapshot {
        let routing_decisions = self
            .decisions
            .iter()
            .filter_map(|(key, current)| {
                let total = current.saturating_sub(*baseline.decisions.get(key).unwrap_or(&0));
                (total > 0).then(|| VgrDecisionStatsSnapshot {
                    predicted: key.predicted.clone(),
                    effective: key.effective.clone(),
                    served: key.served.clone(),
                    branch: key.branch.clone(),
                    readiness_gate: key.readiness_gate.clone(),
                    short_circuit: key.short_circuit.clone(),
                    total,
                })
            })
            .collect();
        VgrStatsSnapshot { routing_decisions }
    }
}

fn collect_decisions(families: &[MetricFamily]) -> BTreeMap<DecisionKey, u64> {
    let mut decisions = BTreeMap::new();
    for metric in metrics(families, VGR_DECISIONS_METRIC) {
        let Some(key) = decision_key(metric) else {
            continue;
        };
        let Some(counter) = metric.get_counter().as_ref() else {
            continue;
        };
        let value = counter.value();
        if value.is_finite() && value > 0.0 {
            let count = decisions.entry(key).or_insert(0u64);
            *count = count.saturating_add(value as u64);
        }
    }
    decisions
}

fn decision_key(metric: &Metric) -> Option<DecisionKey> {
    Some(DecisionKey {
        predicted: label(metric, "predicted")?.to_string(),
        effective: label(metric, "effective")?.to_string(),
        served: label(metric, "served")?.to_string(),
        branch: label(metric, "branch")?.to_string(),
        readiness_gate: label(metric, "readiness_gate")?.to_string(),
        short_circuit: label(metric, "short_circuit")?.to_string(),
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

#[cfg(test)]
mod tests {
    use opentelemetry::KeyValue;
    use opentelemetry::metrics::MeterProvider as _;
    use opentelemetry_sdk::metrics::SdkMeterProvider;
    use prometheus::Registry;

    use crate::stats::StatsAccumulator;

    #[test]
    fn vgr_projection_preserves_bounded_decision_labels_and_reset_baseline() {
        let registry = Registry::new();
        let exporter = opentelemetry_prometheus::exporter()
            .with_registry(registry.clone())
            .build()
            .unwrap_or_else(|error| panic!("failed to build metrics exporter: {error}"));
        let provider = SdkMeterProvider::builder().with_reader(exporter).build();
        let meter = provider.meter("switchyard");
        let stats = StatsAccumulator::new(registry, ["vgr"]);
        let labels = [
            KeyValue::new("predicted", "local"),
            KeyValue::new("effective", "cloud"),
            KeyValue::new("served", "cloud"),
            KeyValue::new("branch", "checks"),
            KeyValue::new("readiness_gate", "secure_checker_missing"),
            KeyValue::new("short_circuit", "none"),
        ];

        meter
            .u64_counter("switchyard.vgr.decisions")
            .build()
            .add(2, &labels);

        let first = stats.snapshot();
        let vgr = first.algorithm_stats.vgr.as_ref().expect("vgr stats");
        let decision = &vgr.routing_decisions[0];
        assert_eq!(decision.predicted, "local");
        assert_eq!(decision.effective, "cloud");
        assert_eq!(decision.served, "cloud");
        assert_eq!(decision.branch, "checks");
        assert_eq!(decision.readiness_gate, "secure_checker_missing");
        assert_eq!(decision.short_circuit, "none");
        assert_eq!(decision.total, 2);

        stats.reset();
        assert!(
            stats
                .snapshot()
                .algorithm_stats
                .vgr
                .expect("vgr stats")
                .routing_decisions
                .is_empty()
        );
    }
}
