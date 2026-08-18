// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Advisor-gate projection from cumulative Prometheus metric families.

use std::collections::BTreeMap;

use prometheus::proto::{Metric, MetricFamily};
use serde::Serialize;

const REVIEWS_METRIC: &str = "switchyard_advisor_gate_reviews_total";
const CONSULT_FAILURES_METRIC: &str = "switchyard_advisor_gate_consult_failures_total";
const DISCARDED_TURNS_METRIC: &str = "switchyard_advisor_gate_discarded_turns_total";
const DISCARDED_TOKENS_METRIC: &str = "switchyard_advisor_gate_discarded_tokens_total";

#[derive(Clone, Debug, Default)]
pub(super) struct AdvisorGateCumulative {
    reviews: BTreeMap<ReviewKey, u64>,
    consult_failures: BTreeMap<String, u64>,
    discarded_turns: u64,
    discarded_tokens: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ReviewKey {
    verdict: String,
    trigger: String,
}

/// Human-readable advisor-gate stats derived from its native metrics.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub(crate) struct AdvisorGateStatsSnapshot {
    /// Verdicts handed down since the last reset, split by what gated the turn.
    pub reviews: BTreeMap<String, ReviewStatsSnapshot>,
    /// Advisor consults that failed outright, by bounded reason label.
    pub consult_failures: BTreeMap<String, u64>,
    /// Executor turns (and their tokens) discarded by REDO verdicts; the
    /// client never saw them, so terminal usage accounting never priced them.
    pub discarded: DiscardedStatsSnapshot,
}

/// One verdict's counts, split by trigger.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub(crate) struct ReviewStatsSnapshot {
    pub total: u64,
    pub by_trigger: BTreeMap<String, u64>,
}

/// REDO-discarded executor turns and their token kinds.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub(crate) struct DiscardedStatsSnapshot {
    pub turns: u64,
    pub tokens: BTreeMap<String, u64>,
}

impl AdvisorGateCumulative {
    pub(super) fn collect(families: &[MetricFamily]) -> Self {
        Self {
            reviews: collect_labeled_pairs(families, REVIEWS_METRIC, "verdict", "trigger")
                .into_iter()
                .map(|((verdict, trigger), count)| (ReviewKey { verdict, trigger }, count))
                .collect(),
            consult_failures: collect_labeled(families, CONSULT_FAILURES_METRIC, "reason"),
            discarded_turns: collect_total(families, DISCARDED_TURNS_METRIC),
            discarded_tokens: collect_labeled(families, DISCARDED_TOKENS_METRIC, "kind"),
        }
    }

    pub(super) fn delta(&self, baseline: &Self) -> AdvisorGateStatsSnapshot {
        let mut reviews: BTreeMap<String, ReviewStatsSnapshot> = BTreeMap::new();
        for (key, current) in &self.reviews {
            let count = current.saturating_sub(*baseline.reviews.get(key).unwrap_or(&0));
            if count == 0 {
                continue;
            }
            let verdict = reviews.entry(key.verdict.clone()).or_default();
            verdict.total = verdict.total.saturating_add(count);
            verdict.by_trigger.insert(key.trigger.clone(), count);
        }
        AdvisorGateStatsSnapshot {
            reviews,
            consult_failures: map_delta(&self.consult_failures, &baseline.consult_failures),
            discarded: DiscardedStatsSnapshot {
                turns: self
                    .discarded_turns
                    .saturating_sub(baseline.discarded_turns),
                tokens: map_delta(&self.discarded_tokens, &baseline.discarded_tokens),
            },
        }
    }
}

fn map_delta(
    current: &BTreeMap<String, u64>,
    baseline: &BTreeMap<String, u64>,
) -> BTreeMap<String, u64> {
    current
        .iter()
        .filter_map(|(key, value)| {
            let delta = value.saturating_sub(*baseline.get(key).unwrap_or(&0));
            (delta > 0).then(|| (key.clone(), delta))
        })
        .collect()
}

fn collect_labeled(
    families: &[MetricFamily],
    metric_name: &str,
    label_name: &str,
) -> BTreeMap<String, u64> {
    let mut counts = BTreeMap::new();
    for metric in metrics(families, metric_name) {
        let Some(key) = label(metric, label_name) else {
            continue;
        };
        if let Some(value) = counter_value(metric) {
            let count = counts.entry(key.to_string()).or_insert(0u64);
            *count = count.saturating_add(value);
        }
    }
    counts
}

fn collect_labeled_pairs(
    families: &[MetricFamily],
    metric_name: &str,
    first_label: &str,
    second_label: &str,
) -> BTreeMap<(String, String), u64> {
    let mut counts = BTreeMap::new();
    for metric in metrics(families, metric_name) {
        let (Some(first), Some(second)) = (label(metric, first_label), label(metric, second_label))
        else {
            continue;
        };
        if let Some(value) = counter_value(metric) {
            let count = counts
                .entry((first.to_string(), second.to_string()))
                .or_insert(0u64);
            *count = count.saturating_add(value);
        }
    }
    counts
}

fn collect_total(families: &[MetricFamily], metric_name: &str) -> u64 {
    metrics(families, metric_name)
        .filter_map(counter_value)
        .fold(0u64, u64::saturating_add)
}

fn counter_value(metric: &Metric) -> Option<u64> {
    let counter = metric.get_counter().as_ref()?;
    let value = counter.value();
    (value.is_finite() && value > 0.0).then_some(value as u64)
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

    use super::*;
    use crate::stats::StatsAccumulator;

    #[test]
    fn advisor_gate_projection_preserves_reviews_discards_and_reset_baseline() {
        let registry = Registry::new();
        let exporter = opentelemetry_prometheus::exporter()
            .with_registry(registry.clone())
            .build()
            .unwrap_or_else(|error| panic!("failed to build metrics exporter: {error}"));
        let provider = SdkMeterProvider::builder().with_reader(exporter).build();
        let meter = provider.meter("switchyard");
        let stats = StatsAccumulator::new(registry, ["advisor_gate"]);

        let reviews = meter.u64_counter("switchyard.advisor_gate.reviews").build();
        reviews.add(
            2,
            &[
                KeyValue::new("verdict", "approve"),
                KeyValue::new("trigger", "no_tool_call"),
            ],
        );
        reviews.add(
            1,
            &[
                KeyValue::new("verdict", "redo"),
                KeyValue::new("trigger", "stall"),
            ],
        );
        meter
            .u64_counter("switchyard.advisor_gate.consult_failures")
            .build()
            .add(1, &[KeyValue::new("reason", "client_error")]);
        meter
            .u64_counter("switchyard.advisor_gate.discarded_turns")
            .build()
            .add(1, &[]);
        let tokens = meter
            .u64_counter("switchyard.advisor_gate.discarded_tokens")
            .build();
        tokens.add(120, &[KeyValue::new("kind", "input")]);
        tokens.add(30, &[KeyValue::new("kind", "output")]);

        let snapshot = stats.snapshot();
        let gate = snapshot
            .algorithm_stats
            .advisor_gate
            .unwrap_or_else(|| panic!("advisor-gate stats missing"));
        assert_eq!(gate.reviews["approve"].total, 2);
        assert_eq!(gate.reviews["approve"].by_trigger["no_tool_call"], 2);
        assert_eq!(gate.reviews["redo"].by_trigger["stall"], 1);
        assert_eq!(gate.consult_failures["client_error"], 1);
        assert_eq!(gate.discarded.turns, 1);
        assert_eq!(gate.discarded.tokens["input"], 120);
        assert_eq!(gate.discarded.tokens["output"], 30);

        stats.reset();
        assert_eq!(
            stats.snapshot().algorithm_stats.advisor_gate,
            Some(AdvisorGateStatsSnapshot::default())
        );

        reviews.add(
            1,
            &[
                KeyValue::new("verdict", "unparseable"),
                KeyValue::new("trigger", "pattern"),
            ],
        );
        let after_reset = stats
            .snapshot()
            .algorithm_stats
            .advisor_gate
            .unwrap_or_else(|| panic!("advisor-gate stats missing after reset"));
        assert_eq!(after_reset.reviews["unparseable"].total, 1);
        assert!(!after_reset.reviews.contains_key("approve"));
    }
}
