// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Query the same OpenTelemetry metric families used by Prometheus, without HTTP.

use std::collections::BTreeMap;

use opentelemetry::global;
use opentelemetry_sdk::metrics::{SdkMeterProvider, Stream};
use prometheus::proto::{Metric, MetricFamily, MetricType};
use prometheus::{Registry, TextEncoder};
use serde::Serialize;

/// Selects metric series. Unspecified fields match all values.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetricFilter<'a> {
    /// Session whose metrics to return.
    pub session_id: Option<&'a str>,
    /// Agent within the session; absent agent IDs are recorded as `root`.
    pub agent_id: Option<&'a str>,
    /// Zero-based inference request index.
    pub agent_turn_idx: Option<i64>,
    /// Zero-based user turn index, shared by its tool continuations.
    pub user_turn_idx: Option<i64>,
}

impl MetricFilter<'_> {
    fn matches(&self, metric: &Metric) -> bool {
        let agent_turn = self.agent_turn_idx.map(|value| value.to_string());
        let user_turn = self.user_turn_idx.map(|value| value.to_string());
        [
            ("session_id", self.session_id),
            ("agent_id", self.agent_id),
            ("agent_turn_idx", agent_turn.as_deref()),
            ("user_turn_idx", user_turn.as_deref()),
        ]
        .into_iter()
        .all(|(key, expected)| {
            expected.is_none_or(|value| {
                metric
                    .get_label()
                    .iter()
                    .any(|label| label.name() == key && label.value() == value)
            })
        })
    }
}

/// An in-process OpenTelemetry provider and metric query handle.
///
/// Keep one collector for the application's lifetime. `new` does not change the
/// global provider; install `meter_provider().clone()` on the host before running
/// requests, or use `install`. Collection is synchronous and requires no runtime
/// or server. All snapshots are cumulative since provider creation.
pub struct Metrics {
    registry: Registry,
    provider: SdkMeterProvider,
}

impl Metrics {
    /// Creates a collector using the SDK's default limit of 2000 series per instrument.
    pub fn new() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::with_cardinality_limit(2000)
    }

    /// Creates and installs a collector as the global OpenTelemetry meter provider.
    /// Install once, before running requests; this replaces any existing provider.
    pub fn install() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let metrics = Self::new()?;
        global::set_meter_provider(metrics.provider.clone());
        Ok(metrics)
    }

    /// Creates a collector with a chosen series limit per instrument.
    ///
    /// Each session/agent/turn creates separate series. When the limit is reached,
    /// the SDK groups further series under `otel_metric_overflow`; those samples
    /// cannot be queried by their original turn identity.
    pub fn with_cardinality_limit(
        limit: usize,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Stream::builder()
            .with_cardinality_limit(limit)
            .build()
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        let registry = Registry::new();
        let reader = opentelemetry_prometheus::exporter()
            .with_registry(registry.clone())
            .build()?;
        let provider = SdkMeterProvider::builder()
            .with_reader(reader)
            .with_view(move |_| Stream::builder().with_cardinality_limit(limit).build().ok())
            .build();
        Ok(Self { registry, provider })
    }

    /// Provider the host can install globally before running requests.
    pub fn meter_provider(&self) -> &SdkMeterProvider {
        &self.provider
    }

    /// Collects all matching metric families, retaining their labels and buckets.
    pub fn gather(&self, filter: MetricFilter<'_>) -> Vec<MetricFamily> {
        self.registry
            .gather()
            .into_iter()
            .filter_map(|mut family| {
                let matching: Vec<_> = family
                    .get_metric()
                    .iter()
                    .filter(|metric| filter.matches(metric))
                    .cloned()
                    .collect();
                if matching.is_empty() {
                    return None;
                }
                family.set_metric(matching);
                Some(family)
            })
            .collect()
    }

    /// Returns Prometheus text, equivalent to a filtered `/metrics` response.
    pub fn prometheus(&self, filter: MetricFilter<'_>) -> prometheus::Result<String> {
        TextEncoder::new().encode_to_string(&self.gather(filter))
    }

    /// Returns typed, serializable stats for every matching metric series.
    /// Unlike the server's model summary, this preserves all metric names and labels.
    pub fn stats(&self, filter: MetricFilter<'_>) -> MetricsSnapshot {
        MetricsSnapshot {
            series: self
                .gather(filter)
                .iter()
                .flat_map(|family| {
                    family
                        .get_metric()
                        .iter()
                        .filter_map(|metric| {
                            Some(MetricSeries {
                                name: family.name().to_owned(),
                                labels: metric
                                    .get_label()
                                    .iter()
                                    .map(|label| {
                                        (label.name().to_owned(), label.value().to_owned())
                                    })
                                    .collect(),
                                value: match family.get_field_type() {
                                    MetricType::COUNTER => MetricValue::Counter {
                                        value: metric.get_counter().value(),
                                    },
                                    MetricType::GAUGE => MetricValue::Gauge {
                                        value: metric.get_gauge().value(),
                                    },
                                    MetricType::HISTOGRAM => {
                                        let histogram = metric.get_histogram();
                                        MetricValue::Histogram {
                                            count: histogram.sample_count(),
                                            sum: histogram.sample_sum(),
                                            buckets: histogram
                                                .get_bucket()
                                                .iter()
                                                .map(|bucket| HistogramBucket {
                                                    upper_bound: bucket.upper_bound(),
                                                    count: bucket.cumulative_count(),
                                                })
                                                .collect(),
                                        }
                                    }
                                    _ => return None,
                                },
                            })
                        })
                        .collect::<Vec<_>>()
                })
                .collect(),
        }
    }
}

/// Typed snapshot containing all selected metrics, including algorithm-specific instruments.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct MetricsSnapshot {
    /// Individual series retain model, outcome, request identity, and other labels.
    pub series: Vec<MetricSeries>,
}

impl MetricsSnapshot {
    /// Sums a counter across the selected series, or returns `None` if it is absent.
    pub fn counter_total(&self, name: &str) -> Option<f64> {
        let mut values = self
            .series
            .iter()
            .filter_map(|series| match series.value {
                MetricValue::Counter { value } if series.name == name => Some(value),
                _ => None,
            })
            .peekable();
        values.peek()?;
        Some(values.sum())
    }
}

/// One labeled metric series in a snapshot. Names use Prometheus spelling.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MetricSeries {
    /// Metric name, for example `switchyard_llm_call_prompt_tokens_total`.
    pub name: String,
    /// All labels on the series.
    pub labels: BTreeMap<String, String>,
    /// Scalar value or histogram statistics.
    pub value: MetricValue,
}

/// Cumulative metric value; gauges retain their current value.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MetricValue {
    /// Monotonically increasing count.
    Counter {
        /// Count recorded so far.
        value: f64,
    },
    /// Current gauge value, including signed up/down counters.
    Gauge {
        /// Current value.
        value: f64,
    },
    /// Distribution retaining cumulative bucket counts.
    Histogram {
        /// Number of observations.
        count: u64,
        /// Sum of observations, in the metric's units.
        sum: f64,
        /// Cumulative finite bucket counts.
        buckets: Vec<HistogramBucket>,
    },
}

/// One cumulative histogram bucket.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct HistogramBucket {
    /// Inclusive upper boundary.
    pub upper_bound: f64,
    /// Number of observations at or below the boundary.
    pub count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentelemetry::KeyValue;
    use opentelemetry::metrics::MeterProvider as _;

    #[test]
    fn queries_beyond_the_default_series_limit_and_retains_gauges_and_histograms() {
        let metrics = Metrics::with_cardinality_limit(3000).unwrap();
        let meter = metrics.meter_provider().meter("switchyard");
        let counter = meter.u64_counter("switchyard.test_turns").build();
        let gauge = meter.i64_up_down_counter("switchyard.test_active").build();
        let histogram = meter.f64_histogram("switchyard.test_duration_ms").build();
        for idx in 0..2100 {
            let labels = [
                KeyValue::new("session_id", "one"),
                KeyValue::new("agent_id", "worker"),
                KeyValue::new("agent_turn_idx", idx),
            ];
            counter.add(1, &labels);
            gauge.add(-1, &labels);
            histogram.record(12.5, &labels);
        }
        let filter = MetricFilter {
            session_id: Some("one"),
            agent_id: Some("worker"),
            agent_turn_idx: Some(2099),
            ..MetricFilter::default()
        };
        let snapshot = metrics.stats(filter);
        assert_eq!(
            snapshot.counter_total("switchyard_test_turns_total"),
            Some(1.0)
        );
        assert!(
            snapshot
                .series
                .iter()
                .any(|series| series.name == "switchyard_test_active"
                    && series.value == MetricValue::Gauge { value: -1.0 })
        );
        assert!(
            snapshot
                .series
                .iter()
                .any(|series| series.name == "switchyard_test_duration_ms"
                    && matches!(
                        series.value,
                        MetricValue::Histogram {
                            count: 1,
                            sum: 12.5,
                            ..
                        }
                    ))
        );
        let all = metrics.stats(MetricFilter::default());
        assert_eq!(
            all.counter_total("switchyard_test_turns_total"),
            Some(2100.0)
        );
        assert_eq!(metrics.stats(filter), snapshot);
        serde_json::to_value(snapshot).unwrap();
        assert!(Metrics::with_cardinality_limit(0).is_err());
    }
}
