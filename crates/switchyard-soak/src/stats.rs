// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded in-memory run state, latency percentiles, and the final pass/fail summary.

use std::collections::BTreeMap;
use std::time::SystemTime;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use serde::Serialize;

/// Cap on retained latency samples; a long run stays within this bound by reservoir sampling.
const RESERVOIR_SIZE: usize = 100_000;

/// Round to three decimals so result files stay readable.
pub fn round3(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

/// Parse a duration such as `30s`, `15m`, or `48h` into seconds.
pub fn parse_duration(value: &str) -> Result<f64, String> {
    let text = value.trim().to_lowercase();
    let bad = || "duration must use s, m, or h, for example 30s or 48h".to_string();
    let unit = text.chars().last().ok_or_else(bad)?;
    let multiplier = match unit {
        's' => 1.0,
        'm' => 60.0,
        'h' => 3600.0,
        _ => return Err(bad()),
    };
    let number = &text[..text.len() - unit.len_utf8()];
    let seconds = number.parse::<f64>().map_err(|_| bad())? * multiplier;
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err("duration must be greater than zero".to_string());
    }
    Ok(seconds)
}

#[derive(Default)]
pub struct LatencyStats {
    pub p50_ms: Option<f64>,
    pub p95_ms: Option<f64>,
    pub p99_ms: Option<f64>,
    pub max_ms: Option<f64>,
}

fn percentile_sorted(values: &[f64], quantile: f64) -> Option<f64> {
    let index = (((values.len().checked_sub(1)?) as f64) * quantile).round_ties_even() as usize;
    Some(round3(values[index]))
}

pub fn latency_stats(values: &mut [f64]) -> LatencyStats {
    values.sort_by(|a, b| a.total_cmp(b));
    LatencyStats {
        p50_ms: percentile_sorted(values, 0.50),
        p95_ms: percentile_sorted(values, 0.95),
        p99_ms: percentile_sorted(values, 0.99),
        max_ms: values.last().copied().map(round3),
    }
}

/// Return four stable prompts with reusable prefixes of *prompt_bytes* filler each.
pub fn build_prompt_pool(prompt_bytes: usize) -> Vec<String> {
    (0..4)
        .map(|index| {
            let prefix = format!("Switchyard soak prefix {index}. ");
            let instruction = "Reply with exactly OK. ";
            let unit = "load test context ";
            let mut filler = unit.repeat(prompt_bytes / unit.len() + 1);
            filler.truncate(prompt_bytes);
            format!("{prefix}{filler}{instruction}")
        })
        .collect()
}

/// ISO-8601 UTC timestamp at second precision, e.g. `2026-07-30T18:04:00Z`.
pub fn now_utc_string() -> String {
    humantime::format_rfc3339_seconds(SystemTime::now()).to_string()
}

/// Compact UTC stamp for a results directory, e.g. `20260730T180400Z`.
pub fn utc_dir_stamp() -> String {
    now_utc_string()
        .chars()
        .filter(|character| !matches!(character, '-' | ':'))
        .collect()
}

/// Request results collected since the previous report.
#[derive(Default)]
pub struct IntervalStats {
    pub successes: u64,
    pub failures: u64,
    pub latencies_ms: Vec<f64>,
}

/// Bounded in-memory state for one soak run.
pub struct RunStats {
    rng: StdRng,
    interval: IntervalStats,
    pub total_successes: u64,
    pub total_failures: u64,
    pub endpoint_successes: BTreeMap<String, u64>,
    pub endpoint_failures: BTreeMap<String, u64>,
    pub error_kinds: BTreeMap<String, u64>,
    latency_reservoir: Vec<f64>,
    latency_count: u64,
    pub health_checks: u64,
    pub health_failures: u64,
    pub metrics_checks: u64,
    pub metrics_failures: u64,
    pub process_checks: u64,
    pub process_failures: u64,
    pub canaries: u64,
    pub canary_failures: u64,
    pub server_restarts: u64,
    pub previous_server_requests: Option<f64>,
    pub rss_samples: Vec<f64>,
    pub completed_duration: bool,
}

impl RunStats {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            interval: IntervalStats::default(),
            total_successes: 0,
            total_failures: 0,
            endpoint_successes: BTreeMap::new(),
            endpoint_failures: BTreeMap::new(),
            error_kinds: BTreeMap::new(),
            latency_reservoir: Vec::new(),
            latency_count: 0,
            health_checks: 0,
            health_failures: 0,
            metrics_checks: 0,
            metrics_failures: 0,
            process_checks: 0,
            process_failures: 0,
            canaries: 0,
            canary_failures: 0,
            server_restarts: 0,
            previous_server_requests: None,
            rss_samples: Vec::new(),
            completed_duration: false,
        }
    }

    /// Record one completed inference request; `error_kind` is `None` on success.
    pub fn record(&mut self, endpoint: &str, latency_ms: f64, error_kind: Option<&str>) {
        match error_kind {
            None => {
                self.interval.successes += 1;
                self.total_successes += 1;
                *self
                    .endpoint_successes
                    .entry(endpoint.to_string())
                    .or_default() += 1;
            }
            Some(kind) => {
                self.interval.failures += 1;
                self.total_failures += 1;
                *self
                    .endpoint_failures
                    .entry(endpoint.to_string())
                    .or_default() += 1;
                *self.error_kinds.entry(kind.to_string()).or_default() += 1;
            }
        }
        self.interval.latencies_ms.push(latency_ms);
        self.latency_count += 1;
        if self.latency_reservoir.len() < RESERVOIR_SIZE {
            self.latency_reservoir.push(latency_ms);
        } else {
            let index = self.rng.random_range(0..self.latency_count) as usize;
            if index < RESERVOIR_SIZE {
                self.latency_reservoir[index] = latency_ms;
            }
        }
    }

    /// Return and reset the current interval.
    pub fn take_interval(&mut self) -> IntervalStats {
        std::mem::take(&mut self.interval)
    }
}

/// Final result written to `summary.json`.
#[derive(Serialize)]
pub struct Summary {
    pub passed: bool,
    pub failure_reasons: Vec<String>,
    pub completed_duration: bool,
    pub elapsed_seconds: f64,
    pub requests: u64,
    pub successes: u64,
    pub failures: u64,
    pub error_rate: f64,
    pub requests_per_second: f64,
    pub endpoint_successes: BTreeMap<String, u64>,
    pub endpoint_failures: BTreeMap<String, u64>,
    pub error_kinds: BTreeMap<String, u64>,
    pub latency_p50_ms: Option<f64>,
    pub latency_p95_ms: Option<f64>,
    pub latency_p99_ms: Option<f64>,
    pub health_checks: u64,
    pub health_failures: u64,
    pub metrics_checks: u64,
    pub metrics_failures: u64,
    pub process_checks: u64,
    pub process_failures: u64,
    pub invalid_request_canaries: u64,
    pub invalid_request_canary_failures: u64,
    pub detected_server_restarts: u64,
    pub rss_first_mib: Option<f64>,
    pub rss_last_mib: Option<f64>,
    pub rss_max_mib: Option<f64>,
    pub rss_growth_mib: Option<f64>,
    pub error_records: u64,
    pub dropped_error_records: u64,
}

/// Build the final result and its release-gate reasons.
pub fn build_summary(
    stats: &RunStats,
    elapsed_seconds: f64,
    max_error_rate: f64,
    max_rss_growth_mib: Option<f64>,
    error_records: u64,
    dropped_error_records: u64,
    task_failures: &[String],
) -> Summary {
    let total = stats.total_successes + stats.total_failures;
    let error_rate = if total > 0 {
        stats.total_failures as f64 / total as f64
    } else {
        1.0
    };
    let rss_first = stats.rss_samples.first().copied();
    let rss_last = stats.rss_samples.last().copied();
    let rss_growth = match (rss_first, rss_last) {
        (Some(first), Some(last)) => Some(last - first),
        _ => None,
    };

    let mut reasons: Vec<String> = task_failures.to_vec();
    if !stats.completed_duration {
        reasons.push("the run stopped before the requested duration".to_string());
    }
    if total == 0 {
        reasons.push("no inference requests completed".to_string());
    }
    if error_rate > max_error_rate {
        reasons.push(format!(
            "request error rate {:.4}% exceeded the {:.4}% limit",
            error_rate * 100.0,
            max_error_rate * 100.0
        ));
    }
    if stats.health_failures > 0 {
        reasons.push(format!("{} liveness checks failed", stats.health_failures));
    }
    if stats.metrics_failures > 0 {
        reasons.push(format!(
            "{} server metrics checks failed",
            stats.metrics_failures
        ));
    }
    if stats.process_failures > 0 {
        reasons.push(format!(
            "{} server process checks failed",
            stats.process_failures
        ));
    }
    if stats.canary_failures > 0 {
        reasons.push(format!(
            "{} invalid-request recovery checks failed",
            stats.canary_failures
        ));
    }
    if stats.server_restarts > 0 {
        reasons.push(format!(
            "server counters reset {} time(s)",
            stats.server_restarts
        ));
    }
    if let (Some(limit), Some(growth)) = (max_rss_growth_mib, rss_growth)
        && growth > limit
    {
        reasons.push(format!(
            "server RSS grew {growth:.1} MiB, above the {limit:.1} MiB limit"
        ));
    }

    let rss_max = stats
        .rss_samples
        .iter()
        .copied()
        .fold(None, |acc: Option<f64>, value| {
            Some(acc.map_or(value, |m| m.max(value)))
        });
    let requests_per_second = if elapsed_seconds > 0.0 {
        total as f64 / elapsed_seconds
    } else {
        0.0
    };

    let mut latency_samples = stats.latency_reservoir.clone();
    let latency = latency_stats(&mut latency_samples);

    Summary {
        passed: reasons.is_empty(),
        failure_reasons: reasons,
        completed_duration: stats.completed_duration,
        elapsed_seconds: round3(elapsed_seconds),
        requests: total,
        successes: stats.total_successes,
        failures: stats.total_failures,
        error_rate,
        requests_per_second,
        endpoint_successes: stats.endpoint_successes.clone(),
        endpoint_failures: stats.endpoint_failures.clone(),
        error_kinds: stats.error_kinds.clone(),
        latency_p50_ms: latency.p50_ms,
        latency_p95_ms: latency.p95_ms,
        latency_p99_ms: latency.p99_ms,
        health_checks: stats.health_checks,
        health_failures: stats.health_failures,
        metrics_checks: stats.metrics_checks,
        metrics_failures: stats.metrics_failures,
        process_checks: stats.process_checks,
        process_failures: stats.process_failures,
        invalid_request_canaries: stats.canaries,
        invalid_request_canary_failures: stats.canary_failures,
        detected_server_restarts: stats.server_restarts,
        rss_first_mib: rss_first,
        rss_last_mib: rss_last,
        rss_max_mib: rss_max,
        rss_growth_mib: rss_growth,
        error_records,
        dropped_error_records,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_duration_accepts_units_and_rejects_invalid_values() {
        assert_eq!(parse_duration("30s"), Ok(30.0));
        assert_eq!(parse_duration(".5s"), Ok(0.5));
        assert_eq!(parse_duration("2.5m"), Ok(150.0));
        assert_eq!(parse_duration("48h"), Ok(172_800.0));
        for value in ["0s", "10", "5x", "-3s", "1d", "NaNs", "infs"] {
            assert!(parse_duration(value).is_err(), "{value} should be rejected");
        }
    }

    #[test]
    fn latency_stats_sorts_once_and_handles_empty_input() {
        let mut values = [60.0, 10.0, 50.0, 20.0, 40.0, 30.0];
        let latency = latency_stats(&mut values);
        assert_eq!(latency.p50_ms, Some(30.0));
        assert_eq!(latency.p95_ms, Some(60.0));
        assert_eq!(latency.p99_ms, Some(60.0));
        assert_eq!(latency.max_ms, Some(60.0));

        let empty = latency_stats(&mut []);
        assert_eq!(empty.p50_ms, None);
        assert_eq!(empty.max_ms, None);
    }

    #[test]
    fn record_tracks_interval_and_cumulative_results() {
        let mut stats = RunStats::new(1);
        stats.record("chat", 10.0, None);
        stats.record("messages", 20.0, Some("timeout"));

        let interval = stats.take_interval();
        assert_eq!(interval.successes, 1);
        assert_eq!(interval.failures, 1);
        assert_eq!(stats.total_successes, 1);
        assert_eq!(stats.total_failures, 1);
        assert_eq!(stats.error_kinds.get("timeout"), Some(&1));
    }

    #[test]
    fn latency_samples_stay_bounded_after_the_reservoir_fills() {
        let mut stats = RunStats::new(1);
        for sample in 0..=RESERVOIR_SIZE {
            stats.record("chat", sample as f64, None);
        }

        assert_eq!(stats.latency_reservoir.len(), RESERVOIR_SIZE);
        assert_eq!(stats.latency_count, RESERVOIR_SIZE as u64 + 1);
    }

    #[test]
    fn summary_reports_all_failed_gates() {
        let mut stats = RunStats::new(1);
        stats.total_successes = 998;
        stats.total_failures = 2;
        stats.health_failures = 1;
        stats.metrics_failures = 1;
        stats.process_failures = 1;
        stats.canary_failures = 1;
        stats.server_restarts = 1;
        stats.rss_samples = vec![100.0, 700.0];

        let summary = build_summary(
            &stats,
            10.0,
            0.001,
            Some(512.0),
            0,
            0,
            &["reporter failed: boom".to_string()],
        );

        assert!(!summary.passed);
        for reason in [
            "run stopped",
            "error rate",
            "liveness",
            "metrics",
            "process",
            "invalid-request",
            "counters reset",
            "RSS grew",
            "reporter failed",
        ] {
            assert!(
                summary
                    .failure_reasons
                    .iter()
                    .any(|message| message.contains(reason)),
                "missing {reason:?}: {:?}",
                summary.failure_reasons,
            );
        }
        assert_eq!(summary.rss_growth_mib, Some(600.0));
    }
}
