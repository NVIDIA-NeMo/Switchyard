// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded in-memory run state, latency percentiles, and the final pass/fail summary.

use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use serde_json::{Value, json};

/// Cap on retained latency samples; a long run stays within this bound by reservoir sampling.
const RESERVOIR_SIZE: usize = 100_000;

/// Cumulative counter names Switchyard exposes on `/metrics`.
pub const SERVER_REQUESTS_METRIC: &str = "switchyard_total_requests";
pub const SERVER_ERRORS_METRIC: &str = "switchyard_total_errors";

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
    // Accept only `\d+(\.\d+)?` before the unit: at least one digit, at most one dot with
    // digits on both sides. This rejects "10", "5x", "-3s", "1.", and ".5".
    let number = &text[..text.len() - unit.len_utf8()];
    let parts: Vec<&str> = number.split('.').collect();
    let well_formed = matches!(parts.len(), 1 | 2)
        && parts
            .iter()
            .all(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()));
    if !well_formed {
        return Err(bad());
    }
    let seconds = number.parse::<f64>().map_err(|_| bad())? * multiplier;
    if seconds <= 0.0 {
        return Err("duration must be greater than zero".to_string());
    }
    Ok(seconds)
}

/// Return the nearest-rank percentile for *values*, or `None` when empty.
pub fn percentile(values: &[f64], quantile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(|a, b| a.total_cmp(b));
    // round_ties_even matches Python's round(), which breaks exact halves to even.
    let index = (((ordered.len() - 1) as f64) * quantile).round_ties_even() as usize;
    Some(ordered[index])
}

/// One equal-width latency bucket for the end-of-run histogram.
struct Bucket {
    low: f64,
    high: f64,
    count: usize,
}

/// Split *values* into *bucket_count* equal-width buckets between the min and max sample.
/// The maximum sample falls in the last bucket rather than spilling past it.
fn histogram(values: &[f64], bucket_count: usize) -> Vec<Bucket> {
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let width = ((max - min) / bucket_count as f64).max(f64::MIN_POSITIVE);
    let mut counts = vec![0usize; bucket_count];
    for &value in values {
        let index = (((value - min) / width) as usize).min(bucket_count - 1);
        counts[index] += 1;
    }
    counts
        .into_iter()
        .enumerate()
        .map(|(index, count)| Bucket {
            low: min + index as f64 * width,
            high: min + (index + 1) as f64 * width,
            count,
        })
        .collect()
}

/// Render an oha-style latency percentile table and ASCII histogram, or empty when no samples.
/// Built from the bounded reservoir, so it reflects the whole run within the reservoir cap.
pub fn latency_report(reservoir: &[f64]) -> String {
    if reservoir.is_empty() {
        return String::new();
    }
    let mut out = String::from("Latency distribution (ms):\n");
    for (label, quantile) in [
        ("p50", 0.50),
        ("p75", 0.75),
        ("p90", 0.90),
        ("p95", 0.95),
        ("p99", 0.99),
        ("p99.9", 0.999),
    ] {
        if let Some(value) = percentile(reservoir, quantile) {
            out.push_str(&format!("  {label:<6}{:>12.3}\n", round3(value)));
        }
    }
    out.push_str("Latency histogram (ms):\n");
    let buckets = histogram(reservoir, 10);
    let widest = buckets
        .iter()
        .map(|bucket| bucket.count)
        .max()
        .unwrap_or(1)
        .max(1);
    for bucket in &buckets {
        let bar = "\u{25a0}".repeat(bucket.count * 40 / widest);
        out.push_str(&format!(
            "  {:>9.1} - {:>9.1} [{:>7}] {bar}\n",
            round3(bucket.low),
            round3(bucket.high),
            bucket.count
        ));
    }
    out
}

fn rounded_percentile(values: &[f64], quantile: f64) -> Option<f64> {
    percentile(values, quantile).map(round3)
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
    format_utc(SystemTime::now())
}

/// Compact UTC stamp for a results directory, e.g. `20260730T180400Z`.
pub fn utc_dir_stamp() -> String {
    let secs = unix_seconds(SystemTime::now());
    let (year, month, day, hour, minute, second) = civil_from_unix(secs);
    format!("{year:04}{month:02}{day:02}T{hour:02}{minute:02}{second:02}Z")
}

fn format_utc(time: SystemTime) -> String {
    let (year, month, day, hour, minute, second) = civil_from_unix(unix_seconds(time));
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

fn unix_seconds(time: SystemTime) -> i64 {
    // Before the epoch never happens for a live run; clamp to 0 rather than panic.
    time.duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Split seconds-since-epoch into UTC (year, month, day, hour, minute, second).
fn civil_from_unix(secs: i64) -> (i64, u32, u32, u32, u32, u32) {
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (hour, minute, second) = (rem / 3600, rem % 3600 / 60, rem % 60);
    // Howard Hinnant's days-to-civil algorithm.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (
        year + i64::from(month <= 2),
        month,
        day,
        hour as u32,
        minute as u32,
        second as u32,
    )
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

    /// The bounded reservoir of latency samples, for the end-of-run distribution report.
    pub fn latency_samples(&self) -> &[f64] {
        &self.latency_reservoir
    }
}

/// Final result of a run: the exit-relevant fields plus the JSON written to `summary.json`.
pub struct Summary {
    pub passed: bool,
    pub reasons: Vec<String>,
    pub requests: u64,
    pub error_rate: f64,
    pub latency_p95_ms: Option<f64>,
    pub json: Value,
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

    let json = json!({
        "passed": reasons.is_empty(),
        "failure_reasons": reasons,
        "completed_duration": stats.completed_duration,
        "elapsed_seconds": round3(elapsed_seconds),
        "requests": total,
        "successes": stats.total_successes,
        "failures": stats.total_failures,
        "error_rate": error_rate,
        "requests_per_second": requests_per_second,
        "endpoint_successes": stats.endpoint_successes,
        "endpoint_failures": stats.endpoint_failures,
        "error_kinds": stats.error_kinds,
        "latency_p50_ms": rounded_percentile(&stats.latency_reservoir, 0.50),
        "latency_p95_ms": rounded_percentile(&stats.latency_reservoir, 0.95),
        "latency_p99_ms": rounded_percentile(&stats.latency_reservoir, 0.99),
        "health_checks": stats.health_checks,
        "health_failures": stats.health_failures,
        "metrics_checks": stats.metrics_checks,
        "metrics_failures": stats.metrics_failures,
        "process_checks": stats.process_checks,
        "process_failures": stats.process_failures,
        "invalid_request_canaries": stats.canaries,
        "invalid_request_canary_failures": stats.canary_failures,
        "detected_server_restarts": stats.server_restarts,
        "rss_first_mib": rss_first,
        "rss_last_mib": rss_last,
        "rss_max_mib": rss_max,
        "rss_growth_mib": rss_growth,
        "error_records": error_records,
        "dropped_error_records": dropped_error_records,
    });

    Summary {
        passed: reasons.is_empty(),
        reasons,
        requests: total,
        error_rate,
        latency_p95_ms: rounded_percentile(&stats.latency_reservoir, 0.95),
        json,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_duration_reads_suffixes() {
        assert_eq!(parse_duration("30s"), Ok(30.0));
        assert_eq!(parse_duration("2.5m"), Ok(150.0));
        assert_eq!(parse_duration("48h"), Ok(172_800.0));
    }

    #[test]
    fn parse_duration_rejects_bad_values() {
        for value in ["0s", "10", "5x", "-3s", "1d", "1.", ".5"] {
            assert!(parse_duration(value).is_err(), "{value} should be rejected");
        }
    }

    #[test]
    fn percentile_uses_nearest_rank() {
        let values = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(percentile(&values, 0.0), Some(1.0));
        assert_eq!(percentile(&values, 0.95), Some(4.0));
        assert_eq!(percentile(&[], 0.5), None);
        // Exact half -> even rank, matching Python round(): (6-1)*0.5 = 2.5 rounds to 2, not 3.
        assert_eq!(
            percentile(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], 0.5),
            Some(30.0)
        );
    }

    #[test]
    fn format_utc_matches_known_instants() {
        let epoch = SystemTime::UNIX_EPOCH;
        assert_eq!(format_utc(epoch), "1970-01-01T00:00:00Z");
        // 2023-11-14T22:13:20Z
        let later = epoch + std::time::Duration::from_secs(1_700_000_000);
        assert_eq!(format_utc(later), "2023-11-14T22:13:20Z");
    }

    #[test]
    fn histogram_buckets_cover_all_samples() {
        let values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let buckets = histogram(&values, 10);
        assert_eq!(buckets.len(), 10);
        // Every sample is counted exactly once, and the max lands in the last bucket.
        assert_eq!(buckets.iter().map(|b| b.count).sum::<usize>(), values.len());
        assert_eq!(buckets[9].count, 2); // 9.0 and 10.0 (the max) share the last bucket
        assert_eq!(buckets[0].low, 0.0);

        // A single repeated value does not panic and lands entirely in one bucket.
        let flat = histogram(&[5.0, 5.0, 5.0], 10);
        assert_eq!(flat.iter().map(|b| b.count).sum::<usize>(), 3);
    }

    #[test]
    fn latency_report_is_empty_without_samples() {
        assert_eq!(latency_report(&[]), "");
        assert!(latency_report(&[1.0, 2.0, 3.0]).contains("Latency histogram (ms):"));
    }

    #[test]
    fn build_prompt_pool_sizes_filler() {
        let pool = build_prompt_pool(32);
        assert_eq!(pool.len(), 4);
        assert!(pool[0].starts_with("Switchyard soak prefix 0. "));
        assert!(pool[0].ends_with("Reply with exactly OK. "));
    }

    #[test]
    fn summary_fails_on_restart_and_error_budget() {
        let mut stats = RunStats::new(1);
        stats.completed_duration = true;
        stats.total_successes = 998;
        stats.total_failures = 2;
        stats.server_restarts = 1;

        let summary = build_summary(&stats, 10.0, 0.001, None, 0, 0, &[]);

        assert!(!summary.passed);
        assert!(summary.reasons.iter().any(|r| r.contains("error rate")));
        assert!(summary.reasons.iter().any(|r| r.contains("counters reset")));
    }

    #[test]
    fn summary_records_task_failures() {
        let mut stats = RunStats::new(1);
        stats.completed_duration = true;
        stats.total_successes = 100;

        let summary = build_summary(
            &stats,
            10.0,
            0.0,
            None,
            0,
            0,
            &["reporter failed: boom".to_string()],
        );

        assert!(!summary.passed);
        assert!(summary.reasons.iter().any(|r| r == "reporter failed: boom"));
    }

    #[test]
    fn summary_flags_rss_growth() {
        let mut stats = RunStats::new(1);
        stats.completed_duration = true;
        stats.total_successes = 100;
        stats.rss_samples = vec![100.0, 700.0];

        let summary = build_summary(&stats, 10.0, 0.0, Some(512.0), 0, 0, &[]);

        assert!(!summary.passed);
        assert_eq!(summary.json["rss_growth_mib"], json!(600.0));
        assert!(summary.reasons.iter().any(|r| r.contains("RSS grew")));
    }
}
