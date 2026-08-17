// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Result files plus the background reporter and invalid-request canary tasks.

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::client::read_server_state;
use crate::stats::{latency_stats, now_utc_string, round3};
use crate::{RunContext, Stop, Workload};

/// Cap on individually recorded failures so a bad run cannot fill the disk.
const MAX_ERROR_RECORDS: u64 = 10_000;

const INTERVAL_FIELDS: [&str; 15] = [
    "timestamp_utc",
    "elapsed_seconds",
    "requests",
    "successes",
    "failures",
    "requests_per_second",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "latency_max_ms",
    "health",
    "server_total_requests",
    "server_total_errors",
    "rss_mib",
    "cpu_percent",
];

/// Write interval rows and bounded error details as the run proceeds.
pub struct ResultsWriter {
    interval_file: File,
    error_file: File,
    error_records: u64,
    dropped_error_records: u64,
}

impl ResultsWriter {
    /// Create a fresh results directory; fails if it already exists.
    pub fn new(results_dir: &Path) -> io::Result<Self> {
        if let Some(parent) = results_dir.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        fs::create_dir(results_dir)?;
        let mut interval_file = File::create(results_dir.join("intervals.csv"))?;
        writeln!(interval_file, "{}", INTERVAL_FIELDS.join(","))?;
        let error_file = File::create(results_dir.join("errors.jsonl"))?;
        Ok(Self {
            interval_file,
            error_file,
            error_records: 0,
            dropped_error_records: 0,
        })
    }

    /// Append one interval row; `cells` are already formatted in `INTERVAL_FIELDS` order.
    pub fn write_interval(&mut self, cells: &[String]) -> io::Result<()> {
        writeln!(self.interval_file, "{}", cells.join(","))?;
        self.interval_file.flush()
    }

    /// Append one failure record, dropping past the bound instead of growing without limit.
    pub fn write_error(&mut self, record: &Value) -> io::Result<()> {
        if self.error_records >= MAX_ERROR_RECORDS {
            self.dropped_error_records += 1;
            return Ok(());
        }
        writeln!(self.error_file, "{record}")?;
        self.error_file.flush()?;
        self.error_records += 1;
        Ok(())
    }

    pub fn error_counts(&self) -> (u64, u64) {
        (self.error_records, self.dropped_error_records)
    }
}

/// Format an optional number for a CSV cell; `None` becomes an empty cell.
fn cell(value: Option<f64>) -> String {
    value
        .map(|value| round3(value).to_string())
        .unwrap_or_default()
}

/// Return RSS MiB and CPU percent for *pid* using the local `ps` command.
pub async fn process_sample(pid: Option<u32>) -> (Option<f64>, Option<f64>) {
    let Some(pid) = pid else {
        return (None, None);
    };
    let output = match tokio::process::Command::new("ps")
        .args(["-o", "rss=,pcpu=", "-p", &pid.to_string()])
        .output()
        .await
    {
        Ok(output) => output,
        // ps missing or fork failed: record a process-check miss, don't crash the reporter.
        Err(_) => return (None, None),
    };
    if !output.status.success() {
        return (None, None);
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut fields = stdout.split_whitespace();
    let (Some(rss), Some(cpu), None) = (fields.next(), fields.next(), fields.next()) else {
        return (None, None);
    };
    match (rss.parse::<f64>(), cpu.parse::<f64>()) {
        (Ok(rss_kib), Ok(cpu_percent)) => (Some(rss_kib / 1024.0), Some(cpu_percent)),
        _ => (None, None),
    }
}

/// Write one liveness, metrics, resource, and latency row per interval.
pub async fn reporter(
    context: RunContext,
    started: Instant,
    interval: Duration,
    target_seconds: f64,
    server_pid: Option<u32>,
    workers_done: std::sync::Arc<Stop>,
) -> Result<(), String> {
    let mut previous_report = started;
    loop {
        tokio::select! {
            _ = tokio::time::sleep(interval) => {}
            _ = context.stop.wait() => {}
        }
        // Once stopping, wait for the workers to drain so the final row counts their last requests.
        if context.stop.is_set() {
            workers_done.wait().await;
        }

        let now = Instant::now();
        let mut interval_stats = context.stats.lock().take_interval();
        let server = read_server_state(&context.client, &context.base_url).await;
        let (rss_mib, cpu_percent) = process_sample(server_pid).await;

        let (total_successes, total_failures) = {
            let mut state = context.stats.lock();
            state.health_checks += 1;
            state.metrics_checks += 1;
            if !server.healthy {
                state.health_failures += 1;
            }
            if server.requests.is_none() || server.errors.is_none() {
                state.metrics_failures += 1;
            }
            if server_pid.is_some() {
                state.process_checks += 1;
                if rss_mib.is_none() || cpu_percent.is_none() {
                    state.process_failures += 1;
                }
            }
            if let Some(rss) = rss_mib {
                state.rss_samples.push(rss);
            }
            if let (Some(current), Some(previous)) =
                (server.requests, state.previous_server_requests)
                && current < previous
            {
                state.server_restarts += 1;
            }
            if let Some(current) = server.requests {
                state.previous_server_requests = Some(current);
            }
            (state.total_successes, state.total_failures)
        };

        let elapsed_interval = (now - previous_report).as_secs_f64().max(0.001);
        let requests = interval_stats.successes + interval_stats.failures;
        let latency = latency_stats(&mut interval_stats.latencies_ms);
        let requests_per_second = round3(requests as f64 / elapsed_interval);
        let timestamp = now_utc_string();
        let elapsed_seconds = round3((now - started).as_secs_f64());
        let health_label = if server.healthy { "ok" } else { "failed" };

        let cells = vec![
            timestamp.clone(),
            elapsed_seconds.to_string(),
            requests.to_string(),
            interval_stats.successes.to_string(),
            interval_stats.failures.to_string(),
            requests_per_second.to_string(),
            cell(latency.p50_ms),
            cell(latency.p95_ms),
            cell(latency.p99_ms),
            cell(latency.max_ms),
            health_label.to_string(),
            cell(server.requests),
            cell(server.errors),
            cell(rss_mib),
            cell(cpu_percent),
        ];
        // Cumulative, progress, and a glanceable status token so a remote tail of the log shows
        // at once that the run is alive, how far along it is, and whether it is healthy.
        let cumulative_requests = total_successes + total_failures;
        let cumulative_error_rate = if cumulative_requests > 0 {
            total_failures as f64 / cumulative_requests as f64
        } else {
            0.0
        };
        let progress = (elapsed_seconds / target_seconds).min(1.0);
        let status = if requests == 0 {
            "stalled"
        } else if !server.healthy || interval_stats.failures > 0 {
            "degraded"
        } else {
            "ok"
        };
        let p95_text = latency.p95_ms.map(|v| v.to_string()).unwrap_or_default();
        let rss_text = rss_mib.map(|v| round3(v).to_string()).unwrap_or_default();

        context
            .writer
            .lock()
            .write_interval(&cells)
            .map_err(|error| error.to_string())?;

        println!(
            "[{timestamp}] progress={elapsed_seconds:.0}s/{target_seconds:.0}s({:.0}%) \
             reqs={cumulative_requests} interval={requests} errors={total_failures}({:.4}%) \
             rps={requests_per_second} p95_ms={p95_text} health={health_label} rss_mib={rss_text} \
             status={}",
            progress * 100.0,
            cumulative_error_rate * 100.0,
            status.to_uppercase(),
        );

        previous_report = now;
        if context.stop.is_set() {
            return Ok(());
        }
    }
}

/// Confirm invalid input returns 400 and the server stays live, on a fixed interval.
pub async fn invalid_request_canary(
    context: RunContext,
    interval: f64,
    workload: std::sync::Arc<Workload>,
) -> Result<(), String> {
    if interval <= 0.0 {
        return Ok(());
    }
    let interval = Duration::from_secs_f64(interval);
    loop {
        tokio::select! {
            _ = tokio::time::sleep(interval) => {}
            _ = context.stop.wait() => {}
        }
        if context.stop.is_set() {
            return Ok(());
        }

        context.stats.lock().canaries += 1;
        let probe = async {
            let invalid = context
                .client
                .post(format!("{}/v1/chat/completions", context.base_url))
                .json(&json!({"model": workload.model, "messages": []}))
                .send()
                .await?;
            let health = context
                .client
                .get(format!("{}/health", context.base_url))
                .send()
                .await?;
            Ok::<(u16, u16), reqwest::Error>((invalid.status().as_u16(), health.status().as_u16()))
        }
        .await;
        let (passed, detail) = match probe {
            Ok((invalid_status, health_status)) => (
                invalid_status == 400 && health_status == 200,
                format!("invalid_status={invalid_status}, health_status={health_status}"),
            ),
            Err(error) => (false, error.to_string()),
        };
        if !passed {
            context.stats.lock().canary_failures += 1;
            context
                .writer
                .lock()
                .write_error(&json!({
                    "timestamp_utc": now_utc_string(),
                    "error": "invalid_request_canary",
                    "detail": detail,
                }))
                .map_err(|error| error.to_string())?;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_details_stop_growing_at_the_file_limit() -> io::Result<()> {
        let parent = tempfile::tempdir()?;
        let results_dir = parent.path().join("results");
        let mut writer = ResultsWriter::new(&results_dir)?;

        for _ in 0..=MAX_ERROR_RECORDS {
            writer.write_error(&serde_json::json!({"error": "upstream"}))?;
        }

        assert_eq!(writer.error_counts(), (MAX_ERROR_RECORDS, 1));
        assert_eq!(
            fs::read_to_string(results_dir.join("errors.jsonl"))?
                .lines()
                .count(),
            10_000
        );
        Ok(())
    }
}
