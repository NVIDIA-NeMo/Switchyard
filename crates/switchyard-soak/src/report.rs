// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Result files plus the background reporter and invalid-request canary tasks.

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use reqwest::Client;
use serde_json::{json, Value};

use crate::client::read_server_state;
use crate::stats::{
    now_utc_string, percentile, round3, RunStats, SERVER_ERRORS_METRIC, SERVER_REQUESTS_METRIC,
};
use crate::Stop;

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
    pub results_dir: PathBuf,
    interval_file: File,
    error_file: File,
    pub error_records: u64,
    pub dropped_error_records: u64,
}

impl ResultsWriter {
    /// Create a fresh results directory; fails if it already exists.
    pub fn new(results_dir: &Path) -> io::Result<Self> {
        if let Some(parent) = results_dir.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::create_dir(results_dir)?;
        let mut interval_file = File::create(results_dir.join("intervals.csv"))?;
        writeln!(interval_file, "{}", INTERVAL_FIELDS.join(","))?;
        let error_file = File::create(results_dir.join("errors.jsonl"))?;
        Ok(Self {
            results_dir: results_dir.to_path_buf(),
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
        // serde_json's map is ordered, so keys serialize sorted, matching the run config file.
        writeln!(self.error_file, "{record}")?;
        self.error_file.flush()?;
        self.error_records += 1;
        Ok(())
    }

    /// Overwrite status.json with the current run snapshot, written atomically (write-then-rename)
    /// so a remote monitor can read a complete file at any moment without racing the writer.
    pub fn write_status(&self, snapshot: &Value) -> io::Result<()> {
        let body = serde_json::to_string_pretty(snapshot).map_err(io::Error::other)?;
        let tmp = self.results_dir.join("status.json.tmp");
        fs::write(&tmp, format!("{body}\n"))?;
        fs::rename(tmp, self.results_dir.join("status.json"))
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
    let fields: Vec<String> = String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .map(str::to_string)
        .collect();
    if !output.status.success() || fields.len() != 2 {
        return (None, None);
    }
    match (fields[0].parse::<f64>(), fields[1].parse::<f64>()) {
        (Ok(rss_kib), Ok(cpu_percent)) => (Some(rss_kib / 1024.0), Some(cpu_percent)),
        _ => (None, None),
    }
}

/// Write one liveness, metrics, resource, and latency row per interval.
#[allow(clippy::too_many_arguments)]
pub async fn reporter(
    client: Client,
    base_url: String,
    started: Instant,
    interval: Duration,
    target_seconds: f64,
    server_pid: Option<u32>,
    stop: Arc<Stop>,
    workers_done: Arc<Stop>,
    stats: Arc<Mutex<RunStats>>,
    writer: Arc<Mutex<ResultsWriter>>,
) -> Result<(), String> {
    let mut previous_report = started;
    loop {
        tokio::select! {
            _ = tokio::time::sleep(interval) => {}
            _ = stop.wait() => {}
        }
        // Once stopping, wait for the workers to drain so the final row counts their last requests.
        if stop.is_set() {
            workers_done.wait().await;
        }

        let now = Instant::now();
        let interval_stats = stats.lock().take_interval();
        let (healthy, metrics) = read_server_state(&client, &base_url).await;
        let (rss_mib, cpu_percent) = process_sample(server_pid).await;
        let server_requests = metrics.get(SERVER_REQUESTS_METRIC).copied();
        let server_errors = metrics.get(SERVER_ERRORS_METRIC).copied();

        let (total_successes, total_failures, server_restarts, completed_duration) = {
            let mut state = stats.lock();
            state.health_checks += 1;
            state.metrics_checks += 1;
            if !healthy {
                state.health_failures += 1;
            }
            if server_requests.is_none() || server_errors.is_none() {
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
                (server_requests, state.previous_server_requests)
            {
                if current < previous {
                    state.server_restarts += 1;
                }
            }
            if let Some(current) = server_requests {
                state.previous_server_requests = Some(current);
            }
            (
                state.total_successes,
                state.total_failures,
                state.server_restarts,
                state.completed_duration,
            )
        };

        let elapsed_interval = (now - previous_report).as_secs_f64().max(0.001);
        let requests = interval_stats.successes + interval_stats.failures;
        let latency_max = interval_stats
            .latencies_ms
            .iter()
            .copied()
            .fold(None, |acc: Option<f64>, value| {
                Some(acc.map_or(value, |m: f64| m.max(value)))
            });
        let requests_per_second = round3(requests as f64 / elapsed_interval);
        let timestamp = now_utc_string();
        let elapsed_seconds = round3((now - started).as_secs_f64());
        let health_label = if healthy { "ok" } else { "failed" };
        let latency_p95 = percentile(&interval_stats.latencies_ms, 0.95).map(round3);

        let cells = vec![
            timestamp.clone(),
            elapsed_seconds.to_string(),
            requests.to_string(),
            interval_stats.successes.to_string(),
            interval_stats.failures.to_string(),
            requests_per_second.to_string(),
            cell(percentile(&interval_stats.latencies_ms, 0.50)),
            cell(latency_p95),
            cell(percentile(&interval_stats.latencies_ms, 0.99)),
            cell(latency_max),
            health_label.to_string(),
            cell(server_requests),
            cell(server_errors),
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
        let progress = if target_seconds > 0.0 {
            (elapsed_seconds / target_seconds).min(1.0)
        } else {
            0.0
        };
        let status = if requests == 0 {
            "stalled"
        } else if !healthy || interval_stats.failures > 0 {
            "degraded"
        } else {
            "ok"
        };
        let p95_text = latency_p95.map(|v| v.to_string()).unwrap_or_default();
        let rss_text = rss_mib.map(|v| round3(v).to_string()).unwrap_or_default();

        let snapshot = json!({
            "timestamp_utc": timestamp,
            "elapsed_seconds": elapsed_seconds,
            "target_seconds": round3(target_seconds),
            "progress": round3(progress),
            "requests": cumulative_requests,
            "successes": total_successes,
            "failures": total_failures,
            "error_rate": cumulative_error_rate,
            "interval_requests": requests,
            "requests_per_second": requests_per_second,
            "latency_p95_ms": latency_p95,
            "health": health_label,
            "rss_mib": rss_mib.map(round3),
            "detected_server_restarts": server_restarts,
            "status": status,
            "completed_duration": completed_duration,
        });
        {
            let mut w = writer.lock();
            w.write_interval(&cells)
                .map_err(|error| error.to_string())?;
            w.write_status(&snapshot)
                .map_err(|error| error.to_string())?;
        }

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
        if stop.is_set() {
            return Ok(());
        }
    }
}

/// Confirm invalid input returns 400 and the server stays live, on a fixed interval.
pub async fn invalid_request_canary(
    client: Client,
    base_url: String,
    interval: f64,
    model: String,
    stop: Arc<Stop>,
    stats: Arc<Mutex<RunStats>>,
    writer: Arc<Mutex<ResultsWriter>>,
) -> Result<(), String> {
    if interval <= 0.0 {
        return Ok(());
    }
    let interval = Duration::from_secs_f64(interval);
    loop {
        tokio::select! {
            _ = tokio::time::sleep(interval) => {}
            _ = stop.wait() => {}
        }
        if stop.is_set() {
            return Ok(());
        }

        stats.lock().canaries += 1;
        let probe = async {
            let invalid = client
                .post(format!("{base_url}/v1/chat/completions"))
                .json(&json!({"model": model, "messages": []}))
                .send()
                .await?;
            let health = client.get(format!("{base_url}/health")).send().await?;
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
            stats.lock().canary_failures += 1;
            writer
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
