// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sustained, closed-loop load test against a live Switchyard server.
//!
//! Workers keep a fixed number of inference requests in flight while a reporter samples
//! liveness, metrics, and process resources, and a canary confirms invalid input is rejected.
//! The run writes result files and exits non-zero when a release gate fails.

pub mod client;
pub mod report;
pub mod stats;

use std::fs;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use parking_lot::Mutex;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};
use reqwest::Client;
use serde_json::json;
use tokio::sync::Notify;

use crate::client::{preflight, request_body, send_request, Endpoint};
use crate::report::{invalid_request_canary, reporter, ResultsWriter};
use crate::stats::{
    build_prompt_pool, build_summary, latency_report, now_utc_string, round3, utc_dir_stamp,
    RunStats,
};

/// Command-line arguments for the soak test.
#[derive(Parser)]
#[command(
    name = "switchyard-soak",
    about = "Run a sustained, closed-loop load test against a live Switchyard server",
    version
)]
pub struct Args {
    /// Base URL of the Switchyard server.
    #[arg(long, default_value = "http://127.0.0.1:4000")]
    base_url: String,

    /// Model id from GET /v1/models; defaults to its first model.
    #[arg(long)]
    model: Option<String>,

    /// Run time with an s, m, or h suffix.
    #[arg(long, value_parser = stats::parse_duration, default_value = "48h")]
    duration: f64,

    /// Number of closed-loop inference workers.
    #[arg(long, default_value_t = 16)]
    concurrency: usize,

    /// Public APIs to exercise; defaults to all three.
    #[arg(long, value_enum, num_args = 1..)]
    endpoints: Vec<Endpoint>,

    /// Fraction of inference requests that use streaming.
    #[arg(long, default_value_t = 0.5)]
    stream_ratio: f64,

    /// Maximum output tokens requested from the backend.
    #[arg(long, default_value_t = 32)]
    max_output_tokens: u32,

    /// Repeated-prefix payload size for each prompt.
    #[arg(long, default_value_t = 1024)]
    prompt_bytes: usize,

    /// Timeout in seconds for one inference request.
    #[arg(long, default_value_t = 120.0)]
    request_timeout: f64,

    /// Seconds between health, metrics, and result samples.
    #[arg(long, default_value_t = 60.0)]
    report_interval: f64,

    /// Seconds between invalid-request recovery checks; zero disables them.
    #[arg(long, default_value_t = 300.0)]
    invalid_canary_interval: f64,

    /// Largest allowed inference error fraction.
    #[arg(long, default_value_t = 0.0)]
    max_error_rate: f64,

    /// Local Switchyard PID to sample for RSS and CPU.
    #[arg(long)]
    server_pid: Option<u32>,

    /// Largest allowed first-to-last RSS increase in MiB.
    #[arg(long, requires = "server_pid")]
    max_rss_growth_mib: Option<f64>,

    /// Environment variable holding a bearer token for the Switchyard endpoint.
    #[arg(long)]
    api_key_env: Option<String>,

    /// New directory for the run result files.
    #[arg(long)]
    results_dir: Option<PathBuf>,
}

impl Args {
    /// Reject inputs clap's types cannot, matching the Python runner's checks.
    pub fn validate(&self) -> Result<(), String> {
        if self.concurrency == 0 {
            return Err("--concurrency must be greater than zero".to_string());
        }
        let mut seen = std::collections::HashSet::new();
        if !self
            .endpoints
            .iter()
            .all(|endpoint| seen.insert(endpoint.as_str()))
        {
            return Err("--endpoints must not repeat a value".to_string());
        }
        if !(0.0..=1.0).contains(&self.stream_ratio) {
            return Err("--stream-ratio must be between 0 and 1".to_string());
        }
        if self.max_output_tokens == 0 || self.prompt_bytes == 0 {
            return Err(
                "--max-output-tokens and --prompt-bytes must be greater than zero".to_string(),
            );
        }
        if self.request_timeout <= 0.0 || self.report_interval <= 0.0 {
            return Err(
                "--request-timeout and --report-interval must be greater than zero".to_string(),
            );
        }
        if self.invalid_canary_interval < 0.0 {
            return Err("--invalid-canary-interval must be zero or greater".to_string());
        }
        if !(0.0..=1.0).contains(&self.max_error_rate) {
            return Err("--max-error-rate must be between 0 and 1".to_string());
        }
        if self.server_pid == Some(0) {
            return Err("--server-pid must be greater than zero".to_string());
        }
        if self.max_rss_growth_mib.is_some_and(|growth| growth < 0.0) {
            return Err("--max-rss-growth-mib must be zero or greater".to_string());
        }
        Ok(())
    }

    fn endpoints(&self) -> Vec<Endpoint> {
        if self.endpoints.is_empty() {
            Endpoint::ALL.to_vec()
        } else {
            self.endpoints.clone()
        }
    }
}

/// A one-shot stop signal that many tasks can wait on and any task can raise.
pub struct Stop {
    flag: AtomicBool,
    notify: Notify,
}

impl Stop {
    pub fn new() -> Self {
        Self {
            flag: AtomicBool::new(false),
            notify: Notify::new(),
        }
    }

    pub fn set(&self) {
        self.flag.store(true, Ordering::SeqCst);
        self.notify.notify_waiters();
    }

    pub fn is_set(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }

    /// Resolve once the signal is raised, now or later.
    pub async fn wait(&self) {
        loop {
            if self.is_set() {
                return;
            }
            // Register before the second flag check so a set() between the two still wakes us.
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.is_set() {
                return;
            }
            notified.await;
        }
    }
}

impl Default for Stop {
    fn default() -> Self {
        Self::new()
    }
}

/// Send closed-loop traffic until the run stops.
#[allow(clippy::too_many_arguments)]
async fn worker(
    client: Client,
    base_url: String,
    worker_id: usize,
    model: String,
    endpoints: Vec<Endpoint>,
    prompt_pool: Vec<String>,
    max_output_tokens: u32,
    stream_ratio: f64,
    stop: Arc<Stop>,
    request_numbers: Arc<AtomicU64>,
    stats: Arc<Mutex<RunStats>>,
    writer: Arc<Mutex<ResultsWriter>>,
) -> Result<(), String> {
    let mut rng = StdRng::seed_from_u64(10_000 + worker_id as u64);
    while !stop.is_set() {
        let request_number = request_numbers.fetch_add(1, Ordering::Relaxed) as usize;
        let endpoint = endpoints[request_number % endpoints.len()];
        let stream = rng.gen::<f64>() < stream_ratio;
        let body = request_body(
            endpoint,
            &model,
            &prompt_pool[request_number % prompt_pool.len()],
            max_output_tokens,
            stream,
        );
        let started = Instant::now();
        let (error_kind, detail) = send_request(&client, &base_url, endpoint, &body).await;
        let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
        stats
            .lock()
            .record(endpoint.as_str(), latency_ms, error_kind.as_deref());
        if let Some(kind) = &error_kind {
            writer
                .lock()
                .write_error(&json!({
                    "timestamp_utc": now_utc_string(),
                    "worker": worker_id,
                    "endpoint": endpoint.as_str(),
                    "stream": stream,
                    "latency_ms": round3(latency_ms),
                    "error": kind,
                    "detail": detail,
                }))
                .map_err(|error| error.to_string())?;
        }
    }
    Ok(())
}

/// Run a background task and raise the stop signal unless it returns `Ok`, so the run ends
/// fail-closed. The drop guard fires on an `Err` return and on a panic unwinding through the
/// await, matching the Python runner, which stopped the run on any task exception; a task that
/// returns `Ok` (a worker/reporter after stop, or a disabled canary) leaves the guard disarmed.
async fn guard_stop(
    stop: Arc<Stop>,
    task: impl Future<Output = Result<(), String>>,
) -> Result<(), String> {
    struct StopOnDrop(Arc<Stop>, bool);
    impl Drop for StopOnDrop {
        fn drop(&mut self) {
            if self.1 {
                self.0.set();
            }
        }
    }
    let mut guard = StopOnDrop(stop, true);
    let result = task.await;
    guard.1 = result.is_err();
    result
}

/// Fold one joined task's outcome into the failure reasons; cancelled tasks are expected.
fn collect_failure(
    name: &str,
    result: Result<Result<(), String>, tokio::task::JoinError>,
    failures: &mut Vec<String>,
) {
    match result {
        Ok(Ok(())) => {}
        Ok(Err(reason)) => failures.push(format!("{name} failed: {reason}")),
        Err(join) if join.is_cancelled() => {}
        Err(join) => failures.push(format!("{name} failed: {join}")),
    }
}

/// Raise *stop* on SIGINT or SIGTERM so an operator can end the run cleanly.
fn spawn_signal_listener(stop: Arc<Stop>) {
    tokio::spawn(async move {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{signal, SignalKind};
            let mut interrupt = signal(SignalKind::interrupt()).ok();
            let mut terminate = signal(SignalKind::terminate()).ok();
            let wait_interrupt = async {
                match interrupt.as_mut() {
                    Some(stream) => {
                        stream.recv().await;
                    }
                    None => std::future::pending::<()>().await,
                }
            };
            let wait_terminate = async {
                match terminate.as_mut() {
                    Some(stream) => {
                        stream.recv().await;
                    }
                    None => std::future::pending::<()>().await,
                }
            };
            tokio::select! {
                _ = wait_interrupt => {}
                _ = wait_terminate => {}
            }
            stop.set();
        }
        #[cfg(not(unix))]
        {
            let _ = tokio::signal::ctrl_c().await;
            stop.set();
        }
    });
}

/// Record every non-secret input, plus the resolved model and the seconds duration.
fn write_config(results_dir: &Path, args: &Args, model: &str) -> Result<(), String> {
    let config = json!({
        "base_url": args.base_url,
        "model": model,
        "duration_seconds": args.duration,
        "concurrency": args.concurrency,
        "endpoints": args.endpoints().iter().map(|endpoint| endpoint.as_str()).collect::<Vec<_>>(),
        "stream_ratio": args.stream_ratio,
        "max_output_tokens": args.max_output_tokens,
        "prompt_bytes": args.prompt_bytes,
        "request_timeout": args.request_timeout,
        "report_interval": args.report_interval,
        "invalid_canary_interval": args.invalid_canary_interval,
        "max_error_rate": args.max_error_rate,
        "server_pid": args.server_pid,
        "max_rss_growth_mib": args.max_rss_growth_mib,
        "api_key_env": args.api_key_env,
    });
    let body = serde_json::to_string_pretty(&config).map_err(|error| error.to_string())?;
    fs::write(results_dir.join("config.json"), format!("{body}\n"))
        .map_err(|error| error.to_string())
}

/// Run the configured soak test and return a process exit code (0 pass, 1 fail).
pub async fn run(args: Args) -> Result<i32, String> {
    let token = match &args.api_key_env {
        Some(var) => {
            let token = std::env::var(var).ok().filter(|value| !value.is_empty());
            if token.is_none() {
                return Err(format!("${var} is not set"));
            }
            token
        }
        None => None,
    };

    // Per-operation timeouts, not a whole-request deadline: a healthy long stream that keeps
    // delivering bytes must not be aborted, so bound connect time and idle time between reads.
    let request_timeout = Duration::from_secs_f64(args.request_timeout);
    let mut builder = Client::builder()
        .no_proxy()
        .connect_timeout(request_timeout)
        .read_timeout(request_timeout)
        .pool_max_idle_per_host(args.concurrency + 4);
    if let Some(token) = &token {
        let mut headers = HeaderMap::new();
        let value = HeaderValue::from_str(&format!("Bearer {token}"))
            .map_err(|_| "the bearer token is not a valid HTTP header value".to_string())?;
        headers.insert(AUTHORIZATION, value);
        builder = builder.default_headers(headers);
    }
    let client = builder.build().map_err(|error| error.to_string())?;
    let base_url = args.base_url.trim_end_matches('/').to_string();

    let model = preflight(&client, &base_url, args.model.as_deref()).await?;
    let results_dir = args
        .results_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from("soak-results").join(utc_dir_stamp()));
    let writer = Arc::new(Mutex::new(
        ResultsWriter::new(&results_dir).map_err(|error| error.to_string())?,
    ));
    write_config(&results_dir, &args, &model)?;

    let endpoints = args.endpoints();
    println!(
        "Soak started: model={model} duration={}s concurrency={} endpoints={} results={}",
        args.duration,
        args.concurrency,
        endpoints
            .iter()
            .map(|endpoint| endpoint.as_str())
            .collect::<Vec<_>>()
            .join(","),
        results_dir.display(),
    );

    let started = Instant::now();
    let stats = Arc::new(Mutex::new(RunStats::new(2026)));
    let stop = Arc::new(Stop::new());
    let workers_done = Arc::new(Stop::new());
    let request_numbers = Arc::new(AtomicU64::new(0));

    spawn_signal_listener(stop.clone());

    let deadline = {
        let stop = stop.clone();
        let stats = stats.clone();
        let duration = Duration::from_secs_f64(args.duration);
        tokio::spawn(async move {
            tokio::time::sleep(duration).await;
            stats.lock().completed_duration = true;
            stop.set();
        })
    };

    let prompt_pool = build_prompt_pool(args.prompt_bytes);
    let mut worker_handles = Vec::new();
    for worker_id in 0..args.concurrency {
        let handle = tokio::spawn(guard_stop(
            stop.clone(),
            worker(
                client.clone(),
                base_url.clone(),
                worker_id,
                model.clone(),
                endpoints.clone(),
                prompt_pool.clone(),
                args.max_output_tokens,
                args.stream_ratio,
                stop.clone(),
                request_numbers.clone(),
                stats.clone(),
                writer.clone(),
            ),
        ));
        worker_handles.push((format!("worker-{worker_id}"), handle));
    }
    let reporter_handle = tokio::spawn(guard_stop(
        stop.clone(),
        reporter(
            client.clone(),
            base_url.clone(),
            started,
            Duration::from_secs_f64(args.report_interval),
            args.duration,
            args.server_pid,
            stop.clone(),
            workers_done.clone(),
            stats.clone(),
            writer.clone(),
        ),
    ));
    let canary_handle = tokio::spawn(guard_stop(
        stop.clone(),
        invalid_request_canary(
            client.clone(),
            base_url.clone(),
            args.invalid_canary_interval,
            model.clone(),
            stop.clone(),
            stats.clone(),
            writer.clone(),
        ),
    ));

    stop.wait().await;
    deadline.abort();

    // A crashed worker/reporter/canary must not discard the run: record it as a failure reason
    // so the summary is still written and the run fails closed.
    let mut task_failures = Vec::new();
    for (name, handle) in worker_handles {
        collect_failure(&name, handle.await, &mut task_failures);
    }
    workers_done.set();
    collect_failure("reporter", reporter_handle.await, &mut task_failures);
    collect_failure(
        "invalid-request-canary",
        canary_handle.await,
        &mut task_failures,
    );

    let elapsed = started.elapsed().as_secs_f64();
    let (error_records, dropped_error_records) = {
        let writer = writer.lock();
        (writer.error_records, writer.dropped_error_records)
    };
    let summary = build_summary(
        &stats.lock(),
        elapsed,
        args.max_error_rate,
        args.max_rss_growth_mib,
        error_records,
        dropped_error_records,
        &task_failures,
    );
    let latency_block = latency_report(stats.lock().latency_samples());
    let summary_path = results_dir.join("summary.json");
    let summary_body =
        serde_json::to_string_pretty(&summary.json).map_err(|error| error.to_string())?;
    fs::write(&summary_path, format!("{summary_body}\n")).map_err(|error| error.to_string())?;

    let label = if summary.passed { "PASS" } else { "FAIL" };
    println!(
        "Soak {label}: requests={} error_rate={:.4}% p95_ms={} summary={}",
        summary.requests,
        summary.error_rate * 100.0,
        summary
            .latency_p95_ms
            .map(|value| value.to_string())
            .unwrap_or_default(),
        summary_path.display(),
    );
    for reason in &summary.reasons {
        println!("- {reason}");
    }
    if !latency_block.is_empty() {
        print!("{latency_block}");
    }
    Ok(if summary.passed { 0 } else { 1 })
}

/// Parse arguments, run the test on a multi-thread runtime, and map the result to an exit code.
pub fn cli_main() -> ExitCode {
    let args = Args::parse();
    if let Err(message) = args.validate() {
        eprintln!("switchyard-soak: {message}");
        return ExitCode::from(2);
    }
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("soak test setup failed: {error}");
            return ExitCode::from(2);
        }
    };
    match runtime.block_on(run(args)) {
        Ok(0) => ExitCode::SUCCESS,
        Ok(_) => ExitCode::from(1),
        Err(error) => {
            eprintln!("soak test setup failed: {error}");
            ExitCode::from(2)
        }
    }
}
