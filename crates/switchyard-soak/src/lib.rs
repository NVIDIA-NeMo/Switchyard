// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![doc = include_str!("../README.md")]

mod client;
mod report;
mod stats;

use std::fs;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use clap::Parser;
use parking_lot::Mutex;
use reqwest::Client;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use serde_json::json;
use tokio::sync::Notify;

use crate::client::{Endpoint, preflight, request_body, send_request};
use crate::report::{ResultsWriter, invalid_request_canary, reporter};
use crate::stats::{
    RunStats, build_prompt_pool, build_summary, now_utc_string, round3, utc_dir_stamp,
};

/// Command-line arguments for the soak test.
#[derive(Parser)]
#[command(
    name = "switchyard-soak",
    about = "Run a sustained, closed-loop load test against a live Switchyard server",
    after_long_help = "Examples:\n  switchyard-soak --model switchyard/general --duration 5m --concurrency 4\n  switchyard-soak --model switchyard/general --duration 48h --server-pid 1234 --max-rss-growth-mib 512\n\nThis command does not require VidaiMock, oha, or AIPerf. The optional scripts/soak_rehearsal.py command uses an embedded VidaiMock helper and warns before it starts when oha or AIPerf is missing.",
    version
)]
pub struct Args {
    /// HTTP base URL of the Switchyard server under test.
    #[arg(long, default_value = "http://127.0.0.1:4000")]
    base_url: String,

    /// Exact route id advertised by GET /v1/models.
    #[arg(long)]
    model: String,

    /// Time to keep sending load; use an s, m, or h suffix.
    #[arg(long, value_parser = stats::parse_duration, default_value = "48h")]
    duration: f64,

    /// Requests kept in flight; each worker sends its next request after the last one ends.
    #[arg(long, default_value_t = 16)]
    concurrency: usize,

    /// Output-token limit sent with every inference request.
    #[arg(long, default_value_t = 32)]
    max_output_tokens: u32,

    /// Bytes of repeated prefix added to each prompt to exercise request memory and caching.
    #[arg(long, default_value_t = 1024)]
    prompt_bytes: usize,

    /// Seconds allowed to connect or wait between response bytes; an active stream may run longer.
    #[arg(long, default_value_t = 120.0)]
    request_timeout: f64,

    /// Seconds between health, metrics, process, and result samples.
    #[arg(long, default_value_t = 60.0)]
    report_interval: f64,

    /// Seconds between malformed-request checks; 0 turns this check off.
    #[arg(long, default_value_t = 300.0)]
    invalid_canary_interval: f64,

    /// Largest passing request-error fraction, from 0 (none) through 1 (all).
    #[arg(long, default_value_t = 0.0)]
    max_error_rate: f64,

    /// PID of a local switchyard-server process whose RSS and CPU should be sampled.
    #[arg(long)]
    server_pid: Option<u32>,

    /// Largest passing first-to-last RSS increase in MiB; requires --server-pid.
    #[arg(long, requires = "server_pid")]
    max_rss_growth_mib: Option<f64>,

    /// Name of an environment variable that holds the endpoint's bearer token.
    #[arg(long)]
    api_key_env: Option<String>,

    /// New directory to create for config, interval, error, and summary files.
    #[arg(long)]
    results_dir: Option<PathBuf>,
}

impl Args {
    /// Reject invalid numeric combinations after clap parses their types.
    pub fn validate(&self) -> Result<(), String> {
        if self.concurrency == 0 {
            return Err("--concurrency must be greater than zero".to_string());
        }
        if self.max_output_tokens == 0 || self.prompt_bytes == 0 {
            return Err(
                "--max-output-tokens and --prompt-bytes must be greater than zero".to_string(),
            );
        }
        if !self.request_timeout.is_finite()
            || !self.report_interval.is_finite()
            || self.request_timeout <= 0.0
            || self.report_interval <= 0.0
        {
            return Err(
                "--request-timeout and --report-interval must be greater than zero".to_string(),
            );
        }
        if !self.invalid_canary_interval.is_finite() || self.invalid_canary_interval < 0.0 {
            return Err("--invalid-canary-interval must be zero or greater".to_string());
        }
        if !(0.0..=1.0).contains(&self.max_error_rate) {
            return Err("--max-error-rate must be between 0 and 1".to_string());
        }
        if self.server_pid == Some(0) {
            return Err("--server-pid must be greater than zero".to_string());
        }
        if self
            .max_rss_growth_mib
            .is_some_and(|growth| !growth.is_finite() || growth < 0.0)
        {
            return Err("--max-rss-growth-mib must be zero or greater".to_string());
        }
        Ok(())
    }
}

#[derive(Clone)]
struct RunContext {
    client: Client,
    base_url: String,
    stop: Arc<Stop>,
    stats: Arc<Mutex<RunStats>>,
    writer: Arc<Mutex<ResultsWriter>>,
}

struct Workload {
    model: String,
    prompts: Vec<String>,
    max_output_tokens: u32,
}

/// A one-shot stop signal that many tasks can wait on and any task can raise.
struct Stop {
    flag: AtomicBool,
    notify: Notify,
}

impl Stop {
    fn new() -> Self {
        Self {
            flag: AtomicBool::new(false),
            notify: Notify::new(),
        }
    }

    fn set(&self) {
        self.flag.store(true, Ordering::SeqCst);
        self.notify.notify_waiters();
    }

    fn is_set(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }

    /// Resolve once the signal is raised, now or later.
    async fn wait(&self) {
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

/// Send closed-loop traffic until the run stops.
async fn worker(
    context: RunContext,
    workload: Arc<Workload>,
    worker_id: usize,
    request_numbers: Arc<AtomicU64>,
) -> Result<(), String> {
    while !context.stop.is_set() {
        let request_number = request_numbers.fetch_add(1, Ordering::Relaxed) as usize;
        let endpoint = Endpoint::ALL[request_number % Endpoint::ALL.len()];
        let stream = (request_number / Endpoint::ALL.len()).is_multiple_of(2);
        let body = request_body(
            endpoint,
            &workload.model,
            &workload.prompts[request_number % workload.prompts.len()],
            workload.max_output_tokens,
            stream,
        );
        let started = Instant::now();
        let result = send_request(&context.client, &context.base_url, endpoint, &body).await;
        let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
        context.stats.lock().record(
            endpoint.as_str(),
            latency_ms,
            result.as_ref().err().map(|error| error.kind.as_str()),
        );
        if let Err(error) = result {
            context
                .writer
                .lock()
                .write_error(&json!({
                    "timestamp_utc": now_utc_string(),
                    "worker": worker_id,
                    "endpoint": endpoint.as_str(),
                    "stream": stream,
                    "latency_ms": round3(latency_ms),
                    "error": error.kind,
                    "detail": error.detail,
                }))
                .map_err(|error| error.to_string())?;
        }
    }
    Ok(())
}

/// Stop the run when a background task fails or panics.
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

/// Add one joined task's error to the failure reasons; cancelled tasks are expected.
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
fn spawn_signal_listener(stop: Arc<Stop>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{SignalKind, signal};
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
    })
}

/// Record every non-secret input with the normalized duration and fixed request variants.
fn write_config(results_dir: &Path, args: &Args, model: &str) -> Result<(), String> {
    let config = json!({
        "base_url": args.base_url,
        "model": model,
        "duration_seconds": args.duration,
        "concurrency": args.concurrency,
        "endpoints": Endpoint::ALL.map(Endpoint::as_str),
        "streaming": [true, false],
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
    args.validate()?;

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

    preflight(&client, &base_url, &args.model).await?;
    let results_dir = args
        .results_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from("soak-results").join(utc_dir_stamp()));
    let writer = Arc::new(Mutex::new(
        ResultsWriter::new(&results_dir).map_err(|error| error.to_string())?,
    ));
    write_config(&results_dir, &args, &args.model)?;

    println!(
        "Soak started: model={} duration={}s concurrency={} endpoints={} results={}",
        args.model,
        args.duration,
        args.concurrency,
        Endpoint::ALL
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
    let context = RunContext {
        client,
        base_url,
        stop: stop.clone(),
        stats: stats.clone(),
        writer: writer.clone(),
    };

    let signal_listener = spawn_signal_listener(stop.clone());

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

    let workload = Arc::new(Workload {
        model: args.model.clone(),
        prompts: build_prompt_pool(args.prompt_bytes),
        max_output_tokens: args.max_output_tokens,
    });
    let mut worker_handles = Vec::new();
    for worker_id in 0..args.concurrency {
        let handle = tokio::spawn(guard_stop(
            stop.clone(),
            worker(
                context.clone(),
                workload.clone(),
                worker_id,
                request_numbers.clone(),
            ),
        ));
        worker_handles.push((format!("worker-{worker_id}"), handle));
    }
    let reporter_handle = tokio::spawn(guard_stop(
        stop.clone(),
        reporter(
            context.clone(),
            started,
            Duration::from_secs_f64(args.report_interval),
            args.duration,
            args.server_pid,
            workers_done.clone(),
        ),
    ));
    let canary_handle = tokio::spawn(guard_stop(
        stop.clone(),
        invalid_request_canary(context, args.invalid_canary_interval, workload),
    ));

    stop.wait().await;
    deadline.abort();
    signal_listener.abort();

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
    let (error_records, dropped_error_records) = { writer.lock().error_counts() };
    let summary = build_summary(
        &stats.lock(),
        elapsed,
        args.max_error_rate,
        args.max_rss_growth_mib,
        error_records,
        dropped_error_records,
        &task_failures,
    );
    let summary_path = results_dir.join("summary.json");
    let summary_body = serde_json::to_string_pretty(&summary).map_err(|error| error.to_string())?;
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
    for reason in &summary.failure_reasons {
        println!("- {reason}");
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
