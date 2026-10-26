// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Periodic worker scrapes and process-local accounting for routed requests.

use anyhow::{Context, ensure};
use futures_util::{StreamExt, future::join_all};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use switchyard_protocol::{InfraSignals, ModelId, WorkerLoad};
use switchyard_runner::Route;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    poll_interval_ms: u64,
    timeout_ms: u64,
    max_concurrent: usize,
    workers: Vec<Worker>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Worker {
    id: String,
    model: ModelId,
    url: String,
    backend: Backend,
    labels: BTreeMap<String, String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Backend {
    Vllm,
    Sglang,
}

impl Backend {
    fn names(&self) -> (&str, &str) {
        match self {
            Self::Vllm => ("vllm:num_requests_running", "vllm:num_requests_waiting"),
            Self::Sglang => ("sglang:num_running_reqs", "sglang:num_queue_reqs"),
        }
    }
}

struct Collector {
    config: Config,
    client: reqwest::Client,
}

impl Collector {
    fn load(path: &Path) -> anyhow::Result<Self> {
        let config: Config = toml::from_str(&std::fs::read_to_string(path)?)?;
        ensure!(
            config.poll_interval_ms > 0
                && config.timeout_ms > 0
                && config.timeout_ms <= config.poll_interval_ms
                && (1..=64).contains(&config.max_concurrent)
                && !config.workers.is_empty(),
            "invalid metrics collection limits"
        );
        let mut ids = BTreeSet::new();
        let mut sources = BTreeSet::new();
        for worker in &config.workers {
            ensure!(
                !worker.id.is_empty() && ids.insert(&worker.id),
                "worker IDs must be nonempty and unique"
            );
            let url = reqwest::Url::parse(&worker.url)?;
            ensure!(
                matches!(url.scheme(), "http" | "https")
                    && url.username().is_empty()
                    && url.password().is_none(),
                "worker metrics URLs must use HTTP without embedded credentials"
            );
            ensure!(
                sources.insert((url.to_string(), &worker.labels)),
                "duplicate worker metrics source"
            );
        }
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(config.timeout_ms))
            .redirect(reqwest::redirect::Policy::none())
            .build()?;
        Ok(Self { config, client })
    }

    async fn scrape(&self, worker: &Worker) -> anyhow::Result<WorkerLoad> {
        let sampled_at = Instant::now();
        let response = self
            .client
            .get(&worker.url)
            .header("accept", "text/plain; version=0.0.4")
            .send()
            .await?
            .error_for_status()?;
        let mut bytes = Vec::new();
        let mut chunks = response.bytes_stream();
        while let Some(chunk) = chunks.next().await {
            let chunk = chunk?;
            ensure!(
                bytes.len() + chunk.len() <= 1024 * 1024,
                "worker metrics exceed 1 MiB"
            );
            bytes.extend_from_slice(&chunk);
        }
        let (_, scrape) = nom_openmetrics::parser::prometheus(std::str::from_utf8(&bytes)?)
            .map_err(|_| anyhow::anyhow!("invalid Prometheus exposition"))?;
        let (running, waiting) = worker.backend.names();
        Ok(WorkerLoad {
            sampled_at,
            running: gauge(&scrape, running, &worker.labels)?,
            waiting: gauge(&scrape, waiting, &worker.labels)?,
        })
    }

    async fn snapshot(&self) -> InfraSignals {
        let mut signals = InfraSignals::new();
        for batch in self.config.workers.chunks(self.config.max_concurrent) {
            let results = join_all(batch.iter().map(|worker| self.scrape(worker))).await;
            for (worker, result) in batch.iter().zip(results) {
                match result {
                    Ok(load) => {
                        signals
                            .entry(worker.model.clone())
                            .or_default()
                            .workers
                            .insert(worker.id.clone(), load);
                    }
                    Err(error) => {
                        tracing::warn!(worker = %worker.id, %error, "worker metrics unavailable")
                    }
                }
            }
        }
        signals
    }
}

fn gauge(
    scrape: &[nom_openmetrics::Family<'_>],
    name: &str,
    labels: &BTreeMap<String, String>,
) -> anyhow::Result<u64> {
    for descriptor in scrape.iter().flat_map(|family| &family.descriptors) {
        if let nom_openmetrics::MetricDescriptor::Type { metric, r#type } = descriptor {
            ensure!(
                *metric != name || matches!(r#type, nom_openmetrics::MetricType::Gauge),
                "scheduler metric must be a gauge"
            );
        }
    }
    let mut samples = scrape
        .iter()
        .flat_map(|family| &family.samples)
        .filter(|sample| {
            sample.name() == name
                && labels.iter().all(|(key, value)| {
                    sample
                        .labels()
                        .iter()
                        .any(|label| label.name == key && label.value == *value)
                })
        });
    let sample = samples.next().context("required scheduler gauge missing")?;
    ensure!(
        samples.next().is_none(),
        "ambiguous scheduler gauge; select one worker/rank with labels"
    );
    let value = sample.number();
    ensure!(
        value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value < u64::MAX as f64,
        "invalid scheduler gauge value"
    );
    Ok(value as u64)
}

pub struct Metrics {
    state: Arc<Mutex<InfraSignals>>,
    poller: tokio::task::JoinHandle<()>,
}

impl Metrics {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let collector = Collector::load(path)?;
        let state = Arc::new(Mutex::new(InfraSignals::new()));
        let shared = state.clone();
        let poller = tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(Duration::from_millis(collector.config.poll_interval_ms));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                interval.tick().await;
                let mut next = collector.snapshot().await;
                let mut current = shared.lock().unwrap_or_else(|p| p.into_inner());
                // Scrape failure removes the worker, while local request lifetimes survive refreshes.
                for (model, counts) in current.iter() {
                    let entry = next.entry(model.clone()).or_default();
                    entry.local_inflight = counts.local_inflight;
                }
                *current = next;
            }
        });
        Ok(Self { state, poller })
    }

    pub fn track(
        &self,
        route: &Route,
        outcome: &mut switchyard_runner::RoutingOutcome,
    ) -> anyhow::Result<InFlight> {
        let mut signals = self.state.lock().unwrap_or_else(|p| p.into_inner());
        route.apply_infra(outcome, &signals)?;
        let model = outcome.selected_model_id()?.clone();
        let counts = signals.entry(model.clone()).or_default();
        counts.local_inflight += 1;
        Ok(InFlight {
            state: self.state.clone(),
            model,
        })
    }
}

impl Drop for Metrics {
    fn drop(&mut self) {
        self.poller.abort();
    }
}

/// Lives until the ExtProc response stream finishes, errors or is dropped.
#[derive(Debug)]
pub struct InFlight {
    state: Arc<Mutex<InfraSignals>>,
    model: ModelId,
}

impl Drop for InFlight {
    fn drop(&mut self) {
        let mut signals = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(counts) = signals.get_mut(&self.model) {
            counts.local_inflight -= 1;
        }
    }
}

#[cfg(test)]
impl Metrics {
    pub fn empty_for_test() -> Self {
        Self {
            state: Arc::new(Mutex::new(InfraSignals::new())),
            poller: tokio::spawn(std::future::pending()),
        }
    }

    pub fn inflight_for_test(&self) -> u64 {
        self.state
            .lock()
            .unwrap()
            .values()
            .map(|v| v.local_inflight)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, extract::Path as HttpPath, routing::get};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use switchyard_protocol::{Request, text_request};
    use switchyard_runner::{RoutingOutcome, Runner};

    #[test]
    fn scheduler_gauges_require_one_finite_nonnegative_integer_series() {
        let labels = [("engine".into(), "0".into())].into();
        for backend in [Backend::Vllm, Backend::Sglang] {
            let (running, _) = backend.names();
            for (value, valid) in [
                ("0", true),
                ("8e0", true),
                ("NaN", false),
                ("-1", false),
                ("1.5", false),
            ] {
                let text = format!(
                    "# TYPE {running} gauge\n{running}{{engine=\"0\"}} {value}\n{running}{{engine=\"1\"}} 9\n"
                );
                let (_, scrape) = nom_openmetrics::parser::prometheus(&text).unwrap();
                assert_eq!(gauge(&scrape, running, &labels).is_ok(), valid, "{value}");
                assert!(gauge(&scrape, running, &BTreeMap::new()).is_err());
                assert!(gauge(&scrape, "missing", &labels).is_err());
            }
        }
    }

    #[tokio::test]
    async fn metrics_refresh_and_tracking_enforce_capacity() {
        let load = Arc::new(AtomicUsize::new(8));
        let serving_load = load.clone();
        let app = Router::new()
            .route("/metrics/{worker}", get(move |HttpPath(worker): HttpPath<String>| {
                let load = serving_load.load(Ordering::SeqCst);
                async move {
                    if load == 99 && worker == "small-b" { return "broken scrape".to_owned(); }
                    let waiting = if worker == "large" { 0 } else { load };
                    format!("# TYPE sglang:num_running_reqs gauge\nsglang:num_running_reqs{{engine=\"0\"}} 0\n# TYPE sglang:num_queue_reqs gauge\nsglang:num_queue_reqs{{engine=\"0\"}} {waiting}\n")
                }
            }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let routes = include_str!("../config/task-classifier.toml");
        let infra = routes.split_once("[routes.task.infra]").unwrap().1;
        let stage = format!(
            "{}\n[routes.stage.infra]\n{infra}",
            include_str!("../config/routes.toml")
        );
        assert!(
            Runner::from_toml(&stage)
                .err()
                .unwrap()
                .to_string()
                .contains("infra requires capability llm_classifier")
        );
        let runner = Runner::from_toml(routes).unwrap();
        let route = runner.route("auto").unwrap();
        let mut config = "poll_interval_ms=100\ntimeout_ms=80\nmax_concurrent=4\n".to_owned();
        for (id, model) in [
            ("small-a", "Qwen/Qwen3-0.6B"),
            ("small-b", "Qwen/Qwen3-0.6B"),
            ("large", "Qwen/Qwen3-1.7B"),
        ] {
            config.push_str(&format!("[[workers]]\nid=\"{id}\"\nmodel=\"{model}\"\nurl=\"http://{address}/metrics/{id}\"\nbackend=\"sglang\"\nlabels={{engine=\"0\"}}\n"));
        }
        let path = std::env::temp_dir().join(format!("switchyard-queues-{}.toml", address.port()));
        std::fs::write(&path, config).unwrap();
        let metrics = Metrics::load(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        for waiting in [8, 0, 99] {
            load.store(waiting, Ordering::SeqCst);
            tokio::time::timeout(Duration::from_secs(3), async {
                loop {
                    let ready = {
                        let state = metrics.state.lock().unwrap();
                        state
                            .get(&ModelId::from("Qwen/Qwen3-0.6B"))
                            .is_some_and(|m| {
                                if waiting == 99 {
                                    !m.workers.contains_key("small-b")
                                } else {
                                    m.workers
                                        .get("small-a")
                                        .is_some_and(|w| w.waiting == waiting as u64)
                                }
                            })
                    };
                    if ready {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            })
            .await
            .unwrap();
            let results = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..8)
                    .map(|_| {
                        let metrics = &metrics;
                        scope.spawn(move || {
                            let mut outcome = RoutingOutcome::route_to(
                                "Qwen/Qwen3-0.6B".into(),
                                vec![],
                                Request {
                                    llm_request: text_request(None, "hello"),
                                    ..Default::default()
                                },
                            );
                            let guard = metrics.track(route, &mut outcome).unwrap();
                            (outcome.selected_model_id().unwrap().to_string(), guard)
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().unwrap())
                    .collect::<Vec<_>>()
            });
            let moved = results
                .iter()
                .filter(|(model, _)| model == "Qwen/Qwen3-1.7B")
                .count();
            assert_eq!(moved, if waiting == 8 { 2 } else { 0 });
            drop(results);
            let state = metrics.state.lock().unwrap();
            assert!(state.values().all(|v| v.local_inflight == 0));
        }
        server.abort();
    }
}
