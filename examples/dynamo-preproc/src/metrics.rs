// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prometheus pool observations and process-local request accounting.

use anyhow::{Context, ensure};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    path::Path,
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use switchyard_protocol::{InfraSignals, ModelId, PoolLoad};
use switchyard_runner::Route;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    endpoint: String,
    poll_interval_ms: u64,
    timeout_ms: u64,
    models: Vec<Model>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Model {
    model: ModelId,
    /// Prometheus target labels shared by up and the engine gauges.
    selector: String,
}

#[derive(Deserialize)]
struct Response {
    status: String,
    data: Vector,
    #[serde(default)]
    warnings: Vec<String>,
}

#[derive(Deserialize)]
struct Vector {
    #[serde(rename = "resultType")]
    kind: String,
    result: Vec<Sample>,
}

#[derive(Deserialize)]
struct Sample {
    metric: BTreeMap<String, String>,
    value: (f64, String),
}

type Targets = BTreeMap<(String, String), u64>;

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
                && !config.models.is_empty(),
            "invalid collection limits"
        );
        let mut models = std::collections::BTreeSet::new();
        for model in &config.models {
            ensure!(
                !model.selector.trim().is_empty() && models.insert(&model.model),
                "selectors must be nonempty and models unique"
            );
        }
        let url = reqwest::Url::parse(&config.endpoint)?;
        ensure!(
            matches!(url.scheme(), "http" | "https")
                && url.username().is_empty()
                && url.password().is_none(),
            "invalid Prometheus endpoint"
        );
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(config.timeout_ms))
            .redirect(reqwest::redirect::Policy::none())
            .build()?;
        Ok(Self { config, client })
    }

    async fn query(&self, query: &str, time: &str) -> anyhow::Result<Vec<Sample>> {
        let mut url = reqwest::Url::parse(&self.config.endpoint)?;
        url.query_pairs_mut()
            .extend_pairs([("query", query), ("time", time)]);
        let mut response = self.client.get(url).send().await?.error_for_status()?;
        let mut bytes = Vec::new();
        while let Some(chunk) = response.chunk().await? {
            ensure!(
                bytes.len() + chunk.len() <= 1024 * 1024,
                "Prometheus response exceeds 1 MiB"
            );
            bytes.extend_from_slice(&chunk);
        }
        let response: Response = serde_json::from_slice(&bytes)?;
        ensure!(
            response.status == "success"
                && response.data.kind == "vector"
                && response.warnings.is_empty(),
            "incomplete Prometheus response"
        );
        Ok(response.data.result)
    }

    async fn collect(&self, model: &Model) -> anyhow::Result<PoolLoad> {
        let started = Instant::now();
        let time = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_secs_f64()
            .to_string();
        let selector = &model.selector;
        let up = format!("up{{{selector}}}");
        let queue = format!("sglang:num_queue_reqs{{{selector}}}");
        let running = format!("sglang:num_running_reqs{{{selector}}}");
        // Use source timestamps: an instant-query result is stamped at evaluation time.
        let timestamps = [(&up, "up"), (&queue, "queue"), (&running, "running")]
            .map(|(metric, signal)| {
                format!("label_replace(timestamp({metric}), \"signal\", \"{signal}\", \"\", \"\")")
            })
            .join(" or ");
        let age = format!("time() - min({timestamps})");
        let (up, queue, running, age) = tokio::try_join!(
            self.query(&up, &time),
            self.query(&queue, &time),
            self.query(&running, &time),
            self.query(&age, &time)
        )?;
        let [age] = age.as_slice() else {
            anyhow::bail!("missing sample age");
        };
        let age = Duration::try_from_secs_f64(age.value.1.parse()?)?;
        let up = targets(up)?;
        let queue = targets(queue)?;
        let running = targets(running)?;
        ensure!(
            !up.is_empty()
                && up.values().all(|v| *v == 1)
                && up.keys().eq(queue.keys())
                && up.keys().eq(running.keys()),
            "missing or unhealthy pool targets"
        );
        Ok(PoolLoad {
            sampled_at: started.checked_sub(age).context("invalid sample age")?,
            min_waiting: *queue.values().min().unwrap(),
            max_waiting: *queue.values().max().unwrap(),
            running: running
                .values()
                .try_fold(0u64, |sum, n| sum.checked_add(*n))
                .context("running count overflow")?,
        })
    }

    async fn snapshot(&self) -> InfraSignals {
        let mut signals = InfraSignals::new();
        for model in &self.config.models {
            match self.collect(model).await {
                Ok(load) => {
                    signals.entry(model.model.clone()).or_default().load = Some(load);
                }
                Err(error) => {
                    tracing::warn!(model = %model.model, %error, "pool metrics unavailable")
                }
            }
        }
        signals
    }
}

fn targets(samples: Vec<Sample>) -> anyhow::Result<Targets> {
    let mut targets = Targets::new();
    for sample in samples {
        let key = (
            sample
                .metric
                .get("job")
                .context("missing job label")?
                .clone(),
            sample
                .metric
                .get("instance")
                .context("missing instance label")?
                .clone(),
        );
        let value: f64 = sample.value.1.parse()?;
        ensure!(
            value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value < u64::MAX as f64,
            "invalid count"
        );
        ensure!(
            targets.insert(key, value as u64).is_none(),
            "multiple engine series per target"
        );
    }
    Ok(targets)
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
                // Failed collection clears observations, but in-flight counts survive refreshes.
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
    use axum::{Json, Router, extract::Query, routing::get};
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use switchyard_protocol::{Request, text_request};
    use switchyard_runner::{RoutingOutcome, Runner};

    fn sample(instance: &str, value: &str) -> Sample {
        Sample {
            metric: [
                ("job".into(), "engines".into()),
                ("instance".into(), instance.into()),
            ]
            .into(),
            value: (1.0, value.into()),
        }
    }

    #[test]
    fn target_counts_require_unique_finite_nonnegative_integers() {
        for value in ["NaN", "-1", "1.5", "inf"] {
            assert!(targets(vec![sample("a", value)]).is_err());
        }
        assert_eq!(
            targets(vec![sample("a", "8e0")])
                .unwrap()
                .values()
                .copied()
                .sum::<u64>(),
            8
        );
        assert!(targets(vec![sample("a", "1"), sample("a", "2")]).is_err());
        let mut missing = sample("a", "1");
        missing.metric.remove("job");
        assert!(targets(vec![missing]).is_err());
    }

    #[tokio::test]
    async fn prometheus_collection_and_tracking_preserve_unknown_and_capacity() {
        let mode = Arc::new(AtomicUsize::new(0));
        let serving = mode.clone();
        let app = Router::new().route("/api/v1/query", get(move |Query(query): Query<BTreeMap<String, String>>| {
            let mode = serving.load(Ordering::SeqCst);
            async move {
                assert!(query.contains_key("time"));
                let q = &query["query"];
                let age = q.starts_with("time()");
                let small = q.contains("small");
                let count = if mode == 1 && q.starts_with("sglang:num_queue") { 1 } else { 2 };
                let result = if age {
                    assert!(q.contains("label_replace(timestamp("));
                    vec![json!({"metric":{}, "value":[1, if mode == 3 { "10" } else { "0.01" }]})]
                } else {
                    (0..count).map(|i| {
                        let n = if q.starts_with("up") { if mode == 2 { 0 } else { 1 } }
                            else if small && q.starts_with("sglang:num_queue") { 4 + i } else { 0 };
                        json!({"metric":{"job":"engines","instance":i.to_string()},"value":[1,n.to_string()]})
                    }).collect()
                };
                Json(json!({"status":"success","data":{"resultType":"vector","result":result},
                    "warnings": if mode == 4 { vec!["partial response"] } else { vec![] }}))
            }
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let config = include_str!("../config/metrics.toml").replace(
            "http://prometheus.monitoring.svc.cluster.local:9090",
            &format!("http://{address}"),
        );
        let path =
            std::env::temp_dir().join(format!("switchyard-prometheus-{}.toml", address.port()));
        std::fs::write(&path, config).unwrap();
        let collector = Collector::load(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        let metrics = Metrics::empty_for_test();
        let runner = Runner::from_toml(include_str!("../config/task-classifier.toml")).unwrap();
        let route = runner.route("auto").unwrap();
        for case in 0..5 {
            mode.store(case, Ordering::SeqCst);
            let snapshot = collector.snapshot().await;
            if case == 0 {
                let load = snapshot[&ModelId::from("Qwen/Qwen3-0.6B")]
                    .load
                    .as_ref()
                    .unwrap();
                assert_eq!(
                    (load.min_waiting, load.max_waiting, load.running),
                    (4, 5, 0)
                );
            } else if case != 3 {
                assert!(snapshot.is_empty());
            }
            *metrics.state.lock().unwrap() = snapshot;
            let guards = std::thread::scope(|scope| {
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
                            metrics.track(route, &mut outcome).unwrap()
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().unwrap())
                    .collect::<Vec<_>>()
            });
            assert_eq!(
                guards
                    .iter()
                    .filter(|g| g.model.as_str() == "Qwen/Qwen3-1.7B")
                    .count(),
                if case == 0 { 2 } else { 0 }
            );
            drop(guards);
            assert_eq!(metrics.inflight_for_test(), 0);
        }
        server.abort();
    }
}
