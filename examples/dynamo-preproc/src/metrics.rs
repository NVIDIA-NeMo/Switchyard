// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prometheus pool observations and process-local request accounting.

use anyhow::{Context, ensure};
use serde::Deserialize;
use std::{
    collections::BTreeSet,
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
    /// Prometheus labels selecting the model pool.
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
    result: Vec<Sample>,
}

#[derive(Deserialize)]
struct Sample {
    value: (f64, String),
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
                && !config.models.is_empty(),
            "invalid collection limits"
        );
        let mut models = BTreeSet::new();
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

    async fn query(&self, query: String, time: &str) -> anyhow::Result<f64> {
        let mut url = reqwest::Url::parse(&self.config.endpoint)?;
        url.query_pairs_mut()
            .extend_pairs([("query", query.as_str()), ("time", time)]);
        let response: Response = self
            .client
            .get(url)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        ensure!(
            response.status == "success" && response.warnings.is_empty(),
            "incomplete Prometheus response"
        );
        let [sample] = response.data.result.as_slice() else {
            anyhow::bail!("expected one pool aggregate");
        };
        let value: f64 = sample.value.1.parse()?;
        ensure!(value.is_finite() && value >= 0.0, "invalid pool metric");
        Ok(value)
    }

    async fn collect(&self, model: &Model) -> anyhow::Result<PoolLoad> {
        let started = Instant::now();
        let time = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_secs_f64()
            .to_string();
        let selector = &model.selector;
        let queue = format!("sglang:num_queue_reqs{{{selector}}}");
        let running = format!("sglang:num_running_reqs{{{selector}}}");
        // Aggregated samples carry query time; freshness needs the original scrape times.
        let (min_waiting, max_waiting, running, queue_age, running_age) = tokio::try_join!(
            self.query(format!("min({queue})"), &time),
            self.query(format!("max({queue})"), &time),
            self.query(format!("sum({running})"), &time),
            self.query(format!("time() - min(timestamp({queue}))"), &time),
            self.query(format!("time() - min(timestamp({running}))"), &time)
        )?;
        Ok(PoolLoad {
            sampled_at: started
                .checked_sub(Duration::try_from_secs_f64(queue_age.max(running_age))?)
                .context("invalid sample age")?,
            min_waiting: count(min_waiting)?,
            max_waiting: count(max_waiting)?,
            running: count(running)?,
        })
    }
}

fn count(value: f64) -> anyhow::Result<u64> {
    ensure!(
        value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value < u64::MAX as f64,
        "invalid count"
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
                for model in &collector.config.models {
                    let load = collector.collect(model).await.inspect_err(|error| {
                        tracing::warn!(model = %model.model, %error, "pool metrics unavailable");
                    }).ok();
                    // Clear failed observations without resetting in-flight counts.
                    shared
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .entry(model.model.clone())
                        .or_default()
                        .load = load;
                }
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
    use std::{
        collections::BTreeMap,
        sync::atomic::{AtomicUsize, Ordering},
    };

    #[test]
    fn counts_require_finite_nonnegative_integers() {
        for value in [f64::NAN, -1.0, 1.5, f64::INFINITY, u64::MAX as f64] {
            assert!(count(value).is_err());
        }
        assert_eq!(count(0.0).unwrap(), 0);
        assert_eq!(count(8.0).unwrap(), 8);
    }

    #[tokio::test]
    async fn collection_preserves_missing_and_stale_signals() {
        let mode = Arc::new(AtomicUsize::new(0));
        let serving = mode.clone();
        let app = Router::new().route(
            "/api/v1/query",
            get(move |Query(query): Query<BTreeMap<String, String>>| {
                let mode = serving.load(Ordering::SeqCst);
                async move {
                    assert!(query.contains_key("time"));
                    let q = &query["query"];
                    let value = match q.split('(').next().unwrap() {
                        "time" if mode == 3 && q.contains("running") => "10",
                        "time" => "0.01",
                        "min" => "4",
                        "max" => "5",
                        "sum" => "2",
                        _ => panic!("unexpected query: {q}"),
                    };
                    let result = if mode == 1 {
                        vec![]
                    } else {
                        vec![json!({"metric":{}, "value":[1,value]})]
                    };
                    Json(
                        json!({"status":"success","data":{"resultType":"vector","result":result},
                    "warnings": if mode == 2 { vec!["partial response"] } else { vec![] }}),
                    )
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let mut config: Config = toml::from_str(include_str!("../config/metrics.toml")).unwrap();
        config.endpoint = format!("http://{address}/api/v1/query");
        let collector = Collector {
            config,
            client: reqwest::Client::new(),
        };
        for case in 0..4 {
            mode.store(case, Ordering::SeqCst);
            let result = collector.collect(&collector.config.models[0]).await;
            if case == 1 || case == 2 {
                assert!(result.is_err());
            } else {
                let load = result.unwrap();
                assert_eq!(
                    (load.min_waiting, load.max_waiting, load.running),
                    (4, 5, 2)
                );
                assert!(
                    load.sampled_at.elapsed()
                        >= Duration::from_secs_f64(if case == 3 { 10.0 } else { 0.01 })
                );
            }
        }
        server.abort();
    }
}
