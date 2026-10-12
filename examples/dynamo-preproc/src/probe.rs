// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-scoped EPP observations; transport stays outside the routing algorithm.

use std::{
    collections::BTreeMap,
    path::Path,
    time::{Duration, Instant},
};

use anyhow::{Context, ensure};
use futures_util::{StreamExt, stream};
use serde::Deserialize;
use switchyard_protocol::{ModelId, ServingObservation};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    mode: Mode,
    timeout_ms: u64,
    max_concurrent: usize,
    targets: BTreeMap<String, String>,
}

#[derive(Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum Mode {
    Observe,
    Routing,
}

pub struct Client {
    config: Config,
    client: reqwest::Client,
    permits: tokio::sync::Semaphore,
}

#[derive(Deserialize)]
struct ProbeResponse {
    model: String,
    prompt_tokens: usize,
    candidate: Candidate,
}

#[derive(Deserialize)]
struct Candidate {
    worker_id: String,
    cache: Cache,
    load: Load,
}

#[derive(Deserialize)]
struct Cache {
    effective_prefill_tokens: usize,
    gpu_prefix_tokens: Option<u64>,
    cpu_prefix_tokens: Option<u64>,
    disk_prefix_tokens: Option<u64>,
}

#[derive(Deserialize)]
struct Load {
    active_prefill_tokens: Option<usize>,
}

impl Client {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let config: Config = toml::from_str(&std::fs::read_to_string(path)?)?;
        ensure!(
            config.timeout_ms > 0 && (1..=64).contains(&config.max_concurrent),
            "invalid probe limits"
        );
        for endpoint in config.targets.values() {
            let url = reqwest::Url::parse(endpoint)?;
            ensure!(
                matches!(url.scheme(), "http" | "https")
                    && url.username().is_empty()
                    && url.password().is_none(),
                "probe targets must be HTTP URLs without credentials"
            );
        }
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(config.timeout_ms))
            .redirect(reqwest::redirect::Policy::none())
            .build()?;
        let permits = tokio::sync::Semaphore::new(config.max_concurrent);
        Ok(Self {
            config,
            client,
            permits,
        })
    }

    pub fn use_signals(&self) -> bool {
        self.config.mode == Mode::Routing
    }

    pub async fn collect(
        &self,
        body: &[u8],
        headers: &http::HeaderMap,
        targets: &[ModelId],
    ) -> BTreeMap<ModelId, ServingObservation> {
        let deadline = tokio::time::Instant::now() + Duration::from_millis(self.config.timeout_ms);
        stream::iter(targets.iter().cloned())
            .map(|model| async move {
                let result =
                    tokio::time::timeout_at(deadline, self.query(body, headers, &model)).await;
                match result {
                    Ok(Ok(signal)) => Some((model, signal)),
                    Ok(Err(error)) => {
                        tracing::info!(%model, %error,
                            request_id = headers.get("x-request-id").and_then(|v| v.to_str().ok()),
                            session_id = headers.get("x-switchyard-session-id").and_then(|v| v.to_str().ok()),
                            signal_status = "unavailable", "predicted cache reuse unavailable");
                        None
                    }
                    Err(_) => {
                        tracing::info!(%model,
                            request_id = headers.get("x-request-id").and_then(|v| v.to_str().ok()),
                            session_id = headers.get("x-switchyard-session-id").and_then(|v| v.to_str().ok()),
                            signal_status = "timeout", "predicted cache reuse unavailable");
                        None
                    }
                }
            })
            .buffer_unordered(self.config.max_concurrent)
            .filter_map(|v| async { v })
            .collect()
            .await
    }

    async fn query(
        &self,
        body: &[u8],
        headers: &http::HeaderMap,
        model: &ModelId,
    ) -> anyhow::Result<ServingObservation> {
        let _permit = self
            .permits
            .try_acquire()
            .context("probe concurrency limit reached")?;
        let endpoint = self
            .config
            .targets
            .get(model.as_str())
            .context("no configured probe target")?;
        let body = crate::router::replace_model(body, model.as_str())?;
        let mut request = self
            .client
            .post(endpoint)
            .header("content-type", "application/json")
            .body(body);
        // PreProc strips Dynamo routing headers from dispatch, so probes must not use them either.
        for name in ["x-request-id", "x-switchyard-session-id", "x-tenant-id"] {
            for value in headers.get_all(name).iter() {
                request = request.header(name, value);
            }
        }
        let response = request.send().await?.error_for_status()?;
        let mut bytes = Vec::new();
        let mut chunks = response.bytes_stream();
        while let Some(chunk) = chunks.next().await {
            let chunk = chunk?;
            ensure!(
                bytes.len() + chunk.len() <= 64 * 1024,
                "probe response too large"
            );
            bytes.extend_from_slice(&chunk);
        }
        let result: ProbeResponse = serde_json::from_slice(&bytes)?;
        ensure!(
            result.model == model.as_str(),
            "probe returned a different model"
        );
        let rate = |tokens: Option<u64>| {
            tokens
                .filter(|n| result.prompt_tokens > 0 && *n <= result.prompt_tokens as u64)
                .map(|n| n as f64 / result.prompt_tokens as f64)
        };
        tracing::info!(
            model = %model,
            candidate_worker_id = %result.candidate.worker_id,
            request_id = headers.get("x-request-id").and_then(|v| v.to_str().ok()),
            session_id = headers.get("x-switchyard-session-id").and_then(|v| v.to_str().ok()),
            prompt_tokens = result.prompt_tokens,
            gpu_prefix_tokens = result.candidate.cache.gpu_prefix_tokens,
            cpu_prefix_tokens = result.candidate.cache.cpu_prefix_tokens,
            disk_prefix_tokens = result.candidate.cache.disk_prefix_tokens,
            predicted_gpu_hit_rate = ?rate(result.candidate.cache.gpu_prefix_tokens),
            predicted_cpu_inclusive_hit_rate = ?rate(result.candidate.cache.cpu_prefix_tokens),
            predicted_disk_inclusive_hit_rate = ?rate(result.candidate.cache.disk_prefix_tokens),
            effective_prefill_tokens = result.candidate.cache.effective_prefill_tokens,
            "predicted cache reuse"
        );
        ensure!(
            rate(result.candidate.cache.gpu_prefix_tokens).is_some(),
            "cache estimate unavailable"
        );
        Ok(ServingObservation {
            received_at: Instant::now(),
            effective_prefill_tokens: result.candidate.cache.effective_prefill_tokens,
            active_prefill_tokens: result.candidate.load.active_prefill_tokens,
        })
    }
}
