// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-scoped EPP observations; transport stays outside the routing algorithm.

use std::{
    collections::BTreeMap,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Context, ensure};
use futures_util::{StreamExt, stream};
use serde::Deserialize;
use switchyard_protocol::{
    InfraProvider, InfraSignals, LlmRequest, ModelId, ModelInfraSignals, Request,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    timeout_ms: u64,
    max_concurrent: usize,
    targets: BTreeMap<String, String>,
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
    effective_prefill_tokens: Option<u64>,
    gpu_prefix_tokens: Option<u64>,
    cpu_prefix_tokens: Option<u64>,
    disk_prefix_tokens: Option<u64>,
}

#[derive(Deserialize)]
struct Load {
    active_prefill_tokens: Option<u64>,
    active_requests: Option<u64>,
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

    pub fn for_request(
        self: &Arc<Self>,
        body: &[u8],
        headers: &http::HeaderMap,
        original: Arc<LlmRequest>,
    ) -> Arc<dyn InfraProvider> {
        Arc::new(RequestProbe {
            client: self.clone(),
            body: body.to_vec(),
            headers: headers.clone(),
            original,
        })
    }

    pub async fn collect(
        &self,
        body: &[u8],
        headers: &http::HeaderMap,
        targets: &[ModelId],
    ) -> InfraSignals {
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
    ) -> anyhow::Result<ModelInfraSignals> {
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
        let started = Instant::now();
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
        let probe_elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
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
            probe_response = %std::str::from_utf8(&bytes)?,
            probe_elapsed_ms,
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
        Ok(ModelInfraSignals {
            sampled_at: started,
            effective_prefill_tokens: result.candidate.cache.effective_prefill_tokens,
            active_prefill_tokens: result.candidate.load.active_prefill_tokens,
            active_requests: result.candidate.load.active_requests,
        })
    }
}

struct RequestProbe {
    client: Arc<Client>,
    body: Vec<u8>,
    headers: http::HeaderMap,
    original: Arc<LlmRequest>,
}

#[async_trait::async_trait]
impl InfraProvider for RequestProbe {
    async fn probe(&self, request: &Request, candidates: &[ModelId]) -> InfraSignals {
        let mut current = request.llm_request.clone();
        current.model = None;
        // Probe the exact body that will be forwarded; rewritten requests violate this example's contract.
        if current != *self.original {
            return InfraSignals::new();
        }
        self.client
            .collect(&self.body, &self.headers, candidates)
            .await
    }
}
