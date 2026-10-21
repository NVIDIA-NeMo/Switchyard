// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sample alternative next actions while preserving the selected assistant response.

use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::future::join_all;
use serde::Deserialize;
use serde_json::json;
use switchyard_protocol::{Category, LlmClientError, Request, Tool};

use super::util::buffered_response::buffer_response;
use super::util::llm_judge::libsy_error_reason;
use super::util::prompts::drop_exact_replay;
use crate::{Algorithm, Driver, LibsyError, Result, RoutingOutcome};

/// Settings for sampling candidate actions from one generator.
#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BestOfNActionsConfig {
    /// Number of independent model calls per request.
    pub num_candidates: usize,
    /// Overrides the caller's sampling temperature, including when sampling once.
    pub temperature: f64,
    /// Deadline for each sampled call and its complete stream when sampling more than once.
    pub candidate_timeout_ms: u64,
}

impl Default for BestOfNActionsConfig {
    fn default() -> Self {
        Self {
            num_candidates: 4,
            temperature: 0.8,
            candidate_timeout_ms: 60_000,
        }
    }
}

/// Samples next actions concurrently and returns the first sample in request order.
pub struct BestOfNActions {
    config: BestOfNActionsConfig,
}

impl BestOfNActions {
    /// Creates a sampler with a positive sample count and deadline, and temperature in `0..=2`.
    pub fn new(config: BestOfNActionsConfig) -> Result<Self> {
        let invalid = if config.num_candidates == 0 {
            Some("num_candidates must be at least 1")
        } else if !(0.0..=2.0).contains(&config.temperature) {
            Some("temperature must be finite and between 0 and 2")
        } else if config.candidate_timeout_ms == 0 {
            Some("candidate_timeout_ms must be at least 1")
        } else {
            None
        };
        if let Some(message) = invalid {
            return Err(LibsyError::AlgorithmError {
                message: format!("best_of_n_actions: {message}"),
            });
        }
        Ok(Self { config })
    }
}

#[async_trait::async_trait]
impl Algorithm for BestOfNActions {
    fn name(&self) -> &str {
        "best_of_n_actions"
    }

    fn prepare_request(&self, request: &mut Request) -> Result<()> {
        if self.config.num_candidates > 1 {
            let fields = &request.llm_request.extensions.fields;
            // TODO: Materialize provider-owned history and isolate state for each candidate.
            let invalid = if ["previous_response_id", "conversation"]
                .iter()
                .any(|key| fields.get(*key).is_some_and(|value| !value.is_null()))
            {
                Some(
                    "sampling requires history in the request, without previous_response_id or conversation",
                )
            } else if request
                .llm_request
                .tools
                .iter()
                .any(|tool| !matches!(tool, Tool::Function(_)))
            {
                // Provider tools can execute before selection, including on discarded samples.
                Some("sampling supports only harness-executed function tools")
            } else if fields
                .get("n")
                .is_some_and(|value| !value.is_null() && value.as_u64() != Some(1))
            {
                Some("sampling requires the provider n field to be absent or 1")
            } else {
                None
            };
            if let Some(message) = invalid {
                return Err(LibsyError::client_call(
                    request.model_id().unwrap_or_default(),
                    LlmClientError::InvalidRequest {
                        message: format!("best_of_n_actions: {message}"),
                    },
                ));
            }
        }
        request.llm_request.sampling.temperature = Some(self.config.temperature);
        // Same-format replay would otherwise silently restore the caller's temperature.
        drop_exact_replay(request);
        request.raw_request = None;
        Ok(())
    }

    async fn route(
        self: Arc<Self>,
        driver: Driver,
        mut request: Request,
    ) -> Result<RoutingOutcome> {
        self.prepare_request(&mut request)?;
        // TODO: Sample from an ordered allocation of multiple generator models.
        let generator = driver.first_model_for(&Category::Any)?.clone();
        if self.config.num_candidates == 1 {
            return Ok(RoutingOutcome::route_to(generator, vec![], request));
        }

        let deadline = Duration::from_millis(self.config.candidate_timeout_ms);
        let candidates = join_all((0..self.config.num_candidates).map(|index| {
            let driver = driver.clone();
            let generator = generator.clone();
            let request = request.clone();
            async move {
                let started = Instant::now();
                let result = tokio::time::timeout(deadline, async {
                    let response = driver
                        .call_model_with_error_recovery(request, vec![generator.clone()], true)
                        .await?;
                    buffer_response(generator.as_str(), response).await
                })
                .await
                .unwrap_or_else(|source| {
                    Err(LibsyError::client_call(
                        generator,
                        LlmClientError::Timeout {
                            source: Box::new(source),
                        },
                    ))
                });
                let reason = result.as_ref().err().map(libsy_error_reason);
                let duration_ms = started.elapsed().as_secs_f64() * 1_000.0;
                tracing::debug!(
                    candidate_index = index,
                    duration_ms,
                    is_success = result.is_ok(),
                    reason_code = reason,
                    "action sample completed"
                );
                let evidence = json!({
                    "index": index,
                    "is_success": result.is_ok(),
                    "duration_ms": duration_ms,
                    "reason_code": reason,
                    "usage": result.as_ref().ok().map(|response| &response.agg.usage),
                });
                (result, evidence)
            }
        }))
        .await;

        driver.set_evidence(json!({
            "source": "best_of_n_actions",
            "num_candidates": self.config.num_candidates,
            "temperature": self.config.temperature,
            "selected_index": 0,
            "candidates": candidates.iter().map(|(_, evidence)| evidence).collect::<Vec<_>>(),
        }));

        // TODO: Filter invalid tool actions and deduplicate identical tool names and arguments.
        // TODO: Add listwise, pointwise, or pairwise verification to select the best action.
        // TODO: Define judge context limits, invalid-verdict fallback, and tie handling.
        // join_all retains request order, so completion speed cannot change the selected action.
        let (selected, _) = candidates
            .into_iter()
            .next()
            .ok_or(LibsyError::MissingFinalResponse)?;
        Ok(RoutingOutcome::answered(
            generator,
            request,
            selected?.into_response(),
        ))
    }
}

#[cfg(test)]
mod tests;
