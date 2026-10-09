// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Choose among operator-approved models using request-specific serving observations.

use std::{sync::Arc, time::Duration};

use switchyard_protocol::{Category, Request};

use crate::{Algorithm, Driver, LibsyError, Result, RoutingOutcome};

/// A relative prefill-work policy. Its scores are not latency predictions.
pub struct CacheAware {
    costs: Vec<f64>,
    max_age: Duration,
}

impl CacheAware {
    /// Costs correspond to runtime targets in order; all must be finite and positive.
    pub fn new(costs: Vec<f64>, max_age: Duration) -> Result<Self> {
        if costs.is_empty()
            || costs.iter().any(|c| !c.is_finite() || *c <= 0.0)
            || max_age.is_zero()
        {
            return Err(LibsyError::AlgorithmError {
                message: "cache_aware requires positive costs and signal age".into(),
            });
        }
        Ok(Self { costs, max_age })
    }
}

#[async_trait::async_trait]
impl Algorithm for CacheAware {
    fn name(&self) -> &str {
        "cache_aware"
    }

    async fn route(self: Arc<Self>, driver: Driver, request: Request) -> Result<RoutingOutcome> {
        let models = driver.models_for(&Category::Any);
        let fallback = models.first().ok_or(LibsyError::NoTargets)?;
        if self.costs.len() != models.len() {
            return Err(LibsyError::AlgorithmError {
                message: "cache_aware costs must match runtime targets".into(),
            });
        }
        let observations = request.metadata.as_ref().map(|m| &m.serving_observations);
        // Incomplete observations must not make an unobserved model look worse.
        let scores: Option<Vec<f64>> = models
            .iter()
            .zip(&self.costs)
            .map(|(model, cost)| {
                let signal = observations?.get(model)?;
                if signal.received_at.elapsed() > self.max_age {
                    return None;
                }
                let queued = signal.active_prefill_tokens?;
                let score = (signal.effective_prefill_tokens as f64 + queued as f64) * cost;
                score.is_finite().then_some(score)
            })
            .collect();
        let selected = scores
            .as_ref()
            .and_then(|scores| {
                scores
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(index, _)| &models[index])
            })
            .unwrap_or(fallback);
        tracing::info!(model = %selected, used_signals = scores.is_some(), "cache-aware routing decision");
        Ok(RoutingOutcome::route_to(
            selected.clone(),
            Vec::new(),
            request,
        ))
    }
}
