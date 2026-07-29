// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend wrapper that records backend-call stats.

use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use crate::{ChatRequest, ChatRequestType, ChatResponse, LlmBackend, ProxyContext, Result};
use async_trait::async_trait;

use crate::stats::{
    selected_stats_model, selected_stats_tier, StatsAccumulator, StatsBackendLatency,
};

/// Transparent backend wrapper that records call success/error and backend latency.
#[derive(Clone)]
pub struct StatsLlmBackend {
    inner: Arc<dyn LlmBackend>,
    accumulator: StatsAccumulator,
}

impl StatsLlmBackend {
    /// Creates a stats wrapper around an existing backend.
    pub fn new(inner: Arc<dyn LlmBackend>, accumulator: StatsAccumulator) -> Self {
        Self { inner, accumulator }
    }

    /// Returns the wrapped backend.
    pub fn inner(&self) -> &dyn LlmBackend {
        self.inner.as_ref()
    }

    /// Returns the shared accumulator.
    pub fn accumulator(&self) -> &StatsAccumulator {
        &self.accumulator
    }
}

impl fmt::Debug for StatsLlmBackend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StatsLlmBackend")
            .field("accumulator", &self.accumulator)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl LlmBackend for StatsLlmBackend {
    fn supported_request_types(&self) -> &[ChatRequestType] {
        self.inner.supported_request_types()
    }

    async fn call(&self, ctx: &mut ProxyContext, request: &ChatRequest) -> Result<ChatResponse> {
        let request_model = request.model().map(str::to_string);
        let started_at = Instant::now();
        match self.inner.call(ctx, request).await {
            Ok(response) => {
                let latency = started_at.elapsed();
                ctx.insert(StatsBackendLatency(latency));
                let model = selected_stats_model(ctx, request_model.as_deref());
                let tier = selected_stats_tier(ctx);
                self.accumulator.record_success(
                    model,
                    Some(latency.as_secs_f64() * 1000.0),
                    tier.as_deref(),
                )?;
                Ok(response)
            }
            Err(error) => {
                let model = selected_stats_model(ctx, request_model.as_deref());
                let tier = selected_stats_tier(ctx);
                self.accumulator.record_error(model, tier.as_deref())?;
                Err(error)
            }
        }
    }

    async fn startup(&self) -> Result<()> {
        self.inner.startup().await
    }

    async fn shutdown(&self) -> Result<()> {
        self.inner.shutdown().await
    }
}
