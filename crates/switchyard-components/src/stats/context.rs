// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed context markers shared by stats components.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use switchyard_core::ProxyContext;

use crate::backends::BackendSelection;
use crate::request_processors::RandomRoutingDecision;

/// Request start time captured by `StatsRequestProcessor`.
#[derive(Clone, Copy, Debug)]
pub struct StatsRequestStart(Instant);

impl StatsRequestStart {
    /// Captures the current monotonic clock instant.
    pub fn now() -> Self {
        Self(Instant::now())
    }

    /// Returns elapsed milliseconds since the captured start.
    pub fn elapsed_ms(self) -> f64 {
        self.0.elapsed().as_secs_f64() * 1000.0
    }
}

/// Backend-call duration captured by `StatsLlmBackend`.
#[derive(Clone, Copy, Debug)]
pub struct StatsBackendLatency(pub Duration);

impl StatsBackendLatency {
    /// Converts the duration to milliseconds.
    pub fn as_millis_f64(self) -> f64 {
        self.0.as_secs_f64() * 1000.0
    }
}

/// Optional generic tier label for non-random routing stats.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct StatsRouteLabel(pub String);

impl StatsRouteLabel {
    /// Creates a route label, preserving caller-provided naming.
    pub fn new(label: impl Into<String>) -> Self {
        Self(label.into())
    }
}

/// Returns the selected model label for stats attribution.
pub fn selected_stats_model(ctx: &ProxyContext, fallback: Option<&str>) -> String {
    ctx.get::<BackendSelection>()
        .map(|selection| selection.model.as_str().to_string())
        .or_else(|| fallback.map(str::to_string))
        .unwrap_or_else(|| "<unknown>".to_string())
}

/// Returns a tier label when a routing component supplied one.
pub fn selected_stats_tier(ctx: &ProxyContext) -> Option<String> {
    if let Some(label) = ctx.get::<StatsRouteLabel>() {
        return Some(label.0.clone());
    }
    if let Some(decision) = ctx.get::<RandomRoutingDecision>() {
        return Some(decision.tier.as_str().to_string());
    }
    // Fallback: Python-based pickers (e.g. stage_router) stamp selected_target but
    // don't write a typed marker — use it as the tier label so /v1/routing/stats
    // populates the `tiers` field for those routes too.
    ctx.selected_target().map(|t| t.as_str().to_string())
}
