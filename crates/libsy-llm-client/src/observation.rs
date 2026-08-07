// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-scoped observations emitted while serving an algorithm run.

use std::sync::Arc;
use std::time::Duration;

use switchyard_protocol::Usage;

/// One completed model call observed at the algorithm offload boundary.
#[derive(Clone, Debug)]
pub struct LlmCallObservation {
    /// Model selected for the completed call.
    pub selected_model: String,
    /// Routing tier attached to the selected model, when present.
    pub tier: Option<String>,
    /// Whether this was the routed backend call rather than classifier or judge overhead.
    pub is_routed: bool,
    /// Whether the call completed successfully.
    pub is_success: bool,
    /// Time spent waiting for the model call to resolve.
    pub duration: Duration,
    /// Normalized usage for a buffered successful response.
    pub usage: Option<Usage>,
}

/// One request-scoped observation emitted by the algorithm runner.
#[derive(Clone, Debug)]
pub enum RunObservation {
    /// A completed model call.
    LlmCall(LlmCallObservation),
    /// Routing time recorded by the `switchyard.routing_overhead_ms` metric.
    RoutingOverhead(Duration),
}

/// Request-scoped callback for algorithm-run observations.
///
/// The runner invokes this callback inline while resolving observations. Several
/// runs may invoke the same observer concurrently, so implementations must be
/// thread-safe, fast, and non-blocking.
pub type RunObserver = Arc<dyn Fn(RunObservation) + Send + Sync>;
