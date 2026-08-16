// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-scoped observations emitted while serving an algorithm run.

use std::sync::Arc;
use std::time::Duration;

use switchyard_protocol::{Decision, LlmRequest, ModelId, Usage};

/// Prompt and model-produced content from one non-answer classifier or judge call.
#[derive(Clone, Debug)]
pub struct ClassifierContentObservation {
    /// Model that produced the routing verdict.
    pub selected_model: ModelId,
    /// Exact normalized request sent to the classifier target, excluding transport headers.
    pub request: LlmRequest,
    /// Model-produced reasoning content, when the provider returned it separately.
    pub reasoning: Option<String>,
    /// Text verdict consumed by the routing policy, including invalid replies.
    pub verdict: Option<String>,
    /// Whether the provider call itself completed successfully.
    pub is_success: bool,
    /// Time spent waiting for the classifier call to resolve.
    pub duration: Duration,
}

/// One model call observed immediately before it is sent to its routed client.
#[derive(Clone, Debug)]
pub struct LlmCallStartObservation {
    /// Model selected for the call.
    pub selected_model: ModelId,
    /// Whether this call generates an answer rather than a routing verdict.
    pub is_answer_call: bool,
}

/// One completed model call observed at the algorithm offload boundary.
#[derive(Clone, Debug)]
pub struct LlmCallObservation {
    /// Model selected for the completed call.
    pub selected_model: ModelId,
    /// Whether this call generated an answer rather than a routing verdict.
    pub is_answer_call: bool,
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
    /// A routing decision, observed before its answer call starts.
    RoutingDecision(Decision),
    /// A model call about to start.
    LlmCallStarted(LlmCallStartObservation),
    /// Prompt, reasoning, and verdict from a classifier or judge call.
    ClassifierContent(ClassifierContentObservation),
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
