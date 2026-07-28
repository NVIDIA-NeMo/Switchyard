// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Concrete Switchyard implementations built on `switchyard-core`.
//!
//! `switchyard-core` owns traits and wire wrappers. This crate owns built-in
//! compatibility implementations: request processors, response processors, and
//! observability helpers. LLM calls belong in `switchyard-llm-client`.

pub mod dimension_collector;
pub mod intake;
pub mod request_processors;
pub mod response_processors;
pub mod selection;
pub mod stage_router;
pub mod stats;
mod telemetry;

pub use dimension_collector::{
    extract_tool_signals, ResponseFlag, ResponseSignals, ToolResultSignal,
};
pub use intake::{
    HttpIntakeSink, IntakeFormat, IntakePayloadBuilder, IntakeQueueFullPolicy,
    IntakeRequestMetadata, IntakeRequestState, IntakeSink, IntakeSinkConfig, IntakeTarget,
    RequestMetadata, SubModelCall, SubModelCalls,
};
pub use request_processors::{
    DimensionCollector, IntakeRequestProcessor, RandomRoutingDecision, RandomRoutingEngine,
    RandomRoutingProcessorConfig, RandomRoutingTier, StatsRequestProcessor,
};
pub use response_processors::{
    IntakeResponseProcessor, ResponseSignalCollector, StatsResponseProcessor,
};
pub use selection::{BackendSelection, BackendSelectionReason};
pub use stats::{
    prefix_probe, tracking_enabled_from_env, ClassifierStatsSnapshot, CostBreakdown, CostEstimate,
    LatencyHistogramSnapshot, ModelStatsSnapshot, PrefixProbe, StatsAccumulator,
    StatsBackendLatency, StatsRequestStart, StatsRouteLabel, StatsSnapshot, TierStatsSnapshot,
    TokenTotals, TokenUsage,
};
