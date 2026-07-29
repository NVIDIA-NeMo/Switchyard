// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compatibility backends and processors for the Python Switchyard runtime.
//!
//! New Rust orchestration belongs in libsy algorithms and clients. The contracts
//! here remain only while the Python compatibility runtime uses these components.

pub mod backends;
mod contracts;
pub mod dimension_collector;
pub mod intake;
pub mod request_processors;
pub mod response_processors;
pub mod stage_router;
pub mod stats;
mod telemetry;

pub use backends::{
    AnthropicNativeBackend, BackendSelection, BackendSelectionReason, LlmTargetBackend,
    MultiLlmBackend, OpenAiNativeBackend, OpenAiPassthroughBackend, StatsLlmBackend,
};
pub use contracts::*;
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
pub use stats::{
    prefix_probe, tracking_enabled_from_env, ClassifierStatsSnapshot, CostBreakdown, CostEstimate,
    LatencyHistogramSnapshot, ModelStatsSnapshot, PrefixProbe, StatsAccumulator,
    StatsBackendLatency, StatsRequestStart, StatsRouteLabel, StatsSnapshot, TierStatsSnapshot,
    TokenTotals, TokenUsage,
};
