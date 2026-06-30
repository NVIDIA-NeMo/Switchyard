// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Concrete Switchyard implementations built on `switchyard-core`.
//!
//! `switchyard-core` owns traits and wire wrappers. This crate owns built-in
//! compatibility implementations: backends, request processors, and response
//! processors. New Rust orchestration belongs in components-v2 profiles.

pub mod backends;
pub mod dimension_collector;
pub mod intake;
pub mod otel_metrics;
pub mod request_processors;
pub mod response_processors;
pub mod stats;
mod telemetry;

pub use backends::{
    AnthropicNativeBackend, BackendSelection, BackendSelectionReason, LlmTargetBackend,
    MultiLlmBackend, OpenAiNativeBackend, OpenAiPassthroughBackend,
};
pub use dimension_collector::{
    extract_tool_signals, ContextSignals, DimensionScore, Keywords, ResponseFlag, ResponseSignals,
    ScoringConfig, ToolResultSignal,
};
pub use intake::{
    HttpIntakeSink, IntakePayloadBuilder, IntakeQueueFullPolicy, IntakeRequestMetadata,
    IntakeRequestState, IntakeSink, IntakeSinkConfig, RequestMetadata,
};
pub use request_processors::{
    DimensionCollector, IntakeRequestProcessor, RandomRoutingDecision, RandomRoutingEngine,
    RandomRoutingProcessorConfig, RandomRoutingTier,
};
pub use response_processors::{IntakeResponseProcessor, ResponseSignalCollector};
pub use stats::{
    prefix_probe, tracking_enabled_from_env, CostBreakdown, PrefixProbe, StatsBackendLatency,
    StatsRouteLabel, TokenUsage,
};
