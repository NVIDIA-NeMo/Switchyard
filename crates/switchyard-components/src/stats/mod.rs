// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared stats accounting used by stats processors and backend wrappers.

mod accumulator;
mod cache_eligibility;
mod context;
mod cost;
mod usage;

pub use accumulator::{
    ClassifierStatsSnapshot, LatencyHistogramSnapshot, ModelStatsSnapshot,
    OpenAiTransportStatsSnapshot, StatsAccumulator, StatsSnapshot, TierStatsSnapshot, TokenTotals,
};
pub use cache_eligibility::{prefix_probe, tracking_enabled_from_env, PrefixProbe};
pub use context::{
    selected_stats_model, selected_stats_tier, StatsBackendLatency, StatsRequestStart,
    StatsRouteLabel,
};
pub use cost::{estimate_model_cost, has_model_price, CostBreakdown, CostEstimate};
pub use usage::{
    openai_chat_usage_from_stream_event, openai_responses_usage_from_stream_event, usage_from_body,
    AnthropicStreamUsage, TokenUsage,
};
