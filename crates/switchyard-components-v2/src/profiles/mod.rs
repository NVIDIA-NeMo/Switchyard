// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Profile implementations in the flatter components-v2 design.

mod latency_service;
mod llm_routing;
mod macros;
mod noop;
mod passthrough;
mod profile_types;
mod random_routing;
mod stage_router;
mod subagent_override;

pub use latency_service::{
    EndpointHealth, EndpointHealthStatus, LatencyServiceProcessedRequest, LatencyServiceProfile,
    LatencyServiceProfileConfig, SelectedTarget,
};
pub use llm_routing::{
    LlmRoutingDecision, LlmRoutingProcessedRequest, LlmRoutingProfile, LlmRoutingProfileConfig,
    LlmRoutingTierMapping,
};
pub use noop::{NoopProfile, NoopProfileConfig};
pub use passthrough::{PassthroughProfile, PassthroughProfileConfig};
pub(crate) use profile_types::{parse_profile_config, ProfileConfigEntry};
pub use random_routing::{
    RandomRoutingProcessedRequest, RandomRoutingProfile, RandomRoutingProfileConfig,
};
pub use stage_router::{
    StageRouterClassifierConfig, StageRouterDecision, StageRouterDecisionSource,
    StageRouterPickerMode, StageRouterProcessedRequest, StageRouterProfile,
    StageRouterProfileConfig, StageRouterTier,
};
pub(crate) use subagent_override::SubagentOverrideProfile;
