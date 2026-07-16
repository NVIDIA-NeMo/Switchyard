// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Experimental components-v2 crate for the flatter profile-owned design.
//!
//! This crate is intentionally separate from `switchyard-components` so the
//! rewrite shape can evolve without contaminating the existing production config surface.

extern crate self as switchyard_components_v2;

mod backend;
mod config;
mod profile;
pub mod profiles;
mod stats;

pub use config::{
    parse_profile_config_path, parse_profile_config_str, parse_profile_config_str_with_env_lookup,
    ProfileConfig, ProfileConfigDocument, ProfileConfigFormat, ProfileConfigPlan,
};
pub use profile::{
    Profile, ProfileHooks, ProfileInput, ProfileResponse, RequestMetadata, RoutingMetadata,
};
pub use profiles::{
    EndpointHealth, EndpointHealthStatus, LatencyServiceProcessedRequest, LatencyServiceProfile,
    LatencyServiceProfileConfig, LlmRoutingDecision, LlmRoutingProcessedRequest, LlmRoutingProfile,
    LlmRoutingProfileConfig, LlmRoutingTierMapping, NoopProfile, NoopProfileConfig,
    PassthroughProfile, PassthroughProfileConfig, PrefillProbeDecision,
    PrefillProbeProcessedRequest, PrefillProbeProfile, PrefillProbeProfileConfig,
    PrefillProbeRoutingPolicyConfig, RandomRoutingProcessedRequest, RandomRoutingProfile,
    RandomRoutingProfileConfig, SelectedTarget, StageRouterClassifierConfig, StageRouterDecision,
    StageRouterDecisionSource, StageRouterPickerMode, StageRouterProcessedRequest,
    StageRouterProfile, StageRouterProfileConfig, StageRouterTier,
};
pub use stats::profile_stats_accumulator;
pub use switchyard_components_v2_macros::profile_config;

/// Implementation details used by generated profile-config code.
///
/// This namespace is public only so proc-macro expansion has a stable path.
/// It is not part of the user-facing components-v2 API.
#[doc(hidden)]
pub mod __private {
    pub use crate::config::{ProfileBuildEnv, ProfileConfigDefinition};
}
