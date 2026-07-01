// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Experimental components-v2 crate for the flatter profile-owned design.
//!
//! This crate is intentionally separate from `switchyard-components` so the
//! rewrite shape can evolve without contaminating the existing production config surface.

extern crate self as switchyard_components_v2;

mod backend;
mod config;
pub mod decision;
mod features;
mod profile;
pub mod profiles;
mod stats;

pub use config::{
    parse_profile_config_path, parse_profile_config_str, parse_profile_config_str_with_env_lookup,
    ProfileConfig, ProfileConfigDocument, ProfileConfigFormat, ProfileConfigPlan,
};
pub use decision::{
    route_endpoint_for_format, route_protocol_for_format, CurrentRequestMaterialization,
    DecisionAttempt, DecisionProfile, DecisionProvider, IdentityQuality, RequestIdentity,
    RequestProtocol, RequestSummary, RoutingDecision, RoutingRequest, RoutingTarget,
    ROUTING_DECISION_SCHEMA_VERSION, ROUTING_REQUEST_SCHEMA_VERSION,
};
pub use features::{
    atof_event_dedupe_key, json_string_at, relay_identity_key_from_atof_event,
    RelayAccumulatorCounters, RelayIdentityKey, RelayIngestReport, RelaySnapshot,
    RelaySnapshotAccumulator, RelaySnapshotLimits, DEFAULT_MAX_ATOF_BATCH_BYTES,
    DEFAULT_MAX_ATOF_EVENT_BYTES, DEFAULT_MAX_RELAY_DEDUPE_ENTRIES,
    DEFAULT_MAX_RELAY_HISTORY_PER_IDENTITY, DEFAULT_MAX_RELAY_IDENTITIES,
    DEFAULT_MAX_RELAY_RETAINED_BYTES,
};
pub use profile::{
    Profile, ProfileHooks, ProfileInput, ProfileResponse, RequestMetadata, RoutingMetadata,
};
pub use profiles::{
    decision_for_llm_routing, decision_for_random_routing, EndpointHealth, EndpointHealthStatus,
    LatencyServiceProcessedRequest, LatencyServiceProfile, LatencyServiceProfileConfig,
    LlmRoutingDecision, LlmRoutingProcessedRequest, LlmRoutingProfile, LlmRoutingProfileConfig,
    LlmRoutingTierMapping, NoopProfile, NoopProfileConfig, PassthroughProfile,
    PassthroughProfileConfig, RandomRoutingProcessedRequest, RandomRoutingProfile,
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
