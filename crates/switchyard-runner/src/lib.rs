// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared configured routing for Switchyard serving surfaces.
//!
//! WASI builds omit HTTP-backed route execution.

mod algorithm;
#[cfg(not(target_os = "wasi"))]
mod algorithm_factory;
#[cfg(not(target_os = "wasi"))]
mod config;
#[cfg(not(target_os = "wasi"))]
mod eligibility;
#[cfg(not(target_os = "wasi"))]
mod failure;
#[cfg(not(target_os = "wasi"))]
mod privacy;
mod provider_key_redactor;
#[cfg(not(target_os = "wasi"))]
mod route;
#[cfg(not(target_os = "wasi"))]
mod runner;

#[cfg(not(target_os = "wasi"))]
pub use config::{ConfiguredTarget, RouteTargets, TargetModel, api_key_variables, route_targets};
/// Compatibility names for the effective target requirements.
#[cfg(not(target_os = "wasi"))]
pub use eligibility::{
    TargetRequirements as EligibilityDecision, TargetRequirements as PrivacyDecision,
};
#[cfg(not(target_os = "wasi"))]
pub use eligibility::{TargetRequirements, require_target_labels};

pub use algorithm::{
    AdvisorTriggerConfig, AlgorithmConfigError, AlgorithmSpec, CategoryModelConfig, ClassifierMode,
    ClassifierPolicyConfig, DecisionJudgeRouteConfig, LlmClassifierRouteConfig, PlanExecuteTrigger,
    StageClassifierConfig, StageProfileConfig, StageSettingsConfig, SubagentRouteConfig,
    VgrModeConfig, VgrRouteConfig,
};
#[cfg(not(target_os = "wasi"))]
pub use algorithm_factory::AlgorithmFactory;
#[cfg(not(target_os = "wasi"))]
pub use failure::{RouteErrorKind, RouteErrorPhase, RouteErrorSummary, stream_error_summary};
// Re-exported because `Route::new` takes it, so a host wiring routes does not need a libsy dep.
pub use libsy::{RoutingOutcome, RuntimeModels};
#[cfg(not(target_os = "wasi"))]
pub use privacy::require_restricted_targets;
pub use provider_key_redactor::ProviderKeyRedactor;
#[cfg(not(target_os = "wasi"))]
pub use route::{
    AuxiliaryTarget, CallerAuthKind, ModelCapabilities, Route, RunOutput, RunnerError,
};
#[cfg(not(target_os = "wasi"))]
pub use runner::{DecisionDescription, DecisionTarget, ModelInfo, Runner};
