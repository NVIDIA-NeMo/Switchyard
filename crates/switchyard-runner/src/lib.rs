// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared configured routing for Switchyard serving surfaces.
//!
//! WASI builds omit HTTP-backed route execution.

mod algorithm;
mod algorithm_factory;
mod config;
mod eligibility;
#[cfg(not(target_os = "wasi"))]
mod failure;
mod provider_key_redactor;
#[cfg(not(target_os = "wasi"))]
mod route;
#[cfg(not(target_os = "wasi"))]
mod runner;
mod runner_error;

pub use config::{ConfiguredTarget, RouteTargets, TargetModel, api_key_variables, route_targets};
pub use eligibility::require_target_labels;

pub use algorithm::{
    AdvisorTriggerConfig, AlgorithmConfigError, AlgorithmSpec, CategoryModelConfig, ClassifierMode,
    ClassifierPolicyConfig, DecisionJudgeRouteConfig, LlmClassifierRouteConfig,
    StageClassifierConfig, StageProfileConfig, StageSettingsConfig, SubagentRouteConfig,
    VgrModeConfig, VgrRouteConfig,
};
pub use algorithm_factory::{AlgorithmFactory, ConfiguredAlgorithm};
#[cfg(not(target_os = "wasi"))]
pub use failure::{RouteErrorKind, RouteErrorPhase, RouteErrorSummary, stream_error_summary};
// Re-exported because `Route::new` takes it, so a host wiring routes does not need a libsy dep.
pub use libsy::{RoutingOutcome, RuntimeModels};
pub use provider_key_redactor::ProviderKeyRedactor;
#[cfg(not(target_os = "wasi"))]
pub use route::{AuxiliaryTarget, ModelCapabilities, Route, RunOutput};
#[cfg(not(target_os = "wasi"))]
pub use runner::{DecisionDescription, DecisionTarget, ModelInfo, Runner};
pub use runner_error::{CallerAuthKind, RunnerError};
