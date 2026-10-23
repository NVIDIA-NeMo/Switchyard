// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared configured routing for Switchyard serving surfaces.
//!
//! WASI builds omit HTTP-backed route execution.

mod algorithm;
#[cfg(not(target_os = "wasi"))]
mod config;
#[cfg(not(target_os = "wasi"))]
mod failure;
mod provider_key_redactor;
#[cfg(not(target_os = "wasi"))]
mod route;
#[cfg(not(target_os = "wasi"))]
mod runner;

#[cfg(not(target_os = "wasi"))]
pub use config::api_key_variables;

pub use algorithm::{
    AdvisorTriggerConfig, AlgorithmConfigError, AlgorithmSpec, CategoryModelConfig, ClassifierMode,
    ClassifierPolicyConfig, DecisionJudgeRouteConfig, LlmClassifierRouteConfig,
    StageClassifierConfig, SubagentRouteConfig, VgrModeConfig, VgrRouteConfig,
};
#[cfg(not(target_os = "wasi"))]
pub use failure::{RouteErrorKind, RouteErrorPhase, RouteErrorSummary, stream_error_summary};
// Re-exported because `Route::new` takes it, so a host wiring routes does not need a libsy dep.
pub use libsy::RuntimeModels;
pub use provider_key_redactor::ProviderKeyRedactor;
#[cfg(not(target_os = "wasi"))]
pub use route::{
    AuxiliaryTarget, CallerAuthKind, ModelCapabilities, Route, RunOutput, RunnerError,
};
#[cfg(not(target_os = "wasi"))]
pub use runner::{DecisionDescription, DecisionTarget, ModelInfo, Runner};
