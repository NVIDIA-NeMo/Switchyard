// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared configured routing for Switchyard serving surfaces.

mod algorithm;
mod config;
mod route;
mod runner;

pub use algorithm::{
    AdvisorTriggerConfig, AlgorithmConfigError, AlgorithmSpec, ClassifierMode,
    ClassifierPolicyConfig, LlmClassifierRouteConfig, StageClassifierConfig, SubagentRouteConfig,
};
pub use route::{
    CallerAuthKind, CountTokensTarget, ModelCapabilities, Route, RunOutput, RunnerError,
};
pub use runner::{DecisionDescription, DecisionTarget, ModelInfo, Runner};
