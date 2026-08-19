// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test-time scaling for agentic coding tasks.
//!
//! The controller runs independent attempts, summarizes them, selects useful summaries for a
//! second set of fresh attempts, and selects one final result. Callers provide the agent harness
//! and model calls through [`ScalingBackend`].

#![deny(missing_docs)]

mod config;
mod controller;
mod error;
mod evaluation;
mod manifest;
mod model;
mod ports;
mod prompts;
mod record;
mod seed;
mod tournament;
mod verdict;

pub use config::{DisplayOrder, InvalidVotePolicy, PairingOrder, ScalingConfig, TiePolicy};
pub use controller::ScalingController;
pub use error::{Result, ScalingError};
pub use evaluation::{
    ExperimentMetrics, RolloutEvaluation, StageMetrics, TaskEvaluation, evaluate_run,
    experiment_metrics,
};
pub use manifest::{
    ExperimentManifest, MANIFEST_SCHEMA_VERSION, ManifestSource, ManifestValue,
    REQUIRED_MANIFEST_FIELDS, ReplicationMode,
};
pub use model::{
    Attempt, Candidate, ComparisonRequest, ComparisonResponse, GroupDecision, Rollout,
    RolloutRequest, ScalingRun, Summary, Task, Tournament, TournamentCheckpoint, Vote,
};
pub use ports::ScalingBackend;
pub use prompts::refinement_prompt;
pub use record::{decode_run, encode_run};
pub use tournament::run_tournament;
pub use verdict::parse_verdict;
