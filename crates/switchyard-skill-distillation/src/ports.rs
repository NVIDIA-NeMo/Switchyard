// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Async extension points implemented by source adapters and runtimes.
//!
//! These traits describe separate workflow steps. They do not call one another or
//! decide when distillation runs or when a candidate becomes the active skill.

use std::future::Future;

use crate::error::Result;
use crate::ids::{SkillNamespace, SkillVersionId};
use crate::model::{
    ActivationRecord, DistillationRequest, SkillCandidate, Trajectory, ValidationReport,
};

/// Loads normalized trajectories for a target namespace.
pub trait TrajectorySource: Send + Sync {
    /// Loads all trajectories currently available for `namespace`.
    fn load(
        &self,
        namespace: &SkillNamespace,
    ) -> impl Future<Output = Result<Vec<Trajectory>>> + Send;
}

/// Converts normalized trajectories into a candidate skill.
pub trait SkillDistiller: Send + Sync {
    /// Produces a candidate without implicitly activating it.
    fn distill(
        &self,
        request: &DistillationRequest,
    ) -> impl Future<Output = Result<SkillCandidate>> + Send;
}

/// Evaluates a candidate against optional evaluation trajectories.
pub trait SkillValidator: Send + Sync {
    /// Returns validation evidence; activation remains a caller decision.
    fn validate(
        &self,
        candidate: &SkillCandidate,
        evaluation: &[Trajectory],
    ) -> impl Future<Output = Result<ValidationReport>> + Send;
}

/// Persists candidates and controls the active skill version.
pub trait SkillStore: Send + Sync {
    /// Returns the active candidate for `namespace`, when one exists.
    fn active(
        &self,
        namespace: &SkillNamespace,
    ) -> impl Future<Output = Result<Option<SkillCandidate>>> + Send;

    /// Persists a candidate without activating it.
    fn save_candidate(&self, candidate: &SkillCandidate)
    -> impl Future<Output = Result<()>> + Send;

    /// Activates a previously saved version.
    fn activate(
        &self,
        namespace: &SkillNamespace,
        version: &SkillVersionId,
    ) -> impl Future<Output = Result<ActivationRecord>> + Send;

    /// Restores the immediately preceding active version.
    fn rollback(
        &self,
        namespace: &SkillNamespace,
    ) -> impl Future<Output = Result<ActivationRecord>> + Send;
}
