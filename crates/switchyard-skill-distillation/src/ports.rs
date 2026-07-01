// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Async extension points implemented by source adapters and runtimes.

use async_trait::async_trait;

use crate::error::Result;
use crate::ids::{SkillNamespace, SkillVersionId};
use crate::model::{
    ActivationRecord, DistillationRequest, SkillCandidate, Trajectory, ValidationReport,
};

/// Loads normalized trajectories for a target namespace.
#[async_trait]
pub trait TrajectorySource: Send + Sync {
    /// Loads all trajectories currently available for `namespace`.
    async fn load(&self, namespace: &SkillNamespace) -> Result<Vec<Trajectory>>;
}

/// Converts normalized trajectories into a candidate skill.
#[async_trait]
pub trait SkillDistiller: Send + Sync {
    /// Produces a candidate without implicitly activating it.
    async fn distill(&self, request: &DistillationRequest) -> Result<SkillCandidate>;
}

/// Evaluates a candidate against optional evaluation trajectories.
#[async_trait]
pub trait SkillValidator: Send + Sync {
    /// Returns validation evidence; activation remains a caller decision.
    async fn validate(
        &self,
        candidate: &SkillCandidate,
        evaluation: &[Trajectory],
    ) -> Result<ValidationReport>;
}

/// Persists candidates and controls the active skill version.
#[async_trait]
pub trait SkillStore: Send + Sync {
    /// Returns the active candidate for `namespace`, when one exists.
    async fn active(&self, namespace: &SkillNamespace) -> Result<Option<SkillCandidate>>;

    /// Persists a candidate without activating it.
    async fn save_candidate(&self, candidate: &SkillCandidate) -> Result<()>;

    /// Activates a previously saved version.
    async fn activate(
        &self,
        namespace: &SkillNamespace,
        version: &SkillVersionId,
    ) -> Result<ActivationRecord>;

    /// Restores the immediately preceding active version.
    async fn rollback(&self, namespace: &SkillNamespace) -> Result<ActivationRecord>;
}
