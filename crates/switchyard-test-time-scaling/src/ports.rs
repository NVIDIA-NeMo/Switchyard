// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Adapter boundary for agent harnesses and model calls.

use async_trait::async_trait;

use crate::{
    ComparisonRequest, ComparisonResponse, Result, Rollout, RolloutRequest, Summary, Task,
};

/// Runs all three model roles for one logical model identity.
///
/// Every rollout returned by [`run_rollouts`](Self::run_rollouts) must use a fresh environment.
/// Official grader results must not be included in summaries or comparison responses.
#[async_trait]
pub trait ScalingBackend: Send + Sync {
    /// Harness-owned rollout output, such as a patch or artifact manifest.
    type Output: Clone + Send + Sync + 'static;

    /// Exact model identity used for attempts, summaries, and comparisons.
    fn model_id(&self) -> &str;

    /// Runs one iteration of independent agent attempts in fresh environments.
    async fn run_rollouts(
        &self,
        task: &Task,
        requests: Vec<RolloutRequest>,
    ) -> Result<Vec<Rollout<Self::Output>>>;

    /// Produces one structured JSON summary for a completed attempt.
    async fn summarize(&self, task: &Task, rollout: &Rollout<Self::Output>) -> Result<Summary>;

    /// Compares candidates and returns a response ending in one strict verdict line.
    async fn compare(&self, task: &Task, request: ComparisonRequest) -> Result<ComparisonResponse>;
}
