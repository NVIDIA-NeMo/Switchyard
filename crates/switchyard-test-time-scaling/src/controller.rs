// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Two-iteration scaling controller.

use std::collections::HashSet;

use futures::future::try_join_all;

use crate::prompts::refinement_prompt;
use crate::seed;
use crate::{
    Attempt, Candidate, ExperimentManifest, Result, Rollout, RolloutRequest, ScalingBackend,
    ScalingConfig, ScalingError, ScalingRun, Summary, Task, run_tournament,
};

/// Runs parallel-distill-refine followed by recursive tournament voting.
pub struct ScalingController<B> {
    backend: B,
    config: ScalingConfig,
    manifest: ExperimentManifest,
}

impl<B> ScalingController<B>
where
    B: ScalingBackend,
{
    /// Creates a controller after validating its configuration.
    pub fn new(backend: B, config: ScalingConfig, manifest: ExperimentManifest) -> Result<Self> {
        config.validate()?;
        validate_controller_schedule(&config)?;
        manifest.validate()?;
        if backend.model_id().trim().is_empty() {
            return Err(ScalingError::InvalidConfig(
                "backend model_id must not be empty".to_string(),
            ));
        }
        if backend.model_id() != manifest.model_id {
            return Err(ScalingError::InvalidConfig(
                "backend model_id must match the experiment manifest".to_string(),
            ));
        }
        Ok(Self {
            backend,
            config,
            manifest,
        })
    }

    /// Returns the active workflow settings.
    pub fn config(&self) -> &ScalingConfig {
        &self.config
    }

    /// Runs the complete workflow for one task.
    pub async fn run(&self, task: Task) -> Result<ScalingRun<B::Output>> {
        validate_task(&task)?;
        let iteration_zero = self.run_iteration(&task, 0, Vec::new()).await?;
        let zero_candidates: Vec<Candidate> =
            iteration_zero.iter().map(Candidate::from_attempt).collect();
        let iteration_zero_tournament = run_tournament(
            &self.backend,
            &task,
            zero_candidates.clone(),
            &self.config,
            1,
            Some(self.config.refinement_count),
            seed::derive(self.config.seed, &[10]),
        )
        .await?;
        let checkpoint = iteration_zero_tournament
            .checkpoint
            .as_ref()
            .ok_or_else(|| {
                ScalingError::InvalidRecord(
                    "iteration-zero tournament did not save refinement candidates".to_string(),
                )
            })?;
        let refinement_summaries = summaries_for_ids(&zero_candidates, &checkpoint.candidate_ids)?;
        let refinement_summary_ids = refinement_summaries
            .iter()
            .map(|summary| summary.id.clone())
            .collect();

        let iteration_one = self.run_iteration(&task, 1, refinement_summaries).await?;
        validate_attempt_records(&iteration_zero, &iteration_one)?;
        let one_candidates: Vec<Candidate> =
            iteration_one.iter().map(Candidate::from_attempt).collect();
        let final_tournament = run_tournament(
            &self.backend,
            &task,
            one_candidates,
            &self.config,
            1,
            None,
            seed::derive(self.config.seed, &[20]),
        )
        .await?;
        let Some(final_candidate_id) = final_tournament.survivor_candidate_ids.first() else {
            return Err(ScalingError::InvalidRecord(
                "final tournament has no survivor".to_string(),
            ));
        };
        let final_rollout = iteration_one
            .iter()
            .find(|attempt| &attempt.rollout.id == final_candidate_id)
            .map(|attempt| attempt.rollout.clone())
            .ok_or_else(|| {
                ScalingError::InvalidRecord(
                    "final tournament survivor has no matching rollout".to_string(),
                )
            })?;

        Ok(ScalingRun {
            manifest: self.manifest.clone(),
            task,
            model_id: self.backend.model_id().to_string(),
            config: self.config.clone(),
            iteration_zero,
            iteration_zero_tournament,
            refinement_summary_ids,
            iteration_one,
            final_tournament,
            final_rollout,
        })
    }

    async fn run_iteration(
        &self,
        task: &Task,
        iteration: u8,
        refinement_summaries: Vec<Summary>,
    ) -> Result<Vec<Attempt<B::Output>>> {
        let rendered_refinement =
            (!refinement_summaries.is_empty()).then(|| refinement_prompt(&refinement_summaries));
        let requests: Vec<RolloutRequest> = (0..self.config.rollout_count)
            .map(|rollout_index| RolloutRequest {
                iteration,
                rollout_index,
                seed: seed::derive(
                    self.config.seed,
                    &[30, iteration as u64, rollout_index as u64],
                ),
                refinement_summaries: refinement_summaries.clone(),
                refinement_prompt: rendered_refinement.clone(),
            })
            .collect();
        let rollouts = self.backend.run_rollouts(task, requests.clone()).await?;
        if rollouts.len() != requests.len() {
            return Err(ScalingError::InvalidRecord(format!(
                "backend returned {} rollouts; expected {}",
                rollouts.len(),
                requests.len()
            )));
        }
        let summaries: Vec<Summary> = try_join_all(
            rollouts
                .iter()
                .map(|rollout| self.backend.summarize(task, rollout)),
        )
        .await?;

        requests
            .into_iter()
            .zip(rollouts)
            .zip(summaries)
            .map(|((request, rollout), summary)| {
                validate_attempt(&request, &rollout, &summary, self.backend.model_id())?;
                Ok(Attempt {
                    request,
                    rollout,
                    summary,
                })
            })
            .collect()
    }
}

fn summaries_for_ids(candidates: &[Candidate], ids: &[String]) -> Result<Vec<Summary>> {
    ids.iter()
        .map(|id| {
            candidates
                .iter()
                .find(|candidate| &candidate.id == id)
                .map(|candidate| candidate.summary.clone())
                .ok_or_else(|| {
                    ScalingError::InvalidRecord(
                        "checkpoint candidate has no matching summary".to_string(),
                    )
                })
        })
        .collect()
}

fn validate_task(task: &Task) -> Result<()> {
    if task.id.trim().is_empty()
        || task.benchmark.trim().is_empty()
        || task.prompt.trim().is_empty()
    {
        return Err(ScalingError::InvalidConfig(
            "task id, benchmark, and prompt must not be empty".to_string(),
        ));
    }
    Ok(())
}

fn validate_attempt<O>(
    request: &RolloutRequest,
    rollout: &Rollout<O>,
    summary: &Summary,
    expected_model_id: &str,
) -> Result<()> {
    if rollout.id.trim().is_empty()
        || rollout.environment_id.trim().is_empty()
        || rollout.output_digest.trim().is_empty()
    {
        return Err(ScalingError::InvalidRecord(
            "rollout, environment, and output identifiers must not be empty".to_string(),
        ));
    }
    if rollout.model_id != expected_model_id || summary.model_id != expected_model_id {
        return Err(ScalingError::InvalidRecord(
            "rollout and summary model IDs must match the experiment model".to_string(),
        ));
    }
    if rollout.iteration != request.iteration || rollout.rollout_index != request.rollout_index {
        return Err(ScalingError::InvalidRecord(
            "rollout iteration and index must match its request".to_string(),
        ));
    }
    if summary.id.trim().is_empty()
        || summary.rollout_id != rollout.id
        || summary.raw_response.trim().is_empty()
        || summary.generation_attempts == 0
    {
        return Err(ScalingError::InvalidRecord(
            "summary must identify its rollout and record generation".to_string(),
        ));
    }
    Ok(())
}

fn validate_controller_schedule(config: &ScalingConfig) -> Result<()> {
    let mut population = config.rollout_count;
    let mut reaches_refinement_count = population == config.refinement_count;
    while population > 1 {
        let group_size = config.group_size.min(population);
        if !population.is_multiple_of(group_size) {
            return Err(ScalingError::InvalidConfig(format!(
                "rollout_count {population} is not divisible by group size {group_size}"
            )));
        }
        population /= group_size;
        reaches_refinement_count |= population == config.refinement_count;
    }
    if !reaches_refinement_count {
        return Err(ScalingError::InvalidConfig(
            "refinement_count is not reached by the tournament schedule".to_string(),
        ));
    }
    Ok(())
}

fn validate_attempt_records<O>(zero: &[Attempt<O>], one: &[Attempt<O>]) -> Result<()> {
    let mut rollout_ids = HashSet::with_capacity(zero.len() + one.len());
    let mut summary_ids = HashSet::with_capacity(zero.len() + one.len());
    let mut environment_ids = HashSet::with_capacity(zero.len() + one.len());
    for attempt in zero.iter().chain(one) {
        if !rollout_ids.insert(&attempt.rollout.id) {
            return Err(ScalingError::InvalidRecord(
                "rollout identifiers must be unique".to_string(),
            ));
        }
        if !summary_ids.insert(&attempt.summary.id) {
            return Err(ScalingError::InvalidRecord(
                "summary identifiers must be unique".to_string(),
            ));
        }
        if !environment_ids.insert(&attempt.rollout.environment_id) {
            return Err(ScalingError::InvalidRecord(
                "every rollout must use a fresh environment identifier".to_string(),
            ));
        }
    }
    Ok(())
}
