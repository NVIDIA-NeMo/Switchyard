// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Post-run grading records and paper stage metrics.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::{Result, ScalingError, ScalingRun};

/// Official result for one rollout, produced only after selection finishes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RolloutEvaluation {
    /// Rollout that was graded.
    pub rollout_id: String,
    /// Whether the official grader accepted the rollout.
    pub passed: bool,
}

/// Stage scores for one completed task.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskEvaluation {
    /// Task that was graded.
    pub task_id: String,
    /// Iteration-zero outcomes in rollout order.
    pub iteration_zero: Vec<bool>,
    /// Outcomes for the four summaries selected for refinement.
    pub selected_for_refinement: Vec<bool>,
    /// Iteration-one outcomes in rollout order.
    pub iteration_one: Vec<bool>,
    /// Outcome of the rollout selected by the final tournament.
    pub final_passed: bool,
}

/// Aggregate values reported for one stage over several tasks.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageMetrics {
    /// Average outcome across all rollouts in this stage.
    pub average_pass_at_one: f64,
    /// Fraction of tasks with at least one passing rollout.
    pub pass_at_n: f64,
    /// Tasks containing both passing and failing rollouts.
    pub mixed_task_count: usize,
}

/// Paper stage metrics over a set of completed tasks.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentMetrics {
    /// Number of evaluated tasks.
    pub task_count: usize,
    /// Metrics for the first 16 rollouts.
    pub iteration_zero: StageMetrics,
    /// Metrics for the summaries selected for refinement.
    pub selected_for_refinement: StageMetrics,
    /// Metrics for the second 16 rollouts.
    pub iteration_one: StageMetrics,
    /// Fraction of tasks whose final selected rollout passed.
    pub final_pass_at_one: f64,
}

/// Joins official outcomes to a completed run without exposing them to the controller.
pub fn evaluate_run<O>(
    run: &ScalingRun<O>,
    evaluations: Vec<RolloutEvaluation>,
) -> Result<TaskEvaluation> {
    let expected_ids: HashSet<&str> = run
        .iteration_zero
        .iter()
        .chain(&run.iteration_one)
        .map(|attempt| attempt.rollout.id.as_str())
        .collect();
    let mut outcomes = HashMap::with_capacity(evaluations.len());
    for evaluation in evaluations {
        if !expected_ids.contains(evaluation.rollout_id.as_str()) {
            return Err(ScalingError::InvalidEvaluation(format!(
                "unknown rollout {}",
                evaluation.rollout_id
            )));
        }
        if outcomes
            .insert(evaluation.rollout_id.clone(), evaluation.passed)
            .is_some()
        {
            return Err(ScalingError::InvalidEvaluation(format!(
                "duplicate rollout {}",
                evaluation.rollout_id
            )));
        }
    }
    if outcomes.len() != expected_ids.len() {
        return Err(ScalingError::InvalidEvaluation(format!(
            "received {} outcomes; expected {}",
            outcomes.len(),
            expected_ids.len()
        )));
    }

    let lookup = |rollout_id: &str| {
        outcomes
            .get(rollout_id)
            .copied()
            .ok_or_else(|| ScalingError::InvalidEvaluation(format!("missing rollout {rollout_id}")))
    };
    let iteration_zero = run
        .iteration_zero
        .iter()
        .map(|attempt| lookup(&attempt.rollout.id))
        .collect::<Result<Vec<_>>>()?;
    let checkpoint = run
        .iteration_zero_tournament
        .checkpoint
        .as_ref()
        .ok_or_else(|| {
            ScalingError::InvalidEvaluation(
                "iteration-zero tournament has no refinement checkpoint".to_string(),
            )
        })?;
    let selected_for_refinement = checkpoint
        .candidate_ids
        .iter()
        .map(|id| lookup(id))
        .collect::<Result<Vec<_>>>()?;
    let iteration_one = run
        .iteration_one
        .iter()
        .map(|attempt| lookup(&attempt.rollout.id))
        .collect::<Result<Vec<_>>>()?;
    let final_passed = lookup(&run.final_rollout.id)?;

    Ok(TaskEvaluation {
        task_id: run.task.id.clone(),
        iteration_zero,
        selected_for_refinement,
        iteration_one,
        final_passed,
    })
}

/// Computes the paper's main stage metrics from completed task evaluations.
pub fn experiment_metrics(tasks: &[TaskEvaluation]) -> Result<ExperimentMetrics> {
    if tasks.is_empty() {
        return Err(ScalingError::InvalidEvaluation(
            "at least one task evaluation is required".to_string(),
        ));
    }
    let mut task_ids = HashSet::with_capacity(tasks.len());
    for task in tasks {
        if task.task_id.trim().is_empty() || !task_ids.insert(&task.task_id) {
            return Err(ScalingError::InvalidEvaluation(
                "task evaluation identifiers must be non-empty and unique".to_string(),
            ));
        }
        if task.iteration_zero.is_empty()
            || task.selected_for_refinement.is_empty()
            || task.iteration_one.is_empty()
        {
            return Err(ScalingError::InvalidEvaluation(
                "every evaluated stage must contain at least one rollout".to_string(),
            ));
        }
    }

    Ok(ExperimentMetrics {
        task_count: tasks.len(),
        iteration_zero: stage_metrics(tasks, |task| &task.iteration_zero),
        selected_for_refinement: stage_metrics(tasks, |task| &task.selected_for_refinement),
        iteration_one: stage_metrics(tasks, |task| &task.iteration_one),
        final_pass_at_one: tasks.iter().filter(|task| task.final_passed).count() as f64
            / tasks.len() as f64,
    })
}

fn stage_metrics<'a>(
    tasks: &'a [TaskEvaluation],
    outcomes: impl Fn(&'a TaskEvaluation) -> &'a [bool],
) -> StageMetrics {
    let total = tasks.iter().map(|task| outcomes(task).len()).sum::<usize>();
    let passed = tasks
        .iter()
        .map(|task| outcomes(task).iter().filter(|value| **value).count())
        .sum::<usize>();
    let tasks_with_pass = tasks
        .iter()
        .filter(|task| outcomes(task).iter().any(|value| *value))
        .count();
    let mixed_task_count = tasks
        .iter()
        .filter(|task| {
            let values = outcomes(task);
            values.iter().any(|value| *value) && values.iter().any(|value| !*value)
        })
        .count();

    StageMetrics {
        average_pass_at_one: passed as f64 / total as f64,
        pass_at_n: tasks_with_pass as f64 / tasks.len() as f64,
        mixed_task_count,
    }
}
