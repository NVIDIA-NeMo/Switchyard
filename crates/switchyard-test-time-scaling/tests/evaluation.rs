// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use switchyard_test_time_scaling::{Result, TaskEvaluation, experiment_metrics};

#[test]
fn stage_metrics_use_task_and_rollout_denominators() -> Result<()> {
    let metrics = experiment_metrics(&[
        TaskEvaluation {
            task_id: "task-1".to_string(),
            iteration_zero: vec![true, false],
            selected_for_refinement: vec![true],
            iteration_one: vec![true, true],
            final_passed: true,
        },
        TaskEvaluation {
            task_id: "task-2".to_string(),
            iteration_zero: vec![false, false],
            selected_for_refinement: vec![false],
            iteration_one: vec![false, true],
            final_passed: false,
        },
    ])?;

    assert_eq!(metrics.task_count, 2);
    assert_eq!(metrics.iteration_zero.average_pass_at_one, 0.25);
    assert_eq!(metrics.iteration_zero.pass_at_n, 0.5);
    assert_eq!(metrics.iteration_zero.mixed_task_count, 1);
    assert_eq!(metrics.selected_for_refinement.average_pass_at_one, 0.5);
    assert_eq!(metrics.iteration_one.average_pass_at_one, 0.75);
    assert_eq!(metrics.iteration_one.pass_at_n, 1.0);
    assert_eq!(metrics.iteration_one.mixed_task_count, 1);
    assert_eq!(metrics.final_pass_at_one, 0.5);
    Ok(())
}

#[test]
fn metrics_reject_empty_or_duplicate_tasks() {
    assert!(experiment_metrics(&[]).is_err());

    let task = TaskEvaluation {
        task_id: "same".to_string(),
        iteration_zero: vec![false],
        selected_for_refinement: vec![false],
        iteration_one: vec![false],
        final_passed: false,
    };
    assert!(experiment_metrics(&[task.clone(), task]).is_err());
}
