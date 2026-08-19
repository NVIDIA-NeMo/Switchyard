// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::{Map, json};
use switchyard_test_time_scaling::{
    Candidate, ComparisonRequest, ComparisonResponse, DisplayOrder, ExperimentManifest,
    InvalidVotePolicy, MANIFEST_SCHEMA_VERSION, ManifestSource, ManifestValue,
    REQUIRED_MANIFEST_FIELDS, Result, Rollout, RolloutEvaluation, RolloutRequest, ScalingBackend,
    ScalingConfig, ScalingController, ScalingError, ScalingRun, Summary, Task, TiePolicy,
    decode_run, encode_run, evaluate_run, run_tournament,
};

#[derive(Clone, Copy)]
enum JudgeScript {
    First,
    Tie,
    FirstVoteInvalid,
    RolloutZero,
    WrongSummaryModel,
    WrongComparisonModel,
}

struct FakeBackend {
    judge_script: JudgeScript,
    rollout_requests: Arc<Mutex<Vec<RolloutRequest>>>,
    comparison_calls: Arc<AtomicUsize>,
}

impl FakeBackend {
    fn new(judge_script: JudgeScript) -> Self {
        Self {
            judge_script,
            rollout_requests: Arc::new(Mutex::new(Vec::new())),
            comparison_calls: Arc::new(AtomicUsize::new(0)),
        }
    }
}

#[async_trait]
impl ScalingBackend for FakeBackend {
    type Output = String;

    fn model_id(&self) -> &str {
        "test-model-v1"
    }

    async fn run_rollouts(
        &self,
        _task: &Task,
        requests: Vec<RolloutRequest>,
    ) -> Result<Vec<Rollout<Self::Output>>> {
        self.rollout_requests
            .lock()
            .expect("request log lock")
            .extend(requests.iter().cloned());
        Ok(requests
            .into_iter()
            .map(|request| {
                let suffix = format!("{}-{}", request.iteration, request.rollout_index);
                Rollout {
                    id: format!("rollout-{suffix}"),
                    iteration: request.iteration,
                    rollout_index: request.rollout_index,
                    model_id: self.model_id().to_string(),
                    environment_id: format!("environment-{suffix}"),
                    output_digest: format!("digest-{suffix}"),
                    output: format!("output-{suffix}"),
                }
            })
            .collect())
    }

    async fn summarize(&self, _task: &Task, rollout: &Rollout<Self::Output>) -> Result<Summary> {
        let value = json!({"evidence": rollout.id});
        Ok(Summary {
            id: format!("summary-{}", rollout.id),
            rollout_id: rollout.id.clone(),
            model_id: if matches!(self.judge_script, JudgeScript::WrongSummaryModel) {
                "different-model".to_string()
            } else {
                self.model_id().to_string()
            },
            value: value
                .as_object()
                .cloned()
                .expect("summary fixture is an object"),
            raw_response: value.to_string(),
            generation_attempts: 1,
        })
    }

    async fn compare(
        &self,
        _task: &Task,
        request: ComparisonRequest,
    ) -> Result<ComparisonResponse> {
        self.comparison_calls.fetch_add(1, Ordering::Relaxed);
        let content = match self.judge_script {
            JudgeScript::First => "Evidence.\nFinal verdict: Solution 1".to_string(),
            JudgeScript::Tie => {
                let position = if request.vote_index < 4 { 1 } else { 2 };
                format!("Final verdict: Solution {position}")
            }
            JudgeScript::FirstVoteInvalid if request.vote_index == 0 => "no verdict".to_string(),
            JudgeScript::FirstVoteInvalid => "Final verdict: Solution 1".to_string(),
            JudgeScript::RolloutZero => {
                let position = request
                    .candidates
                    .iter()
                    .position(|candidate| candidate.id == "rollout-0")
                    .map(|index| index + 1)
                    .unwrap_or(1);
                format!("Final verdict: Solution {position}")
            }
            JudgeScript::WrongSummaryModel | JudgeScript::WrongComparisonModel => {
                "Final verdict: Solution 1".to_string()
            }
        };
        Ok(ComparisonResponse {
            model_id: if matches!(self.judge_script, JudgeScript::WrongComparisonModel) {
                "different-model".to_string()
            } else {
                self.model_id().to_string()
            },
            content,
        })
    }
}

fn task() -> Task {
    Task {
        id: "task-1".to_string(),
        benchmark: "test-benchmark".to_string(),
        prompt: "fix the test task".to_string(),
    }
}

fn manifest() -> ExperimentManifest {
    ExperimentManifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        replication_mode: switchyard_test_time_scaling::ReplicationMode::Conceptual,
        code_revision: "test-revision".to_string(),
        model_id: "test-model-v1".to_string(),
        fields: BTreeMap::from_iter(REQUIRED_MANIFEST_FIELDS.map(|name| {
            (
                name.to_string(),
                ManifestValue {
                    source: ManifestSource::Reconstructed,
                    value: "test choice".to_string(),
                },
            )
        })),
    }
}

fn candidates(count: usize) -> Vec<Candidate> {
    (0..count)
        .map(|index| {
            let rollout_id = format!("rollout-{index}");
            Candidate {
                id: rollout_id.clone(),
                rollout_id: rollout_id.clone(),
                summary: Summary {
                    id: format!("summary-{index}"),
                    rollout_id,
                    model_id: "test-model-v1".to_string(),
                    value: Map::from_iter([("index".to_string(), json!(index))]),
                    raw_response: format!(r#"{{"index":{index}}}"#),
                    generation_attempts: 1,
                },
            }
        })
        .collect()
}

#[tokio::test]
async fn main_workflow_keeps_attempts_isolated_and_returns_the_selected_output() -> Result<()> {
    let backend = FakeBackend::new(JudgeScript::First);
    let request_log = Arc::clone(&backend.rollout_requests);
    let comparison_calls = Arc::clone(&backend.comparison_calls);
    let controller = ScalingController::new(backend, ScalingConfig::default(), manifest())?;

    let run = controller.run(task()).await?;

    let population_sizes =
        |populations: &[Vec<String>]| populations.iter().map(Vec::len).collect::<Vec<_>>();
    assert_eq!(
        population_sizes(&run.iteration_zero_tournament.populations),
        vec![16, 8, 4, 2, 1]
    );
    assert_eq!(
        population_sizes(&run.final_tournament.populations),
        vec![16, 8, 4, 2, 1]
    );
    let checkpoint = run
        .iteration_zero_tournament
        .checkpoint
        .as_ref()
        .expect("refinement checkpoint");
    assert_eq!(checkpoint.completed_rounds, 2);
    assert_eq!(checkpoint.candidate_ids.len(), 4);
    assert_eq!(
        checkpoint.candidate_ids,
        run.iteration_zero_tournament.populations[2]
    );
    assert_eq!(run.refinement_summary_ids.len(), 4);

    for attempt in &run.iteration_one {
        let refinement_prompt = attempt
            .request
            .refinement_prompt
            .as_deref()
            .expect("refined attempt has rendered context");
        assert!(refinement_prompt.contains("PRIOR ATTEMPT SUMMARY 4"));
        let summary_ids: Vec<&str> = attempt
            .request
            .refinement_summaries
            .iter()
            .map(|summary| summary.id.as_str())
            .collect();
        assert_eq!(
            summary_ids,
            run.refinement_summary_ids
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            attempt.request.refinement_prompt,
            run.iteration_one[0].request.refinement_prompt
        );
    }

    let environments: HashSet<&str> = run
        .iteration_zero
        .iter()
        .chain(&run.iteration_one)
        .map(|attempt| attempt.rollout.environment_id.as_str())
        .collect();
    assert_eq!(environments.len(), 32);
    assert_eq!(comparison_calls.load(Ordering::Relaxed), 240);
    assert_eq!(
        run.iteration_zero_tournament
            .rounds
            .iter()
            .flatten()
            .count(),
        15
    );
    assert_eq!(
        run.iteration_zero_tournament
            .rounds
            .iter()
            .flatten()
            .flat_map(|decision| &decision.votes)
            .count(),
        120
    );
    assert_eq!(
        run.final_rollout.id,
        run.final_tournament.survivor_candidate_ids[0]
    );
    assert_eq!(run.final_rollout.output, "output-1-0");
    assert_eq!(run.final_rollout.output_digest, "digest-1-0");
    assert_eq!(run.model_id, "test-model-v1");
    let encoded = encode_run(&run)?;
    let replayed: ScalingRun<String> = decode_run(&encoded)?;
    assert_eq!(replayed, run);

    let evaluations = run
        .iteration_zero
        .iter()
        .chain(&run.iteration_one)
        .map(|attempt| RolloutEvaluation {
            rollout_id: attempt.rollout.id.clone(),
            passed: attempt.rollout.id.ends_with("-0"),
        })
        .collect();
    let evaluated = evaluate_run(&run, evaluations)?;
    assert_eq!(evaluated.iteration_zero.len(), 16);
    assert_eq!(evaluated.selected_for_refinement.len(), 4);
    assert_eq!(evaluated.iteration_one.len(), 16);
    assert!(evaluated.final_passed);

    let requests = request_log.lock().expect("request log lock");
    assert_eq!(requests.len(), 32);
    assert_eq!(
        requests
            .iter()
            .filter(|request| request.iteration == 0)
            .count(),
        16
    );
    assert!(
        requests
            .iter()
            .filter(|request| request.iteration == 0)
            .all(|request| request.refinement_summaries.is_empty()
                && request.refinement_prompt.is_none())
    );
    Ok(())
}

#[tokio::test]
async fn group_size_schedules_match_the_paper() -> Result<()> {
    let cases = [
        (16, vec![16]),
        (8, vec![8, 2]),
        (4, vec![4, 4]),
        (2, vec![2, 2, 2, 2]),
    ];
    for (group_size, expected) in cases {
        let backend = FakeBackend::new(JudgeScript::First);
        let config = ScalingConfig {
            group_size,
            votes_per_group: 1,
            ..ScalingConfig::default()
        };
        let tournament =
            run_tournament(&backend, &task(), candidates(16), &config, 1, None, 42).await?;
        let actual: Vec<usize> = tournament
            .rounds
            .iter()
            .map(|round| round[0].input_candidate_ids.len())
            .collect();
        assert_eq!(actual, expected);
    }
    Ok(())
}

#[tokio::test]
async fn ties_and_replacement_votes_are_recorded() -> Result<()> {
    let tie_backend = FakeBackend::new(JudgeScript::Tie);
    let tie_config = ScalingConfig {
        rollout_count: 2,
        refinement_count: 1,
        votes_per_group: 8,
        tie_policy: TiePolicy::FirstInGroup,
        ..ScalingConfig::default()
    };
    let tied = run_tournament(
        &tie_backend,
        &task(),
        candidates(2),
        &tie_config,
        1,
        None,
        7,
    )
    .await?;
    let decision = &tied.rounds[0][0];
    assert_eq!(
        decision.vote_counts.values().copied().collect::<Vec<_>>(),
        [4, 4]
    );
    assert_eq!(decision.tie_break.as_deref(), Some("first_in_group"));

    let replacement_backend = FakeBackend::new(JudgeScript::FirstVoteInvalid);
    let replacement_config = ScalingConfig {
        rollout_count: 2,
        refinement_count: 1,
        votes_per_group: 2,
        invalid_vote_policy: InvalidVotePolicy::Replace {
            max_calls_per_group: 1,
        },
        ..ScalingConfig::default()
    };
    let replaced = run_tournament(
        &replacement_backend,
        &task(),
        candidates(2),
        &replacement_config,
        1,
        None,
        7,
    )
    .await?;
    assert_eq!(replaced.rounds[0][0].votes.len(), 3);
    assert_eq!(replaced.rounds[0][0].votes[0].selected_candidate_id, None);
    assert!(
        replaced.rounds[0][0].votes[2]
            .selected_candidate_id
            .is_some()
    );
    Ok(())
}

#[tokio::test]
async fn abort_policy_rejects_an_invalid_vote() {
    let backend = FakeBackend::new(JudgeScript::FirstVoteInvalid);
    let config = ScalingConfig {
        rollout_count: 2,
        refinement_count: 1,
        votes_per_group: 2,
        invalid_vote_policy: InvalidVotePolicy::Abort,
        ..ScalingConfig::default()
    };
    let result = run_tournament(&backend, &task(), candidates(2), &config, 1, None, 7).await;
    assert_eq!(
        result,
        Err(ScalingError::IncompleteVotes {
            expected: 2,
            actual: 1,
        })
    );
}

#[tokio::test]
async fn display_permutations_map_back_to_the_stable_candidate() -> Result<()> {
    let backend = FakeBackend::new(JudgeScript::RolloutZero);
    let config = ScalingConfig {
        rollout_count: 2,
        refinement_count: 1,
        votes_per_group: 8,
        display_order: DisplayOrder::Shuffle,
        ..ScalingConfig::default()
    };
    let tournament = run_tournament(&backend, &task(), candidates(2), &config, 1, None, 9).await?;
    let decision = &tournament.rounds[0][0];

    assert_eq!(decision.selected_candidate_id, "rollout-0");
    assert!(
        decision
            .votes
            .iter()
            .all(|vote| vote.selected_candidate_id.as_deref() == Some("rollout-0"))
    );
    assert!(
        decision
            .votes
            .iter()
            .any(|vote| vote.ordered_candidate_ids[0] != "rollout-0")
    );
    Ok(())
}

#[tokio::test]
async fn model_drift_is_rejected() -> Result<()> {
    let mut wrong_manifest = manifest();
    wrong_manifest.model_id = "different-model".to_string();
    assert!(
        ScalingController::new(
            FakeBackend::new(JudgeScript::First),
            ScalingConfig::default(),
            wrong_manifest,
        )
        .is_err()
    );

    let small_config = ScalingConfig {
        rollout_count: 2,
        refinement_count: 1,
        votes_per_group: 1,
        ..ScalingConfig::default()
    };
    let controller = ScalingController::new(
        FakeBackend::new(JudgeScript::WrongSummaryModel),
        small_config.clone(),
        manifest(),
    )?;
    assert!(matches!(
        controller.run(task()).await,
        Err(ScalingError::InvalidRecord(_))
    ));

    let result = run_tournament(
        &FakeBackend::new(JudgeScript::WrongComparisonModel),
        &task(),
        candidates(2),
        &small_config,
        1,
        None,
        1,
    )
    .await;
    assert!(matches!(result, Err(ScalingError::InvalidRecord(_))));
    Ok(())
}
