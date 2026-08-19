// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Records produced by one scaling run.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// One agentic coding task.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Task {
    /// Stable task identifier.
    pub id: String,
    /// Benchmark or task family name.
    pub benchmark: String,
    /// Original task text shown to every attempt.
    pub prompt: String,
}

/// Request for one independent agent attempt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RolloutRequest {
    /// Zero for initial attempts and one for refined attempts.
    pub iteration: u8,
    /// Zero-based attempt position within the iteration.
    pub rollout_index: usize,
    /// Seed for this attempt.
    pub seed: u64,
    /// Prior summaries supplied to a refined attempt.
    pub refinement_summaries: Vec<Summary>,
    /// Rendered prior-attempt context, absent for initial attempts.
    pub refinement_prompt: Option<String>,
}

/// Result returned by an agent harness.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rollout<O> {
    /// Stable attempt identifier.
    pub id: String,
    /// Zero for an initial attempt and one for a refined attempt.
    pub iteration: u8,
    /// Zero-based attempt position within the iteration.
    pub rollout_index: usize,
    /// Exact model used for this attempt.
    pub model_id: String,
    /// Identifier of the fresh environment used for this attempt.
    pub environment_id: String,
    /// Content digest of the returned patch, artifacts, or environment snapshot.
    pub output_digest: String,
    /// Patch, artifact set, snapshot, or other harness-owned result.
    pub output: O,
}

/// Structured summary of one attempt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Summary {
    /// Stable summary identifier.
    pub id: String,
    /// Attempt summarized by this record.
    pub rollout_id: String,
    /// Exact model used to generate this summary.
    pub model_id: String,
    /// JSON object presented to refinement and comparison calls.
    pub value: Map<String, Value>,
    /// Lossless model response saved before parsing.
    pub raw_response: String,
    /// Number of content-generation attempts used to produce the object.
    pub generation_attempts: usize,
}

/// One completed attempt together with its request and summary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Attempt<O> {
    /// Request used to start the attempt.
    pub request: RolloutRequest,
    /// Agent harness result.
    pub rollout: Rollout<O>,
    /// Structured summary of the result.
    pub summary: Summary,
}

/// Summary-backed candidate used by a tournament.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    /// Stable candidate identifier.
    pub id: String,
    /// Attempt represented by the candidate.
    pub rollout_id: String,
    /// Structured summary shown to judges.
    pub summary: Summary,
}

impl Candidate {
    /// Builds a candidate from a completed attempt.
    pub fn from_attempt<O>(attempt: &Attempt<O>) -> Self {
        Self {
            id: attempt.rollout.id.clone(),
            rollout_id: attempt.rollout.id.clone(),
            summary: attempt.summary.clone(),
        }
    }
}

/// One comparison call sent to the backend.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComparisonRequest {
    /// Fully rendered comparison prompt.
    pub prompt: String,
    /// Candidates in the exact order used by the prompt.
    pub candidates: Vec<Candidate>,
    /// Zero-based tournament round.
    pub round_index: usize,
    /// Zero-based group within the round.
    pub group_index: usize,
    /// Stable vote record position, including replacements.
    pub vote_index: usize,
    /// Seed for this judge call.
    pub seed: u64,
}

/// Successful response from one comparison model call.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComparisonResponse {
    /// Exact model used for this comparison.
    pub model_id: String,
    /// Lossless response returned by the model.
    pub content: String,
}

/// One recorded judge call and its parsed choice.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Vote {
    /// Stable vote record position, including replacements.
    pub vote_index: usize,
    /// Candidate identifiers in displayed order.
    pub ordered_candidate_ids: Vec<String>,
    /// Fully rendered prompt sent to the judge.
    pub prompt: String,
    /// Seed used for the judge call.
    pub seed: u64,
    /// Exact model used when the call succeeded.
    pub model_id: Option<String>,
    /// Raw judge response when the model call succeeded.
    pub raw_response: Option<String>,
    /// Selected one-based displayed position when parsing succeeded.
    pub selected_position: Option<usize>,
    /// Stable candidate identifier selected by the vote.
    pub selected_candidate_id: Option<String>,
    /// Short failure reason for an invalid vote.
    pub error: Option<String>,
}

/// Result of one tournament comparison group.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupDecision {
    /// Zero-based round position.
    pub round_index: usize,
    /// Zero-based group position.
    pub group_index: usize,
    /// Candidate identifiers before display reordering.
    pub input_candidate_ids: Vec<String>,
    /// Initial and replacement vote records.
    pub votes: Vec<Vote>,
    /// Valid vote totals by candidate identifier.
    pub vote_counts: BTreeMap<String, usize>,
    /// Candidate that advances to the next round.
    pub selected_candidate_id: String,
    /// Recorded tie policy when the highest count was shared.
    pub tie_break: Option<String>,
}

/// Immutable survivor list captured during a tournament.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TournamentCheckpoint {
    /// Number of completed rounds.
    pub completed_rounds: usize,
    /// Candidate identifiers frozen at this point.
    pub candidate_ids: Vec<String>,
}

/// Complete record of one recursive tournament.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tournament {
    /// Settings used for this tournament.
    pub config: crate::ScalingConfig,
    /// Root seed used for pairing, display order, votes, and ties.
    pub root_seed: u64,
    /// Candidate identifiers at the start and after every round.
    pub populations: Vec<Vec<String>>,
    /// Group decisions in round order.
    pub rounds: Vec<Vec<GroupDecision>>,
    /// Requested final survivor count.
    pub target_survivors: usize,
    /// Optional survivor list frozen before the tournament finished.
    pub checkpoint: Option<TournamentCheckpoint>,
    /// Final surviving candidate identifiers.
    pub survivor_candidate_ids: Vec<String>,
}

/// Result of the complete two-iteration scaling workflow.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalingRun<O> {
    /// Experiment settings needed to interpret and repeat this run.
    pub manifest: crate::ExperimentManifest,
    /// Task processed by the controller.
    pub task: Task,
    /// Model identity reported by the shared backend.
    pub model_id: String,
    /// Settings used for the complete run.
    pub config: crate::ScalingConfig,
    /// Initial independent attempts.
    pub iteration_zero: Vec<Attempt<O>>,
    /// Tournament that selected refinement context and continued to one diagnostic survivor.
    pub iteration_zero_tournament: Tournament,
    /// Summary identifiers given to every refined attempt.
    pub refinement_summary_ids: Vec<String>,
    /// Fresh attempts conditioned on the selected summaries.
    pub iteration_one: Vec<Attempt<O>>,
    /// Tournament that selected the final result.
    pub final_tournament: Tournament,
    /// Exact rollout selected by the final tournament.
    pub final_rollout: Rollout<O>,
}
