// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Recursive tournament selection.

use std::collections::{BTreeMap, HashSet};

use futures::future::try_join_all;

use crate::config::{DisplayOrder, InvalidVotePolicy, PairingOrder, TiePolicy};
use crate::prompts::comparison_prompt;
use crate::seed;
use crate::{
    Candidate, ComparisonRequest, GroupDecision, Result, ScalingBackend, ScalingConfig,
    ScalingError, Task, Tournament, TournamentCheckpoint, Vote, parse_verdict,
};

#[derive(Clone, Copy)]
struct VotePosition {
    round: usize,
    group: usize,
    display: usize,
    vote: usize,
}

/// Runs recursive tournament voting until `target_survivors` candidates remain.
///
/// When `checkpoint_at` is set, the exact survivor population is copied when it first reaches
/// that size. The tournament then continues to its final target.
pub async fn run_tournament<B>(
    backend: &B,
    task: &Task,
    candidates: Vec<Candidate>,
    config: &ScalingConfig,
    target_survivors: usize,
    checkpoint_at: Option<usize>,
    root_seed: u64,
) -> Result<Tournament>
where
    B: ScalingBackend,
{
    config.validate()?;
    validate_candidates(&candidates)?;
    if target_survivors == 0 || target_survivors > candidates.len() {
        return Err(ScalingError::InvalidConfig(
            "target_survivors must be between 1 and the candidate count".to_string(),
        ));
    }
    if checkpoint_at.is_some_and(|count| count < target_survivors || count > candidates.len()) {
        return Err(ScalingError::InvalidConfig(
            "checkpoint size must be between target_survivors and the candidate count".to_string(),
        ));
    }

    let mut population = candidates;
    let mut populations = vec![candidate_ids(&population)];
    let mut rounds = Vec::new();
    let mut checkpoint = checkpoint_at
        .filter(|count| *count == population.len())
        .map(|_| TournamentCheckpoint {
            completed_rounds: 0,
            candidate_ids: candidate_ids(&population),
        });

    while population.len() > target_survivors {
        let round_index = rounds.len();
        let group_size = config.group_size.min(population.len());
        if !population.len().is_multiple_of(group_size) {
            return Err(ScalingError::UnevenGroups {
                population: population.len(),
                group_size,
            });
        }
        if population.len() / group_size < target_survivors {
            return Err(ScalingError::OvershootsTarget {
                population: population.len(),
                group_size,
                target: target_survivors,
            });
        }

        let mut ordered = population;
        if config.pairing_order == PairingOrder::Shuffle {
            seed::shuffle(
                &mut ordered,
                seed::derive(root_seed, &[1, round_index as u64]),
            );
        }
        let groups: Vec<Vec<Candidate>> = ordered
            .chunks(group_size)
            .map(<[Candidate]>::to_vec)
            .collect();
        let results = try_join_all(groups.into_iter().enumerate().map(|(group_index, group)| {
            decide_group(
                backend,
                task,
                group,
                config,
                round_index,
                group_index,
                root_seed,
            )
        }))
        .await?;

        let mut decisions = Vec::with_capacity(results.len());
        let mut next_population = Vec::with_capacity(results.len());
        for (decision, survivor) in results {
            decisions.push(decision);
            next_population.push(survivor);
        }
        rounds.push(decisions);
        population = next_population;
        populations.push(candidate_ids(&population));

        if checkpoint.is_none() && checkpoint_at == Some(population.len()) {
            checkpoint = Some(TournamentCheckpoint {
                completed_rounds: rounds.len(),
                candidate_ids: candidate_ids(&population),
            });
        }
    }

    if checkpoint_at.is_some() && checkpoint.is_none() {
        return Err(ScalingError::InvalidConfig(
            "checkpoint size is not reached by this tournament schedule".to_string(),
        ));
    }

    Ok(Tournament {
        config: config.clone(),
        root_seed,
        populations,
        rounds,
        target_survivors,
        checkpoint,
        survivor_candidate_ids: candidate_ids(&population),
    })
}

async fn decide_group<B>(
    backend: &B,
    task: &Task,
    group: Vec<Candidate>,
    config: &ScalingConfig,
    round_index: usize,
    group_index: usize,
    root_seed: u64,
) -> Result<(GroupDecision, Candidate)>
where
    B: ScalingBackend,
{
    let initial = (0..config.votes_per_group).map(|vote_index| {
        collect_vote(
            backend,
            task,
            &group,
            config,
            VotePosition {
                round: round_index,
                group: group_index,
                display: vote_index,
                vote: vote_index,
            },
            root_seed,
        )
    });
    let mut votes = try_join_all(initial).await?;
    let invalid_slots: Vec<usize> = votes
        .iter()
        .enumerate()
        .filter_map(|(index, vote)| vote.selected_candidate_id.is_none().then_some(index))
        .collect();
    let mut valid_count = votes.len() - invalid_slots.len();

    if valid_count != config.votes_per_group {
        match config.invalid_vote_policy {
            InvalidVotePolicy::Abort => {
                return Err(ScalingError::IncompleteVotes {
                    expected: config.votes_per_group,
                    actual: valid_count,
                });
            }
            InvalidVotePolicy::Replace {
                max_calls_per_group,
            } => {
                for replacement_index in 0..max_calls_per_group {
                    if valid_count == config.votes_per_group {
                        break;
                    }
                    let Some(source_vote_index) = invalid_slots
                        .get(replacement_index % invalid_slots.len())
                        .copied()
                    else {
                        break;
                    };
                    let vote_index = config.votes_per_group + replacement_index;
                    let vote = collect_vote(
                        backend,
                        task,
                        &group,
                        config,
                        VotePosition {
                            round: round_index,
                            group: group_index,
                            display: source_vote_index,
                            vote: vote_index,
                        },
                        root_seed,
                    )
                    .await?;
                    if vote.selected_candidate_id.is_some() {
                        valid_count += 1;
                    }
                    votes.push(vote);
                }
            }
        }
    }
    if valid_count != config.votes_per_group {
        return Err(ScalingError::IncompleteVotes {
            expected: config.votes_per_group,
            actual: valid_count,
        });
    }

    let mut counts = BTreeMap::new();
    for candidate_id in votes
        .iter()
        .filter_map(|vote| vote.selected_candidate_id.as_ref())
    {
        *counts.entry(candidate_id.clone()).or_insert(0) += 1;
    }
    let Some(highest_count) = counts.values().copied().max() else {
        return Err(ScalingError::InvalidRecord(
            "a completed group has no vote counts".to_string(),
        ));
    };
    let winners: Vec<&Candidate> = group
        .iter()
        .filter(|candidate| counts.get(&candidate.id) == Some(&highest_count))
        .collect();
    let Some(first_winner) = winners.first().copied() else {
        return Err(ScalingError::InvalidRecord(
            "vote counts do not match any group candidate".to_string(),
        ));
    };
    let (selected_id, tie_break) = if winners.len() == 1 {
        (first_winner.id.clone(), None)
    } else {
        let selected = match config.tie_policy {
            TiePolicy::FirstInGroup => first_winner,
            TiePolicy::SeededRandom => {
                let index = seed::derive(root_seed, &[5, round_index as u64, group_index as u64])
                    as usize
                    % winners.len();
                winners.get(index).copied().ok_or_else(|| {
                    ScalingError::InvalidRecord("tie selection is out of range".to_string())
                })?
            }
        };
        let policy = match config.tie_policy {
            TiePolicy::FirstInGroup => "first_in_group",
            TiePolicy::SeededRandom => "seeded_random",
        };
        (selected.id.clone(), Some(policy.to_string()))
    };
    let Some(survivor) = group
        .iter()
        .find(|candidate| candidate.id == selected_id)
        .cloned()
    else {
        return Err(ScalingError::InvalidRecord(
            "selected candidate is not in its group".to_string(),
        ));
    };

    Ok((
        GroupDecision {
            round_index,
            group_index,
            input_candidate_ids: candidate_ids(&group),
            votes,
            vote_counts: counts,
            selected_candidate_id: selected_id,
            tie_break,
        },
        survivor,
    ))
}

async fn collect_vote<B>(
    backend: &B,
    task: &Task,
    group: &[Candidate],
    config: &ScalingConfig,
    position: VotePosition,
    root_seed: u64,
) -> Result<Vote>
where
    B: ScalingBackend,
{
    let mut displayed = group.to_vec();
    if config.display_order == DisplayOrder::Shuffle {
        seed::shuffle(
            &mut displayed,
            seed::derive(
                root_seed,
                &[
                    2,
                    position.round as u64,
                    position.group as u64,
                    position.display as u64,
                ],
            ),
        );
    }
    let ordered_candidate_ids = candidate_ids(&displayed);
    let prompt = comparison_prompt(task, &displayed);
    let call_seed = seed::derive(
        root_seed,
        &[
            3,
            position.round as u64,
            position.group as u64,
            position.display as u64,
            position.vote as u64,
        ],
    );
    let request = ComparisonRequest {
        prompt: prompt.clone(),
        candidates: displayed.clone(),
        round_index: position.round,
        group_index: position.group,
        vote_index: position.vote,
        seed: call_seed,
    };

    match backend.compare(task, request).await {
        Ok(response) => {
            if response.model_id != backend.model_id() {
                return Err(ScalingError::InvalidRecord(
                    "comparison model ID must match the experiment model".to_string(),
                ));
            }
            let selected_position = parse_verdict(&response.content, displayed.len());
            let selected_candidate_id = selected_position
                .and_then(|value| displayed.get(value - 1))
                .map(|candidate| candidate.id.clone());
            let error = selected_candidate_id
                .is_none()
                .then(|| "judge response has no single in-range verdict".to_string());
            Ok(Vote {
                vote_index: position.vote,
                ordered_candidate_ids,
                prompt,
                seed: call_seed,
                model_id: Some(response.model_id),
                raw_response: Some(response.content),
                selected_position,
                selected_candidate_id,
                error,
            })
        }
        Err(error) => Ok(Vote {
            vote_index: position.vote,
            ordered_candidate_ids,
            prompt,
            seed: call_seed,
            model_id: None,
            raw_response: None,
            selected_position: None,
            selected_candidate_id: None,
            error: Some(error.to_string()),
        }),
    }
}

fn candidate_ids(candidates: &[Candidate]) -> Vec<String> {
    candidates
        .iter()
        .map(|candidate| candidate.id.clone())
        .collect()
}

fn validate_candidates(candidates: &[Candidate]) -> Result<()> {
    if candidates.is_empty() {
        return Err(ScalingError::InvalidConfig(
            "at least one tournament candidate is required".to_string(),
        ));
    }
    let mut ids = HashSet::with_capacity(candidates.len());
    for candidate in candidates {
        if candidate.id.trim().is_empty() || candidate.rollout_id.trim().is_empty() {
            return Err(ScalingError::InvalidRecord(
                "candidate identifiers must not be empty".to_string(),
            ));
        }
        if !ids.insert(&candidate.id) {
            return Err(ScalingError::InvalidRecord(
                "candidate identifiers must be unique".to_string(),
            ));
        }
        if candidate.rollout_id != candidate.summary.rollout_id {
            return Err(ScalingError::InvalidRecord(
                "candidate summary points to a different rollout".to_string(),
            ));
        }
        if candidate.summary.model_id.trim().is_empty() {
            return Err(ScalingError::InvalidRecord(
                "candidate summary model ID must not be empty".to_string(),
            ));
        }
    }
    Ok(())
}
