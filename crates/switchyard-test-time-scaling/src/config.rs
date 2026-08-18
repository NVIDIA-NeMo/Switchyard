// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Settings that change the scaling workflow.

use serde::{Deserialize, Serialize};

use crate::{Result, ScalingError};

/// How candidates are paired before each tournament round.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PairingOrder {
    /// Keep the current candidate order.
    #[default]
    InOrder,
    /// Shuffle candidates from the configured seed.
    Shuffle,
}

/// How candidates are shown to each judge vote.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DisplayOrder {
    /// Keep the group order.
    #[default]
    InOrder,
    /// Shuffle the displayed positions for every vote.
    Shuffle,
}

/// How a tied vote is resolved.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TiePolicy {
    /// Select the earliest tied candidate in the original group.
    #[default]
    FirstInGroup,
    /// Select a tied candidate from the configured seed.
    SeededRandom,
}

/// How malformed or failed judge votes are handled.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "mode")]
pub enum InvalidVotePolicy {
    /// Stop the tournament when any required vote is invalid.
    #[default]
    Abort,
    /// Replace invalid votes up to the stated call limit per group.
    Replace {
        /// Maximum replacement calls made for one group.
        max_calls_per_group: usize,
    },
}

/// Settings for the two-iteration PDR and RTV workflow.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalingConfig {
    /// Independent attempts run in each iteration.
    pub rollout_count: usize,
    /// Iteration-zero summaries given to every refined attempt.
    pub refinement_count: usize,
    /// Maximum comparison group size.
    pub group_size: usize,
    /// Valid votes required for every comparison group.
    pub votes_per_group: usize,
    /// Root seed used for repeatable ordering and adapter calls.
    pub seed: u64,
    /// Pairing order used by tournaments.
    pub pairing_order: PairingOrder,
    /// Display order used by judge calls.
    pub display_order: DisplayOrder,
    /// Tie policy used by group decisions.
    pub tie_policy: TiePolicy,
    /// Invalid-vote policy used by group decisions.
    pub invalid_vote_policy: InvalidVotePolicy,
}

impl Default for ScalingConfig {
    fn default() -> Self {
        Self {
            rollout_count: 16,
            refinement_count: 4,
            group_size: 2,
            votes_per_group: 8,
            seed: 0,
            pairing_order: PairingOrder::InOrder,
            display_order: DisplayOrder::InOrder,
            tie_policy: TiePolicy::FirstInGroup,
            invalid_vote_policy: InvalidVotePolicy::Abort,
        }
    }
}

impl ScalingConfig {
    /// Rejects settings that cannot run the two-iteration workflow.
    pub fn validate(&self) -> Result<()> {
        if self.rollout_count == 0 {
            return Err(ScalingError::InvalidConfig(
                "rollout_count must be at least 1".to_string(),
            ));
        }
        if self.refinement_count == 0 || self.refinement_count > self.rollout_count {
            return Err(ScalingError::InvalidConfig(
                "refinement_count must be between 1 and rollout_count".to_string(),
            ));
        }
        if self.group_size < 2 {
            return Err(ScalingError::InvalidConfig(
                "group_size must be at least 2".to_string(),
            ));
        }
        if self.votes_per_group == 0 {
            return Err(ScalingError::InvalidConfig(
                "votes_per_group must be at least 1".to_string(),
            ));
        }
        if matches!(
            self.invalid_vote_policy,
            InvalidVotePolicy::Replace {
                max_calls_per_group: 0
            }
        ) {
            return Err(ScalingError::InvalidConfig(
                "replacement vote limit must be at least 1".to_string(),
            ));
        }
        Ok(())
    }
}
