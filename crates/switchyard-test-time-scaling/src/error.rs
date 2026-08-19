// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Errors returned by the scaling workflow.

use thiserror::Error;

/// Result type used by this crate.
pub type Result<T> = std::result::Result<T, ScalingError>;

/// A scaling run could not continue safely.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ScalingError {
    /// The experiment manifest is incomplete or makes an unsupported claim.
    #[error("invalid experiment manifest: {0}")]
    InvalidManifest(String),

    /// Configuration values do not describe a valid workflow.
    #[error("invalid scaling config: {0}")]
    InvalidConfig(String),

    /// The adapter failed to run an attempt or summary call.
    #[error("backend failed: {0}")]
    Backend(String),

    /// An adapter returned records that do not match the request.
    #[error("invalid backend record: {0}")]
    InvalidRecord(String),

    /// Official outcomes do not cover one completed run exactly once.
    #[error("invalid evaluation: {0}")]
    InvalidEvaluation(String),

    /// A completed run could not be encoded or decoded.
    #[error("run record failed: {0}")]
    Record(String),

    /// The population cannot be divided without an unspecified bye policy.
    #[error("population {population} is not divisible by group size {group_size}")]
    UnevenGroups {
        /// Number of current candidates.
        population: usize,
        /// Effective group size for this round.
        group_size: usize,
    },

    /// A round would reduce the population below the requested survivor count.
    #[error(
        "group size {group_size} would reduce population {population} below {target} survivors"
    )]
    OvershootsTarget {
        /// Number of current candidates.
        population: usize,
        /// Effective group size for this round.
        group_size: usize,
        /// Requested survivor count.
        target: usize,
    },

    /// A group did not produce the required number of valid votes.
    #[error("group produced {actual} valid votes; expected {expected}")]
    IncompleteVotes {
        /// Required valid-vote count.
        expected: usize,
        /// Actual valid-vote count.
        actual: usize,
    },
}
