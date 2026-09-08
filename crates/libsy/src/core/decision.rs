// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional metadata explaining a routing decision.

/// Algorithm-owned metadata attached to a successful routing outcome.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecisionMetadata {
    /// Stable name of the algorithm that made the decision.
    pub algorithm: String,
    /// Optional bounded, machine-readable evidence produced by the algorithm.
    pub evidence: Option<String>,
}
