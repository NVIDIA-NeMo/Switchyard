// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metadata describing a routing outcome.

/// Identity and optional algorithm evidence attached to a successful routing outcome.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutcomeMetadata {
    outcome_id: String,
    /// Stable name of the algorithm that produced the outcome.
    pub algorithm: String,
    /// Optional bounded, machine-readable evidence produced by the algorithm.
    pub evidence: Option<String>,
}

impl OutcomeMetadata {
    /// Creates outcome metadata with a new UUIDv7 identifier.
    pub fn new(algorithm: String, evidence: Option<String>) -> Self {
        Self {
            outcome_id: uuid::Uuid::now_v7().to_string(),
            algorithm,
            evidence,
        }
    }

    /// Returns the unique identifier for this outcome.
    pub fn outcome_id(&self) -> &str {
        &self.outcome_id
    }
}
