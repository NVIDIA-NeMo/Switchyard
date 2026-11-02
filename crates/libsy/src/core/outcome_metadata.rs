// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metadata describing a routing outcome.

use serde_json::Value;
use switchyard_protocol::ModelId;

/// A judge's selection after applying its routing policy.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct JudgeDecision {
    /// Selected target, or none when the judge abstains.
    pub target: Option<ModelId>,
    /// Category of the selected target, when known.
    pub category: Option<String>,
    /// Available scores, thresholds or failure reasons.
    pub evidence: Option<Value>,
}

/// Identity and optional algorithm evidence attached to a successful routing outcome.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct OutcomeMetadata {
    outcome_id: String,
    /// Stable name of the algorithm that produced the outcome.
    pub algorithm: String,
    /// Optional algorithm-defined JSON evidence.
    ///
    /// Built-in algorithms include a `source` and relevant scores or decision details.
    /// Not every algorithm or decision produces evidence.
    pub evidence: Option<Value>,
    /// Judge selections retained separately from the final routing reason.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub judgments: Vec<JudgeDecision>,
}

impl OutcomeMetadata {
    /// Creates outcome metadata with a new UUIDv7 identifier.
    pub fn new(algorithm: String, evidence: Option<Value>) -> Self {
        Self {
            outcome_id: uuid::Uuid::now_v7().to_string(),
            algorithm,
            evidence,
            judgments: Vec::new(),
        }
    }

    /// Returns the unique identifier for this outcome.
    pub fn outcome_id(&self) -> &str {
        &self.outcome_id
    }
}
