// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metadata describing a routing outcome.

use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::Value;
use switchyard_protocol::ModelId;

/// Identity, routing measurements, and optional algorithm evidence for a routing attempt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct OutcomeMetadata {
    outcome_id: String,
    /// Stable name of the algorithm that produced the outcome.
    pub algorithm: String,
    /// Algorithm contract version, when defined by the algorithm.
    pub algorithm_version: Option<String>,
    /// Reviewed boolean settings, or `None` when no feature contract is defined.
    pub feature_flags: Option<BTreeMap<String, bool>>,
    /// Candidate models read during routing, in first-seen order, excluding judge models.
    /// `None` means the algorithm did not expose its candidates through the driver.
    pub considered_model_ids: Option<Vec<ModelId>>,
    /// Terminal routing status: `success` or `error`, once the attempt completes.
    pub routing_status: Option<&'static str>,
    /// Whether routing failed because no eligible target was available, when known.
    pub no_eligible_target: Option<bool>,
    /// Bounded error category for a failed routing attempt; never a raw error message.
    pub routing_error_code: Option<&'static str>,
    /// Reasons candidates were excluded, or `None` when the algorithm does not report them.
    pub exclusion_reason_codes: Option<Vec<String>>,
    /// Elapsed routing time in whole milliseconds, excluding a subsequent answer call.
    pub routing_duration_ms: Option<u64>,
    /// Optional algorithm-defined JSON evidence.
    ///
    /// Built-in algorithms emit an object with a stable `source` string and only the
    /// relevant `score`, `confidence`, `threshold`, `verdict`, `trigger`, or `reason_code`
    /// fields. Not every algorithm or decision produces evidence.
    pub evidence: Option<Value>,
}

impl OutcomeMetadata {
    /// Creates outcome metadata with a new UUIDv7 identifier.
    pub fn new(algorithm: String, evidence: Option<Value>) -> Self {
        Self {
            outcome_id: uuid::Uuid::now_v7().to_string(),
            algorithm,
            algorithm_version: None,
            feature_flags: None,
            considered_model_ids: None,
            routing_status: None,
            no_eligible_target: None,
            routing_error_code: None,
            exclusion_reason_codes: None,
            routing_duration_ms: None,
            evidence,
        }
    }

    /// Returns the unique identifier for this outcome.
    pub fn outcome_id(&self) -> &str {
        &self.outcome_id
    }
}
