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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_assigns_unique_uuidv7_and_preserves_inputs() {
        let metadata = OutcomeMetadata::new("test".to_string(), Some("matched".to_string()));
        let another = OutcomeMetadata::new("test".to_string(), None);

        assert_eq!(metadata.algorithm, "test");
        assert_eq!(metadata.evidence.as_deref(), Some("matched"));
        assert_eq!(
            uuid::Uuid::parse_str(metadata.outcome_id())
                .expect("outcome id should be a UUID")
                .get_version_num(),
            7
        );
        assert_ne!(metadata.outcome_id(), another.outcome_id());
    }
}
