// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Settings needed to interpret and repeat one experiment.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{Result, ScalingError};

/// Current serialized manifest version.
pub const MANIFEST_SCHEMA_VERSION: u16 = 1;

/// Paper-critical settings that the paper does not fully disclose.
pub const REQUIRED_MANIFEST_FIELDS: [&str; 22] = [
    "exact_prompts",
    "summary_schema",
    "malformed_summary_policy",
    "model_ids_and_api_revisions",
    "role_inference_settings",
    "scaffold_revisions_and_protocols",
    "agent_limits",
    "summary_serialization",
    "pairing_order",
    "display_order",
    "tie_break",
    "invalid_vote_policy",
    "model_retry_policy",
    "experiment_seeds",
    "benchmark_revisions",
    "terminal_bench_task_list",
    "summary_input_contents",
    "refinement_summary_order",
    "observation_truncation",
    "unfinished_rollout_policy",
    "concurrency_and_rate_limits",
    "ablation_fixed_settings",
];

/// Whether a run claims exact or conceptual replication.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplicationMode {
    /// Every paper-critical setting comes from the paper or author code.
    Exact,
    /// Missing paper details use explicit, recorded choices.
    Conceptual,
}

/// Where one manifest value came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ManifestSource {
    /// The paper states the value.
    Paper,
    /// Author code or configuration states the value.
    Repository,
    /// The replication chooses a value the paper does not state.
    Reconstructed,
}

/// One recorded paper-critical setting.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestValue {
    /// Evidence used for this value.
    pub source: ManifestSource,
    /// Exact value or revision used by the run.
    pub value: String,
}

/// Complete description of one experiment setup.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentManifest {
    /// Serialized manifest version.
    pub schema_version: u16,
    /// Replication claim made by this run.
    pub replication_mode: ReplicationMode,
    /// Source revision that ran the experiment.
    pub code_revision: String,
    /// Exact model shared by rollout, summary, and comparison calls.
    pub model_id: String,
    /// Required paper-critical settings.
    pub fields: BTreeMap<String, ManifestValue>,
}

impl ExperimentManifest {
    /// Rejects missing, unknown, empty, or unsupported manifest fields.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != MANIFEST_SCHEMA_VERSION {
            return Err(ScalingError::InvalidManifest(format!(
                "unsupported schema version {}; expected {MANIFEST_SCHEMA_VERSION}",
                self.schema_version
            )));
        }
        if self.code_revision.trim().is_empty() || self.model_id.trim().is_empty() {
            return Err(ScalingError::InvalidManifest(
                "code_revision and model_id must not be empty".to_string(),
            ));
        }

        for name in REQUIRED_MANIFEST_FIELDS {
            let Some(field) = self.fields.get(name) else {
                return Err(ScalingError::InvalidManifest(format!(
                    "missing required field {name}"
                )));
            };
            if field.value.trim().is_empty() {
                return Err(ScalingError::InvalidManifest(format!(
                    "field {name} must not be empty"
                )));
            }
            if self.replication_mode == ReplicationMode::Exact
                && field.source == ManifestSource::Reconstructed
            {
                return Err(ScalingError::InvalidManifest(format!(
                    "exact replication cannot use reconstructed field {name}"
                )));
            }
        }

        if let Some(name) = self
            .fields
            .keys()
            .find(|name| !REQUIRED_MANIFEST_FIELDS.contains(&name.as_str()))
        {
            return Err(ScalingError::InvalidManifest(format!(
                "unknown field {name}"
            )));
        }
        Ok(())
    }
}
