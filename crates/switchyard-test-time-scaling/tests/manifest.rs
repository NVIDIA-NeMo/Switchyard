// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use switchyard_test_time_scaling::{
    ExperimentManifest, MANIFEST_SCHEMA_VERSION, ManifestSource, ManifestValue,
    REQUIRED_MANIFEST_FIELDS, ReplicationMode,
};

fn manifest(source: ManifestSource) -> ExperimentManifest {
    ExperimentManifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        replication_mode: ReplicationMode::Conceptual,
        code_revision: "revision".to_string(),
        model_id: "model".to_string(),
        fields: BTreeMap::from_iter(REQUIRED_MANIFEST_FIELDS.map(|name| {
            (
                name.to_string(),
                ManifestValue {
                    source,
                    value: "recorded value".to_string(),
                },
            )
        })),
    }
}

#[test]
fn conceptual_manifest_requires_every_recorded_choice() {
    let mut value = manifest(ManifestSource::Reconstructed);
    assert!(value.validate().is_ok());

    value.fields.remove("tie_break");
    assert!(value.validate().is_err());
}

#[test]
fn exact_manifest_rejects_reconstructed_choices() {
    let mut value = manifest(ManifestSource::Reconstructed);
    value.replication_mode = ReplicationMode::Exact;
    assert!(value.validate().is_err());

    for field in value.fields.values_mut() {
        field.source = ManifestSource::Paper;
    }
    assert!(value.validate().is_ok());
}

#[test]
fn manifest_rejects_unknown_and_empty_fields() {
    let mut value = manifest(ManifestSource::Paper);
    value.fields.insert(
        "typo".to_string(),
        ManifestValue {
            source: ManifestSource::Paper,
            value: "value".to_string(),
        },
    );
    assert!(value.validate().is_err());

    value.fields.remove("typo");
    value
        .fields
        .get_mut("agent_limits")
        .expect("agent_limits fixture")
        .value
        .clear();
    assert!(value.validate().is_err());
}
