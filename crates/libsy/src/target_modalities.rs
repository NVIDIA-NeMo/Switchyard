// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared target-capability map and request eligibility checks.

use std::collections::{BTreeMap, BTreeSet};

use switchyard_protocol::{InputModality, ModelId, Request};

use crate::{LibsyError, Result};

/// Input modalities supported by each completion target in one route.
///
/// A router without this map retains legacy behavior. When present, the map is
/// expected to cover every completion target in the route.
pub type TargetModalities = BTreeMap<ModelId, BTreeSet<InputModality>>;

/// Returns compatible targets in configured route order.
pub(crate) fn eligible_targets(
    targets: &[ModelId],
    target_modalities: &TargetModalities,
    request: &Request,
) -> Result<(BTreeSet<InputModality>, Vec<ModelId>)> {
    let required_modalities = request.llm_request.input_modalities();
    let eligible = targets
        .iter()
        .filter(|target| {
            target_modalities
                .get(*target)
                .is_some_and(|supported| required_modalities.is_subset(supported))
        })
        .cloned()
        .collect::<Vec<_>>();

    if eligible.is_empty() {
        let target_modalities = targets
            .iter()
            .filter_map(|target| {
                target_modalities
                    .get(target)
                    .cloned()
                    .map(|modalities| (target.clone(), modalities))
            })
            .collect();
        return Err(LibsyError::NoCompatibleTargets {
            required_modalities,
            target_modalities,
        });
    }

    Ok((required_modalities, eligible))
}
