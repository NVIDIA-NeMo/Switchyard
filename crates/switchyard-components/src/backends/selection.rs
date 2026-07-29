// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend execution metadata for observability components.

use crate::{LlmTargetId, ModelId};
use serde::{Deserialize, Serialize};

/// How a backend resolved the final upstream target/model for a request.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendSelectionReason {
    /// A router or caller set a selected target on `ProxyContext`.
    ContextTarget,
    /// The backend was configured with a deterministic default target.
    DefaultTarget,
    /// Only one target is configured, so there is no routing ambiguity.
    SingleTarget,
    /// The inbound request model uniquely matched a configured target model.
    RequestModel,
    /// A native backend has exactly one configured target.
    NativeTarget,
    /// A passthrough backend used the caller-provided model.
    PassthroughModel,
}

/// Final upstream backend selection for a request.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BackendSelection {
    /// Selected target ID when the backend resolved a concrete configured target.
    pub target_id: Option<LlmTargetId>,
    /// Final upstream model name used for the provider call.
    pub model: ModelId,
    /// Client-provided model name before backend routing or rewriting.
    pub original_model: Option<String>,
    /// Reason the backend selected this target/model.
    pub reason: BackendSelectionReason,
}

impl BackendSelection {
    /// Creates a selection for a concrete target-backed backend call.
    pub fn for_target(
        target_id: LlmTargetId,
        model: ModelId,
        original_model: Option<String>,
        reason: BackendSelectionReason,
    ) -> Self {
        Self {
            target_id: Some(target_id),
            model,
            original_model,
            reason,
        }
    }

    /// Creates a selection for a backend call that only resolved a model.
    pub fn for_model(
        model: ModelId,
        original_model: Option<String>,
        reason: BackendSelectionReason,
    ) -> Self {
        Self {
            target_id: None,
            model,
            original_model,
            reason,
        }
    }

    /// Records a native backend call while preserving an upstream routing reason
    /// when a parent backend already selected the same target/model.
    pub fn native_target_observation(
        previous: Option<&Self>,
        target_id: LlmTargetId,
        model: ModelId,
        original_model: Option<String>,
    ) -> Self {
        let matching_previous = previous.filter(|selection| {
            selection.target_id.as_ref() == Some(&target_id) && selection.model == model
        });
        Self {
            target_id: Some(target_id),
            model,
            original_model: matching_previous
                .and_then(|selection| selection.original_model.clone())
                .or(original_model),
            reason: matching_previous
                .map(|selection| selection.reason)
                .unwrap_or(BackendSelectionReason::NativeTarget),
        }
    }
}
