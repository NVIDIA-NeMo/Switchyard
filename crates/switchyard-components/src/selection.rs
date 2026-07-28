// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-call selection metadata for observability components.

use serde::{Deserialize, Serialize};
use switchyard_core::{LlmTargetId, ModelId};

/// How the LLM client resolved the final upstream target/model for a request.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendSelectionReason {
    /// A router or caller set a selected target on `ProxyContext`.
    ContextTarget,
    /// The client was configured with a deterministic default target.
    DefaultTarget,
    /// Only one target is configured, so there is no routing ambiguity.
    SingleTarget,
    /// The inbound request model uniquely matched a configured target model.
    RequestModel,
    /// A passthrough client used the caller-provided model.
    PassthroughModel,
}

/// Final upstream selection for a request.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BackendSelection {
    /// Selected target ID when the client resolved a concrete configured target.
    pub target_id: Option<LlmTargetId>,
    /// Final upstream model name used for the provider call.
    pub model: ModelId,
    /// Client-provided model name before routing or rewriting.
    pub original_model: Option<String>,
    /// Reason the client selected this target/model.
    pub reason: BackendSelectionReason,
}

impl BackendSelection {
    /// Creates a selection for a concrete target call.
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

    /// Creates a selection for a call that only resolved a model.
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
}
