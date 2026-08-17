// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Target-specific system-prompt policy shared by routing algorithms and hosts.

use std::collections::BTreeMap;

use switchyard_protocol::ModelId;

/// System prompts keyed by routing target. A target left unset is routed untouched.
#[derive(Clone, Debug, Default)]
pub struct TargetPrompts {
    by_target: BTreeMap<ModelId, String>,
}

impl TargetPrompts {
    /// Hand `target` this prompt on every turn it serves.
    pub fn with(mut self, target: impl Into<ModelId>, prompt: impl Into<String>) -> Self {
        self.by_target.insert(target.into(), prompt.into());
        self
    }

    /// The prompt configured for `target`, if any.
    pub fn get(&self, target: &ModelId) -> Option<&str> {
        self.by_target.get(target).map(String::as_str)
    }

    /// Whether any target has a prompt, so a caller can skip empty policy layers.
    pub fn is_empty(&self) -> bool {
        self.by_target.is_empty()
    }
}
