// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Policy knobs that control translation strictness and target capabilities.

use serde::{Deserialize, Serialize};

/// Controls whether known lossy conversions are allowed.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum LossyConversionPolicy {
    AllowWithDiagnostics,
    Reject,
}

/// Controls how missing provider IDs are generated.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum DeterministicIdPolicy {
    Preserve,
    GenerateStable { prefix: String },
}

/// Controls whether exact source payloads are preserved for round trips.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum PreservationPolicy {
    /// Keep exact source bodies in the in-memory IR only.
    InMemory,
    /// Also embed a Switchyard metadata envelope in translated wire bodies.
    ///
    /// This enables multi-hop round trips through wire formats, but should be
    /// used deliberately because provider APIs may reject or limit metadata.
    Embed,
    /// Do not store preservation metadata.
    Disabled,
}

/// Capability constraints for the target model or provider profile.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct TargetCapabilities {
    pub supports_tools: Option<bool>,
    pub supports_images: Option<bool>,
    pub supports_audio: Option<bool>,
    pub supports_video: Option<bool>,
    pub supports_files: Option<bool>,
    pub supports_reasoning_effort: Option<bool>,
    pub supports_json_schema_response_format: Option<bool>,
    pub supports_parallel_tool_calls: Option<bool>,
    pub supports_code_execution: Option<bool>,
    pub supports_safety_settings: Option<bool>,
    pub openai_compatible: Option<bool>,
}

/// End-to-end policy applied by request, response, and stream translation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TranslationPolicy {
    pub lossy_conversion_policy: LossyConversionPolicy,
    pub deterministic_ids: DeterministicIdPolicy,
    pub preservation: PreservationPolicy,
    pub target_capabilities: TargetCapabilities,
}

impl Default for TranslationPolicy {
    fn default() -> Self {
        Self {
            lossy_conversion_policy: LossyConversionPolicy::AllowWithDiagnostics,
            deterministic_ids: DeterministicIdPolicy::GenerateStable {
                prefix: "sw".to_string(),
            },
            preservation: PreservationPolicy::InMemory,
            target_capabilities: TargetCapabilities::default(),
        }
    }
}
