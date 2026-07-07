// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LLM target configuration shared by routing and factory code.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ids::{LlmTargetId, ModelId};

/// Wire format expected by an LLM target.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendFormat {
    /// Format has not been resolved yet and must not reach native backend build.
    #[default]
    Auto,
    /// OpenAI-compatible Chat Completions fallback wire format.
    #[serde(rename = "openai")]
    OpenAi,
    /// OpenAI Responses API wire format for native `/v1/responses` targets.
    Responses,
    /// Anthropic Messages wire format.
    Anthropic,
    /// Gemini generateContent wire format.
    Gemini,
}

/// Optional endpoint overrides for an LLM target.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EndpointConfig {
    /// Optional upstream base URL, usually ending in `/v1`.
    pub base_url: Option<String>,
    /// Optional upstream API key; environment fallback remains backend-specific.
    pub api_key: Option<String>,
    /// Optional upstream request timeout in seconds.
    pub timeout_secs: Option<f64>,
}

impl EndpointConfig {
    /// Merges a shared endpoint with target-local overrides.
    ///
    /// Fields set directly on the target win over fields inherited from the
    /// shared endpoint definition.
    pub fn with_overrides(&self, overrides: &Self) -> Self {
        Self {
            base_url: overrides.base_url.clone().or_else(|| self.base_url.clone()),
            api_key: overrides.api_key.clone().or_else(|| self.api_key.clone()),
            timeout_secs: overrides.timeout_secs.or(self.timeout_secs),
        }
    }
}

/// A concrete upstream model target that routing processors can select.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LlmTarget {
    /// Stable target ID used by routers and config references.
    pub id: LlmTargetId,
    /// Upstream model name sent to the provider.
    pub model: ModelId,
    /// Native wire format expected by the upstream target.
    pub format: BackendFormat,
    /// Connection settings for the upstream target.
    #[serde(default)]
    pub endpoint: EndpointConfig,
    /// Per-target outbound request extensions, merged into the request
    /// body by the wire-specific backend before the upstream call.
    ///
    /// Use cases:
    ///
    /// * **`chat_template_kwargs.enable_thinking=False`** for DeepSeek
    ///   V4 (Flash / Pro) on NVIDIA Inference Hub.  V4 is a chain-of-
    ///   thought model whose default reasoning blows past Hub's proxy
    ///   gateway timeout at ``-n 8`` concurrency (504s on ~5% of
    ///   requests); the flag disables thinking and pegs response
    ///   times at ~5 s flat.  Cannot be set client-side because Hub
    ///   ignores the request-level ``reasoning_effort`` field for
    ///   these models.
    /// * Provider-specific options (vLLM ``guided_json``,
    ///   ``logit_bias``, ``response_format`` shimming) that should
    ///   apply to *every* request to this target.
    ///
    /// Merge semantics are shallow and **caller-wins**: keys already
    /// present in the inbound request body are not overridden.  Set on
    /// the target only what the caller is not expected to set itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_body: Option<Value>,
    /// Per-target outbound HTTP headers attached to every upstream
    /// request issued for this target.
    ///
    /// Distinct from :attr:`extra_body` because some gateway-level
    /// routing keys live in headers, not the request body.  Example:
    /// NVIDIA Inference Hub exposes an evals/benchmarking gateway
    /// behind ``X-Inference-Priority: batch`` — required for DeepSeek
    /// V4 calls during long benchmark sweeps because the regular
    /// gateway enforces a ~6-min timeout that under ``-n 8``
    /// concurrency manifests as cascading 504s.
    ///
    /// Headers added here are appended to whatever the backend would
    /// already send (``Authorization``, ``anthropic-version``,
    /// telemetry).  Reserved header names supplied by the backend
    /// (``Authorization`` / ``x-api-key`` / ``anthropic-version``)
    /// are still authoritative; ``extra_headers`` cannot override
    /// them — the underlying ``reqwest`` builder appends each entry
    /// rather than replacing existing ones, so a duplicate name
    /// would create a multi-valued header rather than a
    /// silent-override security hazard.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra_headers: BTreeMap<String, String>,
}

impl LlmTarget {
    /// Creates a target with automatic format detection and no endpoint overrides.
    pub fn new(id: LlmTargetId, model: ModelId) -> Self {
        Self {
            id,
            model,
            format: BackendFormat::Auto,
            endpoint: EndpointConfig::default(),
            extra_body: None,
            extra_headers: BTreeMap::new(),
        }
    }
}

/// Shallow-merges ``target_extra`` into ``body``: keys already present
/// in ``body`` are preserved (caller wins); new keys from
/// ``target_extra`` are added.  No-op if either side is not an object.
///
/// Used by wire-specific backends to inject per-target
/// :attr:`LlmTarget.extra_body` into outbound request bodies without
/// stomping caller-supplied fields.
pub fn merge_target_extra_body(body: &mut Value, target_extra: Option<&Value>) {
    let Some(Value::Object(extra)) = target_extra else {
        return;
    };
    let Value::Object(body_map) = body else {
        return;
    };
    for (key, value) in extra {
        body_map.entry(key.clone()).or_insert_with(|| value.clone());
    }
}
