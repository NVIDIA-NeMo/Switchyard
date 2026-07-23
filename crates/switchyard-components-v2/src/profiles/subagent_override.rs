// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sub-agent override combinator for v2 profiles.
//!
//! Wraps any profile without changing its behavior for normal traffic. A
//! request whose headers mark delegated sub-agent work
//! ([`switchyard_protocol::Metadata::is_subagent_work`]) is served directly by
//! a fixed worker target — keeping a sub-agent loop on an intentional,
//! cache-compatible target — while every other request delegates to the wrapped
//! profile. A worker failure surfaces as a normal target error and is never
//! silently re-routed through the wrapped profile.

use std::collections::BTreeMap;
use std::time::Instant;

use async_trait::async_trait;
use switchyard_components::StatsAccumulator;
use switchyard_core::{LlmTarget, Result};

use crate::backend::{native_target_backend, TargetBackend};
use crate::profile_stats_accumulator;
use crate::stats_recording::record_usage_or_wrap_stream;
use crate::{Profile, ProfileInput, ProfileResponse};

/// Wraps a profile, routing delegated sub-agent work to a fixed worker target.
pub(crate) struct SubagentOverrideProfile {
    inner: Box<dyn Profile>,
    worker: TargetBackend,
    stats: StatsAccumulator,
}

impl SubagentOverrideProfile {
    /// Wraps `inner`, routing recognized sub-agent work requests to `worker`.
    pub(crate) fn new(inner: Box<dyn Profile>, target: LlmTarget) -> Result<Self> {
        Ok(Self {
            inner,
            worker: native_target_backend(target)?,
            stats: profile_stats_accumulator(),
        })
    }
}

/// Returns true when the request headers signal delegated sub-agent work.
///
/// Normalizes multi-value headers to single-value (first occurrence wins)
/// before calling the protocol-layer detection logic.
fn is_subagent_work(headers: &BTreeMap<String, Vec<String>>) -> bool {
    let flat: BTreeMap<String, String> = headers
        .iter()
        .filter_map(|(k, vs)| vs.first().map(|v| (k.clone(), v.clone())))
        .collect();
    switchyard_protocol::Metadata::from_headers(&flat).is_subagent_work()
}

#[async_trait]
impl Profile for SubagentOverrideProfile {
    /// Routes sub-agent work to the worker target; delegates everything else.
    async fn run(&self, mut input: ProfileInput) -> Result<ProfileResponse> {
        if !is_subagent_work(&input.metadata.headers) {
            return self.inner.run(input).await;
        }

        let profile_started_at = Instant::now();
        let target_model = self.worker.target().model.clone();
        input.request.set_model(target_model.as_str());

        let backend_started_at = Instant::now();
        let response = match self.worker.call(&input.request).await {
            Ok(response) => response,
            Err(error) => {
                self.stats.record_error(target_model.as_str(), None)?;
                return Err(error);
            }
        };

        let backend_latency_ms = backend_started_at.elapsed().as_secs_f64() * 1000.0;
        self.stats
            .record_success(target_model.to_string(), Some(backend_latency_ms), None)?;
        let response = record_usage_or_wrap_stream(
            &self.stats,
            target_model.as_str(),
            None,
            profile_started_at,
            backend_latency_ms,
            response,
        )?;
        Ok(ProfileResponse::from(response))
    }
}
