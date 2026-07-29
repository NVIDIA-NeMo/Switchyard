// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-side stats processor.

use crate::{ChatRequest, ProxyContext, Result};

use crate::stats::{prefix_probe, StatsRequestStart};

/// Records request start time, and optionally the prefix fingerprints used for
/// switch-aware theoretical cache eligibility (gated, since fingerprinting hashes
/// the full prompt each turn and the per-model seen-sets grow over a run).
#[derive(Clone, Copy, Debug, Default)]
pub struct StatsRequestProcessor {
    track_cache_eligibility: bool,
}

impl StatsRequestProcessor {
    /// Creates a processor; `track_cache_eligibility` gates prefix fingerprinting.
    pub fn new(track_cache_eligibility: bool) -> Self {
        Self {
            track_cache_eligibility,
        }
    }

    /// Records request-start metadata and returns the request unchanged.
    pub async fn process(
        &self,
        ctx: &mut ProxyContext,
        request: ChatRequest,
    ) -> Result<ChatRequest> {
        ctx.insert(StatsRequestStart::now());
        if self.track_cache_eligibility {
            ctx.insert(prefix_probe(request.body()));
        }
        Ok(request)
    }
}
