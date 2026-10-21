// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-specific serving observations supplied by trusted hosts.

use std::{collections::BTreeMap, time::Instant};

use crate::{ModelId, Request};
use async_trait::async_trait;

/// Candidate observations keyed by concrete model ID. Missing entries mean unknown.
pub type InfraSignals = BTreeMap<ModelId, ModelInfraSignals>;

/// Observations for the serving system's candidate worker for one model.
#[derive(Clone, Debug)]
pub struct ModelInfraSignals {
    /// Conservative local sampling time, such as the start of a probe call.
    pub sampled_at: Instant,
    /// Remaining prefill work after cache credits, using this model's tokenization.
    pub effective_prefill_tokens: Option<u64>,
    /// Tracked prefill work already assigned to the candidate worker.
    pub active_prefill_tokens: Option<u64>,
    /// In-flight requests on the candidate, not an engine queue length.
    pub active_requests: Option<u64>,
}

/// Host-owned infrastructure lookups. Implementations omit failed or unavailable observations.
///
/// Use the supplied request and each model's own preprocessing. Calls must be read-only;
/// the final serving path owns worker selection and admission.
#[async_trait]
pub trait InfraProvider: Send + Sync {
    /// Collect fresh observations for the supplied candidates. An empty map means unavailable.
    async fn probe(&self, request: &Request, candidates: &[ModelId]) -> InfraSignals;
}
