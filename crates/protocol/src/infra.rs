// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Model-pool observations supplied by the serving integration.

use crate::ModelId;
use std::{collections::BTreeMap, time::Instant};

/// Observations keyed by model, without worker identities.
pub type InfraSignals = BTreeMap<ModelId, ModelInfraSignals>;

/// Pool observations and local request accounting.
#[derive(Clone, Debug, Default)]
pub struct ModelInfraSignals {
    /// None means unknown, never idle.
    pub load: Option<PoolLoad>,
    /// Requests still in flight through this host.
    pub local_inflight: u64,
}

/// Aggregated scheduler counts. The host is responsible for pool coverage.
#[derive(Clone, Debug)]
pub struct PoolLoad {
    /// Oldest underlying observation, expressed in the host's monotonic clock.
    pub sampled_at: Instant,
    /// Lowest queue depth among the pool's engines.
    pub min_waiting: u64,
    /// Highest queue depth among the pool's engines.
    pub max_waiting: u64,
    /// Total running requests across the pool.
    pub running: u64,
}
