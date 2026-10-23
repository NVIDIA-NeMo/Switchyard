// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Serving observations supplied by the host for infrastructure-aware routing.

use crate::ModelId;
use std::{collections::BTreeMap, time::Instant};

/// Model observations. Missing models or workers mean unknown, never idle.
pub type InfraSignals = BTreeMap<ModelId, ModelInfraSignals>;

/// Backend worker gauges and requests still in flight through this host.
#[derive(Clone, Debug, Default)]
pub struct ModelInfraSignals {
    /// One entry per independently scheduled worker, keyed by configured worker ID.
    pub workers: BTreeMap<String, WorkerLoad>,
    /// All requests sent to this model that have not completed at this host.
    pub local_inflight: u64,
    /// Subset of local requests moved from the efficient tier.
    pub local_spillovers: u64,
}

/// Running and waiting gauges from one worker scrape.
#[derive(Clone, Debug)]
pub struct WorkerLoad {
    /// Local monotonic time at the start of collection.
    pub sampled_at: Instant,
    /// Backend requests currently executing.
    pub running: u64,
    /// Backend requests waiting for execution capacity.
    pub waiting: u64,
}
