// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module describes one completed model call in the routing log.

use serde::Deserialize;

#[derive(Debug, Deserialize, serde::Serialize)]
pub struct Entry {
    pub ts: String,
    pub model: String,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub turn_id: Option<String>,
    #[serde(default)]
    pub route_id: String,
    #[serde(default)]
    pub tier: String,
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub cached_tokens: u64,
    #[serde(default)]
    pub cache_creation_tokens: u64,
    #[serde(default)]
    pub cache_creation_1h_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    /// The server records the charge or estimate when it has rates; older records have none.
    #[serde(default)]
    pub cost: Option<switchyard_costs::RecordedCost>,
}
