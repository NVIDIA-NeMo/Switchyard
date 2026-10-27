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
    /// The caller-set origin header, such as `codex` or `claude`, when the harness sent one.
    #[serde(default)]
    pub origin: Option<String>,
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

#[cfg(test)]
mod tests {
    use super::Entry;

    #[test]
    fn origin_is_optional() {
        let with: Entry =
            serde_json::from_str(r#"{"ts":"2026-10-08T12:00:00Z","model":"m","origin":"claude"}"#)
                .expect("entry");
        assert_eq!(with.origin.as_deref(), Some("claude"));
        let without: Entry =
            serde_json::from_str(r#"{"ts":"2026-10-08T12:00:00Z","model":"m"}"#).expect("entry");
        assert!(without.origin.is_none());
    }
}
