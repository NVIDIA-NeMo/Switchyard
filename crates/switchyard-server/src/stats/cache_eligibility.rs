// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Switch-aware cache eligibility for the Rust server.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use serde_json::Value;

/// Set this environment variable (to =1) to enable tracking cache prefixes to estimate
/// kv-cache hit as stat `theoretical_cache_hit_rate`.
/// Memory will grow forever so this is only for evals.
const TRACK_ENV: &str = "SWITCHYARD_THEORETICAL_CACHE";

/// Whether theoretical cache-hit tracking is enabled via the environment.
pub(crate) fn tracking_enabled_from_env() -> bool {
    env_opts_in(std::env::var(TRACK_ENV).ok().as_deref())
}

fn env_opts_in(value: Option<&str>) -> bool {
    matches!(
        value
            .map(|value| value.trim().to_ascii_lowercase())
            .as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

/// Cumulative prefix fingerprints of a request, in message order.
#[derive(Clone, Debug, Default)]
pub(crate) struct PrefixProbe {
    boundaries: Vec<(u64, u64)>,
    total_len: u64,
}

impl PrefixProbe {
    /// Returns the fraction belonging to the longest prefix previously seen by a model.
    pub(crate) fn eligible_fraction(&self, seen: &HashSet<u64>) -> f64 {
        if self.total_len == 0 {
            return 0.0;
        }
        let eligible = self
            .boundaries
            .iter()
            .filter(|(_, hash)| seen.contains(hash))
            .map(|(len, _)| *len)
            .max()
            .unwrap_or(0);
        eligible as f64 / self.total_len as f64
    }

    /// Returns the full prompt fingerprint to retain after routing it to a model.
    pub(crate) fn full_hash(&self) -> Option<u64> {
        self.boundaries.last().map(|(_, hash)| *hash)
    }
}

/// Builds format-agnostic prefix fingerprints from a request body.
pub(crate) fn prefix_probe(body: &Value) -> PrefixProbe {
    let mut boundaries = Vec::new();
    let mut total_len = 0u64;
    let mut hasher = DefaultHasher::new();

    let system = body.get("system");
    let instructions = body.get("instructions");
    let prefix_len = system.map(text_len).unwrap_or(0) + instructions.map(text_len).unwrap_or(0);
    if prefix_len > 0 {
        total_len += prefix_len;
        for value in [system, instructions].into_iter().flatten() {
            hash_text_into(value, &mut hasher);
        }
        boundaries.push((total_len, hasher.finish()));
    }

    let turns = body
        .get("messages")
        .or_else(|| body.get("input"))
        .and_then(Value::as_array);
    if let Some(turns) = turns {
        for turn in turns {
            total_len += text_len(turn);
            hash_text_into(turn, &mut hasher);
            boundaries.push((total_len, hasher.finish()));
        }
    }
    PrefixProbe {
        boundaries,
        total_len,
    }
}

fn text_len(value: &Value) -> u64 {
    match value {
        Value::String(value) => value.len() as u64,
        Value::Array(items) => items.iter().map(text_len).sum(),
        Value::Object(map) => map.values().map(text_len).sum(),
        _ => 0,
    }
}

fn hash_text_into(value: &Value, hasher: &mut DefaultHasher) {
    match value {
        Value::String(value) => value.hash(hasher),
        Value::Number(value) => value.to_string().hash(hasher),
        Value::Bool(value) => value.hash(hasher),
        Value::Array(items) => items.iter().for_each(|item| hash_text_into(item, hasher)),
        Value::Object(map) => map.iter().for_each(|(key, value)| {
            key.hash(hasher);
            hash_text_into(value, hasher);
        }),
        Value::Null => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn environment_opt_in_values_are_explicit() {
        assert!(env_opts_in(Some(" TRUE ")));
        assert!(env_opts_in(Some("on")));
        assert!(!env_opts_in(Some("false")));
        assert!(!env_opts_in(None));
    }

    #[test]
    fn object_field_names_affect_the_fingerprint() {
        let left = prefix_probe(&serde_json::json!({"messages": [{"left": "same"}]}));
        let right = prefix_probe(&serde_json::json!({"messages": [{"right": "same"}]}));

        assert_ne!(left.full_hash(), right.full_hash());
    }
}
