// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Switch-aware cache eligibility: how much of a prompt a model has already been sent.

use std::collections::HashSet;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use serde_json::Value;

/// Env var that opts in to per-model theoretical cache-hit tracking.
const TRACK_ENV: &str = "SWITCHYARD_THEORETICAL_CACHE";

/// Whether theoretical cache-hit tracking is enabled via the environment.
///
/// Off by default: prefix fingerprinting and the per-model seen-sets are skipped
/// unless opted in, so the hot path adds nothing and memory stays flat.
pub fn tracking_enabled_from_env() -> bool {
    env_opts_in(std::env::var(TRACK_ENV).ok().as_deref())
}

fn env_opts_in(value: Option<&str>) -> bool {
    matches!(
        value.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

/// Cumulative prefix fingerprints of a request, in message order.
#[derive(Clone, Debug, Default)]
pub struct PrefixProbe {
    /// `(cumulative_text_len, rolling_hash)` after the prefix fields and each turn.
    boundaries: Vec<(u64, u64)>,
    /// Total prompt text length, including the newest turn.
    total_len: u64,
}

impl PrefixProbe {
    /// Eligible fraction given the prefix fingerprints a model has already seen.
    /// Longest matching boundary over total; the newest turn is unseen, so excluded.
    pub fn eligible_fraction(&self, seen: &HashSet<u64>) -> f64 {
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

    /// Fingerprint of the full prompt, recorded once a model has processed it.
    pub fn full_hash(&self) -> Option<u64> {
        self.boundaries.last().map(|(_, hash)| *hash)
    }
}

/// Builds prefix fingerprints from a request body.
/// Format-agnostic: `system`/`instructions`, then each `messages`/`input` turn.
pub fn prefix_probe(body: &Value) -> PrefixProbe {
    let mut boundaries = Vec::new();
    let mut acc_len = 0u64;
    let mut hasher = DefaultHasher::new();

    let system = body.get("system");
    let instructions = body.get("instructions");
    let prefix_len = system.map(text_len).unwrap_or(0) + instructions.map(text_len).unwrap_or(0);
    if prefix_len > 0 {
        acc_len += prefix_len;
        for value in [system, instructions].into_iter().flatten() {
            hash_text_into(value, &mut hasher);
        }
        boundaries.push((acc_len, hasher.finish()));
    }

    let turns = body
        .get("messages")
        .or_else(|| body.get("input"))
        .and_then(Value::as_array);
    if let Some(turns) = turns {
        for turn in turns {
            acc_len += text_len(turn);
            hash_text_into(turn, &mut hasher);
            boundaries.push((acc_len, hasher.finish()));
        }
    }
    PrefixProbe {
        boundaries,
        total_len: acc_len,
    }
}

/// Recursively sums the byte length of every JSON string value.
fn text_len(value: &Value) -> u64 {
    match value {
        Value::String(s) => s.len() as u64,
        Value::Array(items) => items.iter().map(text_len).sum(),
        Value::Object(map) => map.values().map(text_len).sum(),
        _ => 0,
    }
}

/// Recursively feeds every JSON scalar value into the hasher, in order.
/// Includes numbers and bools so prompts differing only in a scalar don't collide.
fn hash_text_into(value: &Value, hasher: &mut DefaultHasher) {
    match value {
        Value::String(s) => s.hash(hasher),
        Value::Number(n) => n.to_string().hash(hasher),
        Value::Bool(b) => b.hash(hasher),
        Value::Array(items) => items.iter().for_each(|item| hash_text_into(item, hasher)),
        Value::Object(map) => map.values().for_each(|val| hash_text_into(val, hasher)),
        Value::Null => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn env_opt_in_parsing() {
        assert!(env_opts_in(Some("1")));
        assert!(env_opts_in(Some(" TRUE ")));
        assert!(env_opts_in(Some("on")));
        assert!(!env_opts_in(Some("0")));
        assert!(!env_opts_in(Some("false")));
        assert!(!env_opts_in(None));
    }

    #[test]
    fn empty_body_is_zero() {
        assert_eq!(
            prefix_probe(&json!({})).eligible_fraction(&HashSet::new()),
            0.0
        );
    }

    #[test]
    fn unseen_prefix_is_not_eligible() {
        let probe = prefix_probe(&json!({"messages": [{"role": "user", "content": "aaaa"}]}));
        assert_eq!(probe.eligible_fraction(&HashSet::new()), 0.0);
    }

    #[test]
    fn previously_seen_prefix_is_eligible_newest_turn_is_not() {
        let turn1 = prefix_probe(&json!({"messages": [{"role": "user", "content": "aaaa"}]}));
        let mut seen = HashSet::new();
        seen.insert(turn1.full_hash().unwrap());

        // Same first turn plus an equal-length newest turn -> half is re-presentable.
        let turn2 = prefix_probe(&json!({
            "messages": [
                {"role": "user", "content": "aaaa"},
                {"role": "user", "content": "bbbb"},
            ],
        }));
        assert_eq!(turn2.eligible_fraction(&seen), 0.5);
    }
}
