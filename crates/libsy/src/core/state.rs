// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in state shared by processors and classifiers across a session.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::algorithms::util::tool_signals::ToolSignals;

/// A value in a session's [`State`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateValue {
    /// Text value.
    String(String),
    /// Nonnegative counter.
    Count(u32),
    /// Flag value.
    Bool(bool),
}

/// Routing facts accumulated across one session's algorithm runs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct State {
    /// Tool-result signals for the current request, set by the tool-signal
    /// processor. `None` until it runs or when the request has no tool activity,
    /// so routers must treat absence as "no signal yet".
    #[serde(skip)]
    pub tool_signals: Option<ToolSignals>,
    /// Algorithm-specific state keyed by stable internal names.
    #[serde(flatten)]
    pub extra: BTreeMap<String, StateValue>,
}

impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        self.extra == other.extra
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_round_trips_through_json_and_compares_extra_only() -> serde_json::Result<()> {
        let mut state = State::default();
        state
            .extra
            .insert("capable_hold_turns".to_string(), StateValue::Count(2));
        state.extra.insert(
            "fall_open".to_string(),
            StateValue::String("capable".to_string()),
        );
        state
            .extra
            .insert("vgr_handoff_sent".to_string(), StateValue::Bool(true));
        let json = serde_json::json!({
            "capable_hold_turns": {"count": 2},
            "fall_open": {"string": "capable"},
            "vgr_handoff_sent": {"bool": true},
        });

        let with_signals = State {
            tool_signals: Some(ToolSignals::default()),
            ..state.clone()
        };
        assert_eq!(serde_json::to_value(&with_signals)?, json);
        assert_eq!(serde_json::from_value::<State>(json)?, state);
        assert_eq!(with_signals, state);
        Ok(())
    }
}
