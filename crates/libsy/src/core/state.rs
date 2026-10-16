// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in state shared by processors and classifiers across a session.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::algorithms::util::tool_signals::ToolSignals;

/// A value in a session's [`State`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StateValue {
    /// Text value.
    String(String),
    /// Nonnegative counter.
    Count(u32),
    /// Signed integer value.
    Int(i32),
    /// Floating-point value.
    Scalar(f32),
}

/// Routing facts accumulated across one session's algorithm runs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    /// Number of turns processed for this session.
    pub turn_count: u32,
    /// Tool-result signals for the current request, set by the tool-signal
    /// processor. `None` until it runs or when the request has no tool activity,
    /// so routers must treat absence as "no signal yet".
    #[serde(skip)] // Recomputed from the next request; never persist transcript-derived text.
    pub tool_signals: Option<ToolSignals>,
    /// Algorithm-specific state keyed by stable internal names.
    pub extra: HashMap<String, StateValue>,
}

/// Opaque, versioned algorithm state for a host that owns session persistence.
/// Hosts must also bind this to the configuration and session that produced it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionCheckpoint {
    /// Checkpoint schema version, independent of the host's storage format.
    pub version: u32,
    /// Algorithm that owns the payload.
    pub algorithm: String,
    /// Algorithm-owned state. Hosts must not interpret or edit it.
    pub state: serde_json::Value,
}

impl SessionCheckpoint {
    pub(crate) fn new(algorithm: &str, state: impl Serialize) -> crate::Result<Self> {
        Ok(Self {
            version: 1,
            algorithm: algorithm.to_string(),
            state: serde_json::to_value(state)
                .map_err(|error| crate::LibsyError::external("encode session checkpoint", error))?,
        })
    }

    pub(crate) fn decode<T: serde::de::DeserializeOwned>(
        &self,
        algorithm: &str,
    ) -> crate::Result<T> {
        if self.version != 1 || self.algorithm != algorithm {
            return Err(crate::LibsyError::AlgorithmError {
                message: "Incompatible session checkpoint".to_string(),
            });
        }
        serde_json::from_value(self.state.clone())
            .map_err(|error| crate::LibsyError::external("decode session checkpoint", error))
    }
}
