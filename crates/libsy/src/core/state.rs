// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in state shared by processors and classifiers across a session.

use std::collections::HashMap;

use crate::algorithms::util::tool_signals::ToolSignals;

/// A value in a session's [`State`].
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone, Default)]
pub struct State {
    /// Number of turns processed for this session.
    pub turn_count: u32,
    /// Tool-result signals for the current request, set by the tool-signal
    /// processor. `None` until it runs or when the request has no tool activity,
    /// so routers must treat absence as "no signal yet".
    pub tool_signals: Option<ToolSignals>,
    /// Algorithm-specific state keyed by stable internal names.
    pub extra: HashMap<String, StateValue>,
}

/// Provides access to the [`StateValue`] map accumulated during algorithm execution.
///
/// Implemented on state types that carry routing metadata. The map is cloned once and
/// attached to the published [`crate::core::algorithm::Step::Decision`], so downstream
/// observers (e.g. Python bindings) can read signal fields without parsing log output.
pub trait WithExtra {
    /// Returns a snapshot of the extra routing metadata accumulated so far.
    fn routing_extra(&self) -> HashMap<String, StateValue>;
}

impl WithExtra for State {
    fn routing_extra(&self) -> HashMap<String, StateValue> {
        self.extra.clone()
    }
}

/// Stateless compositions carry no extra; callers receive an empty map.
impl WithExtra for () {
    fn routing_extra(&self) -> HashMap<String, StateValue> {
        HashMap::new()
    }
}
