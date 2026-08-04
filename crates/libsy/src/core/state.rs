// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in state shared by processors and classifiers across a session.

use std::collections::HashMap;

use crate::algorithms::util::tool_signals::ToolSignals;

/// A value in a session's [`State`].
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) enum StateValue {
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
#[allow(dead_code)]
pub(crate) struct State {
    /// Number of turns processed for this session.
    pub(crate) turn_count: u32,
    /// Tool-result signals for the current request, set by the tool-signal
    /// processor. `None` until it runs or when the request has no tool activity,
    /// so routers must treat absence as "no signal yet".
    pub(crate) tool_signals: Option<ToolSignals>,
    /// Algorithm-specific state keyed by stable internal names.
    pub(crate) extra: HashMap<String, StateValue>,
}
