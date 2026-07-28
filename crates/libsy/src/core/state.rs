// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in state shared by processors and classifiers across a session.

use std::collections::HashMap;

use crate::ToolSignals;

/// A value in a session's [`State`].
#[derive(Debug, Clone)]
pub enum StateValue {
    String(String),
    Count(u32),
    Int(i32),
    Scalar(f32),
}

/// Routing facts accumulated across one session's algorithm runs.
#[derive(Debug, Clone, Default)]
pub struct State {
    pub turn_count: u32,
    /// Tool-result signals for the current request, set by the tool-signal
    /// processor. `None` until it runs or when the request has no tool activity,
    /// so routers must treat absence as "no signal yet".
    pub tool_signals: Option<ToolSignals>,
    pub extra: HashMap<String, StateValue>,
}
