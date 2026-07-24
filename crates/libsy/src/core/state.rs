// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::Mutex;

use crate::ToolSignals;

/// A value in a session's [`State`].
#[derive(Debug, Clone)]
pub enum StateValue {
    String(String),
    Count(u32),
    Int(i32),
    Scalar(f32),
}

/// State maitaineed by [`Algorithm`]s
#[derive(Debug, Clone, Default)]
pub struct State {
    pub turn_count: u32,
    pub tool_signals: Option<ToolSignals>,
    pub extra: HashMap<String, StateValue>,
}

/// State shared accross tunrs / sessions
pub type SharedState = Arc<Mutex<State>>;
