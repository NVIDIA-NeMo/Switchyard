// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Signal extraction for the stage_router.
//!
//! Owns the pure logic for reading a coding-agent request's tool-call history
//! into a [`ToolResultSignal`] (write/edit/read counts, error severity, streaks),
//! plus response-side signals. The
//! [`crate::request_processors::dimension_collector`] module wraps it as a
//! request-side Switchyard component.

pub mod response;
pub mod tool_signals;

pub use response::{ResponseFlag, ResponseSignals, extract_response_signals};
pub use tool_signals::{
    DEFAULT_RECENT_WINDOW, ToolResultSignal, extract_tool_signals, extract_tool_signals_with_window,
};
