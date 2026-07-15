// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed intake request metadata and per-request intake state.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use switchyard_core::{ChatRequest, ChatRequestType};

/// Header carrying the client session ID used by intake.
pub const PROXY_SESSION_ID_HEADER: &str = "proxy_x_session_id";
/// Header that explicitly enables or disables intake capture.
pub const INTAKE_ENABLED_HEADER: &str = "x-switchyard-intake-enabled";
/// Header carrying the intake app label.
pub const INTAKE_APP_HEADER: &str = "x-switchyard-intake-app";
/// Header carrying the intake task label.
pub const INTAKE_TASK_HEADER: &str = "x-switchyard-intake-task";

/// Intake-specific metadata extracted from request headers.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct IntakeRequestMetadata {
    /// Optional explicit opt-in or opt-out flag.
    pub enabled: Option<bool>,
    /// Optional application label.
    pub app: Option<String>,
    /// Optional task label.
    pub task: Option<String>,
}

/// Per-request metadata shared across processors.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct RequestMetadata {
    /// Client session ID when provided.
    pub session_id: Option<String>,
    /// Intake-specific request controls.
    pub intake: IntakeRequestMetadata,
}

impl RequestMetadata {
    /// Extracts metadata from a case-insensitive header map.
    pub fn from_headers(headers: &BTreeMap<String, String>) -> Self {
        let normalized = headers
            .iter()
            .map(|(name, value)| (name.to_ascii_lowercase(), value.as_str()))
            .collect::<BTreeMap<_, _>>();
        Self {
            session_id: header_value(&normalized, PROXY_SESSION_ID_HEADER),
            intake: IntakeRequestMetadata {
                enabled: parse_bool(header_value(&normalized, INTAKE_ENABLED_HEADER).as_deref()),
                app: header_value(&normalized, INTAKE_APP_HEADER),
                task: header_value(&normalized, INTAKE_TASK_HEADER),
            },
        }
    }
}

/// Request-side state consumed by the response-side intake processor.
#[derive(Clone, Debug, PartialEq)]
pub struct IntakeRequestState {
    /// Request start timestamp in milliseconds since epoch.
    pub started_at_ms: i64,
    /// Original inbound wire format.
    pub inbound_format: ChatRequestType,
    /// Session ID copied from request metadata.
    pub session_id: Option<String>,
    /// Whether response capture should be skipped.
    pub skip: bool,
    /// Request body snapshot used when constructing the intake payload.
    pub request_snapshot: Option<ChatRequest>,
}

impl IntakeRequestState {
    /// Returns true when the response processor should leave the response alone.
    pub fn skipped(&self) -> bool {
        self.skip
    }
}

/// One routing-strategy sub-model call captured for intake: the model a router
/// called to pick a route, plus that call's token usage. Never message content.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubModelCall {
    /// Model the router called to make its routing decision.
    pub model: String,
    /// Prompt tokens the routing call spent.
    pub prompt_tokens: i64,
    /// Completion tokens the routing call spent.
    pub completion_tokens: i64,
    /// Cached prompt tokens the routing call reused.
    pub cached_tokens: i64,
    /// Router family that made the call (e.g. `deterministic`, `stage_router`).
    pub router_type: String,
    /// Route label the router produced (e.g. `classifier`, `planner`).
    pub routed_to: String,
}

/// Routing-strategy sub-model calls made while serving one request. Routers
/// append one entry per call; the response-side intake processor emits each as
/// its own anonymous record.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SubModelCalls(pub Vec<SubModelCall>);

// Header extraction treats empty strings as absent to match Python behavior.
fn header_value(headers: &BTreeMap<String, &str>, name: &str) -> Option<String> {
    headers
        .get(name)
        .copied()
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

// Only exact true/false strings are accepted; malformed values are ignored.
fn parse_bool(raw: Option<&str>) -> Option<bool> {
    match raw.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("true") => Some(true),
        Some("false") => Some(false),
        _ => None,
    }
}
