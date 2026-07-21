// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Intake sink configuration.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Behavior when the async intake queue is full.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntakeQueueFullPolicy {
    /// Drop the payload and keep serving the user response.
    #[default]
    Drop,
    /// Wait for queue capacity before returning from enqueue.
    Block,
}

/// Payload shape a configured intake target expects.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntakeFormat {
    /// Nested OpenAI chat-completions JSON (the nemo-platform ingest shape).
    ChatCompletions,
    /// Flat, top-level type-prefixed telemetry document (data-lake posting shape).
    FlatDocument,
}

/// A configurable intake destination.
///
/// When a sink has no target it posts nested chat-completions documents to the
/// authenticated nemo-platform ingest URL built from `intake_base_url` +
/// `workspace`. Set a target to post a chosen payload shape to an explicit URL.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntakeTarget {
    /// Full posting URL. The caller builds it; the sink POSTs to it verbatim.
    pub url: String,
    /// Payload shape to emit.
    pub format: IntakeFormat,
    /// Send the sink's `api_key` as a bearer token with each POST.
    pub authenticated: bool,
}

/// Runtime configuration for the HTTP intake sink.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct IntakeSinkConfig {
    /// Base URL of the intake API service.
    pub intake_base_url: Option<String>,
    /// Workspace path segment for the intake API.
    pub workspace: Option<String>,
    /// User ID attached to every intake payload.
    pub user_id: String,
    /// Bearer token used when posting intake payloads.
    pub api_key: Option<String>,
    /// Optional explicit destination. Unset posts nested chat-completions
    /// documents to the authenticated nemo-platform ingest.
    pub target: Option<IntakeTarget>,
    /// Maximum buffered payloads before applying `on_queue_full`.
    pub max_queue_size: usize,
    /// Per-request HTTP timeout in seconds.
    pub request_timeout_s: f64,
    /// Number of retry attempts after the first failed POST.
    pub max_retries: u32,
    /// Queue pressure behavior.
    pub on_queue_full: IntakeQueueFullPolicy,
    /// Capture prompt/response text. Off by default (metadata-only).
    #[serde(default)]
    pub capture_content: bool,
}

impl fmt::Debug for IntakeSinkConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IntakeSinkConfig")
            .field("intake_base_url", &self.intake_base_url)
            .field("workspace", &self.workspace)
            .field("user_id", &self.user_id)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("target", &self.target)
            .field("max_queue_size", &self.max_queue_size)
            .field("request_timeout_s", &self.request_timeout_s)
            .field("max_retries", &self.max_retries)
            .field("on_queue_full", &self.on_queue_full)
            .field("capture_content", &self.capture_content)
            .finish()
    }
}

impl Default for IntakeSinkConfig {
    fn default() -> Self {
        Self {
            intake_base_url: None,
            workspace: None,
            user_id: "switchyard".to_string(),
            api_key: None,
            target: None,
            max_queue_size: 1000,
            request_timeout_s: 10.0,
            max_retries: 2,
            on_queue_full: IntakeQueueFullPolicy::Drop,
            capture_content: false,
        }
    }
}

impl IntakeSinkConfig {
    /// Returns the configured workspace or the Python-compatible default.
    pub fn workspace_or_default(&self) -> &str {
        self.workspace.as_deref().unwrap_or("default")
    }
}
