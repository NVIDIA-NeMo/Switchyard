// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Error types used by the Rust core contracts.

use thiserror::Error;

use crate::ids::{InvalidId, ModelId, ProfileId};
use crate::types::ChatRequestType;

/// Result alias for core Switchyard operations.
pub type Result<T> = std::result::Result<T, SwitchyardError>;

/// Shared error enum for configuration, processor, and backend failures.
#[derive(Debug, Error)]
pub enum SwitchyardError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    #[error(transparent)]
    InvalidId(#[from] InvalidId),

    #[error("{kind} {id:?} is already registered")]
    DuplicateRegistration { kind: &'static str, id: String },

    #[error("no model registered for {model}")]
    ModelNotFound { model: ModelId },

    #[error("no Decision API profile registered for {profile_id}")]
    DecisionProfileNotFound { profile_id: ProfileId },

    #[error("profile {profile_id} does not support the Decision API")]
    DecisionUnsupported { profile_id: ProfileId },

    #[error("{component} does not support request type {request_type:?}")]
    UnsupportedRequestType {
        component: String,
        request_type: ChatRequestType,
    },

    // Client sent a structurally valid but semantically invalid request
    // (e.g. an empty `messages` array). Surfaced as a 4xx, never a 5xx,
    // so agents can distinguish a client bug from a transient server failure.
    #[error("{0}")]
    InvalidRequest(String),

    #[error("processor failed: {0}")]
    Processor(String),

    #[error("backend failed: {0}")]
    Backend(String),

    #[error("upstream failed: {0}")]
    Upstream(String),

    #[error("upstream failed: {provider} returned HTTP {status_code}: {body}")]
    UpstreamHttp {
        provider: String,
        status_code: u16,
        body: String,
    },

    // Target hit its context window; routing runtime may evict + retry once.
    #[error("context window exceeded on target {target_id} ({model}): {message}")]
    ContextWindowExceeded {
        target_id: String,
        model: String,
        message: String,
    },

    // Every target was evicted; no fallback left to satisfy the request.
    #[error("context pool exhausted (last target {last_target_id}): {reason}")]
    ContextPoolExhausted {
        last_target_id: String,
        reason: String,
    },

    #[error("{0}")]
    Other(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_window_exceeded_renders_target_and_model() {
        let err = SwitchyardError::ContextWindowExceeded {
            target_id: "weak".into(),
            model: "nvidia/deepseek-ai/evals-deepseek-v4-pro".into(),
            message: "prompt is too long".into(),
        };
        let rendered = err.to_string();
        assert!(rendered.contains("weak"));
        assert!(rendered.contains("evals-deepseek-v4-pro"));
        assert!(rendered.contains("prompt is too long"));
    }

    #[test]
    fn context_pool_exhausted_renders_last_target() {
        let err = SwitchyardError::ContextPoolExhausted {
            last_target_id: "strong".into(),
            reason: "all targets evicted".into(),
        };
        let rendered = err.to_string();
        assert!(rendered.contains("strong"));
        assert!(rendered.contains("all targets evicted"));
    }
}
