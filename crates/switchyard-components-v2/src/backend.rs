// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend helpers for profile-owned runtimes.

use std::sync::Arc;

use async_trait::async_trait;
use switchyard_components::backends::{
    AnthropicNativeBackend, GeminiNativeBackend, OpenAiNativeBackend,
};
use switchyard_core::{
    BackendFormat, ChatRequest, ChatResponse, LlmTarget, Result, SwitchyardError,
};

/// Backend trait used by v2 profiles without exposing `ProxyContext`.
#[async_trait]
pub trait ProfileBackend: Send + Sync {
    /// Calls a backend with an already-prepared request.
    async fn call(&self, request: &ChatRequest) -> Result<ChatResponse>;
}

/// One target and the backend that serves it.
#[derive(Clone)]
pub struct TargetBackend {
    target: LlmTarget,
    backend: Arc<dyn ProfileBackend>,
}

impl TargetBackend {
    /// Creates a target/backend pair.
    pub fn new(target: LlmTarget, backend: Arc<dyn ProfileBackend>) -> Self {
        Self { target, backend }
    }

    /// Returns the target metadata.
    pub fn target(&self) -> &LlmTarget {
        &self.target
    }

    /// Calls the target backend with a request prepared by the profile.
    pub async fn call(&self, request: &ChatRequest) -> Result<ChatResponse> {
        self.backend.call(request).await
    }
}

/// Builds the existing native backend for one fully-resolved target.
pub(crate) fn native_target_backend(target: LlmTarget) -> Result<TargetBackend> {
    let backend: Arc<dyn ProfileBackend> = match target.format {
        BackendFormat::OpenAi | BackendFormat::Responses => {
            Arc::new(OpenAiNativeBackend::new(target.clone())?)
        }
        BackendFormat::Anthropic => Arc::new(AnthropicNativeBackend::new(target.clone())?),
        BackendFormat::Gemini => Arc::new(GeminiNativeBackend::new(target.clone())?),
        BackendFormat::Auto => {
            return Err(SwitchyardError::InvalidConfig(format!(
                "target {} must have a resolved backend format",
                target.id
            )));
        }
    };
    Ok(TargetBackend::new(target, backend))
}

#[async_trait]
impl ProfileBackend for OpenAiNativeBackend {
    async fn call(&self, request: &ChatRequest) -> Result<ChatResponse> {
        self.call_without_context(request).await
    }
}

#[async_trait]
impl ProfileBackend for AnthropicNativeBackend {
    async fn call(&self, request: &ChatRequest) -> Result<ChatResponse> {
        self.call_without_context(request).await
    }
}

#[async_trait]
impl ProfileBackend for GeminiNativeBackend {
    async fn call(&self, request: &ChatRequest) -> Result<ChatResponse> {
        self.call_without_context(request).await
    }
}
