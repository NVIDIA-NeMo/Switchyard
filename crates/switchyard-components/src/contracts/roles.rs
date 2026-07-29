// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend trait shared by compatibility LLM callers.

use async_trait::async_trait;

use super::context::ProxyContext;
use super::error::Result;
use super::types::{ChatRequest, ChatRequestType, ChatResponse};

/// Backend abstraction responsible for making the LLM call.
#[async_trait]
pub trait LlmBackend: Send + Sync {
    /// Returns the request formats this backend can accept directly.
    fn supported_request_types(&self) -> &[ChatRequestType];

    /// Calls the backend with the processed request.
    async fn call(&self, ctx: &mut ProxyContext, request: &ChatRequest) -> Result<ChatResponse>;

    /// Starts any resources owned by the backend.
    async fn startup(&self) -> Result<()> {
        Ok(())
    }

    /// Stops any resources owned by the backend.
    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }
}
