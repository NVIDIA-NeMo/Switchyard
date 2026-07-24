// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-processor adapter for the tool-signal extractor.
//!
//! Thin wrapper around [`crate::dimension_collector::extract_tool_signals_with_window`].
//! It walks the request's tool-call history and stamps the resulting
//! [`ToolResultSignal`] into `ProxyContext` for the stage_router picker to read.

use switchyard_core::{ChatRequest, ProxyContext, Result};

use crate::dimension_collector::{
    extract_tool_signals_with_window, ToolResultSignal, DEFAULT_RECENT_WINDOW,
};

/// Populates `ProxyContext` with a [`ToolResultSignal`] read from the request's
/// tool-call history. The stage_router picker reads it via
/// `ctx.get::<ToolResultSignal>()`.
#[derive(Clone, Debug)]
pub struct DimensionCollector {
    recent_window: usize,
}

impl Default for DimensionCollector {
    fn default() -> Self {
        Self {
            recent_window: DEFAULT_RECENT_WINDOW,
        }
    }
}

impl DimensionCollector {
    /// Construct a collector with a caller-supplied sliding-window size for the
    /// `recent_*` signal counts. Smaller windows make the picker more reactive
    /// to the latest tool call; larger windows smooth over turn-by-turn noise.
    pub fn with_recent_window(recent_window: usize) -> Self {
        Self { recent_window }
    }

    /// Returns the configured `recent_*` sliding-window size.
    pub fn recent_window(&self) -> usize {
        self.recent_window
    }

    /// Extracts the tool-result signal and stores it on the request context.
    pub async fn process(
        &self,
        ctx: &mut ProxyContext,
        request: ChatRequest,
    ) -> Result<ChatRequest> {
        let tool_signal = extract_tool_signals_with_window(&request, self.recent_window);
        ctx.insert::<ToolResultSignal>(tool_signal);
        Ok(request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn stamps_tool_result_signal_into_proxy_context() {
        let collector = DimensionCollector::default();
        let request = ChatRequest::openai_chat(json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        }));

        let mut ctx = ProxyContext::new();
        collector
            .process(&mut ctx, request)
            .await
            .expect("process ok");

        assert!(ctx.get::<ToolResultSignal>().is_some());
    }
}
