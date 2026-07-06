// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Response-side context-signal collector.
//!
//! Thin adapter around
//! [`crate::dimension_collector::response::extract_response_signals`]. Stamps
//! the resulting [`ResponseSignals`] into `ProxyContext` so downstream
//! stage_router / escalation logic can read structured response-quality
//! flags without re-parsing the wire body.

use switchyard_core::{ChatResponse, ProxyContext, Result};

use crate::dimension_collector::response::{extract_response_signals, ResponseSignals};

/// Populates `ProxyContext` with [`ResponseSignals`] derived from the
/// buffered response body.
///
/// Stamps nothing for streaming responses — those can't be introspected
/// without consuming the stream. Consumers that read
/// `ctx.get::<ResponseSignals>()` and find `None` after this processor
/// ran should treat that as "stream / not-yet-checked," not as
/// "response was acceptable."
#[derive(Clone, Copy, Debug, Default)]
pub struct ResponseSignalCollector;

impl ResponseSignalCollector {
    /// Extracts response-side signals from buffered responses and leaves streams untouched.
    pub async fn process(
        &self,
        ctx: &mut ProxyContext,
        response: ChatResponse,
    ) -> Result<ChatResponse> {
        // Only buffered responses carry an inspectable body. Streams pass
        // through unchanged with no signals stamped.
        if response.body().is_some() {
            let signals = extract_response_signals(&response);
            ctx.insert::<ResponseSignals>(signals);
        }
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension_collector::response::ResponseFlag;
    use serde_json::json;

    use switchyard_core::Result;

    #[tokio::test]
    async fn stamps_response_signals_for_buffered_response() -> Result<()> {
        let collector = ResponseSignalCollector;
        let response = ChatResponse::openai_completion(json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":"  // malformed
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }));

        let mut ctx = ProxyContext::new();
        let _ = collector.process(&mut ctx, response).await?;

        let Some(signals) = ctx.get::<ResponseSignals>() else {
            panic!("ResponseSignalCollector did not stamp ResponseSignals onto ctx");
        };
        assert!(signals.contains(ResponseFlag::MalformedToolCallJson));
        Ok(())
    }

    #[tokio::test]
    async fn stamps_empty_signals_for_clean_response() -> Result<()> {
        let collector = ResponseSignalCollector;
        let response = ChatResponse::openai_completion(json!({
            "choices": [{
                "message": { "content": "looks good" },
                "finish_reason": "stop"
            }]
        }));

        let mut ctx = ProxyContext::new();
        let _ = collector.process(&mut ctx, response).await?;

        let Some(signals) = ctx.get::<ResponseSignals>() else {
            panic!("ResponseSignalCollector did not stamp ResponseSignals onto ctx");
        };
        assert!(!signals.has_failures());
        Ok(())
    }
}
