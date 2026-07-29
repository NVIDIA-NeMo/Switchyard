// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-side intake processor.

use crate::{ChatRequest, ProxyContext, Result};
use serde_json::Value;

use crate::intake::context::{IntakeRequestState, RequestMetadata};
use crate::intake::payload::now_millis;

/// Captures request-side intake state for the response processor.
#[derive(Clone, Copy, Debug, Default)]
pub struct IntakeRequestProcessor;

impl IntakeRequestProcessor {
    /// Captures request metadata needed for response-side intake emission.
    pub async fn process(
        &self,
        ctx: &mut ProxyContext,
        request: ChatRequest,
    ) -> Result<ChatRequest> {
        let metadata = ctx.get::<RequestMetadata>().cloned().unwrap_or_default();
        let client_opt_in = metadata
            .intake
            .enabled
            .or_else(|| extract_store_toggle(&request));
        let skip = client_opt_in != Some(true);
        let request_snapshot = (!skip).then(|| request.clone());
        ctx.insert(IntakeRequestState {
            started_at_ms: now_millis(),
            inbound_format: request.request_type(),
            session_id: metadata.session_id,
            skip,
            request_snapshot,
        });
        Ok(request)
    }
}

// The OpenAI `store` flag is treated as an intake opt-in when headers are absent.
fn extract_store_toggle(request: &ChatRequest) -> Option<bool> {
    match request.body() {
        Value::Object(object) => object.get("store").and_then(Value::as_bool),
        _ => None,
    }
}
