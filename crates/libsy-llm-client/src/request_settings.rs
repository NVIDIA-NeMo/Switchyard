// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider request preparation for hosts that own their model clients.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;
use switchyard_protocol::{LlmRequest, ModelId, WireFormat};

use crate::error::Result;

/// Request-body settings independent of HTTP transport, credentials, and retries.
pub struct RequestSettings {
    /// Provider format used to encode the normalized request.
    pub wire_format: WireFormat,
    /// Default top-level fields; explicit request fields take precedence.
    pub extra_body: BTreeMap<String, Value>,
    /// Fields removed before defaults are applied.
    pub omit_body_fields: BTreeSet<String>,
    /// Forced OpenAI reasoning effort, applied after defaults. Ignored for Anthropic.
    pub reasoning_effort: Option<String>,
}

impl RequestSettings {
    /// Encodes a request for `model` using the native client's body preparation.
    /// Does not modify the request or send it. Encoding failures return `RequestEncoding`.
    pub fn prepare_request_body(&self, request: &LlmRequest, model: &ModelId) -> Result<Value> {
        crate::client::prepare_request_body(
            request,
            model,
            self.wire_format,
            &self.extra_body,
            &self.omit_body_fields,
            self.reasoning_effort.as_deref(),
        )
    }
}
