// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tool-result context signals — thin adapter over libsy's extractor.
//!
//! The extraction logic lives in [`switchyard_libsy::ToolSignals`]. This module
//! bridges the crate's [`ChatRequest`] (a format-tagged JSON body) to libsy's
//! [`switchyard_protocol::Request`] (raw body + wire-format metadata) so the two
//! request models share one implementation.

use switchyard_core::{ChatRequest, ChatRequestType};
use switchyard_protocol::{Metadata, Request, WireFormat};

/// The tool-signal output type. Re-exported from libsy so downstream consumers
/// (the `ToolResultSignal` stamped on `ProxyContext`) see a single type.
pub use switchyard_libsy::{ToolSignals as ToolResultSignal, DEFAULT_RECENT_WINDOW};

/// Adapt a format-tagged [`ChatRequest`] to a [`switchyard_protocol::Request`]
/// carrying the raw body and its wire format, which is all libsy's extractor reads.
fn to_protocol_request(request: &ChatRequest) -> Request {
    let wire_format = match request.request_type() {
        ChatRequestType::OpenAiChat => WireFormat::OpenAiChat,
        ChatRequestType::Anthropic => WireFormat::AnthropicMessages,
        ChatRequestType::OpenAiResponses => WireFormat::OpenAiResponses,
    };
    Request {
        raw_request: Some(request.body().clone()),
        metadata: Some(Metadata {
            wire_format: Some(wire_format),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Extract tool-execution signals using the default `recent_*` window.
pub fn extract_tool_signals(request: &ChatRequest) -> ToolResultSignal {
    ToolResultSignal::from_request(&to_protocol_request(request), None)
}

/// Extract tool-execution signals with a caller-supplied `recent_*` window.
pub fn extract_tool_signals_with_window(
    request: &ChatRequest,
    recent_window: usize,
) -> ToolResultSignal {
    ToolResultSignal::from_request(&to_protocol_request(request), Some(recent_window))
}
