// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Neutral conversation IR used for loss-aware provider translation.
//!
//! The type definitions now live in `libsy-protocol` (Switchyard's shared IR
//! vocabulary); this module re-exports them so translation keeps its existing
//! `crate::ir::*` paths and public surface. The two top-level types are named
//! `LlmRequest`/`LlmResponse` in the protocol crate and re-exported here under
//! their long-standing `ConversationRequest`/`ConversationResponse` names.

pub use libsy_protocol::ir::*;
pub use libsy_protocol::ir::{
    AggLlmResponse as ConversationResponse, LlmRequest as ConversationRequest,
};

/// Returns true when `name` is a message role recognized by at least one
/// supported provider API (OpenAI Chat, OpenAI Responses, or Anthropic
/// Messages). Codecs use this to distinguish a genuinely-unsupported role —
/// which is rejected on request decode to preserve the provider contract —
/// from a known role that a given codec maps to a default. `function` is
/// included because it is a legacy OpenAI Chat role that older clients may
/// still send and which has always been coerced to `user` here.
pub(crate) fn is_known_role_name(name: &str) -> bool {
    matches!(
        name,
        "system" | "developer" | "user" | "assistant" | "tool" | "function"
    )
}
