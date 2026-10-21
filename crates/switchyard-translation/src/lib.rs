// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![doc = include_str!("../README.md")]

pub mod codecs;
pub(crate) mod codex_custom_tools;
pub(crate) mod codex_namespaces;
pub mod diagnostic;
pub mod engine;
pub mod error;
mod helpers;
pub mod policy;
mod sse;
pub mod stream;
pub mod util;

pub use codecs::responses::is_anthropic_thinking_envelope;
pub use switchyard_protocol::stream::{
    LlmResponseChunk, LlmResponseStream, LlmResponseStreamEvent, LlmStreamError,
    ProviderStreamEvent,
};
pub use switchyard_protocol::{format, llm};

pub use diagnostic::*;
pub use engine::*;
pub use error::*;
pub use format::*;
pub use helpers::*;
pub use llm::*;
pub use policy::*;
pub use stream::*;
pub use util::{
    PRESERVATION_METADATA_KEY, normalize_anthropic_tool_use_ids, prepare_request_for_target,
    sanitize_anthropic_tool_use_id,
};
