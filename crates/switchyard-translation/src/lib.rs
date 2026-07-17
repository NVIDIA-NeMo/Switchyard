// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure Rust translation engine for Switchyard.
//!
//! The crate translates provider wire formats through a neutral Switchyard
//! conversation IR. It intentionally has no dependency on provider SDKs, HTTP
//! servers, Python objects, or FFI bindings.

pub mod codecs;
pub mod diagnostic;
pub mod engine;
pub mod error;
mod helpers;
pub mod policy;
mod sse;
pub mod stream;
pub mod util;

pub use switchyard_protocol::stream::LlmResponseChunk;
pub use switchyard_protocol::stream::LlmResponseStream;
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
    normalize_anthropic_tool_use_ids, sanitize_anthropic_tool_use_id, PRESERVATION_METADATA_KEY,
};
