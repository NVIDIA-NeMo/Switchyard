// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in backend implementations.

pub mod anthropic;
mod common;
mod context_overflow;
pub mod multi;
pub mod openai;
mod selection;

pub use anthropic::AnthropicNativeBackend;
pub use multi::{LlmTargetBackend, MultiLlmBackend};
pub use openai::{OpenAiNativeBackend, OpenAiPassthroughBackend};
pub use selection::{BackendSelection, BackendSelectionReason};
