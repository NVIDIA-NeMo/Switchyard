// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LLM-facing type names used by libsy algorithms and clients.

pub use switchyard_translation::{
    ContentBlock as LlmContentBlock, ConversationRequest as LlmRequest,
    ConversationResponse as LlmResponse, Message as LlmMessage,
    ResponseOutput as LlmResponseOutput, Role as LlmRole,
};
