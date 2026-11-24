// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Anthropic Messages buffered and streaming codecs.

mod buffered;
mod stream;

pub use buffered::AnthropicMessagesCodec;
pub use stream::AnthropicMessagesStreamCodec;

// Legacy models accept temperature and top_p in 0.0..=1.0 and nonnegative integers for top_k.
// https://platform.claude.com/docs/en/api/messages/create?version=2023-06-01
// API version 2026-10-21 returns HTTP 400 if a request includes temperature, top_p, or top_k.
// https://platform.claude.com/docs/en/api/migration-2026-10-21#sampling-parameters-are-removed
pub(crate) const MAX_TEMPERATURE: f64 = 1.0;
pub(crate) const DEFAULT_TEMPERATURE: f64 = 1.0;
pub(crate) const DEFAULT_TOP_P: f64 = 1.0;
