// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Anthropic Messages buffered and streaming codecs.

mod buffered;
mod stream;

pub use buffered::AnthropicMessagesCodec;
pub use stream::AnthropicMessagesStreamCodec;

// Temperature range: 0.0..=1.0 for legacy models supporting this parameter.
// https://platform.claude.com/docs/en/api/messages/create?version=2023-06-01
// API version 2026-10-21 removes temperature; sending it returns HTTP 400.
// https://platform.claude.com/docs/en/api/migration-2026-10-21#sampling-parameters-are-removed
pub(crate) const MAX_TEMPERATURE: f64 = 1.0;
pub(crate) const DEFAULT_TEMPERATURE: f64 = 1.0;
