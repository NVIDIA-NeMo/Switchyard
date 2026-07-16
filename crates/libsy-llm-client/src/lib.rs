// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP LLM client that speaks Switchyard's neutral IR directly.
//!
//! [`TranslatingLlmClient`] maps a model name (and the wire format resolved from
//! the request) to a [`Backend`],
//! encodes a [`switchyard_protocol::Request`] to that backend's wire format via
//! `switchyard-translation`, applies auth and forwards caller headers, makes the
//! HTTP call with a shared [`reqwest::Client`], and decodes the wire response
//! back to a [`switchyard_protocol::Response`] — supporting both buffered and
//! streamed responses.
//!
//! The crate depends only on `libsy-protocol` and `switchyard-translation`. Neutral
//! IR encode/decode — including SSE stream decoding — lives in
//! `switchyard-translation`; this crate is the HTTP transport around it. The
//! context-overflow detection in [`mod@error`] is vendored from
//! `switchyard-components` (whose copy is crate-private and unavailable here); a
//! future refactor could promote it to a shared location.

pub mod backend;
pub mod client;
pub mod error;
pub mod raw;

pub use backend::{Backend, HttpBackendConfig};
pub use client::{ModelConfig, TranslatingLlmClient};
pub use error::{LlmClientError, Result};
pub use raw::RawResponse;
pub use switchyard_translation::RawEventStream;
