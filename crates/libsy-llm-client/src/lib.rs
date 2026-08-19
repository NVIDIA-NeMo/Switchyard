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
//! [`run()`] pairs the client with a libsy algorithm: it drives
//! [`switchyard_libsy::Algorithm::run_stream`], serves routing-time calls, and makes the terminal
//! answer call from the routing outcome when needed. A host that just wants the answer does not
//! have to drive the step stream itself.

pub mod backend;
pub mod client;
pub mod error;
pub mod metrics;
mod observability;
mod observation;
pub mod raw;
mod responses_reasoning;
pub mod run;

pub use backend::{Backend, DEFAULT_MAX_RETRIES, HttpBackendConfig};
pub use client::{ModelConfig, TranslatingLlmClient};
pub use error::{LlmClientError, Result};
pub use observation::{LlmCallObservation, RunObservation, RunObserver};
pub use raw::RawResponse;
pub use responses_reasoning::ResponsesReasoningPolicy;
pub use run::{ClientRouter, run};
pub use switchyard_translation::RawEventStream;

/// Registers process-wide compatibility gauges with the global meter provider.
pub fn initialize_metrics() {
    metrics::initialize();
}
