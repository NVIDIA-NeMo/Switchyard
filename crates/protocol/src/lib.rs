// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![warn(missing_docs)]

//! Shared request/response protocol for libsy — the vocabulary an [`Algorithm`] reasons
//! over, decoupled from libsy's orchestration.
//!
//! This crate owns Switchyard's neutral conversation IR: [`LlmRequest`] (model, messages,
//! tools, sampling, …), the buffered [`AggLlmResponse`] (outputs, usage, …), and its
//! streaming counterpart [`LlmResponseChunk`]; the [`Request`]/[`Response`] envelope that
//! pairs them with correlation [`Metadata`]; plus the wire-[`format` module](crate::format)
//! identifiers translation keys off. `switchyard-translation` re-exports the conversation
//! and format modules, together with the most commonly used stream types. The IR
//! carries no bare `prompt`/`completion`; the [`text_request`] / [`prompt_text`] /
//! [`text_response`] / [`completion_text`] helpers bridge to and from plain text for the
//! common single-turn case.
//!
//! The streamed-response type itself — a live stream of chunks *or* the terminal
//! aggregate — is the [`LlmResponse`] enum; it owns a `futures::Stream`, so it is the one
//! non-`Clone`, non-data type in this crate.
//!
//! ## Example
//!
//! ```
//! use switchyard_protocol::{LlmRequest, Message, Role};
//!
//! let request = LlmRequest {
//!     model: Some("provider/model".into()),
//!     messages: vec![Message::text(Role::User, "Explain tail latency")],
//!     ..LlmRequest::default()
//! };
//!
//! assert_eq!(request.messages.len(), 1);
//! ```
//!
//! [`Algorithm`]: https://docs.rs/switchyard-libsy/latest/switchyard_libsy/trait.Algorithm.html

pub mod client;
pub mod envelope;
pub mod format;
pub mod llm;
pub mod metadata;
pub mod stream;

pub use client::*;
pub use envelope::*;
pub use format::*;
pub use llm::*;
pub use metadata::*;
pub use stream::*;

/// Builds a single-turn request: one user message carrying `prompt`, for `model`.
///
/// This deliberately creates only the common text shape. Construct [`LlmRequest`]
/// directly for instructions, tools, multimodal content, or sampling controls.
pub fn text_request(model: Option<String>, prompt: impl Into<String>) -> LlmRequest {
    LlmRequest {
        model,
        messages: vec![Message::text(Role::User, prompt)],
        ..LlmRequest::default()
    }
}

/// Returns a lossy text view of all user messages, joined by newlines.
///
/// Only text and refusal blocks are included. Instructions, tool content,
/// reasoning, and media are omitted. Returns an empty string when no user text exists.
pub fn prompt_text(request: &LlmRequest) -> String {
    request
        .messages
        .iter()
        .filter(|message| message.role == Role::User)
        .filter_map(|message| message.text_content("\n"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Builds a single-turn response: one assistant message carrying `completion`, for `model`.
///
/// Construct [`AggLlmResponse`] directly when usage, tools, reasoning, or multiple
/// output items must be represented.
pub fn text_response(model: Option<String>, completion: impl Into<String>) -> AggLlmResponse {
    AggLlmResponse {
        model,
        outputs: vec![ResponseOutput {
            role: Role::Assistant,
            content: vec![ContentBlock::Text {
                text: completion.into(),
            }],
            stop_reason: None,
        }],
        ..AggLlmResponse::default()
    }
}

/// Returns a lossy text view of the first assistant output.
///
/// Only text blocks from the first output are concatenated. Refusals, reasoning,
/// tools, media, and additional outputs are omitted. Returns an empty string when
/// no such text exists.
pub fn completion_text(response: &AggLlmResponse) -> String {
    response
        .outputs
        .first()
        .map(|output| {
            output
                .content
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<String>()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_round_trips_prompt_text() {
        let req = text_request(Some("m".to_string()), "hello world");
        assert_eq!(req.model.as_deref(), Some("m"));
        assert_eq!(prompt_text(&req), "hello world");
    }

    #[test]
    fn response_round_trips_completion_text() {
        let resp = text_response(None, "the answer");
        assert_eq!(completion_text(&resp), "the answer");
    }

    #[test]
    fn empty_text_helpers_are_empty_strings() {
        assert_eq!(prompt_text(&LlmRequest::default()), "");
        assert_eq!(completion_text(&AggLlmResponse::default()), "");
    }
}
