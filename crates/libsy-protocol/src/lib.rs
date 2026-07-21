// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared request/response protocol for libsy — the vocabulary an [`Algorithm`] reasons
//! over, decoupled from libsy's orchestration.
//!
//! This crate owns Switchyard's neutral conversation IR: [`LlmRequest`] (model, messages,
//! tools, sampling, …), the buffered [`AggLlmResponse`] (outputs, usage, …), and its
//! streaming counterpart [`LlmResponseChunk`]; the [`Request`]/[`Response`] envelope that
//! pairs them with correlation [`Metadata`]; plus the wire-[`format`] identifiers
//! translation keys off. `switchyard-translation` re-exports the IR types under its own
//! `ConversationRequest` / `ConversationResponse` / `ConversationStreamEvent` names. The IR
//! carries no bare `prompt`/`completion`; the [`text_request`] / [`prompt_text`] /
//! [`text_response`] / [`completion_text`] helpers bridge to and from plain text for the
//! common single-turn case.
//!
//! The streamed-response type itself — a live stream of chunks *or* the terminal
//! aggregate — is the [`LlmResponse`] enum; it owns a `futures::Stream`, so it is the one
//! non-`Clone`, non-data type in this crate.
//!
//! [`Algorithm`]: https://docs.rs/libsy

pub mod client;
pub mod envelope;
pub mod format;
pub mod llm;
pub mod signals;
pub mod stream;

pub use client::*;
pub use envelope::*;
pub use format::*;
pub use llm::*;
pub use stream::*;

/// Build a single-turn request: one user message carrying `prompt`, for `model`.
pub fn text_request(model: Option<String>, prompt: impl Into<String>) -> LlmRequest {
    LlmRequest {
        model,
        messages: vec![Message::text(Role::User, prompt)],
        ..LlmRequest::default()
    }
}

/// The user's prompt text — the text of every user message, joined by newlines. Empty
/// when the request has no user text.
pub fn prompt_text(request: &LlmRequest) -> String {
    request
        .messages
        .iter()
        .filter(|message| message.role == Role::User)
        .filter_map(|message| message.text_content("\n"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Build a single-turn response: one assistant message carrying `completion`, for `model`.
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

/// The assistant's completion text — the text blocks of the first output, concatenated.
/// Empty when the response has no textual output.
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
