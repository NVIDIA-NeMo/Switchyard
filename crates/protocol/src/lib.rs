// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![warn(missing_docs)]

//! Provider-neutral request, response, streaming, routing, and metadata types shared by
//! Switchyard algorithms, clients, and translation codecs. This crate defines contracts;
//! it does not route requests, translate wire payloads, or perform network calls.
//!
//! ## Setup
//!
//! ```toml
//! [dependencies]
//! switchyard-protocol = { git = "https://github.com/NVIDIA-NeMo/Switchyard.git" }
//! serde_json = "1"
//! ```
//!
//! ## Simple request
//!
//! ```
//! use switchyard_protocol::{ContentBlock, LlmRequest, Message, Role};
//!
//! let request = LlmRequest {
//!     model: Some("provider/model".into()),
//!     messages: vec![Message {
//!         role: Role::User,
//!         content: vec![ContentBlock::Text {
//!             text: "Explain tail latency".into(),
//!         }],
//!     }],
//!     ..LlmRequest::default()
//! };
//!
//! assert_eq!(request.model.as_deref(), Some("provider/model"));
//! assert_eq!(request.messages.len(), 1);
//! ```
//!
//! ## Detailed request
//!
//! Use [`Request`] when routing also needs correlation metadata, and construct
//! [`LlmRequest`] directly for instructions, tools, and generation controls:
//!
//! ```
//! use serde_json::json;
//! use switchyard_protocol::{
//!     ContentBlock, InstructionBlock, LlmRequest, Message, Metadata, OutputParams,
//!     Request, Role, SamplingParams, ToolChoice, ToolDefinition,
//! };
//!
//! let request = Request {
//!     llm_request: LlmRequest {
//!         model: Some("provider/model".into()),
//!         instructions: vec![InstructionBlock {
//!             role: Role::System,
//!             content: vec![ContentBlock::Text {
//!                 text: "Answer with concise operational guidance.".into(),
//!             }],
//!         }],
//!         messages: vec![Message::text(Role::User, "Why is p99 latency rising?")],
//!         tools: vec![ToolDefinition {
//!             name: "lookup_metric".into(),
//!             description: Some("Read one service metric".into()),
//!             parameters: json!({
//!                 "type": "object",
//!                 "properties": { "name": { "type": "string" } },
//!                 "required": ["name"]
//!             }),
//!             strict: Some(true),
//!         }],
//!         tool_choice: Some(ToolChoice::Auto),
//!         sampling: SamplingParams {
//!             temperature: Some(0.2),
//!             ..SamplingParams::default()
//!         },
//!         output: OutputParams {
//!             max_output_tokens: Some(512),
//!             ..OutputParams::default()
//!         },
//!         stream: true,
//!         ..LlmRequest::default()
//!     },
//!     metadata: Some(Metadata {
//!         session_id: Some("session-42".into()),
//!         correlation_id: Some("request-7".into()),
//!         ..Metadata::default()
//!     }),
//!     ..Request::default()
//! };
//!
//! assert_eq!(request.llm_request.tools[0].name, "lookup_metric");
//! assert_eq!(
//!     request
//!         .metadata
//!         .as_ref()
//!         .and_then(|metadata| metadata.session_id.as_deref()),
//!     Some("session-42")
//! );
//! ```
//!
//! [`LlmResponse`] represents either a buffered [`AggLlmResponse`] or a
//! single-consumption stream of [`LlmResponseChunk`]s. See each type's documentation for
//! aggregation and preservation behavior.

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
