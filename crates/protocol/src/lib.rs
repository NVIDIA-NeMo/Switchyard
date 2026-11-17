// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

pub mod category;
pub mod client;
pub mod codex_namespaces;
pub mod decision;
pub mod envelope;
pub mod format;
pub mod infra;
pub mod llm;
pub mod metadata;
pub mod model_id;
pub mod stream;

pub use category::*;
pub use client::*;
pub use decision::*;
pub use envelope::*;
pub use format::*;
pub use infra::*;
pub use llm::*;
pub use metadata::*;
pub use model_id::*;
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

/// Builds a single-turn response: one assistant message carrying `completion`, for `model`.
///
/// Construct [`AggLlmResponse`] directly when usage, tools, reasoning, or multiple
/// output items must be represented.
pub fn text_response(model: Option<String>, completion: impl Into<String>) -> AggLlmResponse {
    AggLlmResponse {
        model,
        outputs: vec![ResponseOutput {
            url_citations: Vec::new(),
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
    fn text_request_builds_one_user_message() {
        let req = text_request(Some("m".to_string()), "hello world");
        assert_eq!(req.model.as_deref(), Some("m"));
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, Role::User);
        assert_eq!(
            req.messages[0].text_content("\n").as_deref(),
            Some("hello world")
        );
    }

    #[test]
    fn response_round_trips_completion_text() {
        let resp = text_response(None, "the answer");
        assert_eq!(completion_text(&resp), "the answer");
    }

    #[test]
    fn empty_completion_text_is_empty_string() {
        assert_eq!(completion_text(&AggLlmResponse::default()), "");
    }
}
