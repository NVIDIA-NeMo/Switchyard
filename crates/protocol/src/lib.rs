// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

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

#[cfg(test)]
mod tests {
    use super::*;

    fn text_request(model: Option<String>, prompt: impl Into<String>) -> LlmRequest {
        LlmRequest {
            model,
            messages: vec![Message::text(Role::User, prompt)],
            ..LlmRequest::default()
        }
    }

    fn prompt_text(request: &LlmRequest) -> String {
        request
            .messages
            .iter()
            .filter(|message| message.role == Role::User)
            .filter_map(|message| message.text_content("\n"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn text_response(model: Option<String>, completion: impl Into<String>) -> AggLlmResponse {
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

    fn completion_text(response: &AggLlmResponse) -> String {
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
