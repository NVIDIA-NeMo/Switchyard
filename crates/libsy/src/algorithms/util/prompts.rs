// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Adding text to a request on its way to the model it was routed to.
//!
//! These helpers mutate the normalized request. A codec asked to encode for the format the
//! request arrived in replays the body captured at decode instead of reading that
//! request — so an addition that leaves exact replay in place never reaches the
//! model. This is not enforced: a future processor that mutates the request and
//! forgets the call reintroduces SWITCH-1224, silently and without a failing
//! test.

use switchyard_protocol::{ContentBlock, InstructionBlock, Message, Request, Role};

/// Appends `note` to the request as conversation text.
///
/// It joins the trailing user message when there is one, rather than opening a
/// turn of its own: Anthropic rejects two consecutive user messages, and a
/// `tool_result` must stay first within its message. A text block after it
/// satisfies both and keeps the addition a cache-safe suffix. Any other trailing
/// role — an empty conversation, or one ending on an assistant turn — takes a
/// fresh user message.
pub fn append_note(request: &mut Request, note: &str) {
    match request.llm_request.messages.last_mut() {
        Some(last) if last.role == Role::User => last.content.push(ContentBlock::Text {
            text: note.to_string(),
        }),
        _ => request
            .llm_request
            .messages
            .push(Message::text(Role::User, note)),
    }
    drop_exact_replay(request);
}

/// Gives up exact same-format replay for this turn.
///
/// A codec replays the preserved inbound body verbatim when the target format
/// matches the source, which is what keeps a same-format hop lossless. That body
/// predates anything added here, so leaving it in place would encode the request
/// as it arrived and silently drop the addition. Dropping it sends the codec down
/// its normal path, which encodes from the request itself.
///
/// Every stored body goes, not just the one for the inbound format: preservation
/// also carries bodies embedded by earlier hops, and the addition is missing from
/// all of them equally.
///
/// Call this from any new code that mutates the request. Nothing checks that you
/// have.
pub(crate) fn drop_exact_replay(request: &mut Request) {
    request.llm_request.preservation.requests.clear();
}

/// Prepends a system prompt and disables exact replay so the edit reaches the provider.
pub(crate) fn prepend_system_prompt(request: &mut Request, prompt: &str) {
    request.llm_request.instructions.insert(
        0,
        InstructionBlock {
            role: Role::System,
            content: vec![ContentBlock::Text {
                text: prompt.to_string(),
            }],
        },
    );
    drop_exact_replay(request);
}

#[cfg(test)]
mod tests {
    use super::*;
    use switchyard_protocol::{LlmRequest, ToolResult, text_request};

    const NOTE: &str = "recovering from an error";

    /// Every test request carries the exact inbound body a codec keeps for
    /// same-format replay, so each assertion below also says what happens to it.
    fn request_with(messages: Vec<Message>) -> Request {
        Request {
            llm_request: LlmRequest {
                messages,
                preservation: preserved_body(),
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: None,
        }
    }

    fn preserved_body() -> switchyard_protocol::PreservationMetadata {
        let mut preservation = switchyard_protocol::PreservationMetadata::default();
        preservation.requests.insert(
            "openai_chat".into(),
            serde_json::json!({"model": "weak", "messages": [{"role": "user", "content": "hi"}]}),
        );
        preservation
    }

    fn replays_exactly(request: &Request) -> bool {
        !request.llm_request.preservation.requests.is_empty()
    }

    #[test]
    fn a_note_joins_a_trailing_user_turn_after_its_tool_result() {
        // The shape a coding-agent turn actually arrives in: the tool result
        // leads the trailing user message, so the note has to follow it.
        let tool_result = ContentBlock::ToolResult(ToolResult {
            tool_call_id: "call_1".to_string(),
            content: vec![ContentBlock::Text {
                text: "exit 1".to_string(),
            }],
            is_error: Some(true),
        });
        let mut request = request_with(vec![Message {
            role: Role::User,
            content: vec![tool_result.clone()],
        }]);

        append_note(&mut request, NOTE);

        let messages = &request.llm_request.messages;
        assert_eq!(messages.len(), 1, "no second consecutive user turn");
        assert_eq!(
            messages[0].content,
            vec![
                tool_result,
                ContentBlock::Text {
                    text: NOTE.to_string()
                }
            ]
        );
    }

    #[test]
    fn a_note_opens_a_user_turn_after_an_assistant_turn() {
        let mut request = request_with(vec![Message::text(Role::Assistant, "done")]);

        append_note(&mut request, NOTE);

        let messages = &request.llm_request.messages;
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1].role, Role::User);
        assert_eq!(messages[1].text_content(""), Some(NOTE.to_string()));
    }

    #[test]
    fn a_note_leaves_the_rest_of_the_conversation_untouched() {
        let mut request = Request {
            llm_request: LlmRequest {
                preservation: preserved_body(),
                ..text_request(Some("auto".to_string()), "fix the build")
            },
            raw_request: None,
            metadata: None,
        };

        append_note(&mut request, NOTE);

        let trail: Vec<String> = request
            .llm_request
            .messages
            .iter()
            .filter_map(|message| message.text_content("|"))
            .collect();
        assert_eq!(trail, vec![format!("fix the build|{NOTE}")]);
        assert!(
            !replays_exactly(&request),
            "a same-format hop would replay the body captured before the note"
        );
    }
}
