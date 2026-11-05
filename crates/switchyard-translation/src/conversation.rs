// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module groups native Responses input items into turns for routing and other APIs.

use std::borrow::Cow;

use crate::{ContentBlock, InstructionBlock, LlmRequest, Message, Role, ToolCall, ToolResult};

/// A temporary conversation view groups native items without changing the request.
pub struct Conversation<'a> {
    /// Instructions include native inline system and developer messages.
    pub instructions: Cow<'a, [InstructionBlock]>,
    /// Reasoning and parallel tool calls share their original assistant turn.
    pub messages: Cow<'a, [Message]>,
}

/// This function groups the current content for routing and non-Responses APIs.
/// The request retains native item order and fields for Responses encoding.
/// Indexes in this view do not refer to the original request's messages.
pub fn conversation(request: &LlmRequest) -> Conversation<'_> {
    if !request
        .messages
        .iter()
        .any(|message| message.responses.is_some())
    {
        return Conversation {
            instructions: Cow::Borrowed(&request.instructions),
            messages: Cow::Borrowed(&request.messages),
        };
    }
    // A state reference can identify provider-held calls that are absent from this request.
    let fields = &request.extensions.fields;
    let stored_state = fields
        .get("previous_response_id")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|id| !id.is_empty())
        || fields.get("conversation").is_some_and(|conversation| {
            conversation
                .as_str()
                .or_else(|| conversation.get("id").and_then(serde_json::Value::as_str))
                .is_some_and(|id| !id.is_empty())
        });
    let mut instructions = request.instructions.clone();
    let mut turns = ConversationTurns::default();
    for message in &request.messages {
        // Inline tool definitions are already present in the request's tool catalog.
        if matches!(message.content.as_slice(), [ContentBlock::Unknown { raw, .. }]
            if raw.get("type").and_then(serde_json::Value::as_str) == Some("additional_tools"))
        {
            continue;
        }
        // Inline instructions leave pending reasoning and tool calls in the same turn.
        if matches!(message.role, Role::System | Role::Developer) {
            instructions.push(InstructionBlock {
                role: message.role,
                content: message.content.clone(),
            });
            continue;
        }
        match message.content.as_slice() {
            [ContentBlock::Reasoning { .. }] if message.role == Role::Assistant => {
                // Reasoning after tool outputs starts the next assistant turn.
                if !turns.outputs.is_empty() {
                    turns.flush_tools();
                }
                turns.reasoning.extend(message.content.clone());
            }
            [ContentBlock::ToolCall(call)] => {
                if !turns.outputs.is_empty() {
                    turns.flush_tools();
                }
                turns.calls.push(call.clone());
            }
            [ContentBlock::ToolResult(result)] => {
                if stored_state
                    || turns
                        .calls
                        .iter()
                        .any(|call| call.id == result.tool_call_id)
                {
                    turns.outputs.push(result.clone());
                } else {
                    let text = result
                        .content
                        .iter()
                        .filter_map(|block| match block {
                            ContentBlock::Text { text, .. } | ContentBlock::Refusal { text } => {
                                Some(text.as_str())
                            }
                            ContentBlock::Unknown { raw, .. } => raw.as_str(),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                    turns.push_message(Message::text(
                        Role::User,
                        format!("Tool result {}: {text}", result.tool_call_id),
                    ));
                }
            }
            _ => {
                let unknown_item =
                    matches!(message.content.as_slice(), [ContentBlock::Unknown { .. }])
                        && message.responses.as_ref().is_some_and(|source| {
                            source
                                .fields
                                .get("type")
                                .and_then(serde_json::Value::as_str)
                                .is_some_and(|kind| kind != "message")
                        });
                let mut message = message.clone();
                message.responses = None;
                if message.role == Role::Assistant && !unknown_item && turns.calls.is_empty() {
                    let mut content = std::mem::take(&mut turns.reasoning);
                    content.append(&mut message.content);
                    message.content = content;
                } else if turns.calls.is_empty() {
                    turns.flush_reasoning();
                }
                turns.push_message(message);
            }
        }
    }
    turns.flush_tools();
    turns.flush_reasoning();
    Conversation {
        instructions: Cow::Owned(instructions),
        messages: Cow::Owned(turns.messages),
    }
}

// These lists hold temporary clones while the view is built; the request owns the native history.
#[derive(Default)]
struct ConversationTurns {
    messages: Vec<Message>,
    calls: Vec<ToolCall>,
    outputs: Vec<ToolResult>,
    deferred: Vec<Message>,
    reasoning: Vec<ContentBlock>,
}

impl ConversationTurns {
    fn push_message(&mut self, message: Message) {
        if !self.calls.is_empty() && self.outputs.is_empty() {
            self.deferred.push(message);
            return;
        }
        self.flush_tools();
        self.messages.push(message);
    }

    fn flush_tools(&mut self) {
        if !self.calls.is_empty() {
            let mut content = std::mem::take(&mut self.reasoning);
            content.extend(
                std::mem::take(&mut self.calls)
                    .into_iter()
                    .map(ContentBlock::ToolCall),
            );
            self.messages.push(Message {
                role: Role::Assistant,
                content,
                responses: None,
            });
        }
        for result in std::mem::take(&mut self.outputs) {
            self.messages.push(Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult(result)],
                responses: None,
            });
        }
        self.messages.append(&mut self.deferred);
    }

    fn flush_reasoning(&mut self) {
        if !self.reasoning.is_empty() {
            let content = std::mem::take(&mut self.reasoning);
            self.push_message(Message {
                role: Role::Assistant,
                content,
                responses: None,
            });
        }
    }
}
