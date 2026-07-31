// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Adding text to a request on its way to the model it was routed to.
//!
//! Two shapes, both target-agnostic — any algorithm routing between named
//! targets can use them, and neither writes anything back into the caller's
//! conversation:
//!
//! * [`append_note`] — a one-off note in the conversation itself, for telling
//!   the model something about *this* turn.
//! * [`SystemPromptProcessor`] — standing instructions per target, applied on
//!   every turn that target serves.
//!
//! Which text, and when, is the caller's policy; this module only knows how to
//! place it so the provider accepts it and the prompt cache survives.

use std::collections::BTreeMap;

use async_trait::async_trait;
use switchyard_protocol::{ContentBlock, InstructionBlock, Message, Request, Role};

use crate::core::processor::{Event, Processor};
use crate::Result;

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
}

/// System prompts keyed by routing target. A target left unset is routed
/// untouched.
#[derive(Clone, Debug, Default)]
pub struct TargetPrompts {
    by_target: BTreeMap<String, String>,
}

impl TargetPrompts {
    /// Hand `target` this prompt on every turn it serves.
    pub fn with(mut self, target: impl Into<String>, prompt: impl Into<String>) -> Self {
        self.by_target.insert(target.into(), prompt.into());
        self
    }

    /// The prompt configured for `target`, if any.
    pub fn get(&self, target: &str) -> Option<&str> {
        self.by_target.get(target).map(String::as_str)
    }

    /// Whether any target has a prompt, so a caller can skip wiring the
    /// processor when none does.
    pub fn is_empty(&self) -> bool {
        self.by_target.is_empty()
    }
}

/// Prepends the routed target's system prompt to the outbound request.
pub struct SystemPromptProcessor {
    prompts: TargetPrompts,
}

impl SystemPromptProcessor {
    /// Hand each target the prompt configured for it.
    pub fn new(prompts: TargetPrompts) -> Self {
        Self { prompts }
    }
}

#[async_trait]
impl<S: Send> Processor<S> for SystemPromptProcessor {
    async fn process(&self, _state: &mut S, event: Event<'_>) -> Result<()> {
        // The decision event carries both the routing outcome and the outbound request,
        // so the target is read straight off it — whichever classifier picked it, and
        // with nothing kept between turns.
        let Event::Decision { request, decision } = event else {
            return Ok(());
        };
        let Some(prompt) = self.prompts.get(decision.selected_model()) else {
            return Ok(());
        };
        // Ahead of the client's own instructions, so this framing is what the
        // model reads first.
        request.llm_request.instructions.insert(
            0,
            InstructionBlock {
                role: Role::System,
                content: vec![ContentBlock::Text {
                    text: prompt.to_string(),
                }],
            },
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use switchyard_protocol::{text_request, LlmRequest, ToolResult};

    const NOTE: &str = "recovering from an error";
    const STRONG_PROMPT: &str = "diagnose before you edit";
    const WEAK_PROMPT: &str = "follow the settled plan";

    fn request_with(messages: Vec<Message>) -> Request {
        Request {
            llm_request: LlmRequest {
                messages,
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: None,
        }
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
            llm_request: text_request(Some("auto".to_string()), "fix the build"),
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
    }

    /// A decision routed to `target`.
    struct RoutedTo(&'static str);
    impl switchyard_protocol::Decision for RoutedTo {
        fn selected_model(&self) -> &str {
            self.0
        }
        fn reasoning(&self) -> Option<&str> {
            None
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    /// The instruction text the request carries.
    fn instructions(request: &Request) -> Vec<String> {
        request
            .llm_request
            .instructions
            .iter()
            .filter_map(|block| {
                block.content.iter().find_map(|content| match content {
                    ContentBlock::Text { text } => Some(text.clone()),
                    _ => None,
                })
            })
            .collect()
    }

    /// Runs one outbound request routed to `target` through `processor`.
    async fn run(processor: &SystemPromptProcessor, target: &'static str) -> Result<Request> {
        let mut request = Request::default();
        processor
            .process(
                &mut (),
                Event::Decision {
                    request: &mut request,
                    decision: &RoutedTo(target),
                },
            )
            .await?;
        Ok(request)
    }

    fn prompts() -> TargetPrompts {
        TargetPrompts::default()
            .with("strong", STRONG_PROMPT)
            .with("weak", WEAK_PROMPT)
    }

    #[tokio::test]
    async fn each_target_gets_its_own_prompt() -> Result<()> {
        let processor = SystemPromptProcessor::new(prompts());
        for (target, expected) in [("strong", STRONG_PROMPT), ("weak", WEAK_PROMPT)] {
            assert_eq!(
                instructions(&run(&processor, target).await?),
                vec![expected]
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn an_unconfigured_target_is_left_untouched() -> Result<()> {
        // One target's prompt must not leak onto another, whatever ran before.
        let processor =
            SystemPromptProcessor::new(TargetPrompts::default().with("strong", STRONG_PROMPT));
        assert_eq!(
            instructions(&run(&processor, "strong").await?),
            vec![STRONG_PROMPT]
        );
        assert!(instructions(&run(&processor, "weak").await?).is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn the_prompt_leads_the_client_instructions() -> Result<()> {
        let processor = SystemPromptProcessor::new(prompts());
        let mut request = Request::default();
        request.llm_request.instructions.push(InstructionBlock {
            role: Role::System,
            content: vec![ContentBlock::Text {
                text: "you are a coding agent".to_string(),
            }],
        });

        processor
            .process(
                &mut (),
                Event::Decision {
                    request: &mut request,
                    decision: &RoutedTo("strong"),
                },
            )
            .await?;

        assert_eq!(
            instructions(&request),
            vec![STRONG_PROMPT, "you are a coding agent"]
        );
        Ok(())
    }

    #[tokio::test]
    async fn the_inbound_request_is_left_alone() -> Result<()> {
        // The inbound hook runs before the cascade has picked anything.
        let processor = SystemPromptProcessor::new(prompts());
        let mut request = Request::default();
        processor
            .process(&mut (), Event::Request(&mut request))
            .await?;
        assert!(instructions(&request).is_empty());
        Ok(())
    }
}
