// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Gate signals, split by the event that carries them: conversation counts
//! fold in on [`Event::Request`], the generated turn's shape on
//! [`Event::ModelResponse`], and the trigger classifier reads both.

use async_trait::async_trait;

use crate::Result;
use crate::algorithms::util::tool_signals::ToolSignals;
use crate::core::processor::{Event, Processor};

use super::turn::{has_tool_use, visible_text};

/// Facts the trigger classifier reads, keyed by the event that produced them.
#[derive(Default)]
pub(super) struct GateSignals {
    /// Conversation counts from the shared [`ToolSignals`] extraction.
    pub(super) conversation: ToolSignals,
    /// Shape of the turn the executor just generated.
    pub(super) turn: TurnSignals,
}

/// The generated turn's reviewable shape.
#[derive(Default)]
pub(super) struct TurnSignals {
    /// The turn carries tool use (a `ToolUse` stop reason or any tool-call block).
    pub(super) has_tool_use: bool,
    /// The turn's visible text; `None` when it has no text blocks.
    pub(super) visible_text: Option<String>,
}

/// Fills [`GateSignals`], each event writing its own side.
pub(super) struct GateSignalProcessor;

#[async_trait]
impl Processor<GateSignals> for GateSignalProcessor {
    async fn process(&self, state: &mut GateSignals, event: Event<'_>) -> Result<()> {
        match event {
            Event::Request { request, .. } => {
                state.conversation = ToolSignals::from_request(request, None);
            }
            Event::ModelResponse(agg) => {
                state.turn = TurnSignals {
                    has_tool_use: has_tool_use(agg),
                    visible_text: visible_text(agg),
                };
            }
            Event::Decision { .. } => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use switchyard_protocol::{
        AggLlmResponse, ContentBlock, LlmRequest, Message, ModelId, Request, ResponseOutput, Role,
        ToolCall, ToolResult,
    };

    fn conversation_request() -> Request {
        Request {
            llm_request: LlmRequest {
                messages: vec![
                    Message::text(Role::User, "build X"),
                    Message::text(Role::Assistant, "working"),
                    Message {
                        role: Role::User,
                        content: vec![ContentBlock::ToolResult(ToolResult {
                            tool_call_id: "t1".to_string(),
                            content: vec![ContentBlock::Text {
                                text: "ok".to_string(),
                            }],
                            is_error: None,
                        })],
                    },
                ],
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: None,
        }
    }

    fn tool_call_agg() -> AggLlmResponse {
        AggLlmResponse {
            outputs: vec![ResponseOutput {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::Text {
                        text: "running a tool".to_string(),
                    },
                    ContentBlock::ToolCall(ToolCall {
                        id: "t1".to_string(),
                        name: "bash".to_string(),
                        arguments: serde_json::json!({}),
                    }),
                ],
                stop_reason: None,
            }],
            ..AggLlmResponse::default()
        }
    }

    #[tokio::test]
    async fn each_event_fills_its_own_signal_side() -> Result<()> {
        let processor = GateSignalProcessor;
        let mut state = GateSignals::default();
        let mut request = conversation_request();

        processor
            .process(
                &mut state,
                Event::Request {
                    request: &mut request,
                    driver: None,
                },
            )
            .await?;
        assert_eq!(state.conversation.tool_result_count, 1);
        assert_eq!(state.conversation.assistant_turn_count, 1);
        assert!(
            !state.turn.has_tool_use,
            "the request never sets turn shape"
        );

        let agg = tool_call_agg();
        processor
            .process(&mut state, Event::ModelResponse(&agg))
            .await?;
        assert!(state.turn.has_tool_use);
        assert_eq!(state.turn.visible_text.as_deref(), Some("running a tool"));
        assert_eq!(
            state.conversation.tool_result_count, 1,
            "the response never rewrites conversation counts"
        );
        Ok(())
    }

    #[tokio::test]
    async fn the_decision_event_is_a_no_op() -> Result<()> {
        let processor = GateSignalProcessor;
        let mut state = GateSignals::default();
        let mut request = conversation_request();
        let selected = ModelId::from("executor");

        processor
            .process(
                &mut state,
                Event::Decision {
                    request: &mut request,
                    selected_model_id: &selected,
                },
            )
            .await?;

        assert_eq!(state.conversation.tool_result_count, 0);
        assert!(state.turn.visible_text.is_none());
        Ok(())
    }
}
