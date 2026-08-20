// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Holds one classifier verdict across the tool-call turns that follow a user message.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use switchyard_protocol::{ContentBlock, Message, ModelId, Request, Response, Role};

use super::decisive;
use crate::Result;
use crate::core::algorithm::Driver;
use crate::core::classifier::{Classification, Classifier};
use crate::core::state::{State, StateValue};

/// `State.extra` key holding the pinned target.
const PINNED_TARGET_KEY: &str = "classifier_pinned_target";

/// How often the classifier re-decides a session's target.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ClassifyTrigger {
    /// Judge every request, tool continuations included.
    #[default]
    EveryRequest,
    /// Judge each new user message and hold that target across the tool calls between.
    UserTurn,
    /// Judge once and reuse that target for the session.
    NewSession,
}

/// True when the user spoke. Anthropic carries tool results as a `Role::User` message, so
/// role alone cannot tell a human turn from a tool continuation. The same invariant is
/// encoded in the Anthropic codec's `message_is_tool_result_only`.
fn is_user_turn(message: &Message) -> bool {
    message.role == Role::User
        && !message
            .content
            .iter()
            .all(|block| matches!(block, ContentBlock::ToolResult(_)))
}

/// True when the conversation ends with the user speaking, so the agent is not mid-task.
fn starts_new_turn(messages: &[Message]) -> bool {
    messages.last().is_some_and(is_user_turn)
}

fn pinned_target(state: &State) -> Option<ModelId> {
    match state.extra.get(PINNED_TARGET_KEY) {
        Some(StateValue::String(target)) => Some(ModelId::new(target.clone())),
        _ => None,
    }
}

/// Pins the inner classifier's verdict until the user speaks again.
///
/// Without a session id there is no retained state, and every turn is classified.
pub(crate) struct TurnPin {
    inner: Arc<dyn Classifier<State>>,
}

impl TurnPin {
    pub(crate) fn new(inner: Arc<dyn Classifier<State>>) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl Classifier<State> for TurnPin {
    fn routing_tier(&self, selected_model_id: &ModelId) -> Option<&'static str> {
        self.inner.routing_tier(selected_model_id)
    }

    async fn score(
        &self,
        state: &mut State,
        request: &mut Request,
        driver: Option<&Driver>,
    ) -> Result<(Classification, Option<Response>)> {
        if let Some(target) = pinned_target(state)
            && !starts_new_turn(&request.llm_request.messages)
        {
            return Ok((decisive(&target), None));
        }

        let (classification, response) = self.inner.score(state, request, driver).await?;
        // An abstention clears the pin. Holding the old target would serve this turn from the
        // fall-through default while its tool continuations reused the previous turn's target.
        match classification.argmax(false)? {
            Some(score) => {
                state.extra.insert(
                    PINNED_TARGET_KEY.to_string(),
                    StateValue::String(score.target.as_str().to_string()),
                );
            }
            None => {
                state.extra.remove(PINNED_TARGET_KEY);
            }
        }
        Ok((classification, response))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use parking_lot::Mutex;
    use serde_json::Value;
    use switchyard_protocol::{LlmRequest, ToolCall, ToolResult};

    /// Returns queued verdicts in order, counting each consultation.
    struct RecordingClassifier {
        verdicts: Mutex<Vec<Option<&'static str>>>,
        consultations: Mutex<u32>,
    }

    impl RecordingClassifier {
        fn new(verdicts: Vec<Option<&'static str>>) -> Arc<Self> {
            Arc::new(Self {
                verdicts: Mutex::new(verdicts.into_iter().rev().collect()),
                consultations: Mutex::new(0),
            })
        }

        fn consultations(&self) -> u32 {
            *self.consultations.lock()
        }
    }

    #[async_trait]
    impl Classifier<State> for RecordingClassifier {
        async fn score(
            &self,
            _state: &mut State,
            _request: &mut Request,
            _driver: Option<&Driver>,
        ) -> Result<(Classification, Option<Response>)> {
            *self.consultations.lock() += 1;
            let verdict = self.verdicts.lock().pop().flatten();
            Ok((
                match verdict {
                    Some(target) => decisive(&ModelId::new(target)),
                    None => Classification::Scores(Vec::new()),
                },
                None,
            ))
        }
    }

    fn user(text: &str) -> Message {
        Message::text(Role::User, text)
    }

    fn tool_result(id: &str) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult(ToolResult {
                tool_call_id: id.to_string(),
                content: Vec::new(),
                is_error: None,
            })],
        }
    }

    fn tool_call(id: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: id.to_string(),
                name: "read".to_string(),
                arguments: Value::Null,
            })],
        }
    }

    async fn selected(
        pin: &TurnPin,
        state: &mut State,
        messages: Vec<Message>,
    ) -> Result<Option<String>> {
        let mut request = Request {
            llm_request: LlmRequest {
                messages,
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: None,
        };
        let (classification, _) = pin.score(state, &mut request, None).await?;
        Ok(classification
            .argmax(false)?
            .map(|score| score.target.as_str().to_string()))
    }

    #[tokio::test]
    async fn tool_continuation_turns_hold_the_pinned_target() -> Result<()> {
        let inner = RecordingClassifier::new(vec![Some("capable"), Some("efficient")]);
        let pin = TurnPin::new(inner.clone());
        let mut state = State::default();
        let opening = vec![user("debug this")];

        assert_eq!(
            selected(&pin, &mut state, opening.clone()).await?,
            Some("capable".to_string())
        );
        let mut continued = opening;
        continued.push(tool_call("call-1"));
        continued.push(tool_result("call-1"));
        assert_eq!(
            selected(&pin, &mut state, continued).await?,
            Some("capable".to_string())
        );
        assert_eq!(inner.consultations(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn a_fresh_user_message_re_classifies() -> Result<()> {
        let inner = RecordingClassifier::new(vec![Some("efficient"), Some("capable")]);
        let pin = TurnPin::new(inner.clone());
        let mut state = State::default();
        let opening = vec![user("summarise this file")];

        assert_eq!(
            selected(&pin, &mut state, opening.clone()).await?,
            Some("efficient".to_string())
        );
        let mut followed_up = opening;
        followed_up.push(tool_call("call-1"));
        followed_up.push(tool_result("call-1"));
        followed_up.push(user("now find the race condition"));
        assert_eq!(
            selected(&pin, &mut state, followed_up).await?,
            Some("capable".to_string())
        );
        assert_eq!(inner.consultations(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn an_abstention_clears_an_earlier_pin() -> Result<()> {
        let inner = RecordingClassifier::new(vec![Some("efficient"), None, Some("capable")]);
        let pin = TurnPin::new(inner.clone());
        let mut state = State::default();
        let opening = vec![user("summarise this file")];

        assert_eq!(
            selected(&pin, &mut state, opening.clone()).await?,
            Some("efficient".to_string())
        );
        let mut followed_up = opening;
        followed_up.push(user("now find the race condition"));
        assert_eq!(selected(&pin, &mut state, followed_up.clone()).await?, None);

        // The abstaining turn is served by the fall-through default, so its tool
        // continuations must not reuse the target pinned by the previous turn.
        followed_up.push(tool_call("call-1"));
        followed_up.push(tool_result("call-1"));
        assert_eq!(
            selected(&pin, &mut state, followed_up).await?,
            Some("capable".to_string())
        );
        assert_eq!(inner.consultations(), 3);
        Ok(())
    }

    #[test]
    fn a_turn_starts_only_when_the_user_spoke_last() {
        let mut messages = vec![user("debug this")];
        assert!(starts_new_turn(&messages));
        messages.push(tool_call("call-1"));
        assert!(!starts_new_turn(&messages));
        // Anthropic sends this as a user-role message, so it must not count as a turn.
        messages.push(tool_result("call-1"));
        assert!(!starts_new_turn(&messages));
        messages.push(user("still broken"));
        assert!(starts_new_turn(&messages));
        assert!(!starts_new_turn(&[]));

        // A tool result the user appended to counts, because the user did speak.
        let mut mixed = tool_result("call-2");
        mixed.content.extend(user("and also rename foo").content);
        assert!(starts_new_turn(&[mixed]));
    }
}
