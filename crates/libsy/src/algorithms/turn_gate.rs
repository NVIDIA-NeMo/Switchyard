// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Early-turn gate: route a fixed target for the opening turns of a conversation.
//!
//! [`TurnGate`] scores a fixed `target` at confidence `1.0` while the conversation is younger
//! than `min_turn`, and abstains afterwards. In the escalation cascade it sits ahead of the
//! judge so early turns route to the cheap tier **without** paying for a judge call — there is
//! too little trajectory to assess yet.

use async_trait::async_trait;
use switchyard_protocol::{Request, Role};

use super::core::{Classifier, Score, State};

/// Boxed, thread-safe error type used across the SDK.
type BoxErr = Box<dyn std::error::Error + Send + Sync>;

/// The conversation's turn number: the count of user-role messages so far.
pub fn conversation_turn(request: &Request) -> usize {
    request
        .llm_request
        .messages
        .iter()
        .filter(|message| message.role == Role::User)
        .count()
}

/// Routes `target` while the conversation turn is `< min_turn`, then abstains.
pub struct TurnGate {
    target: String,
    min_turn: usize,
}

impl TurnGate {
    /// Creates a gate that routes `target` until the conversation reaches `min_turn`.
    pub fn new(target: impl Into<String>, min_turn: usize) -> Self {
        Self {
            target: target.into(),
            min_turn,
        }
    }
}

#[async_trait]
impl Classifier for TurnGate {
    async fn score(
        &self,
        _state: &mut State,
        request: &Request,
        _driver: Option<&crate::Driver>,
    ) -> Result<Vec<Score>, BoxErr> {
        if conversation_turn(request) < self.min_turn {
            Ok(vec![Score {
                confidence: 1.0,
                target: self.target.clone(),
            }])
        } else {
            Ok(Vec::new())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use switchyard_protocol::{LlmRequest, Message};

    /// A request with `user_turns` user messages, each followed by an assistant reply.
    fn request(user_turns: usize) -> Request {
        let mut messages = Vec::new();
        for index in 0..user_turns {
            messages.push(Message::text(Role::User, format!("turn {index}")));
            messages.push(Message::text(Role::Assistant, "ok"));
        }
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
    fn conversation_turn_counts_user_messages() {
        assert_eq!(conversation_turn(&request(0)), 0);
        assert_eq!(conversation_turn(&request(3)), 3);
    }

    #[tokio::test]
    async fn routes_the_target_before_min_turn() -> Result<(), BoxErr> {
        let gate = TurnGate::new("weak", 3);
        let scores = gate.score(&mut State::default(), &request(2), None).await?;
        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].target, "weak");
        assert_eq!(scores[0].confidence, 1.0);
        Ok(())
    }

    #[tokio::test]
    async fn abstains_at_and_after_min_turn() -> Result<(), BoxErr> {
        let gate = TurnGate::new("weak", 3);
        // At the threshold the judge should run, so the gate abstains.
        assert!(gate
            .score(&mut State::default(), &request(3), None)
            .await?
            .is_empty());
        assert!(gate
            .score(&mut State::default(), &request(5), None)
            .await?
            .is_empty());
        Ok(())
    }
}
