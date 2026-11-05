// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use switchyard_libsy::{
    AffinityRouter, Algorithm, Classification, Classifier, Driver, Event, Processor, Result,
    RoutingOutcome, RuntimeModels, State, Step,
};
use switchyard_protocol::{Message, Metadata, ModelId, Request, Role, text_request};

fn request(index: usize) -> Request {
    Request {
        llm_request: text_request(None, "new user turn"),
        raw_request: None,
        metadata: Some(Metadata {
            session_id: Some(format!("session-{index}")),
            ..Metadata::default()
        }),
    }
}

struct Reproducer;

#[async_trait]
impl Algorithm for Reproducer {
    fn name(&self) -> &str {
        "retained-capacity-reproducer"
    }

    async fn route(self: Arc<Self>, driver: Driver, input: Request) -> Result<RoutingOutcome> {
        let router = AffinityRouter::new().with_release_on_user_turn();
        let mut state = State::default();
        let original = ModelId::from("model-a");
        let updated = ModelId::from("model-b");
        // Fill the current retention limit using only the public processor API.
        for index in 0..4096 {
            router
                .process(
                    &mut state,
                    Event::Decision {
                        request: &mut request(index),
                        selected_model_id: &original,
                        category: None,
                        driver: &driver,
                    },
                )
                .await?;
        }
        // Try two distinct updates so the check cannot depend on the arbitrary
        // eviction victim happening to be the identity being updated.
        for index in 0..2 {
            router
                .process(
                    &mut state,
                    Event::Decision {
                        request: &mut request(index),
                        selected_model_id: &updated,
                        category: None,
                        driver: &driver,
                    },
                )
                .await?;
            let mut missing = Vec::new();
            for session in 0..4096 {
                let mut continuation = request(session);
                continuation
                    .llm_request
                    .messages
                    .push(Message::text(Role::Tool, "ok"));
                let (classification, _) =
                    router.score(&mut state, &mut continuation, &driver).await?;
                let Classification::Scores(scores) = classification else {
                    panic!("affinity must return scores");
                };
                if scores.is_empty() {
                    missing.push(session);
                } else {
                    let expected = if session <= index {
                        &updated
                    } else {
                        &original
                    };
                    assert_eq!(&scores[0].target, expected);
                }
            }
            println!(
                "after updating session-{index}: {} unrelated sessions lost",
                missing.len()
            );
            assert!(missing.is_empty(), "lost retained sessions: {missing:?}");
        }
        Ok(RoutingOutcome::route_to(updated, Vec::new(), input))
    }
}

#[tokio::test]
async fn user_turn_updates_preserve_other_sessions_at_capacity() {
    let mut steps = Arc::new(Reproducer).run_stream(
        request(0),
        Arc::new(RuntimeModels::default()),
        State::default(),
    );
    assert!(matches!(steps.next().await, Some(Ok(Step::Done(_)))));
    assert!(steps.next().await.is_none());
}
