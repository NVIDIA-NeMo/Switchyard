// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end check that a target which overflowed is not called again in the same session.

use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use switchyard_libsy::{Algorithm, Driver, Result};
use switchyard_llm_client::{ClientRouter, run};
use switchyard_protocol::{
    LlmClientError, LlmResponse, Metadata, ModelId, Request, Response, RoutedLlmClient,
    text_request, text_response,
};

/// Offers both targets on every turn, weak first, exactly as a two-tier route does.
struct TwoTier;

#[async_trait]
impl Algorithm for TwoTier {
    fn name(&self) -> &str {
        "two_tier"
    }

    async fn route(self: Arc<Self>, driver: Driver, request: Request) -> Result<Response> {
        driver
            .call_model(
                request,
                vec![ModelId::from("weak"), ModelId::from("strong")],
                true,
            )
            .await
    }
}

/// Rejects every call to `weak` with a context-window error and records what it was asked for.
struct OverflowingWeak {
    calls: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl RoutedLlmClient for OverflowingWeak {
    async fn call(&self, request: Request) -> std::result::Result<Response, LlmClientError> {
        let model = request.llm_request.model.clone().unwrap_or_default();
        self.calls.lock().push(model.clone());
        if model == "weak" {
            return Err(LlmClientError::ContextWindowExceeded {
                model: ModelId::from("weak"),
                message: "too long".into(),
            });
        }
        Ok(Response {
            llm_response: LlmResponse::Agg(text_response(None, "answer")),
            metadata: None,
        })
    }
}

fn turn(session: &str) -> Request {
    Request {
        llm_request: text_request(None, "hi"),
        raw_request: None,
        metadata: Some(Metadata {
            session_id: Some(session.into()),
            ..Default::default()
        }),
    }
}

#[tokio::test]
async fn an_overflowed_target_is_not_retried_on_the_next_turn() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let clients = ClientRouter::single(Arc::new(OverflowingWeak {
        calls: Arc::clone(&calls),
    }));

    for _ in 0..3 {
        let (_, response) = run(Arc::new(TwoTier), clients.clone(), turn("s1"), None)
            .await
            .expect("strong answers every turn");
        assert_eq!(response.served_model().map(ModelId::as_str), Some("strong"));
    }

    // Turn one discovers the overflow, turns two and three skip weak entirely.
    assert_eq!(
        *calls.lock(),
        vec!["weak", "strong", "strong", "strong"],
        "weak should be called once, not once per turn"
    );
}

#[tokio::test]
async fn a_different_session_rediscovers_the_overflow() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let clients = ClientRouter::single(Arc::new(OverflowingWeak {
        calls: Arc::clone(&calls),
    }));

    run(Arc::new(TwoTier), clients.clone(), turn("s1"), None)
        .await
        .expect("strong answers");
    run(Arc::new(TwoTier), clients.clone(), turn("s2"), None)
        .await
        .expect("strong answers");

    assert_eq!(*calls.lock(), vec!["weak", "strong", "weak", "strong"]);
}
