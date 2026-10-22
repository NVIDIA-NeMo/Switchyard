// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use switchyard_libsy::{Algorithm, Call, Driver, Result, RoutingOutcome, RuntimeModels};
use switchyard_protocol::{
    DecisionRequest, LlmResponse, Request, Response, text_request, text_response,
};
use tokio::sync::Notify;

struct Abandon {
    decision: bool,
    started: Arc<Notify>,
    dropped: Arc<Notify>,
}

#[async_trait]
impl Algorithm for Abandon {
    fn name(&self) -> &str {
        "abandon"
    }

    async fn route(self: Arc<Self>, driver: Driver, request: Request) -> Result<RoutingOutcome> {
        {
            let call = async {
                if self.decision {
                    driver
                        .call_decision(
                            DecisionRequest {
                                model: None,
                                context: serde_json::json!({}),
                                questions: Default::default(),
                            },
                            "abandoned".into(),
                        )
                        .await
                        .map(|_| ())
                } else {
                    driver
                        .call_model(request.clone(), vec!["abandoned".into()])
                        .await
                        .map(|_| ())
                }
            };
            tokio::pin!(call);
            tokio::select! {
                _ = self.started.notified() => {},
                _ = &mut call => panic!("abandoned work finished"),
            }
        }
        // The next call cannot begin until the host drops the abandoned work.
        self.dropped.notified().await;
        let response = driver
            .call_model(request.clone(), vec!["continuation".into()])
            .await?;
        Ok(RoutingOutcome::answered(
            "continuation".into(),
            request,
            response,
        ))
    }
}

struct DropNotify(Arc<Notify>);
impl Drop for DropNotify {
    fn drop(&mut self) {
        self.0.notify_one();
    }
}

async fn pending<T>(started: Arc<Notify>, dropped: Arc<Notify>) -> Result<T> {
    let _guard = DropNotify(dropped);
    started.notify_one();
    std::future::pending().await
}

fn request() -> Request {
    Request {
        llm_request: text_request(None, "hello"),
        raw_request: None,
        metadata: None,
    }
}

fn response() -> Response {
    Response {
        llm_response: LlmResponse::Agg(text_response(None, "continued")),
        metadata: None,
        upstream_headers: Default::default(),
    }
}

#[tokio::test]
async fn abandoning_model_or_decision_work_cancels_only_that_call() {
    for decision in [false, true] {
        let started = Arc::new(Notify::new());
        let dropped = Arc::new(Notify::new());
        let algorithm = Arc::new(Abandon {
            decision,
            started: started.clone(),
            dropped: dropped.clone(),
        });
        let outcome = tokio::time::timeout(
            Duration::from_secs(1),
            switchyard_libsy::drive(
                algorithm,
                request(),
                Arc::new(RuntimeModels::default()),
                move |call| {
                    let started = started.clone();
                    let dropped = dropped.clone();
                    async move {
                        match call {
                            Call::Model(call) if call.models[0] == "continuation" => {
                                call.respond(std::future::ready(Ok(response()))).await
                            }
                            Call::Model(call) => call.respond(pending(started, dropped)).await,
                            Call::Decision(call) => call.respond(pending(started, dropped)).await,
                        }
                    }
                },
            ),
        )
        .await
        .expect("cancelled call blocked continuation")
        .expect("cancelled call failed the run");
        assert_eq!(outcome.selected_model_id().unwrap(), "continuation");
    }
}
