// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use libsy::{Algorithm, Call, LibsyError, RuntimeModels};
use switchyard_protocol::{
    Category, ContentBlock, LlmRequest, Message, Metadata, ModelId, Request, Role, ToolResult,
    text_request,
};

use crate::{PrefillForward, PrefillRouterAlgo, PrefillRouterError, Result};

struct RecordingForward {
    calls: Arc<AtomicUsize>,
    prompts: Arc<std::sync::Mutex<Vec<String>>>,
}

impl PrefillForward for RecordingForward {
    fn output_count(&self) -> usize {
        2
    }

    fn forward(&mut self, prompts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if prompts.iter().any(|prompt| prompt == "fail inference") {
            return Err(PrefillRouterError::InvalidRequest("test failure".into()));
        }
        self.prompts
            .lock()
            .expect("test prompts lock")
            .extend_from_slice(prompts);
        Ok(prompts
            .iter()
            .map(|prompt| {
                if prompt.contains("follow-up") {
                    vec![0.1, 0.9]
                } else {
                    vec![0.8, 0.2]
                }
            })
            .collect())
    }

    fn unload(&mut self) -> Result<()> {
        Ok(())
    }
}

fn target_set() -> Vec<ModelId> {
    vec![ModelId::from("small"), ModelId::from("large")]
}

fn forward() -> (
    RecordingForward,
    Arc<AtomicUsize>,
    Arc<std::sync::Mutex<Vec<String>>>,
) {
    let calls = Arc::new(AtomicUsize::new(0));
    let prompts = Arc::new(std::sync::Mutex::new(Vec::new()));
    (
        RecordingForward {
            calls: Arc::clone(&calls),
            prompts: Arc::clone(&prompts),
        },
        calls,
        prompts,
    )
}

async fn serve_decision(call: libsy::CallDecision) -> libsy::Result<()> {
    let response = switchyard_protocol::DecisionResponse {
        id: None,
        model: Some(call.model.clone()),
        answers: Default::default(),
        usage: Default::default(),
    };
    call.respond(std::future::ready(Ok(response))).await
}

async fn selected(route: Arc<dyn Algorithm>, request: Request) -> libsy::Result<String> {
    let outcome = libsy::drive(
        route,
        request,
        Arc::new(RuntimeModels::new([(Category::Any, target_set())].into())),
        |call| async move {
            let call = match call {
                Call::Model(call) => *call,
                Call::Decision(call) => return serve_decision(*call).await,
            };
            call.respond(std::future::ready(Ok(switchyard_protocol::Response {
                llm_response: switchyard_protocol::LlmResponse::Agg(
                    switchyard_protocol::text_response(None, "unused"),
                ),
                metadata: None,
                upstream_headers: Default::default(),
            })))
            .await
        },
    )
    .await?;
    Ok(outcome.selected_model_id()?.to_string())
}

fn request(messages: Vec<Message>) -> Request {
    Request {
        llm_request: LlmRequest {
            model: Some("auto".to_string()),
            messages,
            ..LlmRequest::default()
        },
        raw_request: None,
        metadata: None,
    }
}

#[tokio::test]
async fn final_request_releases_its_own_retained_assignment() -> libsy::Result<()> {
    let (forward, _, _) = forward();
    let route: Arc<dyn Algorithm> = Arc::new(
        PrefillRouterAlgo::from_forward(target_set(), forward).map_err(|error| {
            LibsyError::AlgorithmError {
                message: error.to_string(),
            }
        })?,
    );
    let with_session = |id: &str, final_request: bool, messages| {
        let mut req = request(messages);
        req.metadata = Some(Metadata {
            session_id: Some(id.to_string()),
            session_final: Some(final_request),
            ..Metadata::default()
        });
        req
    };
    for id in ["one", "two"] {
        assert_eq!(
            selected(
                Arc::clone(&route),
                with_session(id, false, vec![Message::text(Role::User, "follow-up task")])
            )
            .await?,
            "large"
        );
    }
    assert_eq!(
        selected(Arc::clone(&route), with_session("one", true, vec![])).await?,
        "large"
    );
    assert_eq!(
        selected(Arc::clone(&route), with_session("two", false, vec![])).await?,
        "large"
    );
    assert_eq!(
        selected(route, with_session("one", false, vec![])).await?,
        "small"
    );
    Ok(())
}

#[tokio::test]
async fn failed_final_request_releases_retained_assignment() -> libsy::Result<()> {
    let (forward, _, _) = forward();
    let route: Arc<dyn Algorithm> = Arc::new(
        PrefillRouterAlgo::from_forward(target_set(), forward).map_err(|error| {
            LibsyError::AlgorithmError {
                message: error.to_string(),
            }
        })?,
    );
    let with_session = |final_request: bool, messages| {
        let mut req = request(messages);
        req.metadata = Some(Metadata {
            session_id: Some("one".into()),
            session_final: Some(final_request),
            ..Metadata::default()
        });
        req
    };
    assert_eq!(
        selected(
            Arc::clone(&route),
            with_session(false, vec![Message::text(Role::User, "follow-up task")])
        )
        .await?,
        "large"
    );
    let failed = with_session(
        true,
        vec![
            Message::text(Role::User, "initial"),
            Message::text(Role::Assistant, "done"),
            Message::text(Role::User, "fail inference"),
        ],
    );
    assert!(selected(Arc::clone(&route), failed).await.is_err());
    assert_eq!(selected(route, with_session(false, vec![])).await?, "small");
    Ok(())
}

#[tokio::test]
async fn empty_trailing_turn_does_not_score_the_previous_exchange() -> libsy::Result<()> {
    let (forward, calls, prompts) = forward();
    let route: Arc<dyn Algorithm> = Arc::new(
        PrefillRouterAlgo::from_forward(target_set(), forward).map_err(|error| {
            LibsyError::AlgorithmError {
                message: error.to_string(),
            }
        })?,
    );
    // The newest user turn has no text; the turn before the assistant's reply
    // belongs to a finished exchange and must not be scored in its place.
    let routed = selected(
        route,
        request(vec![
            Message::text(Role::User, "initial task"),
            Message::text(Role::Assistant, "done"),
            Message::text(Role::User, "   "),
        ]),
    )
    .await?;
    assert_eq!(routed, "small");
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert!(prompts.lock().expect("test prompts lock").is_empty());
    Ok(())
}

#[tokio::test]
async fn empty_user_turn_beside_the_person_still_scores_the_person() -> libsy::Result<()> {
    let (forward, _calls, prompts) = forward();
    let route: Arc<dyn Algorithm> = Arc::new(
        PrefillRouterAlgo::from_forward(target_set(), forward).map_err(|error| {
            LibsyError::AlgorithmError {
                message: error.to_string(),
            }
        })?,
    );
    // No assistant reply between the two: the empty turn is skipped and the
    // person's text in the same exchange is scored.
    selected(
        route,
        request(vec![
            Message::text(Role::User, "initial task"),
            Message::text(Role::User, "   "),
        ]),
    )
    .await?;
    assert_eq!(
        prompts.lock().expect("test prompts lock").as_slice(),
        ["initial task"]
    );
    Ok(())
}

#[tokio::test]
async fn tool_result_after_the_person_is_walked_past_not_skipped() -> libsy::Result<()> {
    let (forward, _calls, prompts) = forward();
    let route: Arc<dyn Algorithm> = Arc::new(
        PrefillRouterAlgo::from_forward(target_set(), forward).map_err(|error| {
            LibsyError::AlgorithmError {
                message: error.to_string(),
            }
        })?,
    );
    // A tool result is a text-less user message inside the current exchange;
    // it must not count as a skipped turn that ends the search at the tool call.
    selected(
        route,
        request(vec![
            Message::text(Role::User, "follow-up task"),
            Message::text(Role::Assistant, "calling a tool"),
            Message::new(
                Role::User,
                vec![ContentBlock::ToolResult(ToolResult {
                    tool_call_id: "call-1".to_string(),
                    content: vec![ContentBlock::Text {
                        text: "tool output".to_string(),
                    }],
                    is_error: None,
                })],
            ),
        ]),
    )
    .await?;
    assert_eq!(
        prompts.lock().expect("test prompts lock").as_slice(),
        ["follow-up task"]
    );
    Ok(())
}

#[tokio::test]
async fn routes_on_latest_user_text_and_reuses_decision_for_tool_steps() -> libsy::Result<()> {
    let (forward, calls, prompts) = forward();
    let route: Arc<dyn Algorithm> = Arc::new(
        PrefillRouterAlgo::from_forward(target_set(), forward).map_err(|error| {
            LibsyError::AlgorithmError {
                message: error.to_string(),
            }
        })?,
    );

    let routed = selected(
        Arc::clone(&route),
        request(vec![
            Message::text(Role::User, "initial task"),
            Message::text(Role::Assistant, "working"),
            Message::text(Role::User, "follow-up task"),
        ]),
    )
    .await?;
    assert_eq!(routed, "large");
    assert_eq!(
        prompts.lock().expect("test prompts lock").as_slice(),
        ["follow-up task"]
    );

    let continuation = selected(
        Arc::clone(&route),
        request(vec![
            Message::text(Role::User, "initial task"),
            Message::text(Role::Assistant, "working"),
            Message::text(Role::User, "follow-up task"),
            Message::text(Role::Assistant, "calling a tool"),
            Message::new(
                Role::User,
                vec![ContentBlock::ToolResult(ToolResult {
                    tool_call_id: "call-1".to_string(),
                    content: vec![ContentBlock::Text {
                        text: "tool output".to_string(),
                    }],
                    is_error: None,
                })],
            ),
        ]),
    )
    .await?;
    assert_eq!(continuation, "large");
    assert_eq!(calls.load(Ordering::Relaxed), 1);

    let fallback = selected(
        route,
        Request {
            llm_request: text_request(Some("auto".to_string()), ""),
            raw_request: None,
            metadata: None,
        },
    )
    .await?;
    assert_eq!(fallback, "small");
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    Ok(())
}
