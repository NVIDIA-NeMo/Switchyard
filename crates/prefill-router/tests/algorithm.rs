// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use libsy::{Algorithm, LibsyError};
use prefill_router::{PrefillForward, PrefillRouterAlgo, Result};
use switchyard_protocol::{
    ContentBlock, LlmRequest, Message, Metadata, ModelId, Request, Role, ToolResult, text_request,
};

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

async fn selected(route: Arc<dyn Algorithm>, request: Request) -> libsy::Result<String> {
    let outcome = libsy::drive(route, request, |call| async move {
        call.respond(Ok(switchyard_protocol::Response {
            llm_response: switchyard_protocol::LlmResponse::Agg(
                switchyard_protocol::text_response(None, "unused"),
            ),
            metadata: None,
        }))
    })
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
        metadata: Some(Metadata {
            session_id: Some("session-1".to_string()),
            ..Metadata::default()
        }),
    }
}

#[tokio::test]
async fn routes_on_latest_user_text_and_reuses_decision_for_tool_steps() -> libsy::Result<()> {
    let (forward, calls, prompts) = forward();
    let route: Arc<dyn Algorithm> = Arc::new(
        PrefillRouterAlgo::from_test_forward(target_set(), forward).map_err(|error| {
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
            Message::text(Role::User, "follow-up task"),
            Message::text(Role::Assistant, "calling a tool"),
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult(ToolResult {
                    tool_call_id: "call-1".to_string(),
                    content: vec![ContentBlock::Text {
                        text: "tool output".to_string(),
                    }],
                    is_error: None,
                })],
            },
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
