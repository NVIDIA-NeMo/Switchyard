// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::json;
use switchyard_protocol::{
    AggLlmResponse, Category, ContentBlock, FormatId, LlmClientError, LlmResponse, Message,
    Metadata, ModelId, Request, Response, ResponseOutput, Role, StopReason, ToolCall, WireFormat,
};

use super::super::config::{ACTIVE_APPROVAL, ServingMode, VgrConfig};
use super::super::tests::{call, request, result};
use crate::Result;
use crate::algorithms::vgr::Vgr;
use crate::core::algorithm::Algorithm;
use crate::core::testing::{reply, test_drive_with_models};

fn route(configure: impl FnOnce(&mut VgrConfig)) -> Result<Arc<dyn Algorithm>> {
    let mut config = VgrConfig::new(ModelId::new("local"), ModelId::new("cloud"));
    config.targets.judge = Some(ModelId::new("judge"));
    config.targets.cloud_judge = Some(ModelId::new("cloud-judge"));
    config.mode = ServingMode::Active {
        approval: ACTIVE_APPROVAL.into(),
    };
    config.task_typing = false;
    configure(&mut config);
    Ok(Arc::new(Vgr::new(config)?))
}

fn models() -> HashMap<Category, Vec<ModelId>> {
    HashMap::from([(
        Category::Any,
        vec![ModelId::new("local"), ModelId::new("cloud")],
    )])
}

fn session(messages: Vec<Message>) -> Request {
    let mut request = request(messages);
    request.metadata = Some(Metadata {
        session_id: Some("session-1".into()),
        ..Metadata::default()
    });
    request
}

fn agg(response: AggLlmResponse) -> Response {
    Response {
        llm_response: LlmResponse::Agg(response),
        metadata: None,
        upstream_headers: http::HeaderMap::new(),
    }
}

fn tool_call(arguments: serde_json::Value, stop_reason: StopReason) -> Response {
    agg(AggLlmResponse {
        outputs: vec![ResponseOutput {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "next".into(),
                name: "bash".into(),
                arguments,
            })],
            url_citations: Vec::new(),
            stop_reason: Some(stop_reason),
        }],
        ..Default::default()
    })
}

/// A verifier reply whose first token puts probability `p` on "yes".
fn scored(p: f64) -> Response {
    let mut response = AggLlmResponse::default();
    response.preservation.responses.insert(
        FormatId::from(WireFormat::OpenAiChat),
        json!({"choices": [{"logprobs": {"content": [{"top_logprobs": [
            {"token": "yes", "logprob": p.ln()},
            {"token": "no", "logprob": (1.0 - p).ln()}
        ]}]}}]}),
    );
    agg(response)
}

#[derive(Default)]
struct Calls(AtomicUsize, AtomicUsize);

async fn drive(
    route: &Arc<dyn Algorithm>,
    request: Request,
    calls: Arc<Calls>,
    local: fn() -> Response,
    judge_p: f64,
) -> Result<String> {
    let (selected, _) = test_drive_with_models(
        Arc::clone(route),
        request,
        models(),
        move |target: ModelId, _| {
            let calls = Arc::clone(&calls);
            async move {
                match target.as_str() {
                    "local" => calls.0.fetch_add(1, Ordering::Relaxed),
                    "cloud-judge" => calls.1.fetch_add(1, Ordering::Relaxed),
                    _ => 0,
                };
                Ok(match target.as_str() {
                    "local" => local(),
                    "judge" => scored(judge_p),
                    "cloud-judge" => reply("yes"),
                    _ => reply("cloud answer"),
                })
            }
        },
    )
    .await?;
    Ok(selected.to_string())
}

fn complete_call() -> Response {
    tool_call(json!({"command": "ls"}), StopReason::ToolUse)
}

#[tokio::test]
async fn malformed_tool_calls_escalate_and_complete_ones_are_judged() -> Result<()> {
    let route = route(|_| {})?;
    let calls = Arc::new(Calls::default());
    let truncated = || tool_call(json!(r#"{"command":"l"#), StopReason::MaxTokens);
    let task = || session(vec![Message::text(Role::User, "list files")]);
    assert_eq!(
        drive(&route, task(), calls.clone(), truncated, 0.0).await?,
        "cloud"
    );
    assert_eq!(
        drive(&route, task(), calls, complete_call, 0.1).await?,
        "local"
    );
    Ok(())
}

#[tokio::test]
async fn a_truncated_local_attempt_is_judged_before_it_escalates() -> Result<()> {
    let route = route(|config| config.targets.judge = None)?;
    let calls = Arc::new(Calls::default());
    let task = session(vec![Message::text(Role::User, "explain quicksort")]);
    let cut_off = || {
        let mut response = reply("Quicksort picks a pivot, then recurses because");
        if let LlmResponse::Agg(agg) = &mut response.llm_response {
            agg.outputs[0].stop_reason = Some(StopReason::MaxTokens);
        }
        response
    };
    let selected = drive(&route, task, calls.clone(), cut_off, 0.0).await?;
    assert_eq!(selected, "cloud");
    // Attempt, readout and deliberation; read as absent, only the attempt runs.
    assert_eq!(calls.0.load(Ordering::Relaxed), 3);
    Ok(())
}

#[tokio::test]
async fn two_consecutive_escalation_votes_latch_the_session() -> Result<()> {
    let route = route(|_| {})?;
    let calls = Arc::new(Calls::default());
    let task = || session(vec![Message::text(Role::User, "fix the build")]);
    assert_eq!(
        drive(&route, task(), calls.clone(), complete_call, 0.9).await?,
        "local"
    );
    assert_eq!(
        drive(&route, task(), calls.clone(), complete_call, 0.9).await?,
        "cloud"
    );
    assert_eq!(calls.0.load(Ordering::Relaxed), 2);

    // Latched past the user turn: no further local attempt is produced.
    assert_eq!(
        drive(&route, task(), calls.clone(), complete_call, 0.0).await?,
        "cloud"
    );
    assert_eq!(calls.0.load(Ordering::Relaxed), 2);
    Ok(())
}

#[tokio::test]
async fn final_answers_commit_only_on_a_verified_clean_record() -> Result<()> {
    let answer = || reply("the build passes");
    let run = |tool_output: &str| {
        session(vec![
            Message::text(Role::User, "fix the build"),
            call("c1"),
            result("c1", tool_output),
        ])
    };
    let calls = Arc::new(Calls::default());
    let clean = route(|_| {})?;
    assert_eq!(
        drive(&clean, run("ok"), calls.clone(), answer, 0.3).await?,
        "local"
    );
    assert_eq!(
        drive(&clean, run("ok"), calls.clone(), answer, 0.1).await?,
        "cloud"
    );

    // A failure longer ago than the clean tail is forgiven only on the capable judge.
    let mut long = vec![Message::text(Role::User, "fix the build")];
    long.extend([call("e"), result("e", r#"{"error": "missing file"}"#)]);
    for index in 0..101 {
        let id = format!("c{index}");
        long.extend([call(&id), result(&id, "ok")]);
    }
    assert_eq!(
        drive(&clean, session(long.clone()), calls.clone(), answer, 1.0).await?,
        "cloud"
    );
    assert_eq!(calls.1.load(Ordering::Relaxed), 0);
    let confirmed = route(|config| config.confirmed_recovery_min_clean_tail = Some(1))?;
    assert_eq!(
        drive(&confirmed, session(long), calls.clone(), answer, 1.0).await?,
        "local"
    );
    assert_eq!(calls.1.load(Ordering::Relaxed), 1);
    Ok(())
}

#[tokio::test]
async fn an_unavailable_local_tier_escalates_and_other_failures_surface() -> Result<()> {
    let route = route(|_| {})?;
    for (status, escalates) in [(503, true), (400, false)] {
        let outcome = test_drive_with_models(
            Arc::clone(&route),
            session(vec![Message::text(Role::User, "task")]),
            models(),
            move |target: ModelId, _| async move {
                match target.as_str() {
                    "local" => Err(LlmClientError::UpstreamHttp {
                        status: http::StatusCode::from_u16(status).unwrap_or_default(),
                        body: String::new(),
                    }),
                    _ => Ok(reply("cloud answer")),
                }
            },
        )
        .await;
        assert_eq!(
            outcome.is_ok_and(|(selected, _)| selected == "cloud"),
            escalates
        );
    }
    Ok(())
}

#[tokio::test]
async fn an_escalated_agentic_run_hands_off_once_with_the_unverified_claim() -> Result<()> {
    let route = route(|config| config.agentic_handoff = true)?;
    let sent = Arc::new(std::sync::Mutex::new(Vec::new()));
    let task = || {
        session(vec![
            Message::text(Role::User, "fix the build"),
            call("c1"),
            result("c1", "ok"),
        ])
    };
    for _ in 0..2 {
        let sent = Arc::clone(&sent);
        test_drive_with_models(
            Arc::clone(&route),
            task(),
            models(),
            move |target: ModelId, request: Request| {
                let sent = Arc::clone(&sent);
                async move {
                    Ok(match target.as_str() {
                        "local" => reply("the build passes"),
                        "judge" => scored(0.0),
                        "cloud" => {
                            let text = serde_json::to_string(&request.llm_request.messages)
                                .unwrap_or_default();
                            sent.lock().map(|mut sent| sent.push(text)).ok();
                            reply("cloud answer")
                        }
                        _ => reply("no"),
                    })
                }
            },
        )
        .await?;
    }
    let sent = sent.lock().map(|sent| sent.clone()).unwrap_or_default();
    assert_eq!(sent.len(), 2);
    assert!(sent[0].contains("Routing notice") && sent[0].contains("the build passes"));
    assert!(
        !sent[1].contains("Routing notice"),
        "sent once per user turn"
    );
    Ok(())
}

#[tokio::test]
async fn a_spent_local_turn_budget_skips_the_local_attempt() -> Result<()> {
    let route = route(|config| config.local_turn_budget = Some(std::time::Duration::ZERO))?;
    let calls = Arc::new(Calls::default());
    let task = session(vec![Message::text(Role::User, "fix the build")]);
    assert_eq!(
        drive(&route, task, calls.clone(), complete_call, 0.0).await?,
        "cloud"
    );
    assert_eq!(calls.0.load(Ordering::Relaxed), 0);
    Ok(())
}
