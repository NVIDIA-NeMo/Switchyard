// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
use crate::core::state::{State, StateValue};
use crate::core::testing::{category_models, reply, test_drive_with_models};

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
        category_models(Category::Any, &["local", "cloud"]),
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
    // Attempt, readout, deliberation and its verdict; read as absent, only the attempt runs.
    assert_eq!(calls.0.load(Ordering::Relaxed), 4);
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
        drive(&clean, session(long.clone()), calls.clone(), answer, 0.99).await?,
        "cloud"
    );
    assert_eq!(calls.1.load(Ordering::Relaxed), 0);
    let confirmed = route(|config| config.confirmed_recovery_min_clean_tail = Some(1))?;
    assert_eq!(
        drive(&confirmed, session(long), calls.clone(), answer, 0.99).await?,
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
            category_models(Category::Any, &["local", "cloud"]),
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

fn recovering_classifier() -> super::VgrClassifier {
    let mut config = VgrConfig::new(ModelId::new("local"), ModelId::new("cloud"));
    config.mode = ServingMode::Evaluate;
    config.breaker.threshold = 1;
    config.breaker.cooldown = std::time::Duration::ZERO;
    let classifier = super::VgrClassifier {
        breaker: super::CircuitBreaker::new(config.breaker),
        config,
    };
    classifier.breaker.failure();
    classifier
}

#[tokio::test]
async fn verifier_budget_exhaustion_is_not_an_endpoint_failure() -> Result<()> {
    use std::time::{Duration, Instant};

    use crate::core::algorithm::{Driver, RuntimeModels, Step};

    let mut config = VgrConfig::new(ModelId::new("local"), ModelId::new("cloud"));
    config.deadline = Duration::from_millis(100);
    config.breaker.threshold = 1;
    config.breaker.cooldown = Duration::from_secs(60);
    assert!(config.targets.judge.is_none());
    let classifier = super::VgrClassifier {
        breaker: super::CircuitBreaker::new(config.breaker),
        config,
    };
    let mut state = State::default();
    let task = session(vec![Message::text(Role::User, "list files")]);
    let LlmResponse::Agg(proposed) = complete_call().llm_response else {
        unreachable!();
    };
    let (driver, mut steps) = Driver::new("vgr", Arc::new(RuntimeModels::default()));
    let context = super::RequestContext {
        driver: &driver,
        request: &task,
        started: Instant::now(),
    };
    let mut judging = Box::pin(classifier.verify_turn(&context, &proposed, &mut state));
    assert!(futures::poll!(&mut judging).is_pending());
    let Step::CallModel(pending_call) = steps.try_recv().expect("judge call should be queued")?
    else {
        unreachable!();
    };
    assert_eq!(pending_call.models, vec![ModelId::new("local")]);
    let escalates = judging.await;
    drop(pending_call);
    assert!(!escalates);
    assert!(super::turn_streak(&state) < super::TURN_CONFIRMATIONS);
    assert!(classifier.breaker.attempt().is_some());
    Ok(())
}

#[tokio::test]
async fn skipped_requests_leave_the_recovery_trial_available() -> Result<()> {
    use crate::core::classifier::Classifier;
    use crate::core::testing::empty_driver;
    use switchyard_protocol::ImageSource;

    for skip in ["image", "latched", "off", "kill", "deadline"] {
        let mut classifier = recovering_classifier();
        let mut state = State::default();
        let mut task = request(vec![Message::text(Role::User, "task")]);
        match skip {
            "image" => {
                task.llm_request.messages[0].content = vec![ContentBlock::Image {
                    source: ImageSource::Url {
                        url: "https://example.test/image.png".into(),
                        detail: None,
                    },
                }]
            }
            "latched" => {
                state.extra.insert(
                    super::TURN_STREAK_KEY.to_string(),
                    StateValue::Count(super::TURN_CONFIRMATIONS),
                );
            }
            "off" => classifier.config.mode = ServingMode::Off,
            "kill" => {
                let switch = super::super::KillSwitch::new();
                switch.engage();
                classifier.config.kill_switch = Some(switch);
            }
            "deadline" => classifier.config.deadline = std::time::Duration::ZERO,
            _ => unreachable!(),
        }
        classifier
            .score(&mut state, &mut task, &empty_driver())
            .await?;
        assert!(
            classifier.breaker.attempt().is_some(),
            "{skip} reserved the recovery trial"
        );
    }
    Ok(())
}

#[tokio::test]
async fn cancelling_local_generation_releases_the_recovery_trial() -> Result<()> {
    use crate::core::algorithm::{Driver, RuntimeModels};
    use crate::core::classifier::Classifier;

    let classifier = recovering_classifier();
    let mut state = State::default();
    let mut task = request(vec![Message::text(Role::User, "task")]);
    let (driver, _steps) = Driver::new("vgr", Arc::new(RuntimeModels::default()));
    let mut pending = classifier.score(&mut state, &mut task, &driver);
    assert!(futures::poll!(&mut pending).is_pending());
    assert!(classifier.breaker.attempt().is_none());
    drop(pending);
    assert!(classifier.breaker.attempt().is_some());
    Ok(())
}

#[tokio::test]
async fn decisions_attach_the_reason_and_exclude_local_cloud_fallbacks() -> Result<()> {
    use crate::core::algorithm::Call;
    use crate::core::testing::serve_decision;

    for (mode, reason, served) in [
        (ServingMode::Off, "mode_off", "cloud"),
        (ServingMode::Evaluate, "decided", "local"),
        (ServingMode::Shadow, "decided", "cloud"),
    ] {
        let algorithm = route(|config| config.mode = mode)?;
        let outcome = crate::drive(
            algorithm,
            request(vec![Message::text(Role::User, "task")]),
            Arc::new(category_models(Category::Any, &["local", "cloud"]).into()),
            |call| async move {
                match call {
                    Call::Model(call) => {
                        let response = if call.models[0] == ModelId::new("local") {
                            reply("done")
                        } else {
                            scored(0.9)
                        };
                        call.respond(std::future::ready(Ok(response))).await
                    }
                    Call::Decision(call) => serve_decision(*call).await,
                }
            },
        )
        .await?;
        assert_eq!(outcome.selected_model_ids, vec![ModelId::new(served)]);
        let evidence = outcome
            .metadata
            .and_then(|metadata| metadata.evidence)
            .unwrap_or_default();
        assert_eq!(evidence["source"], "vgr");
        assert_eq!(evidence["reason_code"], reason);
        assert_eq!(evidence["verdict"], served);
        assert_eq!(evidence["policy_version"], "1.0");
        if reason == "decided" {
            assert_eq!(evidence["branch"], "DefaultVerified");
        }
    }
    Ok(())
}

#[tokio::test]
async fn cloud_only_work_does_not_get_a_local_handoff_notice() -> Result<()> {
    for scenario in ["shadow", "kill_switch", "local_turn_budget"] {
        let route = route(|config| {
            config.agentic_handoff = true;
            match scenario {
                "shadow" => config.mode = ServingMode::Shadow,
                "kill_switch" => {
                    let switch = super::super::safety::KillSwitch::new();
                    switch.engage();
                    config.kill_switch = Some(switch);
                }
                _ => config.local_turn_budget = Some(std::time::Duration::ZERO),
            }
        })?;
        let task = session(vec![
            Message::text(Role::User, "fix the build"),
            call("c1"),
            result("c1", "ok"),
        ]);
        let (selected, _) = test_drive_with_models(
            route,
            task,
            category_models(Category::Any, &["local", "cloud"]),
            move |target: ModelId, request: Request| async move {
                Ok(match target.as_str() {
                    "local" => complete_call(),
                    "judge" => scored(0.0),
                    _ => {
                        let text = serde_json::to_string(&request.llm_request.messages)
                            .unwrap_or_default();
                        assert!(!text.contains("Routing notice"), "{scenario}");
                        reply("cloud answer")
                    }
                })
            },
        )
        .await?;
        assert_eq!(selected, "cloud");
    }
    Ok(())
}

#[tokio::test]
async fn latched_sessions_do_not_repeat_the_handoff_on_later_user_turns() -> Result<()> {
    let route = route(|config| config.agentic_handoff = true)?;
    assert_eq!(
        drive(
            &route,
            session(vec![Message::text(Role::User, "fix the build")]),
            Arc::new(Calls::default()),
            complete_call,
            0.9,
        )
        .await?,
        "local"
    );
    let sent = Arc::new(parking_lot::Mutex::new(Vec::new()));
    for new_user_turn in [false, true] {
        let mut messages = vec![
            Message::text(Role::User, "fix the build"),
            call("c1"),
            result("c1", "ok"),
        ];
        if new_user_turn {
            messages.push(Message::text(Role::User, "now run the tests"));
        }
        let sent = Arc::clone(&sent);
        let (selected, _) = test_drive_with_models(
            Arc::clone(&route),
            session(messages),
            category_models(Category::Any, &["local", "cloud"]),
            move |target: ModelId, request: Request| {
                let sent = Arc::clone(&sent);
                async move {
                    Ok(match target.as_str() {
                        "local" => complete_call(),
                        "judge" => scored(0.9),
                        _ => {
                            sent.lock().push(
                                serde_json::to_string(&request.llm_request.messages)
                                    .unwrap_or_default(),
                            );
                            reply("cloud answer")
                        }
                    })
                }
            },
        )
        .await?;
        assert_eq!(selected, "cloud");
    }
    let sent = sent.lock();
    assert_eq!(sent.len(), 2);
    assert!(sent[0].contains("Routing notice"));
    assert!(!sent[1].contains("Routing notice"));
    Ok(())
}

#[tokio::test]
async fn an_escalated_agentic_run_hands_off_once_with_the_unverified_claim() -> Result<()> {
    let route = route(|config| config.agentic_handoff = true)?;
    assert_eq!(
        drive(
            &route,
            session(vec![Message::text(Role::User, "fix the build")]),
            Arc::new(Calls::default()),
            complete_call,
            0.0,
        )
        .await?,
        "local"
    );
    let sent = Arc::new(parking_lot::Mutex::new(Vec::new()));
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
            category_models(Category::Any, &["local", "cloud"]),
            move |target: ModelId, request: Request| {
                let sent = Arc::clone(&sent);
                async move {
                    Ok(match target.as_str() {
                        "local" => reply("the build passes"),
                        "judge" => scored(0.0),
                        "cloud" => {
                            let text = serde_json::to_string(&request.llm_request.messages)
                                .unwrap_or_default();
                            sent.lock().push(text);
                            reply("cloud answer")
                        }
                        _ => reply("no"),
                    })
                }
            },
        )
        .await?;
    }
    let sent = sent.lock();
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

#[test]
fn handoff_preserves_existing_messages_and_non_text_content() {
    use switchyard_protocol::ImageSource;

    let mut classifier = recovering_classifier();
    classifier.config.agentic_handoff = true;
    let mut task = request(vec![
        Message::text(Role::User, "finish the task"),
        Message::text(Role::User, "keep the original files"),
        call("c1"),
        result("c1", "the command failed"),
        Message::text(Role::User, "preserve this requirement"),
        Message::text(Role::User, "continue"),
        Message::text(Role::User, "then report the result"),
    ]);
    if let ContentBlock::ToolResult(result) = &mut task.llm_request.messages[3].content[0] {
        result.content.push(ContentBlock::Image {
            source: ImageSource::Url {
                url: "https://example.test/result.png".into(),
                detail: None,
            },
        });
    }
    let original = task.llm_request.messages.clone();
    let mut state = State::default();
    super::set_flag(&mut state, super::LOCAL_SERVED_KEY, true);
    classifier.hand_off(&mut task, &mut state, Some("unchecked result"));
    // A notice may be appended, but all original content stays unchanged.
    let messages = &task.llm_request.messages;
    assert_eq!(messages.len(), original.len());
    assert_eq!(
        &messages[..original.len() - 1],
        &original[..original.len() - 1]
    );
    let last = messages.last().expect("last user message");
    assert_eq!(
        &last.content[..original
            .last()
            .expect("original last message")
            .content
            .len()],
        &original.last().expect("original last message").content
    );
    assert!(
        last.text_content("")
            .is_some_and(|text| text.contains("unchecked result"))
    );
    let after_first = task.llm_request.messages.clone();
    classifier.hand_off(&mut task, &mut state, Some("duplicate notice"));
    assert_eq!(task.llm_request.messages, after_first);
}

#[tokio::test]
async fn a_blank_prompt_escalates_without_a_local_attempt() -> Result<()> {
    let continuing = route(|_| {})?;
    let route = route(|_| {})?;
    let calls = Arc::new(Calls::default());
    let blank_follow_up = vec![
        Message::text(Role::User, "what is 2 + 2?"),
        Message::text(Role::Assistant, "4"),
        Message::text(Role::User, " "),
    ];
    for messages in [
        vec![Message::text(Role::User, "  \n ")],
        vec![Message::text(Role::User, "\u{200B}\u{FEFF}")],
        blank_follow_up,
    ] {
        assert_eq!(
            drive(&route, session(messages), calls.clone(), complete_call, 0.0).await?,
            "cloud"
        );
    }
    assert_eq!(calls.0.load(Ordering::Relaxed), 0);

    // Tool results sent as a user message continue the task; they are not a blank prompt.
    let mut results = result("a", "file.txt");
    results.role = Role::User;
    let continuation = session(vec![
        Message::text(Role::User, "list files"),
        call("a"),
        results,
    ]);
    drive(&continuing, continuation, calls.clone(), complete_call, 0.0).await?;
    assert_eq!(calls.0.load(Ordering::Relaxed), 1);
    Ok(())
}

#[tokio::test]
async fn an_agent_marked_nudge_keeps_a_single_escalation_on_cloud() -> Result<()> {
    let route = route(|_| {})?;
    let calls = Arc::new(Calls::default());
    let truncated = || tool_call(json!(r#"{"command":"l"#), StopReason::MaxTokens);
    let task = session(vec![Message::text(Role::User, "fix the build")]);
    drive(&route, task, calls.clone(), truncated, 0.0).await?;
    let nudge = vec![
        Message::text(Role::User, "fix the build"),
        Message::text(Role::Assistant, "fixed"),
        Message::text(Role::User, "Verify the change before you stop."),
    ];
    let nudge = initiated_by(session(nudge), "agent");
    assert_eq!(
        drive(&route, nudge, calls.clone(), complete_call, 0.0).await?,
        "cloud"
    );
    assert_eq!(calls.0.load(Ordering::Relaxed), 1);
    Ok(())
}

fn initiated_by(mut request: Request, initiator: &str) -> Request {
    let metadata = request.metadata.get_or_insert_with(Metadata::default);
    metadata
        .http_headers
        .get_or_insert_with(http::HeaderMap::new)
        .insert("x-initiator", initiator.parse().unwrap());
    request
}

#[tokio::test]
async fn a_held_follow_up_records_the_latch_as_its_reason() -> Result<()> {
    use crate::core::algorithm::Call;
    use crate::core::testing::serve_decision;

    let algorithm = route(|config| config.mode = ServingMode::Shadow)?;
    let models = Arc::new(category_models(Category::Any, &["local", "cloud"]).into());
    let serve = |call| async move {
        match call {
            Call::Model(call) => call.respond(std::future::ready(Ok(reply("done")))).await,
            Call::Decision(call) => serve_decision(*call).await,
        }
    };
    let escalated = crate::drive(
        Arc::clone(&algorithm),
        session(vec![Message::text(Role::User, "fix the build")]),
        Arc::clone(&models),
        serve,
    )
    .await?;
    assert_eq!(escalated.selected_model_ids, vec![ModelId::new("cloud")]);

    let held = crate::drive(
        algorithm,
        session(vec![
            Message::text(Role::User, "fix the build"),
            call("c1"),
            result("c1", "ok"),
        ]),
        models,
        serve,
    )
    .await?;
    assert_eq!(held.selected_model_ids, vec![ModelId::new("cloud")]);
    let evidence = held
        .metadata
        .and_then(|metadata| metadata.evidence)
        .unwrap_or_default();
    assert_eq!(evidence["source"], "vgr");
    assert_eq!(evidence["reason_code"], "turn_verification_latched");
    assert_eq!(evidence["verdict"], "cloud");
    Ok(())
}

#[tokio::test]
async fn an_escalation_latch_holds_the_session_past_new_questions() -> Result<()> {
    let route = route(|config| config.latch_escalation = true)?;
    let calls = Arc::new(Calls::default());
    let truncated = || tool_call(json!(r#"{"command":"l"#), StopReason::MaxTokens);
    let task = session(vec![Message::text(Role::User, "fix the build")]);
    assert_eq!(
        drive(&route, task, calls.clone(), truncated, 0.0).await?,
        "cloud"
    );
    let follow_up = session(vec![
        Message::text(Role::User, "fix the build"),
        Message::text(Role::Assistant, "fixed"),
        Message::text(Role::User, "now add a test"),
    ]);
    assert_eq!(
        drive(&route, follow_up, calls.clone(), complete_call, 0.0).await?,
        "cloud"
    );
    assert_eq!(calls.0.load(Ordering::Relaxed), 1);
    Ok(())
}

#[tokio::test]
async fn an_initiator_header_decides_when_the_turn_latch_releases() -> Result<()> {
    let route = route(|_| {})?;
    let calls = Arc::new(Calls::default());
    let task = || session(vec![Message::text(Role::User, "fix the build")]);
    drive(&route, task(), calls.clone(), complete_call, 0.9).await?;
    drive(&route, task(), calls.clone(), complete_call, 0.9).await?;
    assert_eq!(calls.0.load(Ordering::Relaxed), 2);

    // A client nudge after a final reply looks like a new question; marked agent, it keeps the latch.
    let nudge = vec![
        Message::text(Role::User, "fix the build"),
        Message::text(Role::Assistant, "fixed"),
        Message::text(Role::User, "Verify the change before you stop."),
    ];
    assert_eq!(
        drive(
            &route,
            initiated_by(session(nudge), "agent"),
            calls.clone(),
            complete_call,
            0.0
        )
        .await?,
        "cloud"
    );
    assert_eq!(calls.0.load(Ordering::Relaxed), 2);

    // A person's correction after tool results looks like a continuation; marked user, it is verified afresh.
    let steer = vec![
        Message::text(Role::User, "fix the build"),
        call("c1"),
        result("c1", "ok"),
        Message::text(Role::User, "use the staging config instead"),
    ];
    assert_eq!(
        drive(
            &route,
            initiated_by(session(steer), "user"),
            calls.clone(),
            complete_call,
            0.0
        )
        .await?,
        "local"
    );
    assert_eq!(calls.0.load(Ordering::Relaxed), 3);
    Ok(())
}

#[tokio::test]
async fn tool_work_from_an_earlier_user_turn_is_not_handed_off() -> Result<()> {
    let route = route(|config| config.agentic_handoff = true)?;
    let sent = Arc::new(std::sync::Mutex::new(String::new()));
    let recorded = Arc::clone(&sent);
    test_drive_with_models(
        route,
        session(vec![
            Message::text(Role::User, "fix the build"),
            call("c1"),
            result("c1", "ok"),
            Message::text(Role::Assistant, "the build passes"),
            Message::text(Role::User, "What is 41 plus 42?"),
        ]),
        category_models(Category::Any, &["local", "cloud"]),
        move |target: ModelId, request: Request| {
            let recorded = Arc::clone(&recorded);
            async move {
                Ok(match target.as_str() {
                    "local" => reply("83"),
                    "cloud" => {
                        let text = serde_json::to_string(&request.llm_request.messages)
                            .unwrap_or_default();
                        recorded.lock().map(|mut sent| *sent = text).ok();
                        reply("83")
                    }
                    _ => scored(0.0),
                })
            }
        },
    )
    .await?;
    let sent = sent.lock().map(|sent| sent.clone()).unwrap_or_default();
    assert!(sent.contains("41 plus 42"), "escalated to the cloud");
    assert!(!sent.contains("Routing notice"));
    Ok(())
}

#[tokio::test]
async fn a_cloud_decision_falls_back_to_local_without_the_handoff_note() -> Result<()> {
    use crate::core::algorithm::Call;
    use crate::core::testing::serve_decision;

    let task = || {
        session(vec![
            Message::text(Role::User, "fix the build"),
            call("c1"),
            result("c1", "ok"),
        ])
    };
    let noted = |request: &Request| {
        serde_json::to_string(&request.llm_request.messages)
            .is_ok_and(|text| text.contains("Routing notice"))
    };
    for killed in [false, true] {
        let switch = super::super::KillSwitch::new();
        let algorithm = route(|config| {
            config.agentic_handoff = true;
            config.kill_switch = Some(switch.clone());
        })?;
        drive(
            &algorithm,
            session(vec![Message::text(Role::User, "fix the build")]),
            Arc::new(Calls::default()),
            complete_call,
            0.0,
        )
        .await?;
        if killed {
            switch.engage();
        }
        let outcome = crate::drive(
            algorithm,
            task(),
            Arc::new(category_models(Category::Any, &["local", "cloud"]).into()),
            |call| async move {
                match call {
                    Call::Model(call) => {
                        let response = match call.models[0].as_str() {
                            "local" => reply("the build passes"),
                            "judge" => scored(0.0),
                            _ => reply("no"),
                        };
                        call.respond(std::future::ready(Ok(response))).await
                    }
                    Call::Decision(call) => serve_decision(*call).await,
                }
            },
        )
        .await?;
        assert!(noted(&outcome.request));
        if killed {
            assert_eq!(outcome.selected_model_ids, vec![ModelId::new("cloud")]);
            assert!(outcome.fallback_request.is_none());
        } else {
            assert_eq!(
                outcome.selected_model_ids,
                vec![ModelId::new("cloud"), ModelId::new("local")]
            );
            let fallback = outcome.fallback_request.expect("a local fallback request");
            assert_eq!(fallback.llm_request.messages, task().llm_request.messages);
        }
    }
    Ok(())
}
