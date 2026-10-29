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
use super::{FOLLOW_UP_QUIET_SECS, LAST_ANSWER_FINAL_KEY, answers_final_reply, unix_seconds};
use crate::Result;
use crate::algorithms::vgr::Vgr;
use crate::core::algorithm::{Algorithm, AnswerSummary};
use crate::core::state::{State, StateValue};
use crate::core::testing::{category_models, reply, test_drive_with_models};

fn route(configure: impl FnOnce(&mut VgrConfig)) -> Result<Arc<dyn Algorithm>> {
    let mut config = VgrConfig::new(ModelId::new("efficient"), ModelId::new("capable"));
    config.targets.judge = Some(ModelId::new("judge"));
    config.targets.capable_judge = Some(ModelId::new("capable-judge"));
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
    efficient: fn() -> Response,
    judge_p: f64,
) -> Result<String> {
    let (selected, _) = test_drive_with_models(
        Arc::clone(route),
        request,
        category_models(Category::Any, &["efficient", "capable"]),
        move |target: ModelId, _| {
            let calls = Arc::clone(&calls);
            async move {
                match target.as_str() {
                    "efficient" => calls.0.fetch_add(1, Ordering::Relaxed),
                    "capable-judge" => calls.1.fetch_add(1, Ordering::Relaxed),
                    _ => 0,
                };
                Ok(match target.as_str() {
                    "efficient" => efficient(),
                    "judge" => scored(judge_p),
                    "capable-judge" => reply("yes"),
                    _ => reply("capable answer"),
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
        "capable"
    );
    assert_eq!(
        drive(&route, task(), calls, complete_call, 0.1).await?,
        "efficient"
    );
    Ok(())
}

#[tokio::test]
async fn a_truncated_efficient_attempt_is_judged_before_it_escalates() -> Result<()> {
    let route = route(|config| {
        config.targets.judge = None;
        config.targets.capable_judge = None;
    })?;
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
    assert_eq!(selected, "capable");
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
        "efficient"
    );
    assert_eq!(
        drive(&route, task(), calls.clone(), complete_call, 0.9).await?,
        "capable"
    );
    assert_eq!(calls.0.load(Ordering::Relaxed), 2);

    // Latched past the user turn: no further efficient attempt is produced.
    assert_eq!(
        drive(&route, task(), calls.clone(), complete_call, 0.0).await?,
        "capable"
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
        "efficient"
    );
    assert_eq!(
        drive(&clean, run("ok"), calls.clone(), answer, 0.1).await?,
        "capable"
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
        "capable"
    );
    assert_eq!(calls.1.load(Ordering::Relaxed), 0);
    let confirmed = route(|config| config.confirmed_recovery_min_clean_tail = Some(1))?;
    assert_eq!(
        drive(&confirmed, session(long), calls.clone(), answer, 0.99).await?,
        "efficient"
    );
    assert_eq!(calls.1.load(Ordering::Relaxed), 1);
    Ok(())
}

#[tokio::test]
async fn a_typed_follow_up_answer_is_judged_on_the_answer_itself() -> Result<()> {
    let route = route(|config| config.task_typing = true)?;
    let conversation = session(vec![
        Message::text(Role::User, "fix the build"),
        call("c1"),
        result("c1", "ok"),
        Message::text(Role::Assistant, "The build passes."),
        Message::text(Role::User, "What is 1+1?"),
    ]);
    // A confident readout serves on the efficient tier with no capable-tier call; a weak one asks the
    // capable tier's answer verifier, with no efficient deliberation first.
    for (readout, expected_calls) in [
        (0.95, vec!["judge: typing", "judge: answer"]),
        (
            0.5,
            vec!["judge: typing", "judge: answer", "capable-judge: answer"],
        ),
    ] {
        let calls = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let seen = Arc::clone(&calls);
        let (selected, _) = test_drive_with_models(
            Arc::clone(&route),
            conversation.clone(),
            category_models(Category::Any, &["efficient", "capable"]),
            move |target: ModelId, request: Request| {
                let seen = Arc::clone(&seen);
                async move {
                    let system = request
                        .llm_request
                        .instructions
                        .first()
                        .and_then(|block| block.content.first())
                        .and_then(|block| match block {
                            ContentBlock::Text { text } => Some(text.clone()),
                            _ => None,
                        })
                        .unwrap_or_default();
                    let question = if system.starts_with("You are a strict answer verifier") {
                        "answer"
                    } else if request
                        .llm_request
                        .extensions
                        .fields
                        .contains_key("logprobs")
                    {
                        "evidence"
                    } else {
                        "typing"
                    };
                    if target.as_str().contains("judge") {
                        seen.lock().push(format!("{}: {question}", target.as_str()));
                    }
                    Ok(match target.as_str() {
                        "efficient" => reply("2"),
                        "judge" if question == "typing" => reply("Answer"),
                        "judge" => scored(readout),
                        "capable-judge" => reply("yes"),
                        _ => reply("capable answer"),
                    })
                }
            },
        )
        .await?;
        assert_eq!(selected.as_str(), "efficient", "readout {readout}");
        assert_eq!(*calls.lock(), expected_calls, "readout {readout}");
    }
    Ok(())
}

#[tokio::test]
async fn an_unavailable_efficient_tier_escalates_and_other_failures_surface() -> Result<()> {
    let route = route(|_| {})?;
    for (status, escalates) in [(503, true), (400, false)] {
        let outcome = test_drive_with_models(
            Arc::clone(&route),
            session(vec![Message::text(Role::User, "task")]),
            category_models(Category::Any, &["efficient", "capable"]),
            move |target: ModelId, _| async move {
                match target.as_str() {
                    "efficient" => Err(LlmClientError::UpstreamHttp {
                        status: http::StatusCode::from_u16(status).unwrap_or_default(),
                        body: String::new(),
                    }),
                    _ => Ok(reply("capable answer")),
                }
            },
        )
        .await;
        assert_eq!(
            outcome.is_ok_and(|(selected, _)| selected == "capable"),
            escalates
        );
    }
    Ok(())
}

fn recovering_classifier() -> super::VgrClassifier {
    let mut config = VgrConfig::new(ModelId::new("efficient"), ModelId::new("capable"));
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

    let mut config = VgrConfig::new(ModelId::new("efficient"), ModelId::new("capable"));
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
    assert_eq!(pending_call.models, vec![ModelId::new("efficient")]);
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
async fn cancelling_efficient_generation_releases_the_recovery_trial() -> Result<()> {
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
async fn decisions_attach_the_reason_and_exclude_efficient_capable_fallbacks() -> Result<()> {
    use crate::core::algorithm::Call;
    use crate::core::testing::serve_decision;

    for (mode, reason, served) in [
        (ServingMode::Off, "mode_off", "capable"),
        (ServingMode::Evaluate, "decided", "efficient"),
        (ServingMode::Shadow, "decided", "capable"),
    ] {
        let algorithm = route(|config| config.mode = mode)?;
        let outcome = crate::drive(
            algorithm,
            request(vec![Message::text(Role::User, "task")]),
            Arc::new(category_models(Category::Any, &["efficient", "capable"]).into()),
            |call| async move {
                match call {
                    Call::Model(call) => {
                        let response = if call.models[0] == ModelId::new("efficient") {
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
        assert_eq!(evidence["policy_version"], "1.2");
        if reason == "decided" {
            assert_eq!(evidence["branch"], "DefaultVerified");
        }
    }
    Ok(())
}

#[tokio::test]
async fn capable_only_work_does_not_get_an_efficient_handoff_notice() -> Result<()> {
    for scenario in ["shadow", "kill_switch", "efficient_turn_budget"] {
        let route = route(|config| {
            config.agentic_handoff = true;
            match scenario {
                "shadow" => config.mode = ServingMode::Shadow,
                "kill_switch" => {
                    let switch = super::super::safety::KillSwitch::new();
                    switch.engage();
                    config.kill_switch = Some(switch);
                }
                _ => config.efficient_turn_budget = Some(std::time::Duration::ZERO),
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
            category_models(Category::Any, &["efficient", "capable"]),
            move |target: ModelId, request: Request| async move {
                Ok(match target.as_str() {
                    "efficient" => complete_call(),
                    "judge" => scored(0.0),
                    _ => {
                        let text = serde_json::to_string(&request.llm_request.messages)
                            .unwrap_or_default();
                        assert!(!text.contains("Routing notice"), "{scenario}");
                        reply("capable answer")
                    }
                })
            },
        )
        .await?;
        assert_eq!(selected, "capable");
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
        "efficient"
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
            category_models(Category::Any, &["efficient", "capable"]),
            move |target: ModelId, request: Request| {
                let sent = Arc::clone(&sent);
                async move {
                    Ok(match target.as_str() {
                        "efficient" => complete_call(),
                        "judge" => scored(0.9),
                        _ => {
                            sent.lock().push(
                                serde_json::to_string(&request.llm_request.messages)
                                    .unwrap_or_default(),
                            );
                            reply("capable answer")
                        }
                    })
                }
            },
        )
        .await?;
        assert_eq!(selected, "capable");
    }
    let sent = sent.lock();
    assert_eq!(sent.len(), 2);
    assert!(sent[0].contains("Routing notice"));
    assert!(!sent[1].contains("Routing notice"));
    Ok(())
}

#[tokio::test]
async fn compact_handoff_is_optional_and_replays_before_retained_capable_routing() -> Result<()> {
    let previous = vec![
        Message::text(Role::User, "inspect the repository"),
        call("old"),
        result("old", "previous output"),
        Message::text(Role::Assistant, "inspection complete"),
    ];
    for compact_handoff in [None, Some(false), Some(true)] {
        let route = route(|config| {
            config.agentic_handoff = true;
            if let Some(enabled) = compact_handoff {
                config.compact_handoff = enabled;
            }
        })?;
        assert_eq!(
            drive(
                &route,
                session(
                    previous
                        .iter()
                        .cloned()
                        .chain([Message::text(Role::User, "fix the build")])
                        .collect(),
                ),
                Arc::new(Calls::default()),
                complete_call,
                0.0,
            )
            .await?,
            "efficient"
        );
        let sent = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let efficient_calls = Arc::new(AtomicUsize::new(0));
        for follow_up in [false, true] {
            let mut messages = previous.clone();
            messages.extend([
                Message::text(Role::User, "fix the build"),
                call("c1"),
                result("c1", "ok"),
            ]);
            if follow_up {
                messages.extend([call("capable-1"), result("capable-1", "capable output")]);
            }
            let sent = Arc::clone(&sent);
            let efficient_calls = Arc::clone(&efficient_calls);
            let (selected, _) = test_drive_with_models(
                Arc::clone(&route),
                session(messages),
                category_models(Category::Any, &["efficient", "capable"]),
                move |target: ModelId, request: Request| {
                    let sent = Arc::clone(&sent);
                    let efficient_calls = Arc::clone(&efficient_calls);
                    async move {
                        Ok(match target.as_str() {
                            "efficient" => {
                                efficient_calls.fetch_add(1, Ordering::Relaxed);
                                reply("the build passes")
                            }
                            "judge" => scored(0.0),
                            "capable" => {
                                sent.lock().push(request.llm_request.messages);
                                reply("capable answer")
                            }
                            _ => reply("no"),
                        })
                    }
                },
            )
            .await?;
            assert_eq!(selected, "capable");
        }
        assert_eq!(efficient_calls.load(Ordering::Relaxed), 1);
        let sent = sent.lock();
        assert_eq!(sent.len(), 2);
        if compact_handoff == Some(true) {
            assert_eq!(sent[0].len(), 5);
            assert_eq!(&sent[0][..4], previous.as_slice());
            assert_eq!(sent[1].len(), 7);
            assert_eq!(sent[0], sent[1][..5]);
            let digest = sent[0][4].text_content("").unwrap_or_default();
            assert!(digest.starts_with("fix the build"));
            assert!(digest.contains("<condensed_history>"));
            assert!(digest.contains("the build passes"));
            assert!(digest.contains("Routing notice"));
            assert_eq!(sent[1][5], call("capable-1"));
            assert_eq!(sent[1][6], result("capable-1", "capable output"));
        } else {
            let mut original = previous.clone();
            original.extend([
                Message::text(Role::User, "fix the build"),
                call("c1"),
                result("c1", "ok"),
            ]);
            assert_eq!(&sent[0][..7], original.as_slice());
            assert_eq!(sent[0].len(), 8);
            assert!(
                sent[0][7]
                    .text_content("")
                    .unwrap_or_default()
                    .contains("Routing notice")
            );
            let mut follow_up = original;
            follow_up.extend([call("capable-1"), result("capable-1", "capable output")]);
            assert_eq!(sent[1], follow_up);
        }
    }
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
        "efficient"
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
            category_models(Category::Any, &["efficient", "capable"]),
            move |target: ModelId, request: Request| {
                let sent = Arc::clone(&sent);
                async move {
                    Ok(match target.as_str() {
                        "efficient" => reply("the build passes"),
                        "judge" => scored(0.0),
                        "capable" => {
                            let text = serde_json::to_string(&request.llm_request.messages)
                                .unwrap_or_default();
                            sent.lock().push(text);
                            reply("capable answer")
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
async fn a_spent_efficient_turn_budget_skips_the_efficient_attempt() -> Result<()> {
    let route = route(|config| config.efficient_turn_budget = Some(std::time::Duration::ZERO))?;
    let calls = Arc::new(Calls::default());
    let task = session(vec![Message::text(Role::User, "fix the build")]);
    assert_eq!(
        drive(&route, task, calls.clone(), complete_call, 0.0).await?,
        "capable"
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
    super::set_flag(&mut state, super::EFFICIENT_SERVED_KEY, true);
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
async fn a_blank_prompt_escalates_without_an_efficient_attempt() -> Result<()> {
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
            "capable"
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
async fn an_agent_marked_nudge_keeps_a_single_escalation_on_capable() -> Result<()> {
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
        "capable"
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
    let models = Arc::new(category_models(Category::Any, &["efficient", "capable"]).into());
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
    assert_eq!(escalated.selected_model_ids, vec![ModelId::new("capable")]);

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
    assert_eq!(held.selected_model_ids, vec![ModelId::new("capable")]);
    let evidence = held
        .metadata
        .and_then(|metadata| metadata.evidence)
        .unwrap_or_default();
    assert_eq!(evidence["source"], "vgr");
    assert_eq!(evidence["reason_code"], "turn_verification_latched");
    assert_eq!(evidence["verdict"], "capable");
    Ok(())
}

#[tokio::test]
async fn shadow_rechecks_new_user_turns_with_session_latching_enabled() -> Result<()> {
    let route = route(|config| {
        config.mode = ServingMode::Shadow;
        config.latch_escalation = true;
    })?;
    let calls = Arc::new(Calls::default());
    for messages in [
        vec![Message::text(Role::User, "fix the build")],
        vec![
            Message::text(Role::User, "fix the build"),
            Message::text(Role::Assistant, "fixed"),
            Message::text(Role::User, "now add a test"),
        ],
    ] {
        assert_eq!(
            drive(&route, session(messages), calls.clone(), complete_call, 0.0).await?,
            "capable"
        );
    }
    assert_eq!(calls.0.load(Ordering::Relaxed), 2);
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
        "capable"
    );
    let follow_up = session(vec![
        Message::text(Role::User, "fix the build"),
        Message::text(Role::Assistant, "fixed"),
        Message::text(Role::User, "now add a test"),
    ]);
    assert_eq!(
        drive(&route, follow_up, calls.clone(), complete_call, 0.0).await?,
        "capable"
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
        "capable"
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
        "efficient"
    );
    assert_eq!(calls.0.load(Ordering::Relaxed), 3);
    Ok(())
}

#[tokio::test]
async fn tool_work_from_an_earlier_user_turn_is_not_handed_off() -> Result<()> {
    let route = route(|config| config.agentic_handoff = true)?;
    let (selected, _) = test_drive_with_models(
        route,
        session(vec![
            Message::text(Role::User, "fix the build"),
            call("c1"),
            result("c1", "ok"),
            Message::text(Role::Assistant, "the build passes"),
            Message::text(Role::User, "What is 41 plus 42?"),
        ]),
        category_models(Category::Any, &["efficient", "capable"]),
        |target: ModelId, request: Request| async move {
            Ok(match target.as_str() {
                "efficient" => reply("83"),
                "capable" => {
                    let text =
                        serde_json::to_string(&request.llm_request.messages).unwrap_or_default();
                    assert!(text.contains("41 plus 42"));
                    assert!(!text.contains("Routing notice"));
                    reply("83")
                }
                _ => scored(0.0),
            })
        },
    )
    .await?;
    assert_eq!(selected, "capable");
    Ok(())
}

#[tokio::test]
async fn a_capable_decision_falls_back_to_efficient_without_the_handoff_note() -> Result<()> {
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
            Arc::new(category_models(Category::Any, &["efficient", "capable"]).into()),
            |call| async move {
                match call {
                    Call::Model(call) => {
                        let response = match call.models[0].as_str() {
                            "efficient" => reply("the build passes"),
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
            assert_eq!(outcome.selected_model_ids, vec![ModelId::new("capable")]);
            assert!(outcome.fallback_request.is_none());
        } else {
            assert_eq!(
                outcome.selected_model_ids,
                vec![ModelId::new("capable"), ModelId::new("efficient")]
            );
            let fallback = outcome
                .fallback_request
                .expect("an efficient fallback request");
            assert_eq!(fallback.llm_request.messages, task().llm_request.messages);
        }
    }
    Ok(())
}

#[tokio::test]
async fn default_verification_asks_capable_evidence_then_answer() -> Result<()> {
    for (evidence, answer, expected) in [
        ("yes", "yes", "efficient"),
        ("yes", "no", "capable"),
        ("yes", "unclear", "capable"),
        ("no", "yes", "capable"),
        ("unclear", "yes", "capable"),
    ] {
        let route = route(|_| {})?;
        let calls = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let seen = Arc::clone(&calls);
        let (selected, _) = test_drive_with_models(
            route,
            session(vec![Message::text(Role::User, "explain quicksort")]),
            category_models(Category::Any, &["efficient", "capable"]),
            move |target: ModelId, request: Request| {
                let seen = Arc::clone(&seen);
                async move {
                    Ok(match target.as_str() {
                        "efficient" => reply("Choose a pivot and partition the input."),
                        "judge" => scored(0.1),
                        "capable-judge" => {
                            let is_answer = request
                                .llm_request
                                .instructions
                                .iter()
                                .flat_map(|block| &block.content)
                                .any(|block| {
                                    matches!(block, ContentBlock::Text { text }
                                    if text.starts_with("You are a strict answer verifier"))
                                });
                            seen.lock().push(is_answer);
                            reply(if is_answer { answer } else { evidence })
                        }
                        _ => reply("capable answer"),
                    })
                }
            },
        )
        .await?;
        assert_eq!(selected.as_str(), expected, "{evidence}, {answer}");
        assert_eq!(
            *calls.lock(),
            if evidence == "yes" {
                vec![false, true]
            } else {
                vec![false]
            }
        );
    }
    Ok(())
}

#[tokio::test]
async fn shadow_keeps_judging_a_turn_until_vgr_escalates_it() -> Result<()> {
    fn truncated() -> Response {
        tool_call(json!(r#"{"command":"l"#), StopReason::MaxTokens)
    }
    for latch_escalation in [false, true] {
        let route = route(|config| {
            config.mode = ServingMode::Shadow;
            config.latch_escalation = latch_escalation;
        })?;
        let calls = Arc::new(Calls::default());
        let mut messages = vec![Message::text(Role::User, "fix the build")];
        for (step, efficient) in [complete_call, complete_call, truncated, complete_call]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                drive(
                    &route,
                    session(messages.clone()),
                    calls.clone(),
                    efficient,
                    0.1
                )
                .await?,
                "capable"
            );
            let id = format!("c{step}");
            messages.extend([call(&id), result(&id, "ok")]);
        }
        assert_eq!(calls.0.load(Ordering::Relaxed), 3);
        messages.push(Message::text(Role::User, "keep going"));
        drive(
            &route,
            initiated_by(session(messages.clone()), "agent"),
            calls.clone(),
            complete_call,
            0.1,
        )
        .await?;
        assert_eq!(calls.0.load(Ordering::Relaxed), 3);
        drive(
            &route,
            initiated_by(session(messages), "user"),
            calls.clone(),
            complete_call,
            0.1,
        )
        .await?;
        assert_eq!(calls.0.load(Ordering::Relaxed), 4);
    }
    Ok(())
}

#[tokio::test]
async fn shadow_final_answer_commits_do_not_latch_but_rejections_do() -> Result<()> {
    let route = route(|config| {
        config.mode = ServingMode::Shadow;
        config.targets.capable_judge = None;
    })?;
    let calls = Arc::new(Calls::default());
    let mut messages = vec![Message::text(Role::User, "explain quicksort")];
    for probability in [0.9, 0.9, 0.1, 0.9] {
        let task = initiated_by(session(messages.clone()), "agent");
        assert_eq!(
            drive(
                &route,
                task,
                calls.clone(),
                || reply("Partition around a pivot."),
                probability
            )
            .await?,
            "capable"
        );
        messages.push(Message::text(Role::Assistant, "Partition around a pivot."));
        messages.push(Message::text(Role::User, "continue"));
    }
    assert_eq!(calls.0.load(Ordering::Relaxed), 3);
    Ok(())
}

#[tokio::test]
async fn client_recovery_messages_keep_the_turn_latch() -> Result<()> {
    let route = route(|_| {})?;
    let calls = Arc::new(Calls::default());
    let task = || session(vec![Message::text(Role::User, "fix the build")]);
    for _ in 0..2 {
        drive(&route, task(), calls.clone(), complete_call, 0.9).await?;
    }
    // Hermes-style recovery: a nudge after an empty answer, and one sent the moment a
    // complete-looking answer promised a tool call it never made.
    for (final_reply, previous, nudge) in [
        (
            false,
            "(empty)",
            "You just executed tool calls but returned an empty response.",
        ),
        (
            true,
            "Let me run the tests.",
            "Your previous turn indicated a tool call but none was included.",
        ),
    ] {
        route
            .observe_answer("session-1", AnswerSummary { final_reply })
            .await;
        let recovery = session(vec![
            Message::text(Role::User, "fix the build"),
            Message::text(Role::Assistant, previous),
            Message::text(Role::User, nudge),
        ]);
        assert_eq!(
            drive(&route, recovery, calls.clone(), complete_call, 0.0).await?,
            "capable"
        );
    }
    assert_eq!(calls.0.load(Ordering::Relaxed), 2);
    let next = initiated_by(
        session(vec![
            Message::text(Role::User, "fix the build"),
            Message::text(Role::Assistant, "Let me run the tests."),
            Message::text(Role::User, "now do another task"),
        ]),
        "user",
    );
    assert_eq!(
        drive(&route, next, calls.clone(), complete_call, 0.0).await?,
        "efficient"
    );
    assert_eq!(calls.0.load(Ordering::Relaxed), 3);
    Ok(())
}

#[test]
fn only_a_paused_reply_to_a_reported_final_answer_releases_the_latch() {
    let messages = [
        Message::text(Role::User, "fix the build"),
        Message::text(Role::Assistant, "fixed"),
        Message::text(Role::User, "now add a test"),
    ];
    let now = unix_seconds();
    for (reported, releases) in [
        (None, true),
        (Some(0), false),
        (Some(now), false),
        (Some(now - FOLLOW_UP_QUIET_SECS), true),
    ] {
        let mut state = State::default();
        if let Some(finished) = reported {
            state
                .extra
                .insert(LAST_ANSWER_FINAL_KEY.into(), StateValue::Count(finished));
        }
        assert_eq!(
            answers_final_reply(&state, &messages),
            releases,
            "{reported:?}"
        );
    }
}

#[tokio::test]
async fn a_failed_capable_verifier_still_offers_the_efficient_answer() -> Result<()> {
    use crate::core::algorithm::Call;
    use crate::core::testing::serve_decision;

    let algorithm = route(|config| config.task_typing = true)?;
    let task = || session(vec![Message::text(Role::User, "What is 1+1?")]);
    // Exercise production error propagation through CallModel::respond.
    let outcome = crate::drive(
        algorithm,
        task(),
        Arc::new(category_models(Category::Any, &["efficient", "capable"]).into()),
        |call| async move {
            match call {
                Call::Model(call) => {
                    let target = call.models[0].clone();
                    let readout = call
                        .request
                        .llm_request
                        .extensions
                        .fields
                        .contains_key("logprobs");
                    let response = match target.as_str() {
                        "efficient" => Ok(reply("2")),
                        "judge" if readout => Ok(scored(0.05)),
                        "judge" => Ok(reply("Answer")),
                        _ => Err(crate::LibsyError::client_call(
                            target,
                            LlmClientError::UpstreamHttp {
                                status: http::StatusCode::BAD_GATEWAY,
                                body: String::new(),
                            },
                        )),
                    };
                    call.respond(std::future::ready(response)).await
                }
                Call::Decision(call) => serve_decision(*call).await,
            }
        },
    )
    .await?;
    assert_eq!(
        outcome.selected_model_ids,
        vec![ModelId::new("capable"), ModelId::new("efficient")]
    );
    let fallback = outcome
        .fallback_request
        .expect("an efficient fallback request");
    assert_eq!(fallback.llm_request.messages, task().llm_request.messages);
    Ok(())
}
