// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Behavior tests for the advisor review gate.

use std::sync::atomic::{AtomicUsize, Ordering};

use switchyard_protocol::{ResponseOutput, ToolCall, ToolResult, completion_text};

use futures::StreamExt;
use switchyard_protocol::{
    AggLlmResponse, LlmClientError, LlmResponse, LlmResponseChunk, LlmResponseStreamEvent, ModelId,
    StopReason,
};

use super::transcript::{NO_TEXT_PLACEHOLDER, TRUNCATION_MARKER, middle_drop};
use super::*;
use crate::core::testing::{reply, test_drive};

const EXECUTOR: &str = "executor";
const ADVISOR: &str = "advisor";

fn target(name: &str) -> ModelId {
    ModelId::new(name)
}

fn gate(config: AdvisorGateConfig) -> Arc<dyn Algorithm> {
    Arc::new(
        AdvisorGate::new(target(EXECUTOR), target(ADVISOR), config).expect("test config is valid"),
    )
}

fn request(messages: Vec<Message>) -> Request {
    Request {
        llm_request: LlmRequest {
            model: Some("gated".to_string()),
            messages,
            ..LlmRequest::default()
        },
        raw_request: None,
        metadata: None,
    }
}

fn task_request() -> Request {
    request(vec![Message::text(Role::User, "build X")])
}

fn with_bench_header(mut request: Request, id: &str) -> Request {
    let mut headers = http::HeaderMap::new();
    headers.insert(BENCH_SESSION_HEADER, id.parse().expect("header value"));
    let mut metadata = request.metadata.unwrap_or_default();
    metadata.http_headers = Some(headers);
    request.metadata = Some(metadata);
    request
}

fn with_session_id(mut request: Request, id: &str) -> Request {
    let mut metadata = request.metadata.unwrap_or_default();
    metadata.session_id = Some(id.to_string());
    request.metadata = Some(metadata);
    request
}

fn tool_call_turn() -> Response {
    Response {
        llm_response: LlmResponse::Agg(AggLlmResponse {
            outputs: vec![ResponseOutput {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolCall(ToolCall {
                    id: "t1".to_string(),
                    name: "bash".to_string(),
                    arguments: serde_json::json!({}),
                })],
                stop_reason: None,
            }],
            ..AggLlmResponse::default()
        }),
        metadata: None,
    }
}

fn tool_use_stop_turn() -> Response {
    Response {
        llm_response: LlmResponse::Agg(AggLlmResponse {
            outputs: vec![ResponseOutput {
                role: Role::Assistant,
                content: vec![ContentBlock::Text {
                    text: "calling a tool".to_string(),
                }],
                stop_reason: Some(StopReason::ToolUse),
            }],
            ..AggLlmResponse::default()
        }),
        metadata: None,
    }
}

fn reasoning_only_turn() -> Response {
    Response {
        llm_response: LlmResponse::Agg(AggLlmResponse {
            outputs: vec![ResponseOutput {
                role: Role::Assistant,
                content: vec![ContentBlock::Reasoning {
                    text: "thinking about it".to_string(),
                    signature: None,
                }],
                stop_reason: None,
            }],
            ..AggLlmResponse::default()
        }),
        metadata: None,
    }
}

fn empty_turn() -> Response {
    Response {
        llm_response: LlmResponse::Agg(AggLlmResponse {
            outputs: vec![ResponseOutput {
                role: Role::Assistant,
                content: Vec::new(),
                stop_reason: None,
            }],
            ..AggLlmResponse::default()
        }),
        metadata: None,
    }
}

fn streamed(events: Vec<LlmResponseStreamEvent>) -> Response {
    Response {
        llm_response: LlmResponse::Stream(Box::pin(futures::stream::iter(
            events.into_iter().map(Ok),
        ))),
        metadata: None,
    }
}

fn text_stream_events(text: &str) -> Vec<LlmResponseStreamEvent> {
    vec![
        LlmResponseStreamEvent::preserved(
            "anthropic_messages",
            serde_json::json!({"type": "message_start"}),
            vec![LlmResponseChunk::MessageStart {
                id: Some("m1".to_string()),
                model: Some("exec-upstream".to_string()),
            }],
        ),
        LlmResponseStreamEvent::preserved(
            "anthropic_messages",
            serde_json::json!({"type": "content_block_delta", "text": text}),
            vec![LlmResponseChunk::TextDelta {
                index: 0,
                text: text.to_string(),
            }],
        ),
        LlmResponseStreamEvent::preserved(
            "anthropic_messages",
            serde_json::json!({"type": "message_stop"}),
            vec![LlmResponseChunk::MessageStop {
                reason: Some("end_turn".to_string()),
            }],
        ),
    ]
}

/// Serve that answers the advisor with a fixed verdict and the executor
/// from a per-call script, recording every call.
struct Script {
    calls: Arc<parking_lot::Mutex<Vec<(String, Request)>>>,
    executor_calls: Arc<AtomicUsize>,
}

impl Script {
    fn new() -> Self {
        Self {
            calls: Arc::new(parking_lot::Mutex::new(Vec::new())),
            executor_calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn models(&self) -> Vec<String> {
        self.calls
            .lock()
            .iter()
            .map(|(model, _)| model.clone())
            .collect()
    }

    fn advisor_consults(&self) -> usize {
        self.calls
            .lock()
            .iter()
            .filter(|(model, _)| model == ADVISOR)
            .count()
    }

    fn call(&self, index: usize) -> Request {
        self.calls.lock()[index].1.clone()
    }

    /// Serve executor turns from `executor` (indexed per executor call)
    /// and advisor consults with `verdict`.
    fn serve(
        &self,
        verdict: &str,
        executor: impl Fn(usize) -> Response + Send + Sync + 'static,
    ) -> impl Fn(
        Decision,
        Request,
    ) -> futures::future::BoxFuture<
        'static,
        std::result::Result<Response, LlmClientError>,
    > + Send
    + Sync
    + 'static {
        let calls = Arc::clone(&self.calls);
        let executor_calls = Arc::clone(&self.executor_calls);
        let verdict = verdict.to_string();
        let executor = Arc::new(executor);
        move |decision: Decision, request: Request| {
            let calls = Arc::clone(&calls);
            let executor_calls = Arc::clone(&executor_calls);
            let verdict = verdict.clone();
            let executor = Arc::clone(&executor);
            Box::pin(async move {
                let model = decision.selected_model_id().to_string();
                calls.lock().push((model.clone(), request));
                if model == ADVISOR {
                    Ok(reply(verdict))
                } else {
                    let index = executor_calls.fetch_add(1, Ordering::SeqCst);
                    Ok(executor(index))
                }
            })
        }
    }
}

async fn agg_of(response: Response) -> AggLlmResponse {
    response
        .llm_response
        .into_agg()
        .await
        .expect("test response aggregates")
}

// ── Gate behavior ───────────────────────────────────────────────────────

#[tokio::test]
async fn tool_call_turn_replays_without_review() {
    for turn in [tool_call_turn(), tool_use_stop_turn()] {
        let script = Script::new();
        let gate = gate(AdvisorGateConfig::default());
        let serve = script.serve("APPROVE", {
            let turn = parking_lot::Mutex::new(Some(turn));
            move |_| turn.lock().take().expect("one executor call")
        });
        let (_, response) = test_drive(gate, task_request(), serve)
            .await
            .expect("routes");
        assert_eq!(script.models(), vec![EXECUTOR.to_string()]);
        assert!(has_tool_use(&agg_of(response).await));
    }
}

#[tokio::test]
async fn approved_terminal_turn_returns_buffered_body() {
    let script = Script::new();
    let gate = gate(AdvisorGateConfig::default());
    let serve = script.serve("APPROVE", |_| reply("all done"));
    let (trace, response) = test_drive(gate, task_request(), serve)
        .await
        .expect("routes");
    assert_eq!(
        script.models(),
        vec![EXECUTOR.to_string(), ADVISOR.to_string()]
    );
    assert_eq!(completion_text(&agg_of(response).await), "all done");
    // The published trace ends on the executor so hosts attribute the
    // served model correctly.
    let last = trace.last().expect("decision published");
    assert_eq!(last.selected_model_id(), EXECUTOR);
    assert!(last.is_answer_call());
}

#[tokio::test]
async fn advisor_consult_is_not_an_answer_call() {
    let script = Script::new();
    let gate = gate(AdvisorGateConfig::default());
    let consult_shape = Arc::new(parking_lot::Mutex::new(None));
    let shape = Arc::clone(&consult_shape);
    let calls = Arc::clone(&script.calls);
    let serve = move |decision: Decision, request: Request| {
        let shape = Arc::clone(&shape);
        let calls = Arc::clone(&calls);
        Box::pin(async move {
            calls
                .lock()
                .push((decision.selected_model_id().to_string(), request));
            if decision.selected_model_id() == ADVISOR {
                *shape.lock() = Some(decision.is_answer_call());
                Ok(reply("APPROVE"))
            } else {
                Ok(reply("done"))
            }
        })
            as futures::future::BoxFuture<'static, std::result::Result<Response, LlmClientError>>
    };
    test_drive(gate, task_request(), serve)
        .await
        .expect("routes");
    assert_eq!(*consult_shape.lock(), Some(false));
}

#[tokio::test]
async fn redo_appends_echo_and_feedback_then_reinvokes() {
    let script = Script::new();
    let gate = gate(AdvisorGateConfig::default());
    let serve = script.serve("REDO: run the tests", |index| {
        if index == 0 {
            reply("first attempt")
        } else {
            reply("continued")
        }
    });
    let (_, response) = test_drive(gate, task_request(), serve)
        .await
        .expect("routes");
    assert_eq!(
        script.models(),
        vec![
            EXECUTOR.to_string(),
            ADVISOR.to_string(),
            EXECUTOR.to_string()
        ]
    );
    assert_eq!(completion_text(&agg_of(response).await), "continued");
    let redo = script.call(2);
    let messages = &redo.llm_request.messages;
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[1].role, Role::Assistant);
    assert_eq!(
        messages[1].text_content("\n").as_deref(),
        Some("first attempt")
    );
    assert_eq!(messages[2].role, Role::User);
    let feedback = messages[2].text_content("\n").expect("feedback text");
    assert!(feedback.starts_with(REDO_FEEDBACK_PREFIX));
    assert!(feedback.ends_with("run the tests"));
    assert!(redo.llm_request.preservation.requests.is_empty());
}

#[tokio::test]
async fn budget_consumed_once_per_scope() {
    let script = Script::new();
    let gate = gate(AdvisorGateConfig::default());
    let serve = script.serve("APPROVE", |_| reply("done"));
    test_drive(Arc::clone(&gate), task_request(), serve)
        .await
        .expect("first run");
    let serve = script.serve("APPROVE", |_| reply("done again"));
    let (_, response) = test_drive(gate, task_request(), serve)
        .await
        .expect("second run");
    // Headerless requests share the instance scope: exactly one consult.
    assert_eq!(script.advisor_consults(), 1);
    assert_eq!(completion_text(&agg_of(response).await), "done again");
}

#[tokio::test]
async fn budget_keyed_by_bench_header_not_conversation() {
    let script = Script::new();
    let gate = gate(AdvisorGateConfig::default());
    for turn in ["build X", "now build Y", "and Z"] {
        let serve = script.serve("APPROVE", |_| reply("done"));
        let request = with_bench_header(request(vec![Message::text(Role::User, turn)]), "eval-1");
        test_drive(Arc::clone(&gate), request, serve)
            .await
            .expect("routes");
    }
    assert_eq!(script.advisor_consults(), 1);
}

#[tokio::test]
async fn scope_precedence_header_over_session_id() {
    let script = Script::new();
    let gate = gate(AdvisorGateConfig::default());
    // Same bench header, different host session ids: one scope.
    for session in ["s1", "s2"] {
        let serve = script.serve("APPROVE", |_| reply("done"));
        let request = with_bench_header(with_session_id(task_request(), session), "eval-1");
        test_drive(Arc::clone(&gate), request, serve)
            .await
            .expect("routes");
    }
    assert_eq!(script.advisor_consults(), 1);
    // Distinct session ids without the header: distinct scopes.
    for session in ["s3", "s4"] {
        let serve = script.serve("APPROVE", |_| reply("done"));
        test_drive(
            Arc::clone(&gate),
            with_session_id(task_request(), session),
            serve,
        )
        .await
        .expect("routes");
    }
    assert_eq!(script.advisor_consults(), 3);
}

#[tokio::test]
async fn max_reviews_two_reviews_then_passthrough() {
    let script = Script::new();
    let gate = gate(AdvisorGateConfig {
        max_reviews: 2,
        ..AdvisorGateConfig::default()
    });
    for _ in 0..3 {
        let serve = script.serve("APPROVE", |_| reply("done"));
        test_drive(Arc::clone(&gate), task_request(), serve)
            .await
            .expect("routes");
    }
    assert_eq!(script.advisor_consults(), 2);
}

#[tokio::test]
async fn exhausted_scope_passes_live_stream_through() {
    let script = Script::new();
    let gate = gate(AdvisorGateConfig::default());
    let serve = script.serve("APPROVE", |_| reply("done"));
    test_drive(Arc::clone(&gate), task_request(), serve)
        .await
        .expect("spends budget");
    // Post-budget turns pass through as the live stream, events verbatim.
    let events = text_stream_events("streamed continuation");
    let expected = serde_json::to_value(&events).expect("events serialize");
    let serve = script.serve("APPROVE", {
        let events = parking_lot::Mutex::new(Some(events));
        move |_| streamed(events.lock().take().expect("one executor call"))
    });
    let (_, response) = test_drive(gate, task_request(), serve)
        .await
        .expect("routes");
    let LlmResponse::Stream(stream) = response.llm_response else {
        panic!("expected a live stream");
    };
    let replayed: Vec<LlmResponseStreamEvent> = stream
        .map(|item| item.expect("stream item"))
        .collect()
        .await;
    assert_eq!(
        serde_json::to_value(&replayed).expect("serialize"),
        expected
    );
    assert_eq!(script.advisor_consults(), 1);
}

// ── Failure paths ───────────────────────────────────────────────────────

fn failing_advisor(
    script: &Script,
    executor_reply: &'static str,
) -> impl Fn(
    Decision,
    Request,
) -> futures::future::BoxFuture<'static, std::result::Result<Response, LlmClientError>>
+ Send
+ Sync
+ 'static {
    let calls = Arc::clone(&script.calls);
    move |decision: Decision, request: Request| {
        let calls = Arc::clone(&calls);
        Box::pin(async move {
            let model = decision.selected_model_id().to_string();
            calls.lock().push((model.clone(), request));
            if model == ADVISOR {
                Err(LlmClientError::General("advisor down".to_string()))
            } else {
                Ok(reply(executor_reply))
            }
        })
    }
}

#[tokio::test]
async fn fail_open_returns_turn_refunds_and_caps_failures() {
    let script = Script::new();
    let gate = gate(AdvisorGateConfig::default());
    // Three failed consults: each returns the turn and refunds the budget.
    for _ in 0..3 {
        let (_, response) = test_drive(
            Arc::clone(&gate),
            task_request(),
            failing_advisor(&script, "done"),
        )
        .await
        .expect("fail-open run");
        assert_eq!(completion_text(&agg_of(response).await), "done");
    }
    assert_eq!(script.advisor_consults(), 3);
    // The failure cap now stops consulting entirely.
    test_drive(
        Arc::clone(&gate),
        task_request(),
        failing_advisor(&script, "done"),
    )
    .await
    .expect("passthrough run");
    assert_eq!(script.advisor_consults(), 3);
    // A recovered advisor is never consulted again in this scope.
    let serve = script.serve("APPROVE", |_| reply("done"));
    test_drive(gate, task_request(), serve)
        .await
        .expect("still passthrough");
    assert_eq!(script.advisor_consults(), 3);
}

#[tokio::test]
async fn fail_closed_propagates_refunds_and_counts() {
    let script = Script::new();
    let gate = gate(AdvisorGateConfig {
        fail_open: false,
        ..AdvisorGateConfig::default()
    });
    for _ in 0..3 {
        let error = match test_drive(
            Arc::clone(&gate),
            task_request(),
            failing_advisor(&script, "done"),
        )
        .await
        {
            Err(error) => error,
            Ok(_) => panic!("fail-closed surfaces the advisor error"),
        };
        // Wrapped as an algorithm failure so the host renders a 5xx, not
        // the advisor's own (possibly context-window-shaped) client error.
        assert!(matches!(error, LibsyError::AlgorithmError { .. }));
        assert!(error.to_string().contains("advisor consult failed"));
    }
    assert_eq!(script.advisor_consults(), 3);
    // The failure cap bounds fail-closed too: the scope stops consulting
    // and the executor turn flows again.
    let (_, response) = test_drive(gate, task_request(), failing_advisor(&script, "recovered"))
        .await
        .expect("post-cap passthrough");
    assert_eq!(script.advisor_consults(), 3);
    assert_eq!(completion_text(&agg_of(response).await), "recovered");
}

#[tokio::test]
async fn unparseable_verdict_refunds_and_approves() {
    let script = Script::new();
    let gate = gate(AdvisorGateConfig::default());
    let serve = script.serve("I cannot approve this — REDO: run the tests", |_| {
        reply("done")
    });
    let (_, response) = test_drive(Arc::clone(&gate), task_request(), serve)
        .await
        .expect("unparseable run");
    assert_eq!(completion_text(&agg_of(response).await), "done");
    // The refunded budget admits another review.
    let serve = script.serve("APPROVE", |_| reply("done"));
    test_drive(gate, task_request(), serve)
        .await
        .expect("second run");
    assert_eq!(script.advisor_consults(), 2);
}

#[tokio::test]
async fn context_window_error_propagates() {
    let gate = gate(AdvisorGateConfig::default());
    let serve = |_decision: Decision, _request: Request| async move {
        Err(LlmClientError::ContextWindowExceeded {
            model: "exec-upstream".into(),
            message: "prompt is too long".to_string(),
        })
    };
    let error = match test_drive(gate, task_request(), serve).await {
        Err(error) => error,
        Ok(_) => panic!("context-window error propagates"),
    };
    assert!(matches!(
        error,
        LibsyError::ClientCall {
            source: LlmClientError::ContextWindowExceeded { .. },
            ..
        }
    ));
}

#[tokio::test]
async fn mid_stream_error_propagates_while_buffering() {
    let gate = gate(AdvisorGateConfig::default());
    let serve = |_decision: Decision, _request: Request| async move {
        Ok(streamed(vec![
            LlmResponseStreamEvent::new(vec![LlmResponseChunk::TextDelta {
                index: 0,
                text: "partial".to_string(),
            }]),
            LlmResponseStreamEvent::new(vec![LlmResponseChunk::StreamError {
                message: "upstream reset".to_string(),
            }]),
        ]))
    };
    let error = match test_drive(gate, task_request(), serve).await {
        Err(error) => error,
        Ok(_) => panic!("mid-stream error propagates"),
    };
    assert!(matches!(
        error,
        LibsyError::ClientCall {
            source: LlmClientError::UpstreamHttp { status: 502, .. },
            ..
        }
    ));
}

// ── Streaming ───────────────────────────────────────────────────────────

#[tokio::test]
async fn streamed_approval_replays_preserved_events_verbatim() {
    let script = Script::new();
    let gate = gate(AdvisorGateConfig::default());
    let events = text_stream_events("the answer");
    let expected = serde_json::to_value(&events).expect("events serialize");
    let serve = script.serve("APPROVE", {
        let events = parking_lot::Mutex::new(Some(events));
        move |_| streamed(events.lock().take().expect("one executor call"))
    });
    let (_, response) = test_drive(gate, task_request(), serve)
        .await
        .expect("routes");
    assert_eq!(script.advisor_consults(), 1);
    let LlmResponse::Stream(stream) = response.llm_response else {
        panic!("expected replayed stream");
    };
    let replayed: Vec<LlmResponseStreamEvent> = stream
        .map(|item| item.expect("stream item"))
        .collect()
        .await;
    assert_eq!(
        serde_json::to_value(&replayed).expect("serialize"),
        expected
    );
}

#[tokio::test]
async fn streamed_tool_call_turn_replays_without_review() {
    let script = Script::new();
    let gate = gate(AdvisorGateConfig::default());
    let events = vec![LlmResponseStreamEvent::new(vec![
        LlmResponseChunk::ToolCallDelta {
            index: 0,
            id: Some("t1".to_string()),
            name: Some("bash".to_string()),
            arguments_delta: Some("{}".to_string()),
        },
    ])];
    let serve = script.serve("APPROVE", {
        let events = parking_lot::Mutex::new(Some(events));
        move |_| streamed(events.lock().take().expect("one executor call"))
    });
    test_drive(gate, task_request(), serve)
        .await
        .expect("routes");
    assert_eq!(script.models(), vec![EXECUTOR.to_string()]);
}

// ── Triggers ────────────────────────────────────────────────────────────

fn pattern_config() -> AdvisorGateConfig {
    AdvisorGateConfig {
        gate_trigger: GateTrigger::Pattern(r#"task_complete["\s>:]*true"#.to_string()),
        ..AdvisorGateConfig::default()
    }
}

#[tokio::test]
async fn pattern_trigger_gates_matching_text_only() {
    let script = Script::new();
    let gate = gate(pattern_config());
    // Non-matching turns pass through without a consult.
    let serve = script.serve("APPROVE", |_| reply("still working"));
    test_drive(Arc::clone(&gate), task_request(), serve)
        .await
        .expect("routes");
    assert_eq!(script.advisor_consults(), 0);
    // The declared completion gates.
    let serve = script.serve("APPROVE", |_| reply("task_complete: true"));
    test_drive(gate, task_request(), serve)
        .await
        .expect("routes");
    assert_eq!(script.advisor_consults(), 1);
}

#[tokio::test]
async fn pattern_trigger_matches_on_tool_call_turns() {
    // The pattern trigger reads text only; tool use does not exempt a turn.
    let script = Script::new();
    let gate = gate(pattern_config());
    let turn = Response {
        llm_response: LlmResponse::Agg(AggLlmResponse {
            outputs: vec![ResponseOutput {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::Text {
                        text: "task_complete: true".to_string(),
                    },
                    ContentBlock::ToolCall(ToolCall {
                        id: "t1".to_string(),
                        name: "bash".to_string(),
                        arguments: serde_json::json!({}),
                    }),
                ],
                stop_reason: Some(StopReason::ToolUse),
            }],
            ..AggLlmResponse::default()
        }),
        metadata: None,
    };
    let serve = script.serve("APPROVE", {
        let turn = parking_lot::Mutex::new(Some(turn));
        move |_| turn.lock().take().expect("one executor call")
    });
    test_drive(gate, task_request(), serve)
        .await
        .expect("routes");
    assert_eq!(script.advisor_consults(), 1);
}

#[tokio::test]
async fn min_tool_results_defers_gate() {
    let script = Script::new();
    let gate = gate(AdvisorGateConfig {
        gate_min_tool_results: 1,
        ..AdvisorGateConfig::default()
    });
    // Terminal turn before any tool result: passes through unreviewed.
    let serve = script.serve("APPROVE", |_| reply("plan: do X"));
    test_drive(Arc::clone(&gate), task_request(), serve)
        .await
        .expect("routes");
    assert_eq!(script.advisor_consults(), 0);
    // Once the conversation carries a tool result, the gate fires.
    let with_result = request(vec![
        Message::text(Role::User, "build X"),
        Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult(ToolResult {
                tool_call_id: "t1".to_string(),
                content: vec![ContentBlock::Text {
                    text: "ok".to_string(),
                }],
                is_error: None,
            })],
        },
    ]);
    let serve = script.serve("APPROVE", |_| reply("done"));
    test_drive(gate, with_result, serve).await.expect("routes");
    assert_eq!(script.advisor_consults(), 1);
}

#[tokio::test]
async fn stall_checkpoint_reviews_mid_task_once() {
    let script = Script::new();
    let gate = gate(AdvisorGateConfig {
        gate_stall_turns: 2,
        max_reviews: 2,
        ..AdvisorGateConfig::default()
    });
    let grinding = || {
        request(vec![
            Message::text(Role::User, "build X"),
            Message::text(Role::Assistant, "step 1"),
            Message::text(Role::Assistant, "step 2"),
        ])
    };
    // A tool-call turn is not terminal, but the stall checkpoint reviews it.
    let serve = script.serve("APPROVE", {
        let turn = parking_lot::Mutex::new(Some(tool_call_turn()));
        move |_| turn.lock().take().expect("one executor call")
    });
    test_drive(Arc::clone(&gate), grinding(), serve)
        .await
        .expect("routes");
    assert_eq!(script.advisor_consults(), 1);
    // The latch keeps the same conversation from stalling twice.
    let serve = script.serve("APPROVE", {
        let turn = parking_lot::Mutex::new(Some(tool_call_turn()));
        move |_| turn.lock().take().expect("one executor call")
    });
    test_drive(gate, grinding(), serve).await.expect("routes");
    assert_eq!(script.advisor_consults(), 1);
}

#[tokio::test]
async fn simultaneous_trigger_does_not_latch_stall() {
    let script = Script::new();
    let gate = gate(AdvisorGateConfig {
        gate_stall_turns: 1,
        max_reviews: 2,
        ..AdvisorGateConfig::default()
    });
    let conversation = || {
        request(vec![
            Message::text(Role::User, "build X"),
            Message::text(Role::Assistant, "step 1"),
        ])
    };
    // Terminal turn and stall coincide: the trigger review runs, the
    // stall does not latch.
    let serve = script.serve("APPROVE", |_| reply("done"));
    test_drive(Arc::clone(&gate), conversation(), serve)
        .await
        .expect("routes");
    assert_eq!(script.advisor_consults(), 1);
    // The unlatched stall still fires later on a tool-call turn.
    let serve = script.serve("APPROVE", {
        let turn = parking_lot::Mutex::new(Some(tool_call_turn()));
        move |_| turn.lock().take().expect("one executor call")
    });
    test_drive(gate, conversation(), serve)
        .await
        .expect("routes");
    assert_eq!(script.advisor_consults(), 2);
}

// ── Reasoning-only and empty turns ──────────────────────────────────────

#[tokio::test]
async fn reasoning_only_turn_reviewed_and_echoed() {
    let script = Script::new();
    let gate = gate(AdvisorGateConfig::default());
    let serve = script.serve("REDO: verify the output", {
        let turn = parking_lot::Mutex::new(Some(reasoning_only_turn()));
        move |index| {
            if index == 0 {
                turn.lock().take().expect("one gated turn")
            } else {
                reply("continued")
            }
        }
    });
    test_drive(gate, task_request(), serve)
        .await
        .expect("routes");
    // The consult saw the labeled reasoning as the terminal evidence.
    let consult = script.call(1);
    let transcript = consult.llm_request.messages[0]
        .text_content("\n")
        .expect("transcript text");
    assert!(transcript.contains(REASONING_TAIL_LABEL.trim_end()));
    assert!(transcript.contains("thinking about it"));
    // The REDO echo prefers the reasoning over an empty string.
    let redo = script.call(2);
    assert_eq!(
        redo.llm_request.messages[1].text_content("\n").as_deref(),
        Some("thinking about it")
    );
}

#[tokio::test]
async fn empty_turn_redo_echo_uses_placeholder() {
    let script = Script::new();
    let gate = gate(AdvisorGateConfig::default());
    let serve = script.serve("REDO: produce output", {
        let turn = parking_lot::Mutex::new(Some(empty_turn()));
        move |index| {
            if index == 0 {
                turn.lock().take().expect("one gated turn")
            } else {
                reply("continued")
            }
        }
    });
    test_drive(gate, task_request(), serve)
        .await
        .expect("routes");
    let redo = script.call(2);
    assert_eq!(
        redo.llm_request.messages[1].text_content("\n").as_deref(),
        Some(EMPTY_ECHO_PLACEHOLDER)
    );
}

// ── Consult request shape ───────────────────────────────────────────────

#[tokio::test]
async fn consult_request_shape() {
    let script = Script::new();
    let gate = gate(AdvisorGateConfig {
        advisor_temperature: Some(0.2),
        ..AdvisorGateConfig::default()
    });
    let serve = script.serve("APPROVE", |_| reply("done"));
    test_drive(gate, task_request(), serve)
        .await
        .expect("routes");
    let consult = script.call(1).llm_request;
    assert_eq!(consult.instructions.len(), 1);
    assert_eq!(
        consult.instructions[0].content,
        vec![ContentBlock::Text {
            text: REVIEWER_SYSTEM_PROMPT.to_string()
        }]
    );
    assert_eq!(consult.messages.len(), 1);
    assert_eq!(consult.messages[0].role, Role::User);
    assert_eq!(consult.output.max_output_tokens, Some(2048));
    assert_eq!(consult.output.response_format, None);
    assert_eq!(consult.sampling.temperature, Some(0.2));
    assert!(consult.tools.is_empty());
    assert!(!consult.stream);
    let transcript = consult.messages[0].text_content("\n").expect("transcript");
    assert!(transcript.starts_with("Conversation so far (JSON):"));
    assert!(transcript.contains("The executor's latest turn"));
    assert!(transcript.ends_with("done"));
}

#[tokio::test]
async fn consult_transcript_includes_system_instructions() {
    let script = Script::new();
    let gate = gate(AdvisorGateConfig::default());
    let serve = script.serve("APPROVE", |_| reply("done"));
    let mut gated = task_request();
    gated.llm_request.instructions = vec![InstructionBlock {
        role: Role::System,
        content: vec![ContentBlock::Text {
            text: "the deliverable must be a CSV".to_string(),
        }],
    }];
    test_drive(gate, gated, serve).await.expect("routes");
    // System content is normalized out of `messages`; the advisor still
    // sees it, leading the serialized transcript.
    let transcript = script.call(1).llm_request.messages[0]
        .text_content("\n")
        .expect("transcript text");
    assert!(transcript.contains("the deliverable must be a CSV"));
    let task = transcript.find("build X").expect("task present");
    let system = transcript
        .find("the deliverable must be a CSV")
        .expect("system present");
    assert!(system < task);
}

// ── Sessions ────────────────────────────────────────────────────────────

#[tokio::test]
async fn session_final_evicts_scope() {
    let script = Script::new();
    let gate = gate(AdvisorGateConfig::default());
    let serve = script.serve("APPROVE", |_| reply("done"));
    let mut closing = with_session_id(task_request(), "s1");
    if let Some(metadata) = closing.metadata.as_mut() {
        metadata.session_final = Some(true);
    }
    test_drive(Arc::clone(&gate), closing, serve)
        .await
        .expect("routes");
    // The evicted scope re-arms: the same session id is reviewed again.
    let serve = script.serve("APPROVE", |_| reply("done"));
    test_drive(gate, with_session_id(task_request(), "s1"), serve)
        .await
        .expect("routes");
    assert_eq!(script.advisor_consults(), 2);
}

#[tokio::test]
async fn concurrent_same_scope_requests_consult_once() {
    let script = Script::new();
    let gate = gate(AdvisorGateConfig::default());
    let barrier = Arc::new(tokio::sync::Barrier::new(2));
    let calls = Arc::clone(&script.calls);
    let serve = {
        let barrier = Arc::clone(&barrier);
        move |decision: Decision, request: Request| {
            let barrier = Arc::clone(&barrier);
            let calls = Arc::clone(&calls);
            Box::pin(async move {
                let model = decision.selected_model_id().to_string();
                calls.lock().push((model.clone(), request));
                if model == ADVISOR {
                    Ok(reply("APPROVE"))
                } else {
                    // Hold both executor turns until each has generated,
                    // so both runs race for the single review slot.
                    barrier.wait().await;
                    Ok(reply("done"))
                }
            })
                as futures::future::BoxFuture<
                    'static,
                    std::result::Result<Response, LlmClientError>,
                >
        }
    };
    let (first, second) = tokio::join!(
        test_drive(Arc::clone(&gate), task_request(), serve.clone()),
        test_drive(Arc::clone(&gate), task_request(), serve)
    );
    first.expect("first run");
    second.expect("second run");
    assert_eq!(script.advisor_consults(), 1);
}

// ── Pure functions ──────────────────────────────────────────────────────

#[test]
fn verdict_parser_table() {
    let re = regex::Regex::new(VERDICT_PATTERN).expect("pattern compiles");
    let approve = |reply: &str| matches!(parse_verdict(&re, reply), Some(Verdict::Approve));
    let redo_plan = |reply: &str| match parse_verdict(&re, reply) {
        Some(Verdict::Redo { plan }) => Some(plan),
        _ => None,
    };
    assert!(approve("APPROVE"));
    assert!(approve("approve"));
    assert!(approve("  **APPROVE**"));
    assert!(approve("> approve"));
    assert!(approve("Final verdict: APPROVE"));
    assert!(approve("verdict: APPROVE"));
    assert_eq!(
        redo_plan("REDO: run the tests").as_deref(),
        Some("run the tests")
    );
    assert_eq!(redo_plan("REDO\n- fix x").as_deref(), Some("fix x"));
    assert_eq!(
        redo_plan("**Verdict:** REDO fix y").as_deref(),
        Some("fix y")
    );
    // An empty plan falls back to the whole reply.
    assert_eq!(redo_plan("REDO").as_deref(), Some("REDO"));
    // Word boundary: REDOING is not a verdict.
    assert!(parse_verdict(&re, "REDOING the work").is_none());
    // Prose-first replies are not trusted as verdicts.
    assert!(parse_verdict(&re, "I cannot approve this — REDO: run the tests").is_none());
    assert!(parse_verdict(&re, "").is_none());
}

#[test]
fn transcript_middle_drop() {
    assert_eq!(middle_drop("short".to_string(), 256), "short");
    let long: String = "a".repeat(300) + &"b".repeat(300);
    let capped = middle_drop(long, 400);
    assert_eq!(
        capped,
        format!("{}{TRUNCATION_MARKER}{}", "a".repeat(100), "b".repeat(300))
    );
    // Multi-byte characters never split.
    let unicode: String = "é".repeat(600);
    let capped = middle_drop(unicode, 400);
    assert_eq!(
        capped,
        format!("{}{TRUNCATION_MARKER}{}", "é".repeat(100), "é".repeat(300))
    );
    assert_eq!(
        middle_drop("x".to_string(), 256),
        "x",
        "under-cap text passes through"
    );
    let framed = review_transcript(&[Message::text(Role::User, "task")], None, 256);
    assert!(framed.ends_with(NO_TEXT_PLACEHOLDER));
}

#[test]
fn new_validation_errors() {
    let invalid = |config: AdvisorGateConfig, needle: &str| {
        let error = AdvisorGate::new(target(EXECUTOR), target(ADVISOR), config)
            .err()
            .expect("config rejected");
        assert!(error.to_string().contains(needle), "{error}");
    };
    invalid(
        AdvisorGateConfig {
            max_reviews: 0,
            ..AdvisorGateConfig::default()
        },
        "max_reviews",
    );
    invalid(
        AdvisorGateConfig {
            advisor_max_tokens: 0,
            ..AdvisorGateConfig::default()
        },
        "advisor_max_tokens",
    );
    invalid(
        AdvisorGateConfig {
            transcript_max_chars: 255,
            ..AdvisorGateConfig::default()
        },
        "transcript_max_chars",
    );
    invalid(
        AdvisorGateConfig {
            gate_trigger: GateTrigger::Pattern(String::new()),
            ..AdvisorGateConfig::default()
        },
        "non-empty gate_trigger_pattern",
    );
    invalid(
        AdvisorGateConfig {
            gate_trigger: GateTrigger::Pattern("(unclosed".to_string()),
            ..AdvisorGateConfig::default()
        },
        "not a valid regex",
    );
}
