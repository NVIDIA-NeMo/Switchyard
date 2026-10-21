// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicUsize, Ordering};

use futures::{StreamExt, TryStreamExt, stream};
use parking_lot::Mutex;
use switchyard_protocol::{
    LlmResponse, LlmResponseChunk, LlmResponseStreamEvent, Metadata, Response, ToolDefinition,
    Usage, completion_text, text_request,
};
use tokio::sync::{Barrier, Notify};

use super::*;
use crate::core::testing::reply;
use crate::{Call, RuntimeModels, drive};

fn algorithm(num_candidates: usize) -> Arc<BestOfNActions> {
    Arc::new(
        BestOfNActions::new(BestOfNActionsConfig {
            num_candidates,
            ..BestOfNActionsConfig::default()
        })
        .unwrap(),
    )
}

fn models() -> Arc<RuntimeModels> {
    Arc::new(RuntimeModels::new(
        [(Category::Any, vec!["generator".into()])].into(),
    ))
}

fn request() -> Request {
    let mut request = Request {
        llm_request: text_request(Some("route".into()), "fix the tests"),
        ..Request::default()
    };
    request.llm_request.sampling.temperature = Some(0.0);
    request.llm_request.output.max_output_tokens = Some(128);
    request
        .llm_request
        .tools
        .push(Tool::Function(ToolDefinition {
            name: "bash".into(),
            description: None,
            parameters: json!({"type": "object"}),
            strict: None,
        }));
    request
        .llm_request
        .extensions
        .fields
        .insert("seed".into(), json!(42));
    request
        .llm_request
        .preservation
        .requests
        .insert("openai_chat".into(), json!({"temperature": 0.0}));
    request.raw_request = Some(json!({"temperature": 0.0}));
    request
}

#[test]
fn invalid_limits_and_temperatures_are_rejected() {
    for config in [
        BestOfNActionsConfig {
            num_candidates: 0,
            ..Default::default()
        },
        BestOfNActionsConfig {
            candidate_timeout_ms: 0,
            ..Default::default()
        },
        BestOfNActionsConfig {
            temperature: -0.1,
            ..Default::default()
        },
        BestOfNActionsConfig {
            temperature: 2.1,
            ..Default::default()
        },
        BestOfNActionsConfig {
            temperature: f64::NAN,
            ..Default::default()
        },
        BestOfNActionsConfig {
            temperature: f64::INFINITY,
            ..Default::default()
        },
    ] {
        assert!(BestOfNActions::new(config).is_err());
    }
}

#[tokio::test]
async fn single_sample_keeps_the_completion_path_and_applies_temperature() -> Result<()> {
    let outcome = drive(algorithm(1), request(), models(), |_| async {
        panic!("single-sample routing should not make an intermediate call")
    })
    .await?;
    assert!(outcome.response.is_none());
    assert_eq!(outcome.selected_model_id()?, "generator");
    assert_eq!(outcome.request.llm_request.sampling.temperature, Some(0.8));
    assert!(outcome.request.llm_request.preservation.requests.is_empty());
    assert!(outcome.request.raw_request.is_none());
    Ok(())
}

#[tokio::test]
async fn concurrent_samples_replay_sample_zero_even_when_it_finishes_last() -> Result<()> {
    let calls = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(Barrier::new(3));
    let extras = Arc::new(AtomicUsize::new(0));
    let extras_done = Arc::new(Notify::new());
    let finished = Arc::new(Mutex::new(Vec::new()));
    let usage = Usage {
        output_tokens: Some(9),
        ..Default::default()
    };
    let events = vec![
        LlmResponseStreamEvent::preserved(
            "anthropic_messages",
            json!({"type": "content_block_delta", "delta": {"type": "signature_delta", "signature": "signed"}}),
            vec![],
        ),
        LlmResponseChunk::ToolCallDelta {
            index: 0,
            id: Some("call-zero".into()),
            name: Some("bash".into()),
            arguments_delta: Some(r#"{"command":"pytest"}"#.into()),
        }
        .into(),
        LlmResponseChunk::Usage(usage.clone()).into(),
        LlmResponseChunk::MessageStop {
            reason: Some("tool_use".into()),
        }
        .into(),
    ];
    let run = drive(algorithm(3), request(), models(), {
        let calls = calls.clone();
        let finished = finished.clone();
        let events = events.clone();
        move |call| {
            let Call::Model(call) = call else {
                panic!("unexpected judge")
            };
            let index = calls.fetch_add(1, Ordering::SeqCst);
            let expected = request();
            assert!(call.recover_errors);
            assert_eq!(call.request.llm_request.model.as_deref(), Some("generator"));
            assert_eq!(
                call.request.llm_request.messages,
                expected.llm_request.messages
            );
            assert_eq!(call.request.llm_request.tools, expected.llm_request.tools);
            assert_eq!(call.request.llm_request.output, expected.llm_request.output);
            assert_eq!(call.request.llm_request.extensions.fields["seed"], 42);
            assert_eq!(call.request.llm_request.sampling.temperature, Some(0.8));
            assert!(call.request.llm_request.preservation.requests.is_empty());
            assert!(call.request.raw_request.is_none());
            let barrier = barrier.clone();
            let extras = extras.clone();
            let extras_done = extras_done.clone();
            let finished = finished.clone();
            let events = events.clone();
            async move {
                call.respond(async move {
                    barrier.wait().await;
                    if index == 0 {
                        extras_done.notified().await;
                    }
                    finished.lock().push(index);
                    if index != 0 && extras.fetch_add(1, Ordering::SeqCst) == 1 {
                        extras_done.notify_one();
                    }
                    if index == 0 {
                        Ok(Response {
                            llm_response: LlmResponse::Stream(
                                stream::iter(events.into_iter().map(Ok)).boxed(),
                            ),
                            metadata: Some(Metadata {
                                session_id: Some("session".into()),
                                ..Default::default()
                            }),
                            upstream_headers: [(
                                http::header::HeaderName::from_static("x-request-id"),
                                "request-zero".parse().unwrap(),
                            )]
                            .into_iter()
                            .collect(),
                        })
                    } else {
                        Ok(reply(format!("discarded {index}")))
                    }
                })
                .await
            }
        }
    });
    let outcome = tokio::time::timeout(Duration::from_secs(2), run)
        .await
        .unwrap()?;
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(finished.lock().last(), Some(&0));
    let evidence = outcome.metadata.unwrap().evidence.unwrap();
    assert_eq!(evidence["selected_index"], 0);
    assert_eq!(evidence["candidates"][0]["usage"]["output_tokens"], 9);
    let response = outcome.response.unwrap();
    assert_eq!(
        response.metadata.unwrap().session_id.as_deref(),
        Some("session")
    );
    assert_eq!(response.upstream_headers["x-request-id"], "request-zero");
    let LlmResponse::Stream(replay) = response.llm_response else {
        panic!("lost original stream")
    };
    assert_eq!(replay.try_collect::<Vec<_>>().await.unwrap(), events);
    Ok(())
}

#[tokio::test]
async fn only_sample_zero_failure_fails_the_run() {
    for failed_index in [0, 1] {
        let calls = AtomicUsize::new(0);
        let result = drive(algorithm(2), request(), models(), |call| {
            let Call::Model(call) = call else {
                panic!("unexpected judge")
            };
            let index = calls.fetch_add(1, Ordering::SeqCst);
            async move {
                call.respond(async move {
                    if index == failed_index {
                        Err(LibsyError::client_call(
                            "generator",
                            LlmClientError::General("failed".into()),
                        ))
                    } else {
                        Ok(reply(format!("sample {index}")))
                    }
                })
                .await
            }
        })
        .await;
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        if failed_index == 0 {
            assert!(matches!(result, Err(LibsyError::ClientCall { .. })));
        } else {
            let outcome = result.unwrap();
            assert_eq!(
                completion_text(outcome.response.unwrap().llm_response.as_agg().unwrap()),
                "sample 0"
            );
            assert_eq!(
                outcome.metadata.unwrap().evidence.unwrap()["candidates"][1]["is_success"],
                false
            );
        }
    }
}

#[tokio::test]
async fn deadline_includes_draining_each_candidate_stream() {
    for stalled_index in [0, 1] {
        let calls = AtomicUsize::new(0);
        let algorithm = Arc::new(
            BestOfNActions::new(BestOfNActionsConfig {
                num_candidates: 2,
                candidate_timeout_ms: 50,
                ..Default::default()
            })
            .unwrap(),
        );
        let result = drive(algorithm, request(), models(), |call| {
            let Call::Model(call) = call else {
                panic!("unexpected judge")
            };
            let index = calls.fetch_add(1, Ordering::SeqCst);
            async move {
                let response = if index == stalled_index {
                    Response {
                        llm_response: LlmResponse::Stream(stream::pending().boxed()),
                        metadata: None,
                        upstream_headers: Default::default(),
                    }
                } else {
                    reply("sample 0")
                };
                call.respond(std::future::ready(Ok(response))).await
            }
        });
        let result = tokio::time::timeout(Duration::from_secs(2), result)
            .await
            .unwrap();
        if stalled_index == 0 {
            assert!(matches!(
                result,
                Err(LibsyError::ClientCall {
                    source: LlmClientError::Timeout { .. },
                    ..
                })
            ));
        } else {
            assert_eq!(
                result.unwrap().metadata.unwrap().evidence.unwrap()["candidates"][1]["reason_code"],
                "timeout"
            );
        }
    }
}

#[tokio::test]
async fn unsupported_requests_fail_before_dispatch() {
    let mut unsupported = Vec::new();
    for (key, value) in [
        ("previous_response_id", json!("resp")),
        ("conversation", json!({"id": "conv"})),
        ("n", json!(2)),
    ] {
        let mut input = request();
        input
            .llm_request
            .extensions
            .fields
            .insert(key.into(), value);
        unsupported.push(input);
    }
    let mut provider_tools = request();
    provider_tools.llm_request.tools.push(Tool::Provider {
        provider: "openai_responses".into(),
        raw: json!({"type": "web_search"}),
    });
    unsupported.push(provider_tools);
    for input in unsupported {
        let result = drive(algorithm(2), input, models(), |_| async {
            panic!("unsupported request reached a provider")
        })
        .await;
        assert!(matches!(
            result,
            Err(LibsyError::ClientCall {
                source: LlmClientError::InvalidRequest { .. },
                ..
            })
        ));
    }
}

struct Dropped(Arc<AtomicUsize>);

impl Drop for Dropped {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[tokio::test]
async fn cancellation_drops_every_pending_candidate() {
    let started = Arc::new(Barrier::new(4));
    let dropped = Arc::new(AtomicUsize::new(0));
    let run = tokio::spawn(drive(algorithm(3), request(), models(), {
        let started = started.clone();
        let dropped = dropped.clone();
        move |call| {
            let started = started.clone();
            let dropped = dropped.clone();
            async move {
                let Call::Model(call) = call else {
                    panic!("unexpected judge")
                };
                call.respond(async move {
                    let _guard = Dropped(dropped);
                    started.wait().await;
                    std::future::pending().await
                })
                .await
            }
        }
    }));
    tokio::time::timeout(Duration::from_secs(2), started.wait())
        .await
        .unwrap();
    run.abort();
    assert!(matches!(run.await, Err(error) if error.is_cancelled()));
    assert_eq!(dropped.load(Ordering::SeqCst), 3);
}
