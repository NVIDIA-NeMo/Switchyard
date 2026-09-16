// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for one verification-gated turn.
//!
//! Driven through the shared mocked-`Driver` scaffold every other algorithm
//! uses, so what is asserted is the behavior a deployment would see: which tier
//! served the turn, which calls were paid for, and what the capable tier was
//! given when the attempt was rejected.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::StreamExt;
use parking_lot::Mutex;
use switchyard_protocol::{
    AggLlmResponse, Category, ContentBlock, FormatId, ImageSource, InstructionBlock,
    LlmClientError, LlmResponse, LlmResponseChunk, LlmResponseStreamEvent, Message, ModelId,
    PreservationMetadata, Request, Response, ResponseOutput, Role, StopReason, ToolCall,
    ToolResult, Usage, WireFormat, text_request, text_response,
};

#[cfg(any(unix, windows))]
use super::super::checker::{
    CandidateWorkspace, CheckerConfig, CommandWorkspaceProvider, CommandWorkspaceProviderConfig,
    PinnedChecker, SANDBOX_ATTESTATION, WorkspaceProvider,
};
use super::super::config::{Checker, CheckerRequest, ServingMode, ValidatedChecker, VgrConfig};
use super::super::mode::ACTIVE_APPROVAL;
use super::super::safety::{BreakerConfig, KillSwitch};
use crate::core::testing::{Serve, ServeResult, test_drive_with_models};
use crate::{Algorithm, LibsyError, Result, RuntimeModels, Step};
#[cfg(any(unix, windows))]
use tempfile::TempDir;

const LOCAL: &str = "local-tier";
const CLOUD: &str = "cloud-tier";

/// A request carrying one user turn.
fn request(text: &str) -> Request {
    Request {
        llm_request: text_request(Some("auto".to_string()), text),
        raw_request: None,
        metadata: None,
    }
}

/// A request keyed to one session for affinity behavior.
fn session_request(session_id: &str, text: &str) -> Request {
    Request {
        metadata: Some(switchyard_protocol::Metadata {
            session_id: Some(session_id.to_string()),
            agent_id: Some("agent-a".to_string()),
            ..Default::default()
        }),
        ..request(text)
    }
}

/// Extends a user request with one tool call and its result.
fn tool_continuation(mut request: Request) -> Request {
    request.llm_request.messages.push(Message {
        role: Role::Assistant,
        content: vec![ContentBlock::ToolCall(ToolCall {
            id: "call-1".into(),
            name: "terminal".into(),
            arguments: serde_json::json!({"command": "true"}),
        })],
    });
    request.llm_request.messages.push(Message {
        role: Role::User,
        content: vec![ContentBlock::ToolResult(ToolResult {
            tool_call_id: "call-1".into(),
            content: vec![ContentBlock::Text {
                text: "command succeeded".into(),
            }],
            is_error: Some(false),
        })],
    });
    request
}

/// An active route between the two tiers, with verification enabled.
fn active() -> VgrConfig {
    VgrConfig {
        mode: ServingMode::Active {
            approval: ACTIVE_APPROVAL.into(),
        },
        // Most runtime fixtures predate the typing rung and pin a direct
        // verification call sequence. Dedicated tests below cover the default.
        task_typing: false,
        ..VgrConfig::new(ModelId::from(LOCAL), ModelId::from(CLOUD))
    }
}

fn route_models() -> Arc<RuntimeModels> {
    Arc::new(RuntimeModels::new(
        [(
            Category::Any,
            vec![ModelId::from(LOCAL), ModelId::from(CLOUD)],
        )]
        .into(),
    ))
}

async fn test_drive(
    algorithm: Arc<dyn Algorithm>,
    request: Request,
    serve: impl Serve,
) -> Result<(ModelId, Response)> {
    test_drive_with_models(algorithm, request, (*route_models()).clone(), serve).await
}

/// A buffered reply that also reports a readout probability for "yes".
fn reply_with_readout(text: &str, p_yes: f64) -> Response {
    let mut agg = text_response(None, text.to_string());
    let mut preservation = PreservationMetadata::default();
    preservation.responses.insert(
        FormatId::from(WireFormat::OpenAiChat),
        serde_json::json!({"choices": [{"logprobs": {"content": [{"top_logprobs": [
            {"token": "yes", "logprob": p_yes.ln()},
            {"token": "no", "logprob": (1.0 - p_yes).ln()}
        ]}]}}]}),
    );
    agg.preservation = preservation;
    Response {
        llm_response: LlmResponse::Agg(agg),
        metadata: None,
        upstream_headers: Default::default(),
    }
}

/// A plain buffered reply.
fn reply(text: &str) -> Response {
    Response {
        llm_response: LlmResponse::Agg(text_response(None, text.to_string())),
        metadata: None,
        upstream_headers: Default::default(),
    }
}

/// A proposed assistant turn that invokes one tool.
fn tool_aggregate(name: &str) -> AggLlmResponse {
    AggLlmResponse {
        outputs: vec![ResponseOutput {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "call-1".into(),
                name: name.into(),
                arguments: serde_json::json!({"path": "input.txt"}),
            })],
            stop_reason: Some(StopReason::ToolUse),
        }],
        ..Default::default()
    }
}

fn tool_reply(name: &str) -> Response {
    Response {
        llm_response: LlmResponse::Agg(tool_aggregate(name)),
        metadata: None,
        upstream_headers: Default::default(),
    }
}

fn shaped_tool_reply(arguments: serde_json::Value, stop_reason: StopReason) -> Response {
    let mut aggregate = tool_aggregate("write_file");
    aggregate.outputs[0].stop_reason = Some(stop_reason);
    let ContentBlock::ToolCall(call) = &mut aggregate.outputs[0].content[0] else {
        unreachable!()
    };
    call.arguments = arguments;
    Response {
        llm_response: LlmResponse::Agg(aggregate),
        metadata: None,
        upstream_headers: Default::default(),
    }
}

fn is_turn_verification(request: &Request) -> bool {
    request
        .llm_request
        .instructions
        .iter()
        .flat_map(|instruction| &instruction.content)
        .any(|block| {
            matches!(
                block,
                ContentBlock::Text { text } if text.contains("escalation judge inside an agentic router")
            )
        })
}

/// QA's reproduction from the bug report, as filed.
///
/// Before the split this printed `target=cloud-tier calls=["local-tier",
/// "cloud-tier"]`: the attempt, then an immediate escalation with no rung
/// between them. The assertion pins that second call as local.
///
/// Note this closure answers every call with the same truncated response, so
/// the readout it feeds the rung is itself truncated and reads as
/// indeterminate -- the turn still escalates, but only after the evidence was
/// gathered and judged, which is the behaviour the report asked for.
#[tokio::test]
async fn max_tokens_truncated_local_attempt_routing() -> Result<()> {
    let log = CallLog::default();
    let seen = log.clone();
    let route = Arc::new(super::super::Vgr::new(active()).expect("builds"));

    let (target, _response) = test_drive(
        route,
        request("Write a detailed explanation of quicksort."),
        move |t: ModelId, _r: Request| {
            let log = seen.clone();
            async move {
                log.record(&t);
                // A substantial, mostly-complete local answer that got cut off
                // by the token budget, not a genuinely empty/failed response.
                let mut agg = text_response(
                    None,
                    "Quicksort is a divide-and-conquer algorithm. It picks a pivot, \
                     partitions the array around it, then recurses on each half. The \
                     average case is O(n log n) because"
                        .to_string(),
                );
                agg.outputs[0].stop_reason = Some(StopReason::MaxTokens);
                Ok(Response {
                    llm_response: LlmResponse::Agg(agg),
                    metadata: None,
                    upstream_headers: Default::default(),
                })
            }
        },
    )
    .await?;

    // The whole ladder, in order: the attempt, then the readout and the
    // deliberation that judged it, then the escalation they did not avert.
    // The regression was ["local-tier", "cloud-tier"] -- the attempt and an
    // immediate escalation with no rung between them. Pinning the exact
    // sequence also catches a regression that runs the readout but skips the
    // deliberation, which a positional check alone would let through.
    assert_eq!(log.targets(), vec![LOCAL, LOCAL, LOCAL, CLOUD]);
    assert_eq!(target, ModelId::from(CLOUD));
    Ok(())
}

/// Records every target called, in order, so cost can be asserted.
#[derive(Clone, Default)]
struct CallLog(Arc<Mutex<Vec<String>>>);

impl CallLog {
    fn targets(&self) -> Vec<String> {
        self.0.lock().clone()
    }
    fn record(&self, target: &ModelId) {
        self.0.lock().push(target.to_string());
    }
}

/// The text of a served response.
async fn served_text(response: Response) -> Result<String> {
    let agg = response
        .llm_response
        .into_agg()
        .await
        .map_err(|source| LibsyError::client_call(ModelId::from(LOCAL), source))?;
    let Some(output) = agg.first_output() else {
        return Ok(String::new());
    };
    Ok(output
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.clone()),
            _ => None,
        })
        .collect::<String>())
}

/// Binds a fixed checker verdict to test manifest evidence.
fn validated_checker(verdict: Option<bool>) -> Result<ValidatedChecker> {
    ValidatedChecker::new(
        Arc::new(FixedChecker(verdict)),
        "sha256:0123456789abcdef0123456789abcdef",
    )
}

/// A typed failure for test-only control-flow checks.
fn test_error(message: impl Into<String>) -> LibsyError {
    LibsyError::AlgorithmError {
        message: message.into(),
    }
}

#[tokio::test]
async fn an_eligible_local_failure_escalates_to_cloud() -> Result<()> {
    let route: Arc<dyn Algorithm> = Arc::new(super::super::Vgr::new(active())?);
    let stream = route.run_stream(request("hello"), route_models());
    tokio::pin!(stream);

    let step = stream
        .next()
        .await
        .ok_or_else(|| test_error("missing local attempt call"))??;
    let Step::CallModel(call) = step else {
        return Err(test_error("expected local attempt call"));
    };
    assert_eq!(call.models, vec![ModelId::from(LOCAL)]);
    call.respond(Err(LibsyError::client_call(
        ModelId::from(LOCAL),
        LlmClientError::ContextWindowExceeded {
            model: ModelId::from(LOCAL),
            message: "request exceeds local context".to_string(),
        },
    )))?;

    let step = stream
        .next()
        .await
        .ok_or_else(|| test_error("missing cloud completion call"))??;
    let Step::CallModel(call) = step else {
        return Err(test_error("eligible local failure did not call cloud"));
    };
    assert_eq!(call.models, vec![ModelId::from(CLOUD)]);
    call.respond(Ok(reply("cloud answer")))?;

    let step = stream
        .next()
        .await
        .ok_or_else(|| test_error("missing terminal outcome"))??;
    let Step::Done(outcome) = step else {
        return Err(test_error("cloud completion was not terminal"));
    };
    assert_eq!(outcome.selected_model_id()?, &ModelId::from(CLOUD));
    assert_eq!(outcome.selected_model_ids, vec![ModelId::from(CLOUD)]);
    assert!(outcome.response.is_some());
    Ok(())
}

#[tokio::test]
async fn a_cloud_decision_never_falls_back_to_local() -> Result<()> {
    let route: Arc<dyn Algorithm> = Arc::new(super::super::Vgr::new(VgrConfig::new(
        ModelId::from(LOCAL),
        ModelId::from(CLOUD),
    ))?);
    let stream = route.run_stream(request("hello"), route_models());
    tokio::pin!(stream);

    let step = stream
        .next()
        .await
        .ok_or_else(|| test_error("missing cloud completion call"))??;
    let Step::CallModel(call) = step else {
        return Err(test_error("off mode did not call cloud"));
    };
    assert_eq!(call.models, vec![ModelId::from(CLOUD)]);
    call.respond(Ok(reply("cloud answer")))?;

    let step = stream
        .next()
        .await
        .ok_or_else(|| test_error("missing terminal outcome"))??;
    let Step::Done(outcome) = step else {
        return Err(test_error("cloud completion was not terminal"));
    };
    assert_eq!(outcome.selected_model_id()?, &ModelId::from(CLOUD));
    assert_eq!(outcome.selected_model_ids, vec![ModelId::from(CLOUD)]);
    assert!(outcome.response.is_some());
    Ok(())
}

#[tokio::test]
async fn a_confident_readout_commits_the_local_attempt_without_a_cloud_call() -> Result<()> {
    // The point of the cheap rung: a commit reachable from the readout alone
    // pays for the attempt and the readout, and nothing else.
    let log = CallLog::default();
    let seen = log.clone();
    let route = Arc::new(super::super::Vgr::new(active())?);

    let (target, response) = test_drive(
        route,
        request("what is the capital?"),
        move |t: ModelId, _r| {
            let log = seen.clone();
            async move {
                log.record(&t);
                let result: ServeResult = if log.targets().len() == 1 {
                    Ok(reply("Paris."))
                } else {
                    Ok(reply_with_readout("yes", 0.97))
                };
                result
            }
        },
    )
    .await?;

    assert_eq!(target, ModelId::from(LOCAL));
    assert_eq!(served_text(response).await?, "Paris.");
    // The attempt and one readout. No cloud verifier was configured, and none
    // was needed.
    assert_eq!(log.targets(), vec![LOCAL, LOCAL]);
    Ok(())
}

#[tokio::test]
async fn buffering_an_aggregate_preserves_extensions_refusal_and_provider_body() -> Result<()> {
    let mut aggregate = text_response(Some(LOCAL.to_string()), "answer");
    let Some(output) = aggregate.outputs.first_mut() else {
        return Err(test_error("text response has no output"));
    };
    output.content.push(ContentBlock::Refusal {
        text: "restricted detail".to_string(),
    });
    aggregate
        .extensions
        .fields
        .insert("provider_extension".to_string(), serde_json::json!("exact"));
    aggregate.preservation.responses.insert(
        FormatId::from(WireFormat::OpenAiResponses),
        serde_json::json!({"id": "preserved", "output": [{"type": "refusal"}]}),
    );
    let expected = aggregate.clone();

    let buffered = super::BufferedResponse::new(Response {
        llm_response: LlmResponse::Agg(aggregate),
        metadata: None,
        upstream_headers: Default::default(),
    })
    .await
    .map_err(|source| LibsyError::client_call(ModelId::from(LOCAL), source))?;

    assert_eq!(buffered.aggregate(), &expected);
    let returned = buffered
        .into_response()
        .llm_response
        .into_agg()
        .await
        .map_err(|source| LibsyError::client_call(ModelId::from(LOCAL), source))?;
    assert_eq!(returned, expected);
    Ok(())
}

#[tokio::test]
async fn a_streamed_local_commit_replays_exact_events_and_boundaries() -> Result<()> {
    let events = vec![
        LlmResponseStreamEvent::preserved(
            WireFormat::OpenAiChat,
            serde_json::json!({
                "id": "response-1",
                "choices": [{"delta": {"role": "assistant", "content": "hello"}}],
                "provider_extension": "first"
            }),
            vec![
                LlmResponseChunk::MessageStart {
                    id: Some("response-1".to_string()),
                    model: Some(LOCAL.to_string()),
                },
                LlmResponseChunk::TextDelta {
                    index: 0,
                    text: "hello".to_string(),
                },
            ],
        ),
        LlmResponseStreamEvent::preserved(
            WireFormat::OpenAiChat,
            serde_json::json!({
                "id": "response-1",
                "choices": [{"delta": {"refusal": "restricted detail"}}]
            }),
            Vec::new(),
        ),
        LlmResponseStreamEvent::preserved(
            WireFormat::OpenAiChat,
            serde_json::json!({
                "id": "response-1",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 1}
            }),
            vec![
                LlmResponseChunk::MessageStop {
                    reason: Some("stop".to_string()),
                },
                LlmResponseChunk::Usage(Usage {
                    output_tokens: Some(1),
                    ..Usage::default()
                }),
            ],
        ),
    ];
    let expected = events.clone();
    let calls = Arc::new(AtomicUsize::new(0));
    let seen_calls = Arc::clone(&calls);
    let route = Arc::new(super::super::Vgr::new(active())?);
    let mut streamed_request = request("what is the answer?");
    streamed_request.llm_request.stream = true;

    let (target, response) = test_drive(route, streamed_request, move |_target, _request| {
        let events = events.clone();
        let call = seen_calls.fetch_add(1, Ordering::SeqCst);
        async move {
            if call == 0 {
                Ok(Response {
                    llm_response: LlmResponse::Stream(Box::pin(futures::stream::iter(
                        events
                            .into_iter()
                            .map(Ok::<LlmResponseStreamEvent, LlmClientError>),
                    ))),
                    metadata: None,
                    upstream_headers: Default::default(),
                })
            } else {
                Ok(reply_with_readout("yes", 0.97))
            }
        }
    })
    .await?;

    assert_eq!(target, ModelId::from(LOCAL));
    let LlmResponse::Stream(mut replayed) = response.llm_response else {
        return Err(test_error("streamed local attempt returned as aggregate"));
    };
    let mut actual = Vec::new();
    while let Some(event) = replayed.next().await {
        actual.push(event.map_err(|source| LibsyError::client_call(ModelId::from(LOCAL), source))?);
    }
    assert_eq!(actual, expected);
    Ok(())
}

/// Records when an in-flight async operation is cancelled.
struct DropSignal(Arc<AtomicBool>);

impl Drop for DropSignal {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

#[tokio::test]
async fn judged_coding_dial_spends_exactly_one_cloud_confirmation() -> Result<()> {
    let log = CallLog::default();
    let seen = log.clone();
    let config = VgrConfig {
        mode: ServingMode::Evaluate,
        task_typing: false,
        targets: super::super::config::Targets {
            cloud_judge: Some(ModelId::from(CLOUD)),
            ..VgrConfig::new(ModelId::from(LOCAL), ModelId::from(CLOUD)).targets
        },
        ..VgrConfig::new(ModelId::from(LOCAL), ModelId::from(CLOUD))
    };
    let route = Arc::new(super::super::Vgr::new(config)?);
    let (target, _) = test_drive(route, request("fix the build"), move |t: ModelId, _r| {
        let seen = seen.clone();
        async move {
            seen.record(&t);
            let response = match seen.targets().len() {
                1 => reply("```rust\nfn fixed() {}\n```"),
                2 => reply_with_readout("yes", 0.25),
                _ => reply("yes"),
            };
            Ok(response)
        }
    })
    .await?;

    assert_eq!(target, ModelId::from(LOCAL));
    assert_eq!(log.targets(), vec![LOCAL, LOCAL, CLOUD]);
    Ok(())
}

#[tokio::test]
async fn cancelling_stream_buffering_drops_the_source_stream() -> Result<()> {
    let (started_tx, mut started_rx) = tokio::sync::mpsc::unbounded_channel();
    let dropped = Arc::new(AtomicBool::new(false));
    let dropped_by_stream = Arc::clone(&dropped);
    let source = futures::stream::once(async move {
        let _drop_signal = DropSignal(dropped_by_stream);
        let _ = started_tx.send(());
        std::future::pending::<std::result::Result<LlmResponseStreamEvent, LlmClientError>>().await
    });
    let response = Response {
        llm_response: LlmResponse::Stream(Box::pin(source)),
        metadata: None,
        upstream_headers: Default::default(),
    };

    let buffer_task = tokio::spawn(super::BufferedResponse::new(response));
    let started = tokio::time::timeout(Duration::from_secs(1), started_rx.recv())
        .await
        .map_err(|source| LibsyError::external("waiting for response buffering", source))?;
    if started.is_none() {
        return Err(test_error("response buffering never started"));
    }
    buffer_task.abort();
    let join = buffer_task.await;

    assert!(join.is_err_and(|error| error.is_cancelled()));
    assert!(
        dropped.load(Ordering::SeqCst),
        "cancelling buffering did not drop the source stream"
    );
    Ok(())
}

#[tokio::test]
async fn a_weak_readout_escalates_to_the_capable_tier() -> Result<()> {
    let route = Arc::new(super::super::Vgr::new(active())?);
    let (target, _) = test_drive(
        route,
        request("what is the capital?"),
        |t: ModelId, _r| async move {
            let result: ServeResult = if t == *LOCAL {
                // The attempt, then a readout well under every bar.
                Ok(reply_with_readout("Paris.", 0.05))
            } else {
                Ok(reply("cloud answer"))
            };
            result
        },
    )
    .await?;

    assert_eq!(target, ModelId::from(CLOUD));
    Ok(())
}

#[tokio::test]
async fn judged_coding_dial_refutation_escalates() -> Result<()> {
    let log = CallLog::default();
    let seen = log.clone();
    let config = VgrConfig {
        mode: ServingMode::Evaluate,
        task_typing: false,
        targets: super::super::config::Targets {
            cloud_judge: Some(ModelId::from(CLOUD)),
            ..VgrConfig::new(ModelId::from(LOCAL), ModelId::from(CLOUD)).targets
        },
        ..VgrConfig::new(ModelId::from(LOCAL), ModelId::from(CLOUD))
    };
    let route = Arc::new(super::super::Vgr::new(config)?);
    let (target, _) = test_drive(route, request("fix the build"), move |t: ModelId, _r| {
        let seen = seen.clone();
        async move {
            seen.record(&t);
            let response = match seen.targets().len() {
                1 => reply("```rust\nfn broken() {}\n```"),
                2 => reply_with_readout("yes", 0.25),
                3 => reply("no"),
                _ => reply("cloud"),
            };
            Ok(response)
        }
    })
    .await?;

    assert_eq!(target, ModelId::from(CLOUD));
    assert_eq!(log.targets(), vec![LOCAL, LOCAL, CLOUD, CLOUD]);
    Ok(())
}

#[tokio::test]
async fn an_unreadable_verifier_escalates_rather_than_failing_the_request() -> Result<()> {
    // A verifier that produces nothing usable is indeterminate evidence, which
    // escalates. It must not surface as an error to the caller.
    let route = Arc::new(super::super::Vgr::new(active())?);
    let (target, _) = test_drive(route, request("hello"), |t: ModelId, _r| async move {
        let result: ServeResult = if t == *LOCAL {
            // No preserved logprobs at all, so the readout cannot be scored,
            // and the deliberating verifier hedges.
            Ok(reply("an attempt\nMaybe? I cannot say."))
        } else {
            Ok(reply("cloud answer"))
        };
        result
    })
    .await?;

    assert_eq!(target, ModelId::from(CLOUD));
    Ok(())
}

#[tokio::test]
async fn off_mode_serves_cloud_without_producing_an_attempt() -> Result<()> {
    // Off spends nothing: there is no decision to inform, so the local tier is
    // never called at all.
    let log = CallLog::default();
    let seen = log.clone();
    let route = Arc::new(super::super::Vgr::new(VgrConfig::new(
        ModelId::from(LOCAL),
        ModelId::from(CLOUD),
    ))?);

    let (target, _) = test_drive(route, request("hello"), move |t: ModelId, _r| {
        let log = seen.clone();
        async move {
            log.record(&t);
            let result: ServeResult = Ok(reply("cloud answer"));
            result
        }
    })
    .await?;

    assert_eq!(target, ModelId::from(CLOUD));
    assert_eq!(log.targets(), vec![CLOUD]);
    Ok(())
}

#[tokio::test]
async fn image_in_a_tool_result_bypasses_a_text_only_local_tier() -> Result<()> {
    // Agent screenshots are nested in tool results, not always top-level user blocks.
    let mut with_image = request("inspect the screenshot");
    with_image.llm_request.messages.push(Message {
        role: Role::Tool,
        content: vec![ContentBlock::ToolResult(ToolResult {
            tool_call_id: "call-1".into(),
            content: vec![ContentBlock::Image {
                source: ImageSource::Url {
                    url: "https://example.invalid/screenshot.png".into(),
                    detail: None,
                },
            }],
            is_error: Some(false),
        })],
    });
    let log = CallLog::default();
    let seen = log.clone();
    let route = Arc::new(super::super::Vgr::new(active())?);

    let (target, _) = test_drive(route, with_image, move |t: ModelId, _r| {
        let log = seen.clone();
        async move {
            log.record(&t);
            let result: ServeResult = Ok(reply("cloud answer"));
            result
        }
    })
    .await?;

    assert_eq!(target, ModelId::from(CLOUD));
    assert_eq!(log.targets(), vec![CLOUD]);
    Ok(())
}

#[tokio::test]
async fn shadow_mode_produces_the_attempt_but_serves_cloud() -> Result<()> {
    // The observation mode still does the work, so a deployment can measure
    // what would have happened; it just does not act on it.
    let log = CallLog::default();
    let seen = log.clone();
    let config = VgrConfig {
        mode: ServingMode::Shadow,
        ..VgrConfig::new(ModelId::from(LOCAL), ModelId::from(CLOUD))
    };
    let route = Arc::new(super::super::Vgr::new(config)?);

    let (target, _) = test_drive(route, request("hello"), move |t: ModelId, _r| {
        let log = seen.clone();
        async move {
            log.record(&t);
            let result: ServeResult = Ok(reply_with_readout("an attempt", 0.99));
            result
        }
    })
    .await?;

    assert_eq!(target, ModelId::from(CLOUD));
    // The local attempt was produced and verified, despite not being served.
    assert!(log.targets().iter().any(|t| t == LOCAL));
    Ok(())
}

#[tokio::test]
async fn active_mode_requires_the_approval_attestation() {
    let config = VgrConfig {
        mode: ServingMode::Active {
            approval: "sure".into(),
        },
        ..VgrConfig::new(ModelId::from(LOCAL), ModelId::from(CLOUD))
    };
    assert!(super::super::Vgr::new(config).is_err());
}

#[test]
fn config_defaults_enable_task_typing() {
    assert!(VgrConfig::new(ModelId::from(LOCAL), ModelId::from(CLOUD)).task_typing);
}

#[tokio::test]
async fn an_escalation_carries_the_rejected_attempt_as_unverified_reference() -> Result<()> {
    // The local tier's work is not discarded, but the capable tier is told
    // plainly that it was never verified.
    let carried = Arc::new(Mutex::new(String::new()));
    let seen = Arc::clone(&carried);
    let config = VgrConfig {
        speculation_carry: true,
        ..active()
    };
    let route = Arc::new(super::super::Vgr::new(config)?);

    let (target, _) = test_drive(
        route,
        request("solve this"),
        move |t: ModelId, r: Request| {
            let seen = Arc::clone(&seen);
            async move {
                if t == *CLOUD {
                    let text: String = r
                        .llm_request
                        .messages
                        .iter()
                        .flat_map(|m| m.content.iter())
                        .filter_map(|b| match b {
                            ContentBlock::Text { text } => Some(text.clone()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    *seen.lock() = text;
                }
                let result: ServeResult = if t == *LOCAL {
                    Ok(reply_with_readout("my draft solution", 0.01))
                } else {
                    Ok(reply("cloud answer"))
                };
                result
            }
        },
    )
    .await?;

    assert_eq!(target, ModelId::from(CLOUD));
    let carried = carried.lock().clone();
    assert!(carried.contains("my draft solution"), "attempt not carried");
    assert!(carried.contains("NOT verified"), "not labelled unverified");
    // The original request survives alongside the carried attempt.
    assert!(carried.contains("solve this"));
    Ok(())
}

#[tokio::test]
async fn speculation_carry_bounds_the_previous_attempt() -> Result<()> {
    let attempt = format!("HEAD-{}-TAIL", "middle".repeat(1_000));
    let carried = Arc::new(Mutex::new(String::new()));
    let seen = Arc::clone(&carried);
    let config = VgrConfig {
        speculation_carry: true,
        ..active()
    };
    let route = Arc::new(super::super::Vgr::new(config)?);

    let (target, _) = test_drive(
        route,
        request("solve this"),
        move |t: ModelId, r: Request| {
            let seen = Arc::clone(&seen);
            let attempt = attempt.clone();
            async move {
                if t == *CLOUD {
                    let text = r
                        .llm_request
                        .messages
                        .iter()
                        .filter_map(|message| message.text_content("\n"))
                        .collect::<Vec<_>>()
                        .join("\n");
                    *seen.lock() = text;
                    Ok(reply("cloud answer"))
                } else {
                    Ok(reply_with_readout(&attempt, 0.01))
                }
            }
        },
    )
    .await?;

    assert_eq!(target, ModelId::from(CLOUD));
    let carried = carried.lock().clone();
    let opening = "<previous_attempt>\n";
    let start = carried
        .find(opening)
        .map(|index| index + opening.len())
        .ok_or_else(|| test_error("missing carried-attempt opening tag"))?;
    let end = carried[start..]
        .find("\n</previous_attempt>")
        .map(|index| start + index)
        .ok_or_else(|| test_error("missing carried-attempt closing tag"))?;
    let bounded = &carried[start..end];
    assert!(
        bounded.chars().count() <= super::super::render::ATTEMPT_BUDGET,
        "carried attempt exceeded the existing VGR character budget"
    );
    assert!(bounded.starts_with("HEAD-"));
    assert!(bounded.ends_with("-TAIL"));
    assert!(bounded.contains("chars omitted"));
    Ok(())
}

#[tokio::test]
async fn without_speculation_carry_the_capable_tier_sees_the_original_request() -> Result<()> {
    let carried = Arc::new(Mutex::new(String::new()));
    let seen = Arc::clone(&carried);
    let route = Arc::new(super::super::Vgr::new(active())?);

    test_drive(
        route,
        request("solve this"),
        move |t: ModelId, r: Request| {
            let seen = Arc::clone(&seen);
            async move {
                if t == *CLOUD {
                    *seen.lock() = format!("{:?}", r.llm_request.messages);
                }
                let result: ServeResult = if t == *LOCAL {
                    Ok(reply_with_readout("my draft solution", 0.01))
                } else {
                    Ok(reply("cloud answer"))
                };
                result
            }
        },
    )
    .await?;

    assert!(!carried.lock().contains("my draft solution"));
    Ok(())
}

/// A checker that reports a fixed verdict.
struct FixedChecker(Option<bool>);

#[async_trait]
impl Checker for FixedChecker {
    async fn check(&self, _request: CheckerRequest<'_>) -> Option<bool> {
        self.0
    }
}

#[derive(Clone, Debug)]
struct ObservedCheck {
    task_text: String,
    attempt: String,
    deadline: Instant,
    remaining: Duration,
    manifest_identity: String,
}

struct HangingChecker {
    observed: Arc<Mutex<Option<ObservedCheck>>>,
    dropped: Arc<AtomicBool>,
}

#[async_trait]
impl Checker for HangingChecker {
    async fn check(&self, request: CheckerRequest<'_>) -> Option<bool> {
        *self.observed.lock() = Some(ObservedCheck {
            task_text: request.task_text.to_string(),
            attempt: request.attempt.to_string(),
            deadline: request.deadline,
            remaining: request.remaining,
            manifest_identity: request.manifest_identity.to_string(),
        });
        let _drop_signal = DropSignal(Arc::clone(&self.dropped));
        std::future::pending::<Option<bool>>().await
    }
}

#[cfg(any(unix, windows))]
struct EmptyWorkspaceProvider;

#[cfg(any(unix, windows))]
#[async_trait::async_trait]
impl WorkspaceProvider for EmptyWorkspaceProvider {
    fn manifest_identity(&self) -> &str {
        "runtime-test-workspace-v1"
    }

    async fn materialize(
        &self,
        _task_text: &str,
        _attempt: &str,
    ) -> std::io::Result<CandidateWorkspace> {
        CandidateWorkspace::new(TempDir::with_prefix("vgr-runtime-candidate-")?)
    }
}

#[tokio::test]
async fn checker_receives_pinned_deadline_and_is_cancelled_at_the_budget() -> Result<()> {
    const CHECKER_BUDGET: Duration = Duration::from_millis(30);
    const MANIFEST: &str = "sha256:fedcba9876543210fedcba9876543210";

    let observed = Arc::new(Mutex::new(None));
    let dropped = Arc::new(AtomicBool::new(false));
    let checker = HangingChecker {
        observed: Arc::clone(&observed),
        dropped: Arc::clone(&dropped),
    };
    let mut config = active();
    config.deadline = CHECKER_BUDGET;
    config.checker = Some(ValidatedChecker::new(Arc::new(checker), MANIFEST)?);
    let route = Arc::new(super::super::Vgr::new(config)?);
    let started = Instant::now();

    let run = test_drive(
        route,
        request("fix the build"),
        |target: ModelId, _request| async move {
            if target == *LOCAL {
                Ok(reply("a patch"))
            } else {
                Ok(reply("cloud answer"))
            }
        },
    );
    let (target, _) = tokio::time::timeout(Duration::from_secs(1), run)
        .await
        .map_err(|source| LibsyError::external("waiting for checker deadline", source))??;

    assert_eq!(target, ModelId::from(CLOUD));
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "checker exceeded the runtime-enforced deadline"
    );
    assert!(
        dropped.load(Ordering::SeqCst),
        "checker future was not cancelled at its deadline"
    );
    let observed = observed
        .lock()
        .clone()
        .ok_or_else(|| test_error("checker invocation was not observed"))?;
    assert_eq!(observed.task_text, "fix the build");
    assert_eq!(observed.attempt, "a patch");
    assert!(observed.deadline > started);
    assert!(observed.remaining <= CHECKER_BUDGET);
    assert!(!observed.remaining.is_zero());
    assert_eq!(observed.manifest_identity, MANIFEST);
    Ok(())
}

#[tokio::test]
async fn a_checker_pass_commits_and_spends_nothing_on_verifiers() -> Result<()> {
    // The sandboxed checker is ground truth: it supersedes every other rung on
    // its branch, so no verifier call is made at all.
    let log = CallLog::default();
    let seen = log.clone();
    let config = VgrConfig {
        checker: Some(validated_checker(Some(true))?),
        ..active()
    };
    let route = Arc::new(super::super::Vgr::new(config)?);

    let (target, _) = test_drive(route, request("fix the build"), move |t: ModelId, _r| {
        let log = seen.clone();
        async move {
            log.record(&t);
            let result: ServeResult = Ok(reply("a patch"));
            result
        }
    })
    .await?;

    assert_eq!(target, ModelId::from(LOCAL));
    // Only the attempt was paid for.
    assert_eq!(log.targets(), vec![LOCAL]);
    Ok(())
}

#[tokio::test]
async fn a_checker_that_cannot_run_commits_nothing() -> Result<()> {
    // An indeterminate checker is not a pass. The branch has no other rung to
    // fall back on, so the turn escalates.
    let config = VgrConfig {
        checker: Some(validated_checker(None)?),
        ..active()
    };
    let route = Arc::new(super::super::Vgr::new(config)?);

    let (target, _) = test_drive(
        route,
        request("fix the build"),
        |t: ModelId, _r| async move {
            let result: ServeResult = if t == *LOCAL {
                Ok(reply("a patch"))
            } else {
                Ok(reply("cloud answer"))
            };
            result
        },
    )
    .await?;

    assert_eq!(target, ModelId::from(CLOUD));
    Ok(())
}

#[tokio::test]
async fn coding_without_a_validated_checker_cannot_commit() -> Result<()> {
    // The readiness gate: a coding attempt may be decided local, but without a
    // validated checker the deployment is not allowed to serve that decision.
    let route = Arc::new(super::super::Vgr::new(active())?);

    let (target, _) = test_drive(
        route,
        request("fix the build"),
        |t: ModelId, _r| async move {
            let result: ServeResult = if t == *LOCAL {
                Ok(reply("a patch"))
            } else {
                Ok(reply("cloud answer"))
            };
            result
        },
    )
    .await?;

    assert_eq!(target, ModelId::from(CLOUD));
    Ok(())
}

// ─── operator controls ───────────────────────────────────────────────────────

#[tokio::test]
async fn an_engaged_kill_switch_escalates_without_producing_an_attempt() {
    // The operator's stop must cost nothing: no attempt, no verifier, no local
    // call of any kind.
    let switch = KillSwitch::new();
    switch.engage();
    let log = CallLog::default();
    let seen = log.clone();
    let config = VgrConfig {
        kill_switch: Some(switch.clone()),
        ..active()
    };
    let route = Arc::new(super::super::Vgr::new(config).expect("builds"));

    let (target, _) = test_drive(route.clone(), request("what is the capital?"), {
        let seen = seen.clone();
        move |t: ModelId, _r| {
            let log = seen.clone();
            async move {
                log.record(&t);
                let result: ServeResult = Ok(reply("cloud answer"));
                result
            }
        }
    })
    .await
    .expect("routes");

    assert_eq!(target, ModelId::from(CLOUD));
    assert_eq!(log.targets(), vec![CLOUD]);

    // Releasing it restores verification on the very next turn, without the
    // route being rebuilt.
    switch.release();
    let (target, _) = test_drive(
        route,
        request("what is the capital?"),
        move |t: ModelId, _r| {
            let log = seen.clone();
            async move {
                log.record(&t);
                let result: ServeResult = if t == *LOCAL {
                    Ok(reply_with_readout("Paris.", 0.97))
                } else {
                    Ok(reply("cloud answer"))
                };
                result
            }
        },
    )
    .await
    .expect("routes");
    assert_eq!(target, ModelId::from(LOCAL));
}

#[tokio::test]
async fn a_dead_local_endpoint_stops_being_called_once_the_breaker_opens() {
    // Without the breaker every request pays a fresh failed call forever. The
    // turn still escalates either way, so what is asserted is the cost.
    let log = CallLog::default();
    let config = VgrConfig {
        breaker: BreakerConfig {
            threshold: 2,
            cooldown: Duration::from_secs(60),
        },
        ..active()
    };
    let route = Arc::new(super::super::Vgr::new(config).expect("builds"));

    // Two failed turns open the circuit. The local tier answers nothing; the
    // context-overflow class is used because it escalates rather than erroring.
    for _ in 0..2 {
        let seen = log.clone();
        let (target, _) = test_drive(route.clone(), request("hello"), move |t: ModelId, _r| {
            let log = seen.clone();
            async move {
                log.record(&t);
                let result: ServeResult = if t == *LOCAL {
                    Err(LlmClientError::Transport {
                        source: std::io::Error::other("connection refused").into(),
                    })
                } else {
                    Ok(reply("cloud answer"))
                };
                result
            }
        })
        .await
        .expect("routes");
        assert_eq!(target, ModelId::from(CLOUD));
    }
    assert_eq!(log.targets(), vec![LOCAL, CLOUD, LOCAL, CLOUD]);

    // The third turn skips the local tier entirely.
    let seen = log.clone();
    let (target, _) = test_drive(route, request("hello"), move |t: ModelId, _r| {
        let log = seen.clone();
        async move {
            log.record(&t);
            let result: ServeResult = Ok(reply("cloud answer"));
            result
        }
    })
    .await
    .expect("routes");
    assert_eq!(target, ModelId::from(CLOUD));
    assert_eq!(log.targets().iter().filter(|t| *t == LOCAL).count(), 2);
}

#[tokio::test]
async fn repeated_cloud_failures_open_the_terminal_breaker() {
    let calls = Arc::new(AtomicUsize::new(0));
    let config = VgrConfig {
        breaker: BreakerConfig {
            threshold: 2,
            cooldown: Duration::from_secs(60),
        },
        ..VgrConfig::new(ModelId::from(LOCAL), ModelId::from(CLOUD))
    };
    let route = Arc::new(super::super::Vgr::new(config).expect("builds"));

    for _ in 0..2 {
        let seen = calls.clone();
        let error = match test_drive(route.clone(), request("hello"), move |_target, _request| {
            let seen = seen.clone();
            async move {
                seen.fetch_add(1, Ordering::SeqCst);
                Err(LlmClientError::Transport {
                    source: std::io::Error::other("cloud unavailable").into(),
                })
            }
        })
        .await
        {
            Err(error) => error,
            Ok(_) => panic!("cloud call should fail"),
        };
        assert!(matches!(error, LibsyError::ClientCall { .. }));
    }

    let seen = calls.clone();
    let error = match test_drive(route, request("hello"), move |_target, _request| {
        let seen = seen.clone();
        async move {
            seen.fetch_add(1, Ordering::SeqCst);
            Ok(reply("must not be called"))
        }
    })
    .await
    {
        Err(error) => error,
        Ok(_) => panic!("the open cloud breaker should fail fast"),
    };
    assert!(matches!(error, LibsyError::CircuitOpen { .. }));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn a_successful_cloud_half_open_trial_closes_the_breaker() -> Result<()> {
    let calls = Arc::new(AtomicUsize::new(0));
    let config = VgrConfig {
        breaker: BreakerConfig {
            threshold: 1,
            cooldown: Duration::from_millis(20),
        },
        ..VgrConfig::new(ModelId::from(LOCAL), ModelId::from(CLOUD))
    };
    let route = Arc::new(super::super::Vgr::new(config)?);

    let seen = calls.clone();
    let _error = match test_drive(route.clone(), request("hello"), move |_target, _request| {
        let seen = seen.clone();
        async move {
            seen.fetch_add(1, Ordering::SeqCst);
            Err(LlmClientError::Transport {
                source: std::io::Error::other("cloud unavailable").into(),
            })
        }
    })
    .await
    {
        Err(error) => error,
        Ok(_) => panic!("first cloud call should fail"),
    };

    tokio::time::sleep(Duration::from_millis(30)).await;
    for _ in 0..2 {
        let seen = calls.clone();
        let (target, _) = test_drive(route.clone(), request("hello"), move |_target, _request| {
            let seen = seen.clone();
            async move {
                seen.fetch_add(1, Ordering::SeqCst);
                Ok(reply("cloud recovered"))
            }
        })
        .await?;
        assert_eq!(target, ModelId::from(CLOUD));
    }
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    Ok(())
}

#[tokio::test]
async fn dual_endpoint_outage_returns_the_typed_combined_error() {
    let route = Arc::new(super::super::Vgr::new(active()).expect("builds"));
    let error = match test_drive(
        route,
        request("hello"),
        |target: ModelId, _request| async move {
            Err(LlmClientError::Transport {
                source: std::io::Error::other(format!("{target} unavailable")).into(),
            })
        },
    )
    .await
    {
        Err(error) => error,
        Ok(_) => panic!("both tiers should fail"),
    };

    assert!(matches!(
        error,
        LibsyError::VgrTiersUnavailable { local, cloud }
            if matches!(&*local, LibsyError::ClientCall { target, .. } if target == LOCAL)
                && matches!(&*cloud, LibsyError::ClientCall { target, .. } if target == CLOUD)
    ));
}

#[tokio::test]
async fn cloud_context_overflow_remains_the_terminal_client_error() {
    let route = Arc::new(super::super::Vgr::new(active()).expect("builds"));
    let error = match test_drive(
        route,
        request("hello"),
        |target: ModelId, _request| async move {
            Err(LlmClientError::ContextWindowExceeded {
                model: target.clone(),
                message: format!("{target} context exceeded"),
            })
        },
    )
    .await
    {
        Err(error) => error,
        Ok(_) => panic!("both tiers reject the request"),
    };

    assert!(matches!(
        error,
        LibsyError::ClientCall {
            target,
            source: LlmClientError::ContextWindowExceeded { .. },
        } if target == CLOUD
    ));
}

#[tokio::test]
async fn a_hung_verifier_cannot_overrun_the_decision_budget() {
    // The deadline binds each call, not just the gaps between them: a verifier
    // that accepts the request and never answers must not hold the turn open.
    let config = VgrConfig {
        deadline: Duration::from_millis(150),
        ..active()
    };
    let route = Arc::new(super::super::Vgr::new(config).expect("builds"));

    let started = std::time::Instant::now();
    let (target, _) = test_drive(
        route,
        request("what is the capital?"),
        |t: ModelId, _r| async move {
            let result: ServeResult = if t == *LOCAL {
                // The attempt answers; the readout that follows never does.
                static ATTEMPTED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if ATTEMPTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    tokio::time::sleep(Duration::from_secs(30)).await;
                    Ok(reply("never arrives"))
                } else {
                    Ok(reply("Paris."))
                }
            } else {
                Ok(reply("cloud answer"))
            };
            result
        },
    )
    .await
    .expect("routes");

    // Evidence that was never gathered escalates, and the turn ends near the
    // budget rather than near the verifier's own timeout.
    assert_eq!(target, ModelId::from(CLOUD));
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "{:?}",
        started.elapsed()
    );
}

#[cfg(unix)]
#[tokio::test]
async fn the_vgr_deadline_cancels_a_pinned_checker_process_tree()
-> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let marker_owner = TempDir::with_prefix("vgr-runtime-marker-")?;
    let marker = marker_owner.path().join("grandchild-alive");
    let tests = TempDir::with_prefix("vgr-runtime-suite-")?;
    std::fs::write(
        tests.path().join("run.sh"),
        format!(
            "sh -c 'while true; do touch {}; sleep 0.02; done' &\n\
             while [ ! -e {} ]; do :; done\n\
             sleep 30\n",
            marker.display(),
            marker.display()
        ),
    )?;
    let checker = PinnedChecker::new(CheckerConfig {
        timeout: Duration::from_secs(10),
        sandbox_attestation: SANDBOX_ATTESTATION.to_string(),
        ..CheckerConfig::new(
            tests.path(),
            vec!["/bin/sh".to_string(), "{tests}/run.sh".to_string()],
            Arc::new(EmptyWorkspaceProvider),
        )
    })?;
    let manifest_identity = checker.manifest_identity().to_owned();
    let config = VgrConfig {
        checker: Some(ValidatedChecker::new(Arc::new(checker), manifest_identity)?),
        deadline: Duration::from_millis(500),
        ..active()
    };
    let route = Arc::new(super::super::Vgr::new(config)?);

    let (target, _) = test_drive(
        route,
        request("fix the build"),
        |t: ModelId, _r| async move {
            let result: ServeResult = if t == *LOCAL {
                Ok(reply("candidate patch"))
            } else {
                Ok(reply("cloud answer"))
            };
            result
        },
    )
    .await?;
    assert_eq!(target, ModelId::from(CLOUD));
    assert!(marker.exists(), "the checker grandchild never started");

    tokio::time::sleep(Duration::from_millis(200)).await;
    let settled = std::fs::metadata(&marker)?.modified()?;
    tokio::time::sleep(Duration::from_millis(200)).await;
    let after = std::fs::metadata(&marker)?.modified()?;
    assert_eq!(
        settled, after,
        "the checker grandchild survived the outer VGR deadline"
    );
    Ok(())
}

#[cfg(windows)]
#[tokio::test]
async fn the_vgr_deadline_cancels_a_pinned_checker_process_tree()
-> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let marker_owner = TempDir::with_prefix("vgr-runtime-marker-")?;
    let marker = marker_owner.path().join("grandchild-alive");
    let marker_literal = marker.to_string_lossy().replace('\'', "''");
    let tests = TempDir::with_prefix("vgr-runtime-suite-")?;
    std::fs::write(
        tests.path().join("run.ps1"),
        format!(
            "$childScript = \"while (`$true) {{ \
                 [IO.File]::WriteAllText('{marker_literal}', 'alive'); \
                 Start-Sleep -Milliseconds 20 \
             }}\"\r\n\
             Start-Process powershell.exe -ArgumentList @('-NoLogo', '-NoProfile', \
                 '-NonInteractive', '-Command', $childScript) | Out-Null\r\n\
             while (!(Test-Path -LiteralPath '{marker_literal}')) {{ }}\r\n\
             Start-Sleep -Seconds 30\r\n"
        ),
    )?;
    let checker = PinnedChecker::new(CheckerConfig {
        timeout: Duration::from_secs(10),
        sandbox_attestation: SANDBOX_ATTESTATION.to_string(),
        ..CheckerConfig::new(
            tests.path(),
            vec![
                "powershell.exe".to_string(),
                "-NoLogo".to_string(),
                "-NoProfile".to_string(),
                "-NonInteractive".to_string(),
                "-ExecutionPolicy".to_string(),
                "Bypass".to_string(),
                "-Command".to_string(),
                "& (Join-Path $env:TESTS_DIR 'run.ps1')".to_string(),
            ],
            Arc::new(EmptyWorkspaceProvider),
        )
    })?;
    let manifest_identity = checker.manifest_identity().to_owned();
    let config = VgrConfig {
        checker: Some(ValidatedChecker::new(Arc::new(checker), manifest_identity)?),
        deadline: Duration::from_secs(3),
        ..active()
    };
    let route = Arc::new(super::super::Vgr::new(config)?);

    let (target, _) = test_drive(
        route,
        request("fix the build"),
        |t: ModelId, _r| async move {
            let result: ServeResult = if t == *LOCAL {
                Ok(reply("candidate patch"))
            } else {
                Ok(reply("cloud answer"))
            };
            result
        },
    )
    .await?;
    assert_eq!(target, ModelId::from(CLOUD));
    assert!(marker.exists(), "the checker grandchild never started");

    tokio::time::sleep(Duration::from_millis(300)).await;
    let settled = std::fs::metadata(&marker)?.modified()?;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let after = std::fs::metadata(&marker)?.modified()?;
    assert_eq!(
        settled, after,
        "the checker grandchild survived the outer VGR deadline"
    );
    Ok(())
}

#[tokio::test]
async fn local_tool_continuations_and_new_user_turns_reenter_vgr() -> Result<()> {
    let route = Arc::new(super::super::Vgr::new(active())?);
    let opening = session_request("session-1", "what is the capital?");
    let seed_calls = CallLog::default();
    let seen = seed_calls.clone();

    let (target, _) = test_drive(route.clone(), opening.clone(), move |t: ModelId, _r| {
        let seen = seen.clone();
        async move {
            seen.record(&t);
            if seen.targets().len() == 1 {
                Ok(reply("Paris."))
            } else {
                Ok(reply_with_readout("yes", 0.97))
            }
        }
    })
    .await?;
    assert_eq!(target, ModelId::from(LOCAL));

    let continued = tool_continuation(opening);
    let continuation_calls = CallLog::default();
    let seen = continuation_calls.clone();
    let (target, _) = test_drive(route.clone(), continued.clone(), move |t: ModelId, r| {
        let seen = seen.clone();
        async move {
            seen.record(&t);
            if is_turn_verification(&r) {
                Ok(reply_with_readout("no", 0.2))
            } else {
                Ok(tool_reply("read_file"))
            }
        }
    })
    .await?;
    assert_eq!(target, ModelId::from(LOCAL));
    assert_eq!(continuation_calls.targets(), vec![LOCAL, LOCAL]);

    let mut fresh_turn = continued;
    fresh_turn
        .llm_request
        .messages
        .push(Message::text(Role::Assistant, "The command completed."));
    fresh_turn
        .llm_request
        .messages
        .push(Message::text(Role::User, "Now use a different approach."));
    let fresh_calls = CallLog::default();
    let seen = fresh_calls.clone();
    let (target, _) = test_drive(route, fresh_turn, move |t: ModelId, _r| {
        let seen = seen.clone();
        async move {
            seen.record(&t);
            if t == *LOCAL {
                Err(LlmClientError::Transport {
                    source: std::io::Error::other("connection refused").into(),
                })
            } else {
                Ok(reply("cloud answer"))
            }
        }
    })
    .await?;
    assert_eq!(target, ModelId::from(CLOUD));
    assert_eq!(fresh_calls.targets(), vec![LOCAL, CLOUD]);
    Ok(())
}

#[tokio::test]
async fn requests_without_session_identity_are_reverified() -> Result<()> {
    let route = Arc::new(super::super::Vgr::new(active())?);
    let seed_calls = CallLog::default();
    let seen = seed_calls.clone();
    let (target, _) = test_drive(
        route.clone(),
        request("first turn"),
        move |t: ModelId, _r| {
            let seen = seen.clone();
            async move {
                seen.record(&t);
                if seen.targets().len() == 1 {
                    Ok(reply("local answer"))
                } else {
                    Ok(reply_with_readout("yes", 0.97))
                }
            }
        },
    )
    .await?;
    assert_eq!(target, ModelId::from(LOCAL));

    let second_calls = CallLog::default();
    let seen = second_calls.clone();
    let (target, _) = test_drive(route, request("second turn"), move |t: ModelId, _r| {
        let seen = seen.clone();
        async move {
            seen.record(&t);
            if t == *LOCAL {
                Err(LlmClientError::Transport {
                    source: std::io::Error::other("connection refused").into(),
                })
            } else {
                Ok(reply("cloud answer"))
            }
        }
    })
    .await?;
    assert_eq!(target, ModelId::from(CLOUD));
    assert_eq!(second_calls.targets(), vec![LOCAL, CLOUD]);
    Ok(())
}

#[tokio::test]
async fn concurrent_sessions_retain_independent_turn_tiers() -> Result<()> {
    let route = Arc::new(super::super::Vgr::new(active())?);
    let local_opening = session_request("session-local", "solve locally");
    let cloud_opening = session_request("session-cloud", "solve in cloud");

    let local_seed_calls = CallLog::default();
    let seen = local_seed_calls.clone();
    let (target, _) = test_drive(
        route.clone(),
        local_opening.clone(),
        move |t: ModelId, _r| {
            let seen = seen.clone();
            async move {
                seen.record(&t);
                if seen.targets().len() == 1 {
                    Ok(reply("local answer"))
                } else {
                    Ok(reply_with_readout("yes", 0.97))
                }
            }
        },
    )
    .await?;
    assert_eq!(target, ModelId::from(LOCAL));

    let (target, _) = test_drive(
        route.clone(),
        cloud_opening.clone(),
        |t: ModelId, _r| async move {
            if t == *LOCAL {
                Err(LlmClientError::Transport {
                    source: std::io::Error::other("connection refused").into(),
                })
            } else {
                Ok(reply("cloud answer"))
            }
        },
    )
    .await?;
    assert_eq!(target, ModelId::from(CLOUD));

    let local_calls = CallLog::default();
    let seen_local = local_calls.clone();
    let cloud_calls = CallLog::default();
    let seen_cloud = cloud_calls.clone();
    let local_turn = test_drive(
        route.clone(),
        tool_continuation(local_opening),
        move |t: ModelId, r| {
            let seen = seen_local.clone();
            async move {
                seen.record(&t);
                if is_turn_verification(&r) {
                    Ok(reply_with_readout("no", 0.2))
                } else {
                    Ok(tool_reply("read_file"))
                }
            }
        },
    );
    let cloud_turn = test_drive(
        route,
        tool_continuation(cloud_opening),
        move |t: ModelId, _r| {
            let seen = seen_cloud.clone();
            async move {
                seen.record(&t);
                Ok(reply("cloud continuation"))
            }
        },
    );
    let (local_result, cloud_result) = tokio::join!(local_turn, cloud_turn);
    assert_eq!(local_result?.0, ModelId::from(LOCAL));
    assert_eq!(cloud_result?.0, ModelId::from(CLOUD));
    assert_eq!(local_calls.targets(), vec![LOCAL, LOCAL]);
    assert_eq!(cloud_calls.targets(), vec![CLOUD]);
    Ok(())
}

#[tokio::test]
async fn an_escalated_session_stays_on_the_capable_tier() {
    // The whole-session cloud latch runs ahead of per-turn affinity, so a fresh
    // user message remains cloud and does not pay for another local attempt.
    let log = CallLog::default();
    let config = VgrConfig {
        latch_escalation: true,
        ..active()
    };
    let route = Arc::new(super::super::Vgr::new(config).expect("builds"));
    let session = || Request {
        metadata: Some(switchyard_protocol::Metadata {
            session_id: Some("session-1".to_string()),
            agent_id: Some("agent-a".to_string()),
            ..Default::default()
        }),
        ..request("what is the capital?")
    };

    let seen = log.clone();
    let (target, _) = test_drive(route.clone(), session(), move |t: ModelId, _r| {
        let log = seen.clone();
        async move {
            log.record(&t);
            let result: ServeResult = if t == *LOCAL {
                Ok(reply_with_readout("Paris.", 0.02))
            } else {
                Ok(reply("cloud answer"))
            };
            result
        }
    })
    .await
    .expect("routes");
    assert_eq!(target, ModelId::from(CLOUD));
    let verified_turn = log.targets().iter().filter(|t| *t == LOCAL).count();
    assert!(verified_turn > 0, "the first turn was verified");

    let seen = log.clone();
    let (target, _) = test_drive(route, session(), move |t: ModelId, _r| {
        let log = seen.clone();
        async move {
            log.record(&t);
            let result: ServeResult = Ok(reply("cloud answer"));
            result
        }
    })
    .await
    .expect("routes");

    assert_eq!(target, ModelId::from(CLOUD));
    // The latched turn produced no attempt and consulted no verifier.
    assert_eq!(
        log.targets().iter().filter(|t| *t == LOCAL).count(),
        verified_turn
    );
}

// ─── transcript tool evidence ────────────────────────────────────────────────

/// A request whose conversation carries one failed tool result.
fn request_with_tool_error(text: &str) -> Request {
    let mut base = request(text);
    base.llm_request.messages.push(Message {
        role: Role::User,
        content: vec![ContentBlock::ToolResult(ToolResult {
            tool_call_id: "call-1".to_string(),
            content: vec![ContentBlock::Text {
                text: "Traceback (most recent call last)\nModuleNotFoundError: no module named x"
                    .to_string(),
            }],
            is_error: None,
        })],
    });
    base
}

/// A request whose conversation carries one successful tool result.
fn request_with_clean_tool_result(text: &str) -> Request {
    let mut base = request(text);
    base.llm_request.messages.push(Message {
        role: Role::Assistant,
        content: vec![ContentBlock::ToolCall(ToolCall {
            id: "call-1".to_string(),
            name: "terminal".to_string(),
            arguments: serde_json::json!({"command": "deploy"}),
        })],
    });
    base.llm_request.messages.push(Message {
        role: Role::User,
        content: vec![ContentBlock::ToolResult(ToolResult {
            tool_call_id: "call-1".to_string(),
            content: vec![ContentBlock::Text {
                text: "Deployment completed successfully.".to_string(),
            }],
            is_error: Some(false),
        })],
    });
    base
}

#[tokio::test]
async fn a_reported_tool_error_vetoes_a_commit_the_evidence_would_otherwise_license() {
    // An error-bearing agentic run that is not eligible for the bounded
    // recovered-run arm is already terminal: it must spend zero verifier rungs.
    let log = CallLog::default();
    let seen = log.clone();
    let route = Arc::new(super::super::Vgr::new(active()).expect("builds"));
    let (target, _) = test_drive(
        route,
        request_with_tool_error("install the package"),
        move |t: ModelId, _r| {
            let seen = seen.clone();
            async move {
                seen.record(&t);
                let result: ServeResult = if t == *LOCAL {
                    Ok(reply_with_readout("[tool] install\nInstalled it.", 0.99))
                } else {
                    Ok(reply("cloud answer"))
                };
                result
            }
        },
    )
    .await
    .expect("routes");

    assert_eq!(target, ModelId::from(CLOUD));
    assert_eq!(log.targets(), vec![LOCAL, CLOUD]);
}

#[tokio::test]
async fn a_clean_transcript_tool_result_can_authorize_an_agentic_commit() {
    let log = CallLog::default();
    let seen = log.clone();
    let route = Arc::new(super::super::Vgr::new(active()).expect("builds"));
    let mut transcript = request_with_clean_tool_result("run the deployment");
    transcript.llm_request.instructions.push(InstructionBlock {
        role: Role::System,
        content: vec![ContentBlock::Text {
            text: "framework boilerplate ".repeat(500),
        }],
    });
    let (target, _) = test_drive(route, transcript, move |t: ModelId, _r| {
        let seen = seen.clone();
        async move {
            seen.record(&t);
            let result: ServeResult = if t == *LOCAL {
                Ok(reply_with_readout(
                    "[tool] deploy\nAll steps completed.",
                    0.99,
                ))
            } else {
                Ok(reply("cloud answer"))
            };
            result
        }
    })
    .await
    .expect("routes");

    assert_eq!(target, ModelId::from(LOCAL));
    assert_eq!(log.targets(), vec![LOCAL, LOCAL]);
}

// ─── task typing and the agreement rung ──────────────────────────────────────

#[tokio::test]
async fn typing_the_request_makes_the_answer_regime_reachable() -> Result<()> {
    // Without typing, no request reaches the answer regime at all: derivation
    // only carries a final answer when the router typed the task as one.
    let log = CallLog::default();
    let seen = log.clone();
    let config = VgrConfig {
        task_typing: true,
        structured_answer: true,
        ..active()
    };
    let route = Arc::new(super::super::Vgr::new(config).expect("builds"));

    let (target, response) = test_drive(
        route,
        request("what is the capital of France?"),
        move |t: ModelId, r: Request| {
            let log = seen.clone();
            async move {
                log.record(&t);
                let asked = r
                    .llm_request
                    .messages
                    .first()
                    .and_then(|m| m.text_content(""))
                    .unwrap_or_default();
                let system = r
                    .llm_request
                    .instructions
                    .iter()
                    .flat_map(|instruction| &instruction.content)
                    .find_map(|block| match block {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .unwrap_or_default();
                let result: ServeResult = match log.targets().len() {
                    1 => Ok(reply("Paris.")),
                    2 => {
                        assert_eq!(r.llm_request.reasoning.effort.as_deref(), Some("none"));
                        Ok(reply("answer"))
                    }
                    // The local rungs come first and neither commits.
                    3 => Ok(reply_with_readout("yes", 0.1)),
                    4 => {
                        assert!(system.contains("no more than three short sentences"));
                        assert!(system.contains("finish within the available token budget"));
                        assert_eq!(r.llm_request.reasoning.effort, None);
                        assert_eq!(r.llm_request.sampling.temperature, Some(0.0));
                        assert_eq!(
                            r.llm_request.output.max_output_tokens,
                            Some(super::super::rungs::LOCAL_DELIBERATION_MAX_OUTPUT_TOKENS)
                        );
                        Ok(reply("no"))
                    }
                    _ => {
                        // The witness is asked the task and never shown the attempt,
                        // so agreement between the two is evidence, not an echo.
                        assert!(
                            !asked.contains("Paris."),
                            "witness saw the attempt: {asked}"
                        );
                        Ok(reply("Paris."))
                    }
                };
                result
            }
        },
    )
    .await?;

    assert_eq!(target, ModelId::from(LOCAL));
    assert_eq!(served_text(response).await?, "Paris.");
    // Cheapest-first: the attempt, typing, readout and deliberation are all
    // local, and the cloud witness is only reached once none of them settled it.
    assert_eq!(log.targets(), vec![LOCAL, LOCAL, LOCAL, LOCAL, CLOUD]);
    Ok(())
}

#[tokio::test]
async fn a_confident_local_readout_commits_an_answer_without_paying_for_a_witness() -> Result<()> {
    // The witness is a cloud call and the readout is four local tokens, so an
    // answer the readout can settle must never reach the witness.
    let log = CallLog::default();
    let seen = log.clone();
    let config = VgrConfig {
        task_typing: true,
        structured_answer: true,
        ..active()
    };
    let route = Arc::new(super::super::Vgr::new(config)?);

    let (target, _) = test_drive(
        route,
        request("what is the capital of France?"),
        move |t: ModelId, _r| {
            let log = seen.clone();
            async move {
                log.record(&t);
                let result: ServeResult = match log.targets().len() {
                    1 => Ok(reply("Paris.")),
                    2 => Ok(reply("answer")),
                    // Clears the answer branch's dial bar on its own.
                    _ => Ok(reply_with_readout("yes", 0.95)),
                };
                result
            }
        },
    )
    .await?;

    assert_eq!(target, ModelId::from(LOCAL));
    assert_eq!(log.targets(), vec![LOCAL, LOCAL, LOCAL]);
    Ok(())
}

#[tokio::test]
async fn an_unstructured_answer_surface_never_pays_for_a_witness() -> Result<()> {
    // The answer rule reads agreement only on an operator-declared structured
    // surface, so asking a witness anywhere else buys nothing.
    let log = CallLog::default();
    let seen = log.clone();
    let config = VgrConfig {
        task_typing: true,
        structured_answer: false,
        ..active()
    };
    let route = Arc::new(super::super::Vgr::new(config)?);

    let (target, _) = test_drive(
        route,
        request("what is the capital of France?"),
        move |t: ModelId, _r| {
            let log = seen.clone();
            async move {
                log.record(&t);
                let result: ServeResult = match log.targets().len() {
                    1 => Ok(reply("Paris.")),
                    2 => Ok(reply("answer")),
                    3 => Ok(reply_with_readout("yes", 0.1)),
                    4 => Ok(reply("no")),
                    _ => Ok(reply("cloud answer")),
                };
                result
            }
        },
    )
    .await?;

    assert_eq!(target, ModelId::from(CLOUD));
    // No cloud judge is configured, so the turn escalates after the local rungs
    // without a witness call.
    assert_eq!(log.targets(), vec![LOCAL, LOCAL, LOCAL, LOCAL, CLOUD]);
    Ok(())
}

#[tokio::test]
async fn a_witness_that_disagrees_does_not_commit_the_answer() {
    let config = VgrConfig {
        task_typing: true,
        structured_answer: true,
        ..active()
    };
    let route = Arc::new(super::super::Vgr::new(config).expect("builds"));
    let log = CallLog::default();
    let seen = log.clone();

    let (target, _) = test_drive(
        route,
        request("what is the capital of France?"),
        move |t: ModelId, _r| {
            let log = seen.clone();
            async move {
                log.record(&t);
                let result: ServeResult = match log.targets().len() {
                    1 => Ok(reply("Lyon.")),
                    2 => Ok(reply("answer")),
                    // An independently produced answer that contradicts the attempt.
                    3 => Ok(reply("Paris.")),
                    _ => Ok(reply_with_readout("no", 0.01)),
                };
                result
            }
        },
    )
    .await
    .expect("routes");

    assert_eq!(target, ModelId::from(CLOUD));
}

#[tokio::test]
async fn an_unparsable_typing_reply_abstains_to_the_default_regime() -> Result<()> {
    // Abstention must select the default regime, which is more conservative
    // than any type would have been — never a weaker one.
    let config = VgrConfig {
        task_typing: true,
        ..active()
    };
    let route = Arc::new(super::super::Vgr::new(config).expect("builds"));
    let log = CallLog::default();
    let seen = log.clone();

    let (target, response) = test_drive(
        route,
        request("what is the capital?"),
        move |t: ModelId, _r| {
            let log = seen.clone();
            async move {
                log.record(&t);
                let result: ServeResult = match log.targets().len() {
                    1 => Ok(reply("Paris.")),
                    // Neither a known type nor the abstain word.
                    2 => Ok(reply("I think this is probably a question about geography")),
                    _ => Ok(reply_with_readout("yes", 0.97)),
                };
                result
            }
        },
    )
    .await?;

    // The default regime commits on a confident readout, so the turn still
    // resolves — it is just verified by the universal judge rather than by an
    // answer-specific rung.
    assert_eq!(target, ModelId::from(LOCAL));
    assert_eq!(served_text(response).await?, "Paris.");
    Ok(())
}

#[tokio::test]
async fn a_configured_checker_suppresses_the_typing_call() -> Result<()> {
    // The checker's regime is selected by operator configuration, so no type
    // can change it and paying for one would be waste.
    let log = CallLog::default();
    let seen = log.clone();
    let config = VgrConfig {
        task_typing: true,
        checker: Some(validated_checker(Some(true))?),
        ..active()
    };
    let route = Arc::new(super::super::Vgr::new(config)?);

    let (target, _) = test_drive(route, request("fix the build"), move |t: ModelId, _r| {
        let log = seen.clone();
        async move {
            log.record(&t);
            let result: ServeResult = Ok(reply("a patch"));
            result
        }
    })
    .await?;

    assert_eq!(target, ModelId::from(LOCAL));
    // The attempt, and nothing else.
    assert_eq!(log.targets(), vec![LOCAL]);
    Ok(())
}

#[cfg(unix)]
fn real_checker(
    script: &str,
) -> std::result::Result<ValidatedChecker, Box<dyn std::error::Error + Send + Sync>> {
    let suite = tempfile::tempdir()?;
    std::fs::write(suite.path().join("case.txt"), "pinned")?;
    let workspace_provider = CommandWorkspaceProvider::new(CommandWorkspaceProviderConfig::new(
        vec![
            "/bin/sh".to_string(),
            "-c".to_string(),
            "cp \"$ATTEMPT_FILE\" \"$WORKSPACE_DIR/candidate.txt\"".to_string(),
        ],
        Duration::from_secs(10),
    ))?;
    let mut config = CheckerConfig::new(
        suite.path(),
        vec!["/bin/sh".into(), "-c".into(), script.into()],
        Arc::new(workspace_provider),
    );
    config.sandbox_attestation = SANDBOX_ATTESTATION.into();
    let checker = PinnedChecker::new(config)?;
    let manifest_identity = checker.manifest_identity().to_owned();
    Ok(ValidatedChecker::new(Arc::new(checker), manifest_identity)?)
}

#[cfg(windows)]
fn real_checker(
    script: &str,
) -> std::result::Result<ValidatedChecker, Box<dyn std::error::Error + Send + Sync>> {
    let suite = tempfile::tempdir()?;
    std::fs::write(suite.path().join("case.txt"), "pinned")?;
    let workspace_provider = CommandWorkspaceProvider::new(CommandWorkspaceProviderConfig::new(
        vec![
            "powershell.exe".to_string(),
            "-NoLogo".to_string(),
            "-NoProfile".to_string(),
            "-NonInteractive".to_string(),
            "-Command".to_string(),
            "Copy-Item -LiteralPath $env:ATTEMPT_FILE \
             -Destination (Join-Path $env:WORKSPACE_DIR 'candidate.txt')"
                .to_string(),
        ],
        Duration::from_secs(10),
    ))?;
    let mut config = CheckerConfig::new(
        suite.path(),
        vec![
            "powershell.exe".into(),
            "-NoLogo".into(),
            "-NoProfile".into(),
            "-NonInteractive".into(),
            "-Command".into(),
            script.into(),
        ],
        Arc::new(workspace_provider),
    );
    config.sandbox_attestation = SANDBOX_ATTESTATION.into();
    let checker = PinnedChecker::new(config)?;
    let manifest_identity = checker.manifest_identity().to_owned();
    Ok(ValidatedChecker::new(Arc::new(checker), manifest_identity)?)
}

#[cfg(unix)]
#[tokio::test]
async fn active_mode_applies_real_checker_readiness_and_tamper_results()
-> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // These direct libsy cases complement the runner's checker-table
    // integration and prove Active serves only an effective, validated pass.
    for (script, validated, expected) in [
        ("exit 0", true, ModelId::from(LOCAL)),
        ("exit 0", false, ModelId::from(CLOUD)),
        (
            "chmod u+w {tests}/case.txt; printf tampered >> {tests}/case.txt; exit 0",
            true,
            ModelId::from(CLOUD),
        ),
    ] {
        let checker = if validated {
            Some(real_checker(script)?)
        } else {
            None
        };
        let config = VgrConfig {
            checker,
            ..active()
        };
        let route = Arc::new(super::super::Vgr::new(config)?);
        let (target, _) = test_drive(
            route,
            request("fix the build"),
            |t: ModelId, _r| async move {
                if t == *LOCAL {
                    Ok(reply("a patch"))
                } else {
                    Ok(reply("cloud"))
                }
            },
        )
        .await?;
        assert_eq!(target, expected, "{script} validated={validated}");
    }
    Ok(())
}

#[cfg(windows)]
#[tokio::test]
async fn active_mode_applies_real_checker_readiness_and_tamper_results()
-> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // The Windows path applies the same validated-pass and tamper gates using
    // native PowerShell commands and handle-based mutation stamps.
    for (script, validated, expected) in [
        ("exit 0", true, ModelId::from(LOCAL)),
        ("exit 0", false, ModelId::from(CLOUD)),
        (
            "$path = Join-Path $env:TESTS_DIR 'case.txt'; \
             (Get-Item -LiteralPath $path).IsReadOnly = $false; \
             Add-Content -LiteralPath $path -Value 'tampered'; \
             exit 0",
            true,
            ModelId::from(CLOUD),
        ),
    ] {
        let checker = if validated {
            Some(real_checker(script)?)
        } else {
            None
        };
        let config = VgrConfig {
            checker,
            ..active()
        };
        let route = Arc::new(super::super::Vgr::new(config)?);
        let (target, _) = test_drive(
            route,
            request("fix the build"),
            |t: ModelId, _r| async move {
                if t == *LOCAL {
                    Ok(reply("a patch"))
                } else {
                    Ok(reply("cloud"))
                }
            },
        )
        .await?;
        assert_eq!(target, expected, "{script} validated={validated}");
    }
    Ok(())
}

#[tokio::test]
async fn no_tool_turn_skips_in_flight_verification() -> crate::Result<()> {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let route = Arc::new(super::super::Vgr::new(active())?);
    let turn_judges = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&turn_judges);
    let (target, _) = test_drive(route, request("say hello"), move |t: ModelId, r| {
        let seen = Arc::clone(&seen);
        async move {
            if is_turn_verification(&r) {
                seen.fetch_add(1, Ordering::Relaxed);
            }
            let response = if t == *LOCAL {
                reply_with_readout("hello", 0.99)
            } else {
                reply("cloud")
            };
            Ok(response)
        }
    })
    .await?;

    assert_eq!(target, ModelId::from(LOCAL));
    assert_eq!(turn_judges.load(Ordering::Relaxed), 0);
    Ok(())
}

#[tokio::test]
async fn incomplete_tool_turns_escalate_without_verification() -> crate::Result<()> {
    for (arguments, stop_reason) in [
        (
            serde_json::json!({"path": "out.txt"}),
            StopReason::MaxTokens,
        ),
        (
            serde_json::Value::String(r#"{"path":"out.txt"#.into()),
            StopReason::ToolUse,
        ),
    ] {
        let (target, _) = test_drive(
            Arc::new(super::super::Vgr::new(active())?),
            request("write the file"),
            move |target: ModelId, request| {
                let arguments = arguments.clone();
                async move {
                    assert!(!is_turn_verification(&request));
                    Ok(if target == *LOCAL {
                        shaped_tool_reply(arguments, stop_reason)
                    } else {
                        reply("cloud")
                    })
                }
            },
        )
        .await?;
        assert_eq!(target, ModelId::from(CLOUD));
    }
    Ok(())
}

#[tokio::test]
async fn two_tool_turn_votes_latch_the_session_without_more_local_calls() -> crate::Result<()> {
    let route = Arc::new(super::super::Vgr::new(active())?);
    let log = CallLog::default();
    let opening = session_request("turn-session", "inspect the files");
    let turns = [
        (opening.clone(), ModelId::from(LOCAL)),
        (tool_continuation(opening), ModelId::from(CLOUD)),
    ];

    for (request, expected) in turns {
        let seen = log.clone();
        let (target, _) = test_drive(route.clone(), request, move |t: ModelId, r| {
            let seen = seen.clone();
            async move {
                seen.record(&t);
                if is_turn_verification(&r) {
                    assert_eq!(r.llm_request.reasoning.effort.as_deref(), Some("none"));
                    Ok(reply_with_readout("yes", 0.8))
                } else if t == *LOCAL {
                    assert_eq!(r.llm_request.reasoning.effort, None);
                    Ok(tool_reply("read_file"))
                } else {
                    Ok(reply("cloud"))
                }
            }
        })
        .await?;
        assert_eq!(target, expected);
    }

    let local_calls_at_latch = log.targets().iter().filter(|t| *t == LOCAL).count();
    let seen = log.clone();
    let (target, _) = test_drive(
        route,
        session_request("turn-session", "inspect the files"),
        move |t: ModelId, _r| {
            let seen = seen.clone();
            async move {
                seen.record(&t);
                Ok(reply("cloud"))
            }
        },
    )
    .await?;
    assert_eq!(target, ModelId::from(CLOUD));
    assert_eq!(
        log.targets().iter().filter(|t| *t == LOCAL).count(),
        local_calls_at_latch,
        "latched sessions spend neither another candidate nor another judge call"
    );
    Ok(())
}

#[tokio::test]
async fn a_turn_verification_decline_clears_the_confirmation_streak() -> crate::Result<()> {
    let route = Arc::new(super::super::Vgr::new(active())?);
    let session = || Request {
        metadata: Some(switchyard_protocol::Metadata {
            session_id: Some("decline-session".into()),
            ..Default::default()
        }),
        ..request("inspect the files")
    };

    for probability in [0.8, 0.2, 0.8] {
        let (target, _) = test_drive(route.clone(), session(), move |t: ModelId, r| async move {
            if is_turn_verification(&r) {
                Ok(reply_with_readout("verdict", probability))
            } else if t == *LOCAL {
                Ok(tool_reply("read_file"))
            } else {
                Ok(reply("cloud"))
            }
        })
        .await?;
        assert_eq!(target, ModelId::from(LOCAL));
    }

    let (target, _) = test_drive(route, session(), move |t: ModelId, r| async move {
        if is_turn_verification(&r) {
            Ok(reply_with_readout("yes", 0.8))
        } else if t == *LOCAL {
            Ok(tool_reply("read_file"))
        } else {
            Ok(reply("cloud"))
        }
    })
    .await?;
    assert_eq!(target, ModelId::from(CLOUD));
    Ok(())
}

#[tokio::test]
async fn an_indeterminate_turn_vote_is_fail_open() -> crate::Result<()> {
    let route = Arc::new(super::super::Vgr::new(active())?);
    let (target, response) = test_drive(
        route,
        request("inspect the files"),
        |t: ModelId, r| async move {
            if is_turn_verification(&r) {
                Ok(reply("ambiguous"))
            } else if t == *LOCAL {
                Ok(tool_reply("read_file"))
            } else {
                Ok(reply("cloud"))
            }
        },
    )
    .await?;
    assert_eq!(target, ModelId::from(LOCAL));
    let aggregate = response.llm_response.into_agg().await.map_err(|source| {
        crate::LibsyError::AlgorithmError {
            message: format!("buffered tool response failed: {source}"),
        }
    })?;
    assert!(super::super::rungs::has_tool_call(&aggregate));
    Ok(())
}

#[tokio::test]
async fn a_non_transport_turn_judge_error_is_fail_open() -> crate::Result<()> {
    let route = Arc::new(super::super::Vgr::new(active())?);
    let (target, _) = test_drive(
        route,
        request("inspect the files"),
        |t: ModelId, r| async move {
            if is_turn_verification(&r) {
                Err(LlmClientError::General("judge rejected the prompt".into()))
            } else if t == *LOCAL {
                Ok(tool_reply("read_file"))
            } else {
                Ok(reply("cloud"))
            }
        },
    )
    .await?;
    assert_eq!(target, ModelId::from(LOCAL));
    Ok(())
}

#[tokio::test]
async fn local_transport_failure_during_turn_verification_latches_cloud() -> crate::Result<()> {
    let route = Arc::new(super::super::Vgr::new(active())?);
    let log = CallLog::default();
    let session = || Request {
        metadata: Some(switchyard_protocol::Metadata {
            session_id: Some("transport-session".into()),
            ..Default::default()
        }),
        ..request("inspect the files")
    };
    let seen = log.clone();
    let (target, _) = test_drive(route.clone(), session(), move |t: ModelId, r| {
        let seen = seen.clone();
        async move {
            seen.record(&t);
            if is_turn_verification(&r) {
                Err(LlmClientError::Transport {
                    source: std::io::Error::other("connection refused").into(),
                })
            } else if t == *LOCAL {
                Ok(tool_reply("read_file"))
            } else {
                Ok(reply("cloud"))
            }
        }
    })
    .await?;
    assert_eq!(target, ModelId::from(CLOUD));
    let local_calls = log.targets().iter().filter(|t| *t == LOCAL).count();

    let seen = log.clone();
    let (target, _) = test_drive(route, session(), move |t: ModelId, _r| {
        let seen = seen.clone();
        async move {
            seen.record(&t);
            Ok(reply("cloud"))
        }
    })
    .await?;
    assert_eq!(target, ModelId::from(CLOUD));
    assert_eq!(
        log.targets().iter().filter(|t| *t == LOCAL).count(),
        local_calls
    );
    Ok(())
}

#[test]
fn turn_view_preserves_anchors_and_proposed_turn_within_policy_budget() {
    let mut request = request("opening task");
    request
        .llm_request
        .instructions
        .push(switchyard_protocol::InstructionBlock {
            role: Role::System,
            content: vec![ContentBlock::Text {
                text: "system anchor sk-abcdefghijklmnopqrstuvwx".into(),
            }],
        });
    for index in 0..40 {
        request.llm_request.messages.push(Message::text(
            Role::Assistant,
            format!("window-{index} {}", "x".repeat(700)),
        ));
    }
    let policy = super::super::policy::Policy::CURRENT.turn_verification;
    let view = super::super::rungs::turn_view(&request, &tool_aggregate("read_file"), &policy);

    assert!(view.contains("[task] opening task"));
    assert!(view.contains("system anchor [REDACTED]"));
    assert!(view.contains("[proposed next turn]"));
    assert!(view.contains("read_file"));
    assert!(!view.contains("window-0"));
    assert!(view.chars().count() <= policy.max_chars);
}
