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
    ContentBlock, FormatId, LlmClientError, LlmResponse, LlmResponseChunk, LlmResponseStreamEvent,
    ModelId, PreservationMetadata, Request, Response, Usage, WireFormat, text_request,
    text_response,
};

use super::super::config::{Checker, CheckerRequest, ServingMode, ValidatedChecker, VgrConfig};
use super::super::mode::ACTIVE_APPROVAL;
use crate::core::testing::{ServeResult, test_drive};
use crate::{Algorithm, LibsyError, Result, Step};

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

/// An active route between the two tiers, with verification enabled.
fn active() -> VgrConfig {
    VgrConfig {
        mode: ServingMode::Active {
            approval: ACTIVE_APPROVAL.into(),
        },
        ..VgrConfig::new(ModelId::from(LOCAL), ModelId::from(CLOUD))
    }
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
    }
}

/// A plain buffered reply.
fn reply(text: &str) -> Response {
    Response {
        llm_response: LlmResponse::Agg(text_response(None, text.to_string())),
        metadata: None,
    }
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
    let stream = route.run_stream(request("hello"));
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
        .ok_or_else(|| test_error("missing terminal outcome"))??;
    let Step::Done(outcome) = step else {
        return Err(test_error(
            "eligible local failure triggered another routing call",
        ));
    };
    assert_eq!(outcome.selected_model_id, ModelId::from(CLOUD));
    assert!(outcome.fallback_models.is_empty());
    assert!(outcome.response.is_none());
    Ok(())
}

#[tokio::test]
async fn a_cloud_decision_never_falls_back_to_local() -> Result<()> {
    let route: Arc<dyn Algorithm> = Arc::new(super::super::Vgr::new(VgrConfig::new(
        ModelId::from(LOCAL),
        ModelId::from(CLOUD),
    ))?);
    let stream = route.run_stream(request("hello"));
    tokio::pin!(stream);

    let step = stream
        .next()
        .await
        .ok_or_else(|| test_error("missing terminal outcome"))??;
    let Step::Done(outcome) = step else {
        return Err(test_error("off mode unexpectedly made a model call"));
    };
    assert_eq!(outcome.selected_model_id, ModelId::from(CLOUD));
    assert!(outcome.fallback_models.is_empty());
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
