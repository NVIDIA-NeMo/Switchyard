// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The classifier that runs a verification-gated turn.
//!
//! One turn is: call the local tier, hold its answer, gather evidence about
//! that specific answer, then decide whether to release it or escalate.
//!
//! # Why this is one classifier rather than a cascade
//!
//! The evidence rungs look like a cascade, but they are not classifiers: a
//! classifier recommends a *target*, whereas a rung produces a *signal* that
//! only means something once the branch's rule reads all of them together.
//! Composing them as a cascade would also force signals through session state,
//! which holds only scalars. So the ladder is an ordered loop inside one
//! classifier, which is the shape the escalation router already uses.
//!
//! # Cost discipline
//!
//! Rungs run cheapest-first and stop as soon as the decision is settled. A
//! commit reachable from the local readout alone never pays for a cloud call.
//! Evidence that was never gathered stays absent, which the rules distinguish
//! from evidence that was gathered and came back indeterminate.

use std::time::Instant;

use async_trait::async_trait;
use futures::StreamExt;
use http::StatusCode;
use switchyard_protocol::{
    AggLlmResponse, LlmClientError, LlmResponse, LlmResponseStreamEvent, Request, Response,
};

use super::config::{ServingMode, VgrConfig};
use super::decide::{Decision, Readiness, Route, decide_from_signals};
use super::rules::{Signals, Tri};
use super::rungs::{self, Question};
use super::{Branch, Capabilities, ToolErrorsSource, derive_capabilities, matching, readout};
use crate::algorithms::util::decisive;
use crate::algorithms::util::prompts::{append_note, drop_exact_replay};
use crate::core::algorithm::Driver;
use crate::core::classifier::{Classification, Classifier};
use crate::core::state::State;
use crate::{LibsyError, Result};

/// Runs one verification-gated turn.
pub(super) struct VgrClassifier {
    pub(super) config: VgrConfig,
}

#[async_trait]
impl Classifier<State> for VgrClassifier {
    async fn score(
        &self,
        _state: &mut State,
        request: &mut Request,
        driver: Option<&Driver>,
    ) -> Result<(Classification, Option<Response>)> {
        let Some(driver) = driver else {
            return Err(LibsyError::AlgorithmError {
                message: "vgr classifier requires a driver".into(),
            });
        };
        let targets = &self.config.targets;

        // Off never spends anything: there is no decision to inform, so there
        // is no reason to produce an attempt that will not be served.
        if self.config.mode == ServingMode::Off {
            return Ok((decisive(&targets.cloud), None));
        }

        let started = Instant::now();
        let deadline = match started.checked_add(self.config.deadline) {
            Some(deadline) => deadline,
            None => started,
        };

        // Keep the attempt call local-only so a cloud escalation is prepared as
        // the terminal completion call. Eligible local failures select cloud
        // below instead of surfacing or falling backward later.
        let local_response = match driver
            .call_model(request.clone(), vec![targets.local.clone()])
            .await
        {
            Ok(response) => response,
            // An eligible local failure produced no attempt to verify. A host
            // that did not consume the candidate fallback may serve cloud next.
            Err(error) if fallback_eligible(&error) => {
                return Ok((decisive(&targets.cloud), None));
            }
            Err(error) => return Err(error),
        };
        let buffered = match BufferedResponse::new(local_response).await {
            Ok(buffered) => buffered,
            Err(source) if client_error_fallback_eligible(&source) => {
                return Ok((decisive(&targets.cloud), None));
            }
            Err(source) => return Err(LibsyError::client_call(targets.local.clone(), source)),
        };

        let attempt = rungs::response_text(buffered.aggregate()).unwrap_or_default();
        let caps = self.derive(request, &attempt);
        let signals = self
            .gather(driver, &caps, &attempt, request, deadline)
            .await?;
        let decision = decide_from_signals(&caps, &signals, &self.config.policy, &self.readiness());

        tracing::info!(
            branch = ?decision.branch,
            route = ?decision.route,
            effective = ?decision.effective_route,
            gate = ?decision.readiness_gate,
            elapsed_ms = started.elapsed().as_millis() as u64,
            "vgr decision"
        );

        if super::mode::serve_route(&self.config.mode, &decision) == Route::Local {
            // Return the original aggregate or the exact buffered event sequence.
            // No synthetic stream reconstruction is involved.
            return Ok((decisive(&targets.local), Some(buffered.into_response())));
        }

        // Escalating: the capable tier may be given the rejected attempt as
        // reference, so its work is not discarded.
        if self.config.speculation_carry && !attempt.trim().is_empty() {
            carry_attempt(request, &attempt, &decision);
        }
        Ok((decisive(&targets.cloud), None))
    }
}

impl VgrClassifier {
    /// Builds the capabilities this turn is judged under.
    ///
    /// Typing the request with a model call is a rung the reference runs; until
    /// that exists, derivation abstains on task type, which selects the default
    /// verification regime rather than a weaker one.
    fn derive(&self, request: &Request, attempt: &str) -> Capabilities {
        let mut caps = derive_capabilities(
            request,
            attempt,
            self.config.checker.is_some(),
            None,
            None,
            ToolErrorsSource::Host,
        );
        // Operator declarations, which derivation never produces on its own.
        caps.structured_answer = self.config.structured_answer;
        caps
    }

    /// Gathers evidence cheapest-first, stopping once the decision is settled.
    async fn gather(
        &self,
        driver: &Driver,
        caps: &Capabilities,
        attempt: &str,
        request: &Request,
        deadline: Instant,
    ) -> Result<Signals> {
        let mut signals = Signals::default();
        let branch = super::select_branch(caps);
        // Nothing to verify: no rung can change the outcome, so none run.
        if branch == Branch::Unknown {
            return Ok(signals);
        }
        let Some(judged) = caps.transcript.as_deref() else {
            return Ok(signals);
        };

        // The sandboxed checker is ground truth and supersedes every other
        // rung on its branch, so it runs alone and first.
        if branch == Branch::Checks {
            if let Some(checker) = &self.config.checker {
                let task = match caps.task_text.as_deref() {
                    Some(task) => task,
                    None => return Ok(signals),
                };
                signals.tests_pass = Some(match checker.check(task, attempt, deadline).await {
                    Some(true) => Tri::Yes,
                    Some(false) => Tri::No,
                    None => Tri::Unknown,
                });
            }
            return Ok(signals);
        }

        // Typed agreement is free: it compares text this router already holds.
        if branch == Branch::Answer
            && let (Some(answer), Some(task)) =
                (caps.final_answer.as_deref(), caps.task_text.as_deref())
        {
            signals.agreement = Some(matching::match_answer_verdict(task, answer));
            if self.settled(caps, &signals) {
                return Ok(signals);
            }
        }

        // The cheap readout: a few tokens, scored by probability.
        if self.out_of_time(deadline) {
            return Ok(signals);
        }
        signals.readout = self.readout(driver, judged, request, branch).await;
        if self.settled(caps, &signals) {
            return Ok(signals);
        }

        // The deliberating readout: the same question, reasoned before answering.
        if self.out_of_time(deadline) {
            return Ok(signals);
        }
        signals.deliberation = match self
            .ask(
                driver,
                self.config.judge_target().clone(),
                Question::Evidence,
                judged,
                rungs::DELIBERATION_MAX_OUTPUT_TOKENS,
                request,
            )
            .await
        {
            // A deliberating verifier answers yes or no, not a probability, so
            // its verdict is mapped onto the bar it is judged against.
            Tri::Yes => Some(1.0),
            Tri::No => Some(0.0),
            Tri::Unknown => None,
        };
        if self.settled(caps, &signals) {
            return Ok(signals);
        }

        // Cloud confirmation, only where a branch can use it and only if the
        // operator configured a tier to ask.
        let Some(cloud_judge) = self.config.targets.cloud_judge.clone() else {
            return Ok(signals);
        };
        if self.out_of_time(deadline) {
            return Ok(signals);
        }
        match branch {
            Branch::Answer => {
                signals.answer_verifier = Some(
                    self.ask(
                        driver,
                        cloud_judge.clone(),
                        Question::Answer,
                        judged,
                        rungs::DELIBERATION_MAX_OUTPUT_TOKENS,
                        request,
                    )
                    .await,
                );
                if !self.settled(caps, &signals) && !self.out_of_time(deadline) {
                    signals.evidence_verifier = Some(
                        self.ask(
                            driver,
                            cloud_judge,
                            Question::Evidence,
                            judged,
                            rungs::DELIBERATION_MAX_OUTPUT_TOKENS,
                            request,
                        )
                        .await,
                    );
                }
            }
            Branch::CodingNoChecks | Branch::Chat => {
                signals.cloud_judge = Some(
                    self.ask(
                        driver,
                        cloud_judge.clone(),
                        Question::Evidence,
                        judged,
                        rungs::DELIBERATION_MAX_OUTPUT_TOKENS,
                        request,
                    )
                    .await,
                );
                // The second confirmation is only ever consulted after the
                // first affirms, so a refutation costs one call, not two.
                if signals.cloud_judge == Some(Tri::Yes) && !self.out_of_time(deadline) {
                    signals.evidence_confirm = Some(
                        self.ask(
                            driver,
                            cloud_judge,
                            Question::Answer,
                            judged,
                            rungs::DELIBERATION_MAX_OUTPUT_TOKENS,
                            request,
                        )
                        .await,
                    );
                }
            }
            _ => {}
        }
        Ok(signals)
    }

    /// Runs the probability-scored readout, or `None` if it cannot be scored.
    async fn readout(
        &self,
        driver: &Driver,
        judged: &str,
        request: &Request,
        branch: Branch,
    ) -> Option<f64> {
        // Conversation was measured to produce no usable readout at the
        // confident bar, so the branch only pays for one when its dial can use it.
        if branch == Branch::Chat && self.config.policy.offload_dial.chat.is_none() {
            return None;
        }
        let mut call = rungs::build_request(
            Question::Evidence,
            judged,
            readout::MAX_OUTPUT_TOKENS,
            request.metadata.clone(),
        );
        readout::request_logprobs(&mut call);
        let response = driver
            .call_model(call, vec![self.config.judge_target().clone()])
            .await
            .ok()?;
        readout::p_yes(BufferedResponse::new(response).await.ok()?.aggregate())
    }

    /// Puts one question to a verifier, folding every failure into indeterminate.
    ///
    /// A verifier is evidence, not a dependency: failing the caller's request
    /// because a verifier was unavailable would be worse than deciding without
    /// it, and deciding without it already escalates.
    async fn ask(
        &self,
        driver: &Driver,
        target: switchyard_protocol::ModelId,
        question: Question,
        judged: &str,
        max_output_tokens: u64,
        request: &Request,
    ) -> Tri {
        let call = rungs::build_request(
            question,
            judged,
            max_output_tokens,
            request.metadata.clone(),
        );
        match driver.call_model(call, vec![target]).await {
            Ok(response) => match BufferedResponse::new(response).await {
                Ok(buffered) => rungs::parse_verdict(buffered.aggregate()),
                Err(_) => Tri::Unknown,
            },
            Err(error) => {
                tracing::debug!(?question, error = %error, "vgr verifier unavailable");
                Tri::Unknown
            }
        }
    }

    /// Whether the evidence so far already licenses a commit.
    ///
    /// Asked between rungs so a settled decision stops spending. Only a commit
    /// short-circuits: a decision still resolving to escalate may yet be turned
    /// by a rung that has not run.
    fn settled(&self, caps: &Capabilities, signals: &Signals) -> bool {
        decide_from_signals(caps, signals, &self.config.policy, &self.readiness()).route
            == Route::Local
    }

    /// Whether the decision budget is spent.
    fn out_of_time(&self, deadline: Instant) -> bool {
        Instant::now() >= deadline
    }

    /// Readiness evidence is inseparable from the configured checker handle.
    fn readiness(&self) -> Readiness {
        Readiness {
            checker_validated: self.config.checker.is_some(),
        }
    }
}

/// A response buffered for inspection while retaining its original return shape.
struct BufferedResponse {
    aggregate: AggLlmResponse,
    response: Response,
}

impl BufferedResponse {
    /// Buffers a live stream into exact replayable events, or clones an aggregate
    /// for inspection while retaining the original value.
    async fn new(response: Response) -> std::result::Result<Self, LlmClientError> {
        let Response {
            llm_response,
            metadata,
        } = response;
        match llm_response {
            LlmResponse::Agg(aggregate) => Ok(Self {
                aggregate: aggregate.clone(),
                response: Response {
                    llm_response: LlmResponse::Agg(aggregate),
                    metadata,
                },
            }),
            LlmResponse::Stream(mut stream) => {
                let mut events = Vec::new();
                while let Some(event) = stream.next().await {
                    events.push(event?);
                }
                let aggregate = aggregate_events(&events).await?;
                Ok(Self {
                    aggregate,
                    response: Response {
                        llm_response: LlmResponse::Stream(replay_events(events)),
                        metadata,
                    },
                })
            }
        }
    }

    fn aggregate(&self) -> &AggLlmResponse {
        &self.aggregate
    }

    fn into_response(self) -> Response {
        self.response
    }
}

/// Aggregates a clone of buffered events through the protocol's checked API.
async fn aggregate_events(
    events: &[LlmResponseStreamEvent],
) -> std::result::Result<AggLlmResponse, LlmClientError> {
    LlmResponse::Stream(replay_events(events.to_vec()))
        .into_agg()
        .await
}

/// Replays buffered events exactly, including preservation and event boundaries.
fn replay_events(events: Vec<LlmResponseStreamEvent>) -> switchyard_protocol::LlmResponseStream {
    Box::pin(futures::stream::iter(
        events
            .into_iter()
            .map(Ok::<LlmResponseStreamEvent, LlmClientError>),
    ))
}

/// Whether a routed local-call failure is safe to escalate around.
fn fallback_eligible(error: &LibsyError) -> bool {
    matches!(
        error,
        LibsyError::ClientCall { source, .. } if client_error_fallback_eligible(source)
    )
}

/// Matches the host client's existing candidate-fallback policy.
fn client_error_fallback_eligible(error: &LlmClientError) -> bool {
    match error {
        LlmClientError::ContextWindowExceeded { .. }
        | LlmClientError::Transport { .. }
        | LlmClientError::Timeout { .. } => true,
        LlmClientError::UpstreamHttp { status, .. } => {
            matches!(
                *status,
                StatusCode::FORBIDDEN | StatusCode::REQUEST_TIMEOUT | StatusCode::TOO_MANY_REQUESTS
            ) || status.is_server_error()
        }
        _ => false,
    }
}

/// Carries the rejected attempt forward as reference for the capable tier.
///
/// The attempt is labelled as unverified reference rather than as an answer, so
/// the capable tier is not being told a wrong answer is right. Dropping the
/// preserved replay body is mandatory: the request is being mutated, so a
/// stored verbatim copy would be sent instead of the edit.
fn carry_attempt(request: &mut Request, attempt: &str, decision: &Decision) {
    // libsy does not receive per-target context-window capabilities. Reuse the
    // conservative attempt budget already applied to VGR verifier prompts,
    // reserving room for the clipping marker so the carried body stays bounded.
    const CLIPPING_MARKER_RESERVE: usize = 64;
    let attempt = super::text::clip_mid(
        attempt,
        super::render::ATTEMPT_BUDGET.saturating_sub(CLIPPING_MARKER_RESERVE),
        1.0 / 3.0,
    );
    let note = format!(
        "\n\nA previous attempt at this task was produced locally and was NOT verified \
         (verification regime: {:?}). Treat it as unverified reference material, not as a \
         correct answer. Produce your own answer.\n\n<previous_attempt>\n{attempt}\n</previous_attempt>",
        decision.branch
    );
    append_note(request, &note);
    drop_exact_replay(request);
}

#[cfg(test)]
mod tests;
