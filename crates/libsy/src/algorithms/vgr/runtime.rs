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
//!
//! # The decision budget binds every call
//!
//! Each rung is given only the time left in the budget, so a single hung
//! verifier cannot overrun it. A call that does not return in that time is
//! evidence that was not gathered, which is exactly how the rules already treat
//! a verifier that failed: it never commits.

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::StreamExt;
use http::StatusCode;
use switchyard_protocol::{
    AggLlmResponse, ContentBlock, LlmClientError, LlmResponse, LlmResponseStreamEvent, ModelId,
    Request, Response,
};

use super::config::{ServingMode, VgrConfig};
use super::decide::{Decision, Readiness, Route, agentic_can_gather, decide_from_signals};
use super::rules::{Signals, ToolErrorSignal, Tri};
use super::rungs::{self, Question};
use super::safety::{
    CircuitBreaker, indicates_endpoint_failure, indicates_transport_unavailability,
};
use super::telemetry::{Record, Stage, Unknown, count_redactions};
use super::{
    Branch, Capabilities, TaskType, ToolErrorCount, derive_capabilities, matching, readout, text,
};
use crate::algorithms::util::decisive;
use crate::algorithms::util::prompts::{append_note, drop_exact_replay};
use crate::algorithms::util::tool_signals::ToolSignals;
use crate::core::algorithm::Driver;
use crate::core::classifier::{Classification, Classifier};
use crate::core::state::{State, StateValue};
use crate::{LibsyError, Result};

const TURN_STREAK_KEY: &str = "vgr.turn_verification.streak";
const TURN_LATCHED_KEY: &str = "vgr.turn_verification.latched";

/// Which tier a call is billed to, for the record's token accounting.
#[derive(Clone, Copy, Eq, PartialEq)]
enum Billing {
    Local,
    Cloud,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum TurnAction {
    Stay,
    Escalate,
}

/// Runs one verification-gated turn.
pub(super) struct VgrClassifier {
    pub(super) config: VgrConfig,
    /// Health of the local endpoint, shared across every turn this route serves.
    pub(super) local_breaker: Arc<CircuitBreaker>,
    /// Health of the cloud endpoint, including terminal completion calls.
    pub(super) cloud_breaker: Arc<CircuitBreaker>,
}

#[async_trait]
impl Classifier<State> for VgrClassifier {
    async fn score(
        &self,
        state: &mut State,
        request: &mut Request,
        driver: &Driver,
    ) -> Result<(Classification, Option<Response>)> {
        let targets = &self.config.targets;
        let started = Instant::now();
        let mut record = Record::new();

        // Three ways a turn produces no attempt at all. Each escalates without
        // spending anything, which is the direction this router already fails in.
        let short_circuit = if self.config.mode == ServingMode::Off {
            // Off never spends anything: there is no decision to inform, so there
            // is no reason to produce an attempt that will not be served.
            Some("mode_off")
        } else if self
            .config
            .kill_switch
            .as_ref()
            .is_some_and(|switch| switch.is_engaged())
        {
            Some("kill_switch")
        } else if self.local_breaker.is_open() {
            Some("breaker_open")
        } else if turn_latched(state) {
            Some("turn_verification_latched")
        } else if !self.config.local_supports_images && request_has_image(request) {
            Some("local_image_unsupported")
        } else {
            None
        };
        if let Some(reason) = short_circuit {
            record.short_circuit = Some(reason);
            record.served_route = Some(Route::Cloud);
            record.elapsed = started.elapsed();
            record.annotate(request);
            record.emit();
            return Ok((decisive(&targets.cloud), None));
        }

        let deadline = match started.checked_add(self.config.deadline) {
            Some(deadline) => deadline,
            None => started,
        };

        // Keep the attempt call local-only so a cloud escalation is prepared as
        // the terminal completion call. Eligible local failures select cloud
        // below instead of surfacing or falling backward later.
        // The budget covers producing *and* buffering the attempt: an endpoint
        // that accepts the request and then stalls mid-stream would otherwise
        // hold the turn open past the deadline, which is exactly the failure the
        // capable tier is there to absorb.
        let Some(attempt_budget) = self.remaining(started) else {
            record.unknown(Stage::Attempt, Unknown::DeadlineExhausted);
            record.short_circuit = Some("local_timed_out");
            record.served_route = Some(Route::Cloud);
            record.elapsed = started.elapsed();
            record.annotate(request);
            record.emit();
            return Ok((decisive(&targets.cloud), None));
        };
        let attempt_started = Instant::now();
        let attempted = tokio::time::timeout(attempt_budget, async {
            let response = driver
                .call_model(request.clone(), vec![targets.local.clone()])
                .await?;
            BufferedResponse::new(response)
                .await
                .map_err(|source| LibsyError::client_call(targets.local.clone(), source))
        })
        .await;
        record.stage(Stage::Attempt, attempt_started.elapsed());

        let buffered = match attempted {
            // Success is recorded only once the attempt is fully buffered: a
            // call that returns a stream and then fails mid-transport is a
            // failed call, not a healthy one.
            Ok(Ok(attempt)) => {
                self.local_breaker.record_success();
                attempt
            }
            Ok(Err(error)) if fallback_eligible(&error) => {
                if indicates_endpoint_failure(&error) {
                    self.local_breaker.record_failure();
                }
                record.error(Stage::Attempt, &error);
                record.short_circuit = Some("local_unavailable");
                record.served_route = Some(Route::Cloud);
                record.elapsed = started.elapsed();
                record.annotate(request);
                record.emit();
                let response = complete_cloud(
                    driver,
                    request,
                    &targets.cloud,
                    &self.cloud_breaker,
                    Some(error),
                )
                .await?;
                return Ok((decisive(&targets.cloud), Some(response)));
            }
            Ok(Err(error)) => {
                if indicates_endpoint_failure(&error) {
                    self.local_breaker.record_failure();
                }
                record.error(Stage::Attempt, &error);
                return Err(error);
            }
            Err(_elapsed) => {
                // A tier that does not answer within the budget is unhealthy in
                // exactly the way the breaker exists to notice.
                self.local_breaker.record_failure();
                record.unknown(Stage::Attempt, Unknown::TimedOut);
                record.short_circuit = Some("local_timed_out");
                record.served_route = Some(Route::Cloud);
                record.elapsed = started.elapsed();
                record.annotate(request);
                record.emit();
                let local_error = LibsyError::client_call(
                    targets.local.clone(),
                    LlmClientError::Timeout {
                        source: std::io::Error::new(
                            std::io::ErrorKind::TimedOut,
                            "VGR local attempt exceeded the decision deadline",
                        )
                        .into(),
                    },
                );
                let response = complete_cloud(
                    driver,
                    request,
                    &targets.cloud,
                    &self.cloud_breaker,
                    Some(local_error),
                )
                .await?;
                return Ok((decisive(&targets.cloud), Some(response)));
            }
        };
        record.local_tokens += tokens(buffered.aggregate());

        if rungs::has_tool_call(buffered.aggregate()) && !tool_use_is_complete(buffered.aggregate())
        {
            record.short_circuit = Some("malformed_tool_call");
            record.served_route = Some(Route::Cloud);
            record.elapsed = started.elapsed();
            record.annotate(request);
            record.emit();
            return Ok((decisive(&targets.cloud), None));
        }

        if self.config.policy.turn_verification.enabled
            && rungs::has_tool_call(buffered.aggregate())
        {
            let action = self
                .verify_turn(
                    driver,
                    request,
                    buffered.aggregate(),
                    state,
                    &mut record,
                    started,
                )
                .await;
            let served = if action == TurnAction::Escalate
                || matches!(self.config.mode, ServingMode::Off | ServingMode::Shadow)
            {
                Route::Cloud
            } else {
                Route::Local
            };
            record.short_circuit = Some(if action == TurnAction::Escalate {
                "turn_verification_escalated"
            } else {
                "turn_verification_complete"
            });
            record.served_route = Some(served);
            record.elapsed = started.elapsed();
            record.annotate(request);
            record.emit();
            if served == Route::Local {
                return Ok((decisive(&targets.local), Some(buffered.into_response())));
            }
            return Ok((decisive(&targets.cloud), None));
        }

        let attempt = rungs::response_text(buffered.aggregate()).unwrap_or_default();
        let task_type = self.type_task(driver, request, &mut record, started).await;
        record.task_type = task_type;
        let caps = self.derive(request, &attempt, task_type, &mut record);
        let signals = self
            .gather(driver, &caps, request, &mut record, started, deadline)
            .await;
        let decision = self.decide(&caps, &signals);
        record.decision = Some(decision);
        record.tool_errors_source = caps.tool_errors.map(ToolErrorCount::source_label);

        let served = super::mode::serve_route(&self.config.mode, &decision);
        record.served_route = Some(served);
        record.elapsed = started.elapsed();
        record.annotate(request);
        record.emit();

        if served == Route::Local {
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

/// Serves a terminal cloud decision and updates the cloud endpoint breaker.
pub(super) async fn complete_cloud(
    driver: &Driver,
    request: &Request,
    target: &ModelId,
    breaker: &CircuitBreaker,
    local_error: Option<LibsyError>,
) -> Result<Response> {
    if breaker.is_open() {
        return Err(combine_unavailable(
            local_error,
            LibsyError::CircuitOpen {
                target: target.clone(),
            },
        ));
    }

    match driver
        .call_model(request.clone(), vec![target.clone()])
        .await
    {
        Ok(response) => {
            breaker.record_success();
            Ok(response)
        }
        Err(cloud_error) => {
            if indicates_transport_unavailability(&cloud_error) {
                breaker.record_failure();
            } else {
                // A completed request rejection proves the endpoint answered and
                // releases any admitted half-open trial without tripping it.
                breaker.record_success();
            }
            Err(combine_unavailable(local_error, cloud_error))
        }
    }
}

fn combine_unavailable(local_error: Option<LibsyError>, cloud_error: LibsyError) -> LibsyError {
    match local_error {
        Some(local_error)
            if indicates_transport_unavailability(&local_error)
                && indicates_transport_unavailability(&cloud_error) =>
        {
            LibsyError::VgrTiersUnavailable {
                local: Box::new(local_error),
                cloud: Box::new(cloud_error),
            }
        }
        _ => cloud_error,
    }
}

impl VgrClassifier {
    /// Judges one proposed tool-bearing assistant turn and advances its
    /// session-local confirmation streak.
    async fn verify_turn(
        &self,
        driver: &Driver,
        request: &Request,
        proposed: &AggLlmResponse,
        state: &mut State,
        record: &mut Record,
        started: Instant,
    ) -> TurnAction {
        let config = self.config.policy.turn_verification;
        let view = rungs::turn_view(request, proposed, &config);
        let mut call = rungs::build_turn_request(&view, request.metadata.clone());
        readout::request_logprobs(&mut call);

        let Some(budget) = self.remaining(started) else {
            record.unknown(Stage::TurnVerification, Unknown::DeadlineExhausted);
            return TurnAction::Stay;
        };
        let target = self.config.judge_target().clone();
        let call_started = Instant::now();
        let outcome = tokio::time::timeout(budget, async {
            let response = driver.call_model(call, vec![target.clone()]).await?;
            response
                .llm_response
                .into_agg()
                .await
                .map_err(|source| LibsyError::client_call(target.clone(), source))
        })
        .await;
        record.stage(Stage::TurnVerification, call_started.elapsed());

        let agg = match outcome {
            Ok(Ok(agg)) => agg,
            Ok(Err(error)) => {
                record.error(Stage::TurnVerification, &error);
                record.unknown(Stage::TurnVerification, Unknown::CallFailed);
                if target == self.config.targets.local && indicates_transport_unavailability(&error)
                {
                    self.local_breaker.record_failure();
                    latch_turns(state);
                    return TurnAction::Escalate;
                }
                return TurnAction::Stay;
            }
            Err(_elapsed) => {
                record.unknown(Stage::TurnVerification, Unknown::TimedOut);
                if target == self.config.targets.local {
                    self.local_breaker.record_failure();
                    latch_turns(state);
                    return TurnAction::Escalate;
                }
                return TurnAction::Stay;
            }
        };
        if target == self.config.targets.local {
            record.local_tokens += tokens(&agg);
        } else {
            record.cloud_tokens += tokens(&agg);
        }
        let Some(probability) = readout::p_yes(&agg) else {
            record.unknown(Stage::TurnVerification, Unknown::Unparsable);
            return TurnAction::Stay;
        };
        if probability < config.escalate_at {
            state.extra.remove(TURN_STREAK_KEY);
            return TurnAction::Stay;
        }

        let streak = turn_streak(state).saturating_add(1);
        state
            .extra
            .insert(TURN_STREAK_KEY.into(), StateValue::Count(streak));
        if streak >= config.confirmations {
            latch_turns(state);
            TurnAction::Escalate
        } else {
            TurnAction::Stay
        }
    }

    /// Builds the capabilities this turn is judged under.
    ///
    /// VGR treats normalized tool-result history as the execution record used
    /// by the agentic veto and bounded recovery rule.
    fn derive(
        &self,
        request: &Request,
        attempt: &str,
        task_type: Option<TaskType>,
        record: &mut Record,
    ) -> Capabilities {
        let tool_signals = ToolSignals::from_request(request, None);
        let mut caps = derive_capabilities(
            request,
            attempt,
            self.config.checker.is_some(),
            task_type,
            Some(ToolErrorCount::Host(tool_signals.error_count as i32)),
            Some(tool_signals.tool_results as i32),
            Some(tool_signals.tool_tail_clean),
        );
        // Operator declarations, which derivation never produces on its own.
        caps.structured_answer = self.config.structured_answer;
        record.unsupported_content = text::turns(request).1;
        record.redaction_events = caps.transcript.as_deref().map_or(0, count_redactions);
        caps
    }

    /// The decision the current evidence licenses.
    fn decide(&self, caps: &Capabilities, signals: &Signals) -> Decision {
        decide_from_signals(
            caps,
            signals,
            &self.config.policy,
            &Readiness {
                checker_validated: self.config.checker.is_some(),
            },
        )
    }

    /// Types the request so derivation can select a verification regime.
    ///
    /// Skipped when a checker is configured, because the checker's branch is
    /// selected by the operator's configuration and no type can change it.
    /// Abstains on any failure, which selects the default regime — more
    /// conservative than any type would have been, never weaker.
    async fn type_task(
        &self,
        driver: &Driver,
        request: &Request,
        record: &mut Record,
        started: Instant,
    ) -> Option<TaskType> {
        if !self.config.task_typing || self.config.checker.is_some() {
            return None;
        }
        let (turns, _) = text::turns(request);
        let task_text = text::user_task_text(&turns);
        if task_text.trim().is_empty() {
            return None;
        }
        let call = rungs::build_typing_request(&task_text, request.metadata.clone());
        let agg = self
            .call(
                driver,
                call,
                self.config.judge_target().clone(),
                Billing::Local,
                Stage::Typing,
                record,
                started,
            )
            .await?;
        let typed = rungs::parse_task_type(&agg);
        if typed.is_none() && !rungs::task_type_abstained(&agg) {
            record.unknown(Stage::Typing, Unknown::Unparsable);
        }
        typed
    }

    /// Gathers evidence cheapest-first, stopping once the decision is settled.
    async fn gather(
        &self,
        driver: &Driver,
        caps: &Capabilities,
        request: &Request,
        record: &mut Record,
        started: Instant,
        deadline: Instant,
    ) -> Signals {
        // The count derivation produced is the router's own reading of the
        // conversation, so it is the reported entry the veto weighs against
        // whatever the host could attest — which here is nothing.
        let mut signals = Signals {
            tool_errors: match caps.tool_errors {
                Some(count) => ToolErrorSignal::Count(count.count()),
                None => ToolErrorSignal::Absent,
            },
            tool_results: caps.tool_results,
            tool_tail_clean: caps.tool_tail_clean,
            ..Signals::default()
        };
        let branch = super::select_branch(caps);
        // Nothing to verify: no rung can change the outcome, so none run.
        if branch == Branch::Unknown {
            return signals;
        }
        let recovered_run = if branch == Branch::AgenticVerified {
            let (can_gather, recovered) = agentic_can_gather(caps, &signals, &self.config.policy);
            if !can_gather {
                return signals;
            }
            recovered
        } else {
            false
        };
        let Some(judged) = caps.transcript.as_deref() else {
            return signals;
        };

        // The checker is ground truth and supersedes every other rung on its
        // branch, so it runs alone and first.
        if branch == Branch::Checks {
            if let Some(checker) = &self.config.checker {
                record.checker_manifest = Some(checker.manifest_identity().to_owned());
                let task = match caps.task_text.as_deref() {
                    Some(task) => task,
                    None => return signals,
                };
                let checker_started = Instant::now();
                let verdict = checker
                    .check(task, caps.attempt.as_deref().unwrap_or_default(), deadline)
                    .await;
                record.stage(Stage::Checker, checker_started.elapsed());
                signals.tests_pass = Some(match verdict {
                    Some(true) => Tri::Yes,
                    Some(false) => Tri::No,
                    None => {
                        record.unknown(Stage::Checker, Unknown::CallFailed);
                        Tri::Unknown
                    }
                });
            }
            return signals;
        }

        // The cheap readout: a few tokens, scored by probability.
        signals.readout = self
            .readout(driver, judged, request, branch, record, started)
            .await;
        if self.settled(caps, &signals) {
            return signals;
        }
        // The shipped coding dial uses the judged family. A dial candidate
        // buys exactly one strict cloud confirmation immediately: yes commits,
        // while no, unknown, or an unavailable judge blocks that arm. Running
        // local deliberation first cannot change this arm and only adds cost.
        if branch == Branch::CodingNoChecks
            && signals
                .readout
                .zip(self.config.policy.offload_dial.coding)
                .is_some_and(|(readout, dial)| readout >= dial)
        {
            if let Some(cloud_judge) = self.config.targets.cloud_judge.clone() {
                signals.cloud_judge = Some(
                    self.ask(
                        driver,
                        cloud_judge,
                        Billing::Cloud,
                        Question::Evidence,
                        judged,
                        rungs::DELIBERATION_MAX_OUTPUT_TOKENS,
                        request,
                        Stage::CloudJudge,
                        record,
                        started,
                    )
                    .await,
                );
            }
            return signals;
        }

        // The deliberating readout: the same question, briefly reasoned before answering.
        signals.deliberation = match self
            .ask(
                driver,
                self.config.judge_target().clone(),
                Billing::Local,
                Question::Deliberation,
                judged,
                rungs::LOCAL_DELIBERATION_MAX_OUTPUT_TOKENS,
                request,
                Stage::Deliberation,
                record,
                started,
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
            return signals;
        }
        // The bounded recovery arm is intentionally local-only: it may spend a
        // readout and deliberation, but an error-bearing run must not consume a
        // capable-tier verification rung.
        if recovered_run {
            return signals;
        }

        // Typed agreement against an independently produced answer. The witness
        // never sees the attempt, so agreement between them is evidence rather
        // than an echo.
        //
        // It runs after the local rungs, not before: the witness is a cloud call
        // and the readout is four local tokens, so asking it first would spend
        // the expensive rung on answers the cheap one would have committed. And
        // only on a structured surface, because that is the only place the
        // answer rule reads agreement at all — anywhere else the call is waste.
        if branch == Branch::Answer
            && self.config.structured_answer
            && let (Some(answer), Some(task)) =
                (caps.final_answer.as_deref(), caps.task_text.as_deref())
        {
            signals.agreement = Some(
                self.agree(driver, task, answer, request, record, started)
                    .await,
            );
            if self.settled(caps, &signals) {
                return signals;
            }
        }

        // Cloud confirmation, only where a branch can use it and only if the
        // operator configured a tier to ask.
        let Some(cloud_judge) = self.config.targets.cloud_judge.clone() else {
            return signals;
        };
        match branch {
            Branch::Answer => {
                signals.answer_verifier = Some(
                    self.ask(
                        driver,
                        cloud_judge.clone(),
                        Billing::Cloud,
                        Question::Answer,
                        judged,
                        rungs::DELIBERATION_MAX_OUTPUT_TOKENS,
                        request,
                        Stage::CloudJudge,
                        record,
                        started,
                    )
                    .await,
                );
                if !self.settled(caps, &signals) {
                    signals.evidence_verifier = Some(
                        self.ask(
                            driver,
                            cloud_judge,
                            Billing::Cloud,
                            Question::Evidence,
                            judged,
                            rungs::DELIBERATION_MAX_OUTPUT_TOKENS,
                            request,
                            Stage::CloudJudge,
                            record,
                            started,
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
                        Billing::Cloud,
                        Question::Evidence,
                        judged,
                        rungs::DELIBERATION_MAX_OUTPUT_TOKENS,
                        request,
                        Stage::CloudJudge,
                        record,
                        started,
                    )
                    .await,
                );
                if self.settled(caps, &signals) {
                    return signals;
                }
                // The second confirmation is only ever consulted after the
                // first affirms, so a refutation costs one call, not two.
                if signals.cloud_judge == Some(Tri::Yes) {
                    signals.evidence_confirm = Some(
                        self.ask(
                            driver,
                            cloud_judge,
                            Billing::Cloud,
                            Question::Answer,
                            judged,
                            rungs::DELIBERATION_MAX_OUTPUT_TOKENS,
                            request,
                            Stage::CloudJudge,
                            record,
                            started,
                        )
                        .await,
                    );
                }
            }
            _ => {}
        }
        signals
    }

    /// Produces an independent answer and reports whether it agrees.
    ///
    /// The witness is asked of the capable tier, since an answer the local tier
    /// produced twice is one attempt restated, not corroboration.
    async fn agree(
        &self,
        driver: &Driver,
        task: &str,
        answer: &str,
        request: &Request,
        record: &mut Record,
        started: Instant,
    ) -> Tri {
        let target = self
            .config
            .targets
            .cloud_judge
            .clone()
            .unwrap_or_else(|| self.config.targets.cloud.clone());
        let call = rungs::build_witness_request(task, request.metadata.clone());
        let Some(agg) = self
            .call(
                driver,
                call,
                target,
                Billing::Cloud,
                Stage::Witness,
                record,
                started,
            )
            .await
        else {
            return Tri::Unknown;
        };
        let Some(witness) = rungs::parse_witness(&agg) else {
            record.unknown(Stage::Witness, Unknown::Unparsable);
            return Tri::Unknown;
        };
        matching::match_answer_verdict(&witness, answer)
    }

    /// Runs the probability-scored readout, or `None` if it cannot be scored.
    async fn readout(
        &self,
        driver: &Driver,
        judged: &str,
        request: &Request,
        branch: Branch,
        record: &mut Record,
        started: Instant,
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
        let agg = self
            .call(
                driver,
                call,
                self.config.judge_target().clone(),
                Billing::Local,
                Stage::Readout,
                record,
                started,
            )
            .await?;
        let scored = readout::p_yes(&agg);
        if scored.is_none() {
            record.unknown(Stage::Readout, Unknown::Unparsable);
        }
        scored
    }

    /// Puts one question to a verifier, folding every failure into indeterminate.
    ///
    /// A verifier is evidence, not a dependency: failing the caller's request
    /// because a verifier was unavailable would be worse than deciding without
    /// it, and deciding without it already escalates.
    #[allow(clippy::too_many_arguments)]
    async fn ask(
        &self,
        driver: &Driver,
        target: ModelId,
        billing: Billing,
        question: Question,
        judged: &str,
        max_output_tokens: u64,
        request: &Request,
        stage: Stage,
        record: &mut Record,
        started: Instant,
    ) -> Tri {
        let call = rungs::build_request(
            question,
            judged,
            max_output_tokens,
            request.metadata.clone(),
        );
        let Some(agg) = self
            .call(driver, call, target, billing, stage, record, started)
            .await
        else {
            return Tri::Unknown;
        };
        let verdict = rungs::parse_verdict(&agg);
        if verdict == Tri::Unknown {
            record.unknown(stage, Unknown::Unparsable);
        }
        verdict
    }

    /// Makes one model call within the remaining decision budget.
    ///
    /// The budget is enforced *around* the call rather than only before it, so a
    /// verifier that accepts the request and never answers cannot overrun the
    /// deadline. Every outcome other than a readable response is recorded and
    /// folded to `None`, which the rules read as evidence never gathered.
    #[allow(clippy::too_many_arguments)]
    async fn call(
        &self,
        driver: &Driver,
        call: Request,
        target: ModelId,
        billing: Billing,
        stage: Stage,
        record: &mut Record,
        started: Instant,
    ) -> Option<AggLlmResponse> {
        let Some(budget) = self.remaining(started) else {
            record.unknown(stage, Unknown::DeadlineExhausted);
            return None;
        };
        let call_started = Instant::now();
        let call_target = target.clone();
        let outcome = tokio::time::timeout(budget, async {
            let response = driver.call_model(call, vec![call_target.clone()]).await?;
            response
                .llm_response
                .into_agg()
                .await
                .map_err(|source| LibsyError::client_call(call_target, source))
        })
        .await;
        record.stage(stage, call_started.elapsed());

        match outcome {
            Ok(Ok(agg)) => {
                let billed = tokens(&agg);
                match billing {
                    Billing::Local => record.local_tokens += billed,
                    Billing::Cloud => record.cloud_tokens += billed,
                }
                Some(agg)
            }
            Ok(Err(error)) => {
                record.error(stage, &error);
                record.unknown(stage, Unknown::CallFailed);
                None
            }
            Err(_elapsed) => {
                record.unknown(stage, Unknown::TimedOut);
                None
            }
        }
    }

    /// Whether the evidence so far already licenses a commit.
    ///
    /// Asked between rungs so a settled decision stops spending. Only a commit
    /// short-circuits: a decision still resolving to escalate may yet be turned
    /// by a rung that has not run.
    fn settled(&self, caps: &Capabilities, signals: &Signals) -> bool {
        self.decide(caps, signals).route == Route::Local
    }

    /// The time left in the decision budget, or `None` once it is spent.
    fn remaining(&self, started: Instant) -> Option<Duration> {
        self.config.deadline.checked_sub(started.elapsed())
    }
}

/// Tokens a response reported, or zero when the provider reported none.
fn tokens(agg: &AggLlmResponse) -> u64 {
    agg.usage.total_tokens.unwrap_or(0)
}

fn tool_use_is_complete(response: &AggLlmResponse) -> bool {
    response.outputs.iter().all(|output| {
        output.stop_reason != Some(switchyard_protocol::StopReason::MaxTokens)
            && output.content.iter().all(|block| {
                !matches!(
                    block,
                    ContentBlock::ToolCall(call) if call.arguments.is_string()
                )
            })
    })
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
            upstream_headers,
        } = response;
        match llm_response {
            LlmResponse::Agg(aggregate) => Ok(Self {
                aggregate: aggregate.clone(),
                response: Response {
                    llm_response: LlmResponse::Agg(aggregate),
                    metadata,
                    upstream_headers,
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
                        upstream_headers,
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

fn turn_streak(state: &State) -> u32 {
    match state.extra.get(TURN_STREAK_KEY) {
        Some(StateValue::Count(streak)) => *streak,
        _ => 0,
    }
}

fn turn_latched(state: &State) -> bool {
    matches!(
        state.extra.get(TURN_LATCHED_KEY),
        Some(StateValue::Count(value)) if *value > 0
    )
}

fn latch_turns(state: &mut State) {
    state
        .extra
        .insert(TURN_LATCHED_KEY.into(), StateValue::Count(1));
}

/// Whether a request carries image content directly or inside a tool result.
fn request_has_image(request: &Request) -> bool {
    request
        .llm_request
        .instructions
        .iter()
        .any(|instruction| instruction.content.iter().any(block_has_image))
        || request
            .llm_request
            .messages
            .iter()
            .flat_map(|message| &message.content)
            .any(block_has_image)
}

fn block_has_image(block: &ContentBlock) -> bool {
    match block {
        ContentBlock::Image { .. } => true,
        ContentBlock::ToolResult(result) => result.content.iter().any(block_has_image),
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
