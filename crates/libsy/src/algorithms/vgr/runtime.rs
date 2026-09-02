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
use switchyard_protocol::{
    AggLlmResponse, LlmClientError, LlmResponse, Request, Response,
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

        // The local attempt, with a single candidate so its failures surface
        // here rather than being silently served by a fallback.
        let local_response = match driver
            .call_model(request.clone(), vec![targets.local.clone()])
            .await
        {
            Ok(response) => response,
            // A local tier that cannot take the request has produced no
            // attempt to verify, so there is nothing to gate.
            Err(LibsyError::ClientCall {
                source: LlmClientError::ContextWindowExceeded { .. },
                ..
            }) => return Ok((decisive(&targets.cloud), None)),
            Err(error) => return Err(error),
        };
        let agg = match local_response.llm_response.into_agg().await {
            Ok(agg) => agg,
            Err(LlmClientError::Transport { .. }) => {
                return Ok((decisive(&targets.cloud), None));
            }
            Err(source) => return Err(LibsyError::client_call(targets.local.clone(), source)),
        };

        let attempt = rungs::response_text(&agg).unwrap_or_default();
        let caps = self.derive(request, &attempt);
        let signals = self
            .gather(driver, &caps, &attempt, request, started)
            .await?;
        let decision = decide_from_signals(
            &caps,
            &signals,
            &self.config.policy,
            &Readiness {
                checker_validated: self.config.checker_validated,
            },
        );

        tracing::info!(
            branch = ?decision.branch,
            route = ?decision.route,
            effective = ?decision.effective_route,
            gate = ?decision.readiness_gate,
            elapsed_ms = started.elapsed().as_millis() as u64,
            "vgr decision"
        );

        if super::mode::serve_route(&self.config.mode, &decision) == Route::Local {
            // Re-materialize the buffered attempt so the turn is not paid for
            // twice, matching the shape the request arrived in.
            let served = Response {
                llm_response: if request.llm_request.stream {
                    LlmResponse::Stream(agg.into_stream())
                } else {
                    LlmResponse::Agg(agg)
                },
                metadata: local_response.metadata,
            };
            return Ok((decisive(&targets.local), Some(served)));
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
        started: Instant,
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
                let task = caps.task_text.clone().unwrap_or_default();
                signals.tests_pass = Some(match checker.check(&task, attempt) {
                    Some(true) => Tri::Yes,
                    Some(false) => Tri::No,
                    None => Tri::Unknown,
                });
            }
            return Ok(signals);
        }

        // Typed agreement is free: it compares text this router already holds.
        if branch == Branch::Answer
            && let (Some(answer), Some(task)) = (caps.final_answer.as_deref(), caps.task_text.as_deref())
        {
            signals.agreement = Some(matching::match_answer_verdict(task, answer));
            if self.settled(caps, &signals) {
                return Ok(signals);
            }
        }

        // The cheap readout: a few tokens, scored by probability.
        if self.out_of_time(started) {
            return Ok(signals);
        }
        signals.readout = self.readout(driver, judged, request, branch).await;
        if self.settled(caps, &signals) {
            return Ok(signals);
        }

        // The deliberating readout: the same question, reasoned before answering.
        if self.out_of_time(started) {
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
        if self.out_of_time(started) {
            return Ok(signals);
        }
        match branch {
            Branch::Answer => {
                signals.answer_verifier = Some(
                    self.ask(driver, cloud_judge.clone(), Question::Answer, judged,
                             rungs::DELIBERATION_MAX_OUTPUT_TOKENS, request).await,
                );
                if !self.settled(caps, &signals) && !self.out_of_time(started) {
                    signals.evidence_verifier = Some(
                        self.ask(driver, cloud_judge, Question::Evidence, judged,
                                 rungs::DELIBERATION_MAX_OUTPUT_TOKENS, request).await,
                    );
                }
            }
            Branch::CodingNoChecks | Branch::Chat => {
                signals.cloud_judge = Some(
                    self.ask(driver, cloud_judge.clone(), Question::Evidence, judged,
                             rungs::DELIBERATION_MAX_OUTPUT_TOKENS, request).await,
                );
                // The second confirmation is only ever consulted after the
                // first affirms, so a refutation costs one call, not two.
                if signals.cloud_judge == Some(Tri::Yes) && !self.out_of_time(started) {
                    signals.evidence_confirm = Some(
                        self.ask(driver, cloud_judge, Question::Answer, judged,
                                 rungs::DELIBERATION_MAX_OUTPUT_TOKENS, request).await,
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
        readout::p_yes(&buffer(response).await?)
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
        let call = rungs::build_request(question, judged, max_output_tokens, request.metadata.clone());
        match driver.call_model(call, vec![target]).await {
            Ok(response) => match buffer(response).await {
                Some(agg) => rungs::parse_verdict(&agg),
                None => Tri::Unknown,
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
        decide_from_signals(
            caps,
            signals,
            &self.config.policy,
            &Readiness {
                checker_validated: self.config.checker_validated,
            },
        )
        .route
            == Route::Local
    }

    /// Whether the decision budget is spent.
    fn out_of_time(&self, started: Instant) -> bool {
        started.elapsed() >= self.config.deadline
    }
}

/// Buffers a response, discarding it if it cannot be read.
async fn buffer(response: Response) -> Option<AggLlmResponse> {
    response.llm_response.into_agg().await.ok()
}

/// Carries the rejected attempt forward as reference for the capable tier.
///
/// The attempt is labelled as unverified reference rather than as an answer, so
/// the capable tier is not being told a wrong answer is right. Dropping the
/// preserved replay body is mandatory: the request is being mutated, so a
/// stored verbatim copy would be sent instead of the edit.
fn carry_attempt(request: &mut Request, attempt: &str, decision: &Decision) {
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
