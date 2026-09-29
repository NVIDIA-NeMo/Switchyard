// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The classifier that runs one verification-gated turn.
//!
//! A proposed tool call is judged against the session's trajectory; a final
//! answer is typed, judged by rungs that run cheapest-first and stop once the
//! decision is settled, and then committed or escalated. Every call is bounded
//! by what is left of the turn's deadline, and a call that fails or times out
//! is evidence never gathered, which does not commit.

use std::time::{Duration, Instant};

use async_trait::async_trait;
use switchyard_protocol::{
    AggLlmResponse, ContentBlock, LlmClientError, ModelId, Request, Response, StopReason,
};

use super::config::{ServingMode, VgrConfig};
use super::decide::{self, AgenticRun, Route, Signals, Tri};
use super::rungs::{self, Question};
use super::safety::{CircuitBreaker, endpoint_failure, fallback_eligible};
use super::{Branch, Capabilities, TaskType, derive_capabilities, readout, text};
use crate::algorithms::util::buffered_response::buffer_response;
use crate::algorithms::util::decisive;
use crate::core::algorithm::Driver;
use crate::core::classifier::{Classification, Classifier};
use crate::core::state::{State, StateValue};
use crate::{LibsyError, Result};

const TURN_STREAK_KEY: &str = "vgr.turn_verification.streak";
const TURN_LATCHED_KEY: &str = "vgr.turn_verification.latched";
/// Consecutive escalation votes that latch a session to the capable tier.
const TURN_CONFIRMATIONS: u32 = 2;
const TURN_ESCALATE_AT: f64 = 0.5;

type Scored = (Classification, Option<Response>);

pub(super) struct VgrClassifier {
    pub(super) config: VgrConfig,
    pub(super) breaker: CircuitBreaker,
}

#[async_trait]
impl Classifier<State> for VgrClassifier {
    async fn score(
        &self,
        state: &mut State,
        request: &mut Request,
        driver: &Driver,
    ) -> Result<Scored> {
        let started = Instant::now();
        let local = self.config.targets.local.clone();
        let short_circuit = if self.config.mode == ServingMode::Off {
            Some("mode_off")
        } else if self
            .config
            .kill_switch
            .as_ref()
            .is_some_and(|switch| switch.is_engaged())
        {
            Some("kill_switch")
        } else if self.breaker.is_open() {
            Some("breaker_open")
        } else if turn_latched(state) {
            Some("turn_verification_latched")
        } else if !self.config.local_supports_images && request_has_image(request) {
            Some("local_image_unsupported")
        } else {
            None
        };
        if let Some(reason) = short_circuit {
            return Ok(self.escalate(reason, None));
        }

        let Some(budget) = self.remaining(started) else {
            return Ok(self.escalate("local_timed_out", None));
        };
        let attempted = tokio::time::timeout(budget, async {
            let response = driver
                .call_model(request.clone(), vec![local.clone()])
                .await?;
            buffer_response(local.as_str(), response).await
        })
        .await;
        let buffered = match attempted {
            Ok(Ok(buffered)) => {
                self.breaker.success();
                buffered
            }
            Ok(Err(error)) => {
                if endpoint_failure(&error) {
                    self.breaker.failure();
                }
                if !fallback_eligible(&error) {
                    return Err(error);
                }
                return Ok(self.escalate("local_unavailable", None));
            }
            Err(_) => {
                self.breaker.failure();
                return Ok(self.escalate("local_timed_out", None));
            }
        };

        if rungs::has_tool_call(&buffered.agg) {
            if !tool_use_is_complete(&buffered.agg) {
                return Ok(self.escalate("malformed_tool_call", None));
            }
            if self
                .verify_turn(driver, request, &buffered.agg, state, started)
                .await
            {
                return Ok(self.escalate("turn_verification_escalated", None));
            }
            if self.config.mode == ServingMode::Shadow {
                return Ok(self.escalate("turn_verification_complete", None));
            }
            log("turn_verification_complete", None, Route::Local);
            return Ok((decisive(&local), Some(buffered.into_response())));
        }

        let attempt = rungs::response_text(&buffered.agg).unwrap_or_default();
        let task_type = self.type_task(driver, request, started).await;
        let caps = derive_capabilities(request, &attempt, task_type);
        let signals = self.gather(driver, &caps, request, started).await;
        let route = decide::decide(&caps, &signals, self.config.confirmed_min_clean_tail());
        if route == Route::Cloud || self.config.mode == ServingMode::Shadow {
            return Ok(self.escalate("decided", Some(caps.branch)));
        }
        log("decided", Some(caps.branch), Route::Local);
        Ok((decisive(&local), Some(buffered.into_response())))
    }
}

impl VgrClassifier {
    fn escalate(&self, reason: &'static str, branch: Option<Branch>) -> Scored {
        log(reason, branch, Route::Cloud);
        (decisive(&self.config.targets.cloud), None)
    }

    /// Judges one proposed tool call; `true` escalates the rest of the session.
    async fn verify_turn(
        &self,
        driver: &Driver,
        request: &Request,
        proposed: &AggLlmResponse,
        state: &mut State,
        started: Instant,
    ) -> bool {
        let mut call = rungs::build_request(
            Question::TurnTrajectory,
            &rungs::turn_view(request, proposed),
            readout::MAX_OUTPUT_TOKENS,
            request.metadata.clone(),
        );
        readout::request_logprobs(&mut call);
        let judge = self.config.judge();
        let probability = match self.call(driver, call, judge, started).await {
            None => return false,
            Some(Ok(agg)) => readout::p_yes(&agg),
            // Verification fails open, unless the local endpoint itself is down.
            Some(Err(error)) => {
                if *judge == self.config.targets.local && endpoint_failure(&error) {
                    self.breaker.failure();
                    latch_turns(state);
                    return true;
                }
                return false;
            }
        };
        let Some(probability) = probability else {
            return false;
        };
        if probability < TURN_ESCALATE_AT {
            state.extra.remove(TURN_STREAK_KEY);
            return false;
        }
        let streak = match state.extra.get(TURN_STREAK_KEY) {
            Some(StateValue::Count(streak)) => streak.saturating_add(1),
            _ => 1,
        };
        state
            .extra
            .insert(TURN_STREAK_KEY.into(), StateValue::Count(streak));
        if streak >= TURN_CONFIRMATIONS {
            latch_turns(state);
        }
        streak >= TURN_CONFIRMATIONS
    }

    /// Types the request; abstains, selecting the default regime, on any failure.
    async fn type_task(
        &self,
        driver: &Driver,
        request: &Request,
        started: Instant,
    ) -> Option<TaskType> {
        if !self.config.task_typing {
            return None;
        }
        let task = text::user_task_text(&text::turns(request).0);
        if task.trim().is_empty() {
            return None;
        }
        let call = rungs::build_typing_request(&task, request.metadata.clone());
        let agg = self
            .call(driver, call, self.config.judge(), started)
            .await?
            .ok()?;
        rungs::parse_task_type(&agg)
    }

    /// Gathers evidence cheapest-first, stopping once the decision commits.
    async fn gather(
        &self,
        driver: &Driver,
        caps: &Capabilities,
        request: &Request,
        started: Instant,
    ) -> Signals {
        let mut signals = Signals::default();
        let branch = caps.branch;
        let confirmed = self.config.confirmed_min_clean_tail();
        let run = (branch == Branch::Agentic).then(|| decide::agentic_run(caps, confirmed));
        let (Some(judged), false) = (
            caps.transcript.as_deref(),
            matches!(branch, Branch::Coding | Branch::Unknown) || run == Some(AgenticRun::Vetoed),
        ) else {
            return signals;
        };
        let settled = |signals: &Signals| decide::decide(caps, signals, confirmed) == Route::Local;
        let cloud_confirms = run == Some(AgenticRun::ConfirmedRecovery);
        let local = self.config.judge();
        let ask = |target, question, max_output_tokens| {
            self.ask(
                driver,
                target,
                question,
                judged,
                max_output_tokens,
                request,
                started,
            )
        };

        signals.readout = self.readout(driver, judged, request, started).await;
        if settled(&signals) {
            return signals;
        }
        if !(cloud_confirms && decide::locally_verified(branch, &signals)) {
            signals.deliberation = match ask(
                local,
                Question::Deliberation,
                rungs::LOCAL_DELIBERATION_MAX_OUTPUT_TOKENS,
            )
            .await
            {
                Tri::Yes => Some(1.0),
                Tri::No => Some(0.0),
                Tri::Unknown => None,
            };
            if settled(&signals) {
                return signals;
            }
        }
        let Some(cloud) = self.config.targets.cloud_judge.as_ref() else {
            return signals;
        };
        let cloud_ask = |question| ask(cloud, question, rungs::DELIBERATION_MAX_OUTPUT_TOKENS);
        if cloud_confirms {
            if decide::locally_verified(branch, &signals) {
                signals.cloud_judge = Some(cloud_ask(Question::Evidence).await);
            }
            return signals;
        }
        match branch {
            Branch::Answer => {
                signals.answer_verifier = Some(cloud_ask(Question::Answer).await);
                if !settled(&signals) {
                    signals.evidence_verifier = Some(cloud_ask(Question::Evidence).await);
                }
            }
            Branch::Chat => {
                signals.cloud_judge = Some(cloud_ask(Question::Evidence).await);
                // Asked only once the first affirms, so a refutation costs one call.
                if signals.cloud_judge == Some(Tri::Yes) {
                    signals.evidence_confirm = Some(cloud_ask(Question::Answer).await);
                }
            }
            _ => {}
        }
        signals
    }

    async fn readout(
        &self,
        driver: &Driver,
        judged: &str,
        request: &Request,
        started: Instant,
    ) -> Option<f64> {
        let mut call = rungs::build_request(
            Question::Evidence,
            judged,
            readout::MAX_OUTPUT_TOKENS,
            request.metadata.clone(),
        );
        readout::request_logprobs(&mut call);
        let agg = self
            .call(driver, call, self.config.judge(), started)
            .await?
            .ok()?;
        readout::p_yes(&agg)
    }

    #[allow(clippy::too_many_arguments)]
    async fn ask(
        &self,
        driver: &Driver,
        target: &ModelId,
        question: Question,
        judged: &str,
        max_output_tokens: u64,
        request: &Request,
        started: Instant,
    ) -> Tri {
        let call = rungs::build_request(
            question,
            judged,
            max_output_tokens,
            request.metadata.clone(),
        );
        match self.call(driver, call, target, started).await {
            Some(Ok(agg)) => rungs::parse_verdict(&agg),
            _ => Tri::Unknown,
        }
    }

    /// One verifier call inside the remaining budget; `None` once it is spent.
    async fn call(
        &self,
        driver: &Driver,
        call: Request,
        target: &ModelId,
        started: Instant,
    ) -> Option<Result<AggLlmResponse>> {
        let budget = self.remaining(started)?;
        let outcome = tokio::time::timeout(budget, async {
            let response = driver.call_model(call, vec![target.clone()]).await?;
            response
                .llm_response
                .into_agg()
                .await
                .map_err(|source| LibsyError::client_call(target.clone(), source))
        })
        .await;
        Some(outcome.unwrap_or_else(|_| {
            Err(LibsyError::client_call(
                target.clone(),
                LlmClientError::Timeout {
                    source: Box::new(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "VGR verifier exceeded the decision deadline",
                    )),
                },
            ))
        }))
    }

    fn remaining(&self, started: Instant) -> Option<Duration> {
        self.config.deadline.checked_sub(started.elapsed())
    }
}

fn log(reason: &'static str, branch: Option<Branch>, served: Route) {
    tracing::info!(target: "libsy", reason, branch = ?branch, served = ?served, "vgr decision");
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

fn tool_use_is_complete(response: &AggLlmResponse) -> bool {
    response.outputs.iter().all(|output| {
        output.stop_reason != Some(StopReason::MaxTokens)
            && output.content.iter().all(|block| {
                !matches!(block, ContentBlock::ToolCall(call) if call.arguments.is_string())
            })
    })
}

/// Whether a request carries image content directly or inside a tool result.
fn request_has_image(request: &Request) -> bool {
    fn has_image(block: &ContentBlock) -> bool {
        match block {
            ContentBlock::Image { .. } => true,
            ContentBlock::ToolResult(result) => result.content.iter().any(has_image),
            _ => false,
        }
    }
    let llm = &request.llm_request;
    llm.instructions
        .iter()
        .flat_map(|instruction| &instruction.content)
        .chain(llm.messages.iter().flat_map(|message| &message.content))
        .any(has_image)
}

#[cfg(test)]
mod tests;
