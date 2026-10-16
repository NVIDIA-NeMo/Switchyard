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
use switchyard_protocol::{AggLlmResponse, ContentBlock, ModelId, Request, Response, StopReason};

use super::config::{ServingMode, VgrConfig};
use super::decide::{self, AgenticRun, Route, Signals, Tri};
use super::rungs::{self, Question};
use super::safety::{CircuitBreaker, endpoint_failure, fallback_eligible};
use super::{Branch, Capabilities, TaskType, derive_capabilities, readout, text};
use crate::algorithms::util::buffered_response::buffer_response;
use crate::algorithms::util::decisive;
use crate::core::algorithm::Driver;
use crate::core::classifier::{Classification, Classifier};
use crate::{LibsyError, Result};

#[derive(Default)]
pub(super) struct VgrState {
    pub(super) turn_streak: u32,
    pub(super) turn_latched: bool,
}

struct RequestContext<'a> {
    driver: &'a Driver,
    request: &'a Request,
    started: Instant,
}

/// Consecutive escalation votes that latch a session to the capable tier.
const TURN_CONFIRMATIONS: u32 = 2;
const TURN_ESCALATE_AT: f64 = 0.5;

type Scored = (Classification, Option<Response>);

pub(super) struct VgrClassifier {
    pub(super) config: VgrConfig,
    pub(super) breaker: CircuitBreaker,
}

#[async_trait]
impl Classifier<VgrState> for VgrClassifier {
    async fn score(
        &self,
        state: &mut VgrState,
        request: &mut Request,
        driver: &Driver,
    ) -> Result<Scored> {
        let started = Instant::now();
        let local = &self.config.targets.local;
        let short_circuit = if self.config.mode == ServingMode::Off {
            Some("mode_off")
        } else if self
            .config
            .kill_switch
            .as_ref()
            .is_some_and(|switch| switch.is_engaged())
        {
            Some("kill_switch")
        } else if state.turn_latched {
            Some("turn_verification_latched")
        } else if !self.config.local_supports_images && request_has_image(request) {
            Some("local_image_unsupported")
        } else {
            None
        };
        if let Some(reason) = short_circuit {
            return Ok(self.escalate(driver, reason, None));
        }

        let Some(budget) = self.remaining(started) else {
            return Ok(self.escalate(driver, "local_timed_out", None));
        };
        let Some(trial) = self.breaker.attempt() else {
            return Ok(self.escalate(driver, "breaker_open", None));
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
                trial.success();
                buffered
            }
            Ok(Err(error)) => {
                if endpoint_failure(&error) {
                    trial.failure();
                }
                if !fallback_eligible(&error) {
                    return Err(error);
                }
                return Ok(self.escalate(driver, "local_unavailable", None));
            }
            Err(_) => {
                trial.failure();
                return Ok(self.escalate(driver, "local_timed_out", None));
            }
        };

        let context = RequestContext {
            driver,
            request,
            started,
        };
        if rungs::has_tool_call(&buffered.agg) {
            if !tool_use_is_complete(&buffered.agg) {
                return Ok(self.escalate(driver, "malformed_tool_call", None));
            }
            if self.verify_turn(&context, &buffered.agg, state).await {
                return Ok(self.escalate(driver, "turn_verification_escalated", None));
            }
            if self.config.mode == ServingMode::Shadow {
                return Ok(self.escalate(driver, "turn_verification_complete", None));
            }
            log(driver, "turn_verification_complete", None, Route::Local);
            return Ok((decisive(local), Some(buffered.into_response())));
        }

        let attempt = rungs::response_text(&buffered.agg).unwrap_or_default();
        let task_type = self.type_task(&context).await;
        let caps = derive_capabilities(request, &attempt, task_type);
        let signals = self.gather(&context, &caps).await;
        let route = decide::decide(&caps, &signals, self.config.confirmed_min_clean_tail());
        if route == Route::Cloud || self.config.mode == ServingMode::Shadow {
            return Ok(self.escalate(driver, "decided", Some(caps.branch)));
        }
        log(driver, "decided", Some(caps.branch), Route::Local);
        Ok((decisive(local), Some(buffered.into_response())))
    }
}

impl VgrClassifier {
    fn escalate(&self, driver: &Driver, reason: &'static str, branch: Option<Branch>) -> Scored {
        log(driver, reason, branch, Route::Cloud);
        (decisive(&self.config.targets.cloud), None)
    }

    /// Judges one proposed tool call; `true` escalates the rest of the session.
    async fn verify_turn(
        &self,
        context: &RequestContext<'_>,
        proposed: &AggLlmResponse,
        state: &mut VgrState,
    ) -> bool {
        let mut call = rungs::build_request(
            Question::TurnTrajectory,
            &rungs::turn_view(context.request, proposed),
            readout::MAX_OUTPUT_TOKENS,
            context.request.metadata.clone(),
        );
        readout::request_logprobs(&mut call);
        let judge = self.config.judge();
        let probability = match self.call(context, call, judge).await {
            None => return false,
            Some(Ok(agg)) => readout::p_yes(&agg),
            // Verification fails open, unless the local endpoint itself is down.
            Some(Err(error)) => {
                if *judge == self.config.targets.local && endpoint_failure(&error) {
                    self.breaker.failure();
                    state.turn_latched = true;
                    return true;
                }
                return false;
            }
        };
        let Some(probability) = probability else {
            return false;
        };
        if probability < TURN_ESCALATE_AT {
            state.turn_streak = 0;
            return false;
        }
        state.turn_streak = state.turn_streak.saturating_add(1);
        let streak = state.turn_streak;
        if streak >= TURN_CONFIRMATIONS {
            state.turn_latched = true;
        }
        streak >= TURN_CONFIRMATIONS
    }

    /// Types the request; abstains, selecting the default regime, on any failure.
    async fn type_task(&self, context: &RequestContext<'_>) -> Option<TaskType> {
        if !self.config.task_typing {
            return None;
        }
        let (turns, _) = text::turns(context.request);
        let task = text::user_task_text(&turns);
        if task.trim().is_empty() {
            return None;
        }
        let call = rungs::build_typing_request(task, context.request.metadata.clone());
        let agg = self.call(context, call, self.config.judge()).await?.ok()?;
        rungs::parse_task_type(&agg)
    }

    /// Gathers evidence cheapest-first, stopping once the decision commits.
    async fn gather(&self, context: &RequestContext<'_>, caps: &Capabilities) -> Signals {
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
            self.ask(context, target, question, judged, max_output_tokens)
        };

        signals.readout = self.readout(context, judged).await;
        if settled(&signals) {
            return signals;
        }
        if !(cloud_confirms && decide::locally_verified(branch, &signals)) {
            signals.deliberation = Some(
                ask(
                    local,
                    Question::Deliberation,
                    rungs::LOCAL_DELIBERATION_MAX_OUTPUT_TOKENS,
                )
                .await,
            );
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

    async fn readout(&self, context: &RequestContext<'_>, judged: &str) -> Option<f64> {
        let mut call = rungs::build_request(
            Question::Evidence,
            judged,
            readout::MAX_OUTPUT_TOKENS,
            context.request.metadata.clone(),
        );
        readout::request_logprobs(&mut call);
        let agg = self.call(context, call, self.config.judge()).await?.ok()?;
        readout::p_yes(&agg)
    }

    async fn ask(
        &self,
        context: &RequestContext<'_>,
        target: &ModelId,
        question: Question,
        judged: &str,
        max_output_tokens: u64,
    ) -> Tri {
        let call = rungs::build_request(
            question,
            judged,
            max_output_tokens,
            context.request.metadata.clone(),
        );
        match self.call(context, call, target).await {
            Some(Ok(agg)) => rungs::parse_verdict(&agg),
            _ => Tri::Unknown,
        }
    }

    /// One verifier call inside the remaining budget; `None` once it is spent.
    async fn call(
        &self,
        context: &RequestContext<'_>,
        call: Request,
        target: &ModelId,
    ) -> Option<Result<AggLlmResponse>> {
        let budget = self.remaining(context.started)?;
        let outcome = tokio::time::timeout(budget, async {
            let response = context
                .driver
                .call_model(call, vec![target.clone()])
                .await?;
            response
                .llm_response
                .into_agg()
                .await
                .map_err(|source| LibsyError::client_call(target.clone(), source))
        })
        .await;
        outcome.ok()
    }

    fn remaining(&self, started: Instant) -> Option<Duration> {
        self.config.deadline.checked_sub(started.elapsed())
    }
}

fn log(driver: &Driver, reason: &'static str, branch: Option<Branch>, served: Route) {
    driver.set_evidence(serde_json::json!({
        "source": "vgr",
        "reason_code": reason,
        "verdict": if served == Route::Local { "local" } else { "cloud" },
        "branch": branch.map(|branch| format!("{branch:?}")),
    }));
    tracing::info!(target: "libsy", reason, branch = ?branch, served = ?served, "vgr decision");
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
