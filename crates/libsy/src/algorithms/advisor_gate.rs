// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Executor gated by a once-per-session advisor review.
//!
//! The executor answers every client-visible turn. Turns with tool calls pass
//! through unreviewed; the first *terminal* turn — no tool calls (or a text
//! match under the `pattern` trigger) — is buffered and shown to a stronger
//! advisor model together with the full transcript. `APPROVE` releases the
//! buffered turn unchanged; `REDO` appends the discarded turn's text and the
//! advisor's plan as feedback, then re-invokes the executor so it keeps
//! working. Each budget scope (one benchmark evaluation, one session, or the
//! whole instance — see [`budget::budget_scope`]) is reviewed at most
//! `max_reviews` times; afterwards every call is a pure passthrough.
//!
//! This design is a near-superset of solo executor behavior: identical until
//! the executor first claims to be done, plus one quality gate that catches
//! premature convergence. Front-loading advice was measured to suppress the
//! executor's own test-and-iterate loop, so no advice is injected up front.
//!
//! Failure posture: executor errors always propagate (including
//! `ContextWindowExceeded`, which hosts map to a client-visible 400 so agent
//! harnesses can compact). Advisor errors honor `fail_open` — the buffered
//! turn passes through as an implicit APPROVE — refund the consumed review,
//! and count toward a per-scope failure cap that stops consulting a down
//! advisor entirely.
//!
//! Structure: [`AdvisorGate`] is a thin orchestrator over the gate's
//! components. The [`signals`] processor folds each event's facts into the
//! per-turn state — conversation counts on [`Event::Request`], the generated
//! turn's shape on [`Event::ModelResponse`] — the [`trigger`] classifier
//! reads them after the executor call, and the [`budget`] ledger holds the
//! only mutable state: the reserve/refund review budget and the stall latch.

use std::sync::Arc;
use std::time::Instant;

use switchyard_protocol::{
    ContentBlock, InstructionBlock, LlmRequest, Message, ModelId, OutputParams, Request, Role,
    SamplingParams,
};

use crate::core::algorithm::{Algorithm, Driver, RoutingOutcome};
use crate::core::processor::{Event, Processor};
use crate::{LibsyError, Result};

mod budget;
mod signals;
mod telemetry;
#[cfg(test)]
mod tests;
mod transcript;
mod trigger;
mod turn;

use budget::{ReviewBudget, ScopeKey, budget_scope, stall_key};
use signals::{GateSignalProcessor, GateSignals};
use telemetry::{
    ReviewAudit, emit_discarded_audit, emit_review_audit, record_consult_failure, record_discarded,
    record_review,
};
use transcript::{VERDICT_PATTERN, Verdict, advisor_reply_text, parse_verdict, review_transcript};
use trigger::TriggerClassifier;
#[cfg(test)]
use turn::has_tool_use;
use turn::{GatedTurn, buffer_turn, reasoning_text, visible_text};

/// APPROVE/REDO reviewer contract sent as the advisor's system prompt.
pub const REVIEWER_SYSTEM_PROMPT: &str =
    include_str!("../prompts/advisor-gate/reviewer-system-prompt.md");

/// Prepended to the advisor's REDO plan when it is fed back as a user turn,
/// instructing the executor to continue rather than stop.
pub const REDO_FEEDBACK_PREFIX: &str = concat!(
    include_str!("../prompts/advisor-gate/redo-feedback-prefix.md"),
    "\n"
);

/// Labels the executor's internal reasoning when a turn has no visible text,
/// so the advisor still has evidence to review (reasoning models on vLLM/NIM
/// can emit turns whose only output is reasoning).
const REASONING_TAIL_LABEL: &str =
    "(the executor produced no visible text this turn; its internal reasoning follows)\n";
/// REDO echo when the discarded turn had neither text nor reasoning; strict
/// endpoints (Anthropic) reject empty text blocks, so never echo "".
const EMPTY_ECHO_PLACEHOLDER: &str = "(the executor produced no output this turn)";
/// Benchmark harnesses stamp every request of one evaluation — sub-agents
/// included — with this header, so it is the review budget's first-choice
/// scope: "reviews for *this* task" survives gateways shared by many tasks.
const BENCH_SESSION_HEADER: &str = "proxy_x_session_id";

/// How the gate decides a buffered executor turn is terminal.
#[derive(Clone, Debug, PartialEq)]
pub enum GateTrigger {
    /// First turn without tool calls (subject to `gate_min_tool_results`).
    NoToolCall,
    /// First turn whose visible text matches this regex (searched, not anchored) —
    /// for text-protocol harnesses where every turn lacks tool calls and
    /// completion is declared with a textual marker instead.
    Pattern(String),
}

/// Gate knobs; defaults mirror the benchmarked Python advisor configuration.
#[derive(Clone, Debug)]
pub struct AdvisorGateConfig {
    /// System prompt for the advisor's review call; states the APPROVE/REDO contract.
    pub reviewer_system_prompt: String,
    /// Prepended to the advisor's REDO plan when fed back to the executor.
    pub redo_feedback_prefix: String,
    /// What fires the review.
    pub gate_trigger: GateTrigger,
    /// Reviews allowed per budget scope. 1 keeps the original once-per-task
    /// gate; higher values re-review later terminal turns, making the gate a
    /// sequential best-of-(N+1) with the advisor as judge.
    pub max_reviews: u32,
    /// When > 0, additionally review (once per conversation, consuming budget)
    /// the first request already carrying at least this many assistant turns —
    /// a mid-task checkpoint for executors that grind without declaring
    /// completion. 0 disables.
    pub gate_stall_turns: u32,
    /// For the `no_tool_call` trigger: only review once the conversation
    /// carries at least this many tool results, skipping early commentary
    /// turns on chatty harnesses. 0 reviews from the first terminal turn.
    pub gate_min_tool_results: u32,
    /// Cap on the advisor's output per consult.
    pub advisor_max_tokens: u64,
    /// Sampling temperature for the consult; `None` omits the field on the wire.
    pub advisor_temperature: Option<f64>,
    /// Cap on the serialized transcript handed to the advisor; the middle of
    /// an over-cap conversation is dropped (task head + recent tail survive).
    pub transcript_max_chars: usize,
    /// When true (default), an advisor failure degrades to APPROVE; when
    /// false, it propagates as the turn's error.
    pub fail_open: bool,
}

impl Default for AdvisorGateConfig {
    fn default() -> Self {
        Self {
            reviewer_system_prompt: REVIEWER_SYSTEM_PROMPT.to_string(),
            redo_feedback_prefix: REDO_FEEDBACK_PREFIX.to_string(),
            gate_trigger: GateTrigger::NoToolCall,
            max_reviews: 1,
            gate_stall_turns: 0,
            gate_min_tool_results: 0,
            advisor_max_tokens: 2048,
            advisor_temperature: None,
            transcript_max_chars: 200_000,
            fail_open: true,
        }
    }
}

/// Advisor review gate: executor turns pass through until the first terminal
/// turn, which a stronger advisor reviews once per scope budget (APPROVE
/// releases it, REDO feeds the plan back and re-invokes the executor).
pub struct AdvisorGate {
    executor: ModelId,
    advisor: ModelId,
    config: AdvisorGateConfig,
    /// Folds request- and response-side facts into the per-turn [`GateSignals`].
    signals: GateSignalProcessor,
    /// Decides, from the signals, whether the buffered turn warrants review.
    trigger: TriggerClassifier,
    /// Reserve/refund review ledger and stall latch — the gate's only mutable state.
    budget: ReviewBudget,
    verdict_re: regex::Regex,
}

impl AdvisorGate {
    /// Validates ranges and compiles the trigger and verdict patterns.
    pub fn new(executor: ModelId, advisor: ModelId, config: AdvisorGateConfig) -> Result<Self> {
        if config.max_reviews < 1 {
            return Err(algorithm_error("max_reviews must be at least 1"));
        }
        if config.advisor_max_tokens < 1 {
            return Err(algorithm_error("advisor_max_tokens must be at least 1"));
        }
        if config.transcript_max_chars < 256 {
            return Err(algorithm_error("transcript_max_chars must be at least 256"));
        }
        let trigger = TriggerClassifier::new(&config)?;
        let verdict_re = regex::Regex::new(VERDICT_PATTERN).map_err(|error| {
            algorithm_error(format!("verdict pattern failed to compile: {error}"))
        })?;
        let budget = ReviewBudget::new(config.max_reviews);
        Ok(Self {
            executor,
            advisor,
            config,
            signals: GateSignalProcessor,
            trigger,
            budget,
            verdict_re,
        })
    }

    // ── Gate flow ───────────────────────────────────────────────────────────

    async fn route_inner(
        &self,
        driver: &Driver,
        request: Request,
        scope: &ScopeKey,
    ) -> Result<RoutingOutcome> {
        // Spent budget (or failure cap): pure passthrough — live stream,
        // verbatim preserved-body replay, zero buffering. Executor errors
        // (including ContextWindowExceeded) propagate for the host's
        // client-visible mapping.
        if self.budget.check_exhausted(scope) {
            return Ok(RoutingOutcome::route_to(
                self.executor.clone(),
                Vec::new(),
                request,
            ));
        }

        // Request-side signals fold in before the executor runs: conversation
        // counts from the shared ToolSignals extraction — the same definition
        // of tool-result/turn counting the stage router uses.
        let mut request = request;
        let mut signals = GateSignals::default();
        self.signals
            .process(
                &mut signals,
                Event::Request {
                    request: &mut request,
                    driver: Some(driver),
                },
            )
            .await?;

        // Gated phase: generate the turn once, fully buffered, so the gate
        // can inspect it before the client sees anything.
        let response = driver
            .call_model(request.clone(), vec![self.executor.clone()])
            .await?;
        let turn = buffer_turn(self.executor.as_str(), response).await?;

        // Response-side signals fold in after it: a terminal turn never
        // appears on any subsequent request — the harness stops once the
        // executor answers without tools — so the trigger classifier can only
        // run on the model-response event.
        self.signals
            .process(&mut signals, Event::ModelResponse(&turn.agg))
            .await?;

        let decision = self.trigger.classify(&signals);
        // The stall checkpoint fires once per conversation regardless of the
        // turn's shape — even a tool-call turn — for executors that grind
        // without ever declaring completion.
        let stall_key = stall_key(&request);
        let stall = decision.stalled && !self.budget.stall_already_fired(stall_key);
        if decision.fired.is_none() && !stall {
            return Ok(RoutingOutcome::answered(
                self.executor.clone(),
                request,
                turn.into_response(),
            ));
        }
        // A stall consumed by a simultaneous trigger does not latch, so the
        // checkpoint can still fire later if this review is refunded.
        if stall && decision.fired.is_none() {
            self.budget.mark_stall_fired(stall_key);
        }
        if !self.budget.try_reserve(scope) {
            return Ok(RoutingOutcome::answered(
                self.executor.clone(),
                request,
                turn.into_response(),
            ));
        }

        let trigger_label = decision.fired.unwrap_or("stall");
        let review_tail = visible_text(&turn.agg).or_else(|| {
            reasoning_text(&turn.agg).map(|reasoning| format!("{REASONING_TAIL_LABEL}{reasoning}"))
        });
        match self
            .consult(driver, &request, review_tail.as_deref(), trigger_label)
            .await
        {
            Ok(ConsultOutcome::Approve) => Ok(RoutingOutcome::answered(
                self.executor.clone(),
                request,
                turn.into_response(),
            )),
            Ok(ConsultOutcome::Redo { plan }) => Ok(self.redo(request, turn, &plan)),
            Ok(ConsultOutcome::Failed) => {
                self.budget.refund_failure(scope);
                Ok(RoutingOutcome::answered(
                    self.executor.clone(),
                    request,
                    turn.into_response(),
                ))
            }
            Err(error) => {
                self.budget.refund_failure(scope);
                Err(error)
            }
        }
    }

    /// REDO: the client never sees the gated turn. Its text (or reasoning) is
    /// echoed as an assistant message, the advisor's plan follows as user
    /// feedback, and the executor continues as a pure passthrough call.
    fn redo(&self, request: Request, turn: GatedTurn, plan: &str) -> RoutingOutcome {
        record_discarded(&turn.agg.usage);
        emit_discarded_audit(self.executor.as_str(), &turn.agg.usage);
        let echo = visible_text(&turn.agg)
            .or_else(|| reasoning_text(&turn.agg))
            .unwrap_or_else(|| EMPTY_ECHO_PLACEHOLDER.to_string());
        let mut redo = request;
        redo.llm_request
            .messages
            .push(Message::text(Role::Assistant, echo));
        redo.llm_request.messages.push(Message::text(
            Role::User,
            format!("{}{}", self.config.redo_feedback_prefix, plan),
        ));
        // Mandatory after any message mutation: codecs otherwise replay the
        // preserved pre-surgery body verbatim and the feedback never reaches
        // the executor.
        crate::algorithms::util::prompts::drop_exact_replay(&mut redo);
        RoutingOutcome::route_to(self.executor.clone(), Vec::new(), redo)
    }

    /// Consults the advisor over the buffered transcript and parses the
    /// verdict. `Ok(Failed)` covers fail-open errors and unparseable replies
    /// (the caller refunds); fail-closed errors return `Err`.
    async fn consult(
        &self,
        driver: &Driver,
        base: &Request,
        review_tail: Option<&str>,
        trigger: &'static str,
    ) -> Result<ConsultOutcome> {
        // The advisor reviews the FULL transcript: system/developer content is
        // normalized out of `messages` into `instructions`, so prepend it back
        // as leading messages (identical {role, content} shape) — the task
        // constraints the verdict must check against usually live there.
        let transcript_messages: Vec<Message> = base
            .llm_request
            .instructions
            .iter()
            .map(|block| Message {
                role: block.role,
                content: block.content.clone(),
            })
            .chain(base.llm_request.messages.iter().cloned())
            .collect();
        let transcript = review_transcript(
            &transcript_messages,
            review_tail,
            self.config.transcript_max_chars,
        );
        let consult_request = self.build_consult_request(base, transcript);
        let started = Instant::now();
        let reply = match driver
            .call_model(consult_request, vec![self.advisor.clone()])
            .await
        {
            Ok(response) => response
                .llm_response
                .into_agg()
                .await
                .map_err(|source| LibsyError::client_call(self.advisor.clone(), source)),
            Err(error) => Err(error),
        };
        let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
        let agg = match reply {
            Ok(agg) => agg,
            Err(error) => {
                record_consult_failure(crate::algorithms::util::llm_judge::libsy_error_reason(
                    &error,
                ));
                if !self.config.fail_open {
                    // Surface as an algorithm failure (5xx), never as the
                    // advisor's own client error: a typed ContextWindowExceeded
                    // from the consult would otherwise reach the client as 400
                    // context_length_exceeded and trigger compaction of a
                    // healthy conversation.
                    return Err(algorithm_error(format!(
                        "advisor consult failed (fail_open = false): {error}"
                    )));
                }
                tracing::warn!(
                    target: "libsy",
                    error = %error,
                    "advisor gate: consult failed; passing the turn through (fail open)"
                );
                emit_review_audit(ReviewAudit {
                    verdict: "APPROVE",
                    error: Some(error.to_string()),
                    latency_ms,
                    reply_head: None,
                    usage: None,
                });
                return Ok(ConsultOutcome::Failed);
            }
        };
        let reply_text = advisor_reply_text(&agg);
        let reply_head: String = reply_text.chars().take(160).collect();
        match parse_verdict(&self.verdict_re, &reply_text) {
            Some(Verdict::Approve) => {
                record_review("approve", trigger);
                emit_review_audit(ReviewAudit {
                    verdict: "APPROVE",
                    error: None,
                    latency_ms,
                    reply_head: Some(reply_head),
                    usage: Some(&agg.usage),
                });
                Ok(ConsultOutcome::Approve)
            }
            Some(Verdict::Redo { plan }) => {
                record_review("redo", trigger);
                emit_review_audit(ReviewAudit {
                    verdict: "REDO",
                    error: None,
                    latency_ms,
                    reply_head: Some(reply_head),
                    usage: Some(&agg.usage),
                });
                Ok(ConsultOutcome::Redo { plan })
            }
            None => {
                // The advisor spent real tokens on a reply the gate cannot
                // act on; the observer already recorded them. Refunded by
                // the caller so a flaky advisor cannot burn the budget.
                record_review("unparseable", trigger);
                emit_review_audit(ReviewAudit {
                    verdict: "UNPARSEABLE",
                    error: None,
                    latency_ms,
                    reply_head: Some(reply_head),
                    usage: Some(&agg.usage),
                });
                Ok(ConsultOutcome::Failed)
            }
        }
    }

    /// A fresh, buffered, tool-free request carrying the reviewer contract and
    /// the serialized transcript; metadata is kept for session correlation.
    fn build_consult_request(&self, base: &Request, transcript: String) -> Request {
        Request {
            llm_request: LlmRequest {
                model: base.llm_request.model.clone(),
                instructions: vec![InstructionBlock {
                    role: Role::System,
                    content: vec![ContentBlock::Text {
                        text: self.config.reviewer_system_prompt.clone(),
                    }],
                }],
                messages: vec![Message::text(Role::User, transcript)],
                sampling: SamplingParams {
                    temperature: self.config.advisor_temperature,
                    ..SamplingParams::default()
                },
                output: OutputParams {
                    max_output_tokens: Some(self.config.advisor_max_tokens),
                    response_format: None,
                },
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: base.metadata.clone(),
        }
    }
}

#[async_trait::async_trait]
impl Algorithm for AdvisorGate {
    fn name(&self) -> &str {
        "advisor_gate"
    }

    async fn route(self: Arc<Self>, driver: Driver, request: Request) -> Result<RoutingOutcome> {
        let scope = budget_scope(&request);
        let session_final = request
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.session_final)
            == Some(true);
        let result = self.route_inner(&driver, request, &scope).await;
        if session_final {
            self.budget.evict_scope(&scope);
        }
        result
    }
}

/// Outcome of one consult; `Failed` = fail-open error or unparseable reply.
enum ConsultOutcome {
    Approve,
    Redo { plan: String },
    Failed,
}

fn algorithm_error(message: impl Into<String>) -> LibsyError {
    LibsyError::AlgorithmError {
        message: message.into(),
    }
}
