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
//! Structure: [`AdvisorGate`] is a thin orchestrator — the [`signals`]
//! processor folds each event's facts into per-turn state, the [`trigger`]
//! classifier reads them after the executor call, and the [`budget`] ledger
//! holds the only mutable state.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use switchyard_protocol::{
    AggLlmResponse, Category, ContentBlock, InstructionBlock, LlmRequest, Message, ModelId,
    OutputParams, Request, Role, SamplingParams, ToolChoice, WireFormat,
};

use super::util::prompts::drop_exact_replay;
use super::util::tool_signals::is_mutating_tool_call;
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

/// Shape of the request sent to the reviewer.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum ReviewContext {
    /// Sends a bounded JSON transcript under a fresh reviewer instruction.
    #[default]
    Transcript,
    /// Appends the completion and reviewer request to the live conversation.
    Conversation,
    /// Keeps the planning prefix through the first mutation and recent tool evidence.
    ExecutionDelta,
}

/// Gate knobs; defaults mirror the benchmarked Python advisor configuration.
#[derive(Clone, Debug)]
pub struct AdvisorGateConfig {
    /// System prompt for the advisor's review call; states the APPROVE/REDO contract.
    pub reviewer_system_prompt: String,
    /// Optional instruction prefix restored before a conversation review.
    pub reviewer_prefix_prompt: Option<String>,
    /// Prepended to the advisor's REDO plan when fed back to the executor.
    pub redo_feedback_prefix: String,
    /// How conversation state is presented to the reviewer.
    pub review_context: ReviewContext,
    /// What fires the review.
    pub gate_trigger: GateTrigger,
    /// Requires a matching trigger response to contain no tool call.
    pub gate_require_no_tool_call: bool,
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
            reviewer_prefix_prompt: None,
            redo_feedback_prefix: REDO_FEEDBACK_PREFIX.to_string(),
            review_context: ReviewContext::Transcript,
            gate_trigger: GateTrigger::NoToolCall,
            gate_require_no_tool_call: false,
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
    /// Validates the config. Models are supplied when each request runs.
    pub fn new(config: AdvisorGateConfig) -> Result<Self> {
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
        review_base: Option<&Request>,
    ) -> Result<RoutingOutcome> {
        let executor_models = driver.models_for(&Category::Efficient).to_vec();
        let executor = executor_models
            .first()
            .ok_or_else(|| LibsyError::AlgorithmError {
                message: "no models available for category Efficient".to_string(),
            })?;

        // Spent budget (or failure cap): pure passthrough — live stream,
        // verbatim preserved-body replay, zero buffering. Executor errors
        // (including ContextWindowExceeded) propagate for the host's
        // client-visible mapping.
        if self.budget.check_exhausted(scope) {
            return Ok(RoutingOutcome::route_to(
                executor.clone(),
                executor_models[1..].to_vec(),
                request,
            ));
        }

        // Request-side signals fold in before the executor runs.
        let mut request = request;
        let mut signals = GateSignals::default();
        self.signals
            .process(
                &mut signals,
                Event::Request {
                    request: &mut request,
                    driver,
                },
            )
            .await?;

        // Gated phase: generate the turn once, fully buffered, so the gate
        // can inspect it before the client sees anything.
        let response = driver
            .call_model(request.clone(), executor_models.clone())
            .await?;
        let served_executor = response
            .served_model()
            .cloned()
            .unwrap_or_else(|| executor.clone());
        let turn = buffer_turn(served_executor.as_str(), response).await?;

        // Response-side signals fold in after it: the terminal turn never
        // appears on a later request, so the trigger runs on this event.
        self.signals
            .process(&mut signals, Event::ModelResponse(&turn.agg))
            .await?;

        let decision = self.trigger.classify(&signals);
        // The stall checkpoint fires once per conversation regardless of the
        // turn's shape. Only a stall with no simultaneous trigger latches
        // (atomically — one winner per conversation), so a refunded review
        // leaves the checkpoint re-armed.
        let stall = decision.fired.is_none()
            && decision.stalled
            && self.budget.try_mark_stall_fired(stall_key(&request));
        if decision.fired.is_none() && !stall {
            return Ok(RoutingOutcome::answered(
                served_executor.clone(),
                request,
                turn.into_response(),
            ));
        }
        if !self.budget.try_reserve(scope) {
            return Ok(RoutingOutcome::answered(
                served_executor.clone(),
                request,
                turn.into_response(),
            ));
        }

        let trigger_label = decision.fired.unwrap_or("stall");
        let review_tail = visible_text(&turn.agg).or_else(|| {
            reasoning_text(&turn.agg).map(|reasoning| format!("{REASONING_TAIL_LABEL}{reasoning}"))
        });
        match self
            .consult(
                driver,
                review_base.unwrap_or(&request),
                &turn.agg,
                review_tail.as_deref(),
                trigger_label,
            )
            .await
        {
            Ok(ConsultOutcome::Approve) => {
                driver.set_evidence(serde_json::json!({
                    "source": "advisor",
                    "verdict": "approve",
                    "trigger": trigger_label,
                }));
                Ok(RoutingOutcome::answered(
                    served_executor.clone(),
                    request,
                    turn.into_response(),
                ))
            }
            Ok(ConsultOutcome::Redo { plan }) => {
                driver.set_evidence(serde_json::json!({
                    "source": "advisor",
                    "verdict": "redo",
                    "trigger": trigger_label,
                }));
                Ok(self.redo(
                    executor,
                    &executor_models[1..],
                    &served_executor,
                    request,
                    turn,
                    &plan,
                ))
            }
            Ok(ConsultOutcome::Failed { reason }) => {
                self.budget.refund_failure(scope);
                driver.set_evidence(serde_json::json!({
                    "source": "advisor",
                    "verdict": "fail_open",
                    "trigger": trigger_label,
                    "reason_code": reason,
                }));
                Ok(RoutingOutcome::answered(
                    served_executor,
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

    /// Routes an executor turn while using a separate conversation for review.
    pub(super) async fn route_with_review_base(
        self: Arc<Self>,
        driver: Driver,
        request: Request,
        review_base: Option<Request>,
    ) -> Result<RoutingOutcome> {
        let scope = budget_scope(&request);
        let session_final = request
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.session_final)
            == Some(true);
        let result = self
            .route_inner(&driver, request, &scope, review_base.as_ref())
            .await;
        if session_final {
            self.budget.evict_scope(&scope);
        }
        result
    }

    /// REDO: the client never sees the gated turn. Its text (or reasoning) is
    /// echoed as an assistant message, the advisor's plan follows as user
    /// feedback, and the executor continues as a pure passthrough call.
    fn redo(
        &self,
        executor: &ModelId,
        executor_fallbacks: &[ModelId],
        served_executor: &ModelId,
        request: Request,
        turn: GatedTurn,
        plan: &str,
    ) -> RoutingOutcome {
        record_discarded(&turn.agg.usage);
        emit_discarded_audit(served_executor.as_str(), &turn.agg.usage);
        let mut redo = request;
        let echo = match &self.config.review_context {
            ReviewContext::Transcript => {
                let text = visible_text(&turn.agg)
                    .or_else(|| reasoning_text(&turn.agg))
                    .unwrap_or_else(|| EMPTY_ECHO_PLACEHOLDER.to_string());
                Message::text(Role::Assistant, text)
            }
            ReviewContext::Conversation | ReviewContext::ExecutionDelta => {
                response_message(&turn.agg, EMPTY_ECHO_PLACEHOLDER)
            }
        };
        redo.llm_request.messages.push(echo);
        let feedback = format!("{}{}", self.config.redo_feedback_prefix, plan);
        let exact_responses_extended = extend_exact_responses_turn(&mut redo, &turn.agg, &feedback);
        redo.llm_request
            .messages
            .push(Message::text(Role::User, feedback));
        if !exact_responses_extended {
            drop_exact_replay(&mut redo);
        }
        RoutingOutcome::route_to(executor.clone(), executor_fallbacks.to_vec(), redo)
    }

    /// Consults the advisor over the configured review context and parses the
    /// verdict. `Ok(Failed)` covers fail-open errors and unparseable replies
    /// (the caller refunds); fail-closed errors return `Err`.
    async fn consult(
        &self,
        driver: &Driver,
        base: &Request,
        review_response: &AggLlmResponse,
        review_tail: Option<&str>,
        trigger: &'static str,
    ) -> Result<ConsultOutcome> {
        // The advisor reviews the FULL transcript: system/developer content is
        // normalized out of `messages` into `instructions`, so prepend it back
        // as leading messages with identical {role, content} shape. The task
        // constraints the verdict must check against usually live there.
        let consult_request = self.build_consult_request(base, review_response, review_tail);
        let started = Instant::now();
        // An unresolvable advisor is treated like any other consult failure, so
        // fail_open still returns the buffered executor turn to the client.
        let mut review_model = None;
        let reply = match driver.first_model_for(&Category::Judge) {
            Ok(advisor) => {
                let advisor = advisor.clone();
                review_model = Some(advisor.clone());
                let advisor_models = driver.models_for(&Category::Judge).to_vec();
                match driver.call_model(consult_request, advisor_models).await {
                    Ok(response) => {
                        let served_advisor = response
                            .served_model()
                            .cloned()
                            .unwrap_or_else(|| advisor.clone());
                        review_model = Some(served_advisor.clone());
                        response
                            .llm_response
                            .into_agg()
                            .await
                            .map_err(|source| LibsyError::client_call(served_advisor, source))
                    }
                    Err(error) => Err(error),
                }
            }
            Err(error) => Err(error),
        };
        let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
        let agg = match reply {
            Ok(agg) => agg,
            Err(error) => {
                let reason = crate::algorithms::util::llm_judge::libsy_error_reason(&error);
                record_consult_failure(reason);
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
                    model: review_model.as_ref().map_or("unavailable", ModelId::as_str),
                    verdict: "APPROVE",
                    error: Some(error.to_string()),
                    latency_ms,
                    reply_head: None,
                    usage: None,
                });
                return Ok(ConsultOutcome::Failed { reason });
            }
        };
        let reply_text = advisor_reply_text(&agg);
        let reply_head: String = reply_text.chars().take(160).collect();
        match parse_verdict(&self.verdict_re, &reply_text) {
            Some(Verdict::Approve) => {
                record_review("approve", trigger);
                emit_review_audit(ReviewAudit {
                    model: review_model.as_ref().map_or("unavailable", ModelId::as_str),
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
                    model: review_model.as_ref().map_or("unavailable", ModelId::as_str),
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
                    model: review_model.as_ref().map_or("unavailable", ModelId::as_str),
                    verdict: "UNPARSEABLE",
                    error: None,
                    latency_ms,
                    reply_head: Some(reply_head),
                    usage: Some(&agg.usage),
                });
                Ok(ConsultOutcome::Failed {
                    reason: "parse_error",
                })
            }
        }
    }

    /// Builds a buffered, tool-free request carrying the selected review context.
    fn build_consult_request(
        &self,
        base: &Request,
        review_response: &AggLlmResponse,
        review_tail: Option<&str>,
    ) -> Request {
        match &self.config.review_context {
            ReviewContext::Transcript => {
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
            ReviewContext::Conversation | ReviewContext::ExecutionDelta => {
                let mut request = match self.config.review_context {
                    ReviewContext::ExecutionDelta => compact_execution_delta(base),
                    _ => base.clone(),
                };
                if let Some(prefix) = &self.config.reviewer_prefix_prompt
                    && !instructions_start_with(&request.llm_request.instructions, prefix)
                {
                    request.llm_request.instructions.insert(
                        0,
                        InstructionBlock {
                            role: Role::System,
                            content: vec![ContentBlock::Text {
                                text: prefix.clone(),
                            }],
                        },
                    );
                }
                request.llm_request.messages.push(response_message(
                    review_response,
                    review_tail.unwrap_or(transcript::NO_TEXT_PLACEHOLDER),
                ));
                request.llm_request.messages.push(Message::text(
                    Role::User,
                    self.config.reviewer_system_prompt.clone(),
                ));
                request.llm_request.tool_choice = Some(ToolChoice::None);
                request.llm_request.sampling.temperature = self.config.advisor_temperature;
                request.llm_request.output.max_output_tokens = Some(self.config.advisor_max_tokens);
                request.llm_request.output.response_format = None;
                request.raw_request = None;
                if extend_exact_responses_turn(
                    &mut request,
                    review_response,
                    &self.config.reviewer_system_prompt,
                ) {
                    configure_exact_responses_review(
                        &mut request,
                        self.config.advisor_temperature,
                        self.config.reviewer_prefix_prompt.as_deref(),
                    );
                } else {
                    drop_exact_replay(&mut request);
                }
                request
            }
        }
    }
}

const RECENT_REVIEW_TOOL_CALLS: usize = 8;

fn compact_execution_delta(base: &Request) -> Request {
    let mut request = base.clone();
    request.llm_request.messages = compact_messages(&base.llm_request.messages);

    let format = switchyard_protocol::FormatId::known(WireFormat::OpenAiResponses);
    let Some(body) = request.llm_request.preservation.requests.get_mut(&format) else {
        return request;
    };
    let Some(input) = body
        .get_mut("input")
        .and_then(serde_json::Value::as_array_mut)
    else {
        return request;
    };
    *input = compact_responses_input(input);
    request
}

fn compact_messages(messages: &[Message]) -> Vec<Message> {
    let Some(boundary) = messages.iter().position(message_has_mutation) else {
        return messages.to_vec();
    };
    let selected_call_ids = selected_message_call_ids(&messages[boundary..]);
    messages[..boundary]
        .iter()
        .cloned()
        .chain(messages[boundary..].iter().filter_map(|message| {
            let selected_call_in_message = message.content.iter().any(|block| {
                matches!(block, ContentBlock::ToolCall(call) if selected_call_ids.contains(&call.id))
            });
            let content = message
                .content
                .iter()
                .filter(|block| match block {
                    ContentBlock::ToolCall(call) => selected_call_ids.contains(&call.id),
                    ContentBlock::ToolResult(result) => {
                        selected_call_ids.contains(&result.tool_call_id)
                    }
                    _ => selected_call_in_message,
                })
                .cloned()
                .collect::<Vec<_>>();
            (!content.is_empty()).then_some(Message {
                role: message.role,
                content,
            })
        }))
        .collect()
}

fn message_has_mutation(message: &Message) -> bool {
    message.content.iter().any(|block| {
        matches!(block, ContentBlock::ToolCall(call) if is_mutating_tool_call(&call.name, &call.arguments))
    })
}

fn selected_message_call_ids(messages: &[Message]) -> HashSet<String> {
    let calls = messages
        .iter()
        .flat_map(|message| message.content.iter())
        .filter_map(|block| match block {
            ContentBlock::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect::<Vec<_>>();
    let recent_start = calls.len().saturating_sub(RECENT_REVIEW_TOOL_CALLS);
    let first_mutation = calls
        .iter()
        .find(|call| is_mutating_tool_call(&call.name, &call.arguments))
        .map(|call| call.id.as_str());
    calls
        .iter()
        .enumerate()
        .filter(|(index, call)| *index >= recent_start || Some(call.id.as_str()) == first_mutation)
        .map(|(_, call)| call.id.clone())
        .collect()
}

fn compact_responses_input(input: &[serde_json::Value]) -> Vec<serde_json::Value> {
    let Some(boundary) = input.iter().position(raw_item_is_mutation) else {
        return input.to_vec();
    };
    let calls = input[boundary..]
        .iter()
        .filter(|item| {
            item.get("type").and_then(serde_json::Value::as_str) == Some("function_call")
        })
        .collect::<Vec<_>>();
    let recent_start = calls.len().saturating_sub(RECENT_REVIEW_TOOL_CALLS);
    let first_mutation = calls
        .iter()
        .find(|item| raw_item_is_mutation(item))
        .and_then(|item| item.get("call_id"))
        .and_then(serde_json::Value::as_str);
    let selected_call_ids = calls
        .iter()
        .enumerate()
        .filter(|(index, item)| {
            *index >= recent_start
                || item.get("call_id").and_then(serde_json::Value::as_str) == first_mutation
        })
        .filter_map(|(_, item)| item.get("call_id").and_then(serde_json::Value::as_str))
        .collect::<HashSet<_>>();

    input[..boundary]
        .iter()
        .cloned()
        .chain(
            input[boundary..]
                .iter()
                .filter(|item| {
                    let kind = item.get("type").and_then(serde_json::Value::as_str);
                    matches!(kind, Some("function_call") | Some("function_call_output"))
                        && item
                            .get("call_id")
                            .and_then(serde_json::Value::as_str)
                            .is_some_and(|call_id| selected_call_ids.contains(call_id))
                })
                .cloned(),
        )
        .collect()
}

fn raw_item_is_mutation(item: &serde_json::Value) -> bool {
    if item.get("type").and_then(serde_json::Value::as_str) != Some("function_call") {
        return false;
    }
    let Some(name) = item.get("name").and_then(serde_json::Value::as_str) else {
        return false;
    };
    is_mutating_tool_call(
        name,
        item.get("arguments").unwrap_or(&serde_json::Value::Null),
    )
}

fn response_message(response: &AggLlmResponse, fallback: &str) -> Message {
    let content = response
        .outputs
        .iter()
        .flat_map(|output| output.content.iter().cloned())
        .collect::<Vec<_>>();
    if content.is_empty() {
        return Message::text(Role::Assistant, fallback);
    }
    Message {
        role: response
            .outputs
            .first()
            .map_or(Role::Assistant, |output| output.role),
        content,
    }
}

fn extend_exact_responses_turn(
    request: &mut Request,
    response: &AggLlmResponse,
    user_text: &str,
) -> bool {
    let format = switchyard_protocol::FormatId::known(WireFormat::OpenAiResponses);
    let Some(response_output) = response
        .preservation
        .responses
        .get(&format)
        .and_then(|body| body.get("output"))
        .and_then(serde_json::Value::as_array)
        .cloned()
    else {
        return false;
    };
    let Some(input) = request
        .llm_request
        .preservation
        .requests
        .get_mut(&format)
        .and_then(|body| body.get_mut("input"))
        .and_then(serde_json::Value::as_array_mut)
    else {
        return false;
    };
    input.extend(response_output);
    input.push(serde_json::json!({
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": user_text}],
    }));
    request
        .llm_request
        .preservation
        .requests
        .retain(|candidate, _| candidate == &format);
    true
}

fn configure_exact_responses_review(
    request: &mut Request,
    temperature: Option<f64>,
    reviewer_prefix_prompt: Option<&str>,
) {
    let format = switchyard_protocol::FormatId::known(WireFormat::OpenAiResponses);
    let Some(body) = request
        .llm_request
        .preservation
        .requests
        .get_mut(&format)
        .and_then(serde_json::Value::as_object_mut)
    else {
        return;
    };
    body.insert("tool_choice".to_string(), serde_json::json!("none"));
    body.remove("max_output_tokens");
    if let Some(prefix) = reviewer_prefix_prompt {
        let instructions = body
            .get("instructions")
            .and_then(serde_json::Value::as_str)
            .filter(|instructions| !instructions.is_empty())
            .map_or_else(
                || prefix.to_string(),
                |instructions| {
                    if instructions == prefix || instructions.starts_with(&format!("{prefix}\n\n"))
                    {
                        instructions.to_string()
                    } else {
                        format!("{prefix}\n\n{instructions}")
                    }
                },
            );
        body.insert(
            "instructions".to_string(),
            serde_json::Value::String(instructions),
        );
    }
    if let Some(temperature) = temperature {
        body.insert("temperature".to_string(), serde_json::json!(temperature));
    } else {
        body.remove("temperature");
    }
    body.remove("text");
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
        let result = self.route_inner(&driver, request, &scope, None).await;
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
    Failed { reason: &'static str },
}

fn algorithm_error(message: impl Into<String>) -> LibsyError {
    LibsyError::AlgorithmError {
        message: message.into(),
    }
}

fn instructions_start_with(instructions: &[InstructionBlock], prefix: &str) -> bool {
    instructions.first().is_some_and(|block| {
        block.role == Role::System
            && block.content.first().is_some_and(
                |content| matches!(content, ContentBlock::Text { text } if text == prefix),
            )
    })
}
