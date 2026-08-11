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
//! whole instance — see [`budget_scope`]) is reviewed at most `max_reviews`
//! times; afterwards every call is a pure passthrough.
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

use std::collections::{HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;
use std::time::Instant;

use futures::StreamExt;
use opentelemetry::KeyValue;
use parking_lot::Mutex;
use switchyard_protocol::{
    AggLlmResponse, ContentBlock, Decision, InstructionBlock, LlmClientError, LlmRequest,
    LlmResponse, LlmResponseChunk, LlmResponseStreamEvent, Message, OutputParams, Request,
    Response, ResponseAccumulator, Role, SamplingParams, StopReason, Usage,
};

use crate::core::algorithm::{Algorithm, Driver, LlmTarget};
use crate::{LibsyError, Result, observability};

/// APPROVE/REDO reviewer contract sent as the advisor's system prompt.
pub const REVIEWER_SYSTEM_PROMPT: &str = "You are a senior reviewer acting as a quality gate for a faster executor model working a coding/agent task. You are given the full transcript: the task, every action the executor took and every result it saw, and its latest message — in which it has either (a) proposed a plan before doing the work, or (b) concluded the task is complete.\n\nDecide whether to let the executor stop or send it back to keep working. Put your verdict as the FIRST word of your reply:\n\n- APPROVE — the proposed plan is sound, OR the work is genuinely complete and correct. Reply with exactly: APPROVE\n- REDO — the plan has a real flaw, OR the work is incomplete/incorrect: an unhandled edge case, an untested assumption, a subtly wrong approach, missing verification, or a stated requirement not met. Reply: REDO, then a SHORT, concrete, actionable plan naming exactly what is wrong or missing and what to do about it. No generic advice — point at the specific gap.\n\nBias toward APPROVE when the work looks correct and complete; the executor has already done its own iteration. Use REDO specifically to catch a premature \"done\" on a subtly incomplete solution, or a flawed plan before it is executed. A self-claim of success is not proof — check the actual task requirements against what was actually done.\n";

/// Prepended to the advisor's REDO plan when it is fed back as a user turn,
/// instructing the executor to continue rather than stop.
pub const REDO_FEEDBACK_PREFIX: &str = "A senior reviewer examined your work and determined the task is NOT yet complete or correct. Do not stop here — address the following, then keep working until it is genuinely done:\n\n";

/// Labels the executor's internal reasoning when a turn has no visible text,
/// so the advisor still has evidence to review (reasoning models on vLLM/NIM
/// can emit turns whose only output is reasoning).
const REASONING_TAIL_LABEL: &str =
    "(the executor produced no visible text this turn; its internal reasoning follows)\n";
/// Splices the two surviving ends of an over-cap transcript.
const TRUNCATION_MARKER: &str = "\n...<middle of the conversation truncated>...\n";
/// Stands in for a terminal turn with no reviewable text at all.
const NO_TEXT_PLACEHOLDER: &str = "(no text)";
/// REDO echo when the discarded turn had neither text nor reasoning; strict
/// endpoints (Anthropic) reject empty text blocks, so never echo "".
const EMPTY_ECHO_PLACEHOLDER: &str = "(the executor produced no output this turn)";
/// Failed consults tolerated per scope before the gate stops consulting.
/// Failures refund the review budget — a transient advisor error must not
/// silently exhaust `max_reviews` with zero real reviews — so this separate
/// cap is what bounds per-turn consult latency against a down advisor.
const MAX_FAILED_CONSULTS: u32 = 3;
/// Bounds tracked budget scopes and stall keys; a scope dropped at the bound
/// re-arms like a process restart (rare, harmless).
const MAX_TRACKED_SCOPES: usize = 1_024;
/// Benchmark harnesses stamp every request of one evaluation — sub-agents
/// included — with this header, so it is the review budget's first-choice
/// scope: "reviews for *this* task" survives gateways shared by many tasks.
const BENCH_SESSION_HEADER: &str = "proxy_x_session_id";
/// Anchored verdict parse: optional wrapper characters and an optional
/// "(final) verdict:" label, then APPROVE or REDO as the first real word.
/// Anchoring matters — an unanchored scan turns "I cannot approve this —
/// REDO: run the tests" into APPROVE.
const VERDICT_PATTERN: &str =
    r#"(?i)^[\s*_#>"'(\[`]*(?:(?:final\s+)?verdict\s*:\s*[\s*_#>"'(\[`]*)?(APPROVE|REDO)\b"#;

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

/// The trigger with its pattern compiled once at construction.
enum CompiledTrigger {
    NoToolCall,
    Pattern(regex::Regex),
}

/// Review budget scope, in precedence order: the benchmark harness header
/// (exact evaluation identity, sub-agents included), then the host-resolved
/// session id, then one instance-wide scope for headerless clients.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
enum ScopeKey {
    Instance,
    Client(String),
    Session(String),
}

/// Per-scope review ledger.
#[derive(Default)]
struct ScopeState {
    reviews: u32,
    failed_consults: u32,
    exhaustion_logged: bool,
}

/// Shared mutable gate state; every access is a short critical section and
/// the lock is never held across an await.
#[derive(Default)]
struct GateState {
    scopes: HashMap<ScopeKey, ScopeState>,
    stall_fired: HashSet<u64>,
}

/// Advisor review gate: executor turns pass through until the first terminal
/// turn, which a stronger advisor reviews once per scope budget (APPROVE
/// releases it, REDO feeds the plan back and re-invokes the executor).
pub struct AdvisorGate {
    executor: LlmTarget,
    advisor: LlmTarget,
    config: AdvisorGateConfig,
    trigger: CompiledTrigger,
    verdict_re: regex::Regex,
    state: Mutex<GateState>,
}

impl AdvisorGate {
    /// Validates ranges and compiles the trigger and verdict patterns.
    pub fn new(executor: LlmTarget, advisor: LlmTarget, config: AdvisorGateConfig) -> Result<Self> {
        if config.max_reviews < 1 {
            return Err(algorithm_error("max_reviews must be at least 1"));
        }
        if config.advisor_max_tokens < 1 {
            return Err(algorithm_error("advisor_max_tokens must be at least 1"));
        }
        if config.transcript_max_chars < 256 {
            return Err(algorithm_error("transcript_max_chars must be at least 256"));
        }
        let trigger = match &config.gate_trigger {
            GateTrigger::NoToolCall => CompiledTrigger::NoToolCall,
            GateTrigger::Pattern(pattern) => {
                if pattern.is_empty() {
                    return Err(algorithm_error(
                        "gate_trigger 'pattern' requires a non-empty gate_trigger_pattern",
                    ));
                }
                CompiledTrigger::Pattern(regex::Regex::new(pattern).map_err(|error| {
                    algorithm_error(format!(
                        "gate_trigger_pattern is not a valid regex: {error}"
                    ))
                })?)
            }
        };
        let verdict_re = regex::Regex::new(VERDICT_PATTERN).map_err(|error| {
            algorithm_error(format!("verdict pattern failed to compile: {error}"))
        })?;
        Ok(Self {
            executor,
            advisor,
            config,
            trigger,
            verdict_re,
            state: Mutex::new(GateState::default()),
        })
    }

    /// One executor Decision; published immediately before each executor call
    /// so `trace.last()` always names the executor on every return path.
    fn executor_decision(&self, reasoning: &str) -> Decision {
        Decision::new(
            self.executor.semantic_name.clone(),
            Some(format!("advisor gate: {reasoning}")),
            true,
        )
    }

    // ── Scope ledger ────────────────────────────────────────────────────────

    /// Whether the scope's budget or failure cap is spent; logs once per scope.
    fn check_exhausted(&self, scope: &ScopeKey) -> bool {
        let mut state = self.state.lock();
        let Some(entry) = state.scopes.get_mut(scope) else {
            return false;
        };
        let exhausted = entry.reviews >= self.config.max_reviews
            || entry.failed_consults >= MAX_FAILED_CONSULTS;
        if exhausted && !entry.exhaustion_logged {
            entry.exhaustion_logged = true;
            tracing::info!(
                target: "libsy",
                scope = ?scope,
                "advisor gate: review budget spent; passing through"
            );
        }
        exhausted
    }

    /// Atomically re-checks exhaustion and reserves one review. Reserving
    /// before the consult await means concurrent same-scope requests cannot
    /// overdraw `max_reviews`; a loser returns its buffered turn unreviewed.
    fn try_reserve(&self, scope: &ScopeKey) -> bool {
        let mut state = self.state.lock();
        if state.scopes.len() >= MAX_TRACKED_SCOPES && !state.scopes.contains_key(scope) {
            let evict = state
                .scopes
                .keys()
                .find(|key| **key != ScopeKey::Instance)
                .cloned();
            if let Some(key) = evict {
                state.scopes.remove(&key);
            }
        }
        let max_reviews = self.config.max_reviews;
        let entry = state.scopes.entry(scope.clone()).or_default();
        if entry.reviews >= max_reviews || entry.failed_consults >= MAX_FAILED_CONSULTS {
            return false;
        }
        entry.reviews += 1;
        true
    }

    /// Returns a reserved review after a failed consult and counts the
    /// failure; applied on fail-open *and* fail-closed paths so the failure
    /// cap bounds both.
    fn refund_failure(&self, scope: &ScopeKey) {
        let mut state = self.state.lock();
        let entry = state.scopes.entry(scope.clone()).or_default();
        entry.reviews = entry.reviews.saturating_sub(1);
        entry.failed_consults += 1;
    }

    /// Drops a completed session's ledger entry; the instance scope persists.
    fn evict_scope(&self, scope: &ScopeKey) {
        if *scope == ScopeKey::Instance {
            return;
        }
        self.state.lock().scopes.remove(scope);
    }

    fn stall_already_fired(&self, key: u64) -> bool {
        self.state.lock().stall_fired.contains(&key)
    }

    fn mark_stall_fired(&self, key: u64) {
        let mut state = self.state.lock();
        if state.stall_fired.len() >= MAX_TRACKED_SCOPES {
            let drop = state.stall_fired.iter().next().copied();
            if let Some(key) = drop {
                state.stall_fired.remove(&key);
            }
        }
        state.stall_fired.insert(key);
    }

    // ── Gate flow ───────────────────────────────────────────────────────────

    async fn route_inner(
        &self,
        driver: &Driver,
        request: Request,
        scope: &ScopeKey,
    ) -> Result<Response> {
        // Spent budget (or failure cap): pure passthrough — live stream,
        // verbatim preserved-body replay, zero buffering. Executor errors
        // (including ContextWindowExceeded) propagate for the host's
        // client-visible mapping.
        if self.check_exhausted(scope) {
            let decision = self.executor_decision("review budget spent; passthrough");
            driver.decide(decision.clone()).await?;
            return driver.call_model(request, decision).await;
        }

        // Gated phase: generate the turn once, fully buffered, so the gate
        // can inspect it before the client sees anything.
        let decision = self.executor_decision("executor turn");
        driver.decide(decision.clone()).await?;
        let response = driver.call_model(request.clone(), decision).await?;
        let turn = buffer_turn(&self.executor.semantic_name, response).await?;

        // The stall checkpoint fires once per conversation regardless of the
        // turn's shape — even a tool-call turn — for executors that grind
        // without ever declaring completion.
        let stall_key = stall_key(&request);
        let stall = self.config.gate_stall_turns > 0
            && !self.stall_already_fired(stall_key)
            && assistant_turns(&request.llm_request.messages) >= self.config.gate_stall_turns;
        let triggered = match &self.trigger {
            CompiledTrigger::Pattern(pattern) => {
                pattern.is_match(visible_text(&turn.agg).as_deref().unwrap_or(""))
            }
            CompiledTrigger::NoToolCall => {
                !has_tool_use(&turn.agg)
                    && count_tool_results(&request.llm_request.messages)
                        >= self.config.gate_min_tool_results
            }
        };
        if !(triggered || stall) {
            return Ok(turn.into_response());
        }
        // A stall consumed by a simultaneous trigger does not latch, so the
        // checkpoint can still fire later if this review is refunded.
        if stall && !triggered {
            self.mark_stall_fired(stall_key);
        }
        if !self.try_reserve(scope) {
            return Ok(turn.into_response());
        }

        let trigger_label = match (&self.trigger, triggered) {
            (CompiledTrigger::Pattern(_), true) => "pattern",
            (CompiledTrigger::NoToolCall, true) => "no_tool_call",
            _ => "stall",
        };
        let review_tail = visible_text(&turn.agg).or_else(|| {
            reasoning_text(&turn.agg).map(|reasoning| format!("{REASONING_TAIL_LABEL}{reasoning}"))
        });
        match self
            .consult(driver, &request, review_tail.as_deref(), trigger_label)
            .await
        {
            Ok(ConsultOutcome::Approve) => Ok(turn.into_response()),
            Ok(ConsultOutcome::Redo { plan }) => self.redo(driver, request, turn, &plan).await,
            Ok(ConsultOutcome::Failed) => {
                self.refund_failure(scope);
                Ok(turn.into_response())
            }
            Err(error) => {
                self.refund_failure(scope);
                Err(error)
            }
        }
    }

    /// REDO: the client never sees the gated turn. Its text (or reasoning) is
    /// echoed as an assistant message, the advisor's plan follows as user
    /// feedback, and the executor continues as a pure passthrough call.
    async fn redo(
        &self,
        driver: &Driver,
        request: Request,
        turn: GatedTurn,
        plan: &str,
    ) -> Result<Response> {
        record_discarded(&turn.agg.usage);
        emit_discarded_audit(&self.executor.semantic_name, &turn.agg.usage);
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
        let decision = self.executor_decision("REDO continuation");
        driver.decide(decision.clone()).await?;
        driver.call_model(redo, decision).await
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
        let decision = Decision::new(
            self.advisor.semantic_name.clone(),
            Some("advisor gate: review consult".to_string()),
            false,
        );
        let started = Instant::now();
        let reply = match driver.call_model(consult_request, decision).await {
            Ok(response) => response.llm_response.into_agg().await.map_err(|source| {
                LibsyError::client_call(self.advisor.semantic_name.clone(), source)
            }),
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

    async fn route(self: Arc<Self>, driver: Driver, request: Request) -> Result<Response> {
        let scope = budget_scope(&request);
        let session_final = request
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.session_final)
            == Some(true);
        let result = self.route_inner(&driver, request, &scope).await;
        if session_final {
            self.evict_scope(&scope);
        }
        result
    }
}

/// Advisor verdict on one terminal turn.
enum Verdict {
    Approve,
    Redo { plan: String },
}

/// Outcome of one consult; `Failed` = fail-open error or unparseable reply.
enum ConsultOutcome {
    Approve,
    Redo { plan: String },
    Failed,
}

// ── Budget scope ────────────────────────────────────────────────────────────

/// Resolves the review budget scope: the benchmark harness header wins (it is
/// stamped on every request of one evaluation, sub-agents included), then the
/// host-resolved session id, then one shared instance scope.
fn budget_scope(request: &Request) -> ScopeKey {
    let metadata = request.metadata.as_ref();
    if let Some(value) = metadata
        .and_then(|metadata| metadata.http_headers.as_ref())
        .and_then(|headers| headers.get(BENCH_SESSION_HEADER))
        .and_then(|value| value.to_str().ok())
        && !value.is_empty()
    {
        return ScopeKey::Client(value.to_string());
    }
    if let Some(id) = metadata.and_then(|metadata| metadata.session_id.as_deref())
        && !id.is_empty()
    {
        return ScopeKey::Session(id.to_string());
    }
    ScopeKey::Instance
}

/// Latches the stall checkpoint per conversation: hash of the first user
/// message's text, which is constant across a session's turns.
fn stall_key(request: &Request) -> u64 {
    let text = request
        .llm_request
        .messages
        .iter()
        .find(|message| message.role == Role::User)
        .and_then(|message| message.text_content("\n"))
        .unwrap_or_default();
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

// ── Turn buffering and replay ───────────────────────────────────────────────

/// One fully generated executor turn held while the gate decides.
struct GatedTurn {
    /// Buffered provider events for streamed turns, preservation included, so
    /// replay re-emits them verbatim (signed thinking and provider extensions
    /// survive; folding to an aggregate and re-synthesizing would drop them).
    events: Option<Vec<LlmResponseStreamEvent>>,
    /// Folded view for detection, the review tail, the REDO echo, and
    /// discarded-turn usage. For buffered turns this is the original
    /// response, its own preservation intact.
    agg: AggLlmResponse,
    metadata: Option<switchyard_protocol::Metadata>,
}

impl GatedTurn {
    /// Releases the turn to the client: streamed turns replay their buffered
    /// events verbatim, buffered turns return the original aggregate.
    fn into_response(self) -> Response {
        let llm_response = match self.events {
            Some(events) => {
                LlmResponse::Stream(Box::pin(futures::stream::iter(events.into_iter().map(Ok))))
            }
            None => LlmResponse::Agg(self.agg),
        };
        Response {
            llm_response,
            metadata: self.metadata,
        }
    }
}

/// Consumes the executor response to completion. Mid-stream failures — item
/// errors and in-band error chunks — become typed client-call errors exactly
/// as [`LlmResponse::into_agg`] maps them; the client saw nothing yet, so the
/// turn fails whole.
async fn buffer_turn(executor: &str, response: Response) -> Result<GatedTurn> {
    let metadata = response.metadata;
    match response.llm_response {
        LlmResponse::Agg(agg) => Ok(GatedTurn {
            events: None,
            agg,
            metadata,
        }),
        LlmResponse::Stream(mut stream) => {
            let mut events = Vec::new();
            let mut accumulator = ResponseAccumulator::new();
            while let Some(item) = stream.next().await {
                let event =
                    item.map_err(|source| LibsyError::client_call(executor.to_string(), source))?;
                for chunk in event.normalized() {
                    let failure = match chunk {
                        LlmResponseChunk::DecodeError { message } => {
                            Some(LlmClientError::ResponseTranslation(message.clone()))
                        }
                        LlmResponseChunk::StreamError { message } => {
                            Some(LlmClientError::UpstreamHttp {
                                status: 502,
                                body: message.clone(),
                            })
                        }
                        chunk => {
                            accumulator.push(chunk.clone());
                            None
                        }
                    };
                    if let Some(source) = failure {
                        return Err(LibsyError::client_call(executor.to_string(), source));
                    }
                }
                events.push(event);
            }
            Ok(GatedTurn {
                events: Some(events),
                agg: accumulator.finish(),
                metadata,
            })
        }
    }
}

// ── Detection over the folded turn ──────────────────────────────────────────

/// Whether the turn carries tool use on either signal: a `ToolUse` stop
/// reason, or any tool-call block (some OSS servers mislabel tool-call turns
/// as an ordinary stop, so block presence wins).
fn has_tool_use(agg: &AggLlmResponse) -> bool {
    agg.outputs.iter().any(|output| {
        output.stop_reason == Some(StopReason::ToolUse)
            || output
                .content
                .iter()
                .any(|block| matches!(block, ContentBlock::ToolCall(_)))
    })
}

/// The turn's visible text: all text blocks joined; empty means none.
fn visible_text(agg: &AggLlmResponse) -> Option<String> {
    let text: Vec<&str> = agg
        .outputs
        .iter()
        .flat_map(|output| output.content.iter())
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    if text.is_empty() {
        return None;
    }
    let joined = text.join("\n");
    if joined.is_empty() {
        None
    } else {
        Some(joined)
    }
}

/// The turn's internal reasoning, the review evidence of last resort.
fn reasoning_text(agg: &AggLlmResponse) -> Option<String> {
    let text: Vec<&str> = agg
        .outputs
        .iter()
        .flat_map(|output| output.content.iter())
        .filter_map(|block| match block {
            ContentBlock::Reasoning { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    if text.is_empty() {
        return None;
    }
    let joined = text.join("\n");
    if joined.is_empty() {
        None
    } else {
        Some(joined)
    }
}

/// Tool results carried by the conversation so far (both wires normalize
/// tool results into `ContentBlock::ToolResult`).
fn count_tool_results(messages: &[Message]) -> u32 {
    let count = messages
        .iter()
        .flat_map(|message| message.content.iter())
        .filter(|block| matches!(block, ContentBlock::ToolResult(_)))
        .count();
    u32::try_from(count).unwrap_or(u32::MAX)
}

/// Assistant turns already in the request — the stall checkpoint's clock.
fn assistant_turns(messages: &[Message]) -> u32 {
    let count = messages
        .iter()
        .filter(|message| message.role == Role::Assistant)
        .count();
    u32::try_from(count).unwrap_or(u32::MAX)
}

// ── Transcript and verdict ──────────────────────────────────────────────────

/// Serializes the conversation for the advisor. The JSON body is capped with
/// a middle drop — the head keeps the task statement, the tail keeps the
/// recent evidence a completeness review is about — while the terminal turn
/// is appended uncapped.
fn review_transcript(messages: &[Message], review_tail: Option<&str>, cap: usize) -> String {
    let text = serde_json::to_string(messages).unwrap_or_default();
    let text = middle_drop(text, cap);
    format!(
        "Conversation so far (JSON):\n\n{text}\n\nThe executor's latest turn (a plan, or its claim the task is done):\n{}",
        review_tail.unwrap_or(NO_TEXT_PLACEHOLDER)
    )
}

/// Keeps the first `cap / 4` and last `cap - cap / 4` characters of an
/// over-cap string, splicing [`TRUNCATION_MARKER`] between them. Boundaries
/// are computed per character so multi-byte text never splits a code point.
fn middle_drop(text: String, cap: usize) -> String {
    let total = text.chars().count();
    if total <= cap {
        return text;
    }
    let head_chars = cap / 4;
    let tail_chars = cap - head_chars;
    let head_end = text
        .char_indices()
        .nth(head_chars)
        .map(|(index, _)| index)
        .unwrap_or(text.len());
    let tail_start = text
        .char_indices()
        .nth(total - tail_chars)
        .map(|(index, _)| index)
        .unwrap_or(0);
    format!(
        "{}{TRUNCATION_MARKER}{}",
        &text[..head_end],
        &text[tail_start..]
    )
}

/// Text of the advisor's reply: all text blocks across outputs, trimmed.
fn advisor_reply_text(agg: &AggLlmResponse) -> String {
    visible_text(agg).unwrap_or_default().trim().to_string()
}

/// Parses the anchored verdict. A REDO's plan is the remainder after the
/// verdict token with leading separators stripped; an empty plan falls back
/// to the whole reply so the executor still gets actionable feedback. `None`
/// means the reply led with prose and cannot be trusted as a verdict.
fn parse_verdict(verdict_re: &regex::Regex, reply: &str) -> Option<Verdict> {
    let reply = reply.trim();
    let captures = verdict_re.captures(reply)?;
    let token = captures.get(1)?;
    if token.as_str().eq_ignore_ascii_case("APPROVE") {
        return Some(Verdict::Approve);
    }
    let plan = reply[token.end()..]
        .trim_start_matches([' ', '*', '_', ':', '\n', '-'])
        .trim();
    let plan = if plan.is_empty() { reply } else { plan };
    Some(Verdict::Redo {
        plan: plan.to_string(),
    })
}

// ── Accounting ──────────────────────────────────────────────────────────────

/// Inclusive prompt tokens: non-cached input plus both cache buckets, the
/// same fold the routing log uses, so advisor and executor rows reconcile.
fn inclusive_prompt_tokens(usage: &Usage) -> u64 {
    usage
        .input_tokens
        .unwrap_or(0)
        .saturating_add(usage.cached_input_tokens().unwrap_or(0))
        .saturating_add(usage.cache_creation_input_tokens().unwrap_or(0))
}

fn record_review(verdict: &'static str, trigger: &'static str) {
    observability::meter()
        .u64_counter("switchyard.advisor_gate.reviews")
        .build()
        .add(
            1,
            &[
                KeyValue::new("verdict", verdict),
                KeyValue::new("trigger", trigger),
            ],
        );
}

fn record_consult_failure(reason: &'static str) {
    observability::meter()
        .u64_counter("switchyard.advisor_gate.consult_failures")
        .build()
        .add(1, &[KeyValue::new("reason", reason)]);
}

/// Counts a REDO-discarded executor turn and its tokens; the client never
/// sees the turn, so the host's terminal usage accounting never prices it.
fn record_discarded(usage: &Usage) {
    let meter = observability::meter();
    meter
        .u64_counter("switchyard.advisor_gate.discarded_turns")
        .build()
        .add(1, &[]);
    let tokens = meter
        .u64_counter("switchyard.advisor_gate.discarded_tokens")
        .build();
    for (kind, value) in [
        ("input", usage.input_tokens.unwrap_or(0)),
        ("cached", usage.cached_input_tokens().unwrap_or(0)),
        (
            "cache_creation",
            usage.cache_creation_input_tokens().unwrap_or(0),
        ),
        ("output", usage.output_tokens.unwrap_or(0)),
    ] {
        if value > 0 {
            tokens.add(value, &[KeyValue::new("kind", kind)]);
        }
    }
}

/// One review consult's audit payload.
struct ReviewAudit<'a> {
    verdict: &'static str,
    error: Option<String>,
    latency_ms: f64,
    reply_head: Option<String>,
    usage: Option<&'a Usage>,
}

/// Emits the one-line sorted-key JSON audit record benchmark tooling greps
/// for (`advisor_review=`).
fn emit_review_audit(audit: ReviewAudit<'_>) {
    let mut payload = serde_json::Map::new();
    payload.insert("advisor_review".to_string(), true.into());
    payload.insert(
        "latency_ms".to_string(),
        ((audit.latency_ms * 10.0).round() / 10.0).into(),
    );
    payload.insert("verdict".to_string(), audit.verdict.into());
    if let Some(error) = audit.error {
        payload.insert("error".to_string(), error.into());
    }
    if let Some(head) = audit.reply_head
        && !head.is_empty()
    {
        payload.insert("reply_head".to_string(), head.into());
    }
    if let Some(usage) = audit.usage {
        payload.insert(
            "prompt_tokens".to_string(),
            inclusive_prompt_tokens(usage).into(),
        );
        payload.insert(
            "completion_tokens".to_string(),
            usage.output_tokens.unwrap_or(0).into(),
        );
        let cached = usage.cached_input_tokens().unwrap_or(0);
        if cached > 0 {
            payload.insert("cached_tokens".to_string(), cached.into());
        }
        let creation = usage.cache_creation_input_tokens().unwrap_or(0);
        if creation > 0 {
            payload.insert("cache_creation_tokens".to_string(), creation.into());
        }
    }
    tracing::info!(
        target: "libsy",
        "advisor_review={}",
        serde_json::Value::Object(payload)
    );
}

/// Emits the discarded-turn audit record (`advisor_discarded=`), the gate's
/// own accounting for a turn no host-side observer can price.
fn emit_discarded_audit(model: &str, usage: &Usage) {
    let mut payload = serde_json::Map::new();
    payload.insert("advisor_discarded".to_string(), true.into());
    payload.insert("model".to_string(), model.into());
    payload.insert(
        "prompt_tokens".to_string(),
        inclusive_prompt_tokens(usage).into(),
    );
    payload.insert(
        "cached_tokens".to_string(),
        usage.cached_input_tokens().unwrap_or(0).into(),
    );
    payload.insert(
        "cache_creation_tokens".to_string(),
        usage.cache_creation_input_tokens().unwrap_or(0).into(),
    );
    payload.insert(
        "completion_tokens".to_string(),
        usage.output_tokens.unwrap_or(0).into(),
    );
    tracing::info!(
        target: "libsy",
        "advisor_discarded={}",
        serde_json::Value::Object(payload)
    );
}

fn algorithm_error(message: impl Into<String>) -> LibsyError {
    LibsyError::AlgorithmError {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use switchyard_protocol::{ResponseOutput, ToolCall, ToolResult, completion_text};

    use super::*;
    use crate::core::testing::{reply, test_drive};

    const EXECUTOR: &str = "executor";
    const ADVISOR: &str = "advisor";

    fn target(name: &str) -> LlmTarget {
        LlmTarget {
            semantic_name: name.to_string(),
        }
    }

    fn gate(config: AdvisorGateConfig) -> Arc<dyn Algorithm> {
        Arc::new(
            AdvisorGate::new(target(EXECUTOR), target(ADVISOR), config)
                .expect("test config is valid"),
        )
    }

    fn request(messages: Vec<Message>) -> Request {
        Request {
            llm_request: LlmRequest {
                model: Some("gated".to_string()),
                messages,
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: None,
        }
    }

    fn task_request() -> Request {
        request(vec![Message::text(Role::User, "build X")])
    }

    fn with_bench_header(mut request: Request, id: &str) -> Request {
        let mut headers = http::HeaderMap::new();
        headers.insert(BENCH_SESSION_HEADER, id.parse().expect("header value"));
        let mut metadata = request.metadata.unwrap_or_default();
        metadata.http_headers = Some(headers);
        request.metadata = Some(metadata);
        request
    }

    fn with_session_id(mut request: Request, id: &str) -> Request {
        let mut metadata = request.metadata.unwrap_or_default();
        metadata.session_id = Some(id.to_string());
        request.metadata = Some(metadata);
        request
    }

    fn tool_call_turn() -> Response {
        Response {
            llm_response: LlmResponse::Agg(AggLlmResponse {
                outputs: vec![ResponseOutput {
                    role: Role::Assistant,
                    content: vec![ContentBlock::ToolCall(ToolCall {
                        id: "t1".to_string(),
                        name: "bash".to_string(),
                        arguments: serde_json::json!({}),
                    })],
                    stop_reason: None,
                }],
                ..AggLlmResponse::default()
            }),
            metadata: None,
        }
    }

    fn tool_use_stop_turn() -> Response {
        Response {
            llm_response: LlmResponse::Agg(AggLlmResponse {
                outputs: vec![ResponseOutput {
                    role: Role::Assistant,
                    content: vec![ContentBlock::Text {
                        text: "calling a tool".to_string(),
                    }],
                    stop_reason: Some(StopReason::ToolUse),
                }],
                ..AggLlmResponse::default()
            }),
            metadata: None,
        }
    }

    fn reasoning_only_turn() -> Response {
        Response {
            llm_response: LlmResponse::Agg(AggLlmResponse {
                outputs: vec![ResponseOutput {
                    role: Role::Assistant,
                    content: vec![ContentBlock::Reasoning {
                        text: "thinking about it".to_string(),
                        signature: None,
                    }],
                    stop_reason: None,
                }],
                ..AggLlmResponse::default()
            }),
            metadata: None,
        }
    }

    fn empty_turn() -> Response {
        Response {
            llm_response: LlmResponse::Agg(AggLlmResponse {
                outputs: vec![ResponseOutput {
                    role: Role::Assistant,
                    content: Vec::new(),
                    stop_reason: None,
                }],
                ..AggLlmResponse::default()
            }),
            metadata: None,
        }
    }

    fn streamed(events: Vec<LlmResponseStreamEvent>) -> Response {
        Response {
            llm_response: LlmResponse::Stream(Box::pin(futures::stream::iter(
                events.into_iter().map(Ok),
            ))),
            metadata: None,
        }
    }

    fn text_stream_events(text: &str) -> Vec<LlmResponseStreamEvent> {
        vec![
            LlmResponseStreamEvent::preserved(
                "anthropic_messages",
                serde_json::json!({"type": "message_start"}),
                vec![LlmResponseChunk::MessageStart {
                    id: Some("m1".to_string()),
                    model: Some("exec-upstream".to_string()),
                }],
            ),
            LlmResponseStreamEvent::preserved(
                "anthropic_messages",
                serde_json::json!({"type": "content_block_delta", "text": text}),
                vec![LlmResponseChunk::TextDelta {
                    index: 0,
                    text: text.to_string(),
                }],
            ),
            LlmResponseStreamEvent::preserved(
                "anthropic_messages",
                serde_json::json!({"type": "message_stop"}),
                vec![LlmResponseChunk::MessageStop {
                    reason: Some("end_turn".to_string()),
                }],
            ),
        ]
    }

    /// Serve that answers the advisor with a fixed verdict and the executor
    /// from a per-call script, recording every call.
    struct Script {
        calls: Arc<parking_lot::Mutex<Vec<(String, Request)>>>,
        executor_calls: Arc<AtomicUsize>,
    }

    impl Script {
        fn new() -> Self {
            Self {
                calls: Arc::new(parking_lot::Mutex::new(Vec::new())),
                executor_calls: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn models(&self) -> Vec<String> {
            self.calls
                .lock()
                .iter()
                .map(|(model, _)| model.clone())
                .collect()
        }

        fn advisor_consults(&self) -> usize {
            self.calls
                .lock()
                .iter()
                .filter(|(model, _)| model == ADVISOR)
                .count()
        }

        fn call(&self, index: usize) -> Request {
            self.calls.lock()[index].1.clone()
        }

        /// Serve executor turns from `executor` (indexed per executor call)
        /// and advisor consults with `verdict`.
        fn serve(
            &self,
            verdict: &str,
            executor: impl Fn(usize) -> Response + Send + Sync + 'static,
        ) -> impl Fn(
            Decision,
            Request,
        ) -> futures::future::BoxFuture<
            'static,
            std::result::Result<Response, LlmClientError>,
        > + Send
        + Sync
        + 'static {
            let calls = Arc::clone(&self.calls);
            let executor_calls = Arc::clone(&self.executor_calls);
            let verdict = verdict.to_string();
            let executor = Arc::new(executor);
            move |decision: Decision, request: Request| {
                let calls = Arc::clone(&calls);
                let executor_calls = Arc::clone(&executor_calls);
                let verdict = verdict.clone();
                let executor = Arc::clone(&executor);
                Box::pin(async move {
                    let model = decision.selected_model_id().to_string();
                    calls.lock().push((model.clone(), request));
                    if model == ADVISOR {
                        Ok(reply(verdict))
                    } else {
                        let index = executor_calls.fetch_add(1, Ordering::SeqCst);
                        Ok(executor(index))
                    }
                })
            }
        }
    }

    async fn agg_of(response: Response) -> AggLlmResponse {
        response
            .llm_response
            .into_agg()
            .await
            .expect("test response aggregates")
    }

    // ── Gate behavior ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn tool_call_turn_replays_without_review() {
        for turn in [tool_call_turn(), tool_use_stop_turn()] {
            let script = Script::new();
            let gate = gate(AdvisorGateConfig::default());
            let serve = script.serve("APPROVE", {
                let turn = parking_lot::Mutex::new(Some(turn));
                move |_| turn.lock().take().expect("one executor call")
            });
            let (_, response) = test_drive(gate, task_request(), serve)
                .await
                .expect("routes");
            assert_eq!(script.models(), vec![EXECUTOR.to_string()]);
            assert!(has_tool_use(&agg_of(response).await));
        }
    }

    #[tokio::test]
    async fn approved_terminal_turn_returns_buffered_body() {
        let script = Script::new();
        let gate = gate(AdvisorGateConfig::default());
        let serve = script.serve("APPROVE", |_| reply("all done"));
        let (trace, response) = test_drive(gate, task_request(), serve)
            .await
            .expect("routes");
        assert_eq!(
            script.models(),
            vec![EXECUTOR.to_string(), ADVISOR.to_string()]
        );
        assert_eq!(completion_text(&agg_of(response).await), "all done");
        // The published trace ends on the executor so hosts attribute the
        // served model correctly.
        let last = trace.last().expect("decision published");
        assert_eq!(last.selected_model_id(), EXECUTOR);
        assert!(last.is_answer_call());
    }

    #[tokio::test]
    async fn advisor_consult_is_not_an_answer_call() {
        let script = Script::new();
        let gate = gate(AdvisorGateConfig::default());
        let consult_shape = Arc::new(parking_lot::Mutex::new(None));
        let shape = Arc::clone(&consult_shape);
        let calls = Arc::clone(&script.calls);
        let serve = move |decision: Decision, request: Request| {
            let shape = Arc::clone(&shape);
            let calls = Arc::clone(&calls);
            Box::pin(async move {
                calls
                    .lock()
                    .push((decision.selected_model_id().to_string(), request));
                if decision.selected_model_id() == ADVISOR {
                    *shape.lock() = Some(decision.is_answer_call());
                    Ok(reply("APPROVE"))
                } else {
                    Ok(reply("done"))
                }
            })
                as futures::future::BoxFuture<
                    'static,
                    std::result::Result<Response, LlmClientError>,
                >
        };
        test_drive(gate, task_request(), serve)
            .await
            .expect("routes");
        assert_eq!(*consult_shape.lock(), Some(false));
    }

    #[tokio::test]
    async fn redo_appends_echo_and_feedback_then_reinvokes() {
        let script = Script::new();
        let gate = gate(AdvisorGateConfig::default());
        let serve = script.serve("REDO: run the tests", |index| {
            if index == 0 {
                reply("first attempt")
            } else {
                reply("continued")
            }
        });
        let (_, response) = test_drive(gate, task_request(), serve)
            .await
            .expect("routes");
        assert_eq!(
            script.models(),
            vec![
                EXECUTOR.to_string(),
                ADVISOR.to_string(),
                EXECUTOR.to_string()
            ]
        );
        assert_eq!(completion_text(&agg_of(response).await), "continued");
        let redo = script.call(2);
        let messages = &redo.llm_request.messages;
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1].role, Role::Assistant);
        assert_eq!(
            messages[1].text_content("\n").as_deref(),
            Some("first attempt")
        );
        assert_eq!(messages[2].role, Role::User);
        let feedback = messages[2].text_content("\n").expect("feedback text");
        assert!(feedback.starts_with(REDO_FEEDBACK_PREFIX));
        assert!(feedback.ends_with("run the tests"));
        assert!(redo.llm_request.preservation.requests.is_empty());
    }

    #[tokio::test]
    async fn budget_consumed_once_per_scope() {
        let script = Script::new();
        let gate = gate(AdvisorGateConfig::default());
        let serve = script.serve("APPROVE", |_| reply("done"));
        test_drive(Arc::clone(&gate), task_request(), serve)
            .await
            .expect("first run");
        let serve = script.serve("APPROVE", |_| reply("done again"));
        let (_, response) = test_drive(gate, task_request(), serve)
            .await
            .expect("second run");
        // Headerless requests share the instance scope: exactly one consult.
        assert_eq!(script.advisor_consults(), 1);
        assert_eq!(completion_text(&agg_of(response).await), "done again");
    }

    #[tokio::test]
    async fn budget_keyed_by_bench_header_not_conversation() {
        let script = Script::new();
        let gate = gate(AdvisorGateConfig::default());
        for turn in ["build X", "now build Y", "and Z"] {
            let serve = script.serve("APPROVE", |_| reply("done"));
            let request =
                with_bench_header(request(vec![Message::text(Role::User, turn)]), "eval-1");
            test_drive(Arc::clone(&gate), request, serve)
                .await
                .expect("routes");
        }
        assert_eq!(script.advisor_consults(), 1);
    }

    #[tokio::test]
    async fn scope_precedence_header_over_session_id() {
        let script = Script::new();
        let gate = gate(AdvisorGateConfig::default());
        // Same bench header, different host session ids: one scope.
        for session in ["s1", "s2"] {
            let serve = script.serve("APPROVE", |_| reply("done"));
            let request = with_bench_header(with_session_id(task_request(), session), "eval-1");
            test_drive(Arc::clone(&gate), request, serve)
                .await
                .expect("routes");
        }
        assert_eq!(script.advisor_consults(), 1);
        // Distinct session ids without the header: distinct scopes.
        for session in ["s3", "s4"] {
            let serve = script.serve("APPROVE", |_| reply("done"));
            test_drive(
                Arc::clone(&gate),
                with_session_id(task_request(), session),
                serve,
            )
            .await
            .expect("routes");
        }
        assert_eq!(script.advisor_consults(), 3);
    }

    #[tokio::test]
    async fn max_reviews_two_reviews_then_passthrough() {
        let script = Script::new();
        let gate = gate(AdvisorGateConfig {
            max_reviews: 2,
            ..AdvisorGateConfig::default()
        });
        for _ in 0..3 {
            let serve = script.serve("APPROVE", |_| reply("done"));
            test_drive(Arc::clone(&gate), task_request(), serve)
                .await
                .expect("routes");
        }
        assert_eq!(script.advisor_consults(), 2);
    }

    #[tokio::test]
    async fn exhausted_scope_passes_live_stream_through() {
        let script = Script::new();
        let gate = gate(AdvisorGateConfig::default());
        let serve = script.serve("APPROVE", |_| reply("done"));
        test_drive(Arc::clone(&gate), task_request(), serve)
            .await
            .expect("spends budget");
        // Post-budget turns pass through as the live stream, events verbatim.
        let events = text_stream_events("streamed continuation");
        let expected = serde_json::to_value(&events).expect("events serialize");
        let serve = script.serve("APPROVE", {
            let events = parking_lot::Mutex::new(Some(events));
            move |_| streamed(events.lock().take().expect("one executor call"))
        });
        let (_, response) = test_drive(gate, task_request(), serve)
            .await
            .expect("routes");
        let LlmResponse::Stream(stream) = response.llm_response else {
            panic!("expected a live stream");
        };
        let replayed: Vec<LlmResponseStreamEvent> = stream
            .map(|item| item.expect("stream item"))
            .collect()
            .await;
        assert_eq!(
            serde_json::to_value(&replayed).expect("serialize"),
            expected
        );
        assert_eq!(script.advisor_consults(), 1);
    }

    // ── Failure paths ───────────────────────────────────────────────────────

    fn failing_advisor(
        script: &Script,
        executor_reply: &'static str,
    ) -> impl Fn(
        Decision,
        Request,
    ) -> futures::future::BoxFuture<
        'static,
        std::result::Result<Response, LlmClientError>,
    > + Send
    + Sync
    + 'static {
        let calls = Arc::clone(&script.calls);
        move |decision: Decision, request: Request| {
            let calls = Arc::clone(&calls);
            Box::pin(async move {
                let model = decision.selected_model_id().to_string();
                calls.lock().push((model.clone(), request));
                if model == ADVISOR {
                    Err(LlmClientError::General("advisor down".to_string()))
                } else {
                    Ok(reply(executor_reply))
                }
            })
        }
    }

    #[tokio::test]
    async fn fail_open_returns_turn_refunds_and_caps_failures() {
        let script = Script::new();
        let gate = gate(AdvisorGateConfig::default());
        // Three failed consults: each returns the turn and refunds the budget.
        for _ in 0..3 {
            let (_, response) = test_drive(
                Arc::clone(&gate),
                task_request(),
                failing_advisor(&script, "done"),
            )
            .await
            .expect("fail-open run");
            assert_eq!(completion_text(&agg_of(response).await), "done");
        }
        assert_eq!(script.advisor_consults(), 3);
        // The failure cap now stops consulting entirely.
        test_drive(
            Arc::clone(&gate),
            task_request(),
            failing_advisor(&script, "done"),
        )
        .await
        .expect("passthrough run");
        assert_eq!(script.advisor_consults(), 3);
        // A recovered advisor is never consulted again in this scope.
        let serve = script.serve("APPROVE", |_| reply("done"));
        test_drive(gate, task_request(), serve)
            .await
            .expect("still passthrough");
        assert_eq!(script.advisor_consults(), 3);
    }

    #[tokio::test]
    async fn fail_closed_propagates_refunds_and_counts() {
        let script = Script::new();
        let gate = gate(AdvisorGateConfig {
            fail_open: false,
            ..AdvisorGateConfig::default()
        });
        for _ in 0..3 {
            let error = match test_drive(
                Arc::clone(&gate),
                task_request(),
                failing_advisor(&script, "done"),
            )
            .await
            {
                Err(error) => error,
                Ok(_) => panic!("fail-closed surfaces the advisor error"),
            };
            // Wrapped as an algorithm failure so the host renders a 5xx, not
            // the advisor's own (possibly context-window-shaped) client error.
            assert!(matches!(error, LibsyError::AlgorithmError { .. }));
            assert!(error.to_string().contains("advisor consult failed"));
        }
        assert_eq!(script.advisor_consults(), 3);
        // The failure cap bounds fail-closed too: the scope stops consulting
        // and the executor turn flows again.
        let (_, response) = test_drive(gate, task_request(), failing_advisor(&script, "recovered"))
            .await
            .expect("post-cap passthrough");
        assert_eq!(script.advisor_consults(), 3);
        assert_eq!(completion_text(&agg_of(response).await), "recovered");
    }

    #[tokio::test]
    async fn unparseable_verdict_refunds_and_approves() {
        let script = Script::new();
        let gate = gate(AdvisorGateConfig::default());
        let serve = script.serve("I cannot approve this — REDO: run the tests", |_| {
            reply("done")
        });
        let (_, response) = test_drive(Arc::clone(&gate), task_request(), serve)
            .await
            .expect("unparseable run");
        assert_eq!(completion_text(&agg_of(response).await), "done");
        // The refunded budget admits another review.
        let serve = script.serve("APPROVE", |_| reply("done"));
        test_drive(gate, task_request(), serve)
            .await
            .expect("second run");
        assert_eq!(script.advisor_consults(), 2);
    }

    #[tokio::test]
    async fn context_window_error_propagates() {
        let gate = gate(AdvisorGateConfig::default());
        let serve = |_decision: Decision, _request: Request| async move {
            Err(LlmClientError::ContextWindowExceeded {
                model: "exec-upstream".to_string(),
                message: "prompt is too long".to_string(),
            })
        };
        let error = match test_drive(gate, task_request(), serve).await {
            Err(error) => error,
            Ok(_) => panic!("context-window error propagates"),
        };
        assert!(matches!(
            error,
            LibsyError::ClientCall {
                source: LlmClientError::ContextWindowExceeded { .. },
                ..
            }
        ));
    }

    #[tokio::test]
    async fn mid_stream_error_propagates_while_buffering() {
        let gate = gate(AdvisorGateConfig::default());
        let serve = |_decision: Decision, _request: Request| async move {
            Ok(streamed(vec![
                LlmResponseStreamEvent::new(vec![LlmResponseChunk::TextDelta {
                    index: 0,
                    text: "partial".to_string(),
                }]),
                LlmResponseStreamEvent::new(vec![LlmResponseChunk::StreamError {
                    message: "upstream reset".to_string(),
                }]),
            ]))
        };
        let error = match test_drive(gate, task_request(), serve).await {
            Err(error) => error,
            Ok(_) => panic!("mid-stream error propagates"),
        };
        assert!(matches!(
            error,
            LibsyError::ClientCall {
                source: LlmClientError::UpstreamHttp { status: 502, .. },
                ..
            }
        ));
    }

    // ── Streaming ───────────────────────────────────────────────────────────

    #[tokio::test]
    async fn streamed_approval_replays_preserved_events_verbatim() {
        let script = Script::new();
        let gate = gate(AdvisorGateConfig::default());
        let events = text_stream_events("the answer");
        let expected = serde_json::to_value(&events).expect("events serialize");
        let serve = script.serve("APPROVE", {
            let events = parking_lot::Mutex::new(Some(events));
            move |_| streamed(events.lock().take().expect("one executor call"))
        });
        let (_, response) = test_drive(gate, task_request(), serve)
            .await
            .expect("routes");
        assert_eq!(script.advisor_consults(), 1);
        let LlmResponse::Stream(stream) = response.llm_response else {
            panic!("expected replayed stream");
        };
        let replayed: Vec<LlmResponseStreamEvent> = stream
            .map(|item| item.expect("stream item"))
            .collect()
            .await;
        assert_eq!(
            serde_json::to_value(&replayed).expect("serialize"),
            expected
        );
    }

    #[tokio::test]
    async fn streamed_tool_call_turn_replays_without_review() {
        let script = Script::new();
        let gate = gate(AdvisorGateConfig::default());
        let events = vec![LlmResponseStreamEvent::new(vec![
            LlmResponseChunk::ToolCallDelta {
                index: 0,
                id: Some("t1".to_string()),
                name: Some("bash".to_string()),
                arguments_delta: Some("{}".to_string()),
            },
        ])];
        let serve = script.serve("APPROVE", {
            let events = parking_lot::Mutex::new(Some(events));
            move |_| streamed(events.lock().take().expect("one executor call"))
        });
        test_drive(gate, task_request(), serve)
            .await
            .expect("routes");
        assert_eq!(script.models(), vec![EXECUTOR.to_string()]);
    }

    // ── Triggers ────────────────────────────────────────────────────────────

    fn pattern_config() -> AdvisorGateConfig {
        AdvisorGateConfig {
            gate_trigger: GateTrigger::Pattern(r#"task_complete["\s>:]*true"#.to_string()),
            ..AdvisorGateConfig::default()
        }
    }

    #[tokio::test]
    async fn pattern_trigger_gates_matching_text_only() {
        let script = Script::new();
        let gate = gate(pattern_config());
        // Non-matching turns pass through without a consult.
        let serve = script.serve("APPROVE", |_| reply("still working"));
        test_drive(Arc::clone(&gate), task_request(), serve)
            .await
            .expect("routes");
        assert_eq!(script.advisor_consults(), 0);
        // The declared completion gates.
        let serve = script.serve("APPROVE", |_| reply("task_complete: true"));
        test_drive(gate, task_request(), serve)
            .await
            .expect("routes");
        assert_eq!(script.advisor_consults(), 1);
    }

    #[tokio::test]
    async fn pattern_trigger_matches_on_tool_call_turns() {
        // The pattern trigger reads text only; tool use does not exempt a turn.
        let script = Script::new();
        let gate = gate(pattern_config());
        let turn = Response {
            llm_response: LlmResponse::Agg(AggLlmResponse {
                outputs: vec![ResponseOutput {
                    role: Role::Assistant,
                    content: vec![
                        ContentBlock::Text {
                            text: "task_complete: true".to_string(),
                        },
                        ContentBlock::ToolCall(ToolCall {
                            id: "t1".to_string(),
                            name: "bash".to_string(),
                            arguments: serde_json::json!({}),
                        }),
                    ],
                    stop_reason: Some(StopReason::ToolUse),
                }],
                ..AggLlmResponse::default()
            }),
            metadata: None,
        };
        let serve = script.serve("APPROVE", {
            let turn = parking_lot::Mutex::new(Some(turn));
            move |_| turn.lock().take().expect("one executor call")
        });
        test_drive(gate, task_request(), serve)
            .await
            .expect("routes");
        assert_eq!(script.advisor_consults(), 1);
    }

    #[tokio::test]
    async fn min_tool_results_defers_gate() {
        let script = Script::new();
        let gate = gate(AdvisorGateConfig {
            gate_min_tool_results: 1,
            ..AdvisorGateConfig::default()
        });
        // Terminal turn before any tool result: passes through unreviewed.
        let serve = script.serve("APPROVE", |_| reply("plan: do X"));
        test_drive(Arc::clone(&gate), task_request(), serve)
            .await
            .expect("routes");
        assert_eq!(script.advisor_consults(), 0);
        // Once the conversation carries a tool result, the gate fires.
        let with_result = request(vec![
            Message::text(Role::User, "build X"),
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult(ToolResult {
                    tool_call_id: "t1".to_string(),
                    content: vec![ContentBlock::Text {
                        text: "ok".to_string(),
                    }],
                    is_error: None,
                })],
            },
        ]);
        let serve = script.serve("APPROVE", |_| reply("done"));
        test_drive(gate, with_result, serve).await.expect("routes");
        assert_eq!(script.advisor_consults(), 1);
    }

    #[tokio::test]
    async fn stall_checkpoint_reviews_mid_task_once() {
        let script = Script::new();
        let gate = gate(AdvisorGateConfig {
            gate_stall_turns: 2,
            max_reviews: 2,
            ..AdvisorGateConfig::default()
        });
        let grinding = || {
            request(vec![
                Message::text(Role::User, "build X"),
                Message::text(Role::Assistant, "step 1"),
                Message::text(Role::Assistant, "step 2"),
            ])
        };
        // A tool-call turn is not terminal, but the stall checkpoint reviews it.
        let serve = script.serve("APPROVE", {
            let turn = parking_lot::Mutex::new(Some(tool_call_turn()));
            move |_| turn.lock().take().expect("one executor call")
        });
        test_drive(Arc::clone(&gate), grinding(), serve)
            .await
            .expect("routes");
        assert_eq!(script.advisor_consults(), 1);
        // The latch keeps the same conversation from stalling twice.
        let serve = script.serve("APPROVE", {
            let turn = parking_lot::Mutex::new(Some(tool_call_turn()));
            move |_| turn.lock().take().expect("one executor call")
        });
        test_drive(gate, grinding(), serve).await.expect("routes");
        assert_eq!(script.advisor_consults(), 1);
    }

    #[tokio::test]
    async fn simultaneous_trigger_does_not_latch_stall() {
        let script = Script::new();
        let gate = gate(AdvisorGateConfig {
            gate_stall_turns: 1,
            max_reviews: 2,
            ..AdvisorGateConfig::default()
        });
        let conversation = || {
            request(vec![
                Message::text(Role::User, "build X"),
                Message::text(Role::Assistant, "step 1"),
            ])
        };
        // Terminal turn and stall coincide: the trigger review runs, the
        // stall does not latch.
        let serve = script.serve("APPROVE", |_| reply("done"));
        test_drive(Arc::clone(&gate), conversation(), serve)
            .await
            .expect("routes");
        assert_eq!(script.advisor_consults(), 1);
        // The unlatched stall still fires later on a tool-call turn.
        let serve = script.serve("APPROVE", {
            let turn = parking_lot::Mutex::new(Some(tool_call_turn()));
            move |_| turn.lock().take().expect("one executor call")
        });
        test_drive(gate, conversation(), serve)
            .await
            .expect("routes");
        assert_eq!(script.advisor_consults(), 2);
    }

    // ── Reasoning-only and empty turns ──────────────────────────────────────

    #[tokio::test]
    async fn reasoning_only_turn_reviewed_and_echoed() {
        let script = Script::new();
        let gate = gate(AdvisorGateConfig::default());
        let serve = script.serve("REDO: verify the output", {
            let turn = parking_lot::Mutex::new(Some(reasoning_only_turn()));
            move |index| {
                if index == 0 {
                    turn.lock().take().expect("one gated turn")
                } else {
                    reply("continued")
                }
            }
        });
        test_drive(gate, task_request(), serve)
            .await
            .expect("routes");
        // The consult saw the labeled reasoning as the terminal evidence.
        let consult = script.call(1);
        let transcript = consult.llm_request.messages[0]
            .text_content("\n")
            .expect("transcript text");
        assert!(transcript.contains(REASONING_TAIL_LABEL.trim_end()));
        assert!(transcript.contains("thinking about it"));
        // The REDO echo prefers the reasoning over an empty string.
        let redo = script.call(2);
        assert_eq!(
            redo.llm_request.messages[1].text_content("\n").as_deref(),
            Some("thinking about it")
        );
    }

    #[tokio::test]
    async fn empty_turn_redo_echo_uses_placeholder() {
        let script = Script::new();
        let gate = gate(AdvisorGateConfig::default());
        let serve = script.serve("REDO: produce output", {
            let turn = parking_lot::Mutex::new(Some(empty_turn()));
            move |index| {
                if index == 0 {
                    turn.lock().take().expect("one gated turn")
                } else {
                    reply("continued")
                }
            }
        });
        test_drive(gate, task_request(), serve)
            .await
            .expect("routes");
        let redo = script.call(2);
        assert_eq!(
            redo.llm_request.messages[1].text_content("\n").as_deref(),
            Some(EMPTY_ECHO_PLACEHOLDER)
        );
    }

    // ── Consult request shape ───────────────────────────────────────────────

    #[tokio::test]
    async fn consult_request_shape() {
        let script = Script::new();
        let gate = gate(AdvisorGateConfig {
            advisor_temperature: Some(0.2),
            ..AdvisorGateConfig::default()
        });
        let serve = script.serve("APPROVE", |_| reply("done"));
        test_drive(gate, task_request(), serve)
            .await
            .expect("routes");
        let consult = script.call(1).llm_request;
        assert_eq!(consult.instructions.len(), 1);
        assert_eq!(
            consult.instructions[0].content,
            vec![ContentBlock::Text {
                text: REVIEWER_SYSTEM_PROMPT.to_string()
            }]
        );
        assert_eq!(consult.messages.len(), 1);
        assert_eq!(consult.messages[0].role, Role::User);
        assert_eq!(consult.output.max_output_tokens, Some(2048));
        assert_eq!(consult.output.response_format, None);
        assert_eq!(consult.sampling.temperature, Some(0.2));
        assert!(consult.tools.is_empty());
        assert!(!consult.stream);
        let transcript = consult.messages[0].text_content("\n").expect("transcript");
        assert!(transcript.starts_with("Conversation so far (JSON):"));
        assert!(transcript.contains("The executor's latest turn"));
        assert!(transcript.ends_with("done"));
    }

    #[tokio::test]
    async fn consult_transcript_includes_system_instructions() {
        let script = Script::new();
        let gate = gate(AdvisorGateConfig::default());
        let serve = script.serve("APPROVE", |_| reply("done"));
        let mut gated = task_request();
        gated.llm_request.instructions = vec![InstructionBlock {
            role: Role::System,
            content: vec![ContentBlock::Text {
                text: "the deliverable must be a CSV".to_string(),
            }],
        }];
        test_drive(gate, gated, serve).await.expect("routes");
        // System content is normalized out of `messages`; the advisor still
        // sees it, leading the serialized transcript.
        let transcript = script.call(1).llm_request.messages[0]
            .text_content("\n")
            .expect("transcript text");
        assert!(transcript.contains("the deliverable must be a CSV"));
        let task = transcript.find("build X").expect("task present");
        let system = transcript
            .find("the deliverable must be a CSV")
            .expect("system present");
        assert!(system < task);
    }

    // ── Sessions ────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn session_final_evicts_scope() {
        let script = Script::new();
        let gate = gate(AdvisorGateConfig::default());
        let serve = script.serve("APPROVE", |_| reply("done"));
        let mut closing = with_session_id(task_request(), "s1");
        if let Some(metadata) = closing.metadata.as_mut() {
            metadata.session_final = Some(true);
        }
        test_drive(Arc::clone(&gate), closing, serve)
            .await
            .expect("routes");
        // The evicted scope re-arms: the same session id is reviewed again.
        let serve = script.serve("APPROVE", |_| reply("done"));
        test_drive(gate, with_session_id(task_request(), "s1"), serve)
            .await
            .expect("routes");
        assert_eq!(script.advisor_consults(), 2);
    }

    #[tokio::test]
    async fn concurrent_same_scope_requests_consult_once() {
        let script = Script::new();
        let gate = gate(AdvisorGateConfig::default());
        let barrier = Arc::new(tokio::sync::Barrier::new(2));
        let calls = Arc::clone(&script.calls);
        let serve = {
            let barrier = Arc::clone(&barrier);
            move |decision: Decision, request: Request| {
                let barrier = Arc::clone(&barrier);
                let calls = Arc::clone(&calls);
                Box::pin(async move {
                    let model = decision.selected_model_id().to_string();
                    calls.lock().push((model.clone(), request));
                    if model == ADVISOR {
                        Ok(reply("APPROVE"))
                    } else {
                        // Hold both executor turns until each has generated,
                        // so both runs race for the single review slot.
                        barrier.wait().await;
                        Ok(reply("done"))
                    }
                })
                    as futures::future::BoxFuture<
                        'static,
                        std::result::Result<Response, LlmClientError>,
                    >
            }
        };
        let (first, second) = tokio::join!(
            test_drive(Arc::clone(&gate), task_request(), serve.clone()),
            test_drive(Arc::clone(&gate), task_request(), serve)
        );
        first.expect("first run");
        second.expect("second run");
        assert_eq!(script.advisor_consults(), 1);
    }

    // ── Pure functions ──────────────────────────────────────────────────────

    #[test]
    fn verdict_parser_table() {
        let re = regex::Regex::new(VERDICT_PATTERN).expect("pattern compiles");
        let approve = |reply: &str| matches!(parse_verdict(&re, reply), Some(Verdict::Approve));
        let redo_plan = |reply: &str| match parse_verdict(&re, reply) {
            Some(Verdict::Redo { plan }) => Some(plan),
            _ => None,
        };
        assert!(approve("APPROVE"));
        assert!(approve("approve"));
        assert!(approve("  **APPROVE**"));
        assert!(approve("> approve"));
        assert!(approve("Final verdict: APPROVE"));
        assert!(approve("verdict: APPROVE"));
        assert_eq!(
            redo_plan("REDO: run the tests").as_deref(),
            Some("run the tests")
        );
        assert_eq!(redo_plan("REDO\n- fix x").as_deref(), Some("fix x"));
        assert_eq!(
            redo_plan("**Verdict:** REDO fix y").as_deref(),
            Some("fix y")
        );
        // An empty plan falls back to the whole reply.
        assert_eq!(redo_plan("REDO").as_deref(), Some("REDO"));
        // Word boundary: REDOING is not a verdict.
        assert!(parse_verdict(&re, "REDOING the work").is_none());
        // Prose-first replies are not trusted as verdicts.
        assert!(parse_verdict(&re, "I cannot approve this — REDO: run the tests").is_none());
        assert!(parse_verdict(&re, "").is_none());
    }

    #[test]
    fn transcript_middle_drop() {
        assert_eq!(middle_drop("short".to_string(), 256), "short");
        let long: String = "a".repeat(300) + &"b".repeat(300);
        let capped = middle_drop(long, 400);
        assert_eq!(
            capped,
            format!("{}{TRUNCATION_MARKER}{}", "a".repeat(100), "b".repeat(300))
        );
        // Multi-byte characters never split.
        let unicode: String = "é".repeat(600);
        let capped = middle_drop(unicode, 400);
        assert_eq!(
            capped,
            format!("{}{TRUNCATION_MARKER}{}", "é".repeat(100), "é".repeat(300))
        );
        assert_eq!(
            middle_drop("x".to_string(), 256),
            "x",
            "under-cap text passes through"
        );
        let framed = review_transcript(&[Message::text(Role::User, "task")], None, 256);
        assert!(framed.ends_with(NO_TEXT_PLACEHOLDER));
    }

    #[test]
    fn new_validation_errors() {
        let invalid = |config: AdvisorGateConfig, needle: &str| {
            let error = AdvisorGate::new(target(EXECUTOR), target(ADVISOR), config)
                .err()
                .expect("config rejected");
            assert!(error.to_string().contains(needle), "{error}");
        };
        invalid(
            AdvisorGateConfig {
                max_reviews: 0,
                ..AdvisorGateConfig::default()
            },
            "max_reviews",
        );
        invalid(
            AdvisorGateConfig {
                advisor_max_tokens: 0,
                ..AdvisorGateConfig::default()
            },
            "advisor_max_tokens",
        );
        invalid(
            AdvisorGateConfig {
                transcript_max_chars: 255,
                ..AdvisorGateConfig::default()
            },
            "transcript_max_chars",
        );
        invalid(
            AdvisorGateConfig {
                gate_trigger: GateTrigger::Pattern(String::new()),
                ..AdvisorGateConfig::default()
            },
            "non-empty gate_trigger_pattern",
        );
        invalid(
            AdvisorGateConfig {
                gate_trigger: GateTrigger::Pattern("(unclosed".to_string()),
                ..AdvisorGateConfig::default()
            },
            "not a valid regex",
        );
    }
}
