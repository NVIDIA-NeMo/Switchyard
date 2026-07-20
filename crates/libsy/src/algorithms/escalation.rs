// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Escalation routing, assembled from existing components.
//!
//! Escalation routing starts a conversation on the cheap `weak` tier and moves it to `strong`
//! when an LLM judge sees the run is in recoverable trouble; once escalated, it **latches** to
//! `strong` for the rest of the session. It is a [`FallThroughClassification`] over:
//!
//! - a **latch** ([`AffinityRouter`] retaining only the strong tier) — as a processor it pins
//!   the session on a strong decision; as the first classifier it short-circuits a latched
//!   session to `strong` without calling the judge;
//! - a **[`TurnGate`]** routing `weak` before `min_turn` (too little trajectory to judge);
//! - optionally a cheap **[`StagedRouter`]** heuristic (see [`escalation_router_with_staged`])
//!   that decides confident turns for free and defers ambiguous ones to the judge;
//! - a **judge** — an [`LlmClassifier`] over `{strong, weak}` with the escalation rubric that
//!   scores `strong` high to escalate, failing open to `weak`, wrapped in a [`ConfirmingJudge`]
//!   that holds `weak` until `confirmations` escalate verdicts accrue.
//!
//! The persisted [`State`](super::core::State) is the cross-turn memory: it carries the latch
//! and the confirmation streak across a session's turns.

use std::sync::Arc;

use async_trait::async_trait;

use super::affinity::AffinityRouter;
use super::core::{Classifier, Processor, Score, State};
use super::fall_through::FallThroughClassification;
use super::llm_classifier::LlmClassifier;
use super::staged::StagedRouter;
use super::turn_gate::TurnGate;
use crate::{Driver, LlmTarget, LlmTargetSet, Request};

/// Boxed, thread-safe error type used across the SDK.
type BoxErr = Box<dyn std::error::Error + Send + Sync>;

/// System prompt for the escalation judge. Condensed from the reference
/// `escalation_judge.md`, reframed for the per-target score tool: `strong` high = escalate.
pub const ESCALATION_JUDGE_SYSTEM_PROMPT: &str = "You are an escalation judge for an LLM proxy \
running a multi-turn agent that started on the cheaper `weak` tier. Judge the recent \
trajectory — is the agent making real progress toward the stated task, not how hard the task \
looks — and decide whether the run is in trouble a more capable model would fix. Call the \
route-selection tool exactly once, scoring `strong` high to escalate and `weak` high to stay. \
Escalate only on a clear PATTERN of trouble, never a single failed command: the same command \
or edit failing repeatedly with the same error; near-identical tool calls or file re-reads that \
gain no new information; claiming success while the latest output shows failure; finishing \
without the verification the task requires; drifting to unrelated work while the task is \
unstarted; or desperation (giving up, destructive flailing). Hold `weak` on the healthy \
friction the cheaper tier works through on its own: a test written to fail first, an error \
fixed on the next turn, early exploration dead-ends, adaptively handling a missing tool, trying \
a different approach after one fails, a slow command that has not finished, routine planning, \
and clean zero-count summaries (\"0 failed\"). When the evidence is thin or ambiguous, score \
`weak` high. Escalation is one-way and expensive.";

/// Trailing-message window shown to the judge — wider than the classifier default, because
/// loop detection must see the repeats. Mirrors the reference `recent_turn_window`.
pub const ESCALATION_JUDGE_TURN_WINDOW: usize = 14;

/// The per-session escalation streak, held in the router's [`State`].
///
/// Because a `FallThroughClassification`'s `State` persists for one session, this is a plain
/// counter — no session keying, unlike the reference's process-global streak store.
#[derive(Default)]
struct EscalationStreak {
    /// Consecutive escalate verdicts (reset by a confirming route or an aged-out decline).
    escalates: u32,
    /// Escalate-free judged turns since the last escalate verdict.
    declines: u32,
}

/// Requires repeated escalate verdicts before an escalation takes effect.
///
/// Wraps a judge [`Classifier`] over `{escalate_target, hold_target}`. Each turn it reads the
/// judge's top vote and updates the session streak: it routes `escalate_target` only once
/// `confirmations` escalate verdicts have accrued (within `confirmation_window` — how many
/// judged turns an escalate verdict stays live across intervening declines); otherwise it
/// routes `hold_target`. `confirmations = 1` (the default) escalates on the first verdict,
/// reproducing an unwrapped judge.
pub struct ConfirmingJudge {
    judge: Arc<dyn Classifier>,
    escalate_target: String,
    hold_target: String,
    confirmations: u32,
    confirmation_window: u32,
}

impl ConfirmingJudge {
    /// Wraps `judge`, confirming an escalation to `escalate_target` after `confirmations`
    /// verdicts (minimum 1); other turns route `hold_target`. Window defaults to 1.
    pub fn new(
        judge: Arc<dyn Classifier>,
        escalate_target: impl Into<String>,
        hold_target: impl Into<String>,
        confirmations: u32,
    ) -> Self {
        Self {
            judge,
            escalate_target: escalate_target.into(),
            hold_target: hold_target.into(),
            confirmations: confirmations.max(1),
            confirmation_window: 1,
        }
    }

    /// Sets how many judged turns an escalate verdict stays live for confirmation. `1` (the
    /// default) makes any decline reset the streak; `N` tolerates `N - 1` intervening declines.
    pub fn with_confirmation_window(mut self, window: u32) -> Self {
        self.confirmation_window = window.max(1);
        self
    }
}

/// The highest-confidence target in a score set, or `None` when empty.
fn top_target(scores: &[Score]) -> Option<&str> {
    scores
        .iter()
        .max_by(|a, b| a.confidence.total_cmp(&b.confidence))
        .map(|score| score.target.as_str())
}

#[async_trait]
impl Classifier for ConfirmingJudge {
    async fn score(
        &self,
        state: &mut State,
        request: &Request,
        driver: Option<&Driver>,
    ) -> Result<Vec<Score>, BoxErr> {
        let scores = self.judge.score(state, request, driver).await?;
        // The judge abstained — abstain too, deferring to the next classifier.
        let Some(top) = top_target(&scores) else {
            return Ok(Vec::new());
        };
        let escalated = top == self.escalate_target;

        // Update the session streak and decide whether the escalation is confirmed.
        let confirmed = {
            let streak = state.entry_or_insert_with(EscalationStreak::default);
            if escalated {
                streak.escalates += 1;
                streak.declines = 0;
                streak.escalates >= self.confirmations
            } else {
                streak.declines += 1;
                if streak.declines >= self.confirmation_window {
                    *streak = EscalationStreak::default();
                }
                false
            }
        };

        let target = if confirmed {
            self.escalate_target.clone()
        } else {
            self.hold_target.clone()
        };
        Ok(vec![Score {
            confidence: 1.0,
            target,
        }])
    }
}

/// Builds an escalation router: start on `weak`, let the `judge` model escalate to `strong`
/// from `min_turn` on (after `confirmations` escalate verdicts), and latch `strong` for the
/// rest of the session.
///
/// `strong` / `weak` are the routed tiers (their `semantic_name`s key the latch, gate, and
/// judge); `judge` names the model that scores the trajectory. `confirmations` is the number
/// of escalate verdicts required before escalating (`1` escalates on the first). Wire real
/// clients onto the targets (or serve the offloaded calls yourself when driving the step
/// stream).
pub fn escalation_router(
    strong: LlmTarget,
    weak: LlmTarget,
    judge: LlmTarget,
    min_turn: usize,
    confirmations: u32,
) -> FallThroughClassification {
    build(strong, weak, judge, min_turn, confirmations, None)
}

/// Like [`escalation_router`], with a cheap [`StagedRouter`] heuristic layer inserted before
/// the judge.
///
/// The staged classifier scores the turn's tool-result signals for free: a confident verdict
/// (magnitude past `staged_confidence_threshold`) decides the turn — escalating `strong`
/// (which the latch pins) or holding `weak` — while an ambiguous one abstains and the LLM
/// judge runs. This is the "cheap heuristic → expensive judge" cascade, so the judge is only
/// paid for on turns the heuristic cannot call. `StagedRouter` is classifier-only (it extracts
/// its signals inside `score`), so no extra processor is needed.
pub fn escalation_router_with_staged(
    strong: LlmTarget,
    weak: LlmTarget,
    judge: LlmTarget,
    min_turn: usize,
    confirmations: u32,
    staged_confidence_threshold: f64,
) -> FallThroughClassification {
    build(strong, weak, judge, min_turn, confirmations, Some(staged_confidence_threshold))
}

/// Shared assembly for the escalation router, optionally inserting a staged heuristic layer.
fn build(
    strong: LlmTarget,
    weak: LlmTarget,
    judge: LlmTarget,
    min_turn: usize,
    confirmations: u32,
    staged_confidence_threshold: Option<f64>,
) -> FallThroughClassification {
    let strong_name = strong.semantic_name.clone();
    let weak_name = weak.semantic_name.clone();

    // The latch retains only the strong tier: once escalated, later turns stay strong.
    let latch = Arc::new(AffinityRouter::new().with_latch_only([strong_name.clone()]));

    // The judge scores {strong, weak}; the confirmation gate holds weak until enough escalate
    // verdicts accrue, then routes strong (which the latch pins).
    let judge_llm: Arc<dyn Classifier> = Arc::new(
        LlmClassifier::new(judge, vec![strong_name.clone(), weak_name.clone()])
            .with_system_prompt(ESCALATION_JUDGE_SYSTEM_PROMPT)
            .with_recent_turn_window(ESCALATION_JUDGE_TURN_WINDOW)
            .with_fail_open(weak_name.clone()),
    );
    let judge = Arc::new(ConfirmingJudge::new(
        judge_llm,
        strong_name.clone(),
        weak_name.clone(),
        confirmations,
    ));

    let mut router = FallThroughClassification::new(LlmTargetSet::new(vec![strong, weak]))
        .with_processor(latch.clone() as Arc<dyn Processor>)
        .with_classifier(latch as Arc<dyn Classifier>)
        .with_classifier(Arc::new(TurnGate::new(weak_name.clone(), min_turn)));

    // Optional cheap heuristic layer between the gate and the judge: staged maps its capable
    // tier to `strong` and efficient to `weak`, or abstains in its ambiguous band.
    if let Some(threshold) = staged_confidence_threshold {
        let staged =
            StagedRouter::new(strong_name, weak_name).with_confidence_threshold(threshold);
        router = router.with_classifier(Arc::new(staged));
    }

    router.with_classifier(judge)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicUsize, Ordering};

    use serde_json::json;
    use switchyard_protocol::{
        completion_text, AggLlmResponse, ContentBlock, LlmRequest, Message, Metadata,
        ResponseOutput, Role, ToolCall, ToolResult,
    };

    use crate::{Algorithm, Context, Decision, LlmResponse, Response, RoutedLlmClient};

    /// A judge stub that votes a scripted target per call (last vote repeats).
    struct ScriptedJudge {
        votes: Vec<&'static str>,
        calls: AtomicUsize,
    }

    #[async_trait]
    impl Classifier for ScriptedJudge {
        async fn score(
            &self,
            _state: &mut State,
            _request: &Request,
            _driver: Option<&Driver>,
        ) -> Result<Vec<Score>, BoxErr> {
            let index = self
                .calls
                .fetch_add(1, Ordering::SeqCst)
                .min(self.votes.len() - 1);
            Ok(vec![Score {
                confidence: 1.0,
                target: self.votes[index].to_string(),
            }])
        }
    }

    /// Runs one `ConfirmingJudge` turn over shared state, returning the routed target.
    async fn confirm_turn(gate: &ConfirmingJudge, state: &mut State) -> Result<String, BoxErr> {
        let scores = gate.score(state, &request(1), None).await?;
        Ok(scores[0].target.clone())
    }

    #[tokio::test]
    async fn confirmations_hold_weak_until_the_streak_is_met() -> Result<(), BoxErr> {
        // confirmations=2: the first escalate holds weak, the second confirms strong.
        let gate = ConfirmingJudge::new(
            Arc::new(ScriptedJudge {
                votes: vec!["strong"],
                calls: AtomicUsize::new(0),
            }),
            "strong",
            "weak",
            2,
        );
        let mut state = State::default();
        assert_eq!(confirm_turn(&gate, &mut state).await?, "weak");
        assert_eq!(confirm_turn(&gate, &mut state).await?, "strong");
        Ok(())
    }

    #[tokio::test]
    async fn a_decline_resets_the_streak_with_window_one() -> Result<(), BoxErr> {
        // strong, weak, strong, strong — the decline (window=1) resets, so it takes two more
        // escalates to confirm.
        let gate = ConfirmingJudge::new(
            Arc::new(ScriptedJudge {
                votes: vec!["strong", "weak", "strong", "strong"],
                calls: AtomicUsize::new(0),
            }),
            "strong",
            "weak",
            2,
        );
        let mut state = State::default();
        assert_eq!(confirm_turn(&gate, &mut state).await?, "weak"); // esc streak 1
        assert_eq!(confirm_turn(&gate, &mut state).await?, "weak"); // decline resets
        assert_eq!(confirm_turn(&gate, &mut state).await?, "weak"); // esc streak 1 again
        assert_eq!(confirm_turn(&gate, &mut state).await?, "strong"); // esc streak 2 → confirm
        Ok(())
    }

    #[tokio::test]
    async fn a_wider_window_survives_an_intervening_decline() -> Result<(), BoxErr> {
        // strong, weak, strong with window=2 — the escalate survives one decline and confirms.
        let gate = ConfirmingJudge::new(
            Arc::new(ScriptedJudge {
                votes: vec!["strong", "weak", "strong"],
                calls: AtomicUsize::new(0),
            }),
            "strong",
            "weak",
            2,
        )
        .with_confirmation_window(2);
        let mut state = State::default();
        assert_eq!(confirm_turn(&gate, &mut state).await?, "weak"); // esc streak 1
        assert_eq!(confirm_turn(&gate, &mut state).await?, "weak"); // decline kept (window 2)
        assert_eq!(confirm_turn(&gate, &mut state).await?, "strong"); // esc streak 2 → confirm
        Ok(())
    }

    /// Echoes the routed model name back as the completion (for the strong/weak tiers).
    struct EchoClient;

    #[async_trait]
    impl RoutedLlmClient for EchoClient {
        async fn call(
            &self,
            _ctx: Context,
            _request: Request,
            decision: Arc<dyn Decision>,
        ) -> Result<Response, BoxErr> {
            Ok(Response {
                llm_response: LlmResponse::Agg(switchyard_protocol::text_response(
                    None,
                    decision.selected_model().to_string(),
                )),
                metadata: None,
            })
        }
    }

    /// The judge model: returns a `select_route` tool call, escalating on its first call and
    /// holding `weak` after — and counts how many times it was consulted.
    struct JudgeClient(Arc<AtomicUsize>);

    #[async_trait]
    impl RoutedLlmClient for JudgeClient {
        async fn call(
            &self,
            _ctx: Context,
            _request: Request,
            _decision: Arc<dyn Decision>,
        ) -> Result<Response, BoxErr> {
            let (strong, weak) = if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
                (0.9, 0.1) // escalate
            } else {
                (0.1, 0.9) // hold weak
            };
            Ok(Response {
                llm_response: LlmResponse::Agg(AggLlmResponse {
                    outputs: vec![ResponseOutput {
                        role: Role::Assistant,
                        content: vec![ContentBlock::ToolCall(ToolCall {
                            id: "call-1".to_string(),
                            name: crate::DEFAULT_TOOL_NAME.to_string(),
                            arguments: json!({ "strong": strong, "weak": weak }),
                        })],
                        stop_reason: None,
                    }],
                    ..AggLlmResponse::default()
                }),
                metadata: None,
            })
        }
    }

    fn target(name: &str, client: Arc<dyn RoutedLlmClient>) -> LlmTarget {
        LlmTarget {
            semantic_name: name.to_string(),
            llm_client: Some(client),
        }
    }

    /// A same-session request with `user_turns` user messages (each with an assistant reply).
    fn request(user_turns: usize) -> Request {
        let mut messages = Vec::new();
        for index in 0..user_turns {
            messages.push(Message::text(Role::User, format!("turn {index}")));
            messages.push(Message::text(Role::Assistant, "working"));
        }
        Request {
            llm_request: LlmRequest {
                model: Some("auto".to_string()),
                messages,
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: Some(Metadata {
                session_id: Some("sess-1".to_string()),
                ..Metadata::default()
            }),
        }
    }

    /// Runs one turn through the shared router, returning the served completion text.
    async fn run_turn(
        router: &Arc<FallThroughClassification>,
        req: Request,
    ) -> Result<String, BoxErr> {
        let (_, response) = router.clone().run(Context::default(), req).await?;
        response
            .llm_response
            .into_agg()
            .await
            .map(|agg| completion_text(&agg))
    }

    #[tokio::test]
    async fn escalates_then_latches_strong() -> Result<(), BoxErr> {
        let judge_calls = Arc::new(AtomicUsize::new(0));
        let echo = Arc::new(EchoClient) as Arc<dyn RoutedLlmClient>;
        let router = Arc::new(escalation_router(
            target("strong", echo.clone()),
            target("weak", echo),
            target("judge", Arc::new(JudgeClient(judge_calls.clone()))),
            3,
            1,
        ));

        // Turn 1: before min_turn, the gate routes weak and the judge is never called.
        assert_eq!(run_turn(&router, request(1)).await?, "weak");
        assert_eq!(judge_calls.load(Ordering::SeqCst), 0);

        // Turn 3: the judge runs, escalates, and the strong decision pins the session.
        assert_eq!(run_turn(&router, request(3)).await?, "strong");
        assert_eq!(judge_calls.load(Ordering::SeqCst), 1);

        // Later turn: the latch routes strong and short-circuits the judge (its call count is
        // unchanged even though it would now hold weak).
        assert_eq!(run_turn(&router, request(3)).await?, "strong");
        assert_eq!(judge_calls.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[tokio::test]
    async fn judge_failure_holds_weak_without_latching() -> Result<(), BoxErr> {
        /// A judge whose call always fails.
        struct FailingJudge;

        #[async_trait]
        impl RoutedLlmClient for FailingJudge {
            async fn call(
                &self,
                _ctx: Context,
                _request: Request,
                _decision: Arc<dyn Decision>,
            ) -> Result<Response, BoxErr> {
                Err("judge upstream failed".into())
            }
        }

        let echo = Arc::new(EchoClient) as Arc<dyn RoutedLlmClient>;
        let router = Arc::new(escalation_router(
            target("strong", echo.clone()),
            target("weak", echo),
            target("judge", Arc::new(FailingJudge)),
            3,
            1,
        ));

        // Judge fails → fail open to weak, and no strong pin is created...
        assert_eq!(run_turn(&router, request(3)).await?, "weak");
        // ...so the next judged turn still routes weak (the session did not latch).
        assert_eq!(run_turn(&router, request(3)).await?, "weak");
        Ok(())
    }

    /// A judge model whose call always escalates; counts how many times it was consulted.
    struct AlwaysEscalate(Arc<AtomicUsize>);

    #[async_trait]
    impl RoutedLlmClient for AlwaysEscalate {
        async fn call(
            &self,
            _ctx: Context,
            _request: Request,
            _decision: Arc<dyn Decision>,
        ) -> Result<Response, BoxErr> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(Response {
                llm_response: LlmResponse::Agg(AggLlmResponse {
                    outputs: vec![ResponseOutput {
                        role: Role::Assistant,
                        content: vec![ContentBlock::ToolCall(ToolCall {
                            id: "c".to_string(),
                            name: crate::DEFAULT_TOOL_NAME.to_string(),
                            arguments: json!({ "strong": 0.9, "weak": 0.1 }),
                        })],
                        stop_reason: None,
                    }],
                    ..AggLlmResponse::default()
                }),
                metadata: None,
            })
        }
    }

    #[tokio::test]
    async fn two_confirmations_escalate_on_the_second_verdict() -> Result<(), BoxErr> {
        let judge_calls = Arc::new(AtomicUsize::new(0));
        let echo = Arc::new(EchoClient) as Arc<dyn RoutedLlmClient>;
        let router = Arc::new(escalation_router(
            target("strong", echo.clone()),
            target("weak", echo),
            target("judge", Arc::new(AlwaysEscalate(judge_calls.clone()))),
            3,
            2, // require two escalate verdicts
        ));

        // First judged turn: escalate verdict #1 holds weak (pending confirmation).
        assert_eq!(run_turn(&router, request(3)).await?, "weak");
        assert_eq!(judge_calls.load(Ordering::SeqCst), 1);

        // Second judged turn: escalate verdict #2 confirms → strong, and pins the session.
        assert_eq!(run_turn(&router, request(3)).await?, "strong");
        assert_eq!(judge_calls.load(Ordering::SeqCst), 2);

        // Latched: strong without consulting the judge again.
        assert_eq!(run_turn(&router, request(3)).await?, "strong");
        assert_eq!(judge_calls.load(Ordering::SeqCst), 2);
        Ok(())
    }

    // --- staged heuristic layer --------------------------------------------------------

    /// A past-min_turn request whose last tool result is a critical error — the staged
    /// heuristic reads this as a hard escalation signal.
    fn troubled_request() -> Request {
        let messages = vec![
            Message::text(Role::User, "do the task"),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolCall(ToolCall {
                    id: "c".to_string(),
                    name: "Bash".to_string(),
                    arguments: json!({ "command": "make" }),
                })],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult(ToolResult {
                    tool_call_id: "c".to_string(),
                    content: vec![ContentBlock::Text { text: "fatal: out of memory".to_string() }],
                    is_error: None,
                })],
            },
            Message::text(Role::User, "continue"),
        ];
        Request {
            llm_request: LlmRequest { model: Some("auto".to_string()), messages, ..LlmRequest::default() },
            raw_request: None,
            metadata: Some(Metadata { session_id: Some("sess-1".to_string()), ..Metadata::default() }),
        }
    }

    #[tokio::test]
    async fn staged_heuristic_escalates_before_the_judge() -> Result<(), BoxErr> {
        // The staged layer sees a critical error, escalates strong for free, and the judge is
        // never consulted.
        let judge_calls = Arc::new(AtomicUsize::new(0));
        let echo = Arc::new(EchoClient) as Arc<dyn RoutedLlmClient>;
        let router = Arc::new(escalation_router_with_staged(
            target("strong", echo.clone()),
            target("weak", echo),
            target("judge", Arc::new(JudgeClient(judge_calls.clone()))),
            3,
            1,
            0.75,
        ));

        assert_eq!(run_turn(&router, troubled_request()).await?, "strong");
        assert_eq!(judge_calls.load(Ordering::SeqCst), 0);
        Ok(())
    }

    #[tokio::test]
    async fn ambiguous_turn_falls_through_the_staged_layer_to_the_judge() -> Result<(), BoxErr> {
        // A benign turn carries no strong staged signal, so staged abstains and the judge runs.
        let judge_calls = Arc::new(AtomicUsize::new(0));
        let echo = Arc::new(EchoClient) as Arc<dyn RoutedLlmClient>;
        let router = Arc::new(escalation_router_with_staged(
            target("strong", echo.clone()),
            target("weak", echo),
            target("judge", Arc::new(JudgeClient(judge_calls.clone()))),
            3,
            1,
            0.75,
        ));

        assert_eq!(run_turn(&router, request(3)).await?, "strong"); // judge escalates
        assert_eq!(judge_calls.load(Ordering::SeqCst), 1); // staged abstained → judge ran
        Ok(())
    }
}
