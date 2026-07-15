// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ensemble router built on the [`Algorithm`] interfaces.
//!
//! Each request is fanned out to a set of candidate models concurrently; a judge
//! model (e.g. Haiku) then picks the best response, which is returned to the
//! agent. The algorithm tallies which candidate the judge preferred across
//! requests and, after `exploration_turns` ensemble turns, commits to the
//! winningest model — every subsequent request routes straight to that one model
//! with no fan-out and no judge call.
//!
//! Unlike the reference routers, this algorithm is **stateful**: the win tally,
//! turn counter, and committed choice live behind a [`std::sync::Mutex`] so one
//! shared `&self` can serve a session's requests concurrently (see the
//! `Algorithm` docs). In a proxy setup one `EnsembleOrchAlgo` is created per session,
//! so this state is per-session.

use std::collections::BTreeMap;
use std::error::Error;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use libsy::{Algorithm, Context, Decision, Driver, LlmTargetSet, Request, Response, Signals};
use switchyard_protocol::{completion_text, prompt_text, text_request};

/// Which step of the ensemble flow produced a decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnsemblePhase {
    /// A fan-out call to one candidate model during an exploration turn.
    Candidate,
    /// The judge call that scored the candidate responses.
    Judge,
    /// The candidate the judge selected on an exploration turn.
    Winner,
    /// The single model the algorithm committed to after exploration.
    Committed,
}

impl EnsemblePhase {
    /// Stable string form of the phase, used in decision reasoning.
    pub fn as_str(self) -> &'static str {
        match self {
            EnsemblePhase::Candidate => "candidate",
            EnsemblePhase::Judge => "judge",
            EnsemblePhase::Winner => "winner",
            EnsemblePhase::Committed => "committed",
        }
    }
}

/// Decision produced at each step of the ensemble flow.
pub struct EnsembleDecision {
    /// The model this step concerns (a candidate, the judge, the winner, or the
    /// committed model).
    pub selected_model: String,
    /// Human-readable explanation of the step.
    pub reasoning: String,
    /// Which step of the ensemble flow produced this decision.
    pub phase: EnsemblePhase,
}

impl Decision for EnsembleDecision {
    fn selected_model(&self) -> &str {
        &self.selected_model
    }
    fn reasoning(&self) -> Option<&str> {
        Some(&self.reasoning)
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Mutable per-session state, guarded by a [`Mutex`] so `&self` can be shared
/// across concurrent requests. Never held across an `await`.
struct EnsembleState {
    /// Judge-win count per candidate model.
    wins: BTreeMap<String, u64>,
    /// Completed ensemble (exploration) turns.
    turns: u64,
    /// The model committed to once exploration is over; `None` while exploring.
    committed: Option<String>,
}

/// Ensemble router: fan out to candidates, judge the best, then commit.
pub struct EnsembleOrchAlgo {
    candidate_models: Vec<String>,
    judge_model: String,
    /// Number of ensemble turns to run before committing to the best model.
    /// `0` disables committing — the algorithm ensembles on every request.
    exploration_turns: u64,
    target_set: LlmTargetSet,
    state: Mutex<EnsembleState>,
}

impl EnsembleOrchAlgo {
    /// Create an ensemble over `candidate_models`, judged by `judge_model`,
    /// exploring for `exploration_turns` before committing to the winningest
    /// candidate (`0` = never commit, ensemble every request), routing among
    /// `target_set`. Wrap it in an [`Arc`](std::sync::Arc) and drive it with
    /// [`run`](libsy::Algorithm::run) or
    /// [`run_stream`](libsy::Algorithm::run_stream).
    pub fn new(
        candidate_models: Vec<String>,
        judge_model: impl Into<String>,
        exploration_turns: u64,
        target_set: LlmTargetSet,
    ) -> Self {
        Self {
            candidate_models,
            judge_model: judge_model.into(),
            exploration_turns,
            target_set,
            state: Mutex::new(EnsembleState {
                wins: BTreeMap::new(),
                turns: 0,
                committed: None,
            }),
        }
    }

    /// If exploration is over, return the committed model (committing lazily on
    /// the first post-exploration request). `None` means keep ensembling.
    ///
    /// The lock is taken and dropped here, never across an `await`. Under
    /// concurrency two requests may both commit; they pick the same model from
    /// the same tally (with a stable tie-break), so the result is identical.
    fn resolve_committed(&self) -> Result<Option<String>, Box<dyn Error + Send + Sync>> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| "ensemble state lock poisoned")?;
        if let Some(model) = &state.committed {
            return Ok(Some(model.clone()));
        }
        // `exploration_turns == 0` keeps the algorithm in ensemble mode forever.
        if self.exploration_turns > 0 && state.turns >= self.exploration_turns {
            let best = self.pick_best(&state.wins)?;
            state.committed = Some(best.clone());
            return Ok(Some(best));
        }
        Ok(None)
    }

    /// The candidate with the most judge-wins, breaking ties toward the earlier
    /// candidate (stable and deterministic). Errors only if there are no
    /// candidates configured.
    fn pick_best(
        &self,
        wins: &BTreeMap<String, u64>,
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        let mut best = self
            .candidate_models
            .first()
            .ok_or("no candidate models configured")?;
        let mut best_wins = wins.get(best).copied().unwrap_or(0);
        for model in &self.candidate_models[1..] {
            let w = wins.get(model).copied().unwrap_or(0);
            if w > best_wins {
                best = model;
                best_wins = w;
            }
        }
        Ok(best.clone())
    }

    /// Route a request to a single already-chosen model — the committed fast path.
    async fn route_committed(
        &self,
        driver: &Driver,
        ctx: Context,
        request: Request,
        model: String,
    ) -> Result<(Vec<Arc<dyn Decision>>, Response), Box<dyn Error + Send + Sync>> {
        let target = self.target_set.get_target(&model)?;
        let decision: Arc<dyn Decision> = Arc::new(EnsembleDecision {
            reasoning: format!(
                "committed to '{model}' after {} turns",
                self.exploration_turns
            ),
            selected_model: model.clone(),
            phase: EnsemblePhase::Committed,
        });
        // The agent's inbound name rides through; the committed model is on the
        // decision, not stamped onto the request.
        let routed = Request {
            llm_request: text_request(
                request.llm_request.model.clone(),
                prompt_text(&request.llm_request),
            ),
            raw_request: request.raw_request,
            metadata: request.metadata,
        };
        let response = driver
            .call_llm_target(ctx, &target, routed, decision.clone())
            .await?;
        Ok((vec![decision], response))
    }

    /// One exploration turn: fan out to every candidate, judge the survivors,
    /// tally the winner, and return its response.
    async fn ensemble_turn(
        &self,
        driver: &Driver,
        ctx: Context,
        request: Request,
    ) -> Result<(Vec<Arc<dyn Decision>>, Response), Box<dyn Error + Send + Sync>> {
        let user_prompt = prompt_text(&request.llm_request);
        // The agent's inbound name rides through every sub-call unchanged; the model
        // each call hits is carried by its decision, not stamped onto the request.
        let inbound = request.llm_request.model.clone();

        // Fan out to all candidates concurrently with the same user prompt. Each
        // call is annotated with its own candidate decision so the caller can see
        // which model an offloaded call targets.
        let mut candidate_decisions: Vec<Arc<dyn Decision>> = Vec::new();
        let mut calls = Vec::new();
        for model in &self.candidate_models {
            let target = self.target_set.get_target(model)?;
            let decision: Arc<dyn Decision> = Arc::new(EnsembleDecision {
                selected_model: model.clone(),
                reasoning: format!("ensemble candidate '{model}'"),
                phase: EnsemblePhase::Candidate,
            });
            candidate_decisions.push(decision.clone());
            let call_request = Request {
                llm_request: text_request(inbound.clone(), user_prompt.clone()),
                raw_request: request.raw_request.clone(),
                metadata: request.metadata.clone(),
            };
            let model = model.clone();
            let ctx = ctx.clone();
            calls.push(async move {
                (
                    model,
                    driver
                        .call_llm_target(ctx, &target, call_request, decision)
                        .await,
                )
            });
        }
        let results = futures::future::join_all(calls).await;

        // Keep only successful responses, preserving candidate order. A failed
        // candidate is simply excluded from judging rather than failing the turn.
        let mut survivors: Vec<(String, Response)> = Vec::new();
        for (model, result) in results {
            if let Ok(response) = result {
                survivors.push((model, response));
            }
        }
        if survivors.is_empty() {
            return Err("all ensemble candidates failed".into());
        }

        // Pick the winner: judge only when there is a real choice to make.
        let (winner_model, winner_response, judge_decision) = if survivors.len() == 1 {
            let (model, response) = survivors
                .into_iter()
                .next()
                .ok_or("survivor unexpectedly missing")?;
            (model, response, None)
        } else {
            let judge_prompt = build_judge_prompt(&user_prompt, &survivors);
            let judge_target = self.target_set.get_target(&self.judge_model)?;
            let judge_decision: Arc<dyn Decision> = Arc::new(EnsembleDecision {
                selected_model: self.judge_model.clone(),
                reasoning: format!("judging {} candidate responses", survivors.len()),
                phase: EnsemblePhase::Judge,
            });
            let judge_request = Request {
                llm_request: text_request(inbound.clone(), judge_prompt),
                raw_request: request.raw_request.clone(),
                metadata: request.metadata.clone(),
            };
            let judge_response = driver
                .call_llm_target(ctx, &judge_target, judge_request, judge_decision.clone())
                .await?;
            // Fail open: an unparseable pick falls back to the first response.
            let choice = parse_choice(
                &judge_response
                    .llm_response
                    .agg()
                    .map(completion_text)
                    .unwrap_or_default(),
                survivors.len(),
            );
            let (model, response) = survivors
                .into_iter()
                .nth(choice)
                .ok_or("judge choice out of range")?;
            (model, response, Some(judge_decision))
        };

        // Record the win and advance the turn counter under the lock (not held
        // across any await).
        {
            let mut state = self
                .state
                .lock()
                .map_err(|_| "ensemble state lock poisoned")?;
            *state.wins.entry(winner_model.clone()).or_insert(0) += 1;
            state.turns += 1;
        }

        let winner_decision: Arc<dyn Decision> = Arc::new(EnsembleDecision {
            reasoning: format!("judge selected '{winner_model}' as best response"),
            selected_model: winner_model,
            phase: EnsemblePhase::Winner,
        });

        // Trace order: [candidate calls..., judge?, winner].
        let mut trace = candidate_decisions;
        if let Some(judge_decision) = judge_decision {
            trace.push(judge_decision);
        }
        trace.push(winner_decision);
        Ok((trace, winner_response))
    }
}

/// Build the judge prompt. Responses are presented anonymously (no model names)
/// so the judge scores on content alone rather than model reputation.
fn build_judge_prompt(user_prompt: &str, survivors: &[(String, Response)]) -> String {
    let mut prompt = String::from(
        "You are an impartial judge. Choose which response best answers the user request.\n\n",
    );
    prompt.push_str("User request:\n");
    prompt.push_str(user_prompt);
    prompt.push_str("\n\n");
    for (i, (_model, response)) in survivors.iter().enumerate() {
        prompt.push_str(&format!(
            "Response {}:\n{}\n\n",
            i + 1,
            response
                .llm_response
                .agg()
                .map(completion_text)
                .unwrap_or_default()
        ));
    }
    prompt.push_str(&format!(
        "Reply with only the number (1-{}) of the best response.",
        survivors.len()
    ));
    prompt
}

/// Parse the judge's 1-based pick into a 0-based index, failing open to the
/// first response. Reads the first run of digits in the reply, so "2" or
/// "Response 2 is best" both select index 1.
fn parse_choice(completion: &str, count: usize) -> usize {
    let mut digits = String::new();
    for c in completion.chars() {
        if c.is_ascii_digit() {
            digits.push(c);
        } else if !digits.is_empty() {
            break;
        }
    }
    match digits.parse::<usize>() {
        Ok(n) if n >= 1 && n <= count => n - 1,
        _ => 0,
    }
}

#[async_trait]
impl Algorithm for EnsembleOrchAlgo {
    async fn create_run_task(
        self: Arc<Self>,
        ctx: Context,
        driver: Driver,
        request: Request,
    ) -> Result<Response, Box<dyn Error + Send + Sync>> {
        // Fast path: exploration is over — route straight to the committed model;
        // otherwise run a full ensemble turn. Both return a decision trace plus the
        // final response.
        let (trace, response) = if let Some(model) = self.resolve_committed()? {
            self.route_committed(&driver, ctx.clone(), request, model)
                .await?
        } else {
            self.ensemble_turn(&driver, ctx.clone(), request).await?
        };
        // Publish the trace to the stream (candidate..., judge?, winner). The
        // candidate decisions also rode along on their offloaded `CallLlm` steps.
        for decision in trace {
            driver.info(ctx.clone(), decision).await?;
        }
        Ok(response)
    }

    async fn process_signals(
        self: Arc<Self>,
        _signals: Signals,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Success is measured by the judge, not agent-system signals.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libsy::{LlmClient, LlmResponse, LlmTarget, Response, RoutedRequest};
    use std::sync::Mutex as StdMutex;
    use switchyard_protocol::text_response;

    /// Mock client that answers candidate calls with `answer from {model}` and,
    /// for the judge model, returns the 1-based number of the response whose
    /// content mentions `prefer` (so a test controls which candidate "wins").
    /// Records every model it was called with for call-count assertions.
    struct JudgingClient {
        judge_model: String,
        prefer: String,
        calls: Arc<StdMutex<Vec<String>>>,
    }

    #[async_trait]
    impl LlmClient for JudgingClient {
        async fn call(
            &self,
            routed: RoutedRequest,
        ) -> Result<Response, Box<dyn Error + Send + Sync>> {
            let name = routed.decision.selected_model().to_string();
            self.calls
                .lock()
                .map_err(|_| "lock poisoned")?
                .push(name.clone());
            let completion = if name == self.judge_model {
                judge_pick(&prompt_text(&routed.request.llm_request), &self.prefer)
            } else {
                format!("answer from {name}")
            };
            Ok(Response {
                llm_response: LlmResponse::Agg(text_response(None, completion)),
                metadata: None,
            })
        }
    }

    /// Scan a judge prompt for the `Response N:` whose body is `answer from
    /// {prefer}` and return `N` as a string; defaults to "1".
    fn judge_pick(prompt: &str, prefer: &str) -> String {
        let target_line = format!("answer from {prefer}");
        let mut current = 1u32;
        for line in prompt.lines() {
            if let Some(rest) = line.strip_prefix("Response ") {
                if let Ok(num) = rest.trim_end_matches(':').parse::<u32>() {
                    current = num;
                }
            } else if line == target_line {
                return current.to_string();
            }
        }
        "1".to_string()
    }

    /// Build an ensemble algo over `candidates` + a judge, all backed by one
    /// judging client that prefers `prefer`. Returns the algo and the shared
    /// call log.
    fn algo(
        candidates: &[&str],
        judge: &str,
        prefer: &str,
        exploration_turns: u64,
    ) -> (EnsembleOrchAlgo, Arc<StdMutex<Vec<String>>>) {
        let calls = Arc::new(StdMutex::new(Vec::new()));
        let client = Arc::new(JudgingClient {
            judge_model: judge.to_string(),
            prefer: prefer.to_string(),
            calls: Arc::clone(&calls),
        }) as Arc<dyn LlmClient>;
        let target = |name: &str| LlmTarget {
            semantic_name: name.to_string(),
            llm_client: Some(client.clone()),
        };
        let mut targets: Vec<LlmTarget> = candidates.iter().map(|n| target(n)).collect();
        targets.push(target(judge));
        let algo = EnsembleOrchAlgo::new(
            candidates.iter().map(|s| s.to_string()).collect(),
            judge.to_string(),
            exploration_turns,
            LlmTargetSet::new(targets),
        );
        (algo, calls)
    }

    /// Build an ensemble algo whose candidate + judge targets all share `client`.
    /// One such algo models a single session's stateful router.
    fn algo_with_client(
        candidates: &[&str],
        judge: &str,
        exploration_turns: u64,
        client: Arc<dyn LlmClient>,
    ) -> EnsembleOrchAlgo {
        let target = |name: &str| LlmTarget {
            semantic_name: name.to_string(),
            llm_client: Some(client.clone()),
        };
        let mut targets: Vec<LlmTarget> = candidates.iter().map(|n| target(n)).collect();
        targets.push(target(judge));
        EnsembleOrchAlgo::new(
            candidates.iter().map(|s| s.to_string()).collect(),
            judge.to_string(),
            exploration_turns,
            LlmTargetSet::new(targets),
        )
    }

    fn request(prompt: &str) -> Request {
        Request {
            llm_request: text_request(Some("auto".to_string()), prompt),
            raw_request: None,
            metadata: None,
        }
    }

    /// Wrap an ensemble algo as `Arc<dyn Algorithm>` we can drive to completion.
    /// Reuse one handle across requests to exercise the algo's per-session state.
    fn orch(algo: EnsembleOrchAlgo) -> Arc<dyn Algorithm> {
        Arc::new(algo)
    }

    fn as_ensemble(
        d: &Arc<dyn Decision>,
    ) -> Result<&EnsembleDecision, Box<dyn Error + Send + Sync>> {
        d.as_any()
            .downcast_ref::<EnsembleDecision>()
            .ok_or_else(|| "expected an EnsembleDecision".into())
    }

    #[tokio::test]
    async fn exploration_turn_fans_out_judges_and_returns_the_winner(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Judge prefers b/model; it should win and be returned.
        let (algo, calls) = algo(&["a/model", "b/model"], "judge/haiku", "b/model", 100);
        let (trace, response) = orch(algo)
            .run(Context::default(), request("solve it"))
            .await?;
        assert_eq!(
            response
                .llm_response
                .agg()
                .map(completion_text)
                .unwrap_or_default(),
            "answer from b/model"
        );

        // Both candidates and the judge were called.
        let calls = calls.lock().map_err(|_| "lock poisoned")?;
        assert!(calls.contains(&"a/model".to_string()));
        assert!(calls.contains(&"b/model".to_string()));
        assert!(calls.contains(&"judge/haiku".to_string()));

        // Trace: [candidate a, candidate b, judge, winner].
        assert_eq!(trace.len(), 4);
        assert_eq!(as_ensemble(&trace[0])?.phase, EnsemblePhase::Candidate);
        assert_eq!(as_ensemble(&trace[2])?.phase, EnsemblePhase::Judge);
        let winner = as_ensemble(&trace[3])?;
        assert_eq!(winner.phase, EnsemblePhase::Winner);
        assert_eq!(winner.selected_model, "b/model");
        Ok(())
    }

    #[tokio::test]
    async fn commits_to_the_winningest_model_after_exploration(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Judge always prefers b/model over 2 exploration turns, so the algo
        // commits to b/model even though a/model is listed first.
        let (algo, calls) = algo(&["a/model", "b/model"], "judge/haiku", "b/model", 2);
        let orch = orch(algo);

        // Two exploration turns.
        orch.clone().run(Context::default(), request("t1")).await?;
        orch.clone().run(Context::default(), request("t2")).await?;
        let judge_calls_after_exploration = calls
            .lock()
            .map_err(|_| "lock poisoned")?
            .iter()
            .filter(|c| *c == "judge/haiku")
            .count();
        assert_eq!(judge_calls_after_exploration, 2);

        // Third request: committed fast path — routes straight to b/model with no
        // fan-out to a/model and no judge call.
        let (trace, response) = orch.clone().run(Context::default(), request("t3")).await?;
        assert_eq!(
            response
                .llm_response
                .agg()
                .map(completion_text)
                .unwrap_or_default(),
            "answer from b/model"
        );
        assert_eq!(trace.len(), 1);
        let decision = as_ensemble(&trace[0])?;
        assert_eq!(decision.phase, EnsemblePhase::Committed);
        assert_eq!(decision.selected_model, "b/model");

        let calls = calls.lock().map_err(|_| "lock poisoned")?;
        // Judge was not called again on the committed turn.
        assert_eq!(calls.iter().filter(|c| *c == "judge/haiku").count(), 2);
        // a/model was called only on the two exploration turns, not the third.
        assert_eq!(calls.iter().filter(|c| *c == "a/model").count(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn single_candidate_skips_the_judge() -> Result<(), Box<dyn Error + Send + Sync>> {
        let (algo, calls) = algo(&["only/model"], "judge/haiku", "only/model", 100);
        let (trace, response) = orch(algo).run(Context::default(), request("hi")).await?;
        assert_eq!(
            response
                .llm_response
                .agg()
                .map(completion_text)
                .unwrap_or_default(),
            "answer from only/model"
        );
        // No judge call for a lone candidate.
        assert!(!calls
            .lock()
            .map_err(|_| "lock poisoned")?
            .contains(&"judge/haiku".to_string()));
        // Trace: [candidate, winner] — no judge entry.
        assert_eq!(trace.len(), 2);
        assert_eq!(as_ensemble(&trace[1])?.phase, EnsemblePhase::Winner);
        Ok(())
    }

    #[tokio::test]
    async fn zero_exploration_turns_never_commits() -> Result<(), Box<dyn Error + Send + Sync>> {
        // exploration_turns == 0 keeps ensembling forever.
        let (algo, calls) = algo(&["a/model", "b/model"], "judge/haiku", "b/model", 0);
        let orch = orch(algo);
        for _ in 0..3 {
            let (trace, _) = orch.clone().run(Context::default(), request("x")).await?;
            // Always a full ensemble turn (never a lone Committed decision).
            assert_eq!(
                as_ensemble(&trace[trace.len() - 1])?.phase,
                EnsemblePhase::Winner
            );
        }
        // Judge ran on every turn.
        assert_eq!(
            calls
                .lock()
                .map_err(|_| "lock poisoned")?
                .iter()
                .filter(|c| *c == "judge/haiku")
                .count(),
            3
        );
        Ok(())
    }

    #[tokio::test]
    async fn all_candidates_failing_errors() -> Result<(), Box<dyn Error + Send + Sync>> {
        /// Client whose every call fails.
        struct FailingClient;
        #[async_trait]
        impl LlmClient for FailingClient {
            async fn call(
                &self,
                _routed: RoutedRequest,
            ) -> Result<Response, Box<dyn Error + Send + Sync>> {
                Err("upstream down".into())
            }
        }
        let client = Arc::new(FailingClient) as Arc<dyn LlmClient>;
        let target = |name: &str| LlmTarget {
            semantic_name: name.to_string(),
            llm_client: Some(client.clone()),
        };
        let algo = EnsembleOrchAlgo::new(
            vec!["a/model".to_string(), "b/model".to_string()],
            "judge/haiku",
            100,
            LlmTargetSet::new(vec![
                target("a/model"),
                target("b/model"),
                target("judge/haiku"),
            ]),
        );
        assert!(orch(algo)
            .run(Context::default(), request("x"))
            .await
            .is_err());
        Ok(())
    }

    #[tokio::test]
    async fn process_signals_is_a_noop() -> Result<(), Box<dyn Error + Send + Sync>> {
        let (algo, _) = algo(&["a/model"], "judge/haiku", "a/model", 1);
        Arc::new(algo).process_signals(Signals {}).await?;
        Ok(())
    }

    #[test]
    fn parse_choice_reads_first_number_and_fails_open() {
        assert_eq!(parse_choice("2", 3), 1);
        assert_eq!(parse_choice("Response 3 is best", 3), 2);
        assert_eq!(parse_choice("the winner is 1", 3), 0);
        // Out of range and unparseable both fall open to the first response.
        assert_eq!(parse_choice("7", 3), 0);
        assert_eq!(parse_choice("none", 3), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn two_sessions_process_in_parallel() -> Result<(), Box<dyn Error + Send + Sync>> {
        use std::time::Duration;
        use tokio::sync::Barrier;

        // Candidate calls block on a shared barrier; judge calls do not. `run` serves
        // offloaded calls concurrently, so each session has both its candidate calls in
        // flight at once — 2 sessions x 2 candidates = 4 concurrent candidate calls. The
        // barrier releases only when all four have arrived; if calls were serialized
        // (within a session or across sessions), it would never reach 4 and the test
        // would time out instead of passing.
        struct BarrierClient {
            barrier: Arc<Barrier>,
            judge_model: String,
            prefer: String,
        }

        #[async_trait]
        impl LlmClient for BarrierClient {
            async fn call(
                &self,
                routed: RoutedRequest,
            ) -> Result<Response, Box<dyn Error + Send + Sync>> {
                let name = routed.decision.selected_model().to_string();
                let completion = if name == self.judge_model {
                    // Judge runs after the barrier releases; it must not wait.
                    judge_pick(&prompt_text(&routed.request.llm_request), &self.prefer)
                } else {
                    // Hold every candidate call until all sessions have fanned out.
                    self.barrier.wait().await;
                    format!("answer from {name}")
                };
                Ok(Response {
                    llm_response: LlmResponse::Agg(text_response(None, completion)),
                    metadata: None,
                })
            }
        }

        const CANDIDATES_PER_SESSION: usize = 2;
        const SESSIONS: usize = 2;
        let barrier = Arc::new(Barrier::new(CANDIDATES_PER_SESSION * SESSIONS));
        let client = Arc::new(BarrierClient {
            barrier: barrier.clone(),
            judge_model: "judge/haiku".to_string(),
            prefer: "a/model".to_string(),
        }) as Arc<dyn LlmClient>;

        // Two independent sessions: separate algo instances, each with its own
        // per-session state, sharing only the backend client.
        let session_a: Arc<dyn Algorithm> = Arc::new(algo_with_client(
            &["a/model", "b/model"],
            "judge/haiku",
            100,
            client.clone(),
        ));
        let session_b: Arc<dyn Algorithm> = Arc::new(algo_with_client(
            &["a/model", "b/model"],
            "judge/haiku",
            100,
            client.clone(),
        ));

        let run = |session: Arc<dyn Algorithm>, prompt: &'static str| {
            tokio::spawn(async move {
                session
                    .run(Context::default(), request(prompt))
                    .await
                    .map(|(_, response)| {
                        response
                            .llm_response
                            .agg()
                            .map(completion_text)
                            .unwrap_or_default()
                    })
            })
        };
        let handle_a = run(session_a, "from A");
        let handle_b = run(session_b, "from B");

        // The timeout converts a serialization deadlock into a failure, not a hang.
        let completion_a = tokio::time::timeout(Duration::from_secs(5), handle_a).await???;
        let completion_b = tokio::time::timeout(Duration::from_secs(5), handle_b).await???;
        assert_eq!(completion_a, "answer from a/model");
        assert_eq!(completion_b, "answer from a/model");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn two_parallel_sessions_keep_independent_state(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Two sessions run concurrently with judges that prefer different models:
        // session A prefers a/model, session B prefers b/model. Each explores for
        // two turns then commits; because the win tally is per-session, they must
        // commit to *different* models — proving no state leaks between sessions.
        let (session_a, _) = algo(&["a/model", "b/model"], "judge/haiku", "a/model", 2);
        let (session_b, _) = algo(&["a/model", "b/model"], "judge/haiku", "b/model", 2);

        // Drive one session's three requests sequentially (so its two exploration
        // turns complete before the committing third), returning that third
        // request's winning model and decision phase.
        let drive = |session: Arc<dyn Algorithm>| {
            tokio::spawn(async move {
                session
                    .clone()
                    .run(Context::default(), request("t1"))
                    .await?;
                session
                    .clone()
                    .run(Context::default(), request("t2"))
                    .await?;
                let (trace, response) = session
                    .clone()
                    .run(Context::default(), request("t3"))
                    .await?;
                let phase = trace
                    .last()
                    .and_then(|d| d.as_any().downcast_ref::<EnsembleDecision>())
                    .map(|d| d.phase)
                    .ok_or("missing final decision")?;
                Ok::<(String, EnsemblePhase), Box<dyn Error + Send + Sync>>((
                    response
                        .llm_response
                        .agg()
                        .map(completion_text)
                        .unwrap_or_default(),
                    phase,
                ))
            })
        };
        // The two sessions run in parallel; each committed independently.
        let handle_a = drive(orch(session_a));
        let handle_b = drive(orch(session_b));
        let (completion_a, phase_a) = handle_a.await??;
        let (completion_b, phase_b) = handle_b.await??;

        assert_eq!(phase_a, EnsemblePhase::Committed);
        assert_eq!(completion_a, "answer from a/model");
        assert_eq!(phase_b, EnsemblePhase::Committed);
        assert_eq!(completion_b, "answer from b/model");
        Ok(())
    }
}
