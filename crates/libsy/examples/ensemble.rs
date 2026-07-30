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
//! Run with:
//!   NVIDIA_API_KEY=... cargo run -p switchyard-libsy --example ensemble
//!
//! Unlike the reference routers, this algorithm is **stateful**: the win tally,
//! the claimed exploration count, and the committed choice live behind a
//! [`parking_lot::Mutex`] so one shared `&self` can serve a session's
//! requests concurrently (see the `Algorithm` docs); a [`tokio::sync::Notify`]
//! wakes a request waiting for the exploration budget to finish. In a proxy setup
//! one `EnsembleOrchAlgo` is created per session, so this state is per-session.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;

use tokio::sync::Notify;

use switchyard_libsy::{Algorithm, Driver, LibsyError, LlmTarget, LlmTargetSet, Result};
use switchyard_llm_client::{Backend, HttpBackendConfig, ModelConfig, TranslatingLlmClient};
use switchyard_protocol::{
    Context, Decision, LlmResponse, Request, Response, RoutedLlmClient, completion_text,
    prompt_text, text_request,
};

const CANDIDATE_MODELS: [&str; 3] = [
    "nvidia/qwen/qwen3.6-27b",
    "nvidia/nvidia/nemotron-3-super-v3",
    "nvidia/openai/gpt-oss-20b",
];
const JUDGE_MODEL: &str = "nvidia/deepseek-ai/deepseek-v4-flash";

fn ensemble_error(message: &'static str) -> LibsyError {
    LibsyError::external("ensemble", std::io::Error::other(message))
}

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
    /// Running or completed exploration turns. Failed turns return their slot.
    used_slots: u64,
    /// The model committed to once exploration is over; `None` while exploring.
    committed: Option<String>,
}

/// How one request is served: straight to the committed model, or as a fresh
/// exploration turn (a slot was reserved for it under the lock).
enum RoutePlan {
    Committed(String),
    Explore,
}

/// Holds a reserved exploration slot for the length of one turn and returns it if
/// the turn is dropped before it tallies a winner — the run future aborted on a
/// client disconnect, a panic, or an error. A turn that completes calls
/// [`disarm`](ReservationGuard::disarm) and keeps its now-counted slot.
struct ReservationGuard<'a> {
    algo: &'a EnsembleOrchAlgo,
    armed: bool,
}

impl ReservationGuard<'_> {
    /// Give up the claim on the slot: the turn completed and counted it, so it
    /// must not be released.
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for ReservationGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.algo.release_reservation();
        }
    }
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
    /// Notified whenever a reserved exploration finishes, so a request waiting
    /// for the exploration budget to drain can recheck and commit.
    explorations_done: Notify,
}

impl EnsembleOrchAlgo {
    /// Create an ensemble over `candidate_models`, judged by `judge_model`,
    /// exploring for `exploration_turns` before committing to the winningest
    /// candidate (`0` = never commit, ensemble every request), routing among
    /// `target_set`. Wrap it in an [`Arc`](std::sync::Arc) and drive it with
    /// [`run`](switchyard_libsy::Algorithm::run) or
    /// [`run_stream`](switchyard_libsy::Algorithm::run_stream).
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
                used_slots: 0,
                committed: None,
            }),
            explorations_done: Notify::new(),
        }
    }

    /// Decide how to serve one request, reserving an exploration slot under the
    /// lock before any async work begins. This caps concurrent explorations at
    /// `exploration_turns` and commits only from a complete tally:
    ///
    /// - already committed → route to that model;
    /// - a slot is free (`used_slots < exploration_turns`, or unlimited when
    ///   `exploration_turns == 0`) → reserve it and explore;
    /// - the budget is fully reserved but some explorations are still running →
    ///   wait for them rather than commit early or start an extra turn;
    /// - the budget is fully reserved and every reservation has been tallied →
    ///   commit to the winningest model.
    ///
    /// The lock is only held for the synchronous decision; the wait happens after
    /// it is released, with the wakeup registered first so no completion is missed.
    async fn plan_route(&self) -> Result<RoutePlan> {
        loop {
            let wait = self.explorations_done.notified();
            tokio::pin!(wait);
            {
                let mut state = self.state.lock();
                if let Some(model) = &state.committed {
                    return Ok(RoutePlan::Committed(model.clone()));
                }
                // `exploration_turns == 0` keeps the algorithm ensembling forever.
                if self.exploration_turns == 0 || state.used_slots < self.exploration_turns {
                    state.used_slots += 1;
                    return Ok(RoutePlan::Explore);
                }
                if state.wins.values().sum::<u64>() >= self.exploration_turns {
                    let best = self.pick_best(&state.wins)?;
                    state.committed = Some(best.clone());
                    return Ok(RoutePlan::Committed(best));
                }
                // Register for the next completion before dropping the lock so a
                // notify between here and the await below is not lost.
                wait.as_mut().enable();
            }
            wait.await;
        }
    }

    /// Return an exploration slot whose turn failed without tallying a winner, so
    /// a waiting request can take it and the budget is not spent by a failed turn.
    fn release_reservation(&self) {
        self.state.lock().used_slots -= 1;
        self.explorations_done.notify_waiters();
    }

    /// The candidate with the most judge-wins, breaking ties toward the earlier
    /// candidate (stable and deterministic). Errors only if there are no
    /// candidates configured.
    fn pick_best(&self, wins: &BTreeMap<String, u64>) -> Result<String> {
        let mut best = self.candidate_models.first().ok_or(LibsyError::NoTargets)?;
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
    ) -> Result<(Vec<Arc<dyn Decision>>, Response)> {
        let target = self.target_set.get_target(&model)?;
        let decision: Arc<dyn Decision> = Arc::new(EnsembleDecision {
            reasoning: format!(
                "committed to '{model}' after {} turns",
                self.exploration_turns
            ),
            selected_model: model.clone(),
            phase: EnsemblePhase::Committed,
        });
        // Forward the caller's complete request unchanged; the committed model is
        // carried on the decision, not stamped onto the request.
        let response = driver
            .call_llm_target(ctx, &target, request, decision.clone())
            .await?;
        Ok((vec![decision], response))
    }

    /// Call all candidates concurrently and buffer successful responses. A call
    /// or stream error removes only that candidate from the turn.
    async fn fan_out_candidates(
        &self,
        driver: &Driver,
        ctx: &Context,
        request: &Request,
    ) -> Result<(Vec<Arc<dyn Decision>>, Vec<(String, Response)>)> {
        let mut decisions: Vec<Arc<dyn Decision>> = Vec::new();
        let mut calls = Vec::new();
        for model in &self.candidate_models {
            let target = self.target_set.get_target(model)?;
            let decision: Arc<dyn Decision> = Arc::new(EnsembleDecision {
                selected_model: model.clone(),
                reasoning: format!("ensemble candidate '{model}'"),
                phase: EnsemblePhase::Candidate,
            });
            decisions.push(decision.clone());
            let request = request.clone();
            let model = model.clone();
            let ctx = ctx.clone();
            calls.push(async move {
                // Drain each stream inside its candidate future so slow streams do
                // not block the other candidates from making progress.
                let Response {
                    llm_response,
                    metadata,
                } = driver
                    .call_llm_target(ctx, &target, request, decision)
                    .await
                    .ok()?;
                let llm_response = LlmResponse::Agg(llm_response.into_agg().await.ok()?);
                Some((
                    model,
                    Response {
                        llm_response,
                        metadata,
                    },
                ))
            });
        }

        let survivors = futures::future::join_all(calls)
            .await
            .into_iter()
            .flatten()
            .collect();
        Ok((decisions, survivors))
    }

    /// One exploration turn: fan out to every candidate, judge the survivors,
    /// tally the winner, and return its response.
    async fn ensemble_turn(
        &self,
        driver: &Driver,
        ctx: Context,
        request: Request,
    ) -> Result<(Vec<Arc<dyn Decision>>, Response)> {
        let user_prompt = prompt_text(&request.llm_request);
        // Keep the caller's model ID; each decision selects the actual target.
        let inbound = request.llm_request.model.clone();

        let (candidate_decisions, survivors) =
            self.fan_out_candidates(driver, &ctx, &request).await?;
        if survivors.is_empty() {
            return Err(ensemble_error("all candidate calls failed"));
        }

        // Pick the winner: judge only when there is a real choice to make.
        let (winner_model, winner_response, judge_decision) = if survivors.len() == 1 {
            let (model, response) = survivors
                .into_iter()
                .next()
                .ok_or_else(|| ensemble_error("survivor unexpectedly missing"))?;
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
                llm_request: text_request(inbound, judge_prompt),
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
                    .as_agg()
                    .map(completion_text)
                    .unwrap_or_default(),
                survivors.len(),
            );
            let (model, response) = survivors
                .into_iter()
                .nth(choice)
                .ok_or_else(|| ensemble_error("judge choice out of range"))?;
            (model, response, Some(judge_decision))
        };

        *self
            .state
            .lock()
            .wins
            .entry(winner_model.clone())
            .or_insert(0) += 1;
        self.explorations_done.notify_waiters();

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
                .as_agg()
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
    fn name(&self) -> &str {
        "ensemble"
    }

    async fn create_run_task(
        self: Arc<Self>,
        ctx: Context,
        driver: Driver,
        request: Request,
    ) -> Result<Response> {
        // Reserve the route (committed model or a fresh exploration slot) before any
        // async work, then serve it. A reserved exploration that fails releases its
        // slot so the budget is not spent by a failed turn.
        let (trace, response) = match self.plan_route().await? {
            RoutePlan::Committed(model) => {
                self.route_committed(&driver, ctx.clone(), request, model)
                    .await?
            }
            RoutePlan::Explore => {
                // Hold the reserved slot across the turn: an abort or panic
                // mid-turn returns it through the guard's `Drop`, and an error
                // returns it on the way out. A turn that completes disarms the
                // guard and keeps its now-counted slot.
                let mut guard = ReservationGuard {
                    algo: &self,
                    armed: true,
                };
                let result = self.ensemble_turn(&driver, ctx.clone(), request).await?;
                guard.disarm();
                result
            }
        };
        // Publish the trace to the stream (candidate..., judge?, winner). The
        // candidate decisions were also attached to their `CallLlm` steps.
        for decision in trace {
            driver.info(ctx.clone(), decision).await?;
        }
        Ok(response)
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let backend = HttpBackendConfig {
        base_url: std::env::var("NVIDIA_BASE_URL")
            .unwrap_or_else(|_| "https://inference-api.nvidia.com/v1".to_string()),
        api_key: Some(
            std::env::var("NVIDIA_API_KEY")
                .map_err(|error| LibsyError::external("reading NVIDIA_API_KEY", error))?,
        ),
        extra_headers: BTreeMap::new(),
        extra_body: BTreeMap::new(),
        max_retries: 2,
    };
    let models: Vec<_> = CANDIDATE_MODELS
        .iter()
        .copied()
        .chain(std::iter::once(JUDGE_MODEL))
        .map(|model| ModelConfig::new(model, Backend::OpenAiChat(backend.clone()), None))
        .collect();
    let client = Arc::new(
        TranslatingLlmClient::new(&models)
            .map_err(|error| LibsyError::external("building LLM client", error))?,
    ) as Arc<dyn RoutedLlmClient>;
    let targets = LlmTargetSet::new(
        CANDIDATE_MODELS
            .iter()
            .copied()
            .chain(std::iter::once(JUDGE_MODEL))
            .map(|model| LlmTarget {
                semantic_name: model.to_string(),
                llm_client: Some(client.clone()),
            })
            .collect(),
    );
    let algorithm: Arc<dyn Algorithm> = Arc::new(EnsembleOrchAlgo::new(
        CANDIDATE_MODELS.map(str::to_string).to_vec(),
        JUDGE_MODEL,
        1,
        targets,
    ));
    let request = Request {
        llm_request: text_request(
            Some("ensemble".to_string()),
            "Explain how LLM routing improves reliability.",
        ),
        raw_request: None,
        metadata: None,
    };

    let (_, response) = algorithm.run(Context::default(), request).await?;
    println!(
        "{}",
        completion_text(
            &response
                .llm_response
                .into_agg()
                .await
                .map_err(|error| LibsyError::external("aggregating response", error))?,
        )
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    use futures::StreamExt;
    use switchyard_libsy::LlmTarget;
    use switchyard_protocol::{
        LlmRequest, LlmResponse, LlmResponseChunk, LlmResponseStreamEvent, Message, Response, Role,
        RoutedLlmClient, SamplingParams, Signals, ToolChoice, ToolDefinition, completion_text,
        prompt_text, text_request, text_response,
    };
    use tokio::sync::Semaphore;

    /// Mock client that answers candidate calls with `answer from {model}` and,
    /// for the judge model, returns the 1-based number of the response whose
    /// content mentions `prefer` (so a test controls which candidate "wins").
    /// Records every model it was called with for call-count assertions.
    struct JudgingClient {
        judge_model: String,
        prefer: String,
        calls: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl RoutedLlmClient for JudgingClient {
        async fn call(
            &self,
            _ctx: Context,
            request: Request,
            decision: Arc<dyn Decision>,
        ) -> std::result::Result<Response, switchyard_protocol::LlmClientError> {
            let name = decision.selected_model().to_string();
            self.calls.lock().push(name.clone());
            let completion = if name == self.judge_model {
                judge_pick(&prompt_text(&request.llm_request), &self.prefer)
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
    ) -> (EnsembleOrchAlgo, Arc<Mutex<Vec<String>>>) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let client = Arc::new(JudgingClient {
            judge_model: judge.to_string(),
            prefer: prefer.to_string(),
            calls: Arc::clone(&calls),
        }) as Arc<dyn RoutedLlmClient>;
        let algo = algo_with_client(candidates, judge, exploration_turns, client);
        (algo, calls)
    }

    /// Build an ensemble algo whose candidate + judge targets all share `client`.
    /// One such algo models a single session's stateful router.
    fn algo_with_client(
        candidates: &[&str],
        judge: &str,
        exploration_turns: u64,
        client: Arc<dyn RoutedLlmClient>,
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

    fn as_ensemble(d: &Arc<dyn Decision>) -> Result<&EnsembleDecision> {
        d.as_any()
            .downcast_ref::<EnsembleDecision>()
            .ok_or_else(|| {
                switchyard_libsy::DriverError::TypeMismatch {
                    expected: "EnsembleDecision",
                }
                .into()
            })
    }

    #[tokio::test]
    async fn exploration_turn_fans_out_judges_and_returns_the_winner() -> Result<()> {
        // Judge prefers b/model; it should win and be returned.
        let (algo, calls) = algo(&["a/model", "b/model"], "judge/haiku", "b/model", 100);
        let (trace, response) = orch(algo)
            .run(Context::default(), request("solve it"))
            .await?;
        assert_eq!(
            response
                .llm_response
                .as_agg()
                .map(completion_text)
                .unwrap_or_default(),
            "answer from b/model"
        );

        // Both candidates and the judge were called.
        let calls = calls.lock();
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
    async fn commits_to_the_winningest_model_after_exploration() -> Result<()> {
        // Judge always prefers b/model over 2 exploration turns, so the algo
        // commits to b/model even though a/model is listed first.
        let (algo, calls) = algo(&["a/model", "b/model"], "judge/haiku", "b/model", 2);
        let orch = orch(algo);

        // Two exploration turns.
        orch.clone().run(Context::default(), request("t1")).await?;
        orch.clone().run(Context::default(), request("t2")).await?;
        let judge_calls_after_exploration =
            calls.lock().iter().filter(|c| *c == "judge/haiku").count();
        assert_eq!(judge_calls_after_exploration, 2);

        // Third request: committed fast path — routes straight to b/model with no
        // fan-out to a/model and no judge call.
        let (trace, response) = orch.clone().run(Context::default(), request("t3")).await?;
        assert_eq!(
            response
                .llm_response
                .as_agg()
                .map(completion_text)
                .unwrap_or_default(),
            "answer from b/model"
        );
        assert_eq!(trace.len(), 1);
        let decision = as_ensemble(&trace[0])?;
        assert_eq!(decision.phase, EnsemblePhase::Committed);
        assert_eq!(decision.selected_model, "b/model");

        let calls = calls.lock();
        // Judge was not called again on the committed turn.
        assert_eq!(calls.iter().filter(|c| *c == "judge/haiku").count(), 2);
        // a/model was called only on the two exploration turns, not the third.
        assert_eq!(calls.iter().filter(|c| *c == "a/model").count(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn single_candidate_skips_the_judge() -> Result<()> {
        let (algo, calls) = algo(&["only/model"], "judge/haiku", "only/model", 100);
        let (trace, response) = orch(algo).run(Context::default(), request("hi")).await?;
        assert_eq!(
            response
                .llm_response
                .as_agg()
                .map(completion_text)
                .unwrap_or_default(),
            "answer from only/model"
        );
        // No judge call for a lone candidate.
        assert!(!calls.lock().contains(&"judge/haiku".to_string()));
        // Trace: [candidate, winner] — no judge entry.
        assert_eq!(trace.len(), 2);
        assert_eq!(as_ensemble(&trace[1])?.phase, EnsemblePhase::Winner);
        Ok(())
    }

    #[tokio::test]
    async fn zero_exploration_turns_never_commits() -> Result<()> {
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
            calls.lock().iter().filter(|c| *c == "judge/haiku").count(),
            3
        );
        Ok(())
    }

    #[tokio::test]
    async fn all_candidates_failing_errors() -> Result<()> {
        /// Client whose every call fails.
        struct FailingClient;
        #[async_trait]
        impl RoutedLlmClient for FailingClient {
            async fn call(
                &self,
                _ctx: Context,
                _request: Request,
                _decision: Arc<dyn Decision>,
            ) -> std::result::Result<Response, switchyard_protocol::LlmClientError> {
                Err(switchyard_protocol::LlmClientError::General(
                    "upstream down".into(),
                ))
            }
        }
        let client = Arc::new(FailingClient) as Arc<dyn RoutedLlmClient>;
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
        assert!(
            orch(algo)
                .run(Context::default(), request("x"))
                .await
                .is_err()
        );
        Ok(())
    }

    #[tokio::test]
    async fn process_signals_is_a_noop() -> Result<()> {
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
    async fn two_sessions_process_in_parallel() -> Result<()> {
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
        impl RoutedLlmClient for BarrierClient {
            async fn call(
                &self,
                _ctx: Context,
                request: Request,
                decision: Arc<dyn Decision>,
            ) -> std::result::Result<Response, switchyard_protocol::LlmClientError> {
                let name = decision.selected_model().to_string();
                let completion = if name == self.judge_model {
                    // Judge runs after the barrier releases; it must not wait.
                    judge_pick(&prompt_text(&request.llm_request), &self.prefer)
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
        }) as Arc<dyn RoutedLlmClient>;

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
                            .as_agg()
                            .map(completion_text)
                            .unwrap_or_default()
                    })
            })
        };
        let handle_a = run(session_a, "from A");
        let handle_b = run(session_b, "from B");

        // The timeout converts a serialization deadlock into a failure, not a hang.
        let completion_a = tokio::time::timeout(Duration::from_secs(5), handle_a)
            .await
            .map_err(|error| LibsyError::external("waiting for session A", error))???;
        let completion_b = tokio::time::timeout(Duration::from_secs(5), handle_b)
            .await
            .map_err(|error| LibsyError::external("waiting for session B", error))???;
        assert_eq!(completion_a, "answer from a/model");
        assert_eq!(completion_b, "answer from a/model");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn two_parallel_sessions_keep_independent_state() -> Result<()> {
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
                    .ok_or_else(|| ensemble_error("missing final decision"))?;
                Ok::<(String, EnsemblePhase), LibsyError>((
                    response
                        .llm_response
                        .as_agg()
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

    /// Records the `LlmRequest` of every candidate call; answers candidates with
    /// `answer from {model}` and the judge with "1".
    struct CapturingClient {
        judge_model: String,
        candidate_requests: Arc<Mutex<Vec<LlmRequest>>>,
    }

    #[async_trait]
    impl RoutedLlmClient for CapturingClient {
        async fn call(
            &self,
            _ctx: Context,
            request: Request,
            decision: Arc<dyn Decision>,
        ) -> std::result::Result<Response, switchyard_protocol::LlmClientError> {
            let name = decision.selected_model().to_string();
            let completion = if name == self.judge_model {
                "1".to_string()
            } else {
                self.candidate_requests
                    .lock()
                    .push(request.llm_request.clone());
                format!("answer from {name}")
            };
            Ok(Response {
                llm_response: LlmResponse::Agg(text_response(None, completion)),
                metadata: None,
            })
        }
    }

    #[tokio::test]
    async fn candidate_and_committed_calls_preserve_the_full_request() -> Result<()> {
        // A request a single text prompt would flatten away: system + user
        // messages, a tool, a tool_choice, non-default sampling, and stream = true.
        let original = LlmRequest {
            model: Some("auto".to_string()),
            messages: vec![
                Message::text(Role::System, "be terse"),
                Message::text(Role::User, "add 2 and 2"),
            ],
            tools: vec![ToolDefinition {
                name: "calc".to_string(),
                description: Some("do math".to_string()),
                parameters: serde_json::json!({"type": "object"}),
                strict: Some(true),
            }],
            tool_choice: Some(ToolChoice::Required),
            sampling: SamplingParams {
                temperature: Some(0.3),
                ..Default::default()
            },
            stream: true,
            ..Default::default()
        };
        let candidate_requests = Arc::new(Mutex::new(Vec::new()));
        let client = Arc::new(CapturingClient {
            judge_model: "judge/haiku".to_string(),
            candidate_requests: Arc::clone(&candidate_requests),
        }) as Arc<dyn RoutedLlmClient>;
        let session = orch(algo_with_client(
            &["a/model", "b/model"],
            "judge/haiku",
            1,
            client,
        ));
        for _ in 0..2 {
            session
                .clone()
                .run(
                    Context::default(),
                    Request {
                        llm_request: original.clone(),
                        raw_request: None,
                        metadata: None,
                    },
                )
                .await?;
        }

        // The first request reaches both candidates. The second request reaches
        // the committed model, and all three calls keep the original fields.
        let captured = candidate_requests.lock();
        assert_eq!(captured.len(), 3);
        for request in captured.iter() {
            assert_eq!(request, &original);
        }
        Ok(())
    }

    /// Candidate calls stream their answer as text deltas; the judge records the
    /// prompt it saw and picks the response mentioning `prefer`.
    struct StreamingCandidatesClient {
        judge_model: String,
        prefer: String,
        judge_prompts: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl RoutedLlmClient for StreamingCandidatesClient {
        async fn call(
            &self,
            _ctx: Context,
            request: Request,
            decision: Arc<dyn Decision>,
        ) -> std::result::Result<Response, switchyard_protocol::LlmClientError> {
            let name = decision.selected_model().to_string();
            if name == self.judge_model {
                let prompt = prompt_text(&request.llm_request);
                self.judge_prompts.lock().push(prompt.clone());
                return Ok(Response {
                    llm_response: LlmResponse::Agg(text_response(
                        None,
                        judge_pick(&prompt, &self.prefer),
                    )),
                    metadata: None,
                });
            }
            // Candidate answers arrive as a token stream, not a buffered response.
            let answer = format!("answer from {name}");
            let stream = futures::stream::iter(
                [LlmResponseStreamEvent::from(LlmResponseChunk::TextDelta {
                    index: 0,
                    text: answer,
                })]
                .into_iter()
                .map(Ok),
            )
            .boxed();
            Ok(Response {
                llm_response: LlmResponse::Stream(stream),
                metadata: None,
            })
        }
    }

    #[tokio::test]
    async fn streamed_candidates_are_judged_on_their_real_text() -> Result<()> {
        let judge_prompts = Arc::new(Mutex::new(Vec::new()));
        let client = Arc::new(StreamingCandidatesClient {
            judge_model: "judge/haiku".to_string(),
            prefer: "b/model".to_string(),
            judge_prompts: Arc::clone(&judge_prompts),
        }) as Arc<dyn RoutedLlmClient>;
        let algo = algo_with_client(&["a/model", "b/model"], "judge/haiku", 100, client);
        let (_, response) = orch(algo)
            .run(Context::default(), request("solve it"))
            .await?;

        // The judge saw both streamed candidate answers, so it could choose on
        // content rather than fall open to the first response on empty text.
        let prompt = judge_prompts.lock().first().cloned().unwrap_or_default();
        assert!(
            prompt.contains("answer from a/model"),
            "judge prompt: {prompt}"
        );
        assert!(
            prompt.contains("answer from b/model"),
            "judge prompt: {prompt}"
        );
        // The winner is returned as buffered text, not lost with its stream.
        assert_eq!(
            response
                .llm_response
                .as_agg()
                .map(completion_text)
                .unwrap_or_default(),
            "answer from b/model"
        );
        Ok(())
    }

    /// Candidate calls announce arrival then block on `gate` until the test opens
    /// it; the judge answers immediately, preferring `prefer`.
    struct GatedCandidateClient {
        judge_model: String,
        prefer: String,
        calls: Arc<Mutex<Vec<String>>>,
        arrivals: Arc<Semaphore>,
        gate: Arc<Semaphore>,
    }

    #[async_trait]
    impl RoutedLlmClient for GatedCandidateClient {
        async fn call(
            &self,
            _ctx: Context,
            request: Request,
            decision: Arc<dyn Decision>,
        ) -> std::result::Result<Response, switchyard_protocol::LlmClientError> {
            let name = decision.selected_model().to_string();
            self.calls.lock().push(name.clone());
            if name == self.judge_model {
                let completion = judge_pick(&prompt_text(&request.llm_request), &self.prefer);
                return Ok(Response {
                    llm_response: LlmResponse::Agg(text_response(None, completion)),
                    metadata: None,
                });
            }
            // Announce this candidate has arrived, then wait for the test to release.
            self.arrivals.add_permits(1);
            if let Ok(permit) = self.gate.acquire().await {
                permit.forget();
            }
            Ok(Response {
                llm_response: LlmResponse::Agg(text_response(None, format!("answer from {name}"))),
                metadata: None,
            })
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_requests_do_not_exceed_the_exploration_budget() -> Result<()> {
        let calls = Arc::new(Mutex::new(Vec::<String>::new()));
        let arrivals = Arc::new(Semaphore::new(0));
        let gate = Arc::new(Semaphore::new(0));
        let client = Arc::new(GatedCandidateClient {
            judge_model: "judge/haiku".to_string(),
            prefer: "b/model".to_string(),
            calls: Arc::clone(&calls),
            arrivals: Arc::clone(&arrivals),
            gate: Arc::clone(&gate),
        }) as Arc<dyn RoutedLlmClient>;
        // exploration_turns = 1: exactly one turn may explore before committing.
        let algo: Arc<dyn Algorithm> = Arc::new(algo_with_client(
            &["a/model", "b/model"],
            "judge/haiku",
            1,
            client,
        ));

        // Request A reserves the only slot and fans out; its candidate calls block.
        let handle_a = tokio::spawn({
            let algo = algo.clone();
            async move { algo.run(Context::default(), request("A")).await }
        });
        // Wait for A's two candidates to reach the gate — A has reserved the slot.
        let arrived = arrivals
            .acquire_many(2)
            .await
            .map_err(|_| ensemble_error("arrivals semaphore closed"))?;
        arrived.forget();

        // Request B arrives while A is still exploring. With the budget reserved it
        // must wait for A, not start a second exploration or commit early.
        let handle_b = tokio::spawn({
            let algo = algo.clone();
            async move { algo.run(Context::default(), request("B")).await }
        });

        // Open the gate for every remaining call: A's two candidates, and B's later
        // committed route call. The reservation logic, not the gate, is what bounds
        // exploration — so if B wrongly fanned out, its candidate calls would pass
        // here and the counts below would catch it.
        gate.add_permits(100);
        handle_a.await??;
        let (b_trace, _) = handle_b.await??;

        let calls = calls.lock();
        // Exactly one exploration ran: the judge was consulted once, and B did not
        // fan out to a/model (A's single exploration is the only fan-out).
        assert_eq!(calls.iter().filter(|c| *c == "judge/haiku").count(), 1);
        assert_eq!(calls.iter().filter(|c| *c == "a/model").count(), 1);
        // b/model: A's exploration fan-out plus B's committed route to the winner.
        assert_eq!(calls.iter().filter(|c| *c == "b/model").count(), 2);

        // B committed from A's complete tally: b/model won A's turn, ruling out a
        // premature commit on the empty tally (which would pick the first
        // candidate, a/model).
        let decision = as_ensemble(
            b_trace
                .last()
                .ok_or_else(|| ensemble_error("B produced no decision"))?,
        )?;
        assert_eq!(decision.phase, EnsemblePhase::Committed);
        assert_eq!(decision.selected_model, "b/model");
        Ok(())
    }

    /// Fails every candidate call while `fail` is set and answers normally once
    /// the test clears it; the judge always answers "1".
    struct FlakyCandidateClient {
        judge_model: String,
        fail: Arc<AtomicBool>,
    }

    #[async_trait]
    impl RoutedLlmClient for FlakyCandidateClient {
        async fn call(
            &self,
            _ctx: Context,
            _request: Request,
            decision: Arc<dyn Decision>,
        ) -> std::result::Result<Response, switchyard_protocol::LlmClientError> {
            let name = decision.selected_model().to_string();
            if name == self.judge_model {
                return Ok(Response {
                    llm_response: LlmResponse::Agg(text_response(None, "1")),
                    metadata: None,
                });
            }
            if self.fail.load(Ordering::SeqCst) {
                return Err(switchyard_protocol::LlmClientError::General(
                    "candidate unavailable".to_string(),
                ));
            }
            Ok(Response {
                llm_response: LlmResponse::Agg(text_response(None, format!("answer from {name}"))),
                metadata: None,
            })
        }
    }

    #[tokio::test]
    async fn a_failed_exploration_frees_the_slot_for_the_next_request() -> Result<()> {
        let fail = Arc::new(AtomicBool::new(true));
        let client = Arc::new(FlakyCandidateClient {
            judge_model: "judge/haiku".to_string(),
            fail: Arc::clone(&fail),
        }) as Arc<dyn RoutedLlmClient>;
        // exploration_turns = 1: a single slot serves the whole session.
        let session = orch(algo_with_client(
            &["a/model", "b/model"],
            "judge/haiku",
            1,
            client,
        ));

        // Request A reserves the only slot and fails every candidate, so its turn
        // errors — and must return the slot instead of spending the budget.
        assert!(
            session
                .clone()
                .run(Context::default(), request("A"))
                .await
                .is_err()
        );

        // With the slot freed, request B can still explore. A leaked reservation
        // would park B forever waiting for the exploration budget to drain, so the
        // timeout turns that deadlock into a test failure instead of a hang.
        fail.store(false, Ordering::SeqCst);
        let (trace, _) = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            session.run(Context::default(), request("B")),
        )
        .await
        .map_err(|_| ensemble_error("request B hung: the failed turn leaked its slot"))??;
        let decision = as_ensemble(
            trace
                .last()
                .ok_or_else(|| ensemble_error("B produced no decision"))?,
        )?;
        assert_eq!(decision.phase, EnsemblePhase::Winner);
        Ok(())
    }

    /// Streams one candidate answer that fails partway through, so aggregating it
    /// errors; answers other candidates normally and records judge calls.
    struct BrokenStreamClient {
        judge_model: String,
        broken: String,
        judge_calls: Arc<Mutex<u32>>,
    }

    #[async_trait]
    impl RoutedLlmClient for BrokenStreamClient {
        async fn call(
            &self,
            _ctx: Context,
            _request: Request,
            decision: Arc<dyn Decision>,
        ) -> std::result::Result<Response, switchyard_protocol::LlmClientError> {
            let name = decision.selected_model().to_string();
            if name == self.judge_model {
                *self.judge_calls.lock() += 1;
                return Ok(Response {
                    llm_response: LlmResponse::Agg(text_response(None, "1")),
                    metadata: None,
                });
            }
            if name == self.broken {
                // A stream that yields text then an error: draining it to an
                // aggregate fails, so this candidate is excluded from judging.
                let stream = futures::stream::iter([
                    Ok(LlmResponseStreamEvent::from(LlmResponseChunk::TextDelta {
                        index: 0,
                        text: "partial".to_string(),
                    })),
                    Err(switchyard_protocol::LlmClientError::General(
                        "stream broke".to_string(),
                    )),
                ])
                .boxed();
                return Ok(Response {
                    llm_response: LlmResponse::Stream(stream),
                    metadata: None,
                });
            }
            Ok(Response {
                llm_response: LlmResponse::Agg(text_response(None, format!("answer from {name}"))),
                metadata: None,
            })
        }
    }

    #[tokio::test]
    async fn a_candidate_whose_stream_fails_is_excluded_not_fatal() -> Result<()> {
        let judge_calls = Arc::new(Mutex::new(0u32));
        let client = Arc::new(BrokenStreamClient {
            judge_model: "judge/haiku".to_string(),
            broken: "a/model".to_string(),
            judge_calls: Arc::clone(&judge_calls),
        }) as Arc<dyn RoutedLlmClient>;
        let (trace, response) = orch(algo_with_client(
            &["a/model", "b/model"],
            "judge/haiku",
            100,
            client,
        ))
        .run(Context::default(), request("solve it"))
        .await?;

        // b/model is the lone survivor, so it wins with no judge call...
        assert_eq!(*judge_calls.lock(), 0);
        // ...and its buffered text is returned, not lost with a/model's stream.
        assert_eq!(
            response
                .llm_response
                .as_agg()
                .map(completion_text)
                .unwrap_or_default(),
            "answer from b/model"
        );
        let winner = as_ensemble(
            trace
                .last()
                .ok_or_else(|| ensemble_error("no winner decision"))?,
        )?;
        assert_eq!(winner.selected_model, "b/model");
        Ok(())
    }
}
