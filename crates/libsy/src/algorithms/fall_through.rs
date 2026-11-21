// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fall-through classifier routing: a composable [`Algorithm`] that routes each turn
//! through a processor chain and a classifier cascade.
//!
//! Each turn: request-side [`Processor`]s fold facts into the composition's state; the
//! [`Classifier`] cascade is consulted in order and the first to score decides the target
//! (its `argmax`); the selected model is replayed to the processors so
//! stateful ones (latch, affinity) can bind it.
//!
//! The default `FallThrough` carries no session registry. Stateful compositions share one
//! private [`State`] across turns with the same session ID. Requests without a session ID run
//! against the caller state on the [`Driver`].
//!
//! The selected target is offered first, then the rest of the category it was drawn from, then
//! every other runtime target. The consumer may fall through that ordered candidate list when a
//! model call fails.

use std::{
    collections::HashMap,
    sync::{Arc, Once, Weak},
    time::{Duration, Instant},
};

use async_trait::async_trait;
use parking_lot::Mutex;
use tokio::sync::Mutex as AsyncMutex;

use crate::algorithms::util::prompts::prepend_system_prompt;
use crate::core::algorithm::{self, Algorithm, AnswerSummary, Driver, RoutingIdentity};
use crate::core::classifier::{Classifier, Score};
use crate::core::processor::{Event, Processor};
use crate::core::state::State;
use crate::{LibsyError, Result, RoutingOutcome};
use switchyard_protocol::{Category, ModelId, Request, Response};

struct SessionState {
    state: Arc<AsyncMutex<State>>,
    last_accessed: Instant,
}

type SessionStates = Mutex<HashMap<RoutingIdentity, SessionState>>;

/// Upper bound on retained in-memory session and child-agent states.
const MAX_SESSIONS: usize = 4096;

/// Delete sessions that have been inactive this long. Catches sessions that did not terminate
/// cleanly.
/// A user resuming a deleted session is not fatal. Algorithms will be missing some context
/// so may route less well for the first turn or two.
const SESSION_STATE_TTL: Duration = Duration::from_secs(60 * 60);

/// Run the expired session cleanup code this often.
const SESSION_CLEANUP_INTERVAL: Duration = Duration::from_secs(60 * 60);

/// Processor chain → classifier cascade → routed model call. See the module docs.
pub struct FallThrough {
    name: String,
    processors: Vec<Arc<dyn Processor>>,
    classifiers: Vec<Arc<dyn Classifier>>,
    session_states: Option<Arc<SessionStates>>,
    cleanup_started: Once,
}

impl FallThrough {
    /// Creates an empty stateless router.
    pub fn new() -> Self {
        Self {
            name: "fall_through".to_string(),
            processors: Vec::new(),
            classifiers: Vec::new(),
            session_states: None,
            cleanup_started: Once::new(),
        }
    }

    /// Creates a router that retains one private [`State`] per session.
    pub fn new_with_state() -> Self {
        Self {
            name: "fall_through".to_string(),
            processors: Vec::new(),
            classifiers: Vec::new(),
            session_states: Some(Arc::new(Mutex::new(HashMap::new()))),
            cleanup_started: Once::new(),
        }
    }

    /// Sets the stable, low-cardinality telemetry name for this composition.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Appends a processor to the head-of-request chain.
    pub fn with_processor(mut self, processor: Arc<dyn Processor>) -> Self {
        self.processors.push(processor);
        self
    }

    /// Appends a classifier to the cascade.
    pub fn with_classifier(mut self, classifier: Arc<dyn Classifier>) -> Self {
        self.classifiers.push(classifier);
        self
    }
    /// Executes the processor and classifier sequence for wrappers and the trait entrypoint.
    pub(crate) async fn execute(&self, driver: Driver, request: Request) -> Result<RoutingOutcome> {
        if RoutingIdentity::from_request(&request).is_some() {
            self.start_cleanup_task();
        }
        let clear_session = request
            .metadata
            .as_ref()
            .filter(|metadata| metadata.session_final == Some(true))
            .and_then(|metadata| metadata.session_id.as_deref())
            .filter(|id| !id.is_empty())
            .map(str::to_owned);
        let result = self.execute_session(driver, request).await;
        if let Some(session) = clear_session.as_deref() {
            self.remove_session(session);
        }
        result
    }

    /// Starts one cleanup task on the first request handled by a stateful router.
    fn start_cleanup_task(&self) {
        let Some(states) = &self.session_states else {
            return;
        };
        let states = Arc::downgrade(states);
        self.cleanup_started.call_once(move || {
            drop(tokio::spawn(cleanup_inactive_sessions(states)));
        });
    }

    async fn execute_session(&self, driver: Driver, request: Request) -> Result<RoutingOutcome> {
        // The request is threaded mutably through the whole fold: any component may rewrite
        // it, later components see the rewrite, and the final value reaches the model.
        let mut request = request;
        let session_state = self.session_state(&request);
        let (score, served, prompts) = match session_state {
            Some(state) => {
                let mut state = state.lock().await;
                let (score, served) = self.route(&mut state, &driver, &mut request).await?;
                (score, served, std::mem::take(&mut state.target_prompts))
            }
            None => {
                let mut state = driver.take_state();
                let (score, served) = self.route(&mut state, &driver, &mut request).await?;
                let prompts = std::mem::take(&mut state.target_prompts);
                driver.set_state(state);
                (score, served, prompts)
            }
        };

        tracing::info!(algorithm=self.name, target=%score.target, confidence=score.confidence, "Model selected");

        // A classifier that already called a model — because deciding required one, and that
        // call also answers the turn — hands its response back here, so the turn is not paid
        // for twice.
        // Nothing reads it on the way out: streamed or buffered, it reaches the caller
        // untouched.
        let target = score.target;
        match served {
            Some(response) => Ok(RoutingOutcome::answered(target, request, response)),
            None => {
                // The rest of the category the decision came from leads: those are the
                // alternatives to the model the classifier picked, in the order it ranked
                // them. Every other runtime target follows as a last resort, so a route
                // whose category holds one model still fails over to the other tier.
                let chosen = driver.models_for(score.category.as_ref().unwrap_or(&Category::Any));
                let mut fallback_models: Vec<ModelId> = Vec::new();
                for candidate in chosen.iter().chain(driver.models_for(&Category::Any)) {
                    if *candidate != target && !fallback_models.contains(candidate) {
                        fallback_models.push(candidate.clone());
                    }
                }
                let mut outcome = RoutingOutcome::route_to(target, fallback_models, request);
                if !prompts.is_empty() {
                    for target in outcome.selected_model_ids.iter().skip(1) {
                        let mut request = outcome.request.clone();
                        request.llm_request.model = Some(target.to_string());
                        if let Some(prompt) = prompts.get(target) {
                            prepend_system_prompt(&mut request.llm_request, prompt);
                        }
                        outcome.candidate_requests.insert(target.clone(), request);
                    }
                    if let Some(prompt) = prompts.get(&outcome.selected_model_ids[0]) {
                        prepend_system_prompt(&mut outcome.request.llm_request, prompt);
                    }
                }
                Ok(outcome)
            }
        }
    }

    /// Drops retained routing state for the root session and its sub-agents once complete.
    fn remove_session(&self, session: &str) {
        if let Some(states) = &self.session_states {
            states.lock().retain(|key, _| key.session_id() != session);
        }
    }

    async fn observe_identity(&self, key: &RoutingIdentity, answer: AnswerSummary) {
        let Some(states) = &self.session_states else {
            return;
        };
        let Some(state) = states
            .lock()
            .get(key)
            .map(|session| Arc::clone(&session.state))
        else {
            return;
        };
        let mut state = state.lock().await;
        for classifier in &self.classifiers {
            classifier.observe_answer(&mut state, answer);
        }
    }

    /// Returns this request's retained state without holding the registry lock.
    fn session_state(&self, request: &Request) -> Option<Arc<AsyncMutex<State>>> {
        self.session_state_at(request, Instant::now(), MAX_SESSIONS)
    }

    fn session_state_at(
        &self,
        request: &Request,
        now: Instant,
        max_sessions: usize,
    ) -> Option<Arc<AsyncMutex<State>>> {
        let states = self.session_states.as_ref()?;
        let identity = RoutingIdentity::from_request(request)?;
        let mut states = states.lock();
        if let Some(session) = states.get_mut(&identity) {
            session.last_accessed = now;
            return Some(Arc::clone(&session.state));
        }
        if states.len() >= max_sessions {
            states.retain(|_, session| {
                Arc::strong_count(&session.state) > 1
                    || now.saturating_duration_since(session.last_accessed) < SESSION_STATE_TTL
            });
        }
        if states.len() >= max_sessions {
            // Every retained slot may be actively in flight; route against ephemeral state
            // rather than growing past the cap or evicting active work.
            let oldest_idle = states
                .iter()
                .filter(|(_, session)| Arc::strong_count(&session.state) == 1)
                .min_by_key(|(_, session)| session.last_accessed)
                .map(|(key, _)| key.clone())?;
            states.remove(&oldest_idle);
        }
        let state = Arc::new(AsyncMutex::new(State::default()));
        states.insert(
            identity,
            SessionState {
                state: Arc::clone(&state),
                last_accessed: now,
            },
        );
        Some(state)
    }

    pub(crate) async fn route(
        &self,
        state: &mut State,
        driver: &Driver,
        request: &mut Request,
    ) -> Result<(Score, Option<Response>)> {
        // 1. Processor chain accumulates request-side facts into the composition's state.
        //    The driver is offered here so a processor may consult a model first.
        for processor in &self.processors {
            let event = Event::Request { request, driver };
            processor.process(state, event).await?;
        }

        // 2. Fall through the cascade: the first classifier to score decides (argmax). The
        //    per-request driver is offered to each — driver-backed classifiers use it.
        let mut routed = None;
        for classifier in &self.classifiers {
            let (scores, response) = classifier.score(state, request, driver).await?;
            if let Some(score) = scores.argmax(false)? {
                // Only the deciding classifier's response answers the turn; an abstaining
                // classifier selected nothing for it to be the answer to.
                routed = Some((score, response));
                break;
            }
        }
        let Some((score, served)) = routed else {
            return Err(LibsyError::AlgorithmError {
                message: "every classifier abstained".to_string(),
            });
        };

        // 3. Resolve the target.
        algorithm::ensure_model_is_target(driver.models_for(&Category::Any), &score.target)?;
        let target = score.target.clone();

        // 4. Post-decision replay: every processor sees the selection so stateful ones
        //    can bind it, and may rewrite the outbound request (e.g. add a target prompt).
        for processor in &self.processors {
            let event = Event::Decision {
                request,
                selected_model_id: &target,
                category: score.category.clone(),
                driver,
            };
            processor.process(state, event).await?;
        }

        Ok((score, served))
    }
}

async fn cleanup_inactive_sessions(states: Weak<SessionStates>) {
    let start = tokio::time::Instant::now() + SESSION_CLEANUP_INTERVAL;
    let mut interval = tokio::time::interval_at(start, SESSION_CLEANUP_INTERVAL);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        interval.tick().await;
        let Some(states) = states.upgrade() else {
            break;
        };
        remove_inactive_sessions(&states, Instant::now(), SESSION_STATE_TTL);
    }
}

fn remove_inactive_sessions(states: &SessionStates, now: Instant, ttl: Duration) {
    states.lock().retain(|_, session| {
        // The registry owns one strong reference. Any others belong to active requests,
        // which must keep sharing this state until their routing work finishes.
        Arc::strong_count(&session.state) > 1
            || now.saturating_duration_since(session.last_accessed) < ttl
    });
}

#[async_trait]
impl Algorithm for FallThrough {
    fn name(&self) -> &str {
        &self.name
    }

    async fn route(self: Arc<Self>, driver: Driver, request: Request) -> Result<RoutingOutcome> {
        self.execute(driver, request).await
    }

    fn tracks_answers(&self) -> bool {
        self.session_states.is_some()
            && self
                .classifiers
                .iter()
                .any(|classifier| classifier.tracks_answers())
    }

    async fn observe_answer(&self, session_id: &str, answer: AnswerSummary) {
        self.observe_identity(&RoutingIdentity::Session(session_id.to_string()), answer)
            .await;
    }

    async fn observe_request_answer(&self, request: &Request, answer: AnswerSummary) {
        let Some(key) = RoutingIdentity::from_request(request) else {
            return;
        };
        self.observe_identity(&key, answer).await;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Classification;
    use crate::algorithms::llm_class::DefaultCategoryClassifier;
    use crate::algorithms::util::prompts;
    use crate::core::testing::{Serve, category_models, echo, reply, test_drive_with_models};
    use switchyard_protocol::{LlmRequest, Message, Metadata, Role, completion_text, text_request};

    #[derive(Debug, thiserror::Error)]
    #[error("{0}")]
    struct TestError(&'static str);

    fn test_error(message: &'static str) -> LibsyError {
        LibsyError::external("test", TestError(message))
    }

    // --- fixtures ----------------------------------------------------------------------

    /// Echoes the routed model name, capturing the request it was handed so a test can
    /// assert on what actually reached the model.
    fn capturing(into: Arc<Mutex<Option<Request>>>) -> impl Serve {
        move |target: ModelId, request: Request| {
            let into = Arc::clone(&into);
            async move {
                *into.lock() = Some(request);
                Ok(reply(target))
            }
        }
    }

    const NOTE: &str = "the previous model was stalling";

    fn target_set(names: &[&str]) -> Vec<ModelId> {
        names.iter().map(|name| ModelId::from(*name)).collect()
    }

    /// A classifier that emits fixed scores (empty = abstain).
    struct FixedClassifier(Vec<Score>);

    #[async_trait]
    impl Classifier for FixedClassifier {
        async fn score(
            &self,
            _state: &mut State,
            _request: &mut Request,
            _driver: &Driver,
        ) -> Result<(Classification, Option<Response>)> {
            Ok((
                Classification::Scores(
                    self.0
                        .iter()
                        .map(|s| Score {
                            confidence: s.confidence,
                            target: s.target.clone(),
                            category: None,
                        })
                        .collect(),
                ),
                None,
            ))
        }
    }

    fn score(target: &str, confidence: f64) -> Score {
        Score {
            confidence,
            target: ModelId::from(target),
            category: None,
        }
    }

    fn fixed(scores: Vec<Score>) -> Arc<dyn Classifier> {
        Arc::new(FixedClassifier(scores))
    }

    fn request() -> Request {
        Request {
            llm_request: LlmRequest {
                model: Some("auto".to_string()),
                messages: vec![Message::text(Role::User, "hi")],
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: Some(Metadata {
                session_id: Some("session-1".to_string()),
                ..Metadata::default()
            }),
        }
    }

    /// Drives a shared router with one request, returning the completion text and selection.
    async fn run_request(
        router: &Arc<FallThrough>,
        request: Request,
        serve: impl Serve,
    ) -> Result<(String, ModelId)> {
        let (selected_model, response) = test_drive_with_models(
            router.clone(),
            request,
            category_models(Category::Any, &["strong", "weak"]),
            serve,
        )
        .await?;
        let text = response
            .llm_response
            .into_agg()
            .await
            .map(|agg| completion_text(&agg))
            .map_err(|error| LibsyError::external("aggregating fall-through response", error))?;
        Ok((text, selected_model))
    }

    /// Drives a shared router through one turn in the default test session.
    async fn run_turn(router: &Arc<FallThrough>, serve: impl Serve) -> Result<(String, ModelId)> {
        run_request(router, request(), serve).await
    }

    /// Drives a fresh router through one turn with a specific `serve`.
    async fn run_with(router: FallThrough, serve: impl Serve) -> Result<(String, ModelId)> {
        run_turn(&Arc::new(router), serve).await
    }

    #[tokio::test]
    async fn a_served_answer_reaches_only_its_existing_session() -> Result<()> {
        struct AnswerAware;
        #[async_trait]
        impl Classifier for AnswerAware {
            async fn score(
                &self,
                state: &mut State,
                _request: &mut Request,
                _driver: &Driver,
            ) -> Result<(Classification, Option<Response>)> {
                let target = if state.extra.get("answered") == Some(&crate::StateValue::Bool(true))
                {
                    "strong"
                } else {
                    "weak"
                };
                Ok((Classification::Scores(vec![score(target, 1.0)]), None))
            }
            fn tracks_answers(&self) -> bool {
                true
            }

            fn observe_answer(&self, state: &mut State, answer: AnswerSummary) {
                state.extra.insert(
                    "answered".into(),
                    crate::StateValue::Bool(answer.final_reply),
                );
            }
        }
        assert!(!FallThrough::new_with_state().tracks_answers());
        assert!(
            !FallThrough::new()
                .with_classifier(Arc::new(AnswerAware))
                .tracks_answers()
        );
        let router = Arc::new(FallThrough::new_with_state().with_classifier(Arc::new(AnswerAware)));
        assert!(router.tracks_answers());
        let (before, _) = run_turn(&router, echo()).await?;
        router
            .observe_answer("unknown", AnswerSummary { final_reply: true })
            .await;
        let (unknown, _) = run_turn(&router, echo()).await?;
        router
            .observe_answer("session-1", AnswerSummary { final_reply: true })
            .await;
        let (after, _) = run_turn(&router, echo()).await?;
        assert_eq!(
            (before.as_str(), unknown.as_str(), after.as_str()),
            ("weak", "weak", "strong")
        );
        assert_eq!(router.session_states.as_ref().unwrap().lock().len(), 1);

        let subagent =
            Arc::new(FallThrough::new_with_state().with_classifier(Arc::new(AnswerAware)));
        let child = |agent: Option<&str>| Request {
            metadata: Some(Metadata {
                session_id: Some("session-1".to_string()),
                agent_id: agent.map(str::to_string),
                is_subagent: true,
                is_delegated_work: true,
                ..Metadata::default()
            }),
            ..request()
        };
        let (root_init, _) = run_turn(&subagent, echo()).await?;
        let (child_a_init, _) = run_request(&subagent, child(Some("child-a")), echo()).await?;
        let (unidentified, _) = run_request(&subagent, child(None), echo()).await?;
        subagent
            .observe_request_answer(&child(None), AnswerSummary { final_reply: true })
            .await;
        subagent
            .observe_request_answer(&child(Some("child-a")), AnswerSummary { final_reply: true })
            .await;
        let (root_after, _) = run_turn(&subagent, echo()).await?;
        let (child_a_after, _) = run_request(&subagent, child(Some("child-a")), echo()).await?;
        let (child_b_after, _) = run_request(&subagent, child(Some("child-b")), echo()).await?;
        assert_eq!(
            (
                root_init.as_str(),
                child_a_init.as_str(),
                unidentified.as_str(),
                root_after.as_str(),
                child_a_after.as_str(),
                child_b_after.as_str(),
            ),
            ("weak", "weak", "weak", "weak", "strong", "weak")
        );
        Ok(())
    }

    // --- tests -------------------------------------------------------------------------

    #[tokio::test]
    async fn a_note_reaches_the_model_in_the_conversation() -> Result<()> {
        // Appends a note to every outbound request, the way a router would on a turn it
        // wants to explain.
        struct Noting;

        #[async_trait]
        impl Processor for Noting {
            async fn process(&self, _state: &mut State, event: Event<'_>) -> Result<()> {
                if let Event::Decision { request, .. } = event {
                    prompts::append_note(&mut request.llm_request, NOTE);
                }
                Ok(())
            }
        }

        let captured = Arc::new(Mutex::new(None));
        let router = FallThrough::new()
            .with_processor(Arc::new(Noting))
            .with_classifier(Arc::new(DefaultCategoryClassifier(Category::Any)));

        test_drive_with_models(
            Arc::new(router),
            Request {
                llm_request: text_request(Some("auto".to_string()), "fix the build"),
                raw_request: None,
                metadata: None,
            },
            category_models(Category::Any, &["capable", "efficient"]),
            capturing(Arc::clone(&captured)),
        )
        .await?;
        let request = captured
            .lock()
            .take()
            .ok_or_else(|| LibsyError::external("test", TestError("the model was never called")))?;
        let messages: Vec<String> = request
            .llm_request
            .messages
            .iter()
            .filter_map(|message| message.text_content("|"))
            .collect();

        assert_eq!(
            messages,
            vec!["fix the build".to_string(), NOTE.to_string()]
        );
        assert!(
            request.llm_request.instructions.is_empty(),
            "a note is not an instruction"
        );
        Ok(())
    }

    #[tokio::test]
    async fn the_selected_category_leads_the_candidate_list() -> Result<()> {
        use futures::StreamExt;

        // The decision came from `capable`, so the rest of that category is tried before
        // the models it does not contain.
        let router = Arc::new(
            FallThrough::new()
                .with_classifier(Arc::new(DefaultCategoryClassifier(Category::Capable))),
        );
        let models = crate::RuntimeModels::new(
            [
                (Category::Capable, target_set(&["premium", "reasoning"])),
                (Category::Any, target_set(&["fast", "premium", "reasoning"])),
            ]
            .into(),
        );

        let stream = router.run_stream(request(), Arc::new(models), State::default());
        tokio::pin!(stream);
        while let Some(step) = stream.next().await {
            if let crate::Step::Done(outcome) = step? {
                assert_eq!(
                    outcome.selected_model_ids,
                    target_set(&["premium", "reasoning", "fast"])
                );
                return Ok(());
            }
        }
        Err(test_error("expected a Done step"))
    }

    #[tokio::test]
    async fn selected_target_leads_the_ordered_candidate_list() -> Result<()> {
        use futures::StreamExt;

        let router = Arc::new(FallThrough::new().with_classifier(fixed(vec![score("mid", 0.9)])));

        let models = category_models(Category::Any, &["weak", "mid", "strong"]);
        let stream = router.run_stream(request(), Arc::new(models.into()), State::default());
        tokio::pin!(stream);
        while let Some(step) = stream.next().await {
            if let crate::Step::Done(outcome) = step? {
                assert_eq!(
                    outcome.selected_model_ids,
                    target_set(&["mid", "weak", "strong"])
                );
                assert_eq!(outcome.request.llm_request.model.as_deref(), Some("mid"));
                assert!(outcome.response.is_none());
                return Ok(());
            }
        }
        Err(test_error("expected a Done step"))
    }

    #[tokio::test]
    async fn argmax_picks_the_highest_confidence_target() -> Result<()> {
        let router = FallThrough::new()
            .with_classifier(fixed(vec![score("weak", 0.2), score("strong", 0.9)]));
        let (model, selected_model) = run_with(router, echo()).await?;
        assert_eq!(model, "strong");
        assert_eq!(selected_model, "strong");
        Ok(())
    }

    #[tokio::test]
    async fn falls_through_the_first_abstaining_classifier() -> Result<()> {
        // First classifier abstains (empty); the second decides.
        let router = FallThrough::new()
            .with_classifier(fixed(vec![]))
            .with_classifier(fixed(vec![score("weak", 1.0)]));
        let (model, _) = run_with(router, echo()).await?;
        assert_eq!(model, "weak");
        Ok(())
    }

    #[tokio::test]
    async fn first_deciding_classifier_wins_the_cascade() -> Result<()> {
        // The first classifier decides; the second is never consulted.
        let router = FallThrough::new()
            .with_classifier(fixed(vec![score("strong", 0.6)]))
            .with_classifier(fixed(vec![score("weak", 1.0)]));
        let (model, _) = run_with(router, echo()).await?;
        assert_eq!(model, "strong");
        Ok(())
    }

    #[tokio::test]
    async fn all_abstaining_is_an_error() -> Result<()> {
        let router = FallThrough::new().with_classifier(fixed(vec![]));
        let error = run_with(router, echo())
            .await
            .err()
            .ok_or_else(|| test_error("expected classifiers to abstain"))?;
        assert!(matches!(
            error,
            LibsyError::AlgorithmError { message } if message == "every classifier abstained"
        ));
        Ok(())
    }

    #[tokio::test]
    async fn processor_observes_request_then_decision() -> Result<()> {
        use parking_lot::Mutex;

        // Records which event kinds it saw, proving the replay order: the inbound
        // request, then the routing decision (which carries the request to the model).
        struct RecordingProcessor(Arc<Mutex<Vec<&'static str>>>);

        #[async_trait]
        impl Processor for RecordingProcessor {
            async fn process(&self, _state: &mut State, event: Event<'_>) -> Result<()> {
                let kind = match event {
                    Event::Request { .. } => "request",
                    Event::Decision { .. } => "decision",
                    _ => "other",
                };
                self.0.lock().push(kind);
                Ok(())
            }
        }

        let seen = Arc::new(Mutex::new(Vec::new()));
        let router = FallThrough::new()
            .with_processor(Arc::new(RecordingProcessor(seen.clone())))
            .with_classifier(fixed(vec![score("strong", 1.0)]));
        run_with(router, echo()).await?;

        assert_eq!(*seen.lock(), vec!["request", "decision"]);
        Ok(())
    }

    #[tokio::test]
    async fn a_rewrite_propagates_down_the_chain_and_into_the_model_call() -> Result<()> {
        use parking_lot::Mutex;

        /// Appends a marker message to the request it observes.
        struct Appender(&'static str);

        #[async_trait]
        impl Processor for Appender {
            async fn process(&self, _state: &mut State, event: Event<'_>) -> Result<()> {
                if let Event::Request { request, .. } = event {
                    request
                        .llm_request
                        .messages
                        .push(Message::text(Role::User, self.0));
                }
                Ok(())
            }
        }

        /// Records the marker trail it was handed, then appends its own — proving the
        /// classifier scored the processors' rewrite rather than the original request.
        struct TrailClassifier(Arc<Mutex<Vec<String>>>);

        #[async_trait]
        impl Classifier for TrailClassifier {
            async fn score(
                &self,
                _state: &mut State,
                request: &mut Request,
                _driver: &Driver,
            ) -> Result<(Classification, Option<Response>)> {
                *self.0.lock() = request
                    .llm_request
                    .messages
                    .iter()
                    .filter_map(|message| message.text_content(""))
                    .collect();
                request
                    .llm_request
                    .messages
                    .push(Message::text(Role::User, "classifier"));
                Ok((Classification::Scores(vec![score("strong", 1.0)]), None))
            }
        }

        let seen_by_classifier = Arc::new(Mutex::new(Vec::new()));
        let seen_by_model = Arc::new(Mutex::new(None));
        let router = FallThrough::new()
            .with_processor(Arc::new(Appender("first")))
            .with_processor(Arc::new(Appender("second")))
            .with_classifier(Arc::new(TrailClassifier(seen_by_classifier.clone())));

        run_turn(&Arc::new(router), capturing(seen_by_model.clone())).await?;

        // The classifier saw both processors' edits, in chain order, on top of the original.
        assert_eq!(*seen_by_classifier.lock(), vec!["hi", "first", "second"]);

        // ...and the request that reached the model carries the classifier's edit too.
        let routed = seen_by_model
            .lock()
            .take()
            .ok_or_else(|| test_error("the model was never called"))?;
        let trail: Vec<String> = routed
            .llm_request
            .messages
            .iter()
            .filter_map(|message| message.text_content(""))
            .collect();
        assert_eq!(trail, vec!["hi", "first", "second", "classifier"]);
        Ok(())
    }

    #[tokio::test]
    async fn state_is_shared_within_a_session_and_isolated_between_sessions() -> Result<()> {
        use crate::core::state::StateValue;

        fn count(state: &State) -> u32 {
            match state.extra.get("count") {
                Some(StateValue::Count(n)) => *n,
                _ => 0,
            }
        }

        // Increments the session turn count on every request.
        struct CountingProcessor;

        #[async_trait]
        impl Processor for CountingProcessor {
            async fn process(&self, state: &mut State, event: Event<'_>) -> Result<()> {
                if let Event::Request { .. } = event {
                    let next = count(state) + 1;
                    state
                        .extra
                        .insert("count".to_string(), StateValue::Count(next));
                }
                Ok(())
            }
        }

        // Routes weak on a session's first turn and strong on later turns.
        struct ThresholdClassifier;

        #[async_trait]
        impl Classifier for ThresholdClassifier {
            async fn score(
                &self,
                state: &mut State,
                _request: &mut Request,
                _driver: &Driver,
            ) -> Result<(Classification, Option<Response>)> {
                let target = if count(state) >= 2 { "strong" } else { "weak" };
                Ok((Classification::Scores(vec![score(target, 1.0)]), None))
            }
        }

        let router = Arc::new(
            FallThrough::new_with_state()
                .with_processor(Arc::new(CountingProcessor))
                .with_classifier(Arc::new(ThresholdClassifier)),
        );

        let (turn1, _) = run_turn(&router, echo()).await?;
        let (turn2, _) = run_turn(&router, echo()).await?;
        let final_request = Request {
            metadata: Some(Metadata {
                session_id: Some("session-1".to_string()),
                session_final: Some(true),
                ..Metadata::default()
            }),
            ..request()
        };
        let (final_turn, _) = run_request(&router, final_request, echo()).await?;
        let (restarted_session, _) = run_turn(&router, echo()).await?;
        let (second_session, _) = run_request(
            &router,
            Request {
                metadata: Some(Metadata {
                    session_id: Some("session-2".to_string()),
                    ..Metadata::default()
                }),
                ..request()
            },
            echo(),
        )
        .await?;
        let anonymous = Request {
            metadata: None,
            ..request()
        };
        let (anonymous1, _) = run_request(&router, anonymous.clone(), echo()).await?;
        let (anonymous2, _) = run_request(&router, anonymous, echo()).await?;

        assert_eq!(turn1, "weak");
        assert_eq!(turn2, "strong");
        assert_eq!(final_turn, "strong");
        assert_eq!(restarted_session, "weak");
        assert_eq!(second_session, "weak");
        assert_eq!(anonymous1, "weak");
        assert_eq!(anonymous2, "weak");
        Ok(())
    }

    #[tokio::test]
    async fn final_session_is_removed_when_routing_fails() {
        let router = Arc::new(FallThrough::new_with_state());
        let final_request = Request {
            metadata: Some(Metadata {
                session_id: Some("session-1".to_string()),
                session_final: Some(true),
                ..Metadata::default()
            }),
            ..request()
        };

        let result = test_drive_with_models(
            router.clone(),
            final_request,
            category_models(Category::Any, &["strong"]),
            echo(),
        )
        .await;

        assert!(matches!(result, Err(LibsyError::AlgorithmError { .. })));
        let states = router
            .session_states
            .as_ref()
            .expect("stateful router has a session registry")
            .lock();
        assert!(!states.contains_key(&RoutingIdentity::Session("session-1".to_string())));
    }

    #[test]
    fn cleanup_removes_only_inactive_idle_sessions() {
        let router = FallThrough::new_with_state();
        let _active_state = router
            .session_state(&request()) // session-1
            .expect("session state was inserted");
        let inactive_request = Request {
            metadata: Some(Metadata {
                session_id: Some("session-2".to_string()),
                ..Metadata::default()
            }),
            ..request()
        };
        drop(router.session_state(&inactive_request));

        let states = router
            .session_states
            .as_ref()
            .expect("stateful router has a session registry");
        let now = Instant::now() + SESSION_STATE_TTL + Duration::from_secs(1);

        remove_inactive_sessions(states, now, SESSION_STATE_TTL);

        let states = states.lock();
        assert!(states.contains_key(&RoutingIdentity::Session("session-1".to_string())));
        assert!(!states.contains_key(&RoutingIdentity::Session("session-2".to_string())));
    }

    #[tokio::test]
    async fn subagent_sessions_are_isolated_from_parent_and_cleared_appropriately() -> Result<()> {
        use crate::core::state::StateValue;

        struct CountingProcessor;

        #[async_trait]
        impl Processor for CountingProcessor {
            async fn process(&self, state: &mut State, event: Event<'_>) -> Result<()> {
                if let Event::Request { .. } = event {
                    let next = match state.extra.get("count") {
                        Some(StateValue::Count(n)) => *n + 1,
                        _ => 1,
                    };
                    state
                        .extra
                        .insert("count".to_string(), StateValue::Count(next));
                }
                Ok(())
            }
        }

        struct ThresholdClassifier;

        #[async_trait]
        impl Classifier for ThresholdClassifier {
            async fn score(
                &self,
                state: &mut State,
                _request: &mut Request,
                _driver: &Driver,
            ) -> Result<(Classification, Option<Response>)> {
                let count = match state.extra.get("count") {
                    Some(StateValue::Count(n)) => *n,
                    _ => 0,
                };
                let target = if count >= 2 { "strong" } else { "weak" };
                Ok((Classification::Scores(vec![score(target, 1.0)]), None))
            }
        }

        let router = Arc::new(
            FallThrough::new_with_state()
                .with_processor(Arc::new(CountingProcessor))
                .with_classifier(Arc::new(ThresholdClassifier)),
        );

        let (root1, _) = run_turn(&router, echo()).await?;
        let (root2, _) = run_turn(&router, echo()).await?;
        assert_eq!((root1.as_str(), root2.as_str()), ("weak", "strong"));

        let child_req = |agent: Option<&str>, session_final: Option<bool>| Request {
            metadata: Some(Metadata {
                session_id: Some("session-1".to_string()),
                agent_id: agent.map(str::to_string),
                is_subagent: true,
                is_delegated_work: true,
                session_final,
                ..Metadata::default()
            }),
            ..request()
        };

        let (child_a1, _) = run_request(&router, child_req(Some("child-a"), None), echo()).await?;
        let (unidentified, _) = run_request(&router, child_req(None, None), echo()).await?;
        let (child_b1, _) = run_request(&router, child_req(Some("child-b"), None), echo()).await?;
        assert_eq!(child_a1, "weak");
        assert_eq!(unidentified, "weak");
        assert_eq!(child_b1, "weak");

        let (child_a2, _) = run_request(&router, child_req(Some("child-a"), None), echo()).await?;
        let (child_b2, _) = run_request(&router, child_req(Some("child-b"), None), echo()).await?;
        let (root3, _) = run_turn(&router, echo()).await?;
        assert_eq!(child_a2, "strong");
        assert_eq!(child_b2, "strong");
        assert_eq!(root3, "strong");

        let _ = run_request(&router, child_req(Some("child-a"), Some(true)), echo()).await?;
        assert!(
            router
                .session_states
                .as_ref()
                .expect("stateful router")
                .lock()
                .is_empty()
        );
        let (child_b_restarted, _) =
            run_request(&router, child_req(Some("child-b"), None), echo()).await?;
        let (root_restarted, _) = run_turn(&router, echo()).await?;
        assert_eq!(child_b_restarted, "weak");
        assert_eq!(root_restarted, "weak");
        Ok(())
    }

    #[test]
    fn session_cap_evicts_expired_then_oldest_idle_and_protects_active_work() {
        let router = FallThrough::new_with_state();
        let req_for = |id: &str| Request {
            metadata: Some(Metadata {
                session_id: Some(id.to_string()),
                ..Metadata::default()
            }),
            ..request()
        };
        let t0 = Instant::now();
        drop(router.session_state_at(&req_for("s1"), t0, 3));
        drop(router.session_state_at(&req_for("s2"), t0 + Duration::from_secs(10), 3));
        drop(router.session_state_at(&req_for("s3"), t0 + Duration::from_secs(20), 3));

        drop(router.session_state_at(&req_for("s2"), t0 + Duration::from_secs(25), 3));
        assert_eq!(router.session_states.as_ref().unwrap().lock().len(), 3);

        drop(router.session_state_at(&req_for("s4"), t0 + Duration::from_secs(30), 3));
        {
            let states = router.session_states.as_ref().unwrap().lock();
            assert_eq!(states.len(), 3);
            assert!(!states.contains_key(&RoutingIdentity::Session("s1".into())));
            assert!(states.contains_key(&RoutingIdentity::Session("s2".into())));
        }

        let active_2 = router
            .session_state_at(&req_for("s2"), t0 + Duration::from_secs(31), 3)
            .unwrap();
        let active_3 = router
            .session_state_at(&req_for("s3"), t0 + Duration::from_secs(32), 3)
            .unwrap();
        let active_4 = router
            .session_state_at(&req_for("s4"), t0 + Duration::from_secs(33), 3)
            .unwrap();
        assert!(
            router
                .session_state_at(&req_for("s5"), t0 + Duration::from_secs(34), 3)
                .is_none()
        );
        assert_eq!(router.session_states.as_ref().unwrap().lock().len(), 3);
        drop((active_2, active_3, active_4));
    }
}
