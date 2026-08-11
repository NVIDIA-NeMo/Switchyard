// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fall-through classifier routing: a composable [`Algorithm`] that routes each turn
//! through a processor chain and a classifier cascade.
//!
//! Each turn: request-side [`Processor`]s fold facts into the composition's state; the
//! [`Classifier`] cascade is consulted in order and the first to score decides the target
//! (its `argmax`); the [`Decision`] is published and then replayed to the processors so
//! stateful ones (latch, affinity) can bind it.
//!
//! The default `FallThrough<()>` carries no composition state. Stateful compositions share one
//! private state value across turns with the same session ID. Requests without a session ID use
//! unretained per-run state.
//!
//! Every composition retains one thing regardless: a target that overflows its context window is
//! remembered for the rest of its session and skipped on later turns.
//! An unavailable target is skipped only for the current request.

use std::collections::HashSet;
use std::{
    collections::HashMap,
    sync::{Arc, Once, Weak},
    time::{Duration, Instant},
};

use async_trait::async_trait;
use parking_lot::Mutex;
use tokio::sync::Mutex as AsyncMutex;

use crate::core::algorithm::{
    self, Algorithm, Driver, LlmTarget, LlmTargetSet, RoutingIdentity, SessionEvictions,
};
use crate::core::classifier::{Classification, Classifier, Score};
use crate::core::processor::{Event, Processor};
use crate::{LibsyError, Result};
use switchyard_protocol::{Decision, Request, Response, RoutingFallbackReason};

struct SessionState<S> {
    state: Arc<AsyncMutex<S>>,
    last_accessed: Instant,
}

type SessionStates<S> = Mutex<HashMap<String, SessionState<S>>>;

/// Delete sessions that have been inactive this long. Catches sessions that did not terminate
/// cleanly.
/// A user resuming a deleted session is not fatal. Algorithms will be missing some context
/// so may route less well for the first turn or two.
const SESSION_STATE_TTL: Duration = Duration::from_secs(60 * 60);

/// Run the expired session cleanup code this often.
const SESSION_CLEANUP_INTERVAL: Duration = Duration::from_secs(60 * 60);

/// Terminal classifier for a cascade whose classifiers may all abstain.
///
/// A classifier abstains when it cannot decide, which lets the next one try. The
/// last has no next, so a cascade that could abstain all the way through needs a
/// decider that never does. Which target that is belongs to whoever assembles the
/// cascade, not to the classifiers in it.
pub struct DefaultTarget {
    target: String,
}

impl DefaultTarget {
    /// Close a cascade with `target`.
    pub fn new(target: impl Into<String>) -> Self {
        Self {
            target: target.into(),
        }
    }
}

#[async_trait]
impl<S: Send> Classifier<S> for DefaultTarget {
    async fn score(
        &self,
        _state: &mut S,
        _request: &mut Request,
        _driver: Option<&Driver>,
    ) -> Result<(Classification, Option<Response>)> {
        // Zero confidence: this is a fallback, not a judgement.
        Ok((
            Classification::Scores(vec![Score {
                target: self.target.clone(),
                confidence: 0.0,
            }]),
            None,
        ))
    }
}

/// Processor chain → classifier cascade → routed model call. See the module docs.
///
/// The generic state type is shared by every processor and classifier in the composition.
pub struct FallThrough<S = ()> {
    name: String,
    decision_reason: fn(&str, &Score) -> String,
    processors: Vec<Arc<dyn Processor<S>>>,
    classifiers: Vec<Arc<dyn Classifier<S>>>,
    targets: LlmTargetSet,
    session_states: Option<Arc<SessionStates<S>>>,
    cleanup_started: Once,
    session_evictions: SessionEvictions,
}

impl FallThrough<()> {
    /// Creates an empty stateless router.
    pub fn new(targets: LlmTargetSet) -> Self {
        Self {
            name: "fall_through".to_string(),
            decision_reason: default_decision_reason,
            processors: Vec::new(),
            classifiers: Vec::new(),
            targets,
            session_states: None,
            cleanup_started: Once::new(),
            session_evictions: SessionEvictions::default(),
        }
    }
}

impl<S> FallThrough<S>
where
    S: Default + Send + 'static,
{
    /// Creates a router that retains one private `S` per session.
    pub fn new_with_state(targets: LlmTargetSet) -> Self {
        Self {
            name: "fall_through".to_string(),
            decision_reason: default_decision_reason,
            processors: Vec::new(),
            classifiers: Vec::new(),
            targets,
            session_states: Some(Arc::new(Mutex::new(HashMap::new()))),
            cleanup_started: Once::new(),
            session_evictions: SessionEvictions::default(),
        }
    }

    /// Sets the stable, low-cardinality telemetry name for this composition.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Sets the decision reasoning for an algorithm assembled from this cascade.
    pub(crate) fn with_decision_reason(mut self, reason: fn(&str, &Score) -> String) -> Self {
        self.decision_reason = reason;
        self
    }

    /// Appends a processor to the head-of-request chain.
    pub fn with_processor(mut self, processor: Arc<dyn Processor<S>>) -> Self {
        self.processors.push(processor);
        self
    }

    /// Appends a classifier to the cascade.
    pub fn with_classifier(mut self, classifier: Arc<dyn Classifier<S>>) -> Self {
        self.classifiers.push(classifier);
        self
    }
    /// Executes the processor/classifier/target-call sequence for wrappers and the trait entrypoint.
    pub(crate) async fn execute(&self, driver: Driver, request: Request) -> Result<Response> {
        self.start_cleanup_task();
        let session = session_id(&request);
        let session_final = request
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.session_final)
            == Some(true);
        let result = self.execute_session(driver, request).await;
        if session_final && let Some(session) = session.as_deref() {
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

    async fn execute_session(&self, driver: Driver, request: Request) -> Result<Response> {
        // The request is threaded mutably through the whole fold: any component may rewrite
        // it, later components see the rewrite, and the final value reaches the model.
        let mut request = request;
        // Targets this turn must not route to. Scratch state for one run: seeded from the
        // session's overflow history, then grown by each route-level failure below.
        let mut excluded = HashSet::new();
        // Processors may rewrite the request; overflow history stays with its inbound identity.
        let identity = RoutingIdentity::from_request(&request);
        algorithm::exclude_evicted(
            &mut excluded,
            &self.targets,
            &self.session_evictions,
            identity.as_ref(),
        );
        let session_state = self.session_state(&request);
        let (target, decision, served, deciding) = match session_state {
            Some(state) => {
                let mut state = state.lock().await;
                self.route(&mut state, &excluded, &driver, &mut request)
                    .await?
            }
            None => {
                let mut state = S::default();
                self.route(&mut state, &excluded, &driver, &mut request)
                    .await?
            }
        };

        // A classifier that already called a model — because deciding required one, and that
        // call also answers the turn — hands its response back here, so the turn is not paid
        // for twice. There is no outbound call left to overflow, so the fallback is skipped.
        // Nothing reads it on the way out: streamed or buffered, it reaches the caller
        // untouched.
        match served {
            Some(response) => Ok(response),
            None => {
                algorithm::call_model_with_fallback(
                    &mut excluded,
                    &driver,
                    &self.targets,
                    target,
                    decision,
                    request,
                    identity.as_ref(),
                    &self.session_evictions,
                    |request, target| {
                        for classifier in &self.classifiers {
                            classifier.target_unavailable(request, target);
                        }
                    },
                    |from, to, reason| self.fallback_decision(deciding.as_ref(), from, to, reason),
                )
                .await
            }
        }
    }

    /// Drops retained routing state once the host marks a session complete.
    fn remove_session(&self, session: &str) {
        if let Some(states) = &self.session_states {
            states.lock().remove(session);
        }
        self.session_evictions.remove_session(session);
    }

    /// The decision published when a route-level failure selects a different target.
    fn fallback_decision(
        &self,
        deciding: &dyn Classifier<S>,
        from: &LlmTarget,
        to: &LlmTarget,
        reason: RoutingFallbackReason,
    ) -> Decision {
        let failure = match reason {
            RoutingFallbackReason::ContextWindow => "exceeded its context window",
            RoutingFallbackReason::Unavailable => "was unavailable",
        };
        Decision::new(
            to.semantic_name.clone(),
            Some(with_routing_tier(
                format!(
                    "{} {failure}; fell back to {} (fallback reason: {})",
                    from.semantic_name,
                    to.semantic_name,
                    reason.as_str(),
                ),
                deciding.routing_tier(&to.semantic_name),
            )),
            true,
        )
    }

    /// Returns this request's retained state without holding the registry lock.
    fn session_state(&self, request: &Request) -> Option<Arc<AsyncMutex<S>>> {
        let states = self.session_states.as_ref()?;
        let session_id = session_id(request)?;
        let mut states = states.lock();
        let now = Instant::now();
        let session = states.entry(session_id).or_insert_with(|| SessionState {
            state: Arc::new(AsyncMutex::new(S::default())),
            last_accessed: now,
        });
        session.last_accessed = now;
        Some(Arc::clone(&session.state))
    }

    async fn route(
        &self,
        state: &mut S,
        excluded: &HashSet<String>,
        driver: &Driver,
        request: &mut Request,
    ) -> Result<(
        LlmTarget,
        Decision,
        Option<Response>,
        Arc<dyn Classifier<S>>,
    )> {
        // 1. Processor chain accumulates request-side facts into the composition's state.
        for processor in &self.processors {
            processor.process(state, Event::Request(request)).await?;
        }

        // 2. Fall through the cascade: the first classifier to score decides (argmax). The
        //    per-request driver is offered to each — driver-backed classifiers use it.
        let mut routed = None;
        for classifier in &self.classifiers {
            let (scores, response) = classifier.score(state, request, Some(driver)).await?;
            if let Some(score) = scores.argmax(false)? {
                // Only the deciding classifier's response answers the turn; an abstaining
                // classifier selected nothing for it to be the answer to.
                routed = Some((score, Arc::clone(classifier), response));
                break;
            }
        }
        let Some((score, deciding, served)) = routed else {
            return Err(LibsyError::AlgorithmError {
                message: "every classifier abstained".to_string(),
            });
        };

        // 3. Resolve the target and publish the decision. When an excluded target sends
        //    the request elsewhere, the reasoning describes where it actually went.
        let target = self.targets.resolve_target(&score.target, excluded)?;
        let reasoning = if target.semantic_name == score.target {
            (self.decision_reason)(&self.name, &score)
        } else {
            format!(
                "{} exceeded its context window; fell back to {}",
                score.target, target.semantic_name
            )
        };
        let decision: Decision = Decision::new(
            target.semantic_name.clone(),
            Some(with_routing_tier(
                reasoning,
                deciding.routing_tier(&target.semantic_name),
            )),
            true,
        );
        driver.info(decision.clone()).await?;

        // 4. Post-decision replay: every processor sees the decision so stateful ones
        //    can bind it, and may rewrite the outbound request (e.g. add a target prompt).
        for processor in &self.processors {
            let event = Event::Decision {
                request,
                decision: &decision,
            };
            processor.process(state, event).await?;
        }

        Ok((target, decision, served, deciding))
    }
}

async fn cleanup_inactive_sessions<S>(states: Weak<SessionStates<S>>)
where
    S: Send + 'static,
{
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

fn remove_inactive_sessions<S>(states: &SessionStates<S>, now: Instant, ttl: Duration) {
    states.lock().retain(|_, session| {
        // The registry owns one strong reference. Any others belong to active requests,
        // which must keep sharing this state until their routing work finishes.
        Arc::strong_count(&session.state) > 1
            || now.saturating_duration_since(session.last_accessed) < ttl
    });
}

fn session_id(request: &Request) -> Option<String> {
    request
        .metadata
        .as_ref()?
        .session_id
        .as_deref()
        .filter(|id| !id.is_empty())
        .map(str::to_string)
}

fn default_decision_reason(_name: &str, winner: &Score) -> String {
    format!(
        "fall-through selected {} (confidence {:.3})",
        winner.target, winner.confidence
    )
}

// Routing tier is not part of Decision struct, so keeping it in reasoning for now. Remove it later if needed from reasoning.
fn with_routing_tier(reasoning: String, tier: Option<&str>) -> String {
    match tier {
        Some(tier) => format!("{reasoning}; routing tier: {tier}"),
        None => reasoning,
    }
}

#[async_trait]
impl<S> Algorithm for FallThrough<S>
where
    S: Default + Send + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }

    async fn route(self: Arc<Self>, driver: Driver, request: Request) -> Result<Response> {
        self.execute(driver, request).await
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::util::prompts;
    use crate::core::classifier::Classification;
    use crate::{AffinityRouter, SystemPromptProcessor, TargetPrompts};

    use crate::core::testing::{Serve, echo, reply, test_drive};
    use switchyard_protocol::{
        LlmClientError, LlmRequest, Message, Metadata, Role, completion_text, text_request,
    };

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
        move |decision: Decision, request: Request| {
            let into = Arc::clone(&into);
            async move {
                *into.lock() = Some(request);
                Ok(reply(decision.selected_model_id()))
            }
        }
    }

    const CAPABLE_PROMPT: &str = "diagnose before you edit";
    const EFFICIENT_PROMPT: &str = "follow the settled plan";
    const NOTE: &str = "the previous model was stalling";

    /// One model call as the prompt and note tests observe it.
    #[derive(Clone, Debug, Default)]
    struct RecordedCall {
        target: String,
        messages: Vec<String>,
        instructions: Vec<String>,
    }

    /// Captures the prompt-bearing request that reached the selected target.
    #[derive(Default)]
    struct PromptRecorder(Mutex<Option<RecordedCall>>);

    impl PromptRecorder {
        fn serve(self: &Arc<Self>) -> impl Serve {
            let recorder = Arc::clone(self);
            move |decision: Decision, request: Request| {
                let recorder = Arc::clone(&recorder);
                async move {
                    *recorder.0.lock() = Some(RecordedCall {
                        target: decision.selected_model_id().to_string(),
                        messages: request
                            .llm_request
                            .messages
                            .iter()
                            .filter_map(|message| message.text_content("|"))
                            .collect(),
                        instructions: request
                            .llm_request
                            .instructions
                            .iter()
                            .filter_map(|block| block.content.iter().find_map(text_of))
                            .collect(),
                    });
                    Ok(reply(decision.selected_model_id()))
                }
            }
        }
    }

    fn text_of(block: &switchyard_protocol::ContentBlock) -> Option<String> {
        match block {
            switchyard_protocol::ContentBlock::Text { text } => Some(text.clone()),
            _ => None,
        }
    }

    fn target_set(names: &[&str]) -> LlmTargetSet {
        LlmTargetSet::new(
            names
                .iter()
                .map(|name| LlmTarget {
                    semantic_name: name.to_string(),
                })
                .collect(),
        )
    }

    fn target_prompts() -> TargetPrompts {
        TargetPrompts::default()
            .with("capable", CAPABLE_PROMPT)
            .with("efficient", EFFICIENT_PROMPT)
    }

    /// Routes one turn on a prompt test cascade and returns the recorded model call.
    async fn routed_prompt_call(
        recorder: &Arc<PromptRecorder>,
        router: FallThrough,
    ) -> Result<RecordedCall> {
        test_drive(
            Arc::new(router),
            Request {
                llm_request: text_request(Some("auto".to_string()), "fix the build"),
                raw_request: None,
                metadata: None,
            },
            recorder.serve(),
        )
        .await?;
        let call = recorder.0.lock().take();
        match call {
            Some(call) => Ok(call),
            None => panic!("the model was never called"),
        }
    }

    /// A prompt cascade that always routes to `target`.
    fn prompt_router(target: &str, prompts: TargetPrompts) -> FallThrough {
        FallThrough::new(target_set(&["capable", "efficient"]))
            .with_processor(Arc::new(SystemPromptProcessor::new(prompts)))
            .with_classifier(Arc::new(DefaultTarget::new(target)))
    }

    /// A classifier that emits fixed scores (empty = abstain).
    struct FixedClassifier(Vec<Score>);

    #[async_trait]
    impl Classifier for FixedClassifier {
        async fn score(
            &self,
            _state: &mut (),
            _request: &mut Request,
            _driver: Option<&Driver>,
        ) -> Result<(Classification, Option<Response>)> {
            Ok((
                Classification::Scores(
                    self.0
                        .iter()
                        .map(|s| Score {
                            confidence: s.confidence,
                            target: s.target.clone(),
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
            target: target.to_string(),
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

    /// Drives a shared router with one request, returning the completion text + trace.
    async fn run_request<S>(
        router: &Arc<FallThrough<S>>,
        request: Request,
        serve: impl Serve,
    ) -> Result<(String, Vec<Decision>)>
    where
        S: Default + Send + 'static,
    {
        let (trace, response) = test_drive(router.clone(), request, serve).await?;
        let text = response
            .llm_response
            .into_agg()
            .await
            .map(|agg| completion_text(&agg))
            .map_err(|error| LibsyError::external("aggregating fall-through response", error))?;
        Ok((text, trace))
    }

    /// Drives a shared router through one turn in the default test session.
    async fn run_turn<S>(
        router: &Arc<FallThrough<S>>,
        serve: impl Serve,
    ) -> Result<(String, Vec<Decision>)>
    where
        S: Default + Send + 'static,
    {
        run_request(router, request(), serve).await
    }

    /// Drives a fresh router through one turn with a specific `serve`.
    async fn run_with(router: FallThrough, serve: impl Serve) -> Result<(String, Vec<Decision>)> {
        run_turn(&Arc::new(router), serve).await
    }

    /// Discards the recorded calls; the overflow tests that don't assert on them.
    fn overflows(targets: &'static [&'static str]) -> impl Serve {
        overflowing(targets, Arc::new(Mutex::new(Vec::new())))
    }

    /// Rejects the named `overflowing` targets with a context-window error and echoes for
    /// the rest, so the retry path can be driven. Every call is recorded in `calls`.
    fn overflowing(
        overflowing: &'static [&'static str],
        calls: Arc<Mutex<Vec<String>>>,
    ) -> impl Serve {
        move |decision: Decision, _request: Request| {
            let calls = Arc::clone(&calls);
            async move {
                let model = decision.selected_model_id().to_string();
                calls.lock().push(model.clone());
                if overflowing.contains(&model.as_str()) {
                    return Err(LlmClientError::ContextWindowExceeded {
                        model,
                        message: "prompt is too long".to_string(),
                    });
                }
                Ok(reply(model))
            }
        }
    }

    /// Rejects the named `unavailable` targets with a 503 and echoes for the rest.
    /// Every call is recorded in `calls`.
    fn unavailable(
        unavailable: &'static [&'static str],
        calls: Arc<Mutex<Vec<String>>>,
    ) -> impl Serve {
        move |decision: Decision, _request: Request| {
            let calls = Arc::clone(&calls);
            async move {
                let model = decision.selected_model_id().to_string();
                calls.lock().push(model.clone());
                if unavailable.contains(&model.as_str()) {
                    return Err(LlmClientError::UpstreamHttp {
                        status: 503,
                        body: "unavailable".to_string(),
                    });
                }
                Ok(reply(model))
            }
        }
    }

    // --- tests -------------------------------------------------------------------------

    #[tokio::test]
    async fn each_target_gets_its_own_prompt() -> Result<()> {
        for (target, expected) in [("capable", CAPABLE_PROMPT), ("efficient", EFFICIENT_PROMPT)] {
            let recorder = Arc::new(PromptRecorder::default());
            let call =
                routed_prompt_call(&recorder, prompt_router(target, target_prompts())).await?;
            assert_eq!(call.target, target);
            assert_eq!(call.instructions, vec![expected.to_string()]);
        }
        Ok(())
    }

    #[tokio::test]
    async fn a_target_with_no_prompt_is_left_untouched() -> Result<()> {
        let recorder = Arc::new(PromptRecorder::default());
        let only_capable = TargetPrompts::default().with("capable", CAPABLE_PROMPT);

        let call = routed_prompt_call(&recorder, prompt_router("efficient", only_capable)).await?;

        assert!(
            call.instructions.is_empty(),
            "one target's prompt must not leak onto another: {:?}",
            call.instructions
        );
        Ok(())
    }

    #[tokio::test]
    async fn the_prompt_follows_the_target_whichever_classifier_picked_it() -> Result<()> {
        // The first classifier abstains, so the second decides; the prompt follows the
        // target the cascade settled on rather than the classifier that named it.
        struct Abstains;

        #[async_trait]
        impl Classifier for Abstains {
            async fn score(
                &self,
                _state: &mut (),
                _request: &mut Request,
                _driver: Option<&Driver>,
            ) -> Result<(Classification, Option<Response>)> {
                Ok((Classification::Ambiguous(Vec::new()), None))
            }
        }

        let recorder = Arc::new(PromptRecorder::default());
        let router = FallThrough::new(target_set(&["capable", "efficient"]))
            .with_processor(Arc::new(SystemPromptProcessor::new(target_prompts())))
            .with_classifier(Arc::new(Abstains))
            .with_classifier(Arc::new(DefaultTarget::new("capable")));

        let call = routed_prompt_call(&recorder, router).await?;

        assert_eq!(call.target, "capable");
        assert_eq!(call.instructions, vec![CAPABLE_PROMPT.to_string()]);
        Ok(())
    }

    #[tokio::test]
    async fn a_note_reaches_the_model_in_the_conversation() -> Result<()> {
        // Appends a note to every outbound request, the way a router would on a turn it
        // wants to explain.
        struct Noting;

        #[async_trait]
        impl Processor for Noting {
            async fn process(&self, _state: &mut (), event: Event<'_>) -> Result<()> {
                if let Event::Decision { request, .. } = event {
                    prompts::append_note(request, NOTE);
                }
                Ok(())
            }
        }

        let recorder = Arc::new(PromptRecorder::default());
        let router = FallThrough::new(target_set(&["capable", "efficient"]))
            .with_processor(Arc::new(Noting))
            .with_classifier(Arc::new(DefaultTarget::new("capable")));

        let call = routed_prompt_call(&recorder, router).await?;

        assert_eq!(call.messages, vec![format!("fix the build|{NOTE}")]);
        assert!(call.instructions.is_empty(), "a note is not an instruction");
        Ok(())
    }

    #[tokio::test]
    async fn a_target_that_overflowed_is_skipped_for_the_rest_of_the_session() -> Result<()> {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let router = Arc::new(
            FallThrough::<()>::new(target_set(&["weak", "strong"]))
                .with_classifier(fixed(vec![score("weak", 0.9)])),
        );
        for _ in 0..3 {
            let serve = overflowing(&["weak"], calls.clone());
            assert_eq!(run_turn(&router, serve).await?.0, "strong");
        }
        assert_eq!(calls.lock().iter().filter(|m| *m == "weak").count(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn a_different_session_starts_with_an_empty_eviction_set() -> Result<()> {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let router = Arc::new(
            FallThrough::<()>::new(target_set(&["weak", "strong"]))
                .with_classifier(fixed(vec![score("weak", 0.9)])),
        );
        run_turn(&router, overflowing(&["weak"], calls.clone())).await?;
        let mut other = request();
        other.metadata = Some(Metadata {
            session_id: Some("session-2".to_string()),
            ..Metadata::default()
        });
        run_request(&router, other, overflowing(&["weak"], calls.clone())).await?;
        assert_eq!(calls.lock().iter().filter(|m| *m == "weak").count(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn second_turn_after_full_exhaustion_still_reaches_upstream() -> Result<()> {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let router = Arc::new(
            FallThrough::<()>::new(target_set(&["weak", "strong"]))
                .with_classifier(fixed(vec![score("weak", 0.9)])),
        );
        let first = run_turn(&router, overflowing(&["weak", "strong"], calls.clone())).await;
        assert!(first.is_err());
        calls.lock().clear();
        match run_turn(&router, overflowing(&["weak", "strong"], calls.clone())).await {
            Err(LibsyError::ClientCall { .. }) => {}
            Err(other) => panic!("turn 2 gave {other:?}, calls={:?}", calls.lock()),
            Ok(_) => panic!("expected an error"),
        }
        Ok(())
    }

    #[tokio::test]
    async fn an_overflowing_target_is_retried_on_one_that_fits() -> Result<()> {
        let router = FallThrough::<()>::new(target_set(&["weak", "strong"]))
            .with_classifier(fixed(vec![score("weak", 0.9)]));
        let (model, _) = run_with(router, overflows(&["weak"])).await?;
        assert_eq!(model, "strong");
        Ok(())
    }

    #[tokio::test]
    async fn unavailable_target_clears_matching_affinity_before_the_next_turn() -> Result<()> {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let affinity = Arc::new(AffinityRouter::new());
        let router = Arc::new(
            FallThrough::<()>::new(target_set(&["weak", "strong"]))
                .with_processor(affinity.clone())
                .with_classifier(affinity)
                .with_classifier(Arc::new(DefaultTarget::new("weak"))),
        );
        for _ in 0..2 {
            let (model, trace) = run_turn(&router, unavailable(&["weak"], calls.clone())).await?;
            assert_eq!(model, "strong");
            assert!(
                trace
                    .last()
                    .and_then(|decision| decision.reasoning())
                    .is_some_and(|reasoning| reasoning.contains("fallback reason: unavailable"))
            );
        }
        assert_eq!(&*calls.lock(), &["weak", "strong", "weak", "strong"]);
        Ok(())
    }

    #[tokio::test]
    async fn fallback_decision_preserves_tier_and_cause_in_reasoning() -> Result<()> {
        struct TieredClassifier;

        #[async_trait]
        impl Classifier for TieredClassifier {
            fn routing_tier(&self, selected_model_id: &str) -> Option<&'static str> {
                match selected_model_id {
                    "weak" => Some("weak"),
                    "strong" => Some("strong"),
                    _ => None,
                }
            }

            async fn score(
                &self,
                _state: &mut (),
                _request: &mut Request,
                _driver: Option<&Driver>,
            ) -> Result<(Classification, Option<Response>)> {
                Ok((Classification::Scores(vec![score("weak", 1.0)]), None))
            }
        }

        let router = FallThrough::<()>::new(target_set(&["weak", "strong"]))
            .with_classifier(Arc::new(TieredClassifier));
        let (_, trace) = run_with(router, unavailable(&["weak"], Arc::default())).await?;

        assert_eq!(trace.len(), 2);
        assert_eq!(trace[0].selected_model_id(), "weak");
        assert!(trace[0].is_answer_call());
        assert!(
            trace[0]
                .reasoning()
                .is_some_and(|reasoning| reasoning.contains("routing tier: weak"))
        );

        let fallback = &trace[1];
        assert_eq!(fallback.selected_model_id(), "strong");
        assert!(fallback.is_answer_call());
        assert!(fallback.reasoning().is_some_and(|reasoning| {
            reasoning.contains("fallback reason: unavailable")
                && reasoning.contains("routing tier: strong")
        }));
        Ok(())
    }

    #[tokio::test]
    async fn overflowing_targets_are_retried_until_one_fits() -> Result<()> {
        let router = FallThrough::<()>::new(target_set(&["weak", "mid", "strong"]))
            .with_classifier(fixed(vec![score("weak", 0.9)]));
        let (model, _) = run_with(router, overflows(&["weak", "mid"])).await?;
        assert_eq!(model, "strong");
        Ok(())
    }

    #[tokio::test]
    async fn exhausting_every_target_surfaces_the_client_overflow() -> Result<()> {
        // Only the client error maps to a 400 upstream, so it must survive exhaustion.
        let router = FallThrough::<()>::new(target_set(&["weak", "strong"]))
            .with_classifier(fixed(vec![score("weak", 0.9)]));
        match run_with(router, overflows(&["weak", "strong"])).await {
            Ok(_) => panic!("expected an overflow error, got a response"),
            Err(LibsyError::ClientCall {
                source: LlmClientError::ContextWindowExceeded { .. },
                ..
            }) => Ok(()),
            Err(other) => panic!("expected ContextWindowExceeded, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_retried_request_runs_the_processors_once() -> Result<()> {
        // Routing runs before the call loop, so an overflow must not replay processors.
        struct CountingProcessor(Arc<Mutex<Vec<&'static str>>>);

        #[async_trait]
        impl Processor for CountingProcessor {
            async fn process(&self, _state: &mut (), event: Event<'_>) -> Result<()> {
                let kind = match event {
                    Event::Request(_) => "request",
                    Event::Decision { .. } => "decision",
                    _ => "other",
                };
                self.0.lock().push(kind);
                Ok(())
            }
        }

        let seen = Arc::new(Mutex::new(Vec::new()));
        let router = FallThrough::<()>::new(target_set(&["weak", "strong"]))
            .with_classifier(fixed(vec![score("weak", 0.9)]))
            .with_processor(Arc::new(CountingProcessor(seen.clone())));
        let (model, _) = run_with(router, overflows(&["weak"])).await?;
        assert_eq!(model, "strong");
        assert_eq!(seen.lock().iter().filter(|e| **e == "request").count(), 1);
        assert_eq!(seen.lock().iter().filter(|e| **e == "decision").count(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn a_target_barred_by_overflow_history_reports_the_fallback() -> Result<()> {
        // Headers and usage metrics read the decision, so it must describe the real call.
        let router = Arc::new(
            FallThrough::<()>::new(target_set(&["weak", "strong"]))
                .with_classifier(fixed(vec![score("weak", 0.9)])),
        );
        // The first turn teaches the session that "weak" overflows; the second is barred
        // from it before calling, so routing redirects and must say where it went.
        run_turn(&router, overflows(&["weak"])).await?;
        let (text, trace) = run_turn(&router, overflows(&["weak"])).await?;

        assert_eq!(text, "strong");
        assert_eq!(trace[0].selected_model_id(), "strong");
        assert!(
            trace[0]
                .reasoning()
                .is_some_and(|r| r.contains("fell back to strong"))
        );
        Ok(())
    }

    #[tokio::test]
    async fn argmax_picks_the_highest_confidence_target() -> Result<()> {
        let router = FallThrough::<()>::new(target_set(&["strong", "weak"]))
            .with_classifier(fixed(vec![score("weak", 0.2), score("strong", 0.9)]));
        let (model, trace) = run_with(router, echo()).await?;
        assert_eq!(model, "strong");
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].selected_model_id(), "strong");
        Ok(())
    }

    #[tokio::test]
    async fn falls_through_the_first_abstaining_classifier() -> Result<()> {
        // First classifier abstains (empty); the second decides.
        let router = FallThrough::<()>::new(target_set(&["strong", "weak"]))
            .with_classifier(fixed(vec![]))
            .with_classifier(fixed(vec![score("weak", 1.0)]));
        let (model, _) = run_with(router, echo()).await?;
        assert_eq!(model, "weak");
        Ok(())
    }

    #[tokio::test]
    async fn first_deciding_classifier_wins_the_cascade() -> Result<()> {
        // The first classifier decides; the second is never consulted.
        let router = FallThrough::<()>::new(target_set(&["strong", "weak"]))
            .with_classifier(fixed(vec![score("strong", 0.6)]))
            .with_classifier(fixed(vec![score("weak", 1.0)]));
        let (model, _) = run_with(router, echo()).await?;
        assert_eq!(model, "strong");
        Ok(())
    }

    #[tokio::test]
    async fn all_abstaining_is_an_error() -> Result<()> {
        let router =
            FallThrough::<()>::new(target_set(&["strong", "weak"])).with_classifier(fixed(vec![]));
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
    async fn classifiers_receive_the_per_request_driver() -> Result<()> {
        // A classifier that only decides when handed a driver — proving the cascade offers
        // the per-request driver to every classifier (driver-backed ones need it).
        struct NeedsDriver;

        #[async_trait]
        impl Classifier for NeedsDriver {
            async fn score(
                &self,
                _state: &mut (),
                _request: &mut Request,
                driver: Option<&Driver>,
            ) -> Result<(Classification, Option<Response>)> {
                match driver {
                    Some(_) => Ok((Classification::Scores(vec![score("strong", 1.0)]), None)),
                    None => Err(test_error("expected a driver")),
                }
            }
        }

        let router = FallThrough::<()>::new(target_set(&["strong", "weak"]))
            .with_classifier(Arc::new(NeedsDriver));
        let (model, _) = run_with(router, echo()).await?;
        assert_eq!(model, "strong");
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
            async fn process(&self, _state: &mut (), event: Event<'_>) -> Result<()> {
                let kind = match event {
                    Event::Request(_) => "request",
                    Event::Decision { .. } => "decision",
                    _ => "other",
                };
                self.0.lock().push(kind);
                Ok(())
            }
        }

        let seen = Arc::new(Mutex::new(Vec::new()));
        let router = FallThrough::<()>::new(target_set(&["strong", "weak"]))
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
            async fn process(&self, _state: &mut (), event: Event<'_>) -> Result<()> {
                if let Event::Request(request) = event {
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
                _state: &mut (),
                request: &mut Request,
                _driver: Option<&Driver>,
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
        let targets = target_set(&["strong"]);
        let router = FallThrough::new(targets)
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
        #[derive(Default)]
        struct TurnState {
            count: u32,
        }

        // Increments the session turn count on every request.
        struct CountingProcessor;

        #[async_trait]
        impl Processor<TurnState> for CountingProcessor {
            async fn process(&self, state: &mut TurnState, event: Event<'_>) -> Result<()> {
                if let Event::Request(_) = event {
                    state.count += 1;
                }
                Ok(())
            }
        }

        // Routes weak on a session's first turn and strong on later turns.
        struct ThresholdClassifier;

        #[async_trait]
        impl Classifier<TurnState> for ThresholdClassifier {
            async fn score(
                &self,
                state: &mut TurnState,
                _request: &mut Request,
                _driver: Option<&Driver>,
            ) -> Result<(Classification, Option<Response>)> {
                let target = if state.count >= 2 { "strong" } else { "weak" };
                Ok((Classification::Scores(vec![score(target, 1.0)]), None))
            }
        }

        let router = Arc::new(
            FallThrough::<TurnState>::new_with_state(target_set(&["strong", "weak"]))
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
        let router = Arc::new(FallThrough::<u32>::new_with_state(target_set(&["strong"])));
        let final_request = Request {
            metadata: Some(Metadata {
                session_id: Some("session-1".to_string()),
                session_final: Some(true),
                ..Metadata::default()
            }),
            ..request()
        };

        let result = test_drive(router.clone(), final_request, echo()).await;

        assert!(matches!(result, Err(LibsyError::AlgorithmError { .. })));
        let states = router
            .session_states
            .as_ref()
            .expect("stateful router has a session registry")
            .lock();
        assert!(!states.contains_key("session-1"));
    }

    #[test]
    fn cleanup_removes_only_inactive_idle_sessions() {
        let router = FallThrough::<u32>::new_with_state(target_set(&["strong"]));
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
        assert!(states.contains_key("session-1"));
        assert!(!states.contains_key("session-2"));
    }
}
