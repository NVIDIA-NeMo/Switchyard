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

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use async_trait::async_trait;
use parking_lot::Mutex;
use tokio::sync::Mutex as AsyncMutex;

use crate::core::{Classification, Classifier, Event, Processor, Score};
use crate::{
    Algorithm, Context, Decision, Driver, LibsyError, LlmClientError, LlmTarget, LlmTargetSet,
    Request, Response, Result, RoutedLlmClient,
};

type SessionStates<S> = Mutex<HashMap<String, Arc<AsyncMutex<S>>>>;

/// Matches the cap the Python stack uses. Dropping a live session's entry costs one
/// rediscovered overflow, so the victim choice does not need to be exact.
const MAX_EVICTION_SESSIONS: usize = 1_024;

/// The decision a fall-through run produces: the selected model plus a human-readable reason.
pub struct FallThroughDecision {
    /// Target selected by the classifier cascade.
    pub selected_model: String,
    /// Human-readable explanation of the selection.
    pub reasoning: String,
    tier: Option<&'static str>,
}

impl Decision for FallThroughDecision {
    fn selected_model(&self) -> &str {
        &self.selected_model
    }

    fn routing_tier(&self) -> Option<&str> {
        self.tier
    }

    fn reasoning(&self) -> Option<&str> {
        Some(&self.reasoning)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

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
    session_states: Option<SessionStates<S>>,
    session_evictions: Mutex<HashMap<String, HashSet<String>>>,
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
            session_evictions: Mutex::new(HashMap::new()),
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
            session_states: Some(Mutex::new(HashMap::new())),
            session_evictions: Mutex::new(HashMap::new()),
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
    /// Registers one dual-role component in *both* the processor chain and the classifier
    /// cascade.
    ///
    /// A component that writes state as a [`Processor`] and reads it back as a
    /// [`Classifier`] — such as [`AffinityRouter`](crate::algorithms::AffinityRouter) —
    /// shares that state through the instance, so both roles must be the same `Arc`.
    /// Registering the two separately is easy to half-wire: omit the processor and the
    /// classifier silently never sees an assignment. This registers both at once.
    pub fn with_component<T>(self, component: Arc<T>) -> Self
    where
        T: Processor<S> + Classifier<S> + 'static,
    {
        self.with_processor(component.clone())
            .with_classifier(component)
    }

    /// Executes the processor/classifier/target-call sequence for wrappers and the trait entrypoint.
    pub(crate) async fn execute(
        &self,
        ctx: Context,
        driver: Driver,
        request: Request,
    ) -> Result<Response> {
        // The request is threaded mutably through the whole fold: any component may rewrite
        // it, later components see the rewrite, and the final value reaches the model.
        let mut request = request;
        let session = session_id(&request);
        let mut ctx = ctx;
        for target in self.evicted_in_session(session.as_deref()) {
            // Never seed the pool empty: a later turn may be small enough to serve, and the
            // caller should get the upstream's answer rather than a routing error.
            if self.eligible_targets(&ctx) <= 1 {
                break;
            }
            ctx.exclude_target(target);
        }
        let session_state = self.session_state(&request);
        let (target, decision, served) = match session_state {
            Some(state) => {
                let mut state = state.lock().await;
                self.route(&mut state, &ctx, &driver, &mut request).await?
            }
            None => {
                let mut state = S::default();
                self.route(&mut state, &ctx, &driver, &mut request).await?
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
                self.call_with_overflow_fallback(
                    ctx,
                    &driver,
                    target,
                    decision,
                    request,
                    session.as_deref(),
                )
                .await
            }
        }
    }

    fn eligible_targets(&self, ctx: &Context) -> usize {
        self.targets
            .targets()
            .iter()
            .filter(|t| !ctx.is_excluded(&t.semantic_name))
            .count()
    }

    fn evicted_in_session(&self, session: Option<&str>) -> Vec<String> {
        let Some(session) = session else {
            return Vec::new();
        };
        self.session_evictions
            .lock()
            .get(session)
            .map(|targets| targets.iter().cloned().collect())
            .unwrap_or_default()
    }

    fn record_eviction(&self, session: Option<&str>, target: &str) {
        let Some(session) = session else { return };
        let mut sessions = self.session_evictions.lock();
        if sessions.len() >= MAX_EVICTION_SESSIONS && !sessions.contains_key(session) {
            if let Some(oldest) = sessions.keys().next().cloned() {
                sessions.remove(&oldest);
            }
        }
        sessions
            .entry(session.to_string())
            .or_default()
            .insert(target.to_string());
    }

    /// Calls `target`, falling back to the next eligible target whenever one overflows its
    /// context window, until a call succeeds or every target has been tried.
    ///
    /// Routing is deliberately not re-run: the fallback replaces the target in place so the
    /// processor chain and the retained session state still see exactly one turn.
    async fn call_with_overflow_fallback(
        &self,
        mut ctx: Context,
        driver: &Driver,
        mut target: LlmTarget,
        mut decision: Arc<dyn Decision>,
        request: Request,
        session: Option<&str>,
    ) -> Result<Response> {
        loop {
            let result = driver
                .call_llm_target(ctx.clone(), &target, request.clone(), decision.clone())
                .await;
            let Err(error) = result else { return result };
            let LibsyError::ClientCall {
                target: failed,
                source: LlmClientError::ContextWindowExceeded { .. },
            } = &error
            else {
                return Err(error);
            };
            // A target already excluded means the pool is spent; surface the client error
            // so the caller still sees a context overflow rather than an internal failure.
            if !ctx.exclude_target(failed) {
                return Err(error);
            }
            self.record_eviction(session, failed);
            let Ok(next) = self.targets.resolve_target(&target.semantic_name, &ctx) else {
                return Err(error);
            };
            decision = self.fallback_decision(&target, &next);
            target = next;
            driver.info(ctx.clone(), decision.clone()).await?;
        }
    }

    /// The decision published when an overflow sends the request to a different target.
    fn fallback_decision(&self, from: &LlmTarget, to: &LlmTarget) -> Arc<dyn Decision> {
        Arc::new(FallThroughDecision {
            selected_model: to.semantic_name.clone(),
            reasoning: format!(
                "{} exceeded its context window; fell back to {}",
                from.semantic_name, to.semantic_name
            ),
            tier: self
                .classifiers
                .iter()
                .find_map(|c| c.routing_tier(&to.semantic_name)),
        })
    }

    /// Returns this request's retained state without holding the registry lock.
    fn session_state(&self, request: &Request) -> Option<Arc<AsyncMutex<S>>> {
        let states = self.session_states.as_ref()?;
        let session_id = session_id(request)?;
        let mut states = states.lock();
        let state = states
            .entry(session_id)
            .or_insert_with(|| Arc::new(AsyncMutex::new(S::default())));
        Some(Arc::clone(state))
    }

    async fn route(
        &self,
        state: &mut S,
        ctx: &Context,
        driver: &Driver,
        request: &mut Request,
    ) -> Result<(crate::LlmTarget, Arc<dyn Decision>, Option<Response>)> {
        // 1. Processor chain accumulates request-side facts into the composition's state.
        for processor in &self.processors {
            processor.process(state, Event::Request(request)).await?;
        }

        // 2. Fall through the cascade: the first classifier to score decides (argmax). The
        //    per-request driver is offered to each — driver-backed classifiers use it.
        let mut maybe_score: Option<Score> = None;
        let mut deciding: Option<&Arc<dyn Classifier<S>>> = None;
        let mut served: Option<Response> = None;
        for classifier in &self.classifiers {
            let (scores, response) = classifier.score(state, request, Some(driver)).await?;
            maybe_score = scores.argmax(false)?;
            if maybe_score.is_some() {
                deciding = Some(classifier);
                // Only the deciding classifier's response answers the turn; an abstaining
                // classifier selected nothing for it to be the answer to.
                served = response;
                break;
            }
        }
        let Some(score) = maybe_score else {
            return Err(LibsyError::AlgorithmError {
                message: "every classifier abstained".to_string(),
            });
        };

        // 3. Resolve the target and publish the decision. When an excluded target sends
        //    the request elsewhere, the tier and reasoning describe where it actually went.
        let target = self.targets.resolve_target(&score.target, ctx)?;
        let reasoning = if target.semantic_name == score.target {
            (self.decision_reason)(&self.name, &score)
        } else {
            format!(
                "{} exceeded its context window; fell back to {}",
                score.target, target.semantic_name
            )
        };
        let decision: Arc<dyn Decision> = Arc::new(FallThroughDecision {
            selected_model: target.semantic_name.clone(),
            reasoning,
            tier: deciding.and_then(|c| c.routing_tier(&target.semantic_name)),
        });
        driver.info(ctx.clone(), decision.clone()).await?;

        // 4. Post-decision replay: every processor sees the decision so stateful ones
        //    can bind it, and may rewrite the outbound request (e.g. add a tier prompt).
        for processor in &self.processors {
            let event = Event::Decision {
                request,
                decision: decision.as_ref(),
            };
            processor.process(state, event).await?;
        }

        Ok((target, decision, served))
    }
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

#[async_trait]
impl<S> Algorithm for FallThrough<S>
where
    S: Default + Send + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn count_tokens_client(&self) -> Option<Arc<dyn RoutedLlmClient>> {
        self.targets.count_tokens_client()
    }

    async fn create_run_task(
        self: Arc<Self>,
        ctx: Context,
        driver: Driver,
        request: Request,
    ) -> Result<Response> {
        self.execute(ctx, driver, request).await
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Classification;

    use switchyard_protocol::{
        completion_text, text_response, LlmRequest, Message, Metadata, Role,
    };

    use crate::{LlmResponse, LlmTarget, RoutedLlmClient};

    #[derive(Debug, thiserror::Error)]
    #[error("{0}")]
    struct TestError(&'static str);

    fn test_error(message: &'static str) -> LibsyError {
        LibsyError::external("test", TestError(message))
    }

    // --- fixtures ----------------------------------------------------------------------

    /// A client that echoes the routed model name back as the completion.
    struct EchoClient;

    #[async_trait]
    impl RoutedLlmClient for EchoClient {
        async fn call(
            &self,
            _ctx: Context,
            _request: Request,
            decision: Arc<dyn Decision>,
        ) -> std::result::Result<Response, switchyard_protocol::LlmClientError> {
            Ok(Response {
                llm_response: LlmResponse::Agg(text_response(
                    None,
                    decision.selected_model().to_string(),
                )),
                metadata: None,
            })
        }
    }

    /// A client that captures the request it was handed, so a test can assert on what
    /// actually reached the model.
    struct CapturingClient(Arc<parking_lot::Mutex<Option<Request>>>);

    #[async_trait]
    impl RoutedLlmClient for CapturingClient {
        async fn call(
            &self,
            _ctx: Context,
            request: Request,
            decision: Arc<dyn Decision>,
        ) -> std::result::Result<Response, switchyard_protocol::LlmClientError> {
            *self.0.lock() = Some(request);
            Ok(Response {
                llm_response: LlmResponse::Agg(text_response(
                    None,
                    decision.selected_model().to_string(),
                )),
                metadata: None,
            })
        }
    }

    /// A target set whose targets all serve via [`EchoClient`].
    fn target_set(names: &[&str]) -> LlmTargetSet {
        LlmTargetSet::new(
            names
                .iter()
                .map(|name| LlmTarget {
                    semantic_name: name.to_string(),
                    llm_client: Some(Arc::new(EchoClient) as Arc<dyn RoutedLlmClient>),
                })
                .collect(),
        )
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
    ) -> Result<(String, Vec<Arc<dyn Decision>>)>
    where
        S: Default + Send + 'static,
    {
        let (trace, response) = router.clone().run(Context::default(), request).await?;
        let text = response
            .llm_response
            .into_agg()
            .await
            .map(|agg| completion_text(&agg))
            .map_err(|error| LibsyError::external("aggregating fall-through response", error))?;
        Ok((text, trace))
    }

    /// Drives a shared router through one turn in the default test session.
    async fn run_turn<S>(router: &Arc<FallThrough<S>>) -> Result<(String, Vec<Arc<dyn Decision>>)>
    where
        S: Default + Send + 'static,
    {
        run_request(router, request()).await
    }

    /// Drives a fresh router through one turn.
    async fn run(router: FallThrough) -> Result<(String, Vec<Arc<dyn Decision>>)> {
        run_turn(&Arc::new(router)).await
    }

    // --- tests -------------------------------------------------------------------------

    /// Overflows for the named targets and echoes for the rest, recording every call.
    struct OverflowClient {
        overflowing: Vec<&'static str>,
        calls: Option<Arc<Mutex<Vec<String>>>>,
    }

    #[async_trait]
    impl RoutedLlmClient for OverflowClient {
        async fn call(
            &self,
            _ctx: Context,
            _request: Request,
            decision: Arc<dyn Decision>,
        ) -> std::result::Result<Response, LlmClientError> {
            let model = decision.selected_model().to_string();
            if let Some(calls) = &self.calls {
                calls.lock().push(model.clone());
            }
            if self.overflowing.contains(&model.as_str()) {
                return Err(LlmClientError::ContextWindowExceeded {
                    model,
                    message: "prompt is too long".to_string(),
                });
            }
            Ok(Response {
                llm_response: LlmResponse::Agg(text_response(None, model)),
                metadata: None,
            })
        }
    }

    /// `target_set`, but the named `overflowing` targets reject every call with a
    /// context-window error so the retry path can be driven.
    fn target_set_with_overflow(names: &[&str], overflowing: &[&'static str]) -> LlmTargetSet {
        LlmTargetSet::new(
            names
                .iter()
                .map(|name| LlmTarget {
                    semantic_name: name.to_string(),
                    llm_client: Some(Arc::new(OverflowClient {
                        overflowing: overflowing.to_vec(),
                        calls: None,
                    }) as Arc<dyn RoutedLlmClient>),
                })
                .collect(),
        )
    }

    fn counting_overflow_targets(
        names: &[&str],
        overflowing: &[&'static str],
        calls: Arc<Mutex<Vec<String>>>,
    ) -> LlmTargetSet {
        LlmTargetSet::new(
            names
                .iter()
                .map(|name| LlmTarget {
                    semantic_name: name.to_string(),
                    llm_client: Some(Arc::new(OverflowClient {
                        overflowing: overflowing.to_vec(),
                        calls: Some(calls.clone()),
                    }) as Arc<dyn RoutedLlmClient>),
                })
                .collect(),
        )
    }

    #[tokio::test]
    async fn a_target_that_overflowed_is_skipped_for_the_rest_of_the_session() -> Result<()> {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let router = Arc::new(
            FallThrough::<()>::new(counting_overflow_targets(
                &["weak", "strong"],
                &["weak"],
                calls.clone(),
            ))
            .with_classifier(fixed(vec![score("weak", 0.9)])),
        );
        for _ in 0..3 {
            assert_eq!(run_turn(&router).await?.0, "strong");
        }
        assert_eq!(calls.lock().iter().filter(|m| *m == "weak").count(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn a_different_session_starts_with_an_empty_eviction_set() -> Result<()> {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let router = Arc::new(
            FallThrough::<()>::new(counting_overflow_targets(
                &["weak", "strong"],
                &["weak"],
                calls.clone(),
            ))
            .with_classifier(fixed(vec![score("weak", 0.9)])),
        );
        run_turn(&router).await?;
        let mut other = request();
        other.metadata = Some(Metadata {
            session_id: Some("session-2".to_string()),
            ..Metadata::default()
        });
        run_request(&router, other).await?;
        assert_eq!(calls.lock().iter().filter(|m| *m == "weak").count(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn second_turn_after_full_exhaustion_still_reaches_upstream() -> Result<()> {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let router = Arc::new(
            FallThrough::<()>::new(counting_overflow_targets(
                &["weak", "strong"],
                &["weak", "strong"],
                calls.clone(),
            ))
            .with_classifier(fixed(vec![score("weak", 0.9)])),
        );
        let first = run_turn(&router).await;
        assert!(first.is_err());
        calls.lock().clear();
        match run_turn(&router).await {
            Err(LibsyError::ClientCall { .. }) => {}
            Err(other) => panic!("turn 2 gave {other:?}, calls={:?}", calls.lock()),
            Ok(_) => panic!("expected an error"),
        }
        Ok(())
    }

    #[tokio::test]
    async fn an_overflowing_target_is_retried_on_one_that_fits() -> Result<()> {
        let router =
            FallThrough::<()>::new(target_set_with_overflow(&["weak", "strong"], &["weak"]))
                .with_classifier(fixed(vec![score("weak", 0.9)]));
        let (model, _) = run(router).await?;
        assert_eq!(model, "strong");
        Ok(())
    }

    #[tokio::test]
    async fn overflowing_targets_are_retried_until_one_fits() -> Result<()> {
        let router = FallThrough::<()>::new(target_set_with_overflow(
            &["weak", "mid", "strong"],
            &["weak", "mid"],
        ))
        .with_classifier(fixed(vec![score("weak", 0.9)]));
        let (model, _) = run(router).await?;
        assert_eq!(model, "strong");
        Ok(())
    }

    #[tokio::test]
    async fn exhausting_every_target_surfaces_the_client_overflow() -> Result<()> {
        // Only the client error maps to a 400 upstream, so it must survive exhaustion.
        let router = FallThrough::<()>::new(target_set_with_overflow(
            &["weak", "strong"],
            &["weak", "strong"],
        ))
        .with_classifier(fixed(vec![score("weak", 0.9)]));
        match run(router).await {
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
        let router =
            FallThrough::<()>::new(target_set_with_overflow(&["weak", "strong"], &["weak"]))
                .with_classifier(fixed(vec![score("weak", 0.9)]))
                .with_processor(Arc::new(CountingProcessor(seen.clone())));
        let (model, _) = run(router).await?;
        assert_eq!(model, "strong");
        assert_eq!(seen.lock().iter().filter(|e| **e == "request").count(), 1);
        assert_eq!(seen.lock().iter().filter(|e| **e == "decision").count(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn an_excluded_target_reports_the_target_it_fell_back_to() -> Result<()> {
        // Headers and usage metrics read the decision, so it must describe the real call.
        let router = Arc::new(
            FallThrough::<()>::new(target_set(&["weak", "strong"]))
                .with_classifier(fixed(vec![score("weak", 0.9)])),
        );
        let mut ctx = Context::default();
        ctx.exclude_target("weak");
        let (trace, response) = router.run(ctx, request()).await?;
        let text = response
            .llm_response
            .into_agg()
            .await
            .map(|agg| completion_text(&agg))
            .map_err(|error| LibsyError::external("aggregating fall-through response", error))?;

        assert_eq!(text, "strong");
        assert_eq!(trace[0].selected_model(), "strong");
        assert!(trace[0]
            .reasoning()
            .is_some_and(|r| r.contains("fell back to strong")));
        Ok(())
    }

    #[tokio::test]
    async fn argmax_picks_the_highest_confidence_target() -> Result<()> {
        let router = FallThrough::<()>::new(target_set(&["strong", "weak"]))
            .with_classifier(fixed(vec![score("weak", 0.2), score("strong", 0.9)]));
        let (model, trace) = run(router).await?;
        assert_eq!(model, "strong");
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].selected_model(), "strong");
        Ok(())
    }

    #[tokio::test]
    async fn falls_through_the_first_abstaining_classifier() -> Result<()> {
        // First classifier abstains (empty); the second decides.
        let router = FallThrough::<()>::new(target_set(&["strong", "weak"]))
            .with_classifier(fixed(vec![]))
            .with_classifier(fixed(vec![score("weak", 1.0)]));
        let (model, _) = run(router).await?;
        assert_eq!(model, "weak");
        Ok(())
    }

    #[tokio::test]
    async fn first_deciding_classifier_wins_the_cascade() -> Result<()> {
        // The first classifier decides; the second is never consulted.
        let router = FallThrough::<()>::new(target_set(&["strong", "weak"]))
            .with_classifier(fixed(vec![score("strong", 0.6)]))
            .with_classifier(fixed(vec![score("weak", 1.0)]));
        let (model, _) = run(router).await?;
        assert_eq!(model, "strong");
        Ok(())
    }

    #[tokio::test]
    async fn all_abstaining_is_an_error() -> Result<()> {
        let router =
            FallThrough::<()>::new(target_set(&["strong", "weak"])).with_classifier(fixed(vec![]));
        let error = run(router)
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
        let (model, _) = run(router).await?;
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
        run(router).await?;

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
        let targets = LlmTargetSet::new(vec![LlmTarget {
            semantic_name: "strong".to_string(),
            llm_client: Some(Arc::new(CapturingClient(seen_by_model.clone()))),
        }]);
        let router = FallThrough::new(targets)
            .with_processor(Arc::new(Appender("first")))
            .with_processor(Arc::new(Appender("second")))
            .with_classifier(Arc::new(TrailClassifier(seen_by_classifier.clone())));

        run_turn(&Arc::new(router)).await?;

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

        let (turn1, _) = run_turn(&router).await?;
        let (turn2, _) = run_turn(&router).await?;
        let (second_session, _) = run_request(
            &router,
            Request {
                metadata: Some(Metadata {
                    session_id: Some("session-2".to_string()),
                    ..Metadata::default()
                }),
                ..request()
            },
        )
        .await?;
        let anonymous = Request {
            metadata: None,
            ..request()
        };
        let (anonymous1, _) = run_request(&router, anonymous.clone()).await?;
        let (anonymous2, _) = run_request(&router, anonymous).await?;

        assert_eq!(turn1, "weak");
        assert_eq!(turn2, "strong");
        assert_eq!(second_session, "weak");
        assert_eq!(anonymous1, "weak");
        assert_eq!(anonymous2, "weak");
        Ok(())
    }
}
