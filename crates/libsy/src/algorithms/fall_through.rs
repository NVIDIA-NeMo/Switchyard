// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fall-through classifier routing: a stateful [`Algorithm`] that routes each turn
//! through a processor chain and a classifier cascade.
//!
//! Each turn: request-side [`Processor`]s fold facts into the session [`State`]; the
//! [`Classifier`] cascade is consulted in order and the first to score decides the target
//! (its `argmax`); the [`Decision`] is published and then replayed to the processors so
//! stateful ones (latch, affinity) can bind it.
//!
//! [`FallThrough`] itself is stateless — one shared instance serves every session. The
//! per-session [`State`] rides in the threaded [`Context<SharedState>`] the caller passes into
//! each turn, so routing can depend on accumulated history rather than the current request
//! alone. The state is held under a lock, so concurrent turns of the same session serialize
//! on it.

use std::sync::Arc;

use async_trait::async_trait;

use crate::core::{Classifier, Event, Processor, Score, SharedState};
use crate::{
    Algorithm, Context, Decision, Driver, LibsyError, LlmTargetSet, Request, Response, Result,
};

/// The decision a fall-through run produces: the selected model plus a human-readable reason.
pub struct FallThroughDecision {
    model: String,
    reason: String,
}

impl Decision for FallThroughDecision {
    fn selected_model(&self) -> &str {
        &self.model
    }

    fn reasoning(&self) -> Option<&str> {
        Some(&self.reason)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Processor chain → classifier cascade → routed model call. See the [module docs](self).
///
/// Stateless: the per-session [`State`] lives in the threaded [`Context<SharedState>`], so one
/// shared instance serves every session (see the module docs).
pub struct FallThrough {
    processors: Vec<Arc<dyn Processor>>,
    classifiers: Vec<Arc<dyn Classifier>>,
    targets: LlmTargetSet,
}

impl FallThrough {
    /// Creates an empty router over `targets`; add components with the `with_*` builders.
    pub fn new(targets: LlmTargetSet) -> Self {
        Self {
            processors: Vec::new(),
            classifiers: Vec::new(),
            targets,
        }
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
}
#[async_trait]
impl Algorithm<SharedState> for FallThrough {
    fn name(&self) -> &str {
        "fall_through"
    }

    async fn create_run_task(
        self: Arc<Self>,
        ctx: Context<SharedState>,
        driver: Driver,
        request: Request,
    ) -> Result<Response> {
        // Hold the session state (carried in the threaded context) for the whole
        // request→decision fold, so a turn's fact accumulation is atomic and concurrent
        // turns of the same session serialize on it.
        let mut state = ctx.state.lock().await;

        // 1. Processor chain accumulates request-side facts into the session State.
        for processor in &self.processors {
            processor
                .process(&mut state, Event::Request(&request))
                .await?;
        }

        // 2. Fall through the cascade: the first classifier to score decides (argmax). The
        //    per-request driver is offered to each — driver-backed classifiers use it.
        let mut winner: Option<Score> = None;
        for classifier in &self.classifiers {
            let scores = classifier
                .score(&mut state, &request, Some(&driver))
                .await?;
            if let Some(score) = scores.argmax(false)? {
                winner = Some(score);
                break;
            }
        }
        let Some(winner) = winner else {
            return Err(LibsyError::AllClassifiersAbstained);
        };

        // 3. Resolve the target and publish the decision.
        let target = self.targets.get_target(&winner.target)?;
        let decision: Arc<dyn Decision> = Arc::new(FallThroughDecision {
            model: winner.target.clone(),
            reason: format!(
                "fall-through selected {} (confidence {:.3})",
                winner.target, winner.confidence
            ),
        });
        driver.info(ctx.without_state(), decision.clone()).await?;

        // 4. Replay the decision to the processors so stateful ones (latch / affinity) bind
        //    it into the session State.
        for processor in &self.processors {
            processor
                .process(&mut state, Event::Decision(decision.as_ref()))
                .await?;
        }

        driver
            .call_llm_target(ctx.without_state(), &target, request, decision)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Classification, State};

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
        ) -> std::result::Result<Response, Box<dyn std::error::Error + Send + Sync>> {
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
            _state: &mut State,
            _request: &Request,
            _driver: Option<&Driver>,
        ) -> Result<Classification> {
            Ok(Classification::Scores(
                self.0
                    .iter()
                    .map(|s| Score {
                        confidence: s.confidence,
                        target: s.target.clone(),
                    })
                    .collect(),
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

    /// Drives a shared router through one turn on the given session context, returning the
    /// completion text + trace. Reuse the same `ctx` across calls to model a session.
    async fn run_turn(
        router: &Arc<FallThrough>,
        ctx: Context<SharedState>,
    ) -> Result<(String, Vec<Arc<dyn Decision>>)> {
        let (trace, response) = router.clone().run(ctx, request()).await?;
        let text = response
            .llm_response
            .into_agg()
            .await
            .map(|agg| completion_text(&agg))
            .map_err(|error| {
                LibsyError::external_boxed("aggregating fall-through response", error)
            })?;
        Ok((text, trace))
    }

    /// Drives a fresh router through one turn on a fresh session.
    async fn run(router: FallThrough) -> Result<(String, Vec<Arc<dyn Decision>>)> {
        run_turn(&Arc::new(router), Context::default()).await
    }

    // --- tests -------------------------------------------------------------------------

    #[tokio::test]
    async fn argmax_picks_the_highest_confidence_target() -> Result<()> {
        let router = FallThrough::new(target_set(&["strong", "weak"]))
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
        let router = FallThrough::new(target_set(&["strong", "weak"]))
            .with_classifier(fixed(vec![]))
            .with_classifier(fixed(vec![score("weak", 1.0)]));
        let (model, _) = run(router).await?;
        assert_eq!(model, "weak");
        Ok(())
    }

    #[tokio::test]
    async fn first_deciding_classifier_wins_the_cascade() -> Result<()> {
        // The first classifier decides; the second is never consulted.
        let router = FallThrough::new(target_set(&["strong", "weak"]))
            .with_classifier(fixed(vec![score("strong", 0.6)]))
            .with_classifier(fixed(vec![score("weak", 1.0)]));
        let (model, _) = run(router).await?;
        assert_eq!(model, "strong");
        Ok(())
    }

    #[tokio::test]
    async fn all_abstaining_is_an_error() -> Result<()> {
        let router =
            FallThrough::new(target_set(&["strong", "weak"])).with_classifier(fixed(vec![]));
        let error = run(router)
            .await
            .err()
            .ok_or_else(|| test_error("expected classifiers to abstain"))?;
        assert!(matches!(error, LibsyError::AllClassifiersAbstained));
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
                _state: &mut State,
                _request: &Request,
                driver: Option<&Driver>,
            ) -> Result<Classification> {
                match driver {
                    Some(_) => Ok(Classification::Scores(vec![score("strong", 1.0)])),
                    None => Err(test_error("expected a driver")),
                }
            }
        }

        let router = FallThrough::new(target_set(&["strong", "weak"]))
            .with_classifier(Arc::new(NeedsDriver));
        let (model, _) = run(router).await?;
        assert_eq!(model, "strong");
        Ok(())
    }

    #[tokio::test]
    async fn processor_observes_request_and_decision() -> Result<()> {
        use std::sync::Mutex;

        // Records which event kinds it saw, proving the request-then-decision replay.
        struct RecordingProcessor(Arc<Mutex<Vec<&'static str>>>);

        #[async_trait]
        impl Processor for RecordingProcessor {
            async fn process(&self, _state: &mut State, event: Event<'_>) -> Result<()> {
                let kind = match event {
                    Event::Request(_) => "request",
                    Event::Decision(_) => "decision",
                    _ => "other",
                };
                self.0
                    .lock()
                    .map_err(|_| test_error("lock poisoned"))?
                    .push(kind);
                Ok(())
            }
        }

        let seen = Arc::new(Mutex::new(Vec::new()));
        let router = FallThrough::new(target_set(&["strong", "weak"]))
            .with_processor(Arc::new(RecordingProcessor(seen.clone())))
            .with_classifier(fixed(vec![score("strong", 1.0)]));
        run(router).await?;

        assert_eq!(
            *seen.lock().map_err(|_| test_error("lock poisoned"))?,
            vec!["request", "decision"]
        );
        Ok(())
    }

    #[tokio::test]
    async fn state_persists_across_turns() -> Result<()> {
        // Increments the session turn count on every request.
        struct CountingProcessor;

        #[async_trait]
        impl Processor for CountingProcessor {
            async fn process(&self, state: &mut State, event: Event<'_>) -> Result<()> {
                if let Event::Request(_) = event {
                    state.turn_count += 1;
                }
                Ok(())
            }
        }

        // Routes to "weak" until the session has accumulated >= 2 turns, then "strong" —
        // its decision depends only on state carried over from earlier turns.
        struct ThresholdClassifier;

        #[async_trait]
        impl Classifier for ThresholdClassifier {
            async fn score(
                &self,
                state: &mut State,
                _request: &Request,
                _driver: Option<&Driver>,
            ) -> Result<Classification> {
                let target = if state.turn_count >= 2 {
                    "strong"
                } else {
                    "weak"
                };
                Ok(Classification::Scores(vec![score(target, 1.0)]))
            }
        }

        // One shared, stateless router; the session lives in the context we thread through.
        let router = Arc::new(
            FallThrough::new(target_set(&["strong", "weak"]))
                .with_processor(Arc::new(CountingProcessor))
                .with_classifier(Arc::new(ThresholdClassifier)),
        );

        // One session context, reused across three turns. The turn counter accumulates in
        // its `State`, so the classifier crosses its threshold on turn 2.
        let ctx = Context::<SharedState>::default();
        let (turn1, _) = run_turn(&router, ctx.clone()).await?;
        let (turn2, _) = run_turn(&router, ctx.clone()).await?;
        let (turn3, _) = run_turn(&router, ctx.clone()).await?;

        assert_eq!(turn1, "weak"); // count 1 — below threshold
        assert_eq!(turn2, "strong"); // count 2 — state carried over from turn 1
        assert_eq!(turn3, "strong"); // count 3 — still above threshold
        Ok(())
    }
}
