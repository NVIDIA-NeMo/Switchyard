// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fall-through classification: a processor chain, then a classifier cascade.
//!
//! [`FallThroughClassification`] is the shared routing skeleton. For each request it:
//!
//! 1. runs its [`Processor`] chain over [`Event::Request`], accumulating facts into a shared
//!    [`State`];
//! 2. tries each [`Classifier`] in turn — the first to return a non-empty score set decides,
//!    by argmax over that set's [`Score`]s;
//! 3. resolves the winning target in the [`LlmTargetSet`], publishes the [`Decision`], and
//!    replays it to the processors over [`Event::Decision`] so stateful ones (session latch /
//!    affinity) can bind it;
//! 4. dispatches the routed model call through the [`Driver`].
//!
//! Different component lists express different routers over the one skeleton:
//!
//! - `[LlmClassifier]` → LLM-classifier routing;
//! - `[StagedRouter, fallback]` → staged routing (staged abstains in its band; the fallback
//!   decides);
//! - `[latch, turn-gate, judge]` (with a latch processor) → escalation routing.
//!
//! A classifier that offloads its own model call (e.g. [`LlmClassifier`]) receives the
//! per-request [`Driver`] through [`Classifier::score`]; classifiers that decide locally
//! ignore it. Every classifier is a shared instance.
//!
//! ## Session state
//!
//! The [`State`] is **owned by the router and persists for its lifetime** — one
//! `FallThroughClassification` is one session, and its `State` carries facts across turns.
//! That is what lets a stateful component (a session latch / [`AffinityRouter`]) remember a
//! decision from an earlier turn. The `State` sits behind a lock held for the whole
//! request→decision fold of a turn (so a turn's fact accumulation is atomic and turns are
//! serialized), then released before the routed model call.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use super::core::{Classifier, Event, Processor, Score, State};
use crate::{Algorithm, Context, Decision, Driver, LlmTargetSet, Request, Response};

/// Boxed, thread-safe error type used across the SDK.
type BoxErr = Box<dyn std::error::Error + Send + Sync>;

/// The decision a fall-through run produces: the selected model plus a human-readable reason.
struct RoutedDecision {
    model: String,
    reason: String,
}

impl Decision for RoutedDecision {
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
/// Holds the session [`State`], so one instance is one session (see the module docs).
pub struct FallThroughClassification {
    processors: Vec<Arc<dyn Processor>>,
    classifiers: Vec<Arc<dyn Classifier>>,
    targets: LlmTargetSet,
    /// Session state, persisted across turns and shared under a lock (see the module docs).
    state: Mutex<State>,
}

impl FallThroughClassification {
    /// Creates an empty router over `targets`; add components with the `with_*` builders.
    pub fn new(targets: LlmTargetSet) -> Self {
        Self {
            processors: Vec::new(),
            classifiers: Vec::new(),
            targets,
            state: Mutex::new(State::default()),
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

/// The highest-confidence score, or `None` when the set is empty (the classifier abstained).
/// Ties keep the first — cascade order is the tie-break.
fn argmax(scores: Vec<Score>) -> Option<Score> {
    scores.into_iter().reduce(|best, next| {
        if next.confidence > best.confidence {
            next
        } else {
            best
        }
    })
}

#[async_trait]
impl Algorithm for FallThroughClassification {
    async fn create_run_task(
        self: Arc<Self>,
        ctx: Context,
        driver: Driver,
        request: Request,
    ) -> Result<Response, BoxErr> {
        // Hold the session state for the whole request→decision fold, so a turn's fact
        // accumulation is atomic and concurrent turns serialize on it.
        let mut state = self.state.lock().await;

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
            if let Some(score) = argmax(scores) {
                winner = Some(score);
                break;
            }
        }
        let Some(winner) = winner else {
            return Err("fall-through: every classifier abstained".into());
        };

        // 3. Resolve the target and publish the decision.
        let target = self.targets.get_target(&winner.target)?;
        let decision: Arc<dyn Decision> = Arc::new(RoutedDecision {
            model: winner.target.clone(),
            reason: format!(
                "fall-through selected {} (confidence {:.3})",
                winner.target, winner.confidence
            ),
        });
        driver.info(ctx.clone(), decision.clone()).await?;

        // 4. Replay the decision to the processors so stateful ones (latch / affinity) bind
        //    it into the session State.
        for processor in &self.processors {
            processor
                .process(&mut state, Event::Decision(decision.as_ref()))
                .await?;
        }

        // 5. Release the session state before the (long) routed model call.
        drop(state);
        driver
            .call_llm_target(ctx, &target, request, decision)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use switchyard_protocol::{
        completion_text, text_response, LlmRequest, Message, Metadata, Role,
    };

    use crate::{AffinityRouter, LlmResponse, LlmTarget, RoutedLlmClient};

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
        ) -> Result<Response, BoxErr> {
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
        ) -> Result<Vec<Score>, BoxErr> {
            Ok(self
                .0
                .iter()
                .map(|s| Score {
                    confidence: s.confidence,
                    target: s.target.clone(),
                })
                .collect())
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

    /// Drives a shared router through one turn, returning the completion text + trace.
    async fn run_turn(
        router: &Arc<FallThroughClassification>,
    ) -> Result<(String, Vec<Arc<dyn Decision>>), BoxErr> {
        let (trace, response) = router.clone().run(Context::default(), request()).await?;
        let text = response
            .llm_response
            .into_agg()
            .await
            .map(|agg| completion_text(&agg))?;
        Ok((text, trace))
    }

    /// Drives a fresh router through one turn.
    async fn run(
        router: FallThroughClassification,
    ) -> Result<(String, Vec<Arc<dyn Decision>>), BoxErr> {
        run_turn(&Arc::new(router)).await
    }

    // --- tests -------------------------------------------------------------------------

    #[tokio::test]
    async fn argmax_picks_the_highest_confidence_target() -> Result<(), BoxErr> {
        let router = FallThroughClassification::new(target_set(&["strong", "weak"]))
            .with_classifier(fixed(vec![score("weak", 0.2), score("strong", 0.9)]));
        let (model, trace) = run(router).await?;
        assert_eq!(model, "strong");
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].selected_model(), "strong");
        Ok(())
    }

    #[tokio::test]
    async fn falls_through_the_first_abstaining_classifier() -> Result<(), BoxErr> {
        // First classifier abstains (empty); the second decides.
        let router = FallThroughClassification::new(target_set(&["strong", "weak"]))
            .with_classifier(fixed(vec![]))
            .with_classifier(fixed(vec![score("weak", 1.0)]));
        let (model, _) = run(router).await?;
        assert_eq!(model, "weak");
        Ok(())
    }

    #[tokio::test]
    async fn first_deciding_classifier_wins_the_cascade() -> Result<(), BoxErr> {
        // The first classifier decides; the second is never consulted.
        let router = FallThroughClassification::new(target_set(&["strong", "weak"]))
            .with_classifier(fixed(vec![score("strong", 0.6)]))
            .with_classifier(fixed(vec![score("weak", 1.0)]));
        let (model, _) = run(router).await?;
        assert_eq!(model, "strong");
        Ok(())
    }

    #[tokio::test]
    async fn all_abstaining_is_an_error() -> Result<(), BoxErr> {
        let router = FallThroughClassification::new(target_set(&["strong", "weak"]))
            .with_classifier(fixed(vec![]));
        assert!(run(router).await.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn classifiers_receive_the_per_request_driver() -> Result<(), BoxErr> {
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
            ) -> Result<Vec<Score>, BoxErr> {
                match driver {
                    Some(_) => Ok(vec![score("strong", 1.0)]),
                    None => Err("expected a driver".into()),
                }
            }
        }

        let router = FallThroughClassification::new(target_set(&["strong", "weak"]))
            .with_classifier(Arc::new(NeedsDriver));
        let (model, _) = run(router).await?;
        assert_eq!(model, "strong");
        Ok(())
    }

    #[tokio::test]
    async fn processor_observes_request_and_decision() -> Result<(), BoxErr> {
        use std::sync::Mutex;

        // Records which event kinds it saw, proving the request-then-decision replay.
        struct RecordingProcessor(Arc<Mutex<Vec<&'static str>>>);

        #[async_trait]
        impl Processor for RecordingProcessor {
            async fn process(&self, _state: &mut State, event: Event<'_>) -> Result<(), BoxErr> {
                let kind = match event {
                    Event::Request(_) => "request",
                    Event::Decision(_) => "decision",
                    _ => "other",
                };
                self.0.lock().map_err(|_| "lock poisoned")?.push(kind);
                Ok(())
            }
        }

        let seen = Arc::new(Mutex::new(Vec::new()));
        let router = FallThroughClassification::new(target_set(&["strong", "weak"]))
            .with_processor(Arc::new(RecordingProcessor(seen.clone())))
            .with_classifier(fixed(vec![score("strong", 1.0)]));
        run(router).await?;

        assert_eq!(
            *seen.lock().map_err(|_| "lock poisoned")?,
            vec!["request", "decision"]
        );
        Ok(())
    }

    #[tokio::test]
    async fn session_state_persists_the_latch_across_turns() -> Result<(), BoxErr> {
        use std::sync::atomic::{AtomicUsize, Ordering};

        // Picks "strong" on the first call and "weak" thereafter — so a second turn that
        // still routes "strong" can only be the affinity latch, held in the persistent State.
        struct StrongThenWeak(AtomicUsize);

        #[async_trait]
        impl Classifier for StrongThenWeak {
            async fn score(
                &self,
                _state: &mut State,
                _request: &Request,
                _driver: Option<&Driver>,
            ) -> Result<Vec<Score>, BoxErr> {
                let tier = if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
                    "strong"
                } else {
                    "weak"
                };
                Ok(vec![score(tier, 1.0)])
            }
        }

        // AffinityRouter plays both roles; its assignment map lives in the router's State.
        let affinity = Arc::new(AffinityRouter::new());
        let router = Arc::new(
            FallThroughClassification::new(target_set(&["strong", "weak"]))
                .with_processor(affinity.clone() as Arc<dyn Processor>)
                .with_classifier(affinity as Arc<dyn Classifier>)
                .with_classifier(Arc::new(StrongThenWeak(AtomicUsize::new(0)))),
        );

        // Turn 1: affinity abstains, the fallback picks "strong", the decision pins it.
        let (turn1, _) = run_turn(&router).await?;
        assert_eq!(turn1, "strong");

        // Turn 2: the fallback would now pick "weak", but the persisted latch wins — proving
        // State survives across turns.
        let (turn2, _) = run_turn(&router).await?;
        assert_eq!(turn2, "strong");
        Ok(())
    }
}
