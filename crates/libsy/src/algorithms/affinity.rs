// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Model affinity as a single SDK component.
//!
//! [`AffinityRouter`] retains the first model chosen for a request's stable identity and
//! forces that model on later requests sharing the identity. It is one object that plays
//! both SDK roles, so registering it as a processor and a classifier cannot drift apart:
//!
//! - As a [`Processor`] it *writes* the assignment: it captures the request's identity on
//!   [`Event::Request`] and binds it to the chosen model on [`Event::Decision`], the first
//!   assignment for an identity winning.
//! - As a [`Classifier`] it *reads* the assignment: it scores the retained model with
//!   confidence `1.0`, or returns no scores to abstain.
//!
//! Identity is derived from correlation metadata: by default, a request is keyed by its
//! session, and a sub-agent is keyed more finely by `session + agent` — a subset of its
//! session. [`AffinityRouter::for_subagents`] narrows affinity to explicitly identified
//! child agents, leaving root traffic to later classifiers on every turn.

use std::collections::{HashMap, HashSet};

use async_trait::async_trait;
use switchyard_protocol::Request;

use crate::{Classification, Classifier, Event, Processor, Score, State};

/// Boxed, thread-safe error type used across the SDK.
type BoxErr = Box<dyn std::error::Error + Send + Sync>;

/// Upper bound on retained assignments, keeping the process-local map from growing
/// without limit; the oldest entry is evicted once the bound is reached.
const MAX_ASSIGNMENTS: usize = 4096;

/// The stable identity a model assignment is retained against.
///
/// A sub-agent request is keyed by `session + agent` and a root request by session alone,
/// so a sub-agent's assignment is scoped within — but distinct from — its session's.
#[derive(Clone, Hash, PartialEq, Eq)]
enum AffinityKey {
    /// One model per session, for root-agent traffic.
    Session(String),
    /// One model per identified child agent within a session.
    Subagent { session: String, agent: String },
}

/// The affinity memory folded into [`State`]: the retained model per identity, plus the
/// key captured from the current turn's request while its decision is pending.
#[derive(Default)]
struct AffinityAssignments {
    /// Identity → retained model, first assignment winning.
    assignments: HashMap<AffinityKey, String>,
    /// Key captured on this turn's [`Event::Request`], consumed by its [`Event::Decision`].
    pending: Option<AffinityKey>,
}

/// Retains a model per request identity and forces it on later matching requests.
///
/// Register the same instance as both a processor and a classifier; the two roles share
/// the retained assignments through [`State`]. Because a decision event carries only the
/// chosen model, the request's identity is captured first (on [`Event::Request`]) and
/// consumed when the decision arrives — this assumes a turn's request is observed before
/// its decision on the same [`State`], as the driving algorithm folds a turn in order.
///
/// [`with_latch_only`](Self::with_latch_only) narrows *which* models are retained — a
/// decision for any other model routes normally but is not latched (the escalation latch:
/// retain only the strong tier, never the weak one).
#[derive(Default)]
pub struct AffinityRouter {
    /// When set, only these models are retained; a decision for any other model is not latched.
    latch_only: Option<HashSet<String>>,
    /// Whether root-session requests should abstain instead of being retained.
    subagents_only: bool,
}

impl AffinityRouter {
    /// Creates a router that latches every decision.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a router that retains assignments only for explicitly identified child agents.
    ///
    /// Root-agent requests always abstain, so a later classifier selects them on every turn.
    pub fn for_subagents() -> Self {
        Self {
            subagents_only: true,
            ..Self::default()
        }
    }

    /// Restricts latching to `models`; a decision for any other model routes but is not
    /// retained.
    pub fn with_latch_only(mut self, models: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.latch_only = Some(models.into_iter().map(Into::into).collect());
        self
    }

    /// Whether a decision for `model` should be retained.
    fn should_latch(&self, model: &str) -> bool {
        self.latch_only
            .as_ref()
            .is_none_or(|set| set.contains(model))
    }

    /// Derives the stable identity this router should retain for `request`.
    fn affinity_key(&self, request: &Request) -> Option<AffinityKey> {
        let metadata = request.metadata.as_ref()?;
        let session = metadata.session_id.clone()?;
        if metadata.is_subagent {
            Some(AffinityKey::Subagent {
                session,
                agent: metadata.agent_id.clone()?,
            })
        } else if self.subagents_only {
            None
        } else {
            Some(AffinityKey::Session(session))
        }
    }
}

#[async_trait]
impl Processor for AffinityRouter {
    async fn process(&self, state: &mut State, event: Event<'_>) -> Result<(), BoxErr> {
        match event {
            // Capture the identity now; the model it maps to is only known at the decision.
            Event::Request(request) => {
                state
                    .entry_or_insert_with(AffinityAssignments::default)
                    .pending = self.affinity_key(request);
            }
            // Bind the captured identity to the chosen model, keeping any earlier winner.
            Event::Decision(decision) => {
                let model = decision.selected_model();
                let fact = state.entry_or_insert_with(AffinityAssignments::default);
                let Some(key) = fact.pending.take() else {
                    return Ok(());
                };
                // Latch only permitted models; others consume `pending` but are not retained.
                if self.should_latch(model) && !fact.assignments.contains_key(&key) {
                    evict_if_full(&mut fact.assignments);
                    fact.assignments.insert(key, model.to_string());
                }
            }
            _ => {}
        }
        Ok(())
    }
}

#[async_trait]
impl Classifier for AffinityRouter {
    async fn score(
        &self,
        state: &mut State,
        request: &Request,
        _driver: Option<&crate::Driver>,
    ) -> Result<Classification, BoxErr> {
        let Some(key) = self.affinity_key(request) else {
            return Ok(Classification::Scores(Vec::new()));
        };
        let assigned = state
            .get::<AffinityAssignments>()
            .and_then(|fact| fact.assignments.get(&key).cloned());
        Ok(Classification::Scores(match assigned {
            Some(target) => vec![Score {
                confidence: 1.0,
                target,
            }],
            None => Vec::new(),
        }))
    }
}

/// Evicts one arbitrary assignment when the map has reached [`MAX_ASSIGNMENTS`].
fn evict_if_full(assignments: &mut HashMap<AffinityKey, String>) {
    if assignments.len() >= MAX_ASSIGNMENTS {
        if let Some(evicted) = assignments.keys().next().cloned() {
            assignments.remove(&evicted);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use switchyard_protocol::{text_request, Decision, Metadata};

    /// A decision that reports a fixed selected model.
    struct FixedDecision(&'static str);

    impl Decision for FixedDecision {
        fn selected_model(&self) -> &str {
            self.0
        }

        fn reasoning(&self) -> Option<&str> {
            None
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    fn request(metadata: Metadata) -> Request {
        Request {
            llm_request: text_request(Some("auto".to_string()), "hi"),
            raw_request: None,
            metadata: Some(metadata),
        }
    }

    fn session(session_id: &str, agent_id: &str) -> Metadata {
        Metadata {
            session_id: Some(session_id.to_string()),
            agent_id: Some(agent_id.to_string()),
            ..Metadata::default()
        }
    }

    fn subagent(agent_id: &str, task_id: &str) -> Metadata {
        Metadata {
            session_id: Some("session-1".to_string()),
            agent_id: Some(agent_id.to_string()),
            task_id: Some(task_id.to_string()),
            is_subagent: true,
            ..Metadata::default()
        }
    }

    /// Folds a request and its decision through the router, retaining `model`.
    async fn retain(
        router: &AffinityRouter,
        state: &mut State,
        request: &Request,
        model: &'static str,
    ) -> Result<(), BoxErr> {
        router.process(state, Event::Request(request)).await?;
        router
            .process(state, Event::Decision(&FixedDecision(model)))
            .await
    }

    /// Scores through the definitive classification variant used by affinity.
    async fn scores(
        classifier: &dyn Classifier,
        state: &mut State,
        request: &Request,
    ) -> Result<Vec<Score>, BoxErr> {
        match classifier.score(state, request, None).await? {
            Classification::Scores(scores) => Ok(scores),
            Classification::Ambiguous(_) => Err("affinity never returns ambiguous scores".into()),
        }
    }

    #[tokio::test]
    async fn session_retains_first_model_across_requests() -> Result<(), BoxErr> {
        let router = AffinityRouter::new();
        let mut state = State::default();

        let first = request(session("session-1", "agent-a"));
        retain(&router, &mut state, &first, "model-a").await?;

        // A different agent in the same session is scored onto the retained model.
        let second = request(session("session-1", "agent-b"));
        let scores = scores(&router, &mut state, &second).await?;
        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].confidence, 1.0);
        assert_eq!(scores[0].target, "model-a");
        Ok(())
    }

    #[tokio::test]
    async fn subagent_only_retains_children_without_latching_root_traffic() -> Result<(), BoxErr> {
        let router = AffinityRouter::for_subagents();
        let mut state = State::default();

        let root = request(session("session-1", "root-agent"));
        retain(&router, &mut state, &root, "model-a").await?;
        assert!(scores(&router, &mut state, &root).await?.is_empty());

        let first_child_turn = request(subagent("child-1", "task-1"));
        retain(&router, &mut state, &first_child_turn, "model-b").await?;
        let later_child_turn = request(subagent("child-1", "task-2"));
        let scores = scores(&router, &mut state, &later_child_turn).await?;
        assert_eq!(
            scores.first().map(|score| score.target.as_str()),
            Some("model-b")
        );
        Ok(())
    }

    #[tokio::test]
    async fn first_decision_wins() -> Result<(), BoxErr> {
        let router = AffinityRouter::new();
        let mut state = State::default();

        let req = request(session("session-1", "agent-a"));
        retain(&router, &mut state, &req, "model-a").await?;
        // A later decision for the same identity must not overwrite the first.
        retain(&router, &mut state, &req, "model-b").await?;

        let scores = scores(&router, &mut state, &req).await?;
        assert_eq!(
            scores.first().map(|score| score.target.as_str()),
            Some("model-a")
        );
        Ok(())
    }

    #[tokio::test]
    async fn subagent_is_keyed_by_agent_not_task() -> Result<(), BoxErr> {
        let router = AffinityRouter::new();
        let mut state = State::default();

        let first = request(subagent("child-1", "task-1"));
        retain(&router, &mut state, &first, "model-a").await?;

        // Same child, different task: still scored onto the retained model.
        let second = request(subagent("child-1", "task-2"));
        let scores = scores(&router, &mut state, &second).await?;
        assert_eq!(
            scores.first().map(|score| score.target.as_str()),
            Some("model-a")
        );
        Ok(())
    }

    #[tokio::test]
    async fn distinct_subagents_are_assigned_independently() -> Result<(), BoxErr> {
        let router = AffinityRouter::new();
        let mut state = State::default();

        // One child in the session is pinned...
        retain(
            &router,
            &mut state,
            &request(subagent("child-1", "task-1")),
            "model-a",
        )
        .await?;

        // ...a sibling child in the same session has no assignment of its own yet.
        let sibling = request(subagent("child-2", "task-1"));
        assert!(scores(&router, &mut state, &sibling).await?.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn subagent_does_not_inherit_session_assignment() -> Result<(), BoxErr> {
        let router = AffinityRouter::new();
        let mut state = State::default();

        // The session root is pinned, but a sub-agent is keyed separately...
        retain(
            &router,
            &mut state,
            &request(session("session-1", "root-1")),
            "model-a",
        )
        .await?;

        // ...so the sub-agent abstains until it is assigned in its own right.
        let child = request(subagent("child-1", "task-1"));
        assert!(scores(&router, &mut state, &child).await?.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn classifier_abstains_without_a_session() -> Result<(), BoxErr> {
        let router = AffinityRouter::new();
        let mut state = State::default();

        // No session id at all: nothing to key on.
        let req = request(Metadata::default());
        assert!(scores(&router, &mut state, &req).await?.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn one_router_serves_both_roles() -> Result<(), BoxErr> {
        // The same instance is registered under both SDK roles; a decision folded in via
        // the processor handle is read back via the classifier handle.
        let router = Arc::new(AffinityRouter::new());
        let processor: Arc<dyn Processor> = router.clone();
        let classifier: Arc<dyn Classifier> = router;
        let mut state = State::default();

        let first = request(session("session-1", "agent-a"));
        processor
            .process(&mut state, Event::Request(&first))
            .await?;
        processor
            .process(&mut state, Event::Decision(&FixedDecision("model-a")))
            .await?;

        let second = request(session("session-1", "agent-b"));
        let scores = scores(classifier.as_ref(), &mut state, &second).await?;
        assert_eq!(
            scores.first().map(|score| score.target.as_str()),
            Some("model-a")
        );
        Ok(())
    }

    #[tokio::test]
    async fn decision_without_a_request_is_ignored() -> Result<(), BoxErr> {
        let router = AffinityRouter::new();
        let mut state = State::default();

        // A decision with no captured identity (no prior request) stores nothing.
        router
            .process(&mut state, Event::Decision(&FixedDecision("model-a")))
            .await?;

        let req = request(session("session-1", "agent-a"));
        assert!(scores(&router, &mut state, &req).await?.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn unrelated_events_preserve_the_pending_identity() -> Result<(), BoxErr> {
        let router = AffinityRouter::new();
        let mut state = State::default();

        let req = request(session("session-1", "agent-a"));
        router.process(&mut state, Event::Request(&req)).await?;
        // A stray signal between the request and its decision must not drop the captured
        // identity, nor create an assignment of its own.
        router
            .process(&mut state, Event::Signal(&crate::Signals {}))
            .await?;
        router
            .process(&mut state, Event::Decision(&FixedDecision("model-a")))
            .await?;

        let scores = scores(&router, &mut state, &req).await?;
        assert_eq!(
            scores.first().map(|score| score.target.as_str()),
            Some("model-a")
        );
        Ok(())
    }

    #[tokio::test]
    async fn distinct_sessions_are_assigned_independently() -> Result<(), BoxErr> {
        let router = AffinityRouter::new();
        let mut state = State::default();

        retain(
            &router,
            &mut state,
            &request(session("session-1", "agent-a")),
            "model-a",
        )
        .await?;
        retain(
            &router,
            &mut state,
            &request(session("session-2", "agent-a")),
            "model-b",
        )
        .await?;

        let first = scores(&router, &mut state, &request(session("session-1", "other"))).await?;
        let second = scores(&router, &mut state, &request(session("session-2", "other"))).await?;
        assert_eq!(
            first.first().map(|score| score.target.as_str()),
            Some("model-a")
        );
        assert_eq!(
            second.first().map(|score| score.target.as_str()),
            Some("model-b")
        );
        Ok(())
    }

    #[tokio::test]
    async fn subagent_without_an_agent_id_is_not_keyed() -> Result<(), BoxErr> {
        let router = AffinityRouter::new();
        let mut state = State::default();

        // The sub-agent flag is set but no agent id is present, so no key can be formed;
        // the request is neither retained nor scored.
        let metadata = Metadata {
            session_id: Some("session-1".to_string()),
            is_subagent: true,
            ..Metadata::default()
        };
        let req = request(metadata);
        retain(&router, &mut state, &req, "model-a").await?;
        assert!(scores(&router, &mut state, &req).await?.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn assignments_are_bounded_by_the_cap() -> Result<(), BoxErr> {
        let router = AffinityRouter::new();
        let mut state = State::default();

        // One distinct session past the cap forces exactly one eviction.
        for index in 0..=MAX_ASSIGNMENTS {
            let session_id = format!("session-{index}");
            retain(
                &router,
                &mut state,
                &request(session(&session_id, "agent-a")),
                "model-a",
            )
            .await?;
        }

        let len = state
            .get::<AffinityAssignments>()
            .map(|fact| fact.assignments.len());
        assert_eq!(len, Some(MAX_ASSIGNMENTS));
        Ok(())
    }

    #[tokio::test]
    async fn latch_only_retains_matching_models() -> Result<(), BoxErr> {
        let router = AffinityRouter::new().with_latch_only(["strong"]);
        let mut state = State::default();
        let req = request(session("session-1", "agent-a"));

        // A "weak" decision is not retained — a later turn is not latched.
        retain(&router, &mut state, &req, "weak").await?;
        assert!(scores(&router, &mut state, &req).await?.is_empty());

        // A "strong" decision is retained — later turns latch onto it.
        retain(&router, &mut state, &req, "strong").await?;
        assert_eq!(
            scores(&router, &mut state, &req)
                .await?
                .first()
                .map(|s| s.target.as_str()),
            Some("strong")
        );
        Ok(())
    }
}
