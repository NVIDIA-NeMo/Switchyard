// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Model affinity as a single SDK component.
//!
//! [`AffinityRouter`] retains the first model chosen for a request's stable identity and
//! forces that model on later requests sharing the identity. It is one object that plays
//! both SDK roles, so registering it as a processor and a classifier cannot drift apart:
//!
//! - As a [`Processor`] it *writes* the assignment: [`Event::Decision`] carries the request
//!   whose chosen model is retained, with the first assignment for an identity winning.
//! - As a [`Classifier`] it *reads* the assignment: it scores the retained model with
//!   confidence `1.0`, or returns no scores to abstain.
//!
//! Identity is derived from correlation metadata: by default, a request is keyed by its
//! session, and a sub-agent is keyed more finely by `session + agent` — a subset of its
//! session. [`AffinityRouter::for_subagents`] narrows affinity to explicitly identified
//! child agents, leaving root traffic to later classifiers on every turn.

use std::collections::{HashMap, HashSet, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use parking_lot::Mutex;
use switchyard_protocol::{ModelId, Request, Role};

use crate::core::algorithm::{Driver, RoutingIdentity};
use crate::core::classifier::{Classification, Classifier, Score};
use crate::core::processor::{Event, Processor};

/// Upper bound on retained assignments, keeping the process-local map from growing
/// without limit; the oldest entry is evicted once the bound is reached.
const MAX_ASSIGNMENTS: usize = 4096;

/// Retains a model per request identity and forces it on later matching requests.
///
/// Register the same instance as both a processor and a classifier; the two roles share
/// the retained assignments through this instance's interior storage. Decision events carry
/// their originating request, so concurrent turns cannot bind one request's identity to
/// another request's selected model.
///
/// [`with_latch_only`](Self::with_latch_only) narrows *which* models are retained — a
/// decision for any other model routes normally but is not latched (the escalation latch:
/// retain only the strong tier, never the weak one).
#[derive(Default)]
pub struct AffinityRouter {
    /// When set, only these models are retained; a decision for any other model is not latched.
    latch_only: Option<HashSet<ModelId>>,
    /// Whether root-session requests should abstain instead of being retained.
    subagents_only: bool,
    /// In absence of headers, use the message hash based fallback key to do task based routing
    message_hash_fallback: bool,
    /// Retained assignments, shared across this router's processor and classifier roles.
    ///
    /// Held on the instance so the two roles share one process-local map through a
    /// single registered [`Arc`](std::sync::Arc); bounded by [`MAX_ASSIGNMENTS`].
    assignments: Mutex<HashMap<RoutingIdentity, ModelId>>,
    /// Whether the "no identity to key on" warning has already been emitted.
    unkeyed_warning_emitted: AtomicBool,
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

    /// Uses the first user-message text as a fallback key when metadata has no session.
    pub fn with_message_hash_fallback(mut self) -> Self {
        self.message_hash_fallback = true;
        self
    }

    /// Restricts latching to `models`; a decision for any other model routes but is not
    /// retained.
    pub fn with_latch_only(mut self, models: impl IntoIterator<Item = impl Into<ModelId>>) -> Self {
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
    fn affinity_key(&self, request: &Request) -> Option<RoutingIdentity> {
        let metadata = request.metadata.as_ref();
        let is_subagent = metadata.is_some_and(|metadata| metadata.is_subagent);
        // This mode handles only subagent requests; root requests intentionally fall through.
        if self.subagents_only && !is_subagent {
            return None;
        }

        let key = match (RoutingIdentity::from_request(request), is_subagent) {
            (Some(identity), _) => Some(identity),
            // Child requests require their explicit session + agent identity and never fall
            // back to task text.
            (None, true) => None,
            (None, false) if self.message_hash_fallback => {
                first_user_message_hash(request).map(|hash| {
                    tracing::debug!(affinity_key = %hash, "affinity using message hash fallback");
                    RoutingIdentity::Session(hash)
                })
            }
            (None, false) => None,
        };
        // Affinity that never keys anything is silent otherwise: the route reports itself as
        // configured while every turn is classified afresh. Say so once rather than per turn.
        if key.is_none() && self.should_warn_unkeyed() {
            tracing::warn!(
                target: "libsy",
                is_subagent,
                message_hash_fallback = self.message_hash_fallback,
                "affinity is enabled but this request carries no usable identity, so no \
                 affinity is applied; root requests need a session id or message-hash \
                 fallback with usable first-user text, and child requests need both session \
                 and agent ids"
            );
        }
        key
    }

    /// Reports whether this call owns the one-time warning for an unkeyable request.
    fn should_warn_unkeyed(&self) -> bool {
        !self.unkeyed_warning_emitted.swap(true, Ordering::Relaxed)
    }
}

#[async_trait]
impl<S> Processor<S> for AffinityRouter
where
    S: Send + 'static,
{
    async fn process(&self, _state: &mut S, event: Event<'_>) -> crate::Result<()> {
        if let Event::Decision { request, decision } = event
            && let Some(key) = self.affinity_key(request)
        {
            let model = decision.selected_model_id();
            let mut assignments = self.assignments.lock();
            if self.should_latch(model) && !assignments.contains_key(&key) {
                evict_if_full(&mut assignments);
                assignments.insert(key, model.clone());
            }
        }
        Ok(())
    }
}

/// Hashes the first user message so later turns retain the initial task's affinity.
/// For benchmarking purpose with harnesses, task instructions are added as a user prompt to the request so we hash the initial user message.
/// TODO: Have not considered multi-modal payloads yet. That needs to be handled separately.
fn first_user_message_hash(request: &Request) -> Option<String> {
    let message = request
        .llm_request
        .messages
        .iter()
        .find(|message| message.role == Role::User)?;
    let mut hasher = DefaultHasher::new();
    message.text_content("")?.hash(&mut hasher);
    Some(format!("{:016x}", hasher.finish()))
}

#[async_trait]
impl<S> Classifier<S> for AffinityRouter
where
    S: Send + 'static,
{
    async fn score(
        &self,
        _state: &mut S,
        request: &mut Request,
        _driver: Option<&Driver>,
    ) -> crate::Result<(Classification, Option<switchyard_protocol::Response>)> {
        let Some(key) = self.affinity_key(request) else {
            return Ok((Classification::Scores(Vec::new()), None));
        };
        let assigned = self.assignments.lock().get(&key).cloned();
        Ok((
            Classification::Scores(match assigned {
                Some(target) => vec![Score {
                    confidence: 1.0,
                    target,
                }],
                None => Vec::new(),
            }),
            None,
        ))
    }
}

/// Evicts one arbitrary assignment when the map has reached [`MAX_ASSIGNMENTS`].
fn evict_if_full(assignments: &mut HashMap<RoutingIdentity, ModelId>) {
    if assignments.len() >= MAX_ASSIGNMENTS
        && let Some(evicted) = assignments.keys().next().cloned()
    {
        assignments.remove(&evicted);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use switchyard_protocol::{
        ContentBlock, Decision, LlmRequest, Message, Metadata, text_request,
    };

    /// Boxed, thread-safe error type keeping the test helpers ergonomic.
    type BoxErr = Box<dyn std::error::Error + Send + Sync>;

    fn fixed_decision(target: &str) -> Decision {
        Decision::new(target, true)
    }

    fn request(metadata: Metadata) -> Request {
        Request {
            llm_request: text_request(Some("auto".to_string()), "hi"),
            raw_request: None,
            metadata: Some(metadata),
        }
    }

    fn task_request(
        metadata: Option<Metadata>,
        first_user: &str,
        follow_up: Option<&str>,
    ) -> Request {
        let mut messages = vec![
            Message::text(Role::System, "follow repository instructions"),
            Message::text(Role::User, first_user),
        ];
        if let Some(follow_up) = follow_up {
            messages.push(Message::text(Role::Assistant, "I will inspect the code."));
            messages.push(Message::text(Role::User, follow_up));
        }
        Request {
            llm_request: LlmRequest {
                model: Some("auto".to_string()),
                messages,
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata,
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
        state: &mut (),
        request: &mut Request,
        model: &'static str,
    ) -> Result<(), BoxErr> {
        let decision = fixed_decision(model);
        router
            .process(
                state,
                Event::Decision {
                    request,
                    decision: &decision,
                },
            )
            .await?;
        Ok(())
    }

    /// Scores through the definitive classification variant used by affinity.
    async fn scores(
        classifier: &dyn Classifier,
        state: &mut (),
        request: &mut Request,
    ) -> Result<Vec<Score>, BoxErr> {
        match classifier.score(state, request, None).await?.0 {
            Classification::Scores(scores) => Ok(scores),
            Classification::Ambiguous(_) => Err("affinity never returns ambiguous scores".into()),
        }
    }

    #[tokio::test]
    async fn session_retains_first_model_across_requests() -> Result<(), BoxErr> {
        let router = AffinityRouter::new();
        let mut state = ();

        let mut first = request(session("session-1", "agent-a"));
        retain(&router, &mut state, &mut first, "model-a").await?;

        // A different agent in the same session is scored onto the retained model.
        let mut second = request(session("session-1", "agent-b"));
        let scores = scores(&router, &mut state, &mut second).await?;
        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].confidence, 1.0);
        assert_eq!(scores[0].target, "model-a");
        Ok(())
    }

    #[tokio::test]
    async fn subagent_only_retains_children_without_latching_root_traffic() -> Result<(), BoxErr> {
        let router = AffinityRouter::for_subagents();
        let mut state = ();

        let mut root = request(session("session-1", "root-agent"));
        retain(&router, &mut state, &mut root, "model-a").await?;
        assert!(scores(&router, &mut state, &mut root).await?.is_empty());

        let mut first_child_turn = request(subagent("child-1", "task-1"));
        retain(&router, &mut state, &mut first_child_turn, "model-b").await?;
        let mut later_child_turn = request(subagent("child-1", "task-2"));
        let scores = scores(&router, &mut state, &mut later_child_turn).await?;
        assert_eq!(
            scores.first().map(|score| score.target.as_str()),
            Some("model-b")
        );
        Ok(())
    }

    #[tokio::test]
    async fn first_decision_wins() -> Result<(), BoxErr> {
        let router = AffinityRouter::new();
        let mut state = ();

        let mut req = request(session("session-1", "agent-a"));
        retain(&router, &mut state, &mut req, "model-a").await?;
        // A later decision for the same identity must not overwrite the first.
        retain(&router, &mut state, &mut req, "model-b").await?;

        let scores = scores(&router, &mut state, &mut req).await?;
        assert_eq!(
            scores.first().map(|score| score.target.as_str()),
            Some("model-a")
        );
        Ok(())
    }

    #[tokio::test]
    async fn subagent_is_keyed_by_agent_not_task() -> Result<(), BoxErr> {
        let router = AffinityRouter::new();
        let mut state = ();

        let mut first = request(subagent("child-1", "task-1"));
        retain(&router, &mut state, &mut first, "model-a").await?;

        // Same child, different task: still scored onto the retained model.
        let mut second = request(subagent("child-1", "task-2"));
        let scores = scores(&router, &mut state, &mut second).await?;
        assert_eq!(
            scores.first().map(|score| score.target.as_str()),
            Some("model-a")
        );
        Ok(())
    }

    #[tokio::test]
    async fn distinct_subagents_are_assigned_independently() -> Result<(), BoxErr> {
        let router = AffinityRouter::new();
        let mut state = ();

        // One child in the session is pinned...
        retain(
            &router,
            &mut state,
            &mut request(subagent("child-1", "task-1")),
            "model-a",
        )
        .await?;

        // ...a sibling child in the same session has no assignment of its own yet.
        let mut sibling = request(subagent("child-2", "task-1"));
        assert!(scores(&router, &mut state, &mut sibling).await?.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn subagent_does_not_inherit_session_assignment() -> Result<(), BoxErr> {
        let router = AffinityRouter::new();
        let mut state = ();

        // The session root is pinned, but a sub-agent is keyed separately...
        retain(
            &router,
            &mut state,
            &mut request(session("session-1", "root-1")),
            "model-a",
        )
        .await?;

        // ...so the sub-agent abstains until it is assigned in its own right.
        let mut child = request(subagent("child-1", "task-1"));
        assert!(scores(&router, &mut state, &mut child).await?.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn classifier_abstains_without_a_session() -> Result<(), BoxErr> {
        let router = AffinityRouter::new();
        let mut state = ();

        // No session id at all: nothing to key on.
        let mut req = request(Metadata::default());
        assert!(scores(&router, &mut state, &mut req).await?.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn message_hash_fallback_uses_the_first_user_message() -> Result<(), BoxErr> {
        let router = AffinityRouter::new().with_message_hash_fallback();
        let mut state = ();

        let mut first = task_request(
            None,
            "Add a unit test for this function.",
            Some("Now run the test suite."),
        );
        retain(&router, &mut state, &mut first, "weak").await?;

        let mut follow_up = task_request(
            None,
            "Add a unit test for this function.",
            Some("Now file a pull request."),
        );
        assert_eq!(
            scores(&router, &mut state, &mut follow_up)
                .await?
                .first()
                .map(|score| score.target.as_str()),
            Some("weak")
        );

        let mut other_task = task_request(
            None,
            "Reimplement this binary from two input/output pairs.",
            Some("Now run the test suite."),
        );
        assert!(
            scores(&router, &mut state, &mut other_task)
                .await?
                .is_empty()
        );
        Ok(())
    }

    #[tokio::test]
    async fn subagents_only_root_traffic_does_not_warn() -> Result<(), BoxErr> {
        // Abstaining on root traffic is this mode's contract, so it must not warn.
        let router = AffinityRouter::for_subagents();
        let mut state = ();

        let mut root = request(session("session-1", "agent-1"));
        assert!(scores(&router, &mut state, &mut root).await?.is_empty());
        assert!(
            router.should_warn_unkeyed(),
            "an intentional abstention should leave the warning unconsumed"
        );
        Ok(())
    }

    #[test]
    fn user_message_hash_ignores_non_text_provider_payloads() {
        let request = |user_message| Request {
            llm_request: LlmRequest {
                messages: vec![user_message],
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: None,
        };
        let text_only = request(Message::text(Role::User, "Implement the parser."));
        let text_with_reasoning = request(Message {
            role: Role::User,
            content: vec![
                ContentBlock::Text {
                    text: "Implement the parser.".to_string(),
                },
                ContentBlock::Reasoning {
                    text: "Internal provider reasoning.".to_string(),
                    signature: Some("provider-signature".to_string()),
                    details: Vec::new(),
                },
            ],
        });

        assert_eq!(
            first_user_message_hash(&text_only),
            first_user_message_hash(&text_with_reasoning)
        );
    }

    #[tokio::test]
    async fn metadata_session_takes_precedence_over_message_hash() -> Result<(), BoxErr> {
        let router = AffinityRouter::new().with_message_hash_fallback();
        let mut state = ();

        let mut first = task_request(
            Some(session("session-1", "agent-a")),
            "Implement the parser.",
            None,
        );
        retain(&router, &mut state, &mut first, "strong").await?;

        let mut other_session = task_request(
            Some(session("session-2", "agent-a")),
            "Implement the parser.",
            None,
        );
        assert!(
            scores(&router, &mut state, &mut other_session)
                .await?
                .is_empty()
        );
        Ok(())
    }

    #[tokio::test]
    async fn subagent_without_a_session_abstains_and_warns() -> Result<(), BoxErr> {
        for session_id in [None, Some(String::new())] {
            let router = AffinityRouter::new().with_message_hash_fallback();
            let mut state = ();
            let mut subagent = task_request(
                Some(Metadata {
                    session_id,
                    agent_id: Some("agent-1".to_string()),
                    is_subagent: true,
                    ..Metadata::default()
                }),
                "Implement the parser.",
                None,
            );

            retain(&router, &mut state, &mut subagent, "model-a").await?;
            assert!(scores(&router, &mut state, &mut subagent).await?.is_empty());
            assert!(
                !router.should_warn_unkeyed(),
                "an unidentifiable subagent should consume the warning"
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn one_router_serves_both_roles() -> Result<(), BoxErr> {
        // The same instance is registered under both SDK roles; a decision folded in via
        // the processor handle is read back via the classifier handle.
        let router = Arc::new(AffinityRouter::new());
        let processor: Arc<dyn Processor> = router.clone();
        let classifier: Arc<dyn Classifier> = router;
        let mut state = ();

        let mut first = request(session("session-1", "agent-a"));
        processor
            .process(
                &mut state,
                Event::Decision {
                    request: &mut first,
                    decision: &fixed_decision("model-a"),
                },
            )
            .await?;

        let mut second = request(session("session-1", "agent-b"));
        let scores = scores(classifier.as_ref(), &mut state, &mut second).await?;
        assert_eq!(
            scores.first().map(|score| score.target.as_str()),
            Some("model-a")
        );
        Ok(())
    }

    #[tokio::test]
    async fn decision_without_an_affinity_identity_is_ignored() -> Result<(), BoxErr> {
        let router = AffinityRouter::new();
        let mut state = ();
        let mut unkeyed = request(Metadata::default());

        router
            .process(
                &mut state,
                Event::Decision {
                    request: &mut unkeyed,
                    decision: &fixed_decision("model-a"),
                },
            )
            .await?;

        let mut req = request(session("session-1", "agent-a"));
        assert!(scores(&router, &mut state, &mut req).await?.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn decisions_retain_their_originating_request_identity() -> Result<(), BoxErr> {
        let router = AffinityRouter::new();
        let mut state = ();
        let mut first = request(session("session-1", "agent-a"));
        let mut second = request(session("session-2", "agent-b"));

        // Replay decisions in the opposite order. Each decision carries its request,
        // so the assignments cannot cross.
        router
            .process(
                &mut state,
                Event::Decision {
                    request: &mut second,
                    decision: &fixed_decision("model-b"),
                },
            )
            .await?;
        router
            .process(
                &mut state,
                Event::Decision {
                    request: &mut first,
                    decision: &fixed_decision("model-a"),
                },
            )
            .await?;

        let first_scores = scores(&router, &mut state, &mut first).await?;
        let second_scores = scores(&router, &mut state, &mut second).await?;
        assert_eq!(
            first_scores.first().map(|score| score.target.as_str()),
            Some("model-a")
        );
        assert_eq!(
            second_scores.first().map(|score| score.target.as_str()),
            Some("model-b")
        );
        Ok(())
    }

    #[tokio::test]
    async fn distinct_sessions_are_assigned_independently() -> Result<(), BoxErr> {
        let router = AffinityRouter::new();
        let mut state = ();

        retain(
            &router,
            &mut state,
            &mut request(session("session-1", "agent-a")),
            "model-a",
        )
        .await?;
        retain(
            &router,
            &mut state,
            &mut request(session("session-2", "agent-a")),
            "model-b",
        )
        .await?;

        let first = scores(
            &router,
            &mut state,
            &mut request(session("session-1", "other")),
        )
        .await?;
        let second = scores(
            &router,
            &mut state,
            &mut request(session("session-2", "other")),
        )
        .await?;
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
    async fn subagent_without_an_agent_id_abstains_and_warns() -> Result<(), BoxErr> {
        let router = AffinityRouter::new();
        let mut state = ();

        // The sub-agent flag is set but no agent id is present, so no key can be formed;
        // the request is neither retained nor scored.
        let metadata = Metadata {
            session_id: Some("session-1".to_string()),
            is_subagent: true,
            ..Metadata::default()
        };
        let mut req = request(metadata);
        retain(&router, &mut state, &mut req, "model-a").await?;
        assert!(scores(&router, &mut state, &mut req).await?.is_empty());
        assert!(
            !router.should_warn_unkeyed(),
            "an unidentifiable subagent should consume the warning"
        );
        Ok(())
    }

    #[tokio::test]
    async fn assignments_are_bounded_by_the_cap() -> Result<(), BoxErr> {
        let router = AffinityRouter::new();
        let mut state = ();

        // One distinct session past the cap forces exactly one eviction.
        for index in 0..=MAX_ASSIGNMENTS {
            let session_id = format!("session-{index}");
            retain(
                &router,
                &mut state,
                &mut request(session(&session_id, "agent-a")),
                "model-a",
            )
            .await?;
        }

        let len = router.assignments.lock().len();
        assert_eq!(len, MAX_ASSIGNMENTS);
        Ok(())
    }

    #[tokio::test]
    async fn latch_only_retains_matching_models() -> Result<(), BoxErr> {
        let router = AffinityRouter::new().with_latch_only(["strong"]);
        let mut state = ();
        let mut req = request(session("session-1", "agent-a"));

        // A "weak" decision is not retained — a later turn is not latched.
        retain(&router, &mut state, &mut req, "weak").await?;
        assert!(scores(&router, &mut state, &mut req).await?.is_empty());

        // A "strong" decision is retained — later turns latch onto it.
        retain(&router, &mut state, &mut req, "strong").await?;
        assert_eq!(
            scores(&router, &mut state, &mut req)
                .await?
                .first()
                .map(|s| s.target.as_str()),
            Some("strong")
        );
        Ok(())
    }
}
