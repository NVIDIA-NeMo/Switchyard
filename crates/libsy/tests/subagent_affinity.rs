// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for composing the sub-agent override with affinity routing.
//!
//! The two are independent classifiers in one [`FallThrough`] cascade: the override decides
//! *which* target delegated work belongs on, affinity decides *how long* that decision
//! lives. These tests drive them through the crate's public API, so they also pin that the
//! pieces needed to compose a cascade are actually exported.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;

use switchyard_libsy::algorithms::{AffinityRouter, FallThrough, SubagentOverride};
use switchyard_libsy::{
    Algorithm, Classification, Classifier, Context, Decision, Driver, LlmResponse, LlmTarget,
    LlmTargetSet, Metadata, Request, Response, Result, RoutedLlmClient, Score, SharedState, State,
};
use switchyard_protocol::{completion_text, text_request, text_response};

/// A client that echoes the routed target name back as the completion.
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
            llm_response: LlmResponse::Agg(text_response(None, decision.selected_model())),
            metadata: None,
        })
    }
}

/// The cascade's terminal classifier: always picks the orchestrator.
struct AlwaysOrchestrator;

#[async_trait]
impl Classifier for AlwaysOrchestrator {
    async fn score(
        &self,
        _state: &mut State,
        _request: &mut Request,
        _driver: Option<&Driver>,
    ) -> Result<Classification> {
        Ok(Classification::Scores(vec![Score {
            confidence: 0.5,
            target: "orchestrator".to_string(),
        }]))
    }
}

fn targets() -> LlmTargetSet {
    LlmTargetSet::new(
        ["orchestrator", "worker", "reviewer"]
            .iter()
            .map(|name| LlmTarget {
                semantic_name: (*name).to_string(),
                llm_client: Some(Arc::new(EchoClient) as Arc<dyn RoutedLlmClient>),
            })
            .collect(),
    )
}

fn request(headers: &[(&str, &str)]) -> Request {
    Request {
        llm_request: text_request(Some("auto".to_string()), "hi"),
        raw_request: None,
        metadata: Some(Metadata::from_headers(
            &headers
                .iter()
                .map(|(name, value)| ((*name).to_string(), (*value).to_string()))
                .collect::<BTreeMap<_, _>>(),
        )),
    }
}

/// Affinity replays an existing pin, the override seeds one for delegated work, and the
/// terminal classifier serves everything else.
fn router() -> Arc<FallThrough> {
    Arc::new(
        FallThrough::new(targets())
            .with_component(Arc::new(AffinityRouter::for_subagents()))
            .with_classifier(Arc::new(SubagentOverride::new("worker")))
            .with_classifier(Arc::new(AlwaysOrchestrator)),
    )
}

/// Runs one turn on `ctx`, returning the target that served it.
///
/// `ctx` carries the per-session [`State`] the processor chain folds into. The affinity
/// assignments are held on the `AffinityRouter` instance instead, so replay follows the
/// shared router rather than the context.
async fn turn(
    router: &Arc<FallThrough>,
    ctx: Context<SharedState>,
    headers: &[(&str, &str)],
) -> Result<String> {
    let (_, response) = router.clone().run(ctx, request(headers)).await?;
    Ok(response
        .llm_response
        .as_agg()
        .map(completion_text)
        .unwrap_or_default())
}

/// A Claude Code child agent's lineage headers.
fn child(agent: &str) -> Vec<(&str, &str)> {
    vec![
        ("x-claude-code-session-id", "session-1"),
        ("x-claude-code-agent-id", agent),
    ]
}

#[tokio::test]
async fn root_traffic_falls_through_to_the_terminal_classifier() -> Result<()> {
    let ctx = Context::<SharedState>::default();
    // No sub-agent lineage: affinity abstains (subagents only), the override abstains, and
    // the terminal classifier serves the turn.
    let served = turn(&router(), ctx, &[("x-claude-code-session-id", "session-1")]).await?;
    assert_eq!(served, "orchestrator");
    Ok(())
}

#[tokio::test]
async fn delegated_work_is_routed_to_the_worker() -> Result<()> {
    let ctx = Context::<SharedState>::default();
    assert_eq!(turn(&router(), ctx, &child("child-1")).await?, "worker");
    Ok(())
}

#[tokio::test]
async fn the_override_seeds_a_pin_that_affinity_replays() -> Result<()> {
    let router = router();
    let ctx = Context::<SharedState>::default();

    // Turn 1: affinity has no assignment, so the override decides and the decision is
    // replayed into affinity.
    assert_eq!(
        turn(&router, ctx.clone(), &child("child-1")).await?,
        "worker"
    );

    // Turns 2-3: the lineage header still identifies the child, but affinity — not the
    // override — is now the classifier that answers, because it scores first.
    assert_eq!(
        turn(&router, ctx.clone(), &child("child-1")).await?,
        "worker"
    );
    assert_eq!(turn(&router, ctx, &child("child-1")).await?, "worker");
    Ok(())
}

#[tokio::test]
async fn harness_maintenance_turns_are_not_forced_to_the_worker() -> Result<()> {
    let router = router();
    let ctx = Context::<SharedState>::default();

    // A Codex `compact` turn is sub-agent *lineage* but not delegated *work*: the two
    // predicates differ on purpose, so the override abstains and it routes normally.
    let served = turn(
        &router,
        ctx,
        &[
            ("x-codex-session-id", "session-1"),
            ("x-openai-subagent", "compact"),
        ],
    )
    .await?;
    assert_eq!(served, "orchestrator");
    Ok(())
}

/// Builds a cascade whose override scores `worker`, sharing `affinity` across instances.
fn router_overriding_to(affinity: Arc<AffinityRouter>, worker: &str) -> Arc<FallThrough> {
    Arc::new(
        FallThrough::new(targets())
            .with_component(affinity)
            .with_classifier(Arc::new(SubagentOverride::new(worker)))
            .with_classifier(Arc::new(AlwaysOrchestrator)),
    )
}

#[tokio::test]
async fn the_pin_outlives_the_policy_that_seeded_it() -> Result<()> {
    // Two cascades sharing one affinity instance, with overrides that disagree. The first
    // seeds "worker"; the second would score "reviewer" but is never consulted, because
    // affinity scores earlier in the cascade and replays the existing pin. This is what
    // distinguishes a real replay from the override simply repeating itself.
    let affinity = Arc::new(AffinityRouter::for_subagents());
    let seed = router_overriding_to(affinity.clone(), "worker");
    let rebound = router_overriding_to(affinity, "reviewer");

    assert_eq!(
        turn(&seed, Context::default(), &child("child-1")).await?,
        "worker"
    );
    assert_eq!(
        turn(&rebound, Context::default(), &child("child-1")).await?,
        "worker"
    );
    Ok(())
}

#[tokio::test]
async fn without_a_shared_pin_the_second_policy_wins() -> Result<()> {
    // The negative control for the test above: with independent affinity instances there
    // is no pin to replay, so the second cascade's own override decides. Without this,
    // the replay assertion could pass for the wrong reason.
    let seed = router_overriding_to(Arc::new(AffinityRouter::for_subagents()), "worker");
    let rebound = router_overriding_to(Arc::new(AffinityRouter::for_subagents()), "reviewer");

    assert_eq!(
        turn(&seed, Context::default(), &child("child-1")).await?,
        "worker"
    );
    assert_eq!(
        turn(&rebound, Context::default(), &child("child-1")).await?,
        "reviewer"
    );
    Ok(())
}

#[tokio::test]
async fn distinct_children_are_pinned_independently() -> Result<()> {
    // Two cascades sharing one affinity instance but disagreeing on the worker. Routing
    // both children through the *same* override would pass whether or not affinity keys
    // per agent, so the sibling is sent through the cascade that scores "reviewer": it
    // can only come back "reviewer" if it did not inherit child-1's pin.
    let affinity = Arc::new(AffinityRouter::for_subagents());
    let seed = router_overriding_to(affinity.clone(), "worker");
    let sibling = router_overriding_to(affinity, "reviewer");

    assert_eq!(
        turn(&seed, Context::default(), &child("child-1")).await?,
        "worker"
    );
    assert_eq!(
        turn(&sibling, Context::default(), &child("child-2")).await?,
        "reviewer"
    );
    // child-1's own pin is untouched by the sibling's assignment: affinity replays it
    // even through the cascade whose override would otherwise score "reviewer".
    assert_eq!(
        turn(&sibling, Context::default(), &child("child-1")).await?,
        "worker"
    );
    Ok(())
}

#[tokio::test]
async fn a_cascade_without_the_override_still_routes_root_traffic() -> Result<()> {
    // Affinity and the override are independent: dropping the override leaves a valid
    // cascade, which is the point of composing them rather than nesting one in the other.
    let router = Arc::new(
        FallThrough::new(targets())
            .with_component(Arc::new(AffinityRouter::for_subagents()))
            .with_classifier(Arc::new(AlwaysOrchestrator)),
    );
    let ctx = Context::<SharedState>::default();
    assert_eq!(turn(&router, ctx, &child("child-1")).await?, "orchestrator");
    Ok(())
}
