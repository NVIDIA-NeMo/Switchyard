// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use std::sync::Arc;

use switchyard_protocol::{ContentBlock, Decision, LlmRequest, Message, Metadata, text_request};

/// Boxed, thread-safe error type keeping the test helpers ergonomic.
type BoxErr = Box<dyn std::error::Error + Send + Sync>;

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

fn task_request(metadata: Option<Metadata>, first_user: &str, follow_up: Option<&str>) -> Request {
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
    router
        .process(
            state,
            Event::Decision {
                request,
                decision: &FixedDecision(model),
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
async fn message_hash_fallback_abstains_for_subagents() -> Result<(), BoxErr> {
    let router = AffinityRouter::new().with_message_hash_fallback();
    let mut state = ();
    let mut subagent = task_request(
        Some(Metadata {
            is_subagent: true,
            ..Metadata::default()
        }),
        "Implement the parser.",
        None,
    );

    assert!(scores(&router, &mut state, &mut subagent).await?.is_empty());
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
                decision: &FixedDecision("model-a"),
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
                decision: &FixedDecision("model-a"),
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
                decision: &FixedDecision("model-b"),
            },
        )
        .await?;
    router
        .process(
            &mut state,
            Event::Decision {
                request: &mut first,
                decision: &FixedDecision("model-a"),
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
async fn subagent_without_an_agent_id_is_not_keyed() -> Result<(), BoxErr> {
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
