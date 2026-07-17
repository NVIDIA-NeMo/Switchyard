// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reusable model-affinity policies and harness metadata normalization.
//!
//! Algorithms remain responsible for choosing a model. An [`Affinity`] policy only
//! decides whether that choice should be retained and which stable request identity
//! owns the assignment.

use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;

use serde::Deserialize;
use switchyard_protocol::{AgentContext, Metadata, Request};

const CODEX_TURN_METADATA_HEADER: &str = "x-codex-turn-metadata";
const MAX_ASSIGNMENTS: usize = 4096;

/// Namespaced stable identity used to retain a model assignment.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct AffinityKey {
    namespace: String,
    components: Vec<String>,
}

impl AffinityKey {
    /// Creates a key whose namespace prevents collisions with other affinity policies.
    pub fn new(namespace: impl Into<String>, components: Vec<String>) -> Self {
        Self {
            namespace: namespace.into(),
            components,
        }
    }
}

/// Bounded, process-local storage shared by an affinity policy.
#[derive(Default)]
pub struct AffinityState {
    assignments: Mutex<HashMap<AffinityKey, String>>,
}

impl AffinityState {
    /// Returns the retained model for `key`, failing open if the state lock is poisoned.
    fn get(&self, key: &AffinityKey) -> Option<String> {
        self.assignments
            .lock()
            .ok()
            .and_then(|assignments| assignments.get(key).cloned())
    }

    /// Atomically retains the first model assigned to `key` and returns that model.
    fn get_or_insert(&self, key: AffinityKey, proposed: String) -> String {
        let Ok(mut assignments) = self.assignments.lock() else {
            return proposed;
        };
        if let Some(assigned) = assignments.get(&key) {
            return assigned.clone();
        }
        if assignments.len() >= MAX_ASSIGNMENTS {
            if let Some(evicted) = assignments.keys().next().cloned() {
                assignments.remove(&evicted);
            }
        }
        assignments.insert(key, proposed.clone());
        proposed
    }
}

/// Policy that maps eligible requests to stable assignment keys.
///
/// Algorithms consume this trait at the point where their final model is known:
/// check [`assignment`](Self::assignment), run normal selection on a miss, then call
/// [`retain`](Self::retain) before routing the request.
pub trait Affinity: Send + Sync {
    /// Returns the stable key for an eligible request, or `None` when affinity does
    /// not apply.
    fn key(&self, request: &Request) -> Option<AffinityKey>;

    /// Returns this policy's process-local assignment state.
    fn state(&self) -> &AffinityState;

    /// Returns an existing model assignment for `request`.
    fn assignment(&self, request: &Request) -> Option<String> {
        self.key(request).and_then(|key| self.state().get(&key))
    }

    /// Retains `proposed` for an eligible request and returns the canonical model.
    ///
    /// When concurrent requests propose different models, the first insertion wins.
    /// Ineligible requests simply receive `proposed` without storing it.
    fn retain(&self, request: &Request, proposed: String) -> String {
        let Some(key) = self.key(request) else {
            return proposed;
        };
        self.state().get_or_insert(key, proposed)
    }
}

/// Retains one model for every request carrying the same session id.
#[derive(Default)]
pub struct SessionAffinity {
    state: AffinityState,
}

impl SessionAffinity {
    /// Creates an empty process-local session-affinity policy.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Affinity for SessionAffinity {
    fn key(&self, request: &Request) -> Option<AffinityKey> {
        Some(AffinityKey::new(
            "session",
            vec![request.metadata.as_ref()?.session_id.clone()?],
        ))
    }

    fn state(&self) -> &AffinityState {
        &self.state
    }
}

/// Retains one model for each explicitly identified child agent in a session.
#[derive(Default)]
pub struct SubAgentAffinity {
    state: AffinityState,
}

impl SubAgentAffinity {
    /// Creates an empty process-local child-agent-affinity policy.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Affinity for SubAgentAffinity {
    fn key(&self, request: &Request) -> Option<AffinityKey> {
        let metadata = request.metadata.as_ref()?;
        if !metadata.agent_context.as_deref()?.is_subagent {
            return None;
        }
        Some(AffinityKey::new(
            "subagent",
            vec![metadata.session_id.clone()?, metadata.agent_id.clone()?],
        ))
    }

    fn state(&self) -> &AffinityState {
        &self.state
    }
}

#[derive(Default, Deserialize)]
struct CodexTurnMetadata {
    session_id: Option<String>,
    thread_id: Option<String>,
    parent_thread_id: Option<String>,
    turn_id: Option<String>,
    subagent_kind: Option<String>,
    agent_role: Option<String>,
    task_id: Option<String>,
    task_kind: Option<String>,
}

/// Normalizes harness-specific request headers into libsy metadata.
///
/// Explicit `x-switchyard-*` headers win. Claude Code, NeMo Relay, and Dynamo
/// correlation headers are accepted without linking their runtimes. Codex's structured
/// turn metadata is preferred over its compatibility projections.
pub fn metadata_from_headers(headers: &BTreeMap<String, Vec<String>>) -> Metadata {
    let codex = header(headers, CODEX_TURN_METADATA_HEADER)
        .and_then(|value| serde_json::from_str::<CodexTurnMetadata>(&value).ok())
        .unwrap_or_default();

    let switchyard_parent = header(headers, "x-switchyard-parent-agent-id");
    let claude_agent = header(headers, "x-claude-code-agent-id");
    let claude_parent = header(headers, "x-claude-code-parent-agent-id");
    let relay_subagent = header(headers, "x-nemo-relay-subagent-id");
    let dynamo_parent = header(headers, "x-dynamo-parent-session-id");
    let codex_parent = header(headers, "x-codex-parent-thread-id");
    let openai_subagent = header(headers, "x-openai-subagent");
    let inferred_subagent = switchyard_parent.is_some()
        || claude_agent.is_some()
        || claude_parent.is_some()
        || relay_subagent.is_some()
        || dynamo_parent.is_some()
        || codex.parent_thread_id.is_some()
        || codex.subagent_kind.is_some()
        || codex_parent.is_some()
        || openai_subagent.is_some();
    let is_subagent = header(headers, "x-switchyard-is-subagent")
        .as_deref()
        .and_then(parse_bool)
        .unwrap_or(inferred_subagent);

    let agent_context = AgentContext {
        is_subagent,
        parent_agent_id: first_some([
            switchyard_parent,
            claude_parent,
            dynamo_parent,
            codex.parent_thread_id,
            codex_parent,
        ]),
        agent_kind: first_some([
            header(headers, "x-switchyard-agent-kind"),
            codex.subagent_kind,
            openai_subagent,
        ]),
        agent_role: first_some([header(headers, "x-switchyard-agent-role"), codex.agent_role]),
        task_kind: first_some([header(headers, "x-switchyard-task-kind"), codex.task_kind]),
        turn_id: first_some([header(headers, "x-switchyard-turn-id"), codex.turn_id]),
    };
    let agent_context = (agent_context != AgentContext::default()).then(|| Box::new(agent_context));

    Metadata {
        session_id: first_some([
            header(headers, "x-switchyard-session-id"),
            header(headers, "x-claude-code-session-id"),
            header(headers, "x-nemo-relay-session-id"),
            codex.session_id,
            header(headers, "session-id"),
        ]),
        agent_id: first_some([
            header(headers, "x-switchyard-agent-id"),
            claude_agent,
            relay_subagent,
            header(headers, "x-dynamo-session-id"),
            codex.thread_id,
            header(headers, "thread-id"),
        ]),
        task_id: first_some([
            header(headers, "x-switchyard-task-id"),
            codex.task_id,
            header(headers, "x-task-id"),
        ]),
        agent_context,
        correlation_id: first_some([
            header(headers, "x-request-id"),
            header(headers, "x-client-request-id"),
        ]),
        ..Metadata::default()
    }
}

fn header(headers: &BTreeMap<String, Vec<String>>, name: &str) -> Option<String> {
    headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .and_then(|(_, values)| values.iter().find(|value| !value.trim().is_empty()))
        .map(|value| value.trim().to_string())
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn first_some<const N: usize>(values: [Option<String>; N]) -> Option<String> {
    values.into_iter().flatten().next()
}

#[cfg(test)]
mod tests {
    use super::*;
    use switchyard_protocol::{text_request, AgentContext};

    fn request(metadata: Metadata) -> Request {
        Request {
            llm_request: text_request(Some("auto".to_string()), "hi"),
            raw_request: None,
            metadata: Some(metadata),
        }
    }

    fn child_request(task_id: &str) -> Request {
        request(Metadata {
            session_id: Some("session-1".to_string()),
            agent_id: Some("child-1".to_string()),
            task_id: Some(task_id.to_string()),
            agent_context: Some(Box::new(AgentContext {
                is_subagent: true,
                ..AgentContext::default()
            })),
            ..Metadata::default()
        })
    }

    #[test]
    fn session_affinity_keys_all_requests_by_session() {
        let affinity = SessionAffinity::new();
        let first = request(Metadata {
            session_id: Some("session-1".to_string()),
            agent_id: Some("agent-a".to_string()),
            ..Metadata::default()
        });
        let second = request(Metadata {
            session_id: Some("session-1".to_string()),
            agent_id: Some("agent-b".to_string()),
            ..Metadata::default()
        });

        assert_eq!(affinity.retain(&first, "model-a".to_string()), "model-a");
        assert_eq!(affinity.assignment(&second).as_deref(), Some("model-a"));
    }

    #[test]
    fn subagent_affinity_ignores_task_changes() {
        let affinity = SubAgentAffinity::new();
        assert_eq!(
            affinity.retain(&child_request("task-1"), "model-a".to_string()),
            "model-a"
        );
        assert_eq!(
            affinity.assignment(&child_request("task-2")).as_deref(),
            Some("model-a")
        );
    }

    #[test]
    fn subagent_affinity_does_not_apply_to_root_agents() {
        let affinity = SubAgentAffinity::new();
        let root = request(Metadata {
            session_id: Some("session-1".to_string()),
            agent_id: Some("root-1".to_string()),
            agent_context: Some(Box::new(AgentContext::default())),
            ..Metadata::default()
        });

        assert_eq!(affinity.retain(&root, "model-a".to_string()), "model-a");
        assert!(affinity.assignment(&root).is_none());
    }

    #[test]
    fn normalizes_codex_child_metadata() {
        let headers = BTreeMap::from([(
            CODEX_TURN_METADATA_HEADER.to_string(),
            vec![serde_json::json!({
                "session_id": "root-session",
                "thread_id": "child-agent",
                "parent_thread_id": "root-agent",
                "turn_id": "turn-7",
                "subagent_kind": "collab_spawn",
            })
            .to_string()],
        )]);

        let metadata = metadata_from_headers(&headers);
        assert_eq!(metadata.session_id.as_deref(), Some("root-session"));
        assert_eq!(metadata.agent_id.as_deref(), Some("child-agent"));
        let agent = metadata.agent_context.as_deref();
        assert_eq!(agent.map(|value| value.is_subagent), Some(true));
        assert_eq!(
            agent.and_then(|value| value.parent_agent_id.as_deref()),
            Some("root-agent")
        );
    }

    #[test]
    fn root_codex_metadata_is_not_inferred_as_a_subagent() {
        let headers = BTreeMap::from([(
            CODEX_TURN_METADATA_HEADER.to_string(),
            vec![serde_json::json!({
                "session_id": "root-session",
                "thread_id": "root-agent",
                "turn_id": "turn-1",
            })
            .to_string()],
        )]);

        let metadata = metadata_from_headers(&headers);
        assert_eq!(
            metadata
                .agent_context
                .as_deref()
                .map(|agent| agent.is_subagent),
            Some(false)
        );
    }

    #[test]
    fn explicit_switchyard_subagent_flag_overrides_inference() {
        let headers = BTreeMap::from([
            (
                "x-switchyard-is-subagent".to_string(),
                vec!["false".to_string()],
            ),
            (
                "x-switchyard-parent-agent-id".to_string(),
                vec!["parent".to_string()],
            ),
        ]);

        let metadata = metadata_from_headers(&headers);
        assert_eq!(
            metadata
                .agent_context
                .as_deref()
                .map(|agent| agent.is_subagent),
            Some(false)
        );
    }

    #[test]
    fn normalizes_claude_code_session_header() {
        // Claude Code identifies a session with `x-claude-code-session-id`; session
        // affinity keys on it so a whole CLI session pins to one tier.
        let headers = BTreeMap::from([(
            "x-claude-code-session-id".to_string(),
            vec!["fb46caae-eac6-4f5f-83fd-8fc8f5743abb".to_string()],
        )]);

        let metadata = metadata_from_headers(&headers);
        assert_eq!(
            metadata.session_id.as_deref(),
            Some("fb46caae-eac6-4f5f-83fd-8fc8f5743abb")
        );
    }

    #[test]
    fn normalizes_claude_code_subagent_headers() {
        let headers = BTreeMap::from([
            (
                "x-claude-code-session-id".to_string(),
                vec!["root-session".to_string()],
            ),
            (
                "x-claude-code-agent-id".to_string(),
                vec!["child-agent".to_string()],
            ),
            (
                "x-claude-code-parent-agent-id".to_string(),
                vec!["root-agent".to_string()],
            ),
        ]);

        let metadata = metadata_from_headers(&headers);
        assert_eq!(metadata.session_id.as_deref(), Some("root-session"));
        assert_eq!(metadata.agent_id.as_deref(), Some("child-agent"));
        let agent = metadata.agent_context.as_deref();
        assert_eq!(agent.map(|value| value.is_subagent), Some(true));
        assert_eq!(
            agent.and_then(|value| value.parent_agent_id.as_deref()),
            Some("root-agent")
        );
    }

    #[test]
    fn normalizes_relay_and_dynamo_child_headers() {
        let headers = BTreeMap::from([
            (
                "x-nemo-relay-session-id".to_string(),
                vec!["relay-session".to_string()],
            ),
            (
                "x-nemo-relay-subagent-id".to_string(),
                vec!["relay-child".to_string()],
            ),
            (
                "x-dynamo-parent-session-id".to_string(),
                vec!["relay-parent".to_string()],
            ),
        ]);

        let metadata = metadata_from_headers(&headers);
        assert_eq!(metadata.session_id.as_deref(), Some("relay-session"));
        assert_eq!(metadata.agent_id.as_deref(), Some("relay-child"));
        assert_eq!(
            metadata
                .agent_context
                .as_deref()
                .map(|agent| agent.is_subagent),
            Some(true)
        );
    }
}
