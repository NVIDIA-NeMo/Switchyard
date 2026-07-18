// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Correlation metadata and harness header normalization.
//!
//! [`Metadata`] is the correlation/routing envelope carried alongside a request or
//! response. [`Metadata::from_headers`] normalizes the harness-specific HTTP headers
//! that Claude Code, Codex, NeMo Relay, and Dynamo attach into that neutral shape.

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::WireFormat;

/// Header carrying Codex's structured turn metadata as a JSON object.
const CODEX_TURN_METADATA_HEADER: &str = "x-codex-turn-metadata";

// Explicit Switchyard override headers; these take precedence over harness-native headers.
const SWITCHYARD_SESSION_ID_HEADER: &str = "x-switchyard-session-id";
const SWITCHYARD_AGENT_ID_HEADER: &str = "x-switchyard-agent-id";
const SWITCHYARD_PARENT_AGENT_ID_HEADER: &str = "x-switchyard-parent-agent-id";
const SWITCHYARD_IS_SUBAGENT_HEADER: &str = "x-switchyard-is-subagent";
const SWITCHYARD_AGENT_KIND_HEADER: &str = "x-switchyard-agent-kind";
const SWITCHYARD_AGENT_ROLE_HEADER: &str = "x-switchyard-agent-role";
const SWITCHYARD_TASK_ID_HEADER: &str = "x-switchyard-task-id";
const SWITCHYARD_TASK_KIND_HEADER: &str = "x-switchyard-task-kind";
const SWITCHYARD_TURN_ID_HEADER: &str = "x-switchyard-turn-id";

// NeMo Relay correlation headers.
const RELAY_SESSION_ID_HEADER: &str = "x-nemo-relay-session-id";
const RELAY_SUBAGENT_ID_HEADER: &str = "x-nemo-relay-subagent-id";

// Dynamo correlation headers.
const DYNAMO_SESSION_ID_HEADER: &str = "x-dynamo-session-id";
const DYNAMO_PARENT_SESSION_ID_HEADER: &str = "x-dynamo-parent-session-id";
const DYNAMO_SESSION_FINAL_HEADER: &str = "x-dynamo-session-final";

// Codex compatibility projection of its parent thread id.
const CODEX_PARENT_THREAD_ID_HEADER: &str = "x-codex-parent-thread-id";

// OpenAI subagent marker.
const OPENAI_SUBAGENT_HEADER: &str = "x-openai-subagent";

// Claude Code agent-lineage headers.
const CLAUDE_SESSION_ID_HEADER: &str = "x-claude-code-session-id";
const CLAUDE_AGENT_ID_HEADER: &str = "x-claude-code-agent-id";
const CLAUDE_PARENT_AGENT_ID_HEADER: &str = "x-claude-code-parent-agent-id";

// OpenCode session headers.
const OPENCODE_SESSION_ID_HEADER: &str = "x-session-id";
const OPENCODE_PARENT_SESSION_ID_HEADER: &str = "x-parent-session-id";

// Generic Codex-compatible correlation headers.
const SESSION_ID_HEADER: &str = "session-id";
const THREAD_ID_HEADER: &str = "thread-id";
const TASK_ID_HEADER: &str = "x-task-id";
const REQUEST_ID_HEADER: &str = "x-request-id";
const CLIENT_REQUEST_ID_HEADER: &str = "x-client-request-id";

/// Correlation and routing metadata attached to a request or response.
///
/// All fields are optional (or default-empty); algorithms and observers use whichever
/// are present (e.g. to key per-session state or emit correlated telemetry). The
/// agent-lineage fields (`parent_agent_id`, `is_subagent`, `agent_kind`, `agent_role`,
/// `task_kind`, `turn_id`, `session_final`) are populated for requests from a coding
/// agent. `extra_metadata` is a free-form escape hatch for host-specific keys.
#[derive(Clone, Default)]
pub struct Metadata {
    /// Stable id for a multi-request session/conversation.
    pub session_id: Option<String>,
    /// Id of the agent making the request.
    pub agent_id: Option<String>,
    /// Id of the parent agent, when this request comes from a child agent.
    pub parent_agent_id: Option<String>,
    /// Whether the harness identified this request as coming from a child agent.
    pub is_subagent: bool,
    /// Harness-defined kind of agent call, such as `collab_spawn` or `review`.
    pub agent_kind: Option<String>,
    /// Semantic agent role, such as `explorer`, `worker`, or `reviewer`.
    pub agent_role: Option<String>,
    /// Id of the task the request belongs to.
    pub task_id: Option<String>,
    /// Semantic task class supplied by the harness.
    pub task_kind: Option<String>,
    /// Id of the current agent turn.
    pub turn_id: Option<String>,
    /// Whether the harness signalled this is the session's final request (e.g. the
    /// host may evict per-session state). `None` when the harness said nothing.
    pub session_final: Option<bool>,
    /// External trace/request id for joining with the host's telemetry.
    pub correlation_id: Option<String>,
    /// Arbitrary host-defined key/value metadata.
    pub extra_metadata: Option<BTreeMap<String, String>>,
    /// HTTP headers to attach when forwarding the request/response, if any.
    pub http_headers: Option<BTreeMap<String, String>>,
    /// The wire format the request/response was originally encoded in, if known.
    pub wire_format: Option<WireFormat>,
}

impl Metadata {
    /// Normalizes harness-specific request headers into correlation metadata.
    ///
    /// Explicit `x-switchyard-*` headers win. NeMo Relay and Dynamo correlation
    /// headers are accepted without linking either runtime. Codex's structured turn
    /// metadata is preferred over its compatibility projections. Claude Code and
    /// OpenCode carry their agent lineage in native headers: a Claude Code request
    /// naming a distinct `x-claude-code-agent-id` under its session is treated as a
    /// child agent (its parent inferred to be the session when not stated). Subagent
    /// status is taken from an explicit `x-switchyard-is-subagent` header when
    /// present, and otherwise inferred from any parent/child lineage header.
    pub fn from_headers(headers: &BTreeMap<String, String>) -> Self {
        let headers = &normalize_headers(headers);
        let codex = header(headers, CODEX_TURN_METADATA_HEADER)
            .and_then(|value| serde_json::from_str::<CodexTurnMetadata>(value).ok())
            .unwrap_or_default();

        let switchyard_parent = header(headers, SWITCHYARD_PARENT_AGENT_ID_HEADER);
        let relay_subagent = header(headers, RELAY_SUBAGENT_ID_HEADER);
        let dynamo_parent = header(headers, DYNAMO_PARENT_SESSION_ID_HEADER);
        let codex_parent = header(headers, CODEX_PARENT_THREAD_ID_HEADER);
        let openai_subagent = header(headers, OPENAI_SUBAGENT_HEADER);

        // Claude Code names a session and, for a spawned child, a distinct agent id.
        // A child whose agent id differs from the session is a sub-agent; its parent is
        // the explicit parent-agent header, else the session id it was spawned under.
        let claude_session = header(headers, CLAUDE_SESSION_ID_HEADER);
        let claude_agent = header(headers, CLAUDE_AGENT_ID_HEADER);
        let claude_subagent = matches!(
            (&claude_agent, &claude_session),
            (Some(agent), Some(session)) if agent != session
        );
        let claude_parent = claude_subagent
            .then(|| header(headers, CLAUDE_PARENT_AGENT_ID_HEADER).or(claude_session))
            .flatten();

        // OpenCode carries a session id and an optional parent session; the parent
        // header is only meaningful alongside OpenCode's own session header.
        let opencode_session = header(headers, OPENCODE_SESSION_ID_HEADER);
        let opencode_parent = opencode_session
            .as_ref()
            .and_then(|_| header(headers, OPENCODE_PARENT_SESSION_ID_HEADER));

        let inferred_subagent = switchyard_parent.is_some()
            || relay_subagent.is_some()
            || dynamo_parent.is_some()
            || codex.parent_thread_id.is_some()
            || codex.subagent_kind.is_some()
            || codex_parent.is_some()
            || openai_subagent.is_some()
            || claude_subagent
            || opencode_parent.is_some();
        let is_subagent = header(headers, SWITCHYARD_IS_SUBAGENT_HEADER)
            .and_then(parse_bool)
            .unwrap_or(inferred_subagent);

        Metadata {
            session_id: first_some(&[
                header(headers, SWITCHYARD_SESSION_ID_HEADER),
                claude_session,
                header(headers, RELAY_SESSION_ID_HEADER),
                opencode_session,
                codex.session_id.as_deref(),
                header(headers, SESSION_ID_HEADER),
            ]),
            agent_id: first_some(&[
                header(headers, SWITCHYARD_AGENT_ID_HEADER),
                claude_agent,
                relay_subagent,
                header(headers, DYNAMO_SESSION_ID_HEADER),
                codex.thread_id.as_deref(),
                header(headers, THREAD_ID_HEADER),
            ]),
            parent_agent_id: first_some(&[
                switchyard_parent,
                dynamo_parent,
                codex.parent_thread_id.as_deref(),
                codex_parent,
                claude_parent,
                opencode_parent,
            ]),
            is_subagent,
            agent_kind: first_some(&[
                header(headers, SWITCHYARD_AGENT_KIND_HEADER),
                codex.subagent_kind.as_deref(),
                openai_subagent,
            ]),
            agent_role: first_some(&[
                header(headers, SWITCHYARD_AGENT_ROLE_HEADER),
                codex.agent_role.as_deref(),
            ]),
            task_id: first_some(&[
                header(headers, SWITCHYARD_TASK_ID_HEADER),
                codex.task_id.as_deref(),
                header(headers, TASK_ID_HEADER),
            ]),
            task_kind: first_some(&[
                header(headers, SWITCHYARD_TASK_KIND_HEADER),
                codex.task_kind.as_deref(),
            ]),
            turn_id: first_some(&[
                header(headers, SWITCHYARD_TURN_ID_HEADER),
                codex.turn_id.as_deref(),
            ]),
            session_final: header(headers, DYNAMO_SESSION_FINAL_HEADER).and_then(parse_bool),
            correlation_id: first_some(&[
                header(headers, REQUEST_ID_HEADER),
                header(headers, CLIENT_REQUEST_ID_HEADER),
            ]),
            ..Metadata::default()
        }
    }
}

/// Codex's structured turn metadata, carried as JSON in [`CODEX_TURN_METADATA_HEADER`].
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

/// Parses the common textual spellings of a boolean header value.
fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

/// Lowercases header names and keeps the first non-empty, trimmed value per name.
fn normalize_headers(headers: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    let mut normalized = BTreeMap::new();
    for (key, value) in headers {
        let lower_key = key.to_ascii_lowercase();
        let trimmed_value = value.trim();
        if !normalized.contains_key(&lower_key) && !trimmed_value.is_empty() {
            normalized.insert(lower_key, trimmed_value.to_string());
        }
    }

    normalized
}

fn header<'a>(headers: &'a BTreeMap<String, String>, key: &str) -> Option<&'a str> {
    let lower_key = key.to_ascii_lowercase();
    headers.get(&lower_key).map(|s| s.as_str())
}

fn first_some(options: &[Option<&str>]) -> Option<String> {
    options.iter().find_map(|opt| opt.map(|s| s.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_codex_child_metadata() {
        let headers = BTreeMap::from([(
            CODEX_TURN_METADATA_HEADER.to_string(),
            serde_json::json!({
                "session_id": "root-session",
                "thread_id": "child-agent",
                "parent_thread_id": "root-agent",
                "turn_id": "turn-7",
                "subagent_kind": "collab_spawn",
            })
            .to_string(),
        )]);

        let metadata = Metadata::from_headers(&headers);
        assert_eq!(metadata.session_id.as_deref(), Some("root-session"));
        assert_eq!(metadata.agent_id.as_deref(), Some("child-agent"));
        assert!(metadata.is_subagent);
        assert_eq!(metadata.parent_agent_id.as_deref(), Some("root-agent"));
    }

    #[test]
    fn root_codex_metadata_is_not_inferred_as_a_subagent() {
        let headers = BTreeMap::from([(
            CODEX_TURN_METADATA_HEADER.to_string(),
            serde_json::json!({
                "session_id": "root-session",
                "thread_id": "root-agent",
                "turn_id": "turn-1",
            })
            .to_string(),
        )]);

        let metadata = Metadata::from_headers(&headers);
        assert!(!metadata.is_subagent);
    }

    #[test]
    fn explicit_switchyard_subagent_flag_overrides_inference() {
        let headers = BTreeMap::from([
            ("x-switchyard-is-subagent".to_string(), "false".to_string()),
            (
                "x-switchyard-parent-agent-id".to_string(),
                "parent".to_string(),
            ),
        ]);

        let metadata = Metadata::from_headers(&headers);
        assert!(!metadata.is_subagent);
    }

    #[test]
    fn normalizes_claude_code_session_header() {
        // Claude Code identifies a session with `x-claude-code-session-id`; session
        // affinity keys on it so a whole CLI session pins to one tier.
        let headers = BTreeMap::from([(
            "x-claude-code-session-id".to_string(),
            "fb46caae-eac6-4f5f-83fd-8fc8f5743abb".to_string(),
        )]);

        let metadata = Metadata::from_headers(&headers);
        assert_eq!(
            metadata.session_id.as_deref(),
            Some("fb46caae-eac6-4f5f-83fd-8fc8f5743abb")
        );
    }

    #[test]
    fn normalizes_relay_and_dynamo_child_headers() {
        let headers = BTreeMap::from([
            (
                "x-nemo-relay-session-id".to_string(),
                "relay-session".to_string(),
            ),
            (
                "x-nemo-relay-subagent-id".to_string(),
                "relay-child".to_string(),
            ),
            (
                "x-dynamo-parent-session-id".to_string(),
                "relay-parent".to_string(),
            ),
        ]);

        let metadata = Metadata::from_headers(&headers);
        assert_eq!(metadata.session_id.as_deref(), Some("relay-session"));
        assert_eq!(metadata.agent_id.as_deref(), Some("relay-child"));
        assert!(metadata.is_subagent);
    }

    #[test]
    fn claude_code_agent_lineage_marks_subagent_and_infers_parent() {
        // A distinct agent id under a session is a child agent; with no explicit
        // parent header its parent is inferred to be the session it was spawned under.
        let metadata = Metadata::from_headers(&BTreeMap::from([
            (
                "x-claude-code-session-id".to_string(),
                "claude-session".to_string(),
            ),
            (
                "x-claude-code-agent-id".to_string(),
                "claude-agent".to_string(),
            ),
        ]));
        assert_eq!(metadata.session_id.as_deref(), Some("claude-session"));
        assert_eq!(metadata.agent_id.as_deref(), Some("claude-agent"));
        assert!(metadata.is_subagent);
        assert_eq!(metadata.parent_agent_id.as_deref(), Some("claude-session"));
    }

    #[test]
    fn explicit_claude_parent_agent_overrides_inferred_session() {
        let metadata = Metadata::from_headers(&BTreeMap::from([
            (
                "x-claude-code-session-id".to_string(),
                "claude-session".to_string(),
            ),
            (
                "x-claude-code-agent-id".to_string(),
                "claude-agent".to_string(),
            ),
            (
                "x-claude-code-parent-agent-id".to_string(),
                "claude-parent-agent".to_string(),
            ),
        ]));
        assert_eq!(
            metadata.parent_agent_id.as_deref(),
            Some("claude-parent-agent")
        );
    }

    #[test]
    fn claude_root_agent_without_distinct_child_is_not_a_subagent() {
        // Session but no distinct agent id: a root agent. A stray parent-agent header
        // is only meaningful for a distinct child, so it must not leak in.
        let metadata = Metadata::from_headers(&BTreeMap::from([
            (
                "x-claude-code-session-id".to_string(),
                "claude-session".to_string(),
            ),
            (
                "x-claude-code-parent-agent-id".to_string(),
                "claude-parent-agent".to_string(),
            ),
        ]));
        assert_eq!(metadata.session_id.as_deref(), Some("claude-session"));
        assert_eq!(metadata.agent_id, None);
        assert!(!metadata.is_subagent);
        assert_eq!(metadata.parent_agent_id, None);
    }

    #[test]
    fn opencode_parent_session_marks_subagent() {
        let metadata = Metadata::from_headers(&BTreeMap::from([
            ("x-session-id".to_string(), "opencode-run".to_string()),
            (
                "x-parent-session-id".to_string(),
                "opencode-parent".to_string(),
            ),
        ]));
        assert_eq!(metadata.session_id.as_deref(), Some("opencode-run"));
        assert!(metadata.is_subagent);
        assert_eq!(metadata.parent_agent_id.as_deref(), Some("opencode-parent"));
    }

    #[test]
    fn opencode_parent_ignored_without_opencode_session() {
        // The OpenCode parent header only applies with OpenCode's own session header;
        // next to a Codex `session-id` it must not surface as a parent.
        let metadata = Metadata::from_headers(&BTreeMap::from([
            ("session-id".to_string(), "codex-run".to_string()),
            (
                "x-parent-session-id".to_string(),
                "stray-parent".to_string(),
            ),
        ]));
        assert_eq!(metadata.session_id.as_deref(), Some("codex-run"));
        assert!(!metadata.is_subagent);
        assert_eq!(metadata.parent_agent_id, None);
    }

    #[test]
    fn dynamo_session_final_is_captured() {
        let metadata = Metadata::from_headers(&BTreeMap::from([
            ("x-dynamo-session-id".to_string(), "generic-run".to_string()),
            (
                "x-dynamo-parent-session-id".to_string(),
                "generic-parent".to_string(),
            ),
            ("x-dynamo-session-final".to_string(), "true".to_string()),
        ]));
        assert_eq!(metadata.agent_id.as_deref(), Some("generic-run"));
        assert_eq!(metadata.parent_agent_id.as_deref(), Some("generic-parent"));
        assert_eq!(metadata.session_final, Some(true));

        let not_final = Metadata::from_headers(&BTreeMap::from([
            ("x-dynamo-session-id".to_string(), "generic-run".to_string()),
            ("x-dynamo-session-final".to_string(), "false".to_string()),
        ]));
        assert_eq!(not_final.session_final, Some(false));
    }

    #[test]
    fn codex_session_header_is_case_insensitive() {
        let metadata = Metadata::from_headers(&BTreeMap::from([(
            "Session-ID".to_string(),
            "codex-run".to_string(),
        )]));
        assert_eq!(metadata.session_id.as_deref(), Some("codex-run"));
    }
}
