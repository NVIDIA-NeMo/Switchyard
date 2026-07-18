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
    pub fn from_headers(headers: &BTreeMap<String, Vec<String>>) -> Self {
        let codex = header(headers, CODEX_TURN_METADATA_HEADER)
            .and_then(|value| serde_json::from_str::<CodexTurnMetadata>(&value).ok())
            .unwrap_or_default();

        let switchyard_parent = header(headers, "x-switchyard-parent-agent-id");
        let relay_subagent = header(headers, "x-nemo-relay-subagent-id");
        let dynamo_parent = header(headers, "x-dynamo-parent-session-id");
        let codex_parent = header(headers, "x-codex-parent-thread-id");
        let openai_subagent = header(headers, "x-openai-subagent");

        // Claude Code names a session and, for a spawned child, a distinct agent id.
        // A child whose agent id differs from the session is a sub-agent; its parent is
        // the explicit parent-agent header, else the session id it was spawned under.
        let claude_session = header(headers, "x-claude-code-session-id");
        let claude_agent = header(headers, "x-claude-code-agent-id");
        let claude_subagent = matches!(
            (&claude_agent, &claude_session),
            (Some(agent), Some(session)) if agent != session
        );
        let claude_parent = claude_subagent
            .then(|| {
                header(headers, "x-claude-code-parent-agent-id").or_else(|| claude_session.clone())
            })
            .flatten();

        // OpenCode carries a session id and an optional parent session; the parent
        // header is only meaningful alongside OpenCode's own session header.
        let opencode_session = header(headers, "x-session-id");
        let opencode_parent = opencode_session
            .as_ref()
            .and_then(|_| header(headers, "x-parent-session-id"));

        let inferred_subagent = switchyard_parent.is_some()
            || relay_subagent.is_some()
            || dynamo_parent.is_some()
            || codex.parent_thread_id.is_some()
            || codex.subagent_kind.is_some()
            || codex_parent.is_some()
            || openai_subagent.is_some()
            || claude_subagent
            || opencode_parent.is_some();
        let is_subagent = header(headers, "x-switchyard-is-subagent")
            .as_deref()
            .and_then(parse_bool)
            .unwrap_or(inferred_subagent);

        Metadata {
            session_id: first_some([
                header(headers, "x-switchyard-session-id"),
                claude_session,
                header(headers, "x-nemo-relay-session-id"),
                opencode_session,
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
            parent_agent_id: first_some([
                switchyard_parent,
                dynamo_parent,
                codex.parent_thread_id,
                codex_parent,
                claude_parent,
                opencode_parent,
            ]),
            is_subagent,
            agent_kind: first_some([
                header(headers, "x-switchyard-agent-kind"),
                codex.subagent_kind,
                openai_subagent,
            ]),
            agent_role: first_some([header(headers, "x-switchyard-agent-role"), codex.agent_role]),
            task_id: first_some([
                header(headers, "x-switchyard-task-id"),
                codex.task_id,
                header(headers, "x-task-id"),
            ]),
            task_kind: first_some([header(headers, "x-switchyard-task-kind"), codex.task_kind]),
            turn_id: first_some([header(headers, "x-switchyard-turn-id"), codex.turn_id]),
            session_final: header(headers, "x-dynamo-session-final")
                .as_deref()
                .and_then(parse_bool),
            correlation_id: first_some([
                header(headers, "x-request-id"),
                header(headers, "x-client-request-id"),
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

/// Returns the first non-empty, trimmed value for a case-insensitive header name.
fn header(headers: &BTreeMap<String, Vec<String>>, name: &str) -> Option<String> {
    headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .and_then(|(_, values)| values.iter().find(|value| !value.trim().is_empty()))
        .map(|value| value.trim().to_string())
}

/// Parses the common textual spellings of a boolean header value.
fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

/// Returns the first present value in preference order.
fn first_some<const N: usize>(values: [Option<String>; N]) -> Option<String> {
    values.into_iter().flatten().next()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a single-valued header map from `(name, value)` pairs.
    fn headers(pairs: &[(&str, &str)]) -> BTreeMap<String, Vec<String>> {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), vec![value.to_string()]))
            .collect()
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
            vec![serde_json::json!({
                "session_id": "root-session",
                "thread_id": "root-agent",
                "turn_id": "turn-1",
            })
            .to_string()],
        )]);

        let metadata = Metadata::from_headers(&headers);
        assert!(!metadata.is_subagent);
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

        let metadata = Metadata::from_headers(&headers);
        assert!(!metadata.is_subagent);
    }

    #[test]
    fn normalizes_claude_code_session_header() {
        // Claude Code identifies a session with `x-claude-code-session-id`; session
        // affinity keys on it so a whole CLI session pins to one tier.
        let headers = BTreeMap::from([(
            "x-claude-code-session-id".to_string(),
            vec!["fb46caae-eac6-4f5f-83fd-8fc8f5743abb".to_string()],
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

        let metadata = Metadata::from_headers(&headers);
        assert_eq!(metadata.session_id.as_deref(), Some("relay-session"));
        assert_eq!(metadata.agent_id.as_deref(), Some("relay-child"));
        assert!(metadata.is_subagent);
    }

    #[test]
    fn claude_code_agent_lineage_marks_subagent_and_infers_parent() {
        // A distinct agent id under a session is a child agent; with no explicit
        // parent header its parent is inferred to be the session it was spawned under.
        let metadata = Metadata::from_headers(&headers(&[
            ("x-claude-code-session-id", "claude-session"),
            ("x-claude-code-agent-id", "claude-agent"),
        ]));
        assert_eq!(metadata.session_id.as_deref(), Some("claude-session"));
        assert_eq!(metadata.agent_id.as_deref(), Some("claude-agent"));
        assert!(metadata.is_subagent);
        assert_eq!(metadata.parent_agent_id.as_deref(), Some("claude-session"));
    }

    #[test]
    fn explicit_claude_parent_agent_overrides_inferred_session() {
        let metadata = Metadata::from_headers(&headers(&[
            ("x-claude-code-session-id", "claude-session"),
            ("x-claude-code-agent-id", "claude-agent"),
            ("x-claude-code-parent-agent-id", "claude-parent-agent"),
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
        let metadata = Metadata::from_headers(&headers(&[
            ("x-claude-code-session-id", "claude-session"),
            ("x-claude-code-parent-agent-id", "claude-parent-agent"),
        ]));
        assert_eq!(metadata.session_id.as_deref(), Some("claude-session"));
        assert_eq!(metadata.agent_id, None);
        assert!(!metadata.is_subagent);
        assert_eq!(metadata.parent_agent_id, None);
    }

    #[test]
    fn opencode_parent_session_marks_subagent() {
        let metadata = Metadata::from_headers(&headers(&[
            ("x-session-id", "opencode-run"),
            ("x-parent-session-id", "opencode-parent"),
        ]));
        assert_eq!(metadata.session_id.as_deref(), Some("opencode-run"));
        assert!(metadata.is_subagent);
        assert_eq!(metadata.parent_agent_id.as_deref(), Some("opencode-parent"));
    }

    #[test]
    fn opencode_parent_ignored_without_opencode_session() {
        // The OpenCode parent header only applies with OpenCode's own session header;
        // next to a Codex `session-id` it must not surface as a parent.
        let metadata = Metadata::from_headers(&headers(&[
            ("session-id", "codex-run"),
            ("x-parent-session-id", "stray-parent"),
        ]));
        assert_eq!(metadata.session_id.as_deref(), Some("codex-run"));
        assert!(!metadata.is_subagent);
        assert_eq!(metadata.parent_agent_id, None);
    }

    #[test]
    fn dynamo_session_final_is_captured() {
        let metadata = Metadata::from_headers(&headers(&[
            ("x-dynamo-session-id", "generic-run"),
            ("x-dynamo-parent-session-id", "generic-parent"),
            ("x-dynamo-session-final", "true"),
        ]));
        assert_eq!(metadata.agent_id.as_deref(), Some("generic-run"));
        assert_eq!(metadata.parent_agent_id.as_deref(), Some("generic-parent"));
        assert_eq!(metadata.session_final, Some(true));

        let not_final = Metadata::from_headers(&headers(&[
            ("x-dynamo-session-id", "generic-run"),
            ("x-dynamo-session-final", "false"),
        ]));
        assert_eq!(not_final.session_final, Some(false));
    }

    #[test]
    fn codex_session_header_is_case_insensitive() {
        let metadata = Metadata::from_headers(&headers(&[("Session-ID", "codex-run")]));
        assert_eq!(metadata.session_id.as_deref(), Some("codex-run"));
    }
}
