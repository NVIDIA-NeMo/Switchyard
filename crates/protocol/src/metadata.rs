// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Correlation metadata and harness header normalization.
//!
//! [`Metadata`] is the correlation/routing envelope carried alongside a request or
//! response. [`Metadata::from_headers`] normalizes the harness-specific HTTP headers
//! that Claude Code, Codex, NeMo Relay, and Dynamo attach into that neutral shape.

use std::collections::BTreeMap;

use crate::WireFormat;

// Dotted paths addressing fields inside Codex's turn-metadata header JSON value.
const CODEX_SESSION_ID_PATH: &str = "x-codex-turn-metadata.session_id";
const CODEX_THREAD_ID_PATH: &str = "x-codex-turn-metadata.thread_id";
const CODEX_PARENT_THREAD_ID_PATH: &str = "x-codex-turn-metadata.parent_thread_id";
const CODEX_TURN_ID_PATH: &str = "x-codex-turn-metadata.turn_id";
const CODEX_SUBAGENT_KIND_PATH: &str = "x-codex-turn-metadata.subagent_kind";
const CODEX_AGENT_ROLE_PATH: &str = "x-codex-turn-metadata.agent_role";
const CODEX_TASK_ID_PATH: &str = "x-codex-turn-metadata.task_id";
const CODEX_TASK_KIND_PATH: &str = "x-codex-turn-metadata.task_kind";

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
const SWITCHYARD_REQUEST_ID_HEADER: &str = "x-switchyard-request-id";
const SWITCHYARD_SESSION_FINAL_HEADER: &str = "x-switchyard-session-final";

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

/// Header/JSON-path signals that, when any is present, mark a request as a sub-agent.
const SUBAGENT_SIGNAL_PATHS: &[&str] = &[
    SWITCHYARD_PARENT_AGENT_ID_HEADER,
    RELAY_SUBAGENT_ID_HEADER,
    DYNAMO_PARENT_SESSION_ID_HEADER,
    CODEX_PARENT_THREAD_ID_PATH,
    CODEX_SUBAGENT_KIND_PATH,
    CODEX_PARENT_THREAD_ID_HEADER,
    OPENAI_SUBAGENT_HEADER,
];

/// Ordered candidate lookup paths for each correlation field, keyed by the field's
/// canonical `x-switchyard-*` header name.
type HeaderConfig = [(&'static str, &'static [&'static str])];

/// Precedence of harness headers for each correlation field. See [`HeaderConfig`].
const HEADER_CONFIG: &HeaderConfig = &[
    (
        SWITCHYARD_SESSION_ID_HEADER,
        &[
            SWITCHYARD_SESSION_ID_HEADER,
            CLAUDE_SESSION_ID_HEADER,
            RELAY_SESSION_ID_HEADER,
            OPENCODE_SESSION_ID_HEADER,
            CODEX_SESSION_ID_PATH,
            SESSION_ID_HEADER,
        ],
    ),
    (
        SWITCHYARD_AGENT_ID_HEADER,
        &[
            SWITCHYARD_AGENT_ID_HEADER,
            CLAUDE_AGENT_ID_HEADER,
            RELAY_SUBAGENT_ID_HEADER,
            DYNAMO_SESSION_ID_HEADER,
            CODEX_THREAD_ID_PATH,
            THREAD_ID_HEADER,
        ],
    ),
    (
        SWITCHYARD_PARENT_AGENT_ID_HEADER,
        &[
            SWITCHYARD_PARENT_AGENT_ID_HEADER,
            DYNAMO_PARENT_SESSION_ID_HEADER,
            CODEX_PARENT_THREAD_ID_PATH,
            CODEX_PARENT_THREAD_ID_HEADER,
        ],
    ),
    (
        SWITCHYARD_AGENT_KIND_HEADER,
        &[
            SWITCHYARD_AGENT_KIND_HEADER,
            CODEX_SUBAGENT_KIND_PATH,
            OPENAI_SUBAGENT_HEADER,
        ],
    ),
    (
        SWITCHYARD_AGENT_ROLE_HEADER,
        &[SWITCHYARD_AGENT_ROLE_HEADER, CODEX_AGENT_ROLE_PATH],
    ),
    (
        SWITCHYARD_TASK_ID_HEADER,
        &[
            SWITCHYARD_TASK_ID_HEADER,
            CODEX_TASK_ID_PATH,
            TASK_ID_HEADER,
        ],
    ),
    (
        SWITCHYARD_TASK_KIND_HEADER,
        &[SWITCHYARD_TASK_KIND_HEADER, CODEX_TASK_KIND_PATH],
    ),
    (
        SWITCHYARD_TURN_ID_HEADER,
        &[SWITCHYARD_TURN_ID_HEADER, CODEX_TURN_ID_PATH],
    ),
    (
        SWITCHYARD_REQUEST_ID_HEADER,
        &[
            SWITCHYARD_REQUEST_ID_HEADER,
            REQUEST_ID_HEADER,
            CLIENT_REQUEST_ID_HEADER,
        ],
    ),
    (
        SWITCHYARD_SESSION_FINAL_HEADER,
        &[SWITCHYARD_SESSION_FINAL_HEADER, DYNAMO_SESSION_FINAL_HEADER],
    ),
];

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

        let (parent_agent_id, is_subagent) = parse_sub_agent(headers);

        Metadata {
            session_id: sy_header(headers, SWITCHYARD_SESSION_ID_HEADER),
            agent_id: sy_header(headers, SWITCHYARD_AGENT_ID_HEADER),
            parent_agent_id,
            is_subagent,
            agent_kind: sy_header(headers, SWITCHYARD_AGENT_KIND_HEADER),
            agent_role: sy_header(headers, SWITCHYARD_AGENT_ROLE_HEADER),
            task_id: sy_header(headers, SWITCHYARD_TASK_ID_HEADER),
            task_kind: sy_header(headers, SWITCHYARD_TASK_KIND_HEADER),
            turn_id: sy_header(headers, SWITCHYARD_TURN_ID_HEADER),
            session_final: sy_header(headers, SWITCHYARD_SESSION_FINAL_HEADER)
                .as_deref()
                .and_then(parse_bool),
            correlation_id: sy_header(headers, SWITCHYARD_REQUEST_ID_HEADER),
            ..Metadata::default()
        }
    }
}

/// Returns `(parent_agent_id, is_subagent)` from the headers
fn parse_sub_agent(headers: &BTreeMap<String, String>) -> (Option<String>, bool) {
    let mut is_subagent = false;
    let sy_is_sub_agent = header(headers, SWITCHYARD_IS_SUBAGENT_HEADER).and_then(parse_bool);
    let mut parent: Option<String> = sy_header(headers, SWITCHYARD_PARENT_AGENT_ID_HEADER);
    if let Some(is_sub) = sy_is_sub_agent {
        if parent.is_some() {
            return (parent, is_sub);
        }
    }
    is_subagent |= sy_is_sub_agent.unwrap_or(false);

    let (claude_parent, claude_subagent) = claude_lineage(headers);
    is_subagent |= claude_subagent;
    parent = parent.or_else(|| claude_parent.map(str::to_string));
    if is_subagent && parent.is_some() {
        return (parent, is_subagent);
    }

    parent = parent.or_else(|| opencode_parent(headers).map(str::to_string));
    is_subagent = parent.is_some() || is_subagent;
    if is_subagent && parent.is_some() {
        return (parent, is_subagent);
    }

    is_subagent |= SUBAGENT_SIGNAL_PATHS
        .iter()
        .any(|path| resolve_path(headers, path).is_some());

    (parent, is_subagent)
}

/// Claude Code's `(parent_agent, is_subagent)` from its native lineage headers.
///
/// A request naming an `x-claude-code-agent-id` distinct from its session is a child
/// agent; its parent is the explicit parent-agent header, else the session it was
/// spawned under. A root agent (no distinct child id) has no parent.
fn claude_lineage(headers: &BTreeMap<String, String>) -> (Option<&str>, bool) {
    let session = header(headers, CLAUDE_SESSION_ID_HEADER);
    let agent = header(headers, CLAUDE_AGENT_ID_HEADER);
    let is_subagent = matches!((agent, session), (Some(a), Some(s)) if a != s);
    let parent = is_subagent
        .then(|| header(headers, CLAUDE_PARENT_AGENT_ID_HEADER).or(session))
        .flatten();
    (parent, is_subagent)
}

/// OpenCode's parent session, meaningful only alongside its own session header.
fn opencode_parent(headers: &BTreeMap<String, String>) -> Option<&str> {
    header(headers, OPENCODE_SESSION_ID_HEADER)
        .and(header(headers, OPENCODE_PARENT_SESSION_ID_HEADER))
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

/// Resolves the logical field `key` against `headers` using [`HEADER_CONFIG`]'s paths.
///
/// Returns the value of the first configured path that resolves, or `None` when the
/// field is absent from [`HEADER_CONFIG`] or nothing resolves. Descending into JSON
/// yields owned values, so the result is a `String` rather than a borrow of `headers`.
fn sy_header(headers: &BTreeMap<String, String>, key: &str) -> Option<String> {
    let (_, paths) = HEADER_CONFIG
        .iter()
        .find(|(field, _)| field.eq_ignore_ascii_case(key))?;
    paths.iter().find_map(|path| resolve_path(headers, path))
}

/// Follows one dotted path, descending through a JSON-object header value.
fn resolve_path(headers: &BTreeMap<String, String>, path: &str) -> Option<String> {
    let (header_name, nested) = match path.split_once('.') {
        Some((name, rest)) => (name, Some(rest)),
        None => (path, None),
    };
    let raw = headers.get(&header_name.to_ascii_lowercase())?;

    // A bare header name resolves to its value verbatim; no JSON parsing needed.
    let Some(nested) = nested else {
        return Some(raw.clone());
    };

    // Nested path: parse the header value as JSON and descend key by key.
    let mut current: serde_json::Value = serde_json::from_str(raw).ok()?;
    for segment in nested.split('.') {
        current = current.as_object()?.get(segment)?.clone();
    }

    match current {
        serde_json::Value::String(s) => Some(s),
        serde_json::Value::Null => None,
        leaf => Some(leaf.to_string()),
    }
}

fn header<'a>(headers: &'a BTreeMap<String, String>, key: &str) -> Option<&'a str> {
    let lower_key = key.to_ascii_lowercase();
    headers.get(&lower_key).map(|s| s.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Header carrying Codex's structured turn metadata as a JSON object.
    const CODEX_TURN_METADATA_HEADER: &str = "x-codex-turn-metadata";

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
    fn sy_header_resolves_paths_in_order_and_descends_into_json() {
        // Only the JSON-nested Codex path is present, so descent supplies the value.
        let headers = BTreeMap::from([(
            CODEX_TURN_METADATA_HEADER.to_string(),
            serde_json::json!({ "session_id": "codex-session" }).to_string(),
        )]);
        assert_eq!(
            sy_header(&headers, SWITCHYARD_SESSION_ID_HEADER).as_deref(),
            Some("codex-session")
        );

        // The explicit Switchyard header outranks the Codex path when both resolve.
        let headers = BTreeMap::from([
            (
                SWITCHYARD_SESSION_ID_HEADER.to_string(),
                "explicit".to_string(),
            ),
            (
                CODEX_TURN_METADATA_HEADER.to_string(),
                serde_json::json!({ "session_id": "codex-session" }).to_string(),
            ),
        ]);
        assert_eq!(
            sy_header(&headers, SWITCHYARD_SESSION_ID_HEADER).as_deref(),
            Some("explicit")
        );

        // Nothing resolves for an empty header set or an unknown field.
        assert_eq!(
            sy_header(&BTreeMap::new(), SWITCHYARD_SESSION_ID_HEADER),
            None
        );
        assert_eq!(sy_header(&headers, "x-not-a-field"), None);
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
