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

// OpenCode session header — used for session_id correlation only (not a routing signal).
const OPENCODE_SESSION_ID_HEADER: &str = "x-session-id";

// Generic Codex-compatible correlation headers.
const SESSION_ID_HEADER: &str = "session-id";
const THREAD_ID_HEADER: &str = "thread-id";
const TASK_ID_HEADER: &str = "x-task-id";
const REQUEST_ID_HEADER: &str = "x-request-id";
const CLIENT_REQUEST_ID_HEADER: &str = "x-client-request-id";

/// Harness-defined sub-agent kinds that carry delegated user work rather than
/// harness maintenance (`compact`, `memory_consolidation`, ...). Unknown kinds
/// are excluded deliberately; extend with captured request fixtures.
const SUBAGENT_WORK_KINDS: &[&str] = &["collab_spawn", "review"];

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
    /// Whether this request carries delegated sub-agent *work* and should be
    /// routed to the sub-agent target. Computed from raw harness signals only,
    /// independent of [`Self::agent_kind`], which may be set by an unrelated
    /// operator label (`x-switchyard-agent-kind`).
    pub is_delegated_work: bool,
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
    /// headers are accepted for observability without driving routing. Codex's
    /// structured turn metadata is preferred over its compatibility projections.
    /// Claude Code carries agent lineage in native headers: a request with a
    /// non-empty `x-claude-code-agent-id` is treated as a child agent (its parent
    /// inferred from the session when not stated). Sub-agent routing status is taken
    /// from an explicit `x-switchyard-is-subagent` header when present, and otherwise
    /// inferred from Claude Code lineage or Codex harness signals.
    pub fn from_headers(headers: &BTreeMap<String, String>) -> Self {
        let headers = &normalize_headers(headers);

        let (parent_agent_id, is_subagent, is_delegated_work) = parse_sub_agent(headers);

        Metadata {
            session_id: sy_header(headers, SWITCHYARD_SESSION_ID_HEADER),
            agent_id: sy_header(headers, SWITCHYARD_AGENT_ID_HEADER),
            parent_agent_id,
            is_subagent,
            is_delegated_work,
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

    /// Whether this request should be routed to the sub-agent target.
    ///
    /// Returns `self.is_delegated_work`, which is computed in `parse_sub_agent`
    /// from raw harness signals only — independent of `agent_kind`, which may
    /// be populated by an unrelated operator label (`x-switchyard-agent-kind`).
    pub fn is_subagent_work(&self) -> bool {
        self.is_delegated_work
    }
}

/// Returns `(parent_agent_id, is_subagent, is_delegated_work)` from the headers.
///
/// Recognized sub-agent signals: Claude Code `x-claude-code-agent-id`, Codex
/// `x-openai-subagent` / `x-codex-turn-metadata.subagent_kind`, and explicit
/// `x-switchyard-is-subagent`. Correlation-only headers (Relay, Dynamo, OpenCode
/// parent sessions) populate observability fields but do not drive routing.
///
/// `is_delegated_work` is computed from raw harness signals, not from `agent_kind`,
/// which may be set by an unrelated operator label (`x-switchyard-agent-kind`).
fn parse_sub_agent(headers: &BTreeMap<String, String>) -> (Option<String>, bool, bool) {
    let explicit = header(headers, SWITCHYARD_IS_SUBAGENT_HEADER).and_then(parse_bool);

    let (claude_parent, claude_subagent) = claude_lineage(headers);

    // Harness routing signal: Codex turn-metadata kind or flat OpenAI subagent header.
    // `x-switchyard-agent-kind` (operator semantic label) is intentionally excluded.
    let harness_kind = resolve_path(headers, CODEX_SUBAGENT_KIND_PATH)
        .or_else(|| header(headers, OPENAI_SUBAGENT_HEADER).map(str::to_string));

    // Parent resolved via HEADER_CONFIG precedence (covers Dynamo/Codex correlation);
    // falls back to the Claude Code session the child was spawned under.
    let parent = sy_header(headers, SWITCHYARD_PARENT_AGENT_ID_HEADER)
        .or_else(|| claude_parent.map(str::to_string));

    let is_subagent = explicit.unwrap_or(claude_subagent || harness_kind.is_some());

    let is_delegated_work = match explicit {
        Some(false) => false,
        Some(true) => harness_kind
            .as_deref()
            .map(|k| SUBAGENT_WORK_KINDS.contains(&k))
            .unwrap_or(true),
        None => {
            claude_subagent
                || harness_kind
                    .as_deref()
                    .is_some_and(|k| SUBAGENT_WORK_KINDS.contains(&k))
        }
    };

    (parent, is_subagent, is_delegated_work)
}

/// Claude Code's `(parent_agent, is_subagent)` from its native lineage headers.
///
/// Claude Code only sends `x-claude-code-agent-id` for spawned sub-agents and
/// teammates; root agents omit it. Any non-empty value is therefore a
/// sub-agent signal. The parent is the explicit parent-agent header when
/// present, else the session the child was spawned under.
fn claude_lineage(headers: &BTreeMap<String, String>) -> (Option<&str>, bool) {
    let session = header(headers, CLAUDE_SESSION_ID_HEADER);
    let agent = header(headers, CLAUDE_AGENT_ID_HEADER);
    let is_subagent = agent.is_some();
    let parent = is_subagent
        .then(|| header(headers, CLAUDE_PARENT_AGENT_ID_HEADER).or(session))
        .flatten();
    (parent, is_subagent)
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
    fn codex_parent_thread_id_alone_is_not_a_subagent_signal() {
        // Parent-thread-id is correlation data, not a routing signal. A Codex
        // turn that carries a parent thread id but no `x-openai-subagent` must
        // not be treated as sub-agent work.
        let headers = BTreeMap::from([(
            CODEX_TURN_METADATA_HEADER.to_string(),
            serde_json::json!({
                "session_id": "root-session",
                "thread_id": "child-thread",
                "parent_thread_id": "root-thread",
                "turn_id": "turn-3",
                // no subagent_kind
            })
            .to_string(),
        )]);

        let metadata = Metadata::from_headers(&headers);
        // Parent id is still captured for observability.
        assert_eq!(metadata.parent_agent_id.as_deref(), Some("root-thread"));
        // But it must not drive routing.
        assert!(!metadata.is_subagent);
        assert!(!metadata.is_subagent_work());
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
        // Relay and Dynamo headers are correlation data, not routing signals.
        // They populate observability fields but must not trigger sub-agent routing.
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
        assert_eq!(metadata.parent_agent_id.as_deref(), Some("relay-parent"));
        assert!(!metadata.is_subagent);
        assert!(!metadata.is_subagent_work());
    }

    #[test]
    fn claude_code_agent_lineage_marks_subagent_and_infers_parent() {
        // Any non-empty agent id is a child agent. Without an explicit parent
        // header the parent is inferred to be the session it was spawned under.
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
    fn claude_code_agent_id_alone_marks_subagent() {
        // The agent-id header is the detection predicate; the session header is
        // correlation data. A request with only agent-id is still a child agent.
        let metadata = Metadata::from_headers(&BTreeMap::from([(
            "x-claude-code-agent-id".to_string(),
            "claude-agent".to_string(),
        )]));
        assert!(metadata.is_subagent);
        assert_eq!(metadata.agent_id.as_deref(), Some("claude-agent"));
        assert_eq!(metadata.parent_agent_id, None);
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
    fn claude_root_agent_without_agent_id_is_not_a_subagent() {
        // Root agents omit x-claude-code-agent-id entirely. A stray parent-agent
        // header without an agent-id must not mark the request as a child.
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
    fn opencode_session_headers_are_correlation_only() {
        // OpenCode's x-session-id / x-parent-session-id are correlation headers;
        // they populate session_id for observability but do not trigger routing.
        let metadata = Metadata::from_headers(&BTreeMap::from([
            ("x-session-id".to_string(), "opencode-run".to_string()),
            (
                "x-parent-session-id".to_string(),
                "opencode-parent".to_string(),
            ),
        ]));
        assert_eq!(metadata.session_id.as_deref(), Some("opencode-run"));
        assert!(!metadata.is_subagent);
        assert_eq!(metadata.parent_agent_id, None);
    }

    #[test]
    fn opencode_parent_header_is_not_a_parent_agent_id_source() {
        // x-parent-session-id is not listed in HEADER_CONFIG for parent_agent_id;
        // it must not surface as a parent regardless of adjacent session headers.
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

    #[test]
    fn explicit_subagent_flag_decides_without_a_parent_header() {
        // Explicit `false` wins over presence-based inference even when no
        // parent id accompanies it; the flag decides in both directions.
        let metadata = Metadata::from_headers(&BTreeMap::from([
            ("x-switchyard-is-subagent".to_string(), "false".to_string()),
            ("x-openai-subagent".to_string(), "review".to_string()),
        ]));
        assert!(!metadata.is_subagent);

        let metadata = Metadata::from_headers(&BTreeMap::from([(
            "x-switchyard-is-subagent".to_string(),
            "true".to_string(),
        )]));
        assert!(metadata.is_subagent);
    }

    #[test]
    fn operator_agent_kind_does_not_suppress_harness_subagent_routing() {
        // x-switchyard-agent-kind is an operator semantic label and must not filter
        // routing signals from the harness (x-openai-subagent, x-switchyard-is-subagent).
        let with_openai = Metadata::from_headers(&BTreeMap::from([
            ("x-openai-subagent".to_string(), "review".to_string()),
            (
                "x-switchyard-agent-kind".to_string(),
                "researcher".to_string(),
            ),
        ]));
        assert!(with_openai.is_subagent);
        assert!(with_openai.is_subagent_work());

        let with_explicit = Metadata::from_headers(&BTreeMap::from([
            ("x-switchyard-is-subagent".to_string(), "true".to_string()),
            (
                "x-switchyard-agent-kind".to_string(),
                "researcher".to_string(),
            ),
        ]));
        assert!(with_explicit.is_subagent);
        assert!(with_explicit.is_subagent_work());
    }

    #[test]
    fn subagent_work_requires_a_delegated_work_kind_when_kinded() {
        // Kindless lineage (Claude Code child agent) counts as delegated work.
        let claude_child = Metadata::from_headers(&BTreeMap::from([
            ("x-claude-code-session-id".to_string(), "root".to_string()),
            ("x-claude-code-agent-id".to_string(), "worker".to_string()),
        ]));
        assert!(claude_child.is_subagent_work());

        // Codex delegated-work kinds route as sub-agent work.
        let review = Metadata::from_headers(&BTreeMap::from([(
            "x-openai-subagent".to_string(),
            "review".to_string(),
        )]));
        assert!(review.is_subagent_work());

        // Harness maintenance and unknown kinds stay on normal routing even
        // though the lineage fact still marks them as child-agent requests.
        for kind in ["compact", "memory_consolidation", "brand_new_kind"] {
            let metadata = Metadata::from_headers(&BTreeMap::from([(
                "x-openai-subagent".to_string(),
                kind.to_string(),
            )]));
            assert!(metadata.is_subagent, "{kind} keeps the lineage fact");
            assert!(!metadata.is_subagent_work(), "{kind} is not routed as work");
        }

        // A non-subagent request is never work, whatever its kind says.
        assert!(!Metadata::default().is_subagent_work());
    }
}
