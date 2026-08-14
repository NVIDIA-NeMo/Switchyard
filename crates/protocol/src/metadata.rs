// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Correlation metadata and harness header normalization.
//!
//! [`Metadata`] is the correlation/routing envelope carried alongside a request or
//! response. [`Metadata::from_headers`] normalizes host-specific HTTP headers into
//! that neutral shape.

use std::{collections::BTreeMap, str::FromStr as _};

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

// Correlation-header aliases used by integrating hosts.
const RELAY_SESSION_ID_HEADER: &str = "x-nemo-relay-session-id";
const RELAY_SUBAGENT_ID_HEADER: &str = "x-nemo-relay-subagent-id";

// Additional correlation-header aliases used by integrating hosts.
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
    pub http_headers: Option<http::HeaderMap>,
    /// The wire format the request/response was originally encoded in, if known.
    pub wire_format: Option<WireFormat>,
}

impl Metadata {
    /// Create Metadata
    pub fn from_headers(headers: &http::HeaderMap) -> Self {
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
/// Recognized sub-agent signals include `x-claude-code-agent-id`,
/// `x-openai-subagent`, `x-codex-turn-metadata.subagent_kind`, and explicit
/// `x-switchyard-is-subagent`. Other host correlation and parent-session headers
/// may populate metadata but do not drive sub-agent classification.
///
/// `is_delegated_work` is computed from raw harness signals, not from `agent_kind`,
/// which may be set by an unrelated operator label (`x-switchyard-agent-kind`).
fn parse_sub_agent(headers: &http::HeaderMap) -> (Option<String>, bool, bool) {
    let explicit = header(headers, SWITCHYARD_IS_SUBAGENT_HEADER).and_then(parse_bool);

    let (claude_parent, claude_subagent) = claude_lineage(headers);

    // Harness routing signal: Codex turn-metadata kind or flat OpenAI subagent header.
    // `x-switchyard-agent-kind` (operator semantic label) is intentionally excluded.
    let harness_kind = resolve_path(headers, CODEX_SUBAGENT_KIND_PATH)
        .or_else(|| header(headers, OPENAI_SUBAGENT_HEADER).map(str::to_string));

    // Resolve the parent through the configured header precedence, then fall back
    // to the native agent session the child was spawned under.
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
fn claude_lineage(headers: &http::HeaderMap) -> (Option<&str>, bool) {
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

/// Resolves the logical field `key` against `headers` using [`HEADER_CONFIG`]'s paths.
///
/// Returns the value of the first configured path that resolves, or `None` when the
/// field is absent from [`HEADER_CONFIG`] or nothing resolves. Descending into JSON
/// yields owned values, so the result is a `String` rather than a borrow of `headers`.
fn sy_header(headers: &http::HeaderMap, key: &str) -> Option<String> {
    let (_, paths) = HEADER_CONFIG
        .iter()
        .find(|(field, _)| field.eq_ignore_ascii_case(key))?;
    paths.iter().find_map(|path| resolve_path(headers, path))
}

/// Follows one dotted path, descending through a JSON-object header value.
/// Do not use if you expect multiple values for this header.
fn resolve_path(headers: &http::HeaderMap, path: &str) -> Option<String> {
    let (header_name, nested) = match path.split_once('.') {
        Some((name, rest)) => (name, Some(rest)),
        None => (path, None),
    };
    let raw = headers.get(header_name)?.to_str().ok().map(|s| s.trim())?;
    if raw.is_empty() {
        return None;
    }

    // A bare header name resolves to its value verbatim; no JSON parsing needed.
    let Some(nested) = nested else {
        return Some(raw.to_string());
    };

    // Nested path: parse the header value as JSON and descend key by key.
    let mut current: serde_json::Value = serde_json::from_str(raw).ok()?;
    for segment in nested.split('.') {
        current = current.as_object()?.get(segment)?.clone();
    }

    match current {
        serde_json::Value::String(s) => {
            let value = s.trim();
            (!value.is_empty()).then(|| value.to_string())
        }
        serde_json::Value::Null => None,
        leaf => Some(leaf.to_string()),
    }
}

fn header<'a>(headers: &'a http::HeaderMap, key: &str) -> Option<&'a str> {
    headers
        .get(key)
        .and_then(|s| s.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

/// Utility to convert a slice of string pairs into an `http::HeaderMap`.
pub fn slice_to_header_map(sl: &[(&str, &str)]) -> http::HeaderMap {
    let mut m = http::HeaderMap::with_capacity(sl.len());
    for (k, v) in sl {
        m.insert(
            http::HeaderName::from_str(k).unwrap(),
            (*v).try_into().unwrap(),
        );
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Header carrying Codex's structured turn metadata as a JSON object.
    const CODEX_TURN_METADATA_HEADER: &str = "x-codex-turn-metadata";

    fn metadata(headers: &[(&str, &str)]) -> Metadata {
        Metadata::from_headers(&slice_to_header_map(headers))
    }

    #[test]
    fn normalizes_codex_metadata_and_lineage() {
        let child_body = serde_json::json!({
            "session_id": "root-session",
            "thread_id": "child-agent",
            "parent_thread_id": "root-agent",
            "turn_id": "turn-7",
            "subagent_kind": "collab_spawn",
        })
        .to_string();
        let child = metadata(&[(CODEX_TURN_METADATA_HEADER, child_body.as_str())]);
        assert_eq!(child.session_id.as_deref(), Some("root-session"));
        assert_eq!(child.agent_id.as_deref(), Some("child-agent"));
        assert_eq!(child.parent_agent_id.as_deref(), Some("root-agent"));
        assert!(child.is_subagent);

        let root_body = serde_json::json!({
            "session_id": "root-session",
            "thread_id": "root-agent",
            "turn_id": "turn-1",
        })
        .to_string();
        let root = metadata(&[(CODEX_TURN_METADATA_HEADER, root_body.as_str())]);
        assert!(!root.is_subagent);

        // Parent-thread-id is correlation data, not a routing signal. A Codex
        // turn that carries a parent thread id but no `x-openai-subagent` must
        // not be treated as sub-agent work.
        let correlated_body = serde_json::json!({
            "session_id": "root-session",
            "thread_id": "child-thread",
            "parent_thread_id": "root-thread",
            "turn_id": "turn-3",
        })
        .to_string();
        let correlated = metadata(&[(CODEX_TURN_METADATA_HEADER, correlated_body.as_str())]);
        assert_eq!(correlated.parent_agent_id.as_deref(), Some("root-thread"));
        assert!(!correlated.is_subagent);
        assert!(!correlated.is_subagent_work());
    }

    #[test]
    fn normalizes_claude_code_metadata_and_lineage() {
        // Claude Code identifies a session with `x-claude-code-session-id`; session
        // affinity keys on it so a whole CLI session pins to one tier.
        let session = metadata(&[(
            "x-claude-code-session-id",
            "fb46caae-eac6-4f5f-83fd-8fc8f5743abb",
        )]);
        assert_eq!(
            session.session_id.as_deref(),
            Some("fb46caae-eac6-4f5f-83fd-8fc8f5743abb")
        );

        // Any non-empty agent id is a child agent. Without an explicit parent
        // header the parent is inferred to be the session it was spawned under.
        let child = metadata(&[
            ("x-claude-code-session-id", "claude-session"),
            ("x-claude-code-agent-id", "claude-agent"),
        ]);
        assert_eq!(child.session_id.as_deref(), Some("claude-session"));
        assert_eq!(child.agent_id.as_deref(), Some("claude-agent"));
        assert_eq!(child.parent_agent_id.as_deref(), Some("claude-session"));
        assert!(child.is_subagent);

        let child_without_session = metadata(&[("x-claude-code-agent-id", "claude-agent")]);
        assert_eq!(
            child_without_session.agent_id.as_deref(),
            Some("claude-agent")
        );
        assert_eq!(child_without_session.parent_agent_id, None);
        assert!(child_without_session.is_subagent);

        let explicit_parent = metadata(&[
            ("x-claude-code-session-id", "claude-session"),
            ("x-claude-code-agent-id", "claude-agent"),
            ("x-claude-code-parent-agent-id", "claude-parent-agent"),
        ]);
        assert_eq!(
            explicit_parent.parent_agent_id.as_deref(),
            Some("claude-parent-agent")
        );

        // Root agents omit x-claude-code-agent-id entirely. A stray parent-agent
        // header without an agent-id must not mark the request as a child.
        let root = metadata(&[
            ("x-claude-code-session-id", "claude-session"),
            ("x-claude-code-parent-agent-id", "claude-parent-agent"),
        ]);
        assert_eq!(root.session_id.as_deref(), Some("claude-session"));
        assert_eq!(root.agent_id, None);
        assert_eq!(root.parent_agent_id, None);
        assert!(!root.is_subagent);
    }

    #[test]
    fn normalizes_correlation_and_session_headers_without_routing() {
        // Integrating-host headers are correlation data, not routing signals.
        let relay = metadata(&[
            ("x-nemo-relay-session-id", "relay-session"),
            ("x-nemo-relay-subagent-id", "relay-child"),
            ("x-dynamo-parent-session-id", "relay-parent"),
        ]);
        assert_eq!(relay.session_id.as_deref(), Some("relay-session"));
        assert_eq!(relay.agent_id.as_deref(), Some("relay-child"));
        assert_eq!(relay.parent_agent_id.as_deref(), Some("relay-parent"));
        assert!(!relay.is_subagent);
        assert!(!relay.is_subagent_work());

        let opencode = metadata(&[
            ("x-session-id", "opencode-run"),
            ("x-parent-session-id", "opencode-parent"),
        ]);
        assert_eq!(opencode.session_id.as_deref(), Some("opencode-run"));
        assert_eq!(opencode.parent_agent_id, None);
        assert!(!opencode.is_subagent);

        let codex_session = metadata(&[
            ("session-id", "codex-run"),
            ("x-parent-session-id", "stray-parent"),
        ]);
        assert_eq!(codex_session.session_id.as_deref(), Some("codex-run"));
        assert_eq!(codex_session.parent_agent_id, None);
        assert!(!codex_session.is_subagent);

        let final_session = metadata(&[
            ("x-dynamo-session-id", "generic-run"),
            ("x-dynamo-parent-session-id", "generic-parent"),
            ("x-dynamo-session-final", "true"),
        ]);
        assert_eq!(final_session.agent_id.as_deref(), Some("generic-run"));
        assert_eq!(
            final_session.parent_agent_id.as_deref(),
            Some("generic-parent")
        );
        assert_eq!(final_session.session_final, Some(true));

        let active_session = metadata(&[
            ("x-dynamo-session-id", "generic-run"),
            ("x-dynamo-session-final", "false"),
        ]);
        assert_eq!(active_session.session_final, Some(false));
    }

    #[test]
    fn sy_header_resolves_paths_in_order_and_descends_into_json() {
        // Only the JSON-nested Codex path is present, so descent supplies the value.
        let body = serde_json::json!({ "session_id": "codex-session" }).to_string();
        let headers = slice_to_header_map(&[(CODEX_TURN_METADATA_HEADER, body.as_str())]);
        assert_eq!(
            sy_header(&headers, SWITCHYARD_SESSION_ID_HEADER).as_deref(),
            Some("codex-session")
        );

        // The explicit Switchyard header outranks the Codex path when both resolve.
        let headers = slice_to_header_map(&[
            (SWITCHYARD_SESSION_ID_HEADER, "explicit"),
            (CODEX_TURN_METADATA_HEADER, body.as_str()),
        ]);
        assert_eq!(
            sy_header(&headers, SWITCHYARD_SESSION_ID_HEADER).as_deref(),
            Some("explicit")
        );

        // Nothing resolves for an empty header set or an unknown field.
        assert_eq!(
            sy_header(&http::HeaderMap::new(), SWITCHYARD_SESSION_ID_HEADER),
            None
        );
        assert_eq!(sy_header(&headers, "x-not-a-field"), None);
    }

    // Blank nested metadata must not mask a valid lower-priority session header.
    #[test]
    fn nested_metadata_strings_match_flat_header_normalization() {
        let body = serde_json::json!({ "session_id": "  codex-session  " }).to_string();
        let headers = slice_to_header_map(&[(CODEX_TURN_METADATA_HEADER, body.as_str())]);
        assert_eq!(
            sy_header(&headers, SWITCHYARD_SESSION_ID_HEADER).as_deref(),
            Some("codex-session")
        );

        let blank_body = serde_json::json!({ "session_id": "   " }).to_string();
        let headers = slice_to_header_map(&[
            (CODEX_TURN_METADATA_HEADER, blank_body.as_str()),
            (SESSION_ID_HEADER, "fallback-session"),
        ]);
        assert_eq!(
            sy_header(&headers, SWITCHYARD_SESSION_ID_HEADER).as_deref(),
            Some("fallback-session")
        );
    }

    #[test]
    fn subagent_routing_honors_explicit_signals_and_delegated_work_kinds() {
        // Explicit `false` wins over presence-based inference even when no
        // parent id accompanies it; the flag decides in both directions.
        let explicitly_root = metadata(&[
            ("x-switchyard-is-subagent", "false"),
            ("x-openai-subagent", "review"),
        ]);
        assert!(!explicitly_root.is_subagent);

        let explicitly_child = metadata(&[("x-switchyard-is-subagent", "true")]);
        assert!(explicitly_child.is_subagent);

        let child_with_parent = metadata(&[
            ("x-switchyard-is-subagent", "false"),
            ("x-switchyard-parent-agent-id", "parent"),
        ]);
        assert!(!child_with_parent.is_subagent);

        // Operator labels do not filter routing signals from the harness.
        let with_openai = metadata(&[
            ("x-openai-subagent", "review"),
            ("x-switchyard-agent-kind", "researcher"),
        ]);
        assert!(with_openai.is_subagent);
        assert!(with_openai.is_subagent_work());

        let with_explicit = metadata(&[
            ("x-switchyard-is-subagent", "true"),
            ("x-switchyard-agent-kind", "researcher"),
        ]);
        assert!(with_explicit.is_subagent);
        assert!(with_explicit.is_subagent_work());

        // Kindless lineage (Claude Code child agent) counts as delegated work.
        let claude_child = metadata(&[
            ("x-claude-code-session-id", "root"),
            ("x-claude-code-agent-id", "worker"),
        ]);
        assert!(claude_child.is_subagent_work());

        // Codex delegated-work kinds route as sub-agent work.
        let review = metadata(&[("x-openai-subagent", "review")]);
        assert!(review.is_subagent_work());

        // Harness maintenance and unknown kinds stay on normal routing even
        // though the lineage fact still marks them as child-agent requests.
        for kind in ["compact", "memory_consolidation", "brand_new_kind"] {
            let request = metadata(&[("x-openai-subagent", kind)]);
            assert!(request.is_subagent, "{kind} keeps the lineage fact");
            assert!(!request.is_subagent_work(), "{kind} is not routed as work");
        }

        // A non-subagent request is never work, whatever its kind says.
        assert!(!Metadata::default().is_subagent_work());
    }
}
