// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Durable per-request routing records and session snapshots.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use humantime::format_rfc3339_millis;
use serde::{Deserialize, Serialize};
use switchyard_protocol::{Metadata, ModelId, Usage};

use crate::usage_metrics::token_usage;
use crate::{ServerError, ServerResult};

const LEGACY_SESSION_ID_HEADER: &str = "proxy_x_session_id";
const ORIGIN_HEADER: &str = "x-switchyard-origin";
const TASK_HEADER: &str = "x-switchyard-intake-task";
const TRIAL_ID_HEADER: &str = "x-switchyard-trial-id";

/// Append-only writer for one routing JSONL file.
pub(crate) struct RoutingLog(fs::File);

impl RoutingLog {
    pub(crate) fn new(path: impl Into<PathBuf>) -> ServerResult<Self> {
        let path = path.into();
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent).map_err(|error| routing_log_error(&path, error))?;
        }
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|error| routing_log_error(&path, error))?;
        Ok(Self(file))
    }

    /// Writes one record for a finished call as a single JSON line.
    pub(crate) fn append(
        &mut self,
        context: RoutingLogContext,
        model: &str,
        tier: Option<&str>,
        usage: &Usage,
    ) -> std::io::Result<()> {
        let usage = token_usage(usage);
        let record = RoutingRecord {
            ts: format_rfc3339_millis(SystemTime::now()).to_string().into(),
            route_id: context.route_id.into(),
            algorithm: context.algorithm.into(),
            origin: context.origin.map(Cow::Owned),
            task: context.task.map(Cow::Owned),
            trial_id: context.trial_id.map(Cow::Owned),
            session_id: context.session_id.map(Cow::Owned),
            turn_id: context.turn_id.map(Cow::Owned),
            agent_id: context.agent.agent_id.map(Cow::Owned),
            parent_agent_id: context.agent.parent_agent_id.map(Cow::Owned),
            delegated: context.agent.delegated,
            agent_kind: context.agent.kind.map(Cow::Owned),
            agent_role: context.agent.role.map(Cow::Owned),
            subagent_identity_unsupported: context.agent.identity_unsupported,
            model: model.into(),
            tier: tier.unwrap_or("").into(),
            prompt_tokens: usage.prompt_tokens,
            cached_tokens: usage.cached_tokens,
            cache_creation_tokens: usage.cache_creation_tokens,
            completion_tokens: usage.completion_tokens,
            reasoning_tokens: usage.reasoning_tokens,
            total_tokens: usage.prompt_tokens.saturating_add(usage.completion_tokens),
        };
        let mut line = serde_json::to_vec(&record).map_err(std::io::Error::other)?;
        line.push(b'\n');

        self.0.write_all(&line)
    }
}

/// Reads complete records without synchronizing with the writer.
pub(crate) fn snapshot(
    path: &Path,
    session_id: &str,
) -> std::io::Result<Option<SessionStatsSnapshot>> {
    let mut reader = BufReader::with_capacity(64 * 1024, fs::File::open(path)?);
    let mut line = Vec::new();
    let mut snapshot = SessionStatsSnapshot::new(session_id);
    loop {
        line.clear();
        if reader.read_until(b'\n', &mut line)? == 0 {
            break;
        }
        if !line.ends_with(b"\n") {
            break;
        }
        let Ok(record) = serde_json::from_slice::<RoutingRecord>(&line) else {
            continue;
        };
        snapshot.add_record(&record, session_id);
    }
    snapshot.sum_totals();
    Ok((snapshot.total_calls > 0).then_some(snapshot))
}

/// Which agent made a call. A child agent can share its parent's session id, so
/// the session alone cannot tell delegated work apart from the parent's turns.
/// Every field is copied from the request metadata, never inferred.
#[derive(Clone, Default)]
pub(crate) struct AgentIdentity {
    agent_id: Option<String>,
    parent_agent_id: Option<String>,
    /// [`Metadata::is_subagent_work`], the signal that sends a call to the
    /// sub-agent route, rather than the weaker `is_subagent` hint.
    delegated: bool,
    kind: Option<String>,
    role: Option<String>,
    /// The harness cannot identify its children, so a reader should not take
    /// a missing `delegated` flag to mean that no work was delegated.
    identity_unsupported: bool,
}

impl AgentIdentity {
    /// Copies the agent fields from the request metadata.
    fn from_metadata(metadata: &Metadata) -> Self {
        Self {
            agent_id: metadata.agent_id.clone(),
            parent_agent_id: metadata.parent_agent_id.clone(),
            delegated: metadata.is_subagent_work(),
            kind: metadata.agent_kind.clone(),
            role: metadata.agent_role.clone(),
            identity_unsupported: metadata.subagent_identity_unsupported,
        }
    }
}

/// Holds request metadata until route resolution attaches the durable route identity.
#[derive(Clone)]
pub(crate) struct RoutingLogContext {
    route_id: String,
    algorithm: String,
    origin: Option<String>,
    task: Option<String>,
    trial_id: Option<String>,
    session_id: Option<String>,
    turn_id: Option<String>,
    agent: AgentIdentity,
}

impl RoutingLogContext {
    /// Captures the normalized session ID, with the legacy log-only header as a fallback.
    pub(crate) fn from_metadata(metadata: &Metadata) -> Self {
        let headers = metadata.http_headers.as_ref();
        Self {
            route_id: String::new(),
            algorithm: String::new(),
            origin: headers
                .and_then(|headers| nonempty_header(headers, ORIGIN_HEADER))
                .map(str::to_string),
            task: headers
                .and_then(|headers| nonempty_header(headers, TASK_HEADER))
                .map(str::to_string),
            trial_id: headers
                .and_then(|headers| nonempty_header(headers, TRIAL_ID_HEADER))
                .map(str::to_string),
            turn_id: metadata.turn_id.clone(),
            session_id: metadata.session_id.clone().or_else(|| {
                headers
                    .and_then(|headers| nonempty_header(headers, LEGACY_SESSION_ID_HEADER))
                    .map(str::to_string)
            }),
            agent: AgentIdentity::from_metadata(metadata),
        }
    }

    /// Attaches the resolved route identity shared by every log entry for this request.
    pub(crate) fn with_route(mut self, route_id: impl Into<String>, algorithm: &str) -> Self {
        self.route_id = route_id.into();
        self.algorithm = algorithm.to_string();
        self
    }
}

/// One appended routing record, and the read schema [`snapshot`] parses back,
/// so the written and expected shapes cannot drift apart. Missing fields
/// default so a record from an older schema still contributes what it has.
#[derive(Default, Deserialize, Serialize)]
#[serde(default)]
struct RoutingRecord<'a> {
    ts: Cow<'a, str>,
    route_id: Cow<'a, str>,
    algorithm: Cow<'a, str>,
    #[serde(borrow)]
    origin: Option<Cow<'a, str>>,
    #[serde(borrow)]
    task: Option<Cow<'a, str>>,
    #[serde(borrow)]
    trial_id: Option<Cow<'a, str>>,
    #[serde(borrow)]
    session_id: Option<Cow<'a, str>>,
    #[serde(borrow)]
    turn_id: Option<Cow<'a, str>>,
    // Most calls carry no agent identity, so these fields are written only when
    // set. Records without them still parse through `serde(default)`.
    #[serde(borrow, skip_serializing_if = "Option::is_none")]
    agent_id: Option<Cow<'a, str>>,
    #[serde(borrow, skip_serializing_if = "Option::is_none")]
    parent_agent_id: Option<Cow<'a, str>>,
    #[serde(skip_serializing_if = "is_false")]
    delegated: bool,
    #[serde(borrow, skip_serializing_if = "Option::is_none")]
    agent_kind: Option<Cow<'a, str>>,
    #[serde(borrow, skip_serializing_if = "Option::is_none")]
    agent_role: Option<Cow<'a, str>>,
    #[serde(skip_serializing_if = "is_false")]
    subagent_identity_unsupported: bool,
    model: Cow<'a, str>,
    tier: Cow<'a, str>,
    prompt_tokens: u64,
    cached_tokens: u64,
    cache_creation_tokens: u64,
    completion_tokens: u64,
    reasoning_tokens: u64,
    total_tokens: u64,
}

/// Session totals returned by the routing stats endpoint.
#[derive(Serialize)]
pub(crate) struct SessionStatsSnapshot {
    session_id: String,
    total_calls: u64,
    total_prompt_tokens: u64,
    total_cached_tokens: u64,
    total_cache_creation_tokens: u64,
    total_completion_tokens: u64,
    models: BTreeMap<ModelId, SessionModelStats>,
}

#[derive(Default, Serialize)]
struct SessionModelStats {
    calls: u64,
    prompt_tokens: u64,
    cached_tokens: u64,
    cache_creation_tokens: u64,
    completion_tokens: u64,
}

impl SessionStatsSnapshot {
    fn new(session_id: &str) -> Self {
        Self {
            session_id: session_id.to_string(),
            total_calls: 0,
            total_prompt_tokens: 0,
            total_cached_tokens: 0,
            total_cache_creation_tokens: 0,
            total_completion_tokens: 0,
            models: BTreeMap::new(),
        }
    }

    fn add_record(&mut self, record: &RoutingRecord<'_>, session_id: &str) {
        if record.session_id.as_deref() != Some(session_id) {
            return;
        }
        let model = match record.model.as_ref() {
            "" => "unknown",
            model => model,
        };
        let stats = self.models.entry(ModelId::from(model)).or_default();
        stats.calls = stats.calls.saturating_add(1);
        stats.prompt_tokens = stats.prompt_tokens.saturating_add(record.prompt_tokens);
        stats.cached_tokens = stats.cached_tokens.saturating_add(record.cached_tokens);
        stats.cache_creation_tokens = stats
            .cache_creation_tokens
            .saturating_add(record.cache_creation_tokens);
        stats.completion_tokens = stats
            .completion_tokens
            .saturating_add(record.completion_tokens);
    }

    /// Session totals are exactly the sum of the per-model stats, so they are
    /// derived once rather than accumulated alongside them.
    fn sum_totals(&mut self) {
        for stats in self.models.values() {
            self.total_calls = self.total_calls.saturating_add(stats.calls);
            self.total_prompt_tokens = self.total_prompt_tokens.saturating_add(stats.prompt_tokens);
            self.total_cached_tokens = self.total_cached_tokens.saturating_add(stats.cached_tokens);
            self.total_cache_creation_tokens = self
                .total_cache_creation_tokens
                .saturating_add(stats.cache_creation_tokens);
            self.total_completion_tokens = self
                .total_completion_tokens
                .saturating_add(stats.completion_tokens);
        }
    }
}

/// Lets serde leave out a `false` flag, so it is written only when set.
fn is_false(flag: &bool) -> bool {
    !flag
}

/// Returns a header value only when it is present, non-empty, and valid text.
fn nonempty_header<'a>(headers: &'a http::HeaderMap, name: &str) -> Option<&'a str> {
    headers
        .get(name)
        .filter(|value| !value.is_empty())
        .and_then(|v| v.to_str().ok())
}

fn routing_log_error(path: &Path, error: std::io::Error) -> ServerError {
    ServerError::new(format!(
        "failed to initialize routing log {}: {error}",
        path.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Missing, empty, and non-text origin headers remain absent even with a User-Agent.
    #[test]
    fn unusable_origin_is_not_inferred_from_user_agent() {
        for origin in [
            None,
            Some(http::HeaderValue::from_static("")),
            Some(http::HeaderValue::from_bytes(b"\xff").expect("header")),
        ] {
            let mut headers = http::HeaderMap::new();
            headers.insert(
                "user-agent",
                http::HeaderValue::from_static("codex-cli/1.0"),
            );
            if let Some(origin) = origin {
                headers.insert(ORIGIN_HEADER, origin);
            }
            let metadata = Metadata {
                http_headers: Some(headers),
                ..Default::default()
            };
            assert!(RoutingLogContext::from_metadata(&metadata).origin.is_none());
        }
        assert!(
            RoutingLogContext::from_metadata(&Metadata::default())
                .origin
                .is_none()
        );
    }

    #[test]
    fn a_delegated_call_records_its_agent_lineage() {
        let metadata = Metadata {
            session_id: Some("session-1".to_string()),
            turn_id: Some("turn-1".to_string()),
            agent_id: Some("child-7".to_string()),
            parent_agent_id: Some("thread-1".to_string()),
            is_subagent: true,
            is_delegated_work: true,
            agent_kind: Some("review".to_string()),
            agent_role: Some("reviewer".to_string()),
            ..Metadata::default()
        };
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("routing.jsonl");
        let mut log = RoutingLog::new(&path).expect("routing log");
        log.append(
            RoutingLogContext::from_metadata(&metadata).with_route("model-router", "passthrough"),
            "worker-model",
            None,
            &Usage::default(),
        )
        .expect("append");
        let line = fs::read_to_string(&path).expect("read log");
        let record: serde_json::Value = serde_json::from_str(line.trim()).expect("json record");

        assert_eq!(record["session_id"], "session-1");
        assert_eq!(record["turn_id"], "turn-1");
        assert_eq!(record["agent_id"], "child-7");
        assert_eq!(record["parent_agent_id"], "thread-1");
        assert_eq!(record["delegated"], true);
        assert_eq!(record["agent_kind"], "review");
        assert_eq!(record["agent_role"], "reviewer");
        assert!(record.get("subagent_identity_unsupported").is_none());
    }

    /// Only the requested session is counted, absent fields fall back to zero
    /// and `unknown`, and an unparseable line does not abort the scan.
    #[test]
    fn snapshot_counts_only_the_requested_session() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("routing.jsonl");
        fs::write(
            &path,
            concat!(
                r#"{"session_id":"a","model":"m1","fallback_reason":"unavailable","prompt_tokens":10,"completion_tokens":2}"#,
                "\n",
                r#"{"session_id":"b","model":"m1","prompt_tokens":99,"completion_tokens":99}"#,
                "\n",
                "not json\n",
                r#"{"session_id":"a","origin":"custom-agent","prompt_tokens":5}"#,
                "\n",
            ),
        )
        .expect("write log");

        let stats = snapshot(&path, "a").expect("read log").expect("session a");
        assert_eq!(stats.total_calls, 2);
        assert_eq!(stats.total_prompt_tokens, 15);
        assert_eq!(stats.total_completion_tokens, 2);
        assert_eq!(stats.models["m1"].calls, 1);
        assert_eq!(stats.models["unknown"].prompt_tokens, 5);
        assert!(snapshot(&path, "missing").expect("read log").is_none());
    }
}
