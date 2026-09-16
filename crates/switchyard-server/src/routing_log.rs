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
use switchyard_runner::RouteErrorSummary;

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

    pub(crate) fn append(
        &mut self,
        context: RoutingLogContext,
        model: &str,
        tier: Option<&str>,
        usage: &Usage,
        failure: Option<(&str, &RouteErrorSummary)>,
    ) -> std::io::Result<()> {
        let usage = token_usage(usage);
        let terminal_tier = tier
            .map(str::to_string)
            .or_else(|| context.vgr_served.clone())
            .unwrap_or_default();
        let (route, failure_kind, upstream_status) =
            failure.map_or((None, None, None), |(route, failure)| {
                (
                    Some(route),
                    Some(failure.kind.as_str()),
                    failure.upstream_status,
                )
            });
        let record = RoutingRecord {
            ts: format_rfc3339_millis(SystemTime::now()).to_string().into(),
            route_id: context.route_id.into(),
            algorithm: context.algorithm.into(),
            origin: context.origin.map(Cow::Owned),
            task: context.task.map(Cow::Owned),
            trial_id: context.trial_id.map(Cow::Owned),
            session_id: context.session_id.map(Cow::Owned),
            vgr_predicted: context.vgr_predicted.map(Cow::Owned),
            vgr_effective: context.vgr_effective.map(Cow::Owned),
            vgr_served: context.vgr_served.map(Cow::Owned),
            vgr_branch: context.vgr_branch.map(Cow::Owned),
            vgr_readiness_gate: context.vgr_readiness_gate.map(Cow::Owned),
            vgr_short_circuit: context.vgr_short_circuit.map(Cow::Owned),
            route: route.map(Cow::Borrowed),
            model: model.into(),
            tier: terminal_tier.into(),
            failure_kind: failure_kind.map(Cow::Borrowed),
            upstream_status,
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

/// Holds request metadata until route resolution attaches the durable route identity.
#[derive(Clone)]
pub(crate) struct RoutingLogContext {
    route_id: String,
    algorithm: String,
    origin: Option<String>,
    task: Option<String>,
    trial_id: Option<String>,
    session_id: Option<String>,
    vgr_predicted: Option<String>,
    vgr_effective: Option<String>,
    vgr_served: Option<String>,
    vgr_branch: Option<String>,
    vgr_readiness_gate: Option<String>,
    vgr_short_circuit: Option<String>,
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
            session_id: metadata.session_id.clone().or_else(|| {
                headers
                    .and_then(|headers| nonempty_header(headers, LEGACY_SESSION_ID_HEADER))
                    .map(str::to_string)
            }),
            vgr_predicted: None,
            vgr_effective: None,
            vgr_served: None,
            vgr_branch: None,
            vgr_readiness_gate: None,
            vgr_short_circuit: None,
        }
    }

    /// Adds the VGR labels propagated to the terminal response.
    pub(crate) fn with_response_metadata(mut self, metadata: Option<&Metadata>) -> Self {
        self.vgr_predicted = vgr_label(metadata, "switchyard.vgr.predicted");
        self.vgr_effective = vgr_label(metadata, "switchyard.vgr.effective");
        self.vgr_served = vgr_label(metadata, "switchyard.vgr.served");
        self.vgr_branch = vgr_label(metadata, "switchyard.vgr.branch");
        self.vgr_readiness_gate = vgr_label(metadata, "switchyard.vgr.readiness_gate");
        self.vgr_short_circuit = vgr_label(metadata, "switchyard.vgr.short_circuit");
        self
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
    vgr_predicted: Option<Cow<'a, str>>,
    #[serde(borrow)]
    vgr_effective: Option<Cow<'a, str>>,
    #[serde(borrow)]
    vgr_served: Option<Cow<'a, str>>,
    #[serde(borrow)]
    vgr_branch: Option<Cow<'a, str>>,
    #[serde(borrow)]
    vgr_readiness_gate: Option<Cow<'a, str>>,
    #[serde(borrow)]
    vgr_short_circuit: Option<Cow<'a, str>>,
    #[serde(borrow, skip_serializing_if = "Option::is_none")]
    route: Option<Cow<'a, str>>,
    model: Cow<'a, str>,
    tier: Cow<'a, str>,
    #[serde(borrow, skip_serializing_if = "Option::is_none")]
    failure_kind: Option<Cow<'a, str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    upstream_status: Option<u16>,
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

fn nonempty_header<'a>(headers: &'a http::HeaderMap, name: &str) -> Option<&'a str> {
    headers
        .get(name)
        .filter(|value| !value.is_empty())
        .and_then(|v| v.to_str().ok())
}

fn vgr_label(metadata: Option<&Metadata>, key: &str) -> Option<String> {
    metadata?
        .extra_metadata
        .as_ref()?
        .get(key)
        .filter(|value| !value.is_empty())
        .cloned()
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

    #[test]
    fn terminal_record_correlates_vgr_decision_with_request_and_usage() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("routing.jsonl");
        let request_metadata = Metadata {
            session_id: Some("session-1".to_string()),
            http_headers: Some(http::HeaderMap::from_iter([
                (
                    http::HeaderName::from_static(TASK_HEADER),
                    http::HeaderValue::from_static("task-1"),
                ),
                (
                    http::HeaderName::from_static(TRIAL_ID_HEADER),
                    http::HeaderValue::from_static("trial-1"),
                ),
            ])),
            ..Default::default()
        };
        let response_metadata = Metadata {
            extra_metadata: Some(BTreeMap::from([
                ("switchyard.vgr.predicted".to_string(), "local".to_string()),
                ("switchyard.vgr.effective".to_string(), "cloud".to_string()),
                ("switchyard.vgr.served".to_string(), "cloud".to_string()),
                ("switchyard.vgr.branch".to_string(), "checks".to_string()),
                (
                    "switchyard.vgr.readiness_gate".to_string(),
                    "secure_checker_missing".to_string(),
                ),
                (
                    "switchyard.vgr.short_circuit".to_string(),
                    "none".to_string(),
                ),
            ])),
            ..Default::default()
        };
        let context = RoutingLogContext::from_metadata(&request_metadata)
            .with_response_metadata(Some(&response_metadata));
        let mut log = RoutingLog::new(&path).expect("routing log");
        log.append(
            context,
            "cloud-model",
            None,
            &Usage {
                input_tokens: Some(10),
                output_tokens: Some(2),
                ..Default::default()
            },
            None,
        )
        .expect("append routing record");

        let record: serde_json::Value =
            serde_json::from_slice(&fs::read(path).expect("read routing log"))
                .expect("valid routing record");
        assert_eq!(record["task"], "task-1");
        assert_eq!(record["trial_id"], "trial-1");
        assert_eq!(record["session_id"], "session-1");
        assert_eq!(record["vgr_predicted"], "local");
        assert_eq!(record["vgr_effective"], "cloud");
        assert_eq!(record["vgr_served"], "cloud");
        assert_eq!(record["vgr_branch"], "checks");
        assert_eq!(record["vgr_readiness_gate"], "secure_checker_missing");
        assert_eq!(record["model"], "cloud-model");
        assert_eq!(record["tier"], "cloud");
        assert_eq!(record["total_tokens"], 12);
    }
}
