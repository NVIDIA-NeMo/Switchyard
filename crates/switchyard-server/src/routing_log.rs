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
use switchyard_protocol::Usage;

use crate::usage_metrics::token_usage;
use crate::{ServerError, ServerResult};

const SESSION_ID_HEADER: &str = "proxy_x_session_id";
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
    ) -> std::io::Result<()> {
        let usage = token_usage(usage);
        let record = RoutingRecord {
            ts: format_rfc3339_millis(SystemTime::now()).to_string().into(),
            task: context.task.map(Cow::Owned),
            trial_id: context.trial_id.map(Cow::Owned),
            session_id: context.session_id.map(Cow::Owned),
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

/// Request headers retained until terminal usage is available.
#[derive(Clone)]
pub(crate) struct RoutingLogContext {
    task: Option<String>,
    trial_id: Option<String>,
    session_id: Option<String>,
}

impl RoutingLogContext {
    pub(crate) fn from_headers(headers: &http::HeaderMap) -> Self {
        Self {
            task: nonempty_header(headers, TASK_HEADER).map(|s| s.to_string()),
            trial_id: nonempty_header(headers, TRIAL_ID_HEADER).map(|s| s.to_string()),
            session_id: nonempty_header(headers, SESSION_ID_HEADER).map(|s| s.to_string()),
        }
    }
}

/// One appended routing record, and the read schema [`snapshot`] parses back,
/// so the written and expected shapes cannot drift apart. Missing fields
/// default so a record from an older schema still contributes what it has.
#[derive(Default, Deserialize, Serialize)]
#[serde(default)]
struct RoutingRecord<'a> {
    ts: Cow<'a, str>,
    #[serde(borrow)]
    task: Option<Cow<'a, str>>,
    #[serde(borrow)]
    trial_id: Option<Cow<'a, str>>,
    #[serde(borrow)]
    session_id: Option<Cow<'a, str>>,
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
    models: BTreeMap<String, SessionModelStats>,
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
        let stats = self.models.entry(model.to_string()).or_default();
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

fn routing_log_error(path: &Path, error: std::io::Error) -> ServerError {
    ServerError::new(format!(
        "failed to initialize routing log {}: {error}",
        path.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only the requested session is counted, absent fields fall back to zero
    /// and `unknown`, and an unparseable line does not abort the scan.
    #[test]
    fn snapshot_counts_only_the_requested_session() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("routing.jsonl");
        fs::write(
            &path,
            concat!(
                r#"{"session_id":"a","model":"m1","prompt_tokens":10,"completion_tokens":2}"#,
                "\n",
                r#"{"session_id":"b","model":"m1","prompt_tokens":99,"completion_tokens":99}"#,
                "\n",
                "not json\n",
                r#"{"session_id":"a","prompt_tokens":5}"#,
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
