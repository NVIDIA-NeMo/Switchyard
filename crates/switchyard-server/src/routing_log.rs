// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Durable per-request routing records and session snapshots.

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use humantime::format_rfc3339_millis;
use libsy::Usage;
use serde::Serialize;
use serde_json::Value;

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
            ts: format_rfc3339_millis(SystemTime::now()).to_string(),
            task: context.task,
            trial_id: context.trial_id,
            session_id: context.session_id,
            model,
            tier: tier.unwrap_or(""),
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
    let mut reader = BufReader::new(fs::File::open(path)?);
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
        let Ok(record) = serde_json::from_slice::<Value>(&line) else {
            continue;
        };
        snapshot.add_record(&record, session_id);
    }
    Ok((snapshot.total_calls > 0).then_some(snapshot))
}

/// Request headers retained until terminal usage is available.
#[derive(Clone, Debug)]
pub(crate) struct RoutingLogContext {
    task: Option<String>,
    trial_id: Option<String>,
    session_id: Option<String>,
}

impl RoutingLogContext {
    pub(crate) fn from_headers(headers: &BTreeMap<String, String>) -> Self {
        Self {
            task: nonempty_header(headers, TASK_HEADER),
            trial_id: nonempty_header(headers, TRIAL_ID_HEADER),
            session_id: nonempty_header(headers, SESSION_ID_HEADER),
        }
    }
}

#[derive(Serialize)]
struct RoutingRecord<'a> {
    ts: String,
    task: Option<String>,
    trial_id: Option<String>,
    session_id: Option<String>,
    model: &'a str,
    tier: &'a str,
    prompt_tokens: u64,
    cached_tokens: u64,
    cache_creation_tokens: u64,
    completion_tokens: u64,
    reasoning_tokens: u64,
    total_tokens: u64,
}

/// Session totals returned by the routing stats endpoint.
#[derive(Debug, Eq, PartialEq, Serialize)]
pub(crate) struct SessionStatsSnapshot {
    session_id: String,
    total_calls: u64,
    total_prompt_tokens: u64,
    total_cached_tokens: u64,
    total_cache_creation_tokens: u64,
    total_completion_tokens: u64,
    models: BTreeMap<String, SessionModelStats>,
}

#[derive(Debug, Default, Eq, PartialEq, Serialize)]
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

    fn add_record(&mut self, record: &Value, session_id: &str) {
        let Some(record) = record.as_object() else {
            return;
        };
        if record.get("session_id").and_then(Value::as_str) != Some(session_id) {
            return;
        }
        let model = record
            .get("model")
            .and_then(Value::as_str)
            .filter(|model| !model.is_empty())
            .unwrap_or("unknown")
            .to_string();
        let stats = self.models.entry(model).or_default();
        stats.calls = stats.calls.saturating_add(1);
        self.total_calls = self.total_calls.saturating_add(1);

        add_counter(
            record,
            "prompt_tokens",
            &mut stats.prompt_tokens,
            &mut self.total_prompt_tokens,
        );
        add_counter(
            record,
            "cached_tokens",
            &mut stats.cached_tokens,
            &mut self.total_cached_tokens,
        );
        add_counter(
            record,
            "cache_creation_tokens",
            &mut stats.cache_creation_tokens,
            &mut self.total_cache_creation_tokens,
        );
        add_counter(
            record,
            "completion_tokens",
            &mut stats.completion_tokens,
            &mut self.total_completion_tokens,
        );
    }
}

fn add_counter(
    record: &serde_json::Map<String, Value>,
    key: &str,
    model_total: &mut u64,
    session_total: &mut u64,
) {
    if let Some(value) = record.get(key).and_then(Value::as_u64) {
        *model_total = model_total.saturating_add(value);
        *session_total = session_total.saturating_add(value);
    }
}

fn nonempty_header(headers: &BTreeMap<String, String>, name: &str) -> Option<String> {
    headers.get(name).filter(|value| !value.is_empty()).cloned()
}

fn routing_log_error(path: &Path, error: std::io::Error) -> ServerError {
    ServerError::new(format!(
        "failed to initialize routing log {}: {error}",
        path.display()
    ))
}
