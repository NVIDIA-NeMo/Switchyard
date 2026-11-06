// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Self-delegation routing: the efficient model decides, and its cloud sentinel hands the turn
//! to the capable model.
//!
//! The efficient target (the student) is served as an ordinary model whose visible content
//! starts with one route line, `{"route":"local"}` or `{"route":"cloud"}`. A local reply is
//! served as it came. The cloud sentinel makes the route call the capable target with the
//! client's request, unchanged, and serve that reply behind the sentinel, so the transcript
//! carries the decision and the executed step together. Nothing else is parsed or rewritten.
//! Without a capable target (collection mode) nothing is delegated: a reply is kept when it
//! carries a route line and a step, the shape the exporter accepts, and asked again otherwise.
//! An optional reminder is appended to each teacher request there and kept out of the trace.
//!
//! Both upstream calls ask for buffered replies. The answer is converted to a stream when the
//! client asked for one. With `trace_jsonl` set, one JSON object per decision is appended for
//! training-data export; the trace never changes what is served. With
//! `coalesce_assistant_turns`, assistant steps a client replays as separate messages are
//! merged first, and the trace records the merged history.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Instant, SystemTime};

use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{Value, json};
use switchyard_protocol::{
    AggLlmResponse, Category, ContentBlock, LlmRequest, LlmResponse, Message, Metadata, ModelId,
    Request, Response, ResponseOutput, Role, StopReason, Usage, WireFormat,
};
use switchyard_translation::{append_note, encode_request};

use crate::core::algorithm::{Algorithm, Driver, RoutingOutcome};
use crate::{LibsyError, Result};

#[cfg(test)]
mod tests;

/// The cloud sentinel: the student's whole reply when it delegates, and the first line served
/// ahead of the capable model's step.
pub const CLOUD_SENTINEL: &str = "{\"route\":\"cloud\"}";

/// Schema version written on every trace record.
pub const TRACE_SCHEMA_VERSION: &str = "native-v1";

/// What to do with a student reply that is neither a step nor the cloud sentinel once the
/// retries are spent.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InvalidAction {
    /// Serve the last reply as it came.
    #[default]
    Serve,
    /// Ask the capable target instead. Needs a capable target.
    Escalate,
}

/// What to do when the student call fails.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StudentErrorAction {
    /// Return the error to the client.
    #[default]
    Error,
    /// Ask the capable target instead. Needs a capable target.
    Escalate,
}

/// Retry budget and final action for an invalid student reply.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct OnInvalid {
    /// How many times the student is asked again with the same request.
    pub retry: u32,
    /// What happens to the last invalid reply.
    pub action: InvalidAction,
}

impl Default for OnInvalid {
    fn default() -> Self {
        Self {
            retry: 1,
            action: InvalidAction::Serve,
        }
    }
}

/// Settings for [`SelfDelegation`].
#[derive(Clone, Debug)]
pub struct SelfDelegationConfig {
    /// Remove the route line from served text. The trace keeps it.
    pub hide_route: bool,
    /// Treat a sentinel with cosmetic damage (code fences, single quotes, trailing text) as
    /// cloud at serving. The trace labels it `cloud_lenient`.
    pub lenient_cloud: bool,
    /// Policy for a reply that is neither a step nor the sentinel.
    pub on_invalid: OnInvalid,
    /// Policy for a failed student call.
    pub on_student_error: StudentErrorAction,
    /// Collection mode only: a reminder appended to each teacher request as its last user
    /// message. The trace never records it.
    pub collection_note: Option<String>,
    /// Merge an assistant text message with the tool-call-only assistant messages that follow
    /// it before anything else happens, so a history a client replays as separate items reads
    /// as one message per step.
    pub coalesce_assistant_turns: bool,
    /// Append one JSON object per decision to this file.
    pub trace_jsonl: Option<PathBuf>,
}

impl Default for SelfDelegationConfig {
    fn default() -> Self {
        Self {
            hide_route: false,
            lenient_cloud: true,
            on_invalid: OnInvalid::default(),
            on_student_error: StudentErrorAction::Error,
            collection_note: None,
            coalesce_assistant_turns: false,
            trace_jsonl: None,
        }
    }
}

/// Serves the efficient target and delegates to the capable target on the cloud sentinel.
pub struct SelfDelegation {
    config: SelfDelegationConfig,
    trace: Option<TraceWriter>,
}

impl SelfDelegation {
    /// Builds the algorithm. The trace file is opened on the first write.
    pub fn new(config: SelfDelegationConfig) -> Self {
        let trace = config.trace_jsonl.clone().map(TraceWriter::new);
        Self { config, trace }
    }
}

/// What the gateway does with a student reply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Classification {
    /// Served as it came. Any reply with tool calls is local.
    Local,
    /// Exactly the sentinel: the capable target is called.
    Cloud,
    /// The sentinel with cosmetic damage: the capable target is called, the strict validator
    /// rejects the reply.
    CloudLenient,
    /// Neither a step nor the sentinel: the `on_invalid` policy applies.
    Invalid,
}

impl Classification {
    /// The trace label.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Cloud => "cloud",
            Self::CloudLenient => "cloud_lenient",
            Self::Invalid => "invalid",
        }
    }

    /// The label collapsed to the gateway's decision.
    pub fn decision(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Cloud | Self::CloudLenient => "cloud",
            Self::Invalid => "invalid",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RouteLine {
    Local,
    Cloud,
}

impl RouteLine {
    fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Cloud => "cloud",
        }
    }
}

/// A route line is a line that, trimmed, is a JSON object with the single key `route` whose
/// value is `local` or `cloud`.
fn route_line(line: &str) -> Option<RouteLine> {
    let value: Value = serde_json::from_str(line.trim()).ok()?;
    let object = value.as_object()?;
    if object.len() != 1 {
        return None;
    }
    match object.get("route")?.as_str()? {
        "local" => Some(RouteLine::Local),
        "cloud" => Some(RouteLine::Cloud),
        _ => None,
    }
}

/// Splits off the first line, without its line break.
fn split_first_line(text: &str) -> (&str, &str) {
    match text.split_once('\n') {
        Some((first, rest)) => (first.strip_suffix('\r').unwrap_or(first), rest),
        None => (text, ""),
    }
}

fn non_empty_lines(text: &str) -> impl Iterator<Item = &str> {
    text.lines().filter(|line| !line.trim().is_empty())
}

/// The lenient sentinel match: the sentinel behind code fences, single quotes, or spacing, or
/// the strict sentinel line with text after it.
fn lenient_cloud(text: &str) -> bool {
    let mut lines = non_empty_lines(text);
    let Some(first) = lines.next() else {
        return false;
    };
    if route_line(first) == Some(RouteLine::Cloud) {
        return lines.next().is_some();
    }
    non_empty_lines(text)
        .find(|line| !line.trim_start().starts_with("```"))
        .map(|line| {
            line.replace('\'', "\"")
                .chars()
                .filter(|character| !character.is_whitespace())
                .collect::<String>()
        })
        .is_some_and(|line| route_line(&line) == Some(RouteLine::Cloud))
}

/// Classifies a student reply from its concatenated text and tool-call count.
pub fn classify(text: &str, tool_calls: usize, lenient: bool) -> Classification {
    if tool_calls > 0 {
        return Classification::Local;
    }
    if route_line(text) == Some(RouteLine::Cloud) {
        return Classification::Cloud;
    }
    if lenient && lenient_cloud(text) {
        return Classification::CloudLenient;
    }
    let mut lines = non_empty_lines(text);
    if lines
        .next()
        .is_some_and(|first| route_line(first) == Some(RouteLine::Local))
    {
        let rest: Vec<&str> = lines.collect();
        if !rest.is_empty() && !rest.iter().any(|line| route_line(line).is_some()) {
            return Classification::Local;
        }
    }
    Classification::Invalid
}

/// The strict serving-profile validator, recorded in the trace and applied again at export.
pub fn is_valid(text: &str, tool_calls: usize) -> bool {
    let (first, rest) = split_first_line(text.trim_start());
    let Some(route) = route_line(first) else {
        return false;
    };
    if rest.lines().any(|line| route_line(line).is_some()) {
        return false;
    }
    match route {
        RouteLine::Cloud => tool_calls == 0 && rest.trim().is_empty(),
        RouteLine::Local => tool_calls > 0 || !rest.trim().is_empty(),
    }
}

/// The collection profile for teacher replies: a route line first, a non-empty step after it
/// (a tool call or text), and no other route line. Returns the label when the reply passes.
pub fn collection_route(text: &str, tool_calls: usize) -> Option<&'static str> {
    let (first, rest) = split_first_line(text.trim_start());
    let route = route_line(first)?;
    if rest.lines().any(|line| route_line(line).is_some()) {
        return None;
    }
    (tool_calls > 0 || !rest.trim().is_empty()).then_some(route.as_str())
}

/// Drops leading whitespace, a leading route line, and one line break after it. Text without a
/// leading route line is returned unchanged. The flag reports whether a line was dropped.
pub fn strip_route_line(text: &str) -> (&str, bool) {
    let (first, rest) = split_first_line(text.trim_start());
    if route_line(first).is_some() {
        (rest, true)
    } else {
        (text, false)
    }
}

/// Builds the served text for a delegated step: the canonical sentinel ahead of the capable
/// model's text, with a route line the capable model imitated removed. The flag reports that
/// removal.
pub fn merge_cloud(strong_text: &str, hide_route: bool) -> (String, bool) {
    let (step, imitated) = strip_route_line(strong_text);
    let served = if hide_route {
        step.to_string()
    } else if step.is_empty() {
        CLOUD_SENTINEL.to_string()
    } else {
        format!("{CLOUD_SENTINEL}\n{step}")
    };
    (served, imitated)
}

/// The first output's text and its tool calls.
fn text_and_tools(response: &AggLlmResponse) -> (String, Vec<Value>) {
    let mut text = String::new();
    let mut tools = Vec::new();
    for block in response
        .first_output()
        .into_iter()
        .flat_map(|output| &output.content)
    {
        match block {
            ContentBlock::Text { text: part } => text.push_str(part),
            ContentBlock::ToolCall(call) => {
                let arguments = match &call.arguments {
                    Value::String(arguments) => arguments.clone(),
                    other => other.to_string(),
                };
                tools.push(json!({
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": arguments},
                }));
            }
            _ => {}
        }
    }
    (text, tools)
}

/// Replaces the first output's text blocks with one block holding `text`. Empty text removes
/// them: an empty text block breaks the Anthropic client on its next turn.
fn set_text(response: &mut AggLlmResponse, text: &str) {
    if response.outputs.is_empty() {
        if text.is_empty() {
            return;
        }
        response.outputs.push(ResponseOutput {
            role: Role::Assistant,
            content: Vec::new(),
            url_citations: Vec::new(),
            stop_reason: None,
        });
    }
    let content = &mut response.outputs[0].content;
    let is_text = |block: &ContentBlock| matches!(block, ContentBlock::Text { .. });
    let position = content.iter().position(is_text).unwrap_or_else(|| {
        content
            .iter()
            .take_while(|block| matches!(block, ContentBlock::Reasoning { .. }))
            .count()
    });
    content.retain(|block| !is_text(block));
    if !text.is_empty() {
        content.insert(
            position.min(content.len()),
            ContentBlock::Text {
                text: text.to_string(),
            },
        );
    }
}

fn add_usage(total: &mut Usage, more: &Usage) {
    fn add(total: &mut Option<u64>, more: Option<u64>) {
        if let Some(more) = more {
            *total = Some(total.unwrap_or(0) + more);
        }
    }
    add(&mut total.input_tokens, more.input_tokens);
    add(&mut total.output_tokens, more.output_tokens);
    add(&mut total.total_tokens, more.total_tokens);
    add(&mut total.reasoning_tokens, more.reasoning_tokens);
    if let Some(cost) = more.provider_cost {
        total.provider_cost = Some(total.provider_cost.unwrap_or(0.0) + cost);
    }
    if let Some(cache) = &more.cache {
        let totals = total.cache.get_or_insert_with(Default::default);
        add(&mut totals.cached_input_tokens, cache.cached_input_tokens);
        add(
            &mut totals.cache_creation_input_tokens,
            cache.cache_creation_input_tokens,
        );
        add(
            &mut totals.cache_creation_1h_input_tokens,
            cache.cache_creation_1h_input_tokens,
        );
    }
}

/// The OpenAI Chat `finish_reason` spelling of the first output's stop reason.
fn finish_reason(response: &AggLlmResponse) -> Value {
    match response
        .first_output()
        .and_then(|output| output.stop_reason)
    {
        Some(StopReason::EndTurn) => json!("stop"),
        Some(StopReason::MaxTokens) => json!("length"),
        Some(StopReason::ToolUse) => json!("tool_calls"),
        Some(StopReason::ContentFilter) => json!("content_filter"),
        Some(StopReason::Error) => json!("error"),
        Some(StopReason::Unknown) => json!("unknown"),
        None => Value::Null,
    }
}

fn usage_json(usage: &Usage) -> Value {
    json!({"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens})
}

/// Asks the upstream for a buffered reply. The client replays a preserved same-format body
/// verbatim, so its `stream` field is cleared as well.
fn clear_stream(request: &mut LlmRequest) {
    request.stream = false;
    for body in request.preservation.requests.values_mut() {
        if let Some(object) = body.as_object_mut() {
            object.remove("stream");
            object.remove("stream_options");
        }
    }
}

fn is_blank_text(block: &ContentBlock) -> bool {
    matches!(block, ContentBlock::Text { text } if text.trim().is_empty())
}

fn has_visible_text(message: &Message) -> bool {
    message
        .content
        .iter()
        .any(|block| matches!(block, ContentBlock::Text { .. }) && !is_blank_text(block))
}

fn has_tool_calls(message: &Message) -> bool {
    message
        .content
        .iter()
        .any(|block| matches!(block, ContentBlock::ToolCall(_)))
}

/// Merges an assistant message that has text and no tool calls with the assistant messages
/// that directly follow it and have tool calls but no text, so a history a client replays as
/// separate items reads as one message per step, the shape the model produced. Every other
/// message is left alone. Preserved provider bodies still hold the split history, so they are
/// dropped when a merge happens and encoding rebuilds the request. Returns whether anything
/// was merged.
pub fn coalesce_assistant_turns(request: &mut LlmRequest) -> bool {
    let mut merged = false;
    let mut messages: Vec<Message> = Vec::with_capacity(request.messages.len());
    // The text-only assistant message that opened the current step, while its run lasts.
    let mut head: Option<usize> = None;
    for message in std::mem::take(&mut request.messages) {
        if message.role != Role::Assistant {
            head = None;
        } else if let Some(index) = head
            && has_tool_calls(&message)
            && !has_visible_text(&message)
        {
            messages[index].content.extend(
                message
                    .content
                    .into_iter()
                    .filter(|block| !is_blank_text(block)),
            );
            merged = true;
            continue;
        } else {
            head =
                (has_visible_text(&message) && !has_tool_calls(&message)).then_some(messages.len());
        }
        messages.push(message);
    }
    request.messages = messages;
    if merged {
        request.preservation.requests.clear();
    }
    merged
}

/// A buffered upstream reply with the envelope fields the served response keeps.
struct Buffered {
    agg: AggLlmResponse,
    metadata: Option<Metadata>,
    upstream_headers: http::HeaderMap,
}

impl Buffered {
    fn into_response(self, streaming: bool) -> Response {
        let llm_response = if streaming {
            LlmResponse::Stream(self.agg.into_stream())
        } else {
            LlmResponse::Agg(self.agg)
        };
        Response {
            llm_response,
            metadata: self.metadata,
            upstream_headers: self.upstream_headers,
        }
    }
}

/// Calls one target and buffers its reply. Client errors come back here so they can be traced
/// and the configured policy applied.
async fn call_buffered(driver: &Driver, request: Request, target: &ModelId) -> Result<Buffered> {
    let response = driver
        .call_model_with_error_recovery(request, vec![target.clone()], true)
        .await?;
    let Response {
        llm_response,
        metadata,
        upstream_headers,
    } = response;
    let agg = llm_response
        .into_agg()
        .await
        .map_err(|source| LibsyError::client_call(target.clone(), source))?;
    Ok(Buffered {
        agg,
        metadata,
        upstream_headers,
    })
}

/// One student call as recorded in the trace.
struct Attempt {
    content: String,
    tool_calls: Vec<Value>,
    finish_reason: Value,
    usage: Usage,
    latency_ms: u128,
    classification: Option<Classification>,
    valid: bool,
    /// The collection-profile check; `None` in serving mode and for a failed call.
    collection_valid: Option<bool>,
    error: Option<String>,
}

impl Attempt {
    fn to_json(&self) -> Value {
        json!({
            "content": self.content,
            "tool_calls": self.tool_calls,
            "finish_reason": self.finish_reason,
            "usage": usage_json(&self.usage),
            "latency_ms": self.latency_ms,
            "classification": self.classification.map(Classification::as_str),
            "valid": self.valid,
            "collection_valid": self.collection_valid,
            "error": self.error,
        })
    }
}

/// The capable model's call as recorded in the trace.
struct StrongRecord {
    target: ModelId,
    content: String,
    tool_calls: Vec<Value>,
    finish_reason: Value,
    usage: Usage,
    latency_ms: u128,
    imitated_route_line: bool,
    fallback: bool,
    error: Option<String>,
}

impl StrongRecord {
    fn to_json(&self) -> Value {
        json!({
            "target": self.target.as_str(),
            "content": self.content,
            "tool_calls": self.tool_calls,
            "finish_reason": self.finish_reason,
            "usage": usage_json(&self.usage),
            "latency_ms": self.latency_ms,
            "imitated_route_line": self.imitated_route_line,
            "fallback": self.fallback,
            "error": self.error,
        })
    }
}

/// What the client received.
struct Served {
    by: &'static str,
    content: String,
    tool_calls: usize,
    stripped_route_line: bool,
}

/// Everything one trace record is built from.
struct Decision<'a> {
    neutral: &'a Request,
    /// The neutral request as sent upstream: buffered.
    upstream: &'a LlmRequest,
    efficient: &'a ModelId,
    capable: Option<&'a ModelId>,
    attempts: &'a [Attempt],
    decision: &'static str,
    strong: Option<&'a StrongRecord>,
    served: Option<&'a Served>,
}

/// Appends JSON lines to one file. Writes are serialized; a failed write is logged and never
/// fails the request.
struct TraceWriter {
    path: PathBuf,
    file: Mutex<Option<File>>,
}

impl TraceWriter {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            file: Mutex::new(None),
        }
    }

    fn write(&self, record: &Value) {
        let mut line = record.to_string();
        line.push('\n');
        let mut guard = self.file.lock();
        if guard.is_none() {
            match OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.path)
            {
                Ok(file) => *guard = Some(file),
                Err(error) => {
                    tracing::warn!(
                        path = %self.path.display(),
                        %error,
                        "self_delegation could not open the trace file"
                    );
                    return;
                }
            }
        }
        if let Some(file) = guard.as_mut()
            && let Err(error) = file.write_all(line.as_bytes())
        {
            *guard = None;
            tracing::warn!(
                path = %self.path.display(),
                %error,
                "self_delegation could not write a trace record"
            );
        }
    }
}

impl SelfDelegation {
    /// The request body as the student engine receives it: the neutral request in the OpenAI
    /// Chat shape with the provider model stamped.
    fn request_body(driver: &Driver, request: &LlmRequest, efficient: &ModelId) -> Value {
        match encode_request(request, WireFormat::OpenAiChat) {
            Ok(mut body) => {
                if let Some(object) = body.as_object_mut() {
                    object.insert(
                        "model".to_string(),
                        json!(driver.provider_model(efficient).as_str()),
                    );
                }
                body
            }
            Err(error) => json!({"error": error.to_string()}),
        }
    }

    fn trace(&self, driver: &Driver, decision: &Decision<'_>) {
        let Some(trace) = &self.trace else {
            return;
        };
        let request = &decision.neutral.llm_request;
        let metadata = decision.neutral.metadata.as_ref();
        let record = json!({
            "schema_version": TRACE_SCHEMA_VERSION,
            "ts": humantime::format_rfc3339_millis(SystemTime::now()).to_string(),
            "route": request.model,
            "mode": if decision.capable.is_some() { "serve" } else { "collect" },
            "hide_route": self.config.hide_route,
            "lenient_cloud": self.config.lenient_cloud,
            "session_id": metadata.and_then(|metadata| metadata.session_id.as_deref()),
            "agent_id": metadata.and_then(|metadata| metadata.agent_id.as_deref()),
            "turn_id": metadata.and_then(|metadata| metadata.turn_id.as_deref()),
            "task_id": metadata.and_then(|metadata| metadata.task_id.as_deref()),
            "correlation_id": metadata.and_then(|metadata| metadata.correlation_id.as_deref()),
            "wire_format": metadata.and_then(|metadata| metadata.wire_format).map(WireFormat::as_str),
            "is_compaction": request.is_compaction,
            "has_compaction": request.has_compaction,
            "request": Self::request_body(driver, decision.upstream, decision.efficient),
            "student": {
                "target": decision.efficient.as_str(),
                "attempts": decision.attempts.iter().map(Attempt::to_json).collect::<Vec<_>>(),
            },
            "decision": decision.decision,
            "strong": decision.strong.map(StrongRecord::to_json),
            "served": decision.served.map(|served| json!({
                "by": served.by,
                "content": served.content,
                "tool_calls": served.tool_calls,
                "stripped_route_line": served.stripped_route_line,
            })),
        });
        trace.write(&record);
    }
}

#[async_trait::async_trait]
impl Algorithm for SelfDelegation {
    fn name(&self) -> &str {
        "self_delegation"
    }

    async fn route(self: Arc<Self>, driver: Driver, request: Request) -> Result<RoutingOutcome> {
        let efficient = driver.first_model_for(&Category::Efficient)?.clone();
        let capable = driver.models_for(&Category::Capable).first().cloned();
        let streaming = request.llm_request.stream;
        // The request as received, with split assistant steps merged when configured: no
        // prompt is added here. The upstream copy only asks for buffered replies.
        let mut neutral = request;
        if self.config.coalesce_assistant_turns {
            coalesce_assistant_turns(&mut neutral.llm_request);
        }
        let mut upstream = neutral.clone();
        clear_stream(&mut upstream.llm_request);

        let mut attempts = Vec::new();
        let mut student: Option<Buffered> = None;
        let mut student_usage = Usage::default();
        let mut student_error = None;
        let mut classification = None;
        // Collection keeps only replies the exporter accepts: a route line and a step. A reply
        // without them is a wasted trajectory, so it is asked again like an invalid one.
        let collecting = capable.is_none();
        let mut collection_label = None;
        // The teacher sees the reminder as the last user message. The trace records
        // `upstream`, which never carries it, so the student is never trained on it.
        let mut student_request = upstream.clone();
        if collecting && let Some(note) = &self.config.collection_note {
            append_note(&mut student_request.llm_request, note);
        }
        for _ in 0..=self.config.on_invalid.retry {
            let started = Instant::now();
            match call_buffered(&driver, student_request.clone(), &efficient).await {
                Ok(buffered) => {
                    let (text, tool_calls) = text_and_tools(&buffered.agg);
                    let class = classify(&text, tool_calls.len(), self.config.lenient_cloud);
                    let collection = collecting.then(|| collection_route(&text, tool_calls.len()));
                    attempts.push(Attempt {
                        valid: is_valid(&text, tool_calls.len()),
                        collection_valid: collection.map(|label| label.is_some()),
                        content: text,
                        tool_calls,
                        finish_reason: finish_reason(&buffered.agg),
                        usage: buffered.agg.usage.clone(),
                        latency_ms: started.elapsed().as_millis(),
                        classification: Some(class),
                        error: None,
                    });
                    add_usage(&mut student_usage, &buffered.agg.usage);
                    student = Some(buffered);
                    classification = Some(class);
                    collection_label = collection.flatten();
                    let accepted = match collection {
                        Some(label) => label.is_some(),
                        None => class != Classification::Invalid,
                    };
                    if accepted {
                        break;
                    }
                }
                Err(error) => {
                    tracing::warn!(target = %efficient, %error, "self_delegation student call failed");
                    attempts.push(Attempt {
                        content: String::new(),
                        tool_calls: Vec::new(),
                        finish_reason: Value::Null,
                        usage: Usage::default(),
                        latency_ms: started.elapsed().as_millis(),
                        classification: None,
                        valid: false,
                        collection_valid: None,
                        error: Some(error.to_string()),
                    });
                    student_error = Some(error);
                    break;
                }
            }
        }

        let decision = if collecting {
            collection_label.unwrap_or("invalid")
        } else {
            classification.map_or("invalid", Classification::decision)
        };
        let (escalate, fallback) = match (&student_error, classification) {
            (Some(_), _) => (
                self.config.on_student_error == StudentErrorAction::Escalate,
                true,
            ),
            (None, Some(Classification::Invalid) | None) => (
                self.config.on_invalid.action == InvalidAction::Escalate,
                true,
            ),
            (None, Some(Classification::Cloud | Classification::CloudLenient)) => (true, false),
            (None, Some(Classification::Local)) => (false, false),
        };
        let strong_target = capable.as_ref().filter(|_| escalate);
        let mut record = Decision {
            neutral: &neutral,
            upstream: &upstream.llm_request,
            efficient: &efficient,
            capable: capable.as_ref(),
            attempts: &attempts,
            decision,
            strong: None,
            served: None,
        };

        let Some(capable_target) = strong_target else {
            if let Some(error) = student_error {
                self.trace(&driver, &record);
                return Err(error);
            }
            let Some(mut buffered) = student else {
                return Err(LibsyError::AlgorithmError {
                    message: "self_delegation made no student call".to_string(),
                });
            };
            let (text, tool_calls) = text_and_tools(&buffered.agg);
            let mut stripped = false;
            let mut content = text;
            if self.config.hide_route {
                let (rest, removed) = strip_route_line(&content);
                stripped = removed;
                content = rest.to_string();
                set_text(&mut buffered.agg, &content);
            }
            let served = Served {
                by: "student",
                content,
                tool_calls: tool_calls.len(),
                stripped_route_line: stripped,
            };
            record.served = Some(&served);
            self.trace(&driver, &record);
            driver.set_evidence(json!({"source": "self_delegation", "verdict": decision}));
            tracing::info!(target = %efficient, decision, "self_delegation served the student");
            return Ok(RoutingOutcome::answered(
                efficient,
                neutral,
                buffered.into_response(streaming),
            ));
        };

        let started = Instant::now();
        let mut strong = StrongRecord {
            target: capable_target.clone(),
            content: String::new(),
            tool_calls: Vec::new(),
            finish_reason: Value::Null,
            usage: Usage::default(),
            latency_ms: 0,
            imitated_route_line: false,
            fallback,
            error: None,
        };
        let mut buffered = match call_buffered(&driver, upstream.clone(), capable_target).await {
            Ok(buffered) => buffered,
            Err(error) => {
                tracing::warn!(target = %capable_target, %error, "self_delegation strong call failed");
                strong.latency_ms = started.elapsed().as_millis();
                strong.error = Some(error.to_string());
                record.strong = Some(&strong);
                self.trace(&driver, &record);
                return Err(error);
            }
        };
        strong.latency_ms = started.elapsed().as_millis();
        let (text, tool_calls) = text_and_tools(&buffered.agg);
        let (served_text, imitated) = merge_cloud(&text, self.config.hide_route);
        strong.content = text;
        strong.finish_reason = finish_reason(&buffered.agg);
        strong.usage = buffered.agg.usage.clone();
        strong.imitated_route_line = imitated;
        set_text(&mut buffered.agg, &served_text);
        add_usage(&mut buffered.agg.usage, &student_usage);
        let served = Served {
            by: "strong",
            content: served_text,
            tool_calls: tool_calls.len(),
            stripped_route_line: self.config.hide_route,
        };
        strong.tool_calls = tool_calls;
        record.strong = Some(&strong);
        record.served = Some(&served);
        self.trace(&driver, &record);
        let mut evidence = json!({"source": "self_delegation", "verdict": decision});
        if fallback {
            evidence["trigger"] = json!("fallback");
        }
        driver.set_evidence(evidence);
        tracing::info!(
            target = %capable_target,
            decision,
            fallback,
            "self_delegation served the capable model"
        );
        Ok(RoutingOutcome::answered(
            capable_target.clone(),
            neutral,
            buffered.into_response(streaming),
        ))
    }
}
