// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Long-lived routing helper for the Claude Code mod.
//!
//! Claude's process API closes a child's stdin after one write, so requests arrive as
//! `req-<id>.json` files in a private directory that the helper creates and announces
//! on stdout. Each answer is one JSON line on stdout. One `Runner` serves every request,
//! so routing state such as the capable-tier hold lasts as long as the helper.

use std::{
    collections::HashMap,
    env,
    fs::{self, DirBuilder},
    io::Write,
    os::unix::{fs::DirBuilderExt, process::parent_id},
    path::{Path, PathBuf},
    process::ExitCode,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use switchyard_protocol::{
    ContentBlock, LlmClientError, LlmRequest, LlmResponse, Metadata, ModelId, Request, Response,
    RoutedLlmClient, text_response,
};
use switchyard_runner::Runner;
use tokio::{
    signal::unix::{SignalKind, signal},
    time::{sleep, timeout},
};

const POLL: Duration = Duration::from_millis(25);
/// Polls a request file may fail to parse, since its writer may not have finished.
const PARSE_RETRIES: u32 = 40;

#[derive(Deserialize)]
struct RoutingRequest {
    session_id: String,
    messages: Value,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    match serve().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn emit(line: &Value) -> Result<(), String> {
    let mut out = std::io::stdout().lock();
    writeln!(out, "{line}")
        .and_then(|()| out.flush())
        .map_err(|error| error.to_string())
}

async fn serve() -> Result<(), String> {
    // Sampled before any other work, so only a parent that dies this early goes unnoticed.
    let parent = parent_id();
    let config = env::args().nth(1).ok_or("Expected a TOML config path")?;
    let source = fs::read_to_string(config).map_err(|error| error.to_string())?;
    let runner = Runner::from_toml(&source).map_err(|error| error.to_string())?;
    let mut terminate = signal(SignalKind::terminate()).map_err(|error| error.to_string())?;
    let mut hangup = signal(SignalKind::hangup()).map_err(|error| error.to_string())?;

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos());
    let dir = PathBuf::from(env::var_os("TMPDIR").unwrap_or("/tmp".into()))
        .join(format!("switchyard-claude-{}-{nanos}", std::process::id()));
    DirBuilder::new()
        .mode(0o700)
        .create(&dir)
        .map_err(|error| error.to_string())?;
    let result = async {
        emit(&json!({ "dir": dir }))?;
        let mut failed_parses = HashMap::new();
        loop {
            let Some(requests) = pending(&dir).filter(|_| parent_id() == parent) else {
                return Ok(());
            };
            for (id, path) in requests {
                let Some(request) = read_request(&path, id, &mut failed_parses)? else {
                    // Preserve decision order while the earliest file is still being written.
                    break;
                };
                let answer = tokio::select! {
                    answer = decide(&runner, &request.session_id, request.messages, Arc::new(NativeClassifier {
                        dir: dir.clone(), request_id: id, next_call: AtomicU64::new(0),
                        cache_turn: runner.route("switchyard").is_some_and(|route| route.algorithm_name() == "composite"),
                    })) => answer,
                    () = cancelled(&dir, id) => Err("Routing cancelled".into()),
                    // A decision can wait on the network, so it is cancelled when the helper
                    // is told to stop or loses its parent.
                    () = abandoned(&dir, parent) => return Ok(()),
                    _ = terminate.recv() => return Ok(()),
                    _ = hangup.recv() => return Ok(()),
                };
                emit(&match answer {
                    Ok(decision) => json!({ "id": id, "target": decision["target"], "model": decision["model"], "evidence": decision["evidence"], "algorithm": decision["algorithm"] }),
                    Err(error) => json!({ "id": id, "error": error }),
                })?;
            }
            tokio::select! {
                () = sleep(POLL) => {}
                _ = terminate.recv() => return Ok(()),
                _ = hangup.recv() => return Ok(()),
            }
        }
    }
    .await;
    let _ = fs::remove_dir_all(&dir);
    result
}

async fn cancelled(dir: &Path, id: u64) {
    let path = dir.join(format!("cancel-{id}"));
    while !path.exists() {
        sleep(POLL).await;
    }
    let _ = fs::remove_file(path);
}

/// Resolves once the parent has died (the helper is reparented) or the mod has removed
/// the directory to stop the helper.
async fn abandoned(dir: &Path, parent: u32) {
    while parent_id() == parent && dir.is_dir() {
        sleep(POLL).await;
    }
}

/// Request files in id order. `None` means the directory is gone.
fn pending(dir: &Path) -> Option<Vec<(u64, PathBuf)>> {
    let mut found: Vec<_> = fs::read_dir(dir)
        .ok()?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let id = path
                .file_name()?
                .to_str()?
                .strip_prefix("req-")?
                .strip_suffix(".json")?
                .parse()
                .ok()?;
            Some((id, path))
        })
        .collect();
    found.sort();
    Some(found)
}

/// Consumes a request file. `Ok(None)` means the file is not readable yet.
fn read_request(
    path: &Path,
    id: u64,
    failed_parses: &mut HashMap<u64, u32>,
) -> Result<Option<RoutingRequest>, String> {
    let parsed = fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str::<RoutingRequest>(&text).ok())
        .filter(|request| !request.session_id.is_empty());
    if let Some(request) = parsed {
        failed_parses.remove(&id);
        let _ = fs::remove_file(path);
        return Ok(Some(request));
    }
    let attempts = failed_parses.entry(id).or_default();
    *attempts += 1;
    if *attempts >= PARSE_RETRIES {
        failed_parses.remove(&id);
        let _ = fs::remove_file(path);
        emit(&json!({ "id": id, "error": "Invalid routing request" }))?;
    }
    Ok(None)
}

async fn decide(
    runner: &Runner,
    session: &str,
    messages: Value,
    client: Arc<NativeClassifier>,
) -> Result<Value, String> {
    let messages = serde_json::from_value(messages).map_err(|_| "Invalid routing messages")?;
    let route = runner
        .route("switchyard")
        .ok_or("Missing switchyard route")?;
    let request = Request {
        llm_request: LlmRequest {
            model: Some("switchyard".into()),
            messages,
            ..Default::default()
        },
        metadata: Some(Metadata {
            session_id: Some(session.to_owned()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let mut expected = request.llm_request.clone();
    let outcome = route
        .decide_with_client(request, client)
        .await
        .map_err(|error| error.execution_error_summary().kind.as_str().to_owned())?;
    expected.model = outcome.request.llm_request.model.clone();
    if outcome.response.is_some() || outcome.request.llm_request != expected {
        return Err("Routing must only select a model".into());
    }
    let decision = runner
        .describe_decision(&ModelId::from("switchyard"), &outcome)
        .ok_or("Routing selected an unresolved target")?;
    Ok(
        json!({ "target": decision.selected.target, "model": decision.selected.model,
        "evidence": outcome.metadata.and_then(|m| m.evidence), "algorithm": route.algorithm_name() }),
    )
}

/// Routes judge calls through the mod, which owns Claude's authenticated client.
struct NativeClassifier {
    dir: PathBuf,
    request_id: u64,
    next_call: AtomicU64,
    cache_turn: bool,
}

#[derive(Deserialize)]
struct Reply {
    text: Option<String>,
    error: Option<String>,
}

#[async_trait]
impl RoutedLlmClient for NativeClassifier {
    async fn call(&self, request: Request) -> Result<Response, LlmClientError> {
        let call = self.next_call.fetch_add(1, Ordering::Relaxed);
        let path = self
            .dir
            .join(format!("judge-{}-{call}.json", self.request_id));
        let llm = request.llm_request;
        let instructions = llm
            .instructions
            .iter()
            .flat_map(|b| &b.content)
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        // model.complete accepts text, not a provider-specific structured-output field.
        // Send the library's schema as an instruction; libsy still validates the reply.
        emit(&json!({
            "id": self.request_id,
            "call": call,
            "cache_turn": self.cache_turn,
            "classifier": {
                "model": llm.model,
                "system": format!("{instructions}\nReturn only JSON matching this output format: {}", json!(llm.output.response_format)),
                "prompt": serde_json::to_string(&llm.messages).map_err(|e| LlmClientError::General(e.to_string()))?,
                "maxTokens": llm.output.max_output_tokens.unwrap_or(384),
            }
        })).map_err(LlmClientError::General)?;
        // Longer than the mod's native-call timeout, shorter than its decision deadline.
        let reply = timeout(Duration::from_secs(12), async {
            loop {
                if let Ok(text) = fs::read_to_string(&path)
                    && let Ok(reply) = serde_json::from_str::<Reply>(&text)
                {
                    break reply;
                }
                sleep(POLL).await;
            }
        })
        .await;
        let _ = fs::remove_file(path);
        let reply =
            reply.map_err(|_| LlmClientError::General("Native classifier timed out".into()))?;
        if reply.error.is_some() {
            return Err(LlmClientError::General(
                "Native classifier unavailable".into(),
            ));
        }
        let text = reply
            .text
            .ok_or_else(|| LlmClientError::General("Empty classifier reply".into()))?;
        Ok(Response {
            llm_response: LlmResponse::Agg(text_response(llm.model, text)),
            metadata: request.metadata,
            upstream_headers: Default::default(),
        })
    }
}
