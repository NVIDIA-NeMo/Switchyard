// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::error::Error;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::routing::post;
use axum::{Json, Router};
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::Mutex;

type TestResult<T = ()> = Result<T, Box<dyn Error + Send + Sync>>;

struct Server(Child);

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

async fn decision(
    State(calls): State<Arc<Mutex<Vec<Value>>>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> (StatusCode, Json<Value>) {
    assert_eq!(headers["authorization"], "Bearer test-decision-key");
    calls.lock().await.push(body.clone());
    let task = body["state"]["task"].to_string();
    if task.contains("unavailable") {
        return (
            StatusCode::BAD_GATEWAY,
            Json(json!({"error": "unavailable"})),
        );
    }
    let score = if task.contains("simple") { 0.1 } else { 0.9 };
    (
        StatusCode::OK,
        Json(json!({
            "model": body["model"],
            "answers": {"route": {
                "type": "choice",
                "choice": if score > 0.5 { "advantage" } else { "no_advantage" },
                "probabilities": {"advantage": score, "no_advantage": 1.0 - score},
                "confidence": 0.99,
            }},
        })),
    )
}

fn config(endpoint: &str, trigger: &str, fail_open: bool) -> String {
    format!(
        r#"
schema_version = 1
[llm_clients.answers]
format = "openai_chat"
base_url = "http://127.0.0.1:1/v1"
[targets.capable]
id = "model/capable"
llm_client = "answers"
[targets.efficient]
id = "model/efficient"
llm_client = "answers"
[decision_clients.decisions]
format = "system_one"
endpoint = "{endpoint}"
api_key_env = "SWITCHYARD_TEST_DECISION_KEY"
timeout_ms = 1000
[decision_targets.judge]
id = "model/decision"
decision_client = "decisions"
[routes.test]
id = "composed"
type = "composite"
[routes.test.classifier]
target = "judge"
classify_trigger = "{trigger}"
fail_open = {fail_open}
[routes.test.classifier.decision]
cutoff = 0.4
candidates = {{ a = "capable", b = "efficient" }}
[routes.test.classifier.decision.evidence]
description = "Controlled comparison for a routing test"
[routes.test.stage]
capable_target = "capable"
efficient_target = "efficient"
confidence_threshold = 0.4
recent_turn_window = 4
"#
    )
}

async fn start(config: &str) -> TestResult<(Server, String)> {
    let file = tempfile::NamedTempFile::new()?;
    std::fs::write(file.path(), config)?;
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    drop(listener);
    let mut server = Server(
        Command::new(env!("CARGO_BIN_EXE_switchyard-server"))
            .args([
                "--host",
                "127.0.0.1",
                "--port",
                &port.to_string(),
                "--config",
            ])
            .arg(file.path())
            .env("SWITCHYARD_TEST_DECISION_KEY", "test-decision-key")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?,
    );
    let base = format!("http://127.0.0.1:{port}");
    for _ in 0..100 {
        if let Some(status) = server.0.try_wait()? {
            return Err(format!("server exited during startup: {status}").into());
        }
        if reqwest::get(format!("{base}/health")).await.is_ok() {
            return Ok((server, base));
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    Err("server startup timed out".into())
}

async fn select(base: &str, session: &str, messages: &[Value]) -> TestResult<Value> {
    Ok(reqwest::Client::new()
        .post(format!("{base}/v1/decision"))
        .header("x-switchyard-session-id", session)
        .json(&json!({
            "input_format": "openai_chat",
            "request": {"model": "composed", "messages": messages},
        }))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?)
}

fn tool(messages: &mut Vec<Value>, name: &str, arguments: Value, result: Value) {
    let id = format!("tool_{}", messages.len());
    messages.push(json!({"role": "assistant", "tool_calls": [{
        "id": id, "type": "function", "function": {
            "name": name, "arguments": arguments.to_string(),
        },
    }]}));
    messages.push(json!({"role": "tool", "tool_call_id": id, "content": result.to_string()}));
}

#[tokio::test]
async fn composite_decision_judge_retains_tiers_and_reclassifies_user_turns() -> TestResult {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .route("/v1/systemone", post(decision))
        .with_state(Arc::clone(&calls));
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let endpoint = format!("http://{}/v1/systemone", listener.local_addr()?);
    let upstream = tokio::spawn(async move { axum::serve(listener, app).await });
    let (_server, base) = start(&config(&endpoint, "user_turn", false)).await?;
    let mut messages = vec![json!({"role": "user", "content": "complex task"})];
    assert_eq!(
        select(&base, "session", &messages).await?["selected"]["target"],
        "capable"
    );
    for path in ["a.rs", "b.rs", "c.rs"] {
        tool(
            &mut messages,
            "edit",
            json!({"path": path, "oldText": "old", "newText": "new"}),
            json!({"success": true}),
        );
    }
    assert_eq!(
        select(&base, "session", &messages).await?["selected"]["target"],
        "efficient"
    );
    for _ in 0..2 {
        tool(
            &mut messages,
            "bash",
            json!({"command": "cargo test"}),
            json!({"success": false, "exit_code": 1, "output": "AssertionError: test failed"}),
        );
    }
    assert_eq!(
        select(&base, "session", &messages).await?["selected"]["target"],
        "capable"
    );
    assert_eq!(calls.lock().await.len(), 1);
    let call = calls.lock().await[0].clone();
    assert_eq!(call["model"], "model/decision");
    assert_eq!(call["questions"]["route"]["type"], "choice");
    assert_eq!(
        call["state"]["comparison"],
        json!({"capable": "a", "efficient": "b"})
    );
    assert_eq!(
        call["state"]["evidence"]["description"],
        "Controlled comparison for a routing test"
    );

    for _ in 0..3 {
        for path in ["a.rs", "b.rs", "c.rs", "d.rs"] {
            tool(
                &mut messages,
                "edit",
                json!({"path": path, "oldText": "old", "newText": "new"}),
                json!({"success": true}),
            );
        }
        select(&base, "session", &messages).await?;
    }
    assert_eq!(calls.lock().await.len(), 1);
    let followup = [json!({"role": "user", "content": "simple task"})];
    assert_eq!(
        select(&base, "session", &followup).await?["selected"]["target"],
        "efficient"
    );
    assert_eq!(calls.lock().await.len(), 2);
    let unavailable = [json!({"role": "user", "content": "unavailable"})];
    assert!(select(&base, "session", &unavailable).await.is_err());

    let (_server, base) = start(&config(&endpoint, "user_turn", true)).await?;
    select(
        &base,
        "retained",
        &[json!({"role": "user", "content": "complex task"})],
    )
    .await?;
    assert_eq!(
        select(&base, "retained", &unavailable).await?["selected"]["target"],
        "capable"
    );
    assert_eq!(
        select(&base, "new-session", &unavailable).await?["selected"]["target"],
        "efficient"
    );

    let (_server, base) = start(&config(&endpoint, "new_session", false)).await?;
    select(
        &base,
        "once",
        &[json!({"role": "user", "content": "complex task"})],
    )
    .await?;
    let before = calls.lock().await.len();
    assert_eq!(
        select(&base, "once", &followup).await?["selected"]["target"],
        "capable"
    );
    assert_eq!(calls.lock().await.len(), before);
    upstream.abort();
    Ok(())
}

#[test]
fn composite_decision_configuration_rejects_mixed_settings_and_target_kinds() -> TestResult {
    let valid = config("http://127.0.0.1:1/v1/systemone", "user_turn", false);
    for (source, expected) in [
        (
            valid.replace(
                "target = \"judge\"",
                "target = \"judge\"\nbase_threshold = 0.5",
            ),
            "decision cannot use LLM judge settings",
        ),
        (
            valid.replace(
                "target = \"judge\"",
                "target = \"judge\"\nthreshold_step = 0.0",
            ),
            "decision cannot use LLM judge settings",
        ),
        (
            valid.replace("target = \"judge\"", "target = \"capable\""),
            "requires target kind decision",
        ),
        (
            valid.replace("b = \"efficient\"", "b = \"judge\""),
            "must be an LLM target",
        ),
        (
            valid.replace("b = \"efficient\"", "b = \"capable\""),
            "must include target efficient",
        ),
        (valid.replace("cutoff = 0.4", "cutoff = 1.5"), "cutoff"),
        (
            valid.replace("user_turn", "every_request"),
            "classify_trigger must be user_turn or new_session",
        ),
    ] {
        let file = tempfile::NamedTempFile::new()?;
        std::fs::write(file.path(), source)?;
        let output = Command::new(env!("CARGO_BIN_EXE_switchyard-server"))
            .arg("--config")
            .arg(file.path())
            .arg("--dry-run")
            .env("SWITCHYARD_TEST_DECISION_KEY", "test-decision-key")
            .output()?;
        assert!(!output.status.success());
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(
            error.contains(expected),
            "expected {expected:?}, got {error}"
        );
    }
    Ok(())
}
