// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::sync::Arc;

use axum::body::{Body, to_bytes};
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::routing::post;
use axum::{Json, Router};
use parking_lot::Mutex;
use serde_json::{Value, json};
use switchyard_runner::Runner;
use switchyard_server::{ServerState, build_switchyard_router};
use tokio::net::TcpListener;
use tower::ServiceExt;

type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

const CONFIG: &str = include_str!("../../../examples/classifier-handoff-notes.toml");
const NOTES: &str = include_str!("../../../examples/handoff-notes.toml");
const NOTE: &str = "Read the compiler error and fix the first failing operation.";

#[derive(Default)]
struct Upstream {
    verdicts: Mutex<VecDeque<Value>>,
    requests: Mutex<Vec<Value>>,
}

async fn complete(State(state): State<Arc<Upstream>>, Json(body): Json<Value>) -> Json<Value> {
    let text = if body["model"] == "model/judge" {
        state
            .verdicts
            .lock()
            .pop_front()
            .unwrap_or(json!({"target": "any", "note": null}))
            .to_string()
    } else {
        "ok".into()
    };
    let model = body["model"].clone();
    state.requests.lock().push(body);
    Json(json!({
        "id": "test", "object": "chat.completion", "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    }))
}

struct Server(tokio::task::JoinHandle<()>);
impl Drop for Server {
    fn drop(&mut self) {
        self.0.abort();
    }
}

#[tokio::test]
async fn selected_database_notes_reach_the_model_across_input_formats() -> TestResult {
    let upstream = Arc::new(Upstream::default());
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let base_url = format!("http://{}/v1", listener.local_addr()?);
    let mock = Router::new()
        .route("/v1/chat/completions", post(complete))
        .with_state(upstream.clone());
    let _server = Server(tokio::spawn(async move {
        axum::serve(listener, mock).await.expect("mock server");
    }));
    let dir = tempfile::tempdir()?;
    let config = dir.path().join("routes.toml");
    std::fs::write(
        &config,
        CONFIG
            .replace("http://localhost:8000/v1", &base_url)
            .replace("\"enum\": [\"any\"]", "\"minLength\": 1"),
    )?;
    std::fs::write(dir.path().join("handoff-notes.toml"), NOTES)?;
    let app = build_switchyard_router(ServerState::from_runner(Runner::load(&config)?)?);
    let inputs = [
        (
            "/v1/chat/completions",
            json!({"model": "switchyard/notes", "messages": [{"role": "user", "content": "build failed"}]}),
        ),
        (
            "/v1/messages",
            json!({"model": "switchyard/notes", "max_tokens": 16, "messages": [
                {"role": "user", "content": "build"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "name": "build", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "build failed"}]}
            ]}),
        ),
        (
            "/v1/responses",
            json!({"model": "switchyard/notes", "input": [
                {"role": "user", "content": "build"},
                {"type": "function_call", "call_id": "call_1", "name": "build", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "build failed"}
            ]}),
        ),
    ];
    for (path, body) in inputs {
        for (verdict, expected_note) in [
            (json!({"target": "any", "note": "compile_failure"}), true),
            (json!({"target": "any", "note": null}), false),
            (json!({"target": "any", "note": "unknown"}), false),
            (json!({"target": "any", "note": 42}), false),
            (
                json!({"target": "unconfigured", "note": "compile_failure"}),
                false,
            ),
        ] {
            upstream.verdicts.lock().push_back(verdict);
            upstream.requests.lock().clear();
            let response = app
                .clone()
                .oneshot(
                    Request::post(path)
                        .header("content-type", "application/json")
                        .body(Body::from(body.to_string()))?,
                )
                .await?;
            assert_eq!(response.status(), StatusCode::OK);
            to_bytes(response.into_body(), usize::MAX).await?;
            let requests = upstream.requests.lock();
            assert_eq!(requests.len(), 2);
            let judge = &requests[0];
            assert_eq!(judge["model"], "model/judge");
            assert!(
                judge["messages"][0]["content"]
                    .as_str()
                    .is_some_and(|text| text.contains(NOTE))
            );
            let answer = &requests[1];
            assert_eq!(answer["model"], "model/answer");
            assert_eq!(
                answer["messages"].to_string().contains(NOTE),
                expected_note,
                "{path}: {answer}"
            );
            assert!(answer["messages"].to_string().contains("build failed"));
        }
    }
    std::fs::write(
        &config,
        CONFIG
            .replace("http://localhost:8000/v1", &base_url)
            .replace("every_request", "new_session"),
    )?;
    let app = build_switchyard_router(ServerState::from_runner(Runner::load(&config)?)?);
    upstream.requests.lock().clear();
    upstream
        .verdicts
        .lock()
        .push_back(json!({"target": "any", "note": "compile_failure"}));
    for _ in 0..2 {
        let body = json!({"model": "switchyard/notes", "messages": [{"role": "user", "content": "build failed"}]});
        let response = app
            .clone()
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .header("x-switchyard-session-id", "one-session")
                    .body(Body::from(body.to_string()))?,
            )
            .await?;
        assert_eq!(response.status(), StatusCode::OK);
        to_bytes(response.into_body(), usize::MAX).await?;
    }
    let requests = upstream.requests.lock();
    assert_eq!(requests.len(), 3, "one judge call and two answer calls");
    assert!(requests[1]["messages"].to_string().contains(NOTE));
    assert!(
        !requests[2]["messages"].to_string().contains(NOTE),
        "retained targets do not replay notes"
    );
    Ok(())
}

#[test]
fn handoff_note_database_is_validated_when_loading_the_route() -> TestResult {
    let dir = tempfile::tempdir()?;
    let config = dir.path().join("routes.toml");
    std::fs::write(&config, CONFIG)?;
    assert!(Runner::load(&config).is_err(), "missing note file");
    for database in [
        "",
        "note = 42",
        "note = ''",
        "'' = 'text'",
        "not valid TOML",
    ] {
        std::fs::write(dir.path().join("handoff-notes.toml"), database)?;
        assert!(Runner::load(&config).is_err(), "accepted {database:?}");
    }
    std::fs::write(dir.path().join("handoff-notes.toml"), NOTES)?;
    assert!(Runner::load(&config).is_ok());
    std::fs::write(
        &config,
        CONFIG.replace("selector = \"/note\"", "selector = \"invalid\""),
    )?;
    assert!(Runner::load(&config).is_err());
    Ok(())
}
