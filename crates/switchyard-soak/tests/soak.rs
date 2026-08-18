// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests against a local server that implements the public Switchyard APIs.

use std::collections::HashSet;
use std::error::Error;
use std::sync::Arc;

use axum::Router;
use axum::extract::{Json, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use clap::Parser;
use parking_lot::Mutex;
use serde_json::{Value, json};

use switchyard_soak::{Args, run};

type TestResult = Result<(), Box<dyn Error>>;

struct TestServer {
    base_url: String,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn serve(app: Router) -> Result<TestServer, Box<dyn Error>> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let task = tokio::spawn(async move {
        let _ = axum::serve(listener, app.into_make_service()).await;
    });
    Ok(TestServer {
        base_url: format!("http://{addr}"),
        task,
    })
}

#[derive(Clone, Default)]
struct MockState {
    is_chat_broken: bool,
    seen: Arc<Mutex<HashSet<String>>>,
}

impl MockState {
    fn record(&self, endpoint: &str, body: &Value) {
        let stream = body.get("stream").and_then(Value::as_bool) == Some(true);
        self.seen.lock().insert(format!("{endpoint}:{stream}"));
    }
}

fn switchyard(is_chat_broken: bool) -> (Router, Arc<Mutex<HashSet<String>>>) {
    async fn health() -> Response {
        Json(json!({"status": "ok"})).into_response()
    }
    async fn metrics() -> Response {
        "switchyard_total_requests 1\nswitchyard_total_errors 0\n".into_response()
    }
    async fn models() -> Response {
        Json(json!({"data": [{"id": "soak-route"}]})).into_response()
    }
    async fn chat(State(state): State<MockState>, Json(body): Json<Value>) -> Response {
        let invalid_messages = body.get("messages").is_some_and(|messages| {
            messages
                .as_array()
                .is_none_or(|messages| messages.is_empty())
        });
        if invalid_messages {
            return if state.is_chat_broken {
                Json(json!({"choices": []})).into_response()
            } else {
                (StatusCode::BAD_REQUEST, "messages must not be empty").into_response()
            };
        }
        state.record("chat", &body);
        if state.is_chat_broken {
            return if body.get("stream") == Some(&Value::Bool(true)) {
                Json(json!({"choices": []})).into_response()
            } else {
                Json(json!({})).into_response()
            };
        }
        response(&body, "choices")
    }
    async fn messages(State(state): State<MockState>, Json(body): Json<Value>) -> Response {
        state.record("messages", &body);
        response(&body, "content")
    }
    async fn responses(State(state): State<MockState>, Json(body): Json<Value>) -> Response {
        state.record("responses", &body);
        response(&body, "output")
    }
    fn response(body: &Value, field: &str) -> Response {
        if body.get("stream") == Some(&Value::Bool(true)) {
            let stream = match field {
                "choices" => "data: {\"choices\":[]}\n\ndata: [DONE]\n\n",
                "content" => "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
                "output" => {
                    "event: response.completed\ndata: {\"type\":\"response.completed\"}\n\n"
                }
                _ => "",
            };
            ([(header::CONTENT_TYPE, "text/event-stream")], stream).into_response()
        } else {
            let mut response = serde_json::Map::new();
            response.insert(field.to_string(), json!([]));
            Json(Value::Object(response)).into_response()
        }
    }

    let state = MockState {
        is_chat_broken,
        ..MockState::default()
    };
    let seen = state.seen.clone();
    let app = Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/messages", post(messages))
        .route("/v1/responses", post(responses))
        .with_state(state);
    (app, seen)
}

fn args(base_url: &str, results_dir: &str) -> Result<Args, clap::Error> {
    Args::try_parse_from([
        "switchyard-soak",
        "--base-url",
        base_url,
        "--model",
        "soak-route",
        "--duration",
        "0.8s",
        "--concurrency",
        "2",
        "--report-interval",
        "0.1",
        "--invalid-canary-interval",
        "0.1",
        "--results-dir",
        results_dir,
    ])
}

#[tokio::test]
async fn short_run_exercises_every_response_variant_and_passes() -> TestResult {
    let (app, seen) = switchyard(false);
    let server = serve(app).await?;
    let dir = tempfile::tempdir()?;
    let results_dir = dir.path().join("soak-results");
    let args = args(
        &server.base_url,
        results_dir.to_str().ok_or("non-utf8 results dir")?,
    )?;

    args.validate()?;
    assert_eq!(run(args).await?, 0);

    let summary: Value =
        serde_json::from_str(&std::fs::read_to_string(results_dir.join("summary.json"))?)?;
    assert_eq!(summary["passed"], json!(true));
    assert!(summary["invalid_request_canaries"].as_u64().unwrap_or(0) > 0);
    assert_eq!(
        *seen.lock(),
        HashSet::from([
            "chat:false".to_string(),
            "chat:true".to_string(),
            "messages:false".to_string(),
            "messages:true".to_string(),
            "responses:false".to_string(),
            "responses:true".to_string(),
        ])
    );
    Ok(())
}

#[tokio::test]
async fn bad_responses_and_canary_write_a_failing_summary() -> TestResult {
    let (app, _) = switchyard(true);
    let server = serve(app).await?;
    let dir = tempfile::tempdir()?;
    let results_dir = dir.path().join("soak-results");
    let args = args(
        &server.base_url,
        results_dir.to_str().ok_or("non-utf8 results dir")?,
    )?;

    assert_eq!(run(args).await?, 1);

    let summary: Value =
        serde_json::from_str(&std::fs::read_to_string(results_dir.join("summary.json"))?)?;
    assert_eq!(summary["passed"], json!(false));
    assert!(
        summary["error_kinds"]["invalid_response"]
            .as_u64()
            .unwrap_or(0)
            > 0
    );
    assert!(
        summary["error_kinds"]["invalid_stream"]
            .as_u64()
            .unwrap_or(0)
            > 0
    );
    assert!(
        summary["invalid_request_canary_failures"]
            .as_u64()
            .unwrap_or(0)
            > 0
    );
    Ok(())
}

#[tokio::test]
async fn unknown_model_is_rejected_before_the_run_starts() -> TestResult {
    let (app, _) = switchyard(false);
    let server = serve(app).await?;
    let dir = tempfile::tempdir()?;
    let results_dir = dir.path().join("soak-results");
    let args = Args::try_parse_from([
        "switchyard-soak",
        "--base-url",
        &server.base_url,
        "--model",
        "missing",
        "--duration",
        "1s",
        "--results-dir",
        results_dir.to_str().ok_or("non-utf8 results dir")?,
    ])?;

    let error = run(args).await.unwrap_err();
    assert!(error.contains("is not listed"), "{error}");
    Ok(())
}
