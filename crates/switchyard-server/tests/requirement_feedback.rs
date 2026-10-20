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
const CONFIG: &str = include_str!("../../../examples/classifier-requirement-feedback.toml");
const LEDGER: &str = include_str!("../../../examples/requirement-ledger.toml");
const TASK: &str = "Fix the parser and add a regression test.";
const NOTE: &str = "Requirement evidence review (advisory;";

#[derive(Default)]
struct Upstream {
    verdicts: Mutex<VecDeque<Value>>,
    requests: Mutex<Vec<Value>>,
}

async fn complete(
    State(state): State<Arc<Upstream>>,
    Json(body): Json<Value>,
) -> (StatusCode, Json<Value>) {
    let model = body["model"].clone();
    state.requests.lock().push(body);
    let text = if model == "model/judge" {
        let verdict = state
            .verdicts
            .lock()
            .pop_front()
            .expect("queued judge reply");
        if verdict.is_null() {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error":{"message":"judge unavailable"}})),
            );
        }
        verdict.to_string()
    } else {
        "ok".into()
    };
    (
        StatusCode::OK,
        Json(json!({
            "id":"test", "object":"chat.completion", "model":model,
            "choices":[{"index":0,"message":{"role":"assistant","content":text},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}
        })),
    )
}

struct Server(tokio::task::JoinHandle<()>);
impl Drop for Server {
    fn drop(&mut self) {
        self.0.abort();
    }
}

fn assessment(status: &str, verification: &str) -> Value {
    json!({"through_message":2,"requirements":(["R01","R02"].map(|id| json!({
        "id":id,"status":status,"verification":verification,
        "explanation":"The visible parser regression check is relevant.",
        "evidence":[{"message":2,"block":0,"call_id":"check-1","quote":"parser regression passed"}]
    })))})
}

#[tokio::test]
async fn cited_feedback_reaches_fixed_target_and_failures_omit_it_across_formats() -> TestResult {
    let upstream = Arc::new(Upstream::default());
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let url = format!("http://{}/v1", listener.local_addr()?);
    let mock = Router::new()
        .route("/v1/chat/completions", post(complete))
        .with_state(upstream.clone());
    let _server = Server(tokio::spawn(async move {
        axum::serve(listener, mock).await.expect("mock server")
    }));
    let dir = tempfile::tempdir()?;
    let config = dir.path().join("routes.toml");
    std::fs::write(
        &config,
        CONFIG
            .replace("http://localhost:8000/v1", &url)
            .replace(
                "[targets.target]",
                "[targets.other]\nid = \"model/other\"\nllm_client = \"local\"\n\n[targets.target]",
            )
            .replace("any = [\"target\"]", "any = [\"other\", \"target\"]"),
    )?;
    std::fs::write(dir.path().join("requirement-ledger.toml"), LEDGER)?;
    let app = build_switchyard_router(ServerState::from_runner(Runner::load(&config)?)?);
    let inputs = [
        (
            "/v1/chat/completions",
            json!({"model":"switchyard/requirements","messages":[
                {"role":"user","content":TASK},
                {"role":"assistant","tool_calls":[{"id":"check-1","type":"function","function":{"name":"test","arguments":"{}"}}]},
                {"role":"tool","tool_call_id":"check-1","content":"parser regression passed"}
            ]}),
        ),
        (
            "/v1/messages",
            json!({"model":"switchyard/requirements","max_tokens":16,"messages":[
                {"role":"user","content":TASK},
                {"role":"assistant","content":[{"type":"tool_use","id":"check-1","name":"test","input":{}}]},
                {"role":"user","content":[{"type":"tool_result","tool_use_id":"check-1","content":"parser regression passed"}]}
            ]}),
        ),
        (
            "/v1/responses",
            json!({"model":"switchyard/requirements","input":[
                {"role":"user","content":TASK},
                {"type":"function_call","call_id":"check-1","name":"test","arguments":"{}"},
                {"type":"function_call_output","call_id":"check-1","output":"parser regression passed"}
            ]}),
        ),
    ];
    for (path, body) in inputs {
        let mut invented = assessment("supported_as_met", "current");
        invented["requirements"][0]["evidence"][0]["quote"] = json!("invented evidence");
        let mut duplicate = assessment("supported_as_met", "current");
        duplicate["requirements"][1]["id"] = json!("R01");
        for (verdict, expected) in [
            (assessment("supported_as_met", "current"), true),
            (assessment("incomplete", "stale"), true),
            (assessment("supported_as_met", "stale"), false),
            (invented, false),
            (duplicate, false),
            (json!({}), false),
            (Value::Null, false),
        ] {
            upstream.verdicts.lock().push_back(verdict);
            upstream.requests.lock().clear();
            let response = app
                .clone()
                .oneshot(
                    Request::post(path)
                        .header("content-type", "application/json")
                        .header("x-switchyard-session-id", "one-agent")
                        .body(Body::from(body.to_string()))?,
                )
                .await?;
            assert_eq!(response.status(), StatusCode::OK);
            to_bytes(response.into_body(), usize::MAX).await?;
            let requests = upstream.requests.lock();
            assert_eq!(
                requests.len(),
                2,
                "one observer call and one acting-model call"
            );
            let judge = &requests[0];
            assert_eq!(judge["model"], "model/judge");
            let judge_messages = judge["messages"].to_string();
            assert!(judge_messages.contains("Frozen requirement ledger"));
            assert!(judge_messages.contains("parser regression passed"));
            assert!(judge_messages.contains("check-1"));
            let target = &requests[1];
            assert_eq!(
                target["model"], "model/target",
                "judge failure still uses policy.target"
            );
            assert_eq!(
                target["messages"].to_string().contains(NOTE),
                expected,
                "{path}: {target}"
            );
            assert!(
                target["messages"]
                    .to_string()
                    .contains("parser regression passed")
            );
        }
    }
    Ok(())
}

#[test]
fn ledger_and_full_prefix_configuration_are_checked_at_load() -> TestResult {
    let dir = tempfile::tempdir()?;
    let config = dir.path().join("routes.toml");
    let ledger = dir.path().join("requirement-ledger.toml");
    std::fs::write(&config, CONFIG)?;
    assert!(Runner::load(&config).is_err(), "missing ledger");
    for invalid in [
        "".to_string(),
        "invalid TOML".to_string(),
        LEDGER.replace("id = \"R02\"", "id = \"R01\""),
        LEDGER.replace(
            "source_quote = \"Fix the parser\"",
            "source_quote = \"invented\"",
        ),
    ] {
        std::fs::write(&ledger, invalid)?;
        assert!(Runner::load(&config).is_err());
    }
    std::fs::write(&ledger, LEDGER)?;
    assert!(Runner::load(&config).is_ok());
    for invalid in [
        CONFIG.replace("every_request", "new_session"),
        CONFIG.replace("every_request", "user_turn"),
        CONFIG.replace(
            "max_output_tokens = 4096",
            "max_output_tokens = 4096\nrecent_turn_window = 8",
        ),
        CONFIG.replace("target = \"target\"", "target = \"judge\""),
        CONFIG.replace("target = \"target\"", "target = \"unknown\""),
    ] {
        std::fs::write(&config, invalid)?;
        assert!(Runner::load(&config).is_err());
    }
    Ok(())
}
