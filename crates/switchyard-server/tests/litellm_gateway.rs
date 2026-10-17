// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Regression test for Codex traffic sent through a LiteLLM gateway to Bedrock Claude.
//!
//! The stub gateway streams a freeform tool call the way the real gateway does: as
//! function-style argument deltas whose JSON differs from the finished call. The test runs the
//! production server, config loader, router, and HTTP client against that stub.

use std::convert::Infallible;
use std::error::Error;
use std::io::Write;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use switchyard_server::{build_switchyard_router, config::load_server_state};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tower::ServiceExt;

type TestResult<T = ()> = Result<T, Box<dyn Error + Send + Sync>>;

/// Both route types send the test's requests to the `claude` target.
const CLAUDE_ROUTES: [&str; 2] = ["gateway/passthrough", "gateway/stage"];

// `Gateway::drop` stops the server task even when an assertion fails.
struct Gateway {
    root: String,
    task: JoinHandle<std::io::Result<()>>,
}

impl Gateway {
    async fn start() -> TestResult<Self> {
        let app = Router::new().route("/v1/responses", post(gateway));
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let root = format!("http://{}", listener.local_addr()?);
        let task = tokio::spawn(async move { axum::serve(listener, app).await });
        Ok(Self { root, task })
    }
}

impl Drop for Gateway {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// Builds the server from a config file, as `switchyard-server --config` does.
fn app(gateway: &Gateway) -> TestResult<Router> {
    let mut config = tempfile::Builder::new().suffix(".toml").tempfile()?;
    write!(
        config,
        r#"
schema_version = 1

[llm_clients.gateway]
format = "openai_responses"
base_url = "{root}/v1"
max_retries = 0

[targets.claude]
id = "model/claude"
llm_client = "gateway"

[targets.gpt]
id = "model/gpt"
llm_client = "gateway"

[routes.passthrough]
id = "gateway/passthrough"
type = "passthrough"
target = "claude"

[routes.stage]
id = "gateway/stage"
type = "stage_router"
capable_target = "gpt"
efficient_target = "claude"
picker = "efficient_first"
confidence_threshold = 0.5
"#,
        root = gateway.root,
    )?;
    Ok(build_switchyard_router(load_server_state(config.path())?))
}

async fn post_responses(app: &Router, body: &Value) -> TestResult<(StatusCode, String)> {
    let request = Request::builder()
        .method("POST")
        .uri("/v1/responses")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(body)?))?;
    let response = app.clone().oneshot(request).await?;
    let status = response.status();
    let bytes = response.into_body().collect().await?.to_bytes();
    Ok((status, String::from_utf8(bytes.to_vec())?))
}

/// Every `data:` frame of an SSE body, parsed as JSON.
fn sse_events(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|data| serde_json::from_str(data).ok())
        .collect()
}

fn event_types(events: &[Value]) -> Vec<&str> {
    events
        .iter()
        .map(|event| event["type"].as_str().unwrap_or_default())
        .collect()
}

/// A streamed Codex turn that offers a freeform tool and a function tool.
fn codex_turn(route: &str) -> Value {
    json!({
        "model": route,
        "stream": true,
        "tools": [
            {"type": "custom", "name": "exec", "description": "Run JavaScript", "format": {"type": "text"}},
            {"type": "function", "name": "shell", "description": "Run a command",
                "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}}
        ],
        "tool_choice": "auto",
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "List files"}]}]
    })
}

/// Answers like LiteLLM in front of Bedrock Claude: a freeform call, then a function call.
async fn gateway(Json(body): Json<Value>) -> Response {
    let (custom_events, custom) = litellm_custom_call(0, "ls");
    let (function_events, function) = function_call(1, r#"{"cmd": "pwd"}"#);
    let mut events = [custom_events, function_events].concat();
    let response = json!({
        "id": "resp_gateway", "object": "response", "model": body["model"], "status": "completed",
        "output": [custom, function], "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
    });
    let mut created = response.clone();
    created["status"] = json!("in_progress");
    created["output"] = json!([]);
    events.insert(0, json!({"type": "response.created", "response": created}));
    events.push(json!({"type": "response.completed", "response": response}));
    // The real gateway sends data-only frames without `event:` names.
    let frames = events
        .into_iter()
        .map(|event| Ok::<Event, Infallible>(Event::default().data(event.to_string())));
    Sse::new(futures_util::stream::iter(frames)).into_response()
}

/// A freeform call in the gateway's shape: function-style argument deltas that spell
/// `{"content": ...}`, then a finished `custom_tool_call` whose `input` is the bare text.
fn litellm_custom_call(index: u64, input: &str) -> (Vec<Value>, Value) {
    let added = json!({"type": "custom_tool_call", "id": "ctc_1", "call_id": "call_1", "name": "exec", "input": ""});
    let mut done = added.clone();
    done["input"] = json!(input);
    let arguments = json!({"content": input}).to_string();
    let (first, second) = arguments.split_at(arguments.len() / 2);
    let events = vec![
        json!({"type": "response.output_item.added", "output_index": index, "item": added}),
        json!({"type": "response.function_call_arguments.delta", "item_id": "ctc_1", "output_index": index, "delta": first}),
        json!({"type": "response.function_call_arguments.delta", "item_id": "ctc_1", "output_index": index, "delta": second}),
        json!({"type": "response.function_call_arguments.done", "item_id": "ctc_1", "output_index": index, "arguments": arguments}),
        json!({"type": "response.output_item.done", "output_index": index, "item": done}),
    ];
    (events, done)
}

fn function_call(index: u64, arguments: &str) -> (Vec<Value>, Value) {
    let added = json!({"type": "function_call", "id": "fc_2", "call_id": "call_2", "name": "shell", "arguments": ""});
    let mut done = added.clone();
    done["arguments"] = json!(arguments);
    let events = vec![
        json!({"type": "response.output_item.added", "output_index": index, "item": added}),
        json!({"type": "response.function_call_arguments.delta", "item_id": "fc_2", "output_index": index, "delta": arguments}),
        json!({"type": "response.output_item.done", "output_index": index, "item": done}),
    ];
    (events, done)
}

/// Codex showed "Reconnecting... (stream closed before response.completed)" after every tool
/// call because the server ended the stream after the gateway's freeform call.
#[tokio::test]
async fn codex_tool_call_streams_reach_response_completed() -> TestResult {
    let gateway = Gateway::start().await?;
    let app = app(&gateway)?;
    for route in CLAUDE_ROUTES {
        let (status, body) = post_responses(&app, &codex_turn(route)).await?;
        assert_eq!(status, StatusCode::OK, "{route}: {body}");
        let events = sse_events(&body);
        let types = event_types(&events);
        assert_eq!(
            types.last(),
            Some(&"response.completed"),
            "{route}: {types:?}"
        );
        let calls: Vec<_> = events
            .iter()
            .filter(|event| event["type"] == "response.output_item.done")
            .map(|event| {
                let item = &event["item"];
                (
                    &item["name"],
                    item.get("input").unwrap_or(&item["arguments"]),
                )
            })
            .collect();
        assert_eq!(
            calls,
            [
                (&json!("exec"), &json!("ls")),
                (&json!("shell"), &json!(r#"{"cmd": "pwd"}"#))
            ],
            "{route}"
        );
    }
    Ok(())
}
