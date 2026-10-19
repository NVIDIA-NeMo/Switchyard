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

/// The charge that NVIDIA's LiteLLM gateway reported in `x-litellm-response-cost` for a buffered
/// Chat Completions answer. The estimate from `RATES` is 0.000089 instead.
const HEADER_CHARGE: &str = "6.141e-05";
/// The charge that the gateway reported as `usage.cost` at the end of a stream. The estimate
/// from `RATES` is 0.000059 instead.
const STREAM_CHARGE: f64 = 4.071e-05;
const RATES: &str = r#"{
  "model/claude": {"input": 1, "output": 5, "cache_read": 0.1, "cache_write": 2},
  "model/plain": {"input": 1, "output": 5, "cache_read": 0.1, "cache_write": 2}
}"#;

/// Answers Chat Completions the way the gateway does. A buffered answer from `model/claude`
/// carries its charge only in the header. A streamed answer has zeroed breakdown headers and the
/// charge in the final usage. `model/plain` reports no charge.
async fn charging_gateway(Json(body): Json<Value>) -> Response {
    let model = body["model"].as_str().unwrap_or_default().to_string();
    let mut usage = json!({"prompt_tokens": 9, "completion_tokens": 16, "total_tokens": 25});
    if body["stream"] == true {
        usage = json!({"prompt_tokens": 9, "completion_tokens": 10, "total_tokens": 19, "cost": STREAM_CHARGE});
        let chunk = |delta: Value, finish: Value, usage: Value| {
            json!({"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], "usage": usage})
        };
        let frames = [
            chunk(
                json!({"role": "assistant", "content": "hi"}),
                Value::Null,
                Value::Null,
            ),
            chunk(json!({}), json!("stop"), usage),
        ]
        .into_iter()
        .map(|event| event.to_string())
        .chain(["[DONE]".to_string()])
        .map(|data| Ok::<Event, Infallible>(Event::default().data(data)));
        let mut response = Sse::new(futures_util::stream::iter(frames)).into_response();
        let zero = axum::http::HeaderValue::from_static("0.0");
        response
            .headers_mut()
            .insert("x-litellm-response-cost-original", zero);
        return response;
    }
    let mut response = Json(json!({
        "id": "c1", "object": "chat.completion", "created": 1, "model": model, "usage": usage,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}]
    }))
    .into_response();
    if model == "model/claude" {
        let charge = axum::http::HeaderValue::from_static(HEADER_CHARGE);
        response
            .headers_mut()
            .insert("x-litellm-response-cost", charge);
    }
    response
}

/// The routing log records the gateway's charge when it reports one and estimates from the
/// pricing file only when it does not.
#[tokio::test]
async fn routing_log_records_the_gateway_charge() -> TestResult {
    let app = Router::new().route("/v1/chat/completions", post(charging_gateway));
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let root = format!("http://{}", listener.local_addr()?);
    let gateway = Gateway {
        root,
        task: tokio::spawn(async move { axum::serve(listener, app).await }),
    };
    let directory = tempfile::tempdir()?;
    let config = directory.path().join("config.toml");
    std::fs::write(
        &config,
        format!(
            r#"
schema_version = 1

[llm_clients.gateway]
format = "openai_chat"
base_url = "{root}/v1"
max_retries = 0

[targets.claude]
id = "model/claude"
llm_client = "gateway"

[targets.plain]
id = "model/plain"
llm_client = "gateway"

[routes.reported]
id = "gateway/reported"
type = "passthrough"
target = "claude"

[routes.unreported]
id = "gateway/unreported"
type = "passthrough"
target = "plain"
"#,
            root = gateway.root
        ),
    )?;
    let prices = directory.path().join("prices.json");
    std::fs::write(&prices, RATES)?;
    let state = load_server_state(&config)?
        .with_routing_log(directory.path().join("routing.jsonl"))?
        .with_pricing_file(&prices)?;
    let app = build_switchyard_router(state);
    let header_charge: f64 = HEADER_CHARGE.parse()?;
    for (route, stream, usd, source) in [
        (
            "gateway/reported",
            false,
            header_charge,
            "provider_reported_calls",
        ),
        (
            "gateway/reported",
            true,
            STREAM_CHARGE,
            "provider_reported_calls",
        ),
        ("gateway/unreported", false, 0.000089, "estimated_calls"),
    ] {
        let session = format!("{route}-{stream}");
        let request = Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .header("x-switchyard-session-id", &session)
            .body(Body::from(
                json!({"model": route, "stream": stream, "messages": [{"role": "user", "content": "Say hi"}]})
                    .to_string(),
            ))?;
        let response = app.clone().oneshot(request).await?;
        assert_eq!(response.status(), StatusCode::OK, "{session}");
        response.into_body().collect().await?;
        let request = Request::get(format!("/v1/routing/session-stats?session_id={session}"))
            .body(Body::empty())?;
        let stats = app.clone().oneshot(request).await?;
        let stats: Value = serde_json::from_slice(&stats.into_body().collect().await?.to_bytes())?;
        assert_eq!(stats["cost"][source], 1, "{session}: {stats}");
        let recorded = stats["cost"]["total_usd"].as_f64().ok_or("missing cost")?;
        assert!(
            (recorded - usd).abs() < 1e-12,
            "{session}: {recorded} != {usd}"
        );
    }
    Ok(())
}
