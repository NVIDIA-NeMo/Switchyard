// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for Codex traffic sent through a LiteLLM gateway to Bedrock Claude.
//!
//! The stub gateway reproduces three behaviors of the real gateway that broke Codex:
//! it streams a freeform tool call as function-style argument deltas whose JSON differs from
//! the finished call, it stops at 4,096 output tokens when a request sets no limit, and it
//! rejects tool-call history in a request that defines no tools. Each test runs the
//! production server, config loader, router, and HTTP client against that stub.

use std::convert::Infallible;
use std::error::Error;
use std::io::Write;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode, Uri};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use switchyard_server::{build_switchyard_router, config::load_server_state};
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tower::ServiceExt;

type TestResult<T = ()> = Result<T, Box<dyn Error + Send + Sync>>;

/// Output tokens the gateway allows Bedrock Claude when a request sets no limit.
const GATEWAY_DEFAULT_OUTPUT_CAP: u64 = 4096;
/// Output limit that the `claude` target adds through `extra_body`.
const CONFIGURED_OUTPUT_LIMIT: u64 = 64000;
const LITELLM_TOOLS_ERROR: &str = "litellm.UnsupportedParamsError: Bedrock doesn't support tool calling without `tools=` param specified.";
const CONFLICT_MESSAGE: &str = "Responses snapshot conflicts with streamed content";
/// Both route types send the requests in these tests to the `claude` target.
const CLAUDE_ROUTES: [&str; 2] = ["gateway/passthrough", "gateway/stage"];

// `Gateway::drop` stops the server task even when an assertion fails.
struct Gateway {
    root: String,
    calls: Arc<Mutex<Vec<Value>>>,
    task: JoinHandle<std::io::Result<()>>,
}

impl Gateway {
    async fn start() -> TestResult<Self> {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let app = Router::new()
            .route("/v1/responses", post(gateway))
            .route("/v1/chat/completions", post(gateway))
            .with_state(Arc::clone(&calls));
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let root = format!("http://{}", listener.local_addr()?);
        let task = tokio::spawn(async move { axum::serve(listener, app).await });
        Ok(Self { root, calls, task })
    }

    /// The single request the gateway received since the last call.
    async fn only_call(&self) -> TestResult<Value> {
        let mut calls = self.calls.lock().await;
        let [call] = std::mem::take(&mut *calls)
            .try_into()
            .map_err(|calls: Vec<Value>| {
                format!("expected one gateway call, got {}", calls.len())
            })?;
        Ok(call)
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

[llm_clients.responses]
format = "openai_responses"
base_url = "{root}/v1"
max_retries = 0

[llm_clients.chat]
format = "openai_chat"
base_url = "{root}/v1"
max_retries = 0

[targets.claude]
id = "model/claude"
llm_client = "responses"
extra_body = {{ max_output_tokens = {CONFIGURED_OUTPUT_LIMIT} }}

[targets.claude_uncapped]
id = "model/claude-uncapped"
llm_client = "responses"

[targets.gpt]
id = "model/gpt"
llm_client = "responses"

[targets.claude_chat]
id = "model/claude-chat"
llm_client = "chat"

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

[routes.uncapped]
id = "gateway/uncapped"
type = "passthrough"
target = "claude_uncapped"

[routes.chat]
id = "gateway/chat"
type = "passthrough"
target = "claude_chat"
"#,
        root = gateway.root,
    )?;
    Ok(build_switchyard_router(load_server_state(config.path())?))
}

async fn post_json(app: &Router, path: &str, body: &Value) -> TestResult<(StatusCode, String)> {
    let request = Request::builder()
        .method("POST")
        .uri(path)
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
        .filter(|data| *data != "[DONE]")
        .filter_map(|data| serde_json::from_str(data).ok())
        .collect()
}

fn event_types(events: &[Value]) -> Vec<&str> {
    events
        .iter()
        .map(|event| event["type"].as_str().unwrap_or_default())
        .collect()
}

/// A Codex turn: the tools Codex offers and one user message naming the gateway scenario.
fn codex_turn(route: &str, scenario: &str, stream: bool) -> Value {
    json!({
        "model": route,
        "stream": stream,
        "tools": [
            {"type": "custom", "name": "exec", "description": "Run JavaScript", "format": {"type": "text"}},
            {"type": "function", "name": "shell", "description": "Run a command",
                "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}}
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": true,
        "input": [user_text(&format!("scenario:{scenario}"))]
    })
}

fn user_text(text: &str) -> Value {
    json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]})
}

/// Answers like LiteLLM in front of Bedrock Claude. The `scenario:<name>` text in the request
/// selects the response stream; see [`gateway_output`].
async fn gateway(
    State(calls): State<Arc<Mutex<Vec<Value>>>>,
    uri: Uri,
    Json(body): Json<Value>,
) -> Response {
    calls.lock().await.push(body.clone());
    if calls_a_tool(&body) && !defines_tools(&body) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": {"message": LITELLM_TOOLS_ERROR, "type": "invalid_request_error", "code": "400"}})),
        )
            .into_response();
    }
    let model = body["model"].as_str().unwrap_or_default();
    if uri.path().ends_with("/chat/completions") {
        return Json(json!({
            "id": "chatcmpl-gateway", "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        }))
        .into_response();
    }
    let (mut events, terminal) = gateway_output(&body);
    if body["stream"] != true {
        return Json(terminal["response"].clone()).into_response();
    }
    let mut created = terminal["response"].clone();
    created["status"] = json!("in_progress");
    created["output"] = json!([]);
    events.insert(0, json!({"type": "response.created", "response": created}));
    events.push(terminal);
    // The real gateway sends data-only frames without `event:` names.
    let frames = events
        .into_iter()
        .map(|event| Ok::<Event, Infallible>(Event::default().data(event.to_string())));
    Sse::new(futures_util::stream::iter(frames)).into_response()
}

fn calls_a_tool(body: &Value) -> bool {
    let responses_call = body["input"].as_array().into_iter().flatten().any(|item| {
        matches!(
            item["type"].as_str(),
            Some("function_call" | "custom_tool_call")
        )
    });
    let chat_call = body["messages"]
        .as_array()
        .into_iter()
        .flatten()
        .any(|message| message.get("tool_calls").is_some());
    responses_call || chat_call
}

fn defines_tools(body: &Value) -> bool {
    body["tools"]
        .as_array()
        .is_some_and(|tools| !tools.is_empty())
        || body["input"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|item| item["type"] == "additional_tools")
}

/// The streamed events between `response.created` and the terminal event, plus the terminal
/// event itself, for the scenario named in the request.
fn gateway_output(body: &Value) -> (Vec<Value>, Value) {
    let text = body.to_string();
    let scenario: String = text
        .split("scenario:")
        .nth(1)
        .unwrap_or_default()
        .chars()
        .take_while(|c| c.is_ascii_lowercase() || *c == '-')
        .collect();
    let mut items = Vec::new();
    match scenario.as_str() {
        "custom" => items.push(litellm_custom_call(1, "call_1", "ls")),
        "two-custom" => {
            items.push(litellm_custom_call(1, "call_1", "ls"));
            items.push(litellm_custom_call(2, "call_2", "pwd"));
        }
        "custom-and-function" => {
            items.push(litellm_custom_call(1, "call_1", "ls"));
            let arguments = r#"{"cmd": "pwd"}"#;
            items.push(function_call(2, "call_2", arguments, arguments));
        }
        "native-custom" => items.push(openai_custom_call(0, "call_1", "ls\n")),
        "conflicting-arguments" => {
            items.push(function_call(
                0,
                "call_1",
                r#"{"cmd": "ls"}"#,
                r#"{"cmd": "pwd"}"#,
            ));
        }
        "conflicting-text" => items.push(text_message(0, "Hello", "Goodbye")),
        "long-answer" if body.get("max_output_tokens").is_none() => {
            let (events, item) = text_message(0, "1\n2\n3\n", "1\n2\n3\n");
            let mut response =
                response_body(body, "incomplete", vec![item], GATEWAY_DEFAULT_OUTPUT_CAP);
            response["incomplete_details"] = json!({"reason": "max_output_tokens"});
            let terminal = json!({"type": "response.incomplete", "response": response});
            return (events, terminal);
        }
        _ => items.push(text_message(0, "ok", "ok")),
    }
    // After tool calls, the gateway closes an empty text message at output index 0 that it
    // never announced.
    if items
        .iter()
        .any(|(_, item)| item["type"] == "custom_tool_call")
        && scenario != "native-custom"
    {
        items.push(litellm_empty_message());
    }
    let events = items
        .iter()
        .flat_map(|(events, _)| events.clone())
        .collect();
    let mut output: Vec<(u64, Value)> = items
        .into_iter()
        .map(|(events, item)| (output_index(&events), item))
        .collect();
    output.sort_by_key(|(index, _)| *index);
    let output = output.into_iter().map(|(_, item)| item).collect();
    let terminal = json!({"type": "response.completed", "response": response_body(body, "completed", output, 1)});
    (events, terminal)
}

fn output_index(events: &[Value]) -> u64 {
    events
        .first()
        .and_then(|event| event["output_index"].as_u64())
        .unwrap_or_default()
}

fn response_body(body: &Value, status: &str, output: Vec<Value>, output_tokens: u64) -> Value {
    json!({
        "id": "resp_gateway", "object": "response", "model": body["model"], "status": status,
        "output": output,
        "usage": {"input_tokens": 1, "output_tokens": output_tokens, "total_tokens": 1 + output_tokens}
    })
}

/// A freeform call in the gateway's shape: function-style argument deltas that spell
/// `{"content": ...}`, then a finished `custom_tool_call` whose `input` is the bare text.
fn litellm_custom_call(index: u64, call_id: &str, input: &str) -> (Vec<Value>, Value) {
    let id = format!("ctc_{call_id}");
    let call = json!({"type": "custom_tool_call", "id": id, "call_id": call_id, "name": "exec"});
    let mut added = call.clone();
    added["status"] = json!("in_progress");
    added["input"] = json!("");
    let mut done = call;
    done["status"] = json!("completed");
    done["input"] = json!(input);
    let arguments = json!({"content": input}).to_string();
    let (first, second) = arguments.split_at(arguments.len() / 2);
    let events = vec![
        json!({"type": "response.output_item.added", "output_index": index, "item": added}),
        json!({"type": "response.function_call_arguments.delta", "item_id": id, "output_index": index, "delta": first}),
        json!({"type": "response.function_call_arguments.delta", "item_id": id, "output_index": index, "delta": second}),
        json!({"type": "response.function_call_arguments.done", "item_id": id, "output_index": index, "arguments": arguments}),
        json!({"type": "response.output_item.done", "output_index": index, "item": done}),
    ];
    (events, done)
}

/// A freeform call in OpenAI's own shape, which streams the input with custom input deltas.
fn openai_custom_call(index: u64, call_id: &str, input: &str) -> (Vec<Value>, Value) {
    let id = format!("ctc_{call_id}");
    let call = json!({"type": "custom_tool_call", "id": id, "call_id": call_id, "name": "exec"});
    let mut added = call.clone();
    added["status"] = json!("in_progress");
    added["input"] = json!("");
    let mut done = call;
    done["status"] = json!("completed");
    done["input"] = json!(input);
    let events = vec![
        json!({"type": "response.output_item.added", "output_index": index, "item": added}),
        json!({"type": "response.custom_tool_call_input.delta", "item_id": id, "output_index": index, "delta": input}),
        json!({"type": "response.custom_tool_call_input.done", "item_id": id, "output_index": index, "input": input}),
        json!({"type": "response.output_item.done", "output_index": index, "item": done}),
    ];
    (events, done)
}

/// A function call whose streamed arguments are `streamed` and whose finished item says `done`.
fn function_call(
    index: u64,
    call_id: &str,
    streamed: &str,
    done_arguments: &str,
) -> (Vec<Value>, Value) {
    let id = format!("fc_{call_id}");
    let call = json!({"type": "function_call", "id": id, "call_id": call_id, "name": "shell"});
    let mut added = call.clone();
    added["status"] = json!("in_progress");
    added["arguments"] = json!("");
    let mut done = call;
    done["status"] = json!("completed");
    done["arguments"] = json!(done_arguments);
    let events = vec![
        json!({"type": "response.output_item.added", "output_index": index, "item": added}),
        json!({"type": "response.function_call_arguments.delta", "item_id": id, "output_index": index, "delta": streamed}),
        json!({"type": "response.function_call_arguments.done", "item_id": id, "output_index": index, "arguments": streamed}),
        json!({"type": "response.output_item.done", "output_index": index, "item": done}),
    ];
    (events, done)
}

/// A text message whose streamed text is `streamed` and whose finished item says `done`.
fn text_message(index: u64, streamed: &str, done_text: &str) -> (Vec<Value>, Value) {
    let added = json!({"type": "message", "id": "msg_1", "role": "assistant", "status": "in_progress", "content": []});
    let done = json!({"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": done_text, "annotations": []}]});
    let events = vec![
        json!({"type": "response.output_item.added", "output_index": index, "item": added}),
        json!({"type": "response.content_part.added", "item_id": "msg_1", "output_index": index, "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []}}),
        json!({"type": "response.output_text.delta", "item_id": "msg_1", "output_index": index, "content_index": 0, "delta": streamed}),
        json!({"type": "response.output_text.done", "item_id": "msg_1", "output_index": index, "content_index": 0, "text": done_text}),
        json!({"type": "response.output_item.done", "output_index": index, "item": done}),
    ];
    (events, done)
}

/// The empty text message the gateway closes at output index 0 after a tool call.
fn litellm_empty_message() -> (Vec<Value>, Value) {
    let done = json!({"type": "message", "id": "msg_empty", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": "", "annotations": []}]});
    let events = vec![
        json!({"type": "response.output_text.done", "item_id": "msg_empty", "output_index": 0, "content_index": 0, "text": ""}),
        json!({"type": "response.content_part.done", "item_id": "msg_empty", "output_index": 0, "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []}}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": done}),
    ];
    (events, done)
}

/// The tool calls a stream finished, as `(type, name, input or arguments)`.
fn finished_tool_calls(events: &[Value]) -> Vec<(String, String, String)> {
    events
        .iter()
        .filter(|event| event["type"] == "response.output_item.done")
        .map(|event| &event["item"])
        .filter(|item| {
            matches!(
                item["type"].as_str(),
                Some("custom_tool_call" | "function_call")
            )
        })
        .map(|item| {
            let value = item["input"].as_str().or(item["arguments"].as_str());
            (
                item["type"].as_str().unwrap_or_default().to_string(),
                item["name"].as_str().unwrap_or_default().to_string(),
                value.unwrap_or_default().to_string(),
            )
        })
        .collect()
}

/// An expected finished tool call: `(type, name, input or arguments)`.
type ToolCall<'a> = (&'a str, &'a str, &'a str);

/// Codex showed "Reconnecting... (stream closed before response.completed)" after every tool
/// call because the server ended the stream after the gateway's freeform call.
#[tokio::test]
async fn codex_tool_call_streams_reach_response_completed() -> TestResult {
    let gateway = Gateway::start().await?;
    let app = app(&gateway)?;
    let cases: [(&str, &[ToolCall]); 4] = [
        ("custom", &[("custom_tool_call", "exec", "ls")]),
        (
            "two-custom",
            &[
                ("custom_tool_call", "exec", "ls"),
                ("custom_tool_call", "exec", "pwd"),
            ],
        ),
        (
            "custom-and-function",
            &[
                ("custom_tool_call", "exec", "ls"),
                ("function_call", "shell", r#"{"cmd": "pwd"}"#),
            ],
        ),
        // OpenAI streams freeform input with its own events; it must keep working.
        ("native-custom", &[("custom_tool_call", "exec", "ls\n")]),
    ];
    for route in CLAUDE_ROUTES {
        for (scenario, expected) in cases {
            let case = format!("{route} {scenario}");
            let (status, body) =
                post_json(&app, "/v1/responses", &codex_turn(route, scenario, true)).await?;
            assert_eq!(status, StatusCode::OK, "{case}: {body}");
            let events = sse_events(&body);
            let types = event_types(&events);
            assert_eq!(
                types.last(),
                Some(&"response.completed"),
                "{case}: {types:?}"
            );
            assert!(!types.contains(&"error"), "{case}: {types:?}");
            let expected: Vec<_> = expected
                .iter()
                .map(|(kind, name, value)| (kind.to_string(), name.to_string(), value.to_string()))
                .collect();
            assert_eq!(finished_tool_calls(&events), expected, "{case}");
            assert_eq!(
                gateway.only_call().await?["model"],
                "model/claude",
                "{case}"
            );
        }
    }
    Ok(())
}

/// A finished item that contradicts content that was already streamed ends the stream with an
/// error event in its place. The contradicting event must not be forwarded as if it were the
/// error, which ended the stream with no error event and no log.
#[tokio::test]
async fn contradicting_snapshots_end_the_stream_with_an_error_event() -> TestResult {
    let gateway = Gateway::start().await?;
    let app = app(&gateway)?;
    for route in CLAUDE_ROUTES {
        for scenario in ["conflicting-arguments", "conflicting-text"] {
            let case = format!("{route} {scenario}");
            let (status, body) =
                post_json(&app, "/v1/responses", &codex_turn(route, scenario, true)).await?;
            assert_eq!(status, StatusCode::OK, "{case}: {body}");
            let events = sse_events(&body);
            let types = event_types(&events);
            let last = events.last().ok_or("the stream must not be empty")?;
            assert_eq!(last["type"], "error", "{case}: {types:?}");
            assert_eq!(last["message"], CONFLICT_MESSAGE, "{case}");
            // The contradicting item is replaced by the error, and nothing follows it.
            assert!(
                !types.contains(&"response.output_item.done"),
                "{case}: {types:?}"
            );
            assert!(!types.contains(&"response.completed"), "{case}: {types:?}");
            gateway.only_call().await?;
        }
    }
    Ok(())
}

/// Codex sends no `max_output_tokens`, so the gateway stopped Sonnet at 4,096 tokens and Codex
/// ended long turns with an empty reply. A target's `extra_body` limit lifts the cap without
/// overriding a limit the caller sets.
#[tokio::test]
async fn target_output_limit_lifts_the_gateway_default_cap() -> TestResult {
    let gateway = Gateway::start().await?;
    let app = app(&gateway)?;

    // Without a configured limit the gateway cuts the answer off, and the client receives
    // `response.incomplete`.
    let (status, body) = post_json(
        &app,
        "/v1/responses",
        &codex_turn("gateway/uncapped", "long-answer", true),
    )
    .await?;
    assert_eq!(status, StatusCode::OK, "{body}");
    let events = sse_events(&body);
    assert_eq!(
        event_types(&events).last(),
        Some(&"response.incomplete"),
        "{body}"
    );
    assert!(
        gateway
            .only_call()
            .await?
            .get("max_output_tokens")
            .is_none()
    );

    for route in CLAUDE_ROUTES {
        for stream in [true, false] {
            let case = format!("{route} stream={stream}");
            let (status, body) = post_json(
                &app,
                "/v1/responses",
                &codex_turn(route, "long-answer", stream),
            )
            .await?;
            assert_eq!(status, StatusCode::OK, "{case}: {body}");
            if stream {
                let events = sse_events(&body);
                assert_eq!(
                    event_types(&events).last(),
                    Some(&"response.completed"),
                    "{case}: {body}"
                );
            } else {
                assert_eq!(
                    serde_json::from_str::<Value>(&body)?["status"],
                    "completed",
                    "{case}"
                );
            }
            let call = gateway.only_call().await?;
            assert_eq!(call["max_output_tokens"], CONFIGURED_OUTPUT_LIMIT, "{case}");
        }

        let mut limited = codex_turn(route, "long-answer", true);
        limited["max_output_tokens"] = json!(1000);
        let (status, body) = post_json(&app, "/v1/responses", &limited).await?;
        assert_eq!(status, StatusCode::OK, "{route}: {body}");
        assert_eq!(
            gateway.only_call().await?["max_output_tokens"],
            1000,
            "{route}"
        );
    }
    Ok(())
}

/// The `(type, name)` of each tool definition in a gateway request, for either OpenAI API.
fn defined_tools(call: &Value) -> Vec<(String, String)> {
    call["tools"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|tool| {
            let name = tool["name"].as_str().or(tool["function"]["name"].as_str());
            (
                tool["type"].as_str().unwrap_or_default().to_string(),
                name.unwrap_or_default().to_string(),
            )
        })
        .collect()
}

fn codex_compaction(route: &str, stream: bool) -> Value {
    json!({
        "model": route,
        "stream": stream,
        "tools": [],
        "tool_choice": "auto",
        "parallel_tool_calls": false,
        "input": [
            user_text("List files"),
            {"type": "custom_tool_call", "call_id": "call_1", "name": "exec", "input": "ls"},
            {"type": "custom_tool_call_output", "call_id": "call_1", "output": "README.md"},
            {"type": "function_call", "call_id": "call_2", "name": "shell", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_2", "output": "ok"},
            {"type": "function_call", "call_id": "call_3", "name": "shell", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_3", "output": "ok"},
            {"type": "function_call", "call_id": "call_4", "namespace": "collaboration",
                "name": "spawn_agent", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_4", "output": "ok"},
            user_text("Summarize the conversation.")
        ]
    })
}

/// Codex compacts a long thread by sending its tool-call history with an empty `tools` list.
/// The gateway rejected those requests with HTTP 400, so compaction never finished.
#[tokio::test]
async fn tool_history_without_tools_gets_definitions_the_gateway_accepts() -> TestResult {
    let gateway = Gateway::start().await?;
    let app = app(&gateway)?;
    let codex_tools = [
        ("custom".to_string(), "exec".to_string()),
        ("function".to_string(), "shell".to_string()),
        (
            "function".to_string(),
            "collaboration__spawn_agent".to_string(),
        ),
    ];
    for route in CLAUDE_ROUTES {
        for stream in [true, false] {
            let case = format!("{route} compaction stream={stream}");
            let (status, body) =
                post_json(&app, "/v1/responses", &codex_compaction(route, stream)).await?;
            assert_eq!(status, StatusCode::OK, "{case}: {body}");
            if stream {
                let types = event_types(&sse_events(&body)).join(",");
                assert!(types.ends_with("response.completed"), "{case}: {types}");
            }
            let call = gateway.only_call().await?;
            assert_eq!(call["model"], "model/claude", "{case}");
            assert_eq!(defined_tools(&call), codex_tools, "{case}");
            assert_eq!(call["tool_choice"], "none", "{case}");
        }

        // Anthropic Messages clients such as Claude Code reach the same target through
        // translation.
        let messages = json!({
            "model": route,
            "max_tokens": 300,
            "messages": [
                {"role": "user", "content": "List files"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "README.md"},
                    {"type": "text", "text": "Summarize the conversation."}
                ]}
            ]
        });
        let (status, body) = post_json(&app, "/v1/messages", &messages).await?;
        assert_eq!(status, StatusCode::OK, "{route} messages: {body}");
        let call = gateway.only_call().await?;
        assert_eq!(
            defined_tools(&call),
            [("function".to_string(), "Bash".to_string())],
            "{route} messages"
        );
        assert_eq!(call["tool_choice"], "none", "{route} messages");
    }

    let chat = json!({
        "model": "gateway/chat",
        "messages": [
            {"role": "user", "content": "List files"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "shell", "arguments": "{}"}},
                {"id": "call_2", "type": "function", "function": {"name": "shell", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "README.md"},
            {"role": "tool", "tool_call_id": "call_2", "content": "README.md"},
            {"role": "user", "content": "Summarize the conversation."}
        ]
    });
    let (status, body) = post_json(&app, "/v1/chat/completions", &chat).await?;
    assert_eq!(status, StatusCode::OK, "chat: {body}");
    let call = gateway.only_call().await?;
    assert_eq!(
        defined_tools(&call),
        [("function".to_string(), "shell".to_string())]
    );
    assert_eq!(call["tool_choice"], "none");
    Ok(())
}

/// Requests that define their own tools, or that call no tools, reach the gateway unchanged.
#[tokio::test]
async fn requests_with_their_own_tools_or_no_tool_history_are_unchanged() -> TestResult {
    let gateway = Gateway::start().await?;
    let app = app(&gateway)?;
    for route in CLAUDE_ROUTES {
        let mut defined = codex_compaction(route, true);
        defined["tools"] = json!([{"type": "function", "name": "shell", "description": "Run a command",
            "parameters": {"type": "object"}}]);
        let (status, body) = post_json(&app, "/v1/responses", &defined).await?;
        assert_eq!(status, StatusCode::OK, "{route} defined: {body}");
        let call = gateway.only_call().await?;
        assert_eq!(call["tools"], defined["tools"], "{route} defined");
        assert_eq!(call["tool_choice"], "auto", "{route} defined");

        // Codex's Responses-lite requests carry their tools inside `input`.
        let mut lite = codex_compaction(route, true);
        let lite_tools = json!({"type": "additional_tools", "role": "developer",
            "tools": [{"type": "function", "name": "shell", "parameters": {"type": "object"}}]});
        lite.as_object_mut()
            .ok_or("request must be an object")?
            .remove("tools");
        lite["input"]
            .as_array_mut()
            .ok_or("input must be an array")?
            .insert(0, lite_tools.clone());
        let (status, body) = post_json(&app, "/v1/responses", &lite).await?;
        assert_eq!(status, StatusCode::OK, "{route} lite: {body}");
        let call = gateway.only_call().await?;
        assert!(call.get("tools").is_none(), "{route} lite: {call}");
        assert_eq!(call["input"][0], lite_tools, "{route} lite");
        assert_eq!(call["tool_choice"], "auto", "{route} lite");

        let mut plain = codex_turn(route, "ok", true);
        plain["tools"] = json!([]);
        let (status, body) = post_json(&app, "/v1/responses", &plain).await?;
        assert_eq!(status, StatusCode::OK, "{route} plain: {body}");
        let call = gateway.only_call().await?;
        assert_eq!(defined_tools(&call), [], "{route} plain");
        assert_eq!(call["tool_choice"], "auto", "{route} plain");
    }
    Ok(())
}
