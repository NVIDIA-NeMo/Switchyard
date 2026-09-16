// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Run deadline cases in one test because they share process-wide attempt counters.

use std::convert::Infallible;
use std::error::Error;
use std::io::Write;
use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{Request, StatusCode};
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

// `Upstream::drop` stops the server task even when an assertion fails.
struct Upstream {
    url: String,
    calls: Arc<Mutex<Vec<Value>>>,
    task: JoinHandle<std::io::Result<()>>,
}

impl Upstream {
    async fn start() -> TestResult<Self> {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let app = Router::new()
            .route("/v1/chat/completions", post(upstream))
            .with_state(Arc::clone(&calls));
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let url = format!("http://{}/v1", listener.local_addr()?);
        let task = tokio::spawn(async move { axum::serve(listener, app).await });
        Ok(Self { url, calls, task })
    }
}

impl Drop for Upstream {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn upstream(
    State(calls): State<Arc<Mutex<Vec<Value>>>>,
    Json(body): Json<Value>,
) -> Response {
    let model = body["model"].as_str().unwrap_or_default();
    let attempt = {
        let mut calls = calls.lock().await;
        calls.push(body.clone());
        calls.iter().filter(|call| call["model"] == model).count()
    };
    if model == "backoff" || (model == "retry-body" && attempt == 1) {
        if model == "retry-body" {
            tokio::time::sleep(Duration::from_millis(300)).await;
        }
        let delay = if model == "backoff" { "1" } else { "0" };
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            [("retry-after", delay)],
            "retry",
        )
            .into_response();
    }
    if model == "stalled" {
        tokio::time::sleep(Duration::from_secs(3)).await;
    }
    if body["stream"] == true {
        let model = model.to_string();
        let stream = async_stream::stream! {
            let first = json!({"id":"test", "model":model, "choices":[{"index":0,"delta":{"role":"assistant","content":"PONG"},"finish_reason":null}]});
            yield Ok::<_, Infallible>(Bytes::from(format!("data: {first}\n\n")));
            let delay = match model.as_str() { "slow-body" => 3000, "streaming" => 150, _ => 5 };
            tokio::time::sleep(Duration::from_millis(delay)).await;
            if model == "stream-error" {
                yield Ok(Bytes::from_static(b"data: {\"error\":{\"message\":\"upstream stream failed\"}}\n\ndata: [DONE]\n\n"));
                return;
            }
            let last = json!({"id":"test", "model":model, "choices":[{"index":0,"delta":{},"finish_reason":"stop"}]});
            yield Ok(Bytes::from(format!("data: {last}\n\ndata: [DONE]\n\n")));
        };
        return (
            [("content-type", "text/event-stream")],
            Body::from_stream(stream),
        )
            .into_response();
    }
    let content = if model == "retry-body" {
        r#"{"crux":"bounded task","primary_rule":"SUP-1","capability_boundary":"supported","p_solve":0.9}"#
    } else {
        "PONG"
    };
    let response = json!({"id":"test","object":"chat.completion","model":model,
        "choices":[{"index":0,"message":{"role":"assistant","content":content},"finish_reason":"stop"}],
        "usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}});
    if model == "slow-body" || model == "retry-body" {
        let delay = if model == "retry-body" { 450 } else { 3000 };
        let stream = async_stream::stream! {
            yield Ok::<_, Infallible>(Bytes::from_static(b" "));
            tokio::time::sleep(Duration::from_millis(delay)).await;
            yield Ok(Bytes::from(response.to_string()));
        };
        return (
            [("content-type", "application/json")],
            Body::from_stream(stream),
        )
            .into_response();
    }
    Json(response).into_response()
}

fn app(
    upstream: &Upstream,
    mode: &str,
    judge: &str,
    weak: &str,
    timeout: u64,
) -> TestResult<Router> {
    let route = match mode {
        "escalation" => {
            "type = \"llm_classifier\"\nmode = \"escalation\"\nclassifier_target = \"judge\"\nstrong_target = \"strong\"\nweak_target = \"weak\"\nescalation = { confirmations = 1 }"
        }
        "candidates" => {
            "type = \"random\"\ntargets = [\"weak\", \"strong\"]\nweights = [1000000, 1]\nseed = 17"
        }
        "terminal" => "type = \"passthrough\"\ntarget = \"weak\"",
        _ => {
            "type = \"llm_classifier\"\nclassifier_target = \"judge\"\nstrong_target = \"strong\"\nweak_target = \"weak\"\nbase_threshold = 0.5"
        }
    };
    let mut config = tempfile::Builder::new().suffix(".toml").tempfile()?;
    write!(
        config,
        r#"
schema_version = 1
[llm_clients.http]
format = "openai_chat"
base_url = "{}"
max_retries = 1
timeout_ms = {timeout}
[targets]
judge = {{ id = "{judge}", llm_client = "http", extra_body = {{ stream = {judge_stream} }} }}
weak = {{ id = "{weak}", llm_client = "http" }}
strong = {{ id = "strong", llm_client = "http" }}
[routes.test]
id = "test"
{route}
"#,
        upstream.url,
        judge_stream = judge == "stream-error",
    )?;
    Ok(build_switchyard_router(load_server_state(config.path())?))
}

async fn send(app: &Router, path: &str, body: Option<Value>) -> TestResult<(StatusCode, String)> {
    let (method, body) = match body {
        Some(body) => ("POST", Body::from(serde_json::to_vec(&body)?)),
        None => ("GET", Body::empty()),
    };
    let response = tokio::time::timeout(
        Duration::from_secs(2),
        app.clone().oneshot(
            Request::builder()
                .method(method)
                .uri(path)
                .header("content-type", "application/json")
                .body(body)?,
        ),
    )
    .await??;
    let status = response.status();
    let bytes = tokio::time::timeout(Duration::from_secs(2), response.into_body().collect())
        .await??
        .to_bytes();
    Ok((status, String::from_utf8(bytes.to_vec())?))
}

fn metric(text: &str, code: &str) -> f64 {
    text.lines()
        .find(|line| {
            line.starts_with("switchyard_upstream_attempts_total{")
                && line.contains(&format!("code=\"{code}\""))
        })
        .and_then(|line| line.split_whitespace().last())
        .and_then(|value| value.parse().ok())
        .unwrap_or_default()
}

async fn check(
    upstream: &Upstream,
    app: &Router,
    endpoint: &str,
    expected_calls: &[&str],
    expected_counts: [f64; 3],
) -> TestResult<(StatusCode, String)> {
    upstream.calls.lock().await.clear();
    let (_, before) = send(app, "/metrics", None).await?;
    let mut body = if endpoint == "/v1/responses" {
        json!({"model":"test", "input":"Say PONG", "stream":true})
    } else {
        json!({"model":"test", "messages":[{"role":"user","content":"Say PONG"}], "max_tokens":32, "stream":true})
    };
    if endpoint == "/v1/decision" {
        body = json!({"input_format":"openai_chat", "request":body});
    }
    let result = send(app, endpoint, Some(body)).await?;
    let calls = upstream.calls.lock().await.clone();
    let actual: Vec<_> = calls
        .iter()
        .filter_map(|call| call["model"].as_str())
        .collect();
    assert_eq!(actual, expected_calls, "{endpoint}: {}", result.1);
    let (_, after) = send(app, "/metrics", None).await?;
    // Count HTTP 200, HTTP 503, and pre-response failures separately. A stream
    // that times out after HTTP 200 must not count as another attempt.
    let counts = ["200", "503", "none"].map(|code| metric(&after, code) - metric(&before, code));
    assert_eq!(counts, expected_counts, "{endpoint}: {}", result.1);
    Ok(result)
}

#[tokio::test]
async fn client_deadline_stops_routing_and_counts_attempts() -> TestResult {
    let upstream = Upstream::start().await?;
    let stalled = app(&upstream, "classifier", "stalled", "weak", 100)?;
    let streaming = app(&upstream, "terminal", "judge", "streaming", 500)?;
    let slow_stream = app(&upstream, "terminal", "judge", "slow-body", 100)?;
    for endpoint in [
        "/v1/chat/completions",
        "/v1/messages",
        "/v1/responses",
        "/v1/decision",
    ] {
        let (status, body) =
            check(&upstream, &stalled, endpoint, &["stalled"], [0., 0., 1.]).await?;
        assert_eq!(status, StatusCode::GATEWAY_TIMEOUT, "{body}");
        assert!(body.contains("100 ms"), "{body}");
        if endpoint == "/v1/decision" {
            continue;
        }
        let (status, body) = check(
            &upstream,
            &streaming,
            endpoint,
            &["streaming"],
            [1., 0., 0.],
        )
        .await?;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(body.contains("PONG"), "{body}");
        let success = match endpoint {
            "/v1/messages" => "message_stop",
            "/v1/responses" => "response.completed",
            _ => "[DONE]",
        };
        assert!(body.contains(success), "{body}");
        let (status, body) = check(
            &upstream,
            &slow_stream,
            endpoint,
            &["slow-body"],
            [1., 0., 0.],
        )
        .await?;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(body.contains("PONG") && body.contains("100 ms"), "{body}");
        assert!(!body.contains(success), "{body}");
        assert!(!body.contains("\"finish_reason\":\"stop\""), "{body}");
    }

    // These response and retry failures use the same client code for every endpoint.
    for (judge, expected_calls, counts) in [
        ("slow-body", &["slow-body"][..], [0., 0., 1.]),
        ("backoff", &["backoff"][..], [0., 1., 0.]),
        // The first attempt waits 300 ms and the retry body waits 450 ms.
        // Each fits the 600 ms deadline alone; together they exceed it.
        (
            "retry-body",
            &["retry-body", "retry-body"][..],
            [0., 1., 1.],
        ),
    ] {
        let app = app(&upstream, "classifier", judge, "weak", 600)?;
        let (status, body) = check(
            &upstream,
            &app,
            "/v1/chat/completions",
            expected_calls,
            counts,
        )
        .await?;
        assert_eq!(status, StatusCode::GATEWAY_TIMEOUT, "{judge}: {body}");
        assert!(body.contains("600 ms"), "{body}");
    }
    let failed_stream = app(&upstream, "classifier", "stream-error", "weak", 100)?;
    let (status, body) = check(
        &upstream,
        &failed_stream,
        "/v1/chat/completions",
        &["stream-error"],
        [1., 0., 0.],
    )
    .await?;
    assert_eq!(status, StatusCode::BAD_GATEWAY, "{body}");
    for (mode, weak, counts) in [
        ("escalation", "slow-body", [1., 0., 0.]),
        ("candidates", "stalled", [0., 0., 1.]),
    ] {
        let app = app(&upstream, mode, "judge", weak, 100)?;
        let (status, body) =
            check(&upstream, &app, "/v1/chat/completions", &[weak], counts).await?;
        assert_eq!(status, StatusCode::GATEWAY_TIMEOUT, "{mode}: {body}");
        if mode == "escalation" {
            assert_eq!(upstream.calls.lock().await[0]["stream"], true);
        }
    }
    Ok(())
}
