// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the libsy Rust server.

use std::collections::{BTreeMap, HashSet};
use std::convert::Infallible;
use std::error::Error;
use std::io::Write;
use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, Request as HttpRequest, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response as HttpResponse};
use axum::routing::post;
use axum::{Json, Router};
use http_body_util::BodyExt;
use libsy::{Algorithm, Random};
use serde_json::{Value, json};
use switchyard_llm_client::{
    Backend, ClientRouter, HttpBackendConfig, ModelConfig, TranslatingLlmClient,
};
use switchyard_protocol::ModelId;
use switchyard_protocol::RoutedLlmClient;
use switchyard_server::config::load_server_state;
use switchyard_server::{ServerState, build_switchyard_router};
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tower::ServiceExt;

type TestError = Box<dyn Error + Send + Sync>;
type TestResult<T = ()> = Result<T, TestError>;

const ROUTE_MODEL: &str = "switchyard/random";
const VERSION: &str = env!("CARGO_PKG_VERSION");

struct MockUpstream {
    base_url: String,
    calls: Arc<Mutex<Vec<Value>>>,
    task: JoinHandle<()>,
}

impl MockUpstream {
    async fn start() -> TestResult<Self> {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let app = Router::new()
            .route("/v1/chat/completions", post(upstream_chat))
            .route(
                "/v1/messages",
                post(upstream_messages_requires_forwarded_oauth),
            )
            .route(
                "/v1/responses",
                post(upstream_responses_requires_forwarded_auth),
            )
            .route("/capture", post(upstream_redirect_capture))
            .route("/v1/messages/count_tokens", post(upstream_count_tokens))
            .layer(DefaultBodyLimit::disable())
            .with_state(Arc::clone(&calls));
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        let task = tokio::spawn(async move {
            if let Err(error) = axum::serve(listener, app).await {
                tracing::error!(error = %error, "mock upstream stopped");
            }
        });
        Ok(Self {
            base_url: format!("http://{addr}/v1"),
            calls,
            task,
        })
    }

    /// The upstream model id of every request this upstream received, in order.
    async fn models(&self) -> Vec<String> {
        self.calls
            .lock()
            .await
            .iter()
            .filter_map(|call| call.get("model")?.as_str().map(str::to_string))
            .collect()
    }
}

impl Drop for MockUpstream {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn upstream_chat(
    State(calls): State<Arc<Mutex<Vec<Value>>>>,
    Json(body): Json<Value>,
) -> HttpResponse {
    calls.lock().await.push(body.clone());
    if body["messages"][0]["content"] == "fail" {
        return (
            StatusCode::IM_A_TEAPOT,
            Json(json!({"error": {"message": "upstream rejected request"}})),
        )
            .into_response();
    }
    if body["messages"][0]["content"] == "auth-fail" {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({"error": {"message": "upstream authentication failed"}})),
        )
            .into_response();
    }

    let model = body["model"].as_str().unwrap_or("unknown").to_string();
    let prompt = body["messages"][0]["content"].as_str().unwrap_or("");
    if (model == "model/weak" && prompt == "unavailable") || prompt == "all-unavailable" {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": {"message": "upstream is unavailable"}})),
        )
            .into_response();
    }
    if model == "model/weak" && body["messages"][0]["content"] == "overflow" {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": {
                    "code": "context_length_exceeded",
                    "message": "request exceeds this model's context window"
                }
            })),
        )
            .into_response();
    }
    if body["stream"].as_bool() == Some(true) {
        if body["messages"][0]["content"] == "stream-error" {
            let events = [
                json!({"id": "chatcmpl-stream-error", "model": model, "choices": [{"index": 0, "delta": {"role": "assistant"}}]}).to_string(),
                json!({"id": "chatcmpl-stream-error", "model": model, "choices": [{"index": 0, "delta": {"content": "before"}}]}).to_string(),
                json!({"id": "chatcmpl-stream-error", "model": model, "choices": [{"index": 0, "delta": {"content": "still here"}}], "usage": {"prompt_tokens": 6, "completion_tokens": 2, "total_tokens": 8}}).to_string(),
                json!({"error": {"message": "upstream stream failed", "type": "server_error"}}).to_string(),
            ];
            let stream = futures_util::stream::iter(
                events
                    .into_iter()
                    .map(|data| Ok::<Event, Infallible>(Event::default().data(data))),
            );
            return Sse::new(stream).into_response();
        }
        let events = [
            json!({"id": "chatcmpl-stream", "model": model, "choices": [{"index": 0, "delta": {"role": "assistant"}}]}).to_string(),
            json!({"id": "chatcmpl-stream", "model": model, "choices": [{"index": 0, "delta": {"content": "hello"}}]}).to_string(),
            json!({"id": "chatcmpl-stream", "model": model, "choices": [{"index": 0, "delta": {"content": "-partial"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6, "prompt_tokens_details": {"cached_tokens": 2, "cache_creation_tokens": 1}}}).to_string(),
            json!({"id": "chatcmpl-stream", "model": model, "choices": [{"index": 0, "delta": {"content": "-final"}}], "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17, "prompt_tokens_details": {"cached_tokens": 7, "cache_creation_tokens": 2}, "completion_tokens_details": {"reasoning_tokens": 3}}}).to_string(),
            json!({"id": "chatcmpl-stream", "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}).to_string(),
            "[DONE]".to_string(),
        ];
        let stream = futures_util::stream::iter(
            events
                .into_iter()
                .map(|data| Ok::<Event, Infallible>(Event::default().data(data))),
        );
        return Sse::new(stream).into_response();
    }

    let custom_target_schema = body
        .pointer("/response_format/json_schema/schema/properties/decision/properties/target")
        .is_some();
    let requests_invalid_verdict = body["messages"].as_array().is_some_and(|messages| {
        messages.iter().any(|message| {
            message["content"]
                .as_str()
                .is_some_and(|content| content.contains("invalid verdict"))
        })
    });
    let requests_schema_invalid_verdict = body["messages"].as_array().is_some_and(|messages| {
        messages.iter().any(|message| {
            message["content"]
                .as_str()
                .is_some_and(|content| content.contains("schema-invalid verdict"))
        })
    });
    let content = if model == "model/classifier" && custom_target_schema {
        if requests_invalid_verdict {
            r#"{"decision":{"target":"unknown"}}"#
        } else {
            r#"{"decision":{"target":"premium"}}"#
        }
    } else if model == "model/classifier"
        && body
            .pointer("/response_format/json_schema/schema/properties/escalate")
            .is_some()
    {
        r#"{"escalate":false,"reason":"making progress"}"#
    } else if model == "model/classifier" && requests_schema_invalid_verdict {
        r#"{"crux":"bounded task","primary_rule":"SUP-1","capability_boundary":"supported","p_solve":0.1,"unexpected":true}"#
    } else if model == "model/classifier" {
        r#"{"crux":"bounded task","primary_rule":"SUP-1","capability_boundary":"supported","p_solve":0.9}"#
    } else {
        "ok"
    };
    Json(json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": 7}
        }
    }))
    .into_response()
}

async fn upstream_messages_requires_forwarded_oauth(
    State(calls): State<Arc<Mutex<Vec<Value>>>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> HttpResponse {
    calls.lock().await.push(body.clone());
    let has_expected_headers = headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        == Some("Bearer claude-oauth-token")
        && headers
            .get("anthropic-beta")
            .and_then(|value| value.to_str().ok())
            == Some("oauth-2025-04-20")
        && headers
            .get("anthropic-version")
            .and_then(|value| value.to_str().ok())
            == Some("2023-06-01")
        && !headers.contains_key("chatgpt-account-id")
        && !headers.contains_key("x-openai-fedramp");
    if !has_expected_headers {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({"error": {"message": "missing forwarded Anthropic OAuth headers"}})),
        )
            .into_response();
    }
    Json(json!({
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": body["model"],
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {"input_tokens": 1, "output_tokens": 1}
    }))
    .into_response()
}

async fn upstream_responses_requires_forwarded_auth(
    State(calls): State<Arc<Mutex<Vec<Value>>>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> HttpResponse {
    calls.lock().await.push(body.clone());
    if headers.contains_key("x-test-redirect") {
        return (StatusCode::TEMPORARY_REDIRECT, [("location", "/capture")]).into_response();
    }
    if headers.contains_key("x-test-echo-auth") {
        let authorization = headers
            .get("authorization")
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default();
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({"error": {"message": authorization}})),
        )
            .into_response();
    }
    let has_expected_headers = headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        == Some("Bearer codex-login-token")
        && headers
            .get("chatgpt-account-id")
            .and_then(|value| value.to_str().ok())
            == Some("account-123")
        && headers
            .get("x-openai-fedramp")
            .and_then(|value| value.to_str().ok())
            == Some("true")
        && !headers.contains_key("x-api-key")
        && !headers.contains_key("anthropic-beta");
    if !has_expected_headers {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({"error": {"message": "missing forwarded OpenAI login"}})),
        )
            .into_response();
    }
    Json(json!({
        "id": "resp_test",
        "object": "response",
        "model": body["model"],
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "ok"}]
        }],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
    }))
    .into_response()
}

async fn upstream_redirect_capture(
    State(calls): State<Arc<Mutex<Vec<Value>>>>,
    headers: HeaderMap,
) -> HttpResponse {
    calls.lock().await.push(json!({
        "redirected": true,
        "has_authorization": headers.contains_key("authorization")
    }));
    StatusCode::OK.into_response()
}

async fn upstream_count_tokens(
    State(calls): State<Arc<Mutex<Vec<Value>>>>,
    Json(body): Json<Value>,
) -> HttpResponse {
    calls.lock().await.push(body.clone());
    Json(json!({"input_tokens": 7})).into_response()
}

fn random_state(base_url: &str, routes: &[(&str, &[&str])]) -> TestResult<ServerState> {
    let backend = Backend::OpenAiChat(HttpBackendConfig {
        base_url: base_url.to_string(),
        api_key: Some("test-key".to_string()),
        forward_auth: false,
        extra_headers: BTreeMap::new(),
        extra_body: BTreeMap::new(),
        max_retries: 0,
    });
    let target_models = routes
        .iter()
        .flat_map(|(_, targets)| targets.iter().copied())
        .collect::<HashSet<_>>();
    let model_configs = target_models
        .into_iter()
        .map(|model| ModelConfig::new(model, backend.clone(), None))
        .collect::<Vec<_>>();
    let client: Arc<dyn RoutedLlmClient> = Arc::new(TranslatingLlmClient::new(&model_configs)?);
    let entries = routes
        .iter()
        .map(|(route_model, targets)| {
            let target_set = targets.iter().map(|model| ModelId::from(*model)).collect();
            let algorithm: Arc<dyn Algorithm> = Arc::new(Random::new(target_set, None, None)?);
            Ok((
                ModelId::from(*route_model),
                algorithm,
                ClientRouter::single(Arc::clone(&client)),
            ))
        })
        .collect::<TestResult<Vec<_>>>()?;
    Ok(ServerState::new(entries)?)
}

async fn test_app(routes: &[(&str, &[&str])]) -> TestResult<(MockUpstream, Router)> {
    let upstream = MockUpstream::start().await?;
    let app = build_switchyard_router(random_state(&upstream.base_url, routes)?);
    Ok((upstream, app))
}

fn empty_token_totals() -> Value {
    json!({
        "prompt": 0,
        "completion": 0,
        "cached": 0,
        "cache_creation": 0,
        "reasoning": 0,
        "total": 0
    })
}

#[tokio::test]
async fn stats_exposes_the_exact_empty_schema_and_no_legacy_alias() -> TestResult {
    let (_upstream, app) = test_app(&[(ROUTE_MODEL, &["model/a"])]).await?;
    let response = send(&app, "GET", "/v1/stats", None).await?;
    assert_eq!(response.status, StatusCode::OK);
    assert_eq!(
        response.json()?,
        json!({
            "total_requests": 0,
            "total_errors": 0,
            "total_tokens": empty_token_totals(),
            "models": {},
            "routing_overhead": {
                "count": 0,
                "total_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "avg_ms": 0.0,
                "p50_ms": 0.0,
                "p99_ms": 0.0
            },
            "routing_fallbacks": {
                "context_window": 0,
                "unavailable": 0
            },
            "classifier": {
                "total_requests": 0,
                "total_errors": 0,
                "total_tokens": empty_token_totals(),
                "models": {},
            },
            "algorithm_stats": {},
        })
    );
    assert_eq!(
        send(&app, "GET", "/v1/routing/stats", None).await?.status,
        StatusCode::NOT_FOUND
    );
    Ok(())
}

#[tokio::test]
async fn stats_accumulates_buffered_success_error_and_shared_routes() -> TestResult {
    let (_upstream, app) = test_app(&[
        ("switchyard/one", &["gemini-3.5-flash"]),
        ("switchyard/two", &["model/unknown"]),
    ])
    .await?;
    for route in ["switchyard/one", "switchyard/two"] {
        assert_eq!(
            send(
                &app,
                "POST",
                "/v1/chat/completions",
                Some(json!({
                    "model": route,
                    "messages": [{"role": "user", "content": "hello"}]
                })),
            )
            .await?
            .status,
            StatusCode::OK
        );
    }
    assert_eq!(
        send(
            &app,
            "POST",
            "/v1/chat/completions",
            Some(json!({
                "model": "switchyard/one",
                "messages": [{"role": "user", "content": "fail"}]
            })),
        )
        .await?
        .status,
        StatusCode::IM_A_TEAPOT
    );

    let stats = send(&app, "GET", "/v1/stats", None).await?.json()?;
    assert_eq!(stats["total_requests"], 3);
    assert_eq!(stats["total_errors"], 1);
    assert_eq!(
        stats["total_tokens"],
        json!({
            "prompt": 20,
            "completion": 4,
            "cached": 14,
            "cache_creation": 0,
            "reasoning": 0,
            "total": 24
        })
    );
    assert_eq!(stats["models"]["gemini-3.5-flash"]["calls"], 1);
    assert_eq!(stats["models"]["gemini-3.5-flash"]["errors"], 1);
    assert_eq!(stats["models"]["model/unknown"]["calls"], 1);
    assert_eq!(stats["routing_overhead"]["count"], 2);
    Ok(())
}

#[tokio::test]
async fn stats_reset_returns_confirmation_and_clears_all_stats() -> TestResult {
    let (_upstream, app) = test_app(&[(ROUTE_MODEL, &["model/a"])]).await?;
    assert_eq!(
        send(
            &app,
            "POST",
            "/v1/chat/completions",
            Some(json!({
                "model": ROUTE_MODEL,
                "messages": [{"role": "user", "content": "hello"}]
            })),
        )
        .await?
        .status,
        StatusCode::OK
    );

    let reset = send(&app, "POST", "/v1/stats/reset", None).await?;
    assert_eq!(reset.status, StatusCode::OK);
    assert_eq!(reset.json()?, json!({"status": "reset"}));

    let stats = send(&app, "GET", "/v1/stats", None).await?.json()?;
    assert_eq!(stats["total_requests"], 0);
    assert_eq!(stats["total_errors"], 0);
    assert_eq!(stats["total_tokens"], empty_token_totals());
    assert_eq!(stats["models"], json!({}));
    assert_eq!(stats["routing_overhead"]["count"], 0);
    assert_eq!(stats["classifier"]["total_requests"], 0);
    assert_eq!(stats["classifier"]["models"], json!({}));
    Ok(())
}

#[tokio::test]
async fn metrics_exposes_switchyard_otel_instruments() -> TestResult {
    const MODEL: &str = "model/metrics-buffered";
    let (_upstream, app) = test_app(&[(ROUTE_MODEL, &[MODEL])]).await?;

    let before = send(&app, "GET", "/metrics", None).await?;
    assert_eq!(before.status, StatusCode::OK);
    assert_eq!(
        before
            .headers
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("text/plain; version=0.0.4; charset=utf-8")
    );
    let seeded = before.text()?;
    for expected in [
        "# TYPE switchyard_client_responses_total counter",
        "switchyard_client_responses_total{outcome=\"success\",",
        "switchyard_client_responses_total{outcome=\"retryable_error\",",
        "switchyard_client_responses_total{outcome=\"other_error\",",
        "# TYPE switchyard_upstream_attempts_total counter",
        "switchyard_upstream_attempts_total{code=\"200\",outcome=\"success\",",
        "switchyard_upstream_attempts_total{code=\"429\",outcome=\"retryable_error\",",
        "switchyard_upstream_attempts_total{code=\"500\",outcome=\"retryable_error\",",
        "switchyard_upstream_attempts_total{code=\"504\",outcome=\"retryable_error\",",
        "switchyard_upstream_attempts_total{code=\"none\",outcome=\"retryable_error\",",
        "# TYPE switchyard_router_retry_recovered_total counter",
        "switchyard_router_retry_recovered_total{otel_scope_name=\"switchyard\"} 0",
    ] {
        assert!(
            seeded.contains(expected),
            "missing seeded {expected:?} in metrics:\n{seeded}"
        );
    }

    let response = send(
        &app,
        "POST",
        "/v1/chat/completions",
        Some(json!({
            "model": ROUTE_MODEL,
            "messages": [{"role": "user", "content": "hello"}]
        })),
    )
    .await?;
    assert_eq!(response.status, StatusCode::OK);

    let after = send(&app, "GET", "/metrics", None).await?;
    let metrics = after.text()?;
    for expected in [
        "# TYPE switchyard_build_info gauge",
        &format!("switchyard_build_info{{version=\"{VERSION}\""),
        "# TYPE switchyard_total_requests gauge",
        "# TYPE switchyard_total_errors gauge",
        "# TYPE switchyard_requests_total counter",
        "# TYPE switchyard_model_call_latency_ms histogram",
        "switchyard_client_responses_total{outcome=\"success\",",
        "switchyard_upstream_attempts_total{code=\"200\",outcome=\"success\",",
        "# TYPE switchyard_runs_total counter",
        "# TYPE switchyard_llm_calls_total counter",
        "# TYPE switchyard_run_duration_ms histogram",
        "# TYPE switchyard_llm_call_duration_ms histogram",
        "# TYPE switchyard_prompt_tokens_total counter",
        "# TYPE switchyard_completion_tokens_total counter",
        "# TYPE switchyard_cached_tokens_total counter",
        "# TYPE switchyard_total_latency_ms histogram",
        "# TYPE switchyard_routing_overhead_ms histogram",
        "algorithm=\"random\"",
        &format!("selected_model=\"{MODEL}\""),
    ] {
        assert!(
            metrics.contains(expected),
            "missing {expected:?} in metrics:\n{metrics}"
        );
    }
    for (name, expected_delta) in [
        ("switchyard_prompt_tokens_total", 10.0),
        ("switchyard_completion_tokens_total", 2.0),
        ("switchyard_cached_tokens_total", 7.0),
        ("switchyard_total_latency_ms_count", 1.0),
    ] {
        assert_eq!(
            metric_delta(seeded, metrics, name, &[("model", MODEL)]),
            Some(expected_delta),
            "unexpected delta for {name}"
        );
    }
    // A sub-millisecond boundary exists only because of the server's bucket view.
    assert!(
        metric_line(
            metrics,
            "switchyard_routing_overhead_ms_bucket",
            &[("algorithm", "random"), ("le", "0.1")]
        )
        .is_some()
    );
    for metric in [
        "switchyard_model_call_latency_ms_bucket",
        "switchyard_total_latency_ms_bucket",
    ] {
        assert!(
            metric_line(metrics, metric, &[("model", MODEL), ("le", "300000")]).is_some(),
            "missing five-minute bucket for {metric}"
        );
    }
    assert!(
        metric_line(
            metrics,
            "switchyard_cache_creation_tokens_total",
            &[("model", MODEL)]
        )
        .is_none()
    );
    assert!(
        metric_line(
            metrics,
            "switchyard_reasoning_tokens_total",
            &[("model", MODEL)]
        )
        .is_none()
    );
    for metric in [
        "switchyard_prompt_tokens_total",
        "switchyard_completion_tokens_total",
        "switchyard_cached_tokens_total",
        "switchyard_total_latency_ms_count",
    ] {
        let line = metric_line(metrics, metric, &[("model", MODEL)])
            .ok_or_else(|| format!("missing {metric} series for {MODEL}"))?;
        assert!(!line.contains("tier="), "unexpected tier label in {line}");
    }
    Ok(())
}

#[tokio::test]
async fn accepts_requests_larger_than_the_axum_default_body_limit() -> TestResult {
    let (_upstream, app) = test_app(&[(ROUTE_MODEL, &["model/a"])]).await?;
    let content = "x".repeat(2 * 1024 * 1024);

    let response = send(
        &app,
        "POST",
        "/v1/chat/completions",
        Some(json!({
            "model": ROUTE_MODEL,
            "messages": [{"role": "user", "content": content}]
        })),
    )
    .await?;

    assert_eq!(response.status, StatusCode::OK);
    Ok(())
}

fn load_test_config(toml: &str) -> TestResult<ServerState> {
    let mut config = tempfile::Builder::new()
        .prefix("switchyard-server-config-")
        .suffix(".toml")
        .tempfile()?;
    config.write_all(toml.as_bytes())?;
    config.flush()?;
    Ok(load_server_state(config.path())?)
}

/// A `random` route that selects `first` before any request-local fallback.
fn fallback_state(base_url: &str) -> TestResult<ServerState> {
    load_test_config(&format!(
        r#"
schema_version = 1

[llm_clients.mock]
format = "openai_chat"
base_url = "{base_url}"
max_retries = 0

[targets.first]
id = "{first}"
llm_client = "mock"

[targets.second]
id = "{second}"
llm_client = "mock"

[routes.random]
id = "{ROUTE_MODEL}"
type = "random"
targets = ["first", "second"]
weights = [1, 0]
"#,
        first = "model/weak",
        second = "model/strong",
    ))
}

async fn send(app: &Router, method: &str, path: &str, body: Option<Value>) -> TestResult<Response> {
    send_with_headers(app, method, path, body, &[]).await
}

async fn send_with_headers(
    app: &Router,
    method: &str,
    path: &str,
    body: Option<Value>,
    headers: &[(&str, &str)],
) -> TestResult<Response> {
    let mut builder = HttpRequest::builder().method(method).uri(path);
    for (name, value) in headers {
        builder = builder.header(*name, *value);
    }
    let request_body = if let Some(body) = body {
        builder = builder.header("content-type", "application/json");
        Body::from(serde_json::to_vec(&body)?)
    } else {
        Body::empty()
    };
    let response = app.clone().oneshot(builder.body(request_body)?).await?;
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = response.into_body().collect().await?.to_bytes();
    Ok(Response {
        status,
        headers,
        bytes,
    })
}

struct Response {
    status: StatusCode,
    headers: axum::http::HeaderMap,
    bytes: Bytes,
}

impl Response {
    fn json(&self) -> TestResult<Value> {
        Ok(serde_json::from_slice(&self.bytes)?)
    }

    fn text(&self) -> TestResult<&str> {
        Ok(std::str::from_utf8(&self.bytes)?)
    }
}

fn metric_line<'a>(metrics: &'a str, name: &str, labels: &[(&str, &str)]) -> Option<&'a str> {
    metrics.lines().find(|line| {
        line.starts_with(name)
            && labels
                .iter()
                .all(|(key, value)| line.contains(&format!("{key}=\"{value}\"")))
    })
}

fn metric_value(metrics: &str, name: &str, labels: &[(&str, &str)]) -> Option<f64> {
    metric_line(metrics, name, labels)?
        .split_whitespace()
        .last()?
        .parse()
        .ok()
}

fn metric_delta(before: &str, after: &str, name: &str, labels: &[(&str, &str)]) -> Option<f64> {
    metric_value(after, name, labels)
        .map(|after| after - metric_value(before, name, labels).unwrap_or_default())
}

fn assert_in_order(haystack: &str, needles: &[&str]) {
    let mut remainder = haystack;
    for needle in needles {
        let offset = remainder
            .find(needle)
            .unwrap_or_else(|| panic!("missing {needle:?} after prior events in:\n{haystack}"));
        remainder = &remainder[offset + needle.len()..];
    }
}

/// A route is a synthetic model (`switchyard/classify`) with no upstream of its own; its
/// algorithm picks real targets, and *those* name the client. One request can emit several
/// `Step::CallModel` for different targets, and two targets may sit on different
/// `[llm_clients.*]` sections — here the judge is on one provider and the serving models on
/// another. Pin that each call reaches its own target's upstream, rather than one client
/// chosen per route serving all of them.
#[tokio::test]
async fn each_target_in_one_request_is_served_by_its_own_client() -> TestResult {
    let judge_upstream = MockUpstream::start().await?;
    let model_upstream = MockUpstream::start().await?;
    let state = load_test_config(&format!(
        r#"
schema_version = 1

[llm_clients.judge_provider]
format = "openai_chat"
base_url = "{judge_url}"

[llm_clients.model_provider]
format = "openai_chat"
base_url = "{model_url}"

[targets.judge]
id = "model/judge"
llm_client = "judge_provider"

[targets.strong]
id = "model/strong"
llm_client = "model_provider"

[targets.weak]
id = "model/weak"
llm_client = "model_provider"

[routes.classify]
id = "switchyard/classify"
type = "llm_classifier"
classifier_target = "judge"
strong_target = "strong"
weak_target = "weak"
base_threshold = 0.5
"#,
        judge_url = judge_upstream.base_url,
        model_url = model_upstream.base_url,
    ))?;
    let app = build_switchyard_router(state);

    let response = send(
        &app,
        "POST",
        "/v1/chat/completions",
        Some(json!({
            "model": "switchyard/classify",
            "messages": [{"role": "user", "content": "hi"}]
        })),
    )
    .await?;
    assert_eq!(response.status, StatusCode::OK);

    // The judge call went to the judge's provider and nowhere else; the serving call went to
    // the models' provider. A single per-route client would have sent both to one upstream.
    assert_eq!(
        judge_upstream.models().await,
        vec!["model/judge".to_string()]
    );
    let served = model_upstream.models().await;
    assert_eq!(
        served.len(),
        1,
        "expected exactly one routed call: {served:?}"
    );
    assert!(
        served[0] == "model/weak" || served[0] == "model/strong",
        "routed call went to {served:?}"
    );
    Ok(())
}

/// A critical tool error must reach the stage router's signal scorer, which reads
/// the decoded conversation. The endpoint records no inbound wire format, so a
/// scorer that parsed the raw body instead would find nothing and route every turn
/// as if the conversation had no signals at all.
#[tokio::test]
async fn stage_route_escalates_on_a_signal_in_the_conversation() -> TestResult {
    let upstream = MockUpstream::start().await?;
    let state = load_test_config(&format!(
        r#"
schema_version = 1

[llm_clients.upstream]
format = "openai_chat"
base_url = "{base_url}"

[targets.strong]
id = "model/stats-strong"
llm_client = "upstream"

[targets.weak]
id = "model/stats-weak"
llm_client = "upstream"

[routes.stage]
id = "switchyard/stage"
type = "stage_router"
capable_target = "strong"
efficient_target = "weak"
picker = "efficient_first"
confidence_threshold = 0.5
"#,
        base_url = upstream.base_url
    ))?;
    let app = build_switchyard_router(state);

    let response = send(
        &app,
        "POST",
        "/v1/chat/completions",
        Some(json!({
            "model": "switchyard/stage",
            "messages": [
                {"role": "user", "content": "fix the build"},
                {"role": "assistant", "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "Bash", "arguments": "{\"command\": \"cargo test\"}"}
                }]},
                {"role": "tool", "tool_call_id": "call_1", "content": "fatal runtime error: out of memory"},
            ]
        })),
    )
    .await?;

    assert_eq!(response.status, StatusCode::OK);
    assert_eq!(
        response
            .headers
            .get("x-model-router-selected-model")
            .and_then(|value| value.to_str().ok()),
        Some("model/stats-strong"),
        "a critical error should escalate on the signals alone"
    );
    let stats = send(&app, "GET", "/v1/stats", None).await?.json()?;
    assert_eq!(
        stats["algorithm_stats"]["stage_router"]["routing_decisions"]["override"]["targets"]["model/stats-strong"],
        1
    );
    Ok(())
}

#[tokio::test]
async fn toml_config_constructs_and_serves_multiple_algorithms() -> TestResult {
    let upstream = MockUpstream::start().await?;
    let state = load_test_config(&format!(
        r#"
schema_version = 1

[llm_clients.upstream]
format = "openai_chat"
base_url = "{base_url}"

[targets.classifier]
id = "model/classifier"
llm_client = "upstream"

[targets.strong]
id = "model/strong"
llm_client = "upstream"

[targets.weak]
id = "model/weak"
llm_client = "upstream"

[routes.random]
id = "switchyard/random"
type = "random"
targets = ["weak"]

[routes.classifier]
id = "switchyard/classifier"
type = "llm_classifier"
classifier_target = "classifier"
strong_target = "strong"
weak_target = "weak"
base_threshold = 0.5

[routes.passthrough]
id = "switchyard/passthrough"
type = "passthrough"
target = "weak"

[routes.stage]
id = "switchyard/stage"
type = "stage_router"
capable_target = "strong"
efficient_target = "weak"
picker = "efficient_first"
confidence_threshold = 0.5
recent_turn_window = 3
capable_system_prompt = "diagnose before you edit"
efficient_system_prompt = "follow the settled plan"

[routes.stage.handoff_notes]
escalation_note = "the previous model was stalling"

[routes.stage.classifier]
target = "classifier"
base_threshold = 0.5
"#,
        base_url = upstream.base_url
    ))?;
    let app = build_switchyard_router(state);

    for (route, selected) in [
        ("switchyard/random", "model/weak"),
        ("switchyard/classifier", "model/weak"),
        ("switchyard/passthrough", "model/weak"),
    ] {
        let response = send(
            &app,
            "POST",
            "/v1/chat/completions",
            Some(json!({
                "model": route,
                "messages": [{"role": "user", "content": "hi"}]
            })),
        )
        .await?;
        assert_eq!(response.status, StatusCode::OK);
        assert_eq!(
            response
                .headers
                .get("x-model-router-selected-model")
                .and_then(|value| value.to_str().ok()),
            Some(selected)
        );
    }

    let calls = upstream.calls.lock().await;
    assert_eq!(calls.len(), 4);
    assert_eq!(calls[0]["model"], "model/weak");
    assert_eq!(calls[1]["model"], "model/classifier");
    assert_eq!(calls[2]["model"], "model/weak");
    assert_eq!(calls[3]["model"], "model/weak");
    drop(calls);

    let stats = send(&app, "GET", "/v1/stats", None).await?.json()?;
    assert_eq!(stats["total_requests"], 3);
    assert_eq!(stats["models"]["model/weak"]["calls"], 3);
    assert_eq!(stats["classifier"]["total_requests"], 1);
    assert_eq!(
        stats["classifier"]["models"]["model/classifier"]["calls"],
        1
    );
    assert_eq!(stats["classifier"]["total_tokens"]["prompt"], 10);
    Ok(())
}

#[tokio::test]
async fn custom_classifier_routes_four_targets_and_falls_back_on_an_invalid_verdict() -> TestResult
{
    let upstream = MockUpstream::start().await?;
    let state = load_test_config(&format!(
        r#"
schema_version = 1

[llm_clients.upstream]
format = "openai_chat"
base_url = "{base_url}"

[targets.classifier]
id = "model/classifier"
llm_client = "upstream"

[targets.strong]
id = "model/strong"
llm_client = "upstream"

[targets.middle]
id = "model/middle"
llm_client = "upstream"

[targets.premium]
id = "model/premium"
llm_client = "upstream"

[targets.weak]
id = "model/weak"
llm_client = "upstream"

[routes.custom]
id = "switchyard/custom"
type = "llm_classifier"
mode = "custom"
classifier_target = "classifier"
targets = ["weak", "middle", "strong", "premium"]
default_target = "strong"
prompt = "CUSTOM MULTI TARGET"
response_schema = '''
{{
  "type": "object",
  "properties": {{
    "decision": {{
      "type": "object",
      "properties": {{
        "target": {{"type": "string", "enum": ["weak", "middle", "strong", "premium"]}}
      }},
      "required": ["target"],
      "additionalProperties": false
    }}
  }},
  "required": ["decision"],
  "additionalProperties": false
}}
'''

[routes.custom.policy]
type = "target_selector"
selector = "/decision/target"
"#,
        base_url = upstream.base_url
    ))?;
    let app = build_switchyard_router(state);

    for (task, selected) in [
        ("route this task", "model/premium"),
        ("return an invalid verdict", "model/strong"),
    ] {
        let response = send(
            &app,
            "POST",
            "/v1/chat/completions",
            Some(json!({
                "model": "switchyard/custom",
                "messages": [{"role": "user", "content": task}]
            })),
        )
        .await?;
        assert_eq!(response.status, StatusCode::OK);
        assert_eq!(
            response
                .headers
                .get("x-model-router-selected-model")
                .and_then(|value| value.to_str().ok()),
            Some(selected)
        );
    }

    let calls = upstream.calls.lock().await;
    let judge_call = calls
        .iter()
        .find(|call| call["model"] == "model/classifier")
        .ok_or("custom classifier target was not called")?;
    let prompt = judge_call["messages"][0]["content"]
        .as_str()
        .ok_or("custom classifier prompt was not text")?;
    assert_eq!(prompt, "CUSTOM MULTI TARGET");
    assert_eq!(judge_call["response_format"]["type"], "json_schema");
    assert_eq!(
        judge_call["response_format"]["json_schema"]["name"],
        "switchyard_classifier_response"
    );
    assert_eq!(judge_call["response_format"]["json_schema"]["strict"], true);
    assert_eq!(
        judge_call["response_format"]["json_schema"]["schema"]["properties"]["decision"]["properties"]
            ["target"]["enum"],
        json!(["weak", "middle", "strong", "premium"])
    );
    Ok(())
}

#[tokio::test]
async fn classifier_contract_overrides_reach_every_server_mode() -> TestResult {
    let upstream = MockUpstream::start().await?;
    let state = load_test_config(&format!(
        r#"
schema_version = 1

[llm_clients.upstream]
format = "openai_chat"
base_url = "{base_url}"

[targets.classifier]
id = "model/classifier"
llm_client = "upstream"

[targets.strong]
id = "model/strong"
llm_client = "upstream"

[targets.weak]
id = "model/weak"
llm_client = "upstream"

[routes.capability]
id = "switchyard/capability"
type = "llm_classifier"
mode = "capability"
classifier_target = "classifier"
strong_target = "strong"
weak_target = "weak"
base_threshold = 0.5
prompt = "CUSTOM CAPABILITY"
response_format_type = "json_object"

[routes.escalation]
id = "switchyard/escalation"
type = "llm_classifier"
mode = "escalation"
classifier_target = "classifier"
strong_target = "strong"
weak_target = "weak"
prompt = "CUSTOM ESCALATION"
response_format_type = "json_object"
escalation = {{ confirmations = 1 }}

[routes.stage]
id = "switchyard/stage"
type = "stage_router"
capable_target = "strong"
efficient_target = "weak"
picker = "efficient_first"
confidence_threshold = 1.0

[routes.stage.classifier]
target = "classifier"
base_threshold = 0.5
prompt = "CUSTOM STAGE"
"#,
        base_url = upstream.base_url
    ))?;
    let app = build_switchyard_router(state);

    for (route, prompt_prefix, schema_field, json_object) in [
        (
            "switchyard/capability",
            "CUSTOM CAPABILITY",
            "p_solve",
            true,
        ),
        (
            "switchyard/escalation",
            "CUSTOM ESCALATION",
            "escalate",
            true,
        ),
        ("switchyard/stage", "CUSTOM STAGE", "p_solve", false),
    ] {
        upstream.calls.lock().await.clear();
        let response = send(
            &app,
            "POST",
            "/v1/chat/completions",
            Some(json!({
                "model": route,
                "messages": [{"role": "user", "content": "bounded task"}]
            })),
        )
        .await?;

        assert_eq!(response.status, StatusCode::OK);
        let calls = upstream.calls.lock().await;
        let judge_call = calls
            .iter()
            .find(|call| call["model"] == "model/classifier")
            .ok_or("classifier target was not called")?;
        let prompt = judge_call["messages"][0]["content"]
            .as_str()
            .ok_or("classifier prompt was not text")?;
        assert!(prompt.starts_with(prompt_prefix), "{route}: {prompt}");
        if json_object {
            assert_eq!(
                judge_call["response_format"],
                json!({"type": "json_object"}),
                "{route}: {judge_call}"
            );
            assert!(prompt.contains("JSON Schema"), "{route}: {prompt}");
            assert!(
                prompt.contains(&format!("\"{schema_field}\"")),
                "{route}: missing {schema_field} in {prompt}"
            );
        } else {
            assert!(
                judge_call["response_format"]["json_schema"]["schema"]["properties"]
                    .get(schema_field)
                    .is_some(),
                "{route}: missing {schema_field} in {judge_call}"
            );
        }
    }
    Ok(())
}

#[tokio::test]
async fn stage_classifier_can_request_json_object_output() -> TestResult {
    let upstream = MockUpstream::start().await?;
    let state = load_test_config(&format!(
        r#"
schema_version = 1

[llm_clients.upstream]
format = "openai_chat"
base_url = "{base_url}"

[targets.classifier]
id = "model/classifier"
llm_client = "upstream"

[targets.strong]
id = "model/strong"
llm_client = "upstream"

[targets.weak]
id = "model/weak"
llm_client = "upstream"

[routes.stage]
id = "switchyard/stage"
type = "stage_router"
capable_target = "strong"
efficient_target = "weak"
picker = "efficient_first"
confidence_threshold = 1.0

[routes.stage.classifier]
target = "classifier"
base_threshold = 0.5
response_format_type = "json_object"
"#,
        base_url = upstream.base_url
    ))?;
    let app = build_switchyard_router(state);

    let response = send(
        &app,
        "POST",
        "/v1/chat/completions",
        Some(json!({
            "model": "switchyard/stage",
            "messages": [{"role": "user", "content": "bounded task"}]
        })),
    )
    .await?;

    assert_eq!(response.status, StatusCode::OK);
    let calls = upstream.calls.lock().await;
    let judge_call = calls
        .iter()
        .find(|call| call["model"] == "model/classifier")
        .ok_or("classifier target was not called")?;
    assert_eq!(
        judge_call["response_format"],
        json!({"type": "json_object"})
    );
    let prompt = judge_call["messages"][0]["content"]
        .as_str()
        .ok_or("classifier prompt was not text")?;
    assert!(prompt.contains("JSON Schema"), "{prompt}");
    assert!(prompt.contains("\"p_solve\""), "{prompt}");

    drop(calls);
    let invalid_response = send(
        &app,
        "POST",
        "/v1/chat/completions",
        Some(json!({
            "model": "switchyard/stage",
            "messages": [{"role": "user", "content": "return a schema-invalid verdict"}]
        })),
    )
    .await?;
    assert_eq!(invalid_response.status, StatusCode::OK);
    assert_eq!(
        invalid_response
            .headers
            .get("x-model-router-selected-model")
            .and_then(|value| value.to_str().ok()),
        Some("model/weak")
    );
    Ok(())
}

#[tokio::test]
async fn count_tokens_forwards_to_configured_anthropic_target() -> TestResult {
    let upstream = MockUpstream::start().await?;
    let state = load_test_config(&format!(
        r#"
schema_version = 1

[llm_clients.claude]
format = "anthropic_messages"
base_url = "{base_url}"

[targets.strong]
id = "real/opus"
llm_client = "claude"

[targets.other]
id = "real/sonnet"
llm_client = "claude"

[routes.random]
id = "switchyard/random"
type = "random"
targets = ["other", "strong"]
"#,
        base_url = upstream.base_url
    ))?;
    let app = build_switchyard_router(state);

    let response = send(
        &app,
        "POST",
        "/v1/messages/count_tokens",
        Some(json!({
            "model": "switchyard/random",
            "messages": [{"role": "user", "content": "hi"}]
        })),
    )
    .await?;
    assert_eq!(response.status, StatusCode::OK);
    assert_eq!(response.json()?["input_tokens"], 7);

    let calls = upstream.calls.lock().await;
    assert_eq!(calls.len(), 1);
    // The inbound route name is rewritten to the real upstream model.
    assert_eq!(calls[0]["model"], "real/opus");
    Ok(())
}

#[tokio::test]
async fn anthropic_client_forwards_oauth_when_configured() -> TestResult {
    let upstream = MockUpstream::start().await?;
    let state = load_test_config(&format!(
        r#"
schema_version = 1

[llm_clients.claude]
format = "anthropic_messages"
base_url = "{base_url}"
forward_auth = true
max_retries = 0

[targets.claude]
id = "claude-opus"
llm_client = "claude"

[routes.claude]
id = "switchyard/claude"
type = "passthrough"
target = "claude"
"#,
        base_url = upstream.base_url
    ))?;
    let app = build_switchyard_router(state);

    let response = send_with_headers(
        &app,
        "POST",
        "/v1/messages",
        Some(json!({
            "model": "switchyard/claude",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hello"}]
        })),
        &[
            ("authorization", "Bearer claude-oauth-token"),
            ("anthropic-beta", "oauth-2025-04-20,unsupported-beta"),
            ("chatgpt-account-id", "must-not-cross-providers"),
            ("x-openai-fedramp", "must-not-cross-providers"),
        ],
    )
    .await?;
    assert_eq!(response.status, StatusCode::OK);

    let wrong_api = send_with_headers(
        &app,
        "POST",
        "/v1/responses",
        Some(json!({"model": "switchyard/claude", "input": "hello"})),
        &[("authorization", "Bearer codex-login-token")],
    )
    .await?;
    assert_eq!(wrong_api.status, StatusCode::BAD_REQUEST);
    assert_eq!(upstream.calls.lock().await.len(), 1);

    Ok(())
}

#[tokio::test]
async fn responses_client_forwards_openai_login_when_configured() -> TestResult {
    let upstream = MockUpstream::start().await?;
    let state = load_test_config(&format!(
        r#"
schema_version = 1

[llm_clients.openai]
format = "openai_responses"
base_url = "{base_url}"
forward_auth = true
max_retries = 0

[targets.openai]
id = "gpt-codex"
llm_client = "openai"

[routes.openai]
id = "switchyard/codex"
type = "passthrough"
target = "openai"
"#,
        base_url = upstream.base_url
    ))?;
    let app = build_switchyard_router(state);

    let response = send_with_headers(
        &app,
        "POST",
        "/v1/responses",
        Some(json!({"model": "switchyard/codex", "input": "hello"})),
        &[
            ("authorization", "Bearer codex-login-token"),
            ("chatgpt-account-id", "account-123"),
            ("x-openai-fedramp", "true"),
            ("x-api-key", "must-not-cross-providers"),
            ("anthropic-beta", "oauth-must-not-cross-providers"),
        ],
    )
    .await?;
    assert_eq!(response.status, StatusCode::OK);

    let redirect = send_with_headers(
        &app,
        "POST",
        "/v1/responses",
        Some(json!({"model": "switchyard/codex", "input": "hello"})),
        &[
            ("authorization", "Bearer codex-login-token"),
            ("chatgpt-account-id", "account-123"),
            ("x-openai-fedramp", "true"),
            ("x-test-redirect", "1"),
        ],
    )
    .await?;
    assert_eq!(redirect.status, StatusCode::TEMPORARY_REDIRECT);
    assert_eq!(upstream.calls.lock().await.len(), 2);

    let echoed_auth = send_with_headers(
        &app,
        "POST",
        "/v1/responses",
        Some(json!({"model": "switchyard/codex", "input": "hello"})),
        &[
            ("authorization", "Bearer codex-login-token"),
            ("x-test-echo-auth", "1"),
        ],
    )
    .await?;
    assert_eq!(echoed_auth.status, StatusCode::UNAUTHORIZED);
    let error = echoed_auth.text()?;
    assert!(error.contains("[REDACTED]"));
    assert!(!error.contains("codex-login-token"));

    Ok(())
}

#[tokio::test]
async fn count_tokens_without_anthropic_target_returns_bad_request() -> TestResult {
    let upstream = MockUpstream::start().await?;
    let state = load_test_config(&format!(
        r#"
schema_version = 1

[llm_clients.upstream]
format = "openai_chat"
base_url = "{base_url}"

[targets.weak]
id = "model/weak"
llm_client = "upstream"

[routes.random]
id = "switchyard/random"
type = "random"
targets = ["weak"]
"#,
        base_url = upstream.base_url
    ))?;
    let app = build_switchyard_router(state);

    let response = send(
        &app,
        "POST",
        "/v1/messages/count_tokens",
        Some(json!({
            "model": "switchyard/random",
            "messages": [{"role": "user", "content": "hi"}]
        })),
    )
    .await?;
    assert_eq!(response.status, StatusCode::BAD_REQUEST);
    // This route has no Anthropic-format target.
    assert_eq!(
        response.json()?,
        json!({
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "route has no Anthropic target for token counting"
            }
        })
    );
    Ok(())
}

#[tokio::test]
async fn routes_dispatch_and_discovery_endpoints_are_stable() -> TestResult {
    let (upstream, app) = test_app(&[
        ("switchyard/coding", &["model/code"]),
        ("switchyard/general", &["model/general"]),
    ])
    .await?;

    let health = send(&app, "GET", "/health", None).await?;
    assert_eq!(health.status, StatusCode::OK);
    assert_eq!(health.json()?, json!({"status": "ok"}));

    let models = send(&app, "GET", "/v1/models", None).await?;
    assert_eq!(models.status, StatusCode::OK);
    assert_eq!(
        models.json()?["model_pool"],
        json!(["switchyard/coding", "switchyard/general"])
    );

    let missing = send(&app, "GET", "/missing", None).await?;
    assert_eq!(missing.status, StatusCode::NOT_FOUND);
    assert_eq!(missing.json()?["error"]["code"], "endpoint_not_found");

    for (route_model, target_model) in [
        ("switchyard/general", "model/general"),
        ("switchyard/coding", "model/code"),
    ] {
        let response = send(
            &app,
            "POST",
            "/v1/chat/completions",
            Some(json!({
                "model": route_model,
                "messages": [{"role": "user", "content": "hi"}]
            })),
        )
        .await?;
        assert_eq!(response.status, StatusCode::OK);
        assert_eq!(
            response
                .headers
                .get("x-model-router-selected-model")
                .and_then(|value| value.to_str().ok()),
            Some(target_model)
        );
    }

    let calls = upstream.calls.lock().await;
    assert_eq!(calls[0]["model"], "model/general");
    assert_eq!(calls[1]["model"], "model/code");
    Ok(())
}

#[tokio::test]
async fn models_endpoint_reports_declared_route_capabilities_and_null_when_undeclared() -> TestResult
{
    const CONFIG: &str = r#"
schema_version = 1

[llm_clients.primary]
format = "openai_chat"
base_url = "https://example.test/v1"

[targets.shared]
id = "nvidia/deepseek-ai/deepseek-v4-pro"
llm_client = "primary"

[routes.declared]
id = "declared"
type = "passthrough"
target = "shared"
context_window = 1000000
tool_calling = true

[routes.restricted]
id = "restricted"
type = "passthrough"
target = "shared"
context_window = 262000
tool_calling = false

[routes.reasoning]
id = "reasoning"
type = "passthrough"
target = "shared"
reasoning = true

[routes.undeclared]
id = "undeclared"
type = "passthrough"
target = "shared"
"#;
    let app = build_switchyard_router(load_test_config(CONFIG)?);
    let models = send(&app, "GET", "/v1/models", None).await?;
    assert_eq!(models.status, StatusCode::OK);
    let body = models.json()?;
    let data = body["data"].as_array().cloned().unwrap_or_default();
    let capabilities = data
        .iter()
        .filter_map(|entry| entry["id"].as_str().map(|id| (id, &entry["capabilities"])))
        .collect::<BTreeMap<_, _>>();

    assert_eq!(capabilities["declared"]["context_window"], json!(1_000_000));
    assert_eq!(capabilities["declared"]["tool_calling"], json!(true));
    assert_eq!(capabilities["restricted"]["context_window"], json!(262_000));
    assert_eq!(capabilities["restricted"]["tool_calling"], json!(false));
    assert_eq!(capabilities["undeclared"]["context_window"], json!(null));
    assert_eq!(capabilities["undeclared"]["tool_calling"], json!(null));

    let codex_models = body["models"].as_array().cloned().unwrap_or_default();
    let codex_metadata = codex_models
        .iter()
        .filter_map(|entry| entry["slug"].as_str().map(|slug| (slug, entry)))
        .collect::<BTreeMap<_, _>>();
    // This checks the shape the server emits. That Codex 0.144.5 actually decodes it
    // (context_window: null included) is verified by a live Codex run in SWITCH-1225.
    assert_eq!(codex_metadata.len(), 4);
    assert_eq!(
        codex_metadata["declared"]["context_window"],
        json!(1_000_000)
    );
    assert_eq!(codex_metadata["declared"]["shell_type"], "shell_command");
    assert_eq!(
        codex_metadata["declared"]["apply_patch_tool_type"],
        "freeform"
    );
    // Constant fields Codex requires: a typo here would fail its decode, so pin them.
    assert_eq!(codex_metadata["declared"]["visibility"], "list");
    assert_eq!(codex_metadata["declared"]["supported_in_api"], json!(true));
    assert_eq!(codex_metadata["declared"]["web_search_tool_type"], "text");
    assert_eq!(
        codex_metadata["declared"]["input_modalities"],
        json!(["text"])
    );
    assert_eq!(
        codex_metadata["declared"]["truncation_policy"],
        json!({"mode": "tokens", "limit": 10_000})
    );
    assert_eq!(
        codex_metadata["restricted"]["context_window"],
        json!(262_000)
    );
    assert_eq!(codex_metadata["restricted"]["shell_type"], "disabled");
    assert_eq!(
        codex_metadata["restricted"]["apply_patch_tool_type"],
        json!(null)
    );
    // A reasoning route advertises the effort presets and reasoning controls.
    assert_eq!(
        codex_metadata["reasoning"]["default_reasoning_level"],
        "xhigh"
    );
    assert_eq!(
        codex_metadata["reasoning"]["supported_reasoning_levels"]
            .as_array()
            .map(Vec::len),
        Some(4)
    );
    assert_eq!(
        codex_metadata["reasoning"]["supports_reasoning_summaries"],
        json!(true)
    );
    assert_eq!(
        codex_metadata["reasoning"]["support_verbosity"],
        json!(true)
    );
    assert_eq!(codex_metadata["reasoning"]["default_verbosity"], "low");
    // An undeclared route: null context window, non-reasoning, but tools default on
    // so `switchyard launch codex` stays usable out of the box.
    assert_eq!(codex_metadata["undeclared"]["context_window"], json!(null));
    assert_eq!(
        codex_metadata["undeclared"]["supported_reasoning_levels"],
        json!([])
    );
    assert_eq!(
        codex_metadata["undeclared"]["default_reasoning_level"],
        json!(null)
    );
    assert_eq!(
        codex_metadata["undeclared"]["supports_reasoning_summaries"],
        json!(false)
    );
    assert_eq!(codex_metadata["undeclared"]["shell_type"], "shell_command");
    assert_eq!(
        codex_metadata["undeclared"]["apply_patch_tool_type"],
        "freeform"
    );
    assert_eq!(
        codex_metadata["undeclared"]["supports_parallel_tool_calls"],
        json!(true)
    );
    Ok(())
}

#[tokio::test]
async fn all_inbound_formats_run_libsy_and_return_the_caller_format() -> TestResult {
    let (upstream, app) = test_app(&[(ROUTE_MODEL, &["model/a"])]).await?;

    let cases = [
        (
            "/v1/chat/completions",
            json!({
                "model": ROUTE_MODEL,
                "messages": [{"role": "user", "content": "hi"}]
            }),
        ),
        (
            "/v1/messages",
            json!({
                "model": ROUTE_MODEL,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}]
            }),
        ),
        (
            "/v1/responses",
            json!({"model": ROUTE_MODEL, "input": "hi"}),
        ),
    ];

    let mut responses = Vec::new();
    for (path, body) in cases {
        responses.push(send(&app, "POST", path, Some(body)).await?);
    }

    assert!(
        responses
            .iter()
            .all(|response| response.status == StatusCode::OK)
    );
    assert_eq!(
        responses[0].json()?["choices"][0]["message"]["content"],
        "ok"
    );
    assert_eq!(responses[1].json()?["content"][0]["text"], "ok");
    assert_eq!(
        responses[2].json()?["output"][0]["content"][0]["text"],
        "ok"
    );
    assert_eq!(responses[0].json()?["usage"]["prompt_tokens"], 10);
    assert_eq!(
        responses[0].json()?["usage"]["prompt_tokens_details"]["cached_tokens"],
        7
    );
    assert_eq!(responses[1].json()?["usage"]["input_tokens"], 3);
    assert_eq!(responses[1].json()?["usage"]["cache_read_input_tokens"], 7);
    assert_eq!(responses[2].json()?["usage"]["input_tokens"], 10);
    assert_eq!(
        responses[2].json()?["usage"]["input_tokens_details"]["cached_tokens"],
        7
    );
    for response in &responses {
        assert_eq!(
            response
                .headers
                .get("x-model-router-selected-model")
                .and_then(|value| value.to_str().ok()),
            Some("model/a")
        );
        // The body names the model that answered, not the route id the caller
        // addressed, so it agrees with the routing header above.
        assert_eq!(response.json()?["model"], "model/a");
    }

    let calls = upstream.calls.lock().await;
    assert_eq!(calls.len(), 3);
    assert!(calls.iter().all(|call| call["model"] == "model/a"));
    Ok(())
}

// Normalized metadata is authoritative when both ID forms are present;
// legacy-only callers remain supported for backward compatibility.
#[tokio::test]
async fn routing_log_prefers_canonical_and_preserves_legacy_fallback() -> TestResult {
    let upstream = MockUpstream::start().await?;
    let temp_dir = tempfile::tempdir()?;
    let log_path = temp_dir.path().join("routing.jsonl");
    let state = random_state(&upstream.base_url, &[(ROUTE_MODEL, &["model/a"])])?
        .with_routing_log(&log_path)?;
    let app = build_switchyard_router(state);

    let request = HttpRequest::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("x-switchyard-session-id", "canonical-session")
        .header("proxy_x_session_id", "legacy-session")
        .body(Body::from(serde_json::to_vec(&json!({
            "model": ROUTE_MODEL,
            "messages": [{"role": "user", "content": "hello"}]
        }))?))?;
    let response = app.clone().oneshot(request).await?;
    assert_eq!(response.status(), StatusCode::OK);

    let stats = send(
        &app,
        "GET",
        "/v1/routing/session-stats?session_id=canonical-session",
        None,
    )
    .await?;
    assert_eq!(stats.status, StatusCode::OK);
    let stats = stats.json()?;
    assert_eq!(stats["total_calls"], 1);
    assert_eq!(stats["total_prompt_tokens"], 10);
    assert_eq!(stats["total_cached_tokens"], 7);
    assert_eq!(stats["models"]["model/a"]["completion_tokens"], 2);

    let legacy = send(
        &app,
        "GET",
        "/v1/routing/session-stats?session_id=legacy-session",
        None,
    )
    .await?;
    assert_eq!(legacy.status, StatusCode::NOT_FOUND);

    let legacy_only = send_with_headers(
        &app,
        "POST",
        "/v1/chat/completions",
        Some(json!({
            "model": ROUTE_MODEL,
            "messages": [{"role": "user", "content": "hello"}]
        })),
        &[("proxy_x_session_id", "legacy-only-session")],
    )
    .await?;
    assert_eq!(legacy_only.status, StatusCode::OK);

    let legacy_stats = send(
        &app,
        "GET",
        "/v1/routing/session-stats?session_id=legacy-only-session",
        None,
    )
    .await?;
    assert_eq!(legacy_stats.status, StatusCode::OK);
    assert_eq!(legacy_stats.json()?["total_calls"], 1);

    let records = std::fs::read_to_string(log_path)?;
    let first: Value =
        serde_json::from_str(records.lines().next().ok_or("routing log was empty")?)?;
    assert_eq!(first["session_id"], "canonical-session");
    assert!(
        first["ts"]
            .as_str()
            .is_some_and(|value| value.ends_with('Z'))
    );
    Ok(())
}

#[tokio::test]
async fn routing_log_keeps_the_canonical_session_id_until_a_stream_drains() -> TestResult {
    let upstream = MockUpstream::start().await?;
    let temp_dir = tempfile::tempdir()?;
    let state = random_state(&upstream.base_url, &[(ROUTE_MODEL, &["model/a"])])?
        .with_routing_log(temp_dir.path().join("routing.jsonl"))?;
    let app = build_switchyard_router(state);

    // `send_with_headers` collects the response body, so the stream wrapper reaches
    // its terminal usage record before the stats query runs.
    let response = send_with_headers(
        &app,
        "POST",
        "/v1/chat/completions",
        Some(json!({
            "model": ROUTE_MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true
        })),
        &[("x-switchyard-session-id", "streaming-session")],
    )
    .await?;
    assert_eq!(response.status, StatusCode::OK);
    assert!(response.text()?.contains("data: [DONE]"));

    let stats = send(
        &app,
        "GET",
        "/v1/routing/session-stats?session_id=streaming-session",
        None,
    )
    .await?;
    assert_eq!(stats.status, StatusCode::OK);
    let stats = stats.json()?;
    assert_eq!(stats["total_calls"], 1);
    assert_eq!(stats["total_prompt_tokens"], 12);
    assert_eq!(stats["total_cached_tokens"], 7);
    assert_eq!(stats["total_cache_creation_tokens"], 2);
    assert_eq!(stats["total_completion_tokens"], 5);
    Ok(())
}

#[tokio::test]
async fn unavailable_target_fails_over_across_endpoints_and_stops_when_exhausted() -> TestResult {
    let upstream = MockUpstream::start().await?;
    let temp_dir = tempfile::tempdir()?;
    let log_path = temp_dir.path().join("routing.jsonl");
    let state = fallback_state(&upstream.base_url)?.with_routing_log(&log_path)?;
    let app = build_switchyard_router(state);
    let cases = [
        (
            "/v1/chat/completions",
            json!({
                "model": ROUTE_MODEL,
                "messages": [{"role": "user", "content": "unavailable"}]
            }),
        ),
        (
            "/v1/messages",
            json!({
                "model": ROUTE_MODEL,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "unavailable"}]
            }),
        ),
        (
            "/v1/responses",
            json!({"model": ROUTE_MODEL, "input": "unavailable"}),
        ),
    ];

    for (path, body) in cases {
        let previous_call_count = upstream.calls.lock().await.len();
        let response = send(&app, "POST", path, Some(body)).await?;
        assert_eq!(response.status, StatusCode::OK);
        assert_eq!(
            response
                .headers
                .get("x-model-router-selected-model")
                .and_then(|value| value.to_str().ok()),
            Some("model/strong")
        );
        assert_eq!(response.json()?["model"], "model/strong");
        let calls = upstream.calls.lock().await;
        assert_eq!(
            calls[previous_call_count..]
                .iter()
                .map(|call| call["model"].as_str().unwrap_or(""))
                .collect::<Vec<_>>(),
            ["model/weak", "model/strong"]
        );
    }

    let stats = send(&app, "GET", "/v1/stats", None).await?.json()?;
    // Fallback causes are logged rather than accumulated in the legacy stats counters.
    assert_eq!(stats["routing_fallbacks"]["unavailable"], 0);
    assert_eq!(stats["routing_fallbacks"]["context_window"], 0);
    assert_eq!(stats["models"]["model/strong"]["calls"], 3);
    assert_eq!(stats["models"]["model/weak"]["errors"], 3);

    let records = std::fs::read_to_string(&log_path)?;
    let records = records
        .lines()
        .map(serde_json::from_str::<Value>)
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(records.len(), 3);
    assert!(records.iter().all(|record| {
        record["model"] == "model/strong" && record.get("fallback_reason").is_none()
    }));

    let previous_call_count = upstream.calls.lock().await.len();
    let response = send(
        &app,
        "POST",
        "/v1/chat/completions",
        Some(json!({
            "model": ROUTE_MODEL,
            "messages": [{"role": "user", "content": "all-unavailable"}]
        })),
    )
    .await?;
    assert_eq!(response.status, StatusCode::SERVICE_UNAVAILABLE);
    let error = response.json()?;
    assert_eq!(error["error"]["type"], "upstream_error");
    assert_eq!(error["error"]["code"], "upstream_error");
    let calls = upstream.calls.lock().await;
    assert_eq!(
        calls[previous_call_count..]
            .iter()
            .map(|call| call["model"].as_str().unwrap_or(""))
            .collect::<Vec<_>>(),
        ["model/weak", "model/strong"]
    );
    Ok(())
}

#[tokio::test]
async fn streaming_response_is_framed_for_the_inbound_api() -> TestResult {
    let (_upstream, app) = test_app(&[(ROUTE_MODEL, &["model/a"])]).await?;

    let response = send(
        &app,
        "POST",
        "/v1/chat/completions",
        Some(json!({
            "model": ROUTE_MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        })),
    )
    .await?;

    assert_eq!(response.status, StatusCode::OK);
    assert!(response.text()?.contains("hello"));
    assert!(response.text()?.contains("data: [DONE]"));
    Ok(())
}

// SWITCH-922: every streaming codec must report the routed target, not the route
// id the caller addressed — the route id is meaningless to anything reading the
// trajectory (a Bench UI, a spend log, the client's own display).
#[tokio::test]
async fn streamed_response_model_names_the_served_model_not_the_route() -> TestResult {
    let (_upstream, app) = test_app(&[(ROUTE_MODEL, &["model/a"])]).await?;

    // Each case names the JSON pointer to the model on that format's first event.
    let cases = [
        (
            "/v1/chat/completions",
            json!({
                "model": ROUTE_MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true
            }),
            vec!["model"],
        ),
        (
            "/v1/messages",
            json!({
                "model": ROUTE_MODEL,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true
            }),
            vec!["message", "model"],
        ),
        (
            "/v1/responses",
            json!({"model": ROUTE_MODEL, "input": "hi", "stream": true}),
            vec!["response", "model"],
        ),
    ];

    for (path, body, pointer) in cases {
        let response = send(&app, "POST", path, Some(body)).await?;
        assert_eq!(response.status, StatusCode::OK, "{path}");

        let first = first_sse_event(response.text()?)
            .ok_or_else(|| format!("{path} produced no SSE data frames"))?;
        let model = pointer
            .iter()
            .try_fold(&first, |value, key| value.get(key))
            .and_then(Value::as_str);
        assert_eq!(model, Some("model/a"), "{path}");
    }
    Ok(())
}

// Returns the first `data:` frame of an SSE body as JSON, skipping `[DONE]`.
fn first_sse_event(body: &str) -> Option<Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .find_map(|data| serde_json::from_str(data).ok())
}

#[tokio::test]
async fn streaming_success_records_only_final_usage_and_one_latency() -> TestResult {
    const MODEL: &str = "model/stream-success";
    let (_upstream, app) = test_app(&[(ROUTE_MODEL, &[MODEL])]).await?;
    let before = send(&app, "GET", "/metrics", None).await?;
    let before = before.text()?;

    let response = send(
        &app,
        "POST",
        "/v1/chat/completions",
        Some(json!({
            "model": ROUTE_MODEL,
            "messages": [{"role": "user", "content": "stream-success"}],
            "stream": true
        })),
    )
    .await?;

    assert_eq!(response.status, StatusCode::OK);
    assert_in_order(
        response.text()?,
        &[
            "hello",
            "-partial",
            "-final",
            "\"finish_reason\":\"stop\"",
            "[DONE]",
        ],
    );

    let after = send(&app, "GET", "/metrics", None).await?;
    let after = after.text()?;
    for (name, expected_delta) in [
        ("switchyard_prompt_tokens_total", 12.0),
        ("switchyard_completion_tokens_total", 5.0),
        ("switchyard_cached_tokens_total", 7.0),
        ("switchyard_cache_creation_tokens_total", 2.0),
        ("switchyard_reasoning_tokens_total", 3.0),
        ("switchyard_total_latency_ms_count", 1.0),
    ] {
        assert_eq!(
            metric_delta(before, after, name, &[("model", MODEL)]),
            Some(expected_delta),
            "unexpected delta for {name}"
        );
    }
    let stats = send(&app, "GET", "/v1/stats", None).await?.json()?;
    assert_eq!(stats["total_requests"], 1);
    assert_eq!(
        stats["total_tokens"],
        json!({
            "prompt": 12, "completion": 5, "cached": 7,
            "cache_creation": 2, "reasoning": 3, "total": 17
        })
    );
    assert_eq!(stats["models"][MODEL]["model_call_latency"]["count"], 1);
    assert_eq!(stats["models"][MODEL]["total_latency"]["count"], 1);
    Ok(())
}

#[tokio::test]
// A terminal stream failure records errors without usage or terminal latency.
async fn streaming_error_records_error_without_usage_or_latency() -> TestResult {
    const MODEL: &str = "model/stream-error";
    let (_upstream, app) = test_app(&[(ROUTE_MODEL, &[MODEL])]).await?;
    let before = send(&app, "GET", "/metrics", None).await?;
    let before = before.text()?;

    let response = send(
        &app,
        "POST",
        "/v1/chat/completions",
        Some(json!({
            "model": ROUTE_MODEL,
            "messages": [{"role": "user", "content": "stream-error"}],
            "stream": true
        })),
    )
    .await?;

    assert_eq!(response.status, StatusCode::OK);
    assert_in_order(
        response.text()?,
        &["before", "still here", "upstream stream failed"],
    );

    let after = send(&app, "GET", "/metrics", None).await?;
    let after = after.text()?;
    for name in [
        "switchyard_prompt_tokens_total",
        "switchyard_completion_tokens_total",
        "switchyard_cached_tokens_total",
        "switchyard_cache_creation_tokens_total",
        "switchyard_reasoning_tokens_total",
        "switchyard_total_latency_ms_count",
    ] {
        assert_eq!(
            metric_value(after, name, &[("model", MODEL)]),
            metric_value(before, name, &[("model", MODEL)]),
            "{name} changed after a failed stream"
        );
    }
    assert_eq!(
        metric_delta(
            before,
            after,
            "switchyard_errors_total",
            &[("model", MODEL)]
        ),
        Some(1.0)
    );
    let stats = send(&app, "GET", "/v1/stats", None).await?.json()?;
    assert_eq!(stats["total_requests"], 1);
    assert_eq!(stats["total_errors"], 1);
    assert_eq!(stats["total_tokens"], empty_token_totals());
    assert_eq!(stats["models"][MODEL]["calls"], 1);
    assert_eq!(stats["models"][MODEL]["errors"], 1);
    assert_eq!(stats["models"][MODEL]["total_latency"]["count"], 0);
    assert_eq!(stats["routing_overhead"]["count"], 1);
    Ok(())
}

#[tokio::test]
async fn responses_stream_error_does_not_emit_success_terminal_events() -> TestResult {
    // A distinct target keeps this test's error-counter increments off the shared
    // model/stream-error metric that streaming_error_records_... asserts an exact delta on.
    let (_upstream, app) = test_app(&[(ROUTE_MODEL, &["model/responses-stream-error"])]).await?;

    let response = send(
        &app,
        "POST",
        "/v1/responses",
        Some(json!({
            "model": ROUTE_MODEL,
            "input": "stream-error",
            "stream": true
        })),
    )
    .await?;

    assert_eq!(response.status, StatusCode::OK);
    let body = response.text()?;
    assert_in_order(body, &["before", "upstream stream failed"]);
    for event_type in [
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ] {
        assert!(
            !body.contains(event_type),
            "{event_type} followed an upstream stream error"
        );
    }
    Ok(())
}

#[tokio::test]
async fn chat_stream_error_does_not_emit_success_terminal_chunk() -> TestResult {
    // A distinct target keeps this test's error-counter increments off the shared
    // model/stream-error metric that streaming_error_records_... asserts an exact delta on.
    let (_upstream, app) = test_app(&[(ROUTE_MODEL, &["model/chat-stream-error"])]).await?;

    let response = send(
        &app,
        "POST",
        "/v1/chat/completions",
        Some(json!({
            "model": ROUTE_MODEL,
            "messages": [{"role": "user", "content": "stream-error"}],
            "stream": true
        })),
    )
    .await?;

    assert_eq!(response.status, StatusCode::OK);
    let body = response.text()?;
    assert_in_order(body, &["before", "still here", "upstream stream failed"]);
    // The finalizer must not synthesize a `finish_reason: stop` completion chunk after the error.
    let after_error = body
        .split_once("upstream stream failed")
        .map(|(_, rest)| rest)
        .unwrap_or_default();
    assert!(
        !after_error.contains(r#""finish_reason":"stop""#),
        "a finish_reason=stop chunk followed an upstream stream error:\n{body}"
    );
    Ok(())
}

#[tokio::test]
async fn anthropic_stream_error_does_not_emit_success_terminal_events() -> TestResult {
    // A distinct target keeps this test's error-counter increments off the shared
    // model/stream-error metric that streaming_error_records_... asserts an exact delta on.
    let (_upstream, app) = test_app(&[(ROUTE_MODEL, &["model/anthropic-stream-error"])]).await?;

    let response = send(
        &app,
        "POST",
        "/v1/messages",
        Some(json!({
            "model": ROUTE_MODEL,
            "messages": [{"role": "user", "content": "stream-error"}],
            "max_tokens": 16,
            "stream": true
        })),
    )
    .await?;

    assert_eq!(response.status, StatusCode::OK);
    let body = response.text()?;
    assert_in_order(body, &["before", "upstream stream failed"]);
    // The finalizer must not close the turn with message_delta/message_stop after the error.
    let after_error = body
        .split_once("upstream stream failed")
        .map(|(_, rest)| rest)
        .unwrap_or_default();
    for event_type in ["message_delta", "message_stop"] {
        assert!(
            !after_error.contains(event_type),
            "{event_type} followed an upstream stream error:\n{body}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn request_and_upstream_errors_use_the_inbound_wire_format() -> TestResult {
    let (_upstream, app) = test_app(&[(ROUTE_MODEL, &["model/a"])]).await?;

    let unknown = send(
        &app,
        "POST",
        "/v1/chat/completions",
        Some(json!({
            "model": "other",
            "messages": [{"role": "user", "content": "hi"}]
        })),
    )
    .await?;
    assert_eq!(unknown.status, StatusCode::NOT_FOUND);
    assert_eq!(unknown.json()?["error"]["code"], "model_not_found");

    let missing_model = send(
        &app,
        "POST",
        "/v1/chat/completions",
        Some(json!({"messages": [{"role": "user", "content": "hi"}]})),
    )
    .await?;
    assert_eq!(missing_model.status, StatusCode::BAD_REQUEST);
    assert_eq!(
        missing_model.json()?["error"]["code"],
        "invalid_request_error"
    );

    let upstream_cases = [
        (
            "/v1/chat/completions",
            json!({
                "model": ROUTE_MODEL,
                "messages": [{"role": "user", "content": "auth-fail"}]
            }),
            json!({
                "error": {
                    "message": "upstream authentication failed",
                    "type": "upstream_error",
                    "code": "upstream_error"
                }
            }),
        ),
        (
            "/v1/responses",
            json!({"model": ROUTE_MODEL, "input": "auth-fail"}),
            json!({
                "error": {
                    "message": "upstream authentication failed",
                    "type": "upstream_error",
                    "code": "upstream_error"
                }
            }),
        ),
        (
            "/v1/messages",
            json!({
                "model": ROUTE_MODEL,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "auth-fail"}]
            }),
            json!({
                "type": "error",
                "error": {
                    "type": "authentication_error",
                    "message": "upstream authentication failed"
                }
            }),
        ),
    ];
    for (path, body, expected) in upstream_cases {
        let response = send(&app, "POST", path, Some(body)).await?;
        assert_eq!(response.status, StatusCode::UNAUTHORIZED, "{path}");
        assert_eq!(response.json()?, expected, "{path}");
    }

    let anthropic_unknown = send(
        &app,
        "POST",
        "/v1/messages",
        Some(json!({
            "model": "other",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}]
        })),
    )
    .await?;
    assert_eq!(anthropic_unknown.status, StatusCode::NOT_FOUND);
    assert_eq!(
        anthropic_unknown.json()?,
        json!({
            "type": "error",
            "error": {
                "type": "not_found_error",
                "message": "No route registered for model other"
            }
        })
    );
    Ok(())
}
