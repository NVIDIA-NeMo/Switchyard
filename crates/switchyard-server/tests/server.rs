// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the libsy Rust server.

use std::collections::{BTreeMap, HashSet};
use std::convert::Infallible;
use std::error::Error;
use std::io::Write;
use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{Request as HttpRequest, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response as HttpResponse};
use axum::routing::post;
use axum::{Json, Router};
use http_body_util::BodyExt;
use libsy::algorithms::Random;
use libsy::{Algorithm, LlmTarget, LlmTargetSet, RoutedLlmClient};
use serde_json::{json, Value};
use switchyard_llm_client::{Backend, HttpBackendConfig, ModelConfig, TranslatingLlmClient};
use switchyard_server::config::load_server_state;
use switchyard_server::{build_switchyard_router, ServerState};
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tower::ServiceExt;

type TestError = Box<dyn Error + Send + Sync>;
type TestResult<T = ()> = Result<T, TestError>;

const ROUTE_MODEL: &str = "switchyard/random";

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

    let model = body["model"].as_str().unwrap_or("unknown").to_string();
    if body["stream"].as_bool() == Some(true) {
        let events = [
            json!({"id": "chatcmpl-stream", "model": model, "choices": [{"index": 0, "delta": {"role": "assistant"}}]}).to_string(),
            json!({"id": "chatcmpl-stream", "model": model, "choices": [{"index": 0, "delta": {"content": "hello"}}]}).to_string(),
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

    let content = if model == "model/classifier" {
        "0.9"
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

fn random_state(base_url: &str, routes: &[(&str, &[&str])]) -> TestResult<ServerState> {
    let backend = Backend::OpenAiChat(HttpBackendConfig {
        base_url: base_url.to_string(),
        api_key: Some("test-key".to_string()),
        extra_headers: BTreeMap::new(),
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
    let entries = routes.iter().map(|(route_model, targets)| {
        let target_set = LlmTargetSet::new(
            targets
                .iter()
                .map(|model| LlmTarget {
                    semantic_name: (*model).to_string(),
                    llm_client: Some(Arc::clone(&client)),
                })
                .collect(),
        );
        let algorithm: Arc<dyn Algorithm> = Arc::new(Random::new(target_set));
        ((*route_model).to_string(), algorithm)
    });
    Ok(ServerState::new(entries)?)
}

async fn test_app(routes: &[(&str, &[&str])]) -> TestResult<(MockUpstream, Router)> {
    let upstream = MockUpstream::start().await?;
    let app = build_switchyard_router(random_state(&upstream.base_url, routes)?);
    Ok((upstream, app))
}

#[tokio::test]
async fn metrics_exposes_libsy_otel_instruments() -> TestResult {
    let (_upstream, app) = test_app(&[(ROUTE_MODEL, &["model/a"])]).await?;

    let before = send(&app, "GET", "/metrics", None).await?;
    assert_eq!(before.status, StatusCode::OK);
    assert_eq!(
        before
            .headers
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("text/plain; version=0.0.4; charset=utf-8")
    );

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
        "# TYPE libsy_runs_total counter",
        "# TYPE libsy_llm_calls_total counter",
        "# TYPE libsy_run_duration_ms histogram",
        "# TYPE libsy_llm_call_duration_ms histogram",
        "algorithm=\"random\"",
        "selected_model=\"model/a\"",
    ] {
        assert!(
            metrics.contains(expected),
            "missing {expected:?} in metrics:\n{metrics}"
        );
    }
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

async fn send(app: &Router, method: &str, path: &str, body: Option<Value>) -> TestResult<Response> {
    let mut builder = HttpRequest::builder().method(method).uri(path);
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
threshold = 0.5
"#,
        base_url = upstream.base_url
    ))?;
    let app = build_switchyard_router(state);

    for (route, selected) in [
        ("switchyard/random", "model/weak"),
        ("switchyard/classifier", "model/strong"),
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
    assert_eq!(calls.len(), 3);
    assert_eq!(calls[0]["model"], "model/weak");
    assert_eq!(calls[1]["model"], "model/classifier");
    assert_eq!(calls[2]["model"], "model/strong");
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

    assert!(responses
        .iter()
        .all(|response| response.status == StatusCode::OK));
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
    }

    let calls = upstream.calls.lock().await;
    assert_eq!(calls.len(), 3);
    assert!(calls.iter().all(|call| call["model"] == "model/a"));
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

#[tokio::test]
async fn request_and_upstream_errors_use_the_canonical_envelope() -> TestResult {
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

    let upstream_error = send(
        &app,
        "POST",
        "/v1/chat/completions",
        Some(json!({
            "model": ROUTE_MODEL,
            "messages": [{"role": "user", "content": "fail"}]
        })),
    )
    .await?;
    assert_eq!(upstream_error.status, StatusCode::IM_A_TEAPOT);
    assert_eq!(upstream_error.json()?["error"]["code"], "upstream_error");
    Ok(())
}
