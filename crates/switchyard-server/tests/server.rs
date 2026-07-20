// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the libsy Rust server.

use std::collections::{BTreeMap, HashSet};
use std::convert::Infallible;
use std::error::Error;
use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{Request as HttpRequest, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response as HttpResponse};
use axum::routing::post;
use axum::{Json, Router};
use http_body_util::BodyExt;
use libsy::{Algorithm, LlmTarget, LlmTargetSet, RandomAlgo, RoutedLlmClient};
use serde_json::{json, Value};
use switchyard_llm_client::{Backend, HttpBackendConfig, ModelConfig, TranslatingLlmClient};
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

    Json(json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    }))
    .into_response()
}

fn random_state(base_url: &str, targets: &[&str]) -> TestResult<ServerState> {
    let backend = Backend::OpenAiChat(HttpBackendConfig {
        base_url: base_url.to_string(),
        api_key: Some("test-key".to_string()),
        extra_headers: BTreeMap::new(),
    });
    let model_configs = targets
        .iter()
        .map(|model| ModelConfig::new(*model, backend.clone(), None))
        .collect::<Vec<_>>();
    let client: Arc<dyn RoutedLlmClient> = Arc::new(TranslatingLlmClient::new(&model_configs)?);
    let target_set = LlmTargetSet::new(
        targets
            .iter()
            .map(|model| LlmTarget {
                semantic_name: (*model).to_string(),
                llm_client: Some(Arc::clone(&client)),
            })
            .collect(),
    );
    let algorithm: Arc<dyn Algorithm> = Arc::new(RandomAlgo::new(target_set));
    Ok(ServerState::new(
        ROUTE_MODEL,
        "uniform random routing",
        algorithm,
    )?)
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
async fn health_models_and_unknown_paths_have_stable_shapes() -> TestResult {
    let upstream = MockUpstream::start().await?;
    let app = build_switchyard_router(random_state(&upstream.base_url, &["model/a"])?);

    let health = send(&app, "GET", "/health", None).await?;
    assert_eq!(health.status, StatusCode::OK);
    assert_eq!(health.json()?, json!({"status": "ok"}));

    let models = send(&app, "GET", "/v1/models", None).await?;
    assert_eq!(models.status, StatusCode::OK);
    assert_eq!(models.json()?["model_pool"], json!([ROUTE_MODEL]));

    let missing = send(&app, "GET", "/missing", None).await?;
    assert_eq!(missing.status, StatusCode::NOT_FOUND);
    assert_eq!(missing.json()?["error"]["code"], "endpoint_not_found");
    Ok(())
}

#[tokio::test]
async fn all_inbound_formats_run_libsy_and_return_the_caller_format() -> TestResult {
    let upstream = MockUpstream::start().await?;
    let app = build_switchyard_router(random_state(&upstream.base_url, &["model/a"])?);

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
async fn random_routing_only_calls_configured_targets() -> TestResult {
    let upstream = MockUpstream::start().await?;
    let targets = ["model/a", "model/b", "model/c"];
    let app = build_switchyard_router(random_state(&upstream.base_url, &targets)?);

    for _ in 0..20 {
        let response = send(
            &app,
            "POST",
            "/v1/chat/completions",
            Some(json!({
                "model": ROUTE_MODEL,
                "messages": [{"role": "user", "content": "hi"}]
            })),
        )
        .await?;
        assert_eq!(response.status, StatusCode::OK);
    }

    let configured = targets.into_iter().collect::<HashSet<_>>();
    let calls = upstream.calls.lock().await;
    assert!(calls.iter().all(|call| {
        call["model"]
            .as_str()
            .map(|model| configured.contains(model))
            .unwrap_or(false)
    }));
    Ok(())
}

#[tokio::test]
async fn streaming_response_is_framed_for_the_inbound_api() -> TestResult {
    let upstream = MockUpstream::start().await?;
    let app = build_switchyard_router(random_state(&upstream.base_url, &["model/a"])?);

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
    let upstream = MockUpstream::start().await?;
    let app = build_switchyard_router(random_state(&upstream.base_url, &["model/a"])?);

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
