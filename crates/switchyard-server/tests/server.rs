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
use axum::http::{Request as HttpRequest, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response as HttpResponse};
use axum::routing::post;
use axum::{Json, Router};
use http_body_util::BodyExt;
use libsy::algorithms::{FallThrough, Random};
use libsy::stage_router::{PickerMode, StageClassifier};
use libsy::{Algorithm, LlmTarget, LlmTargetSet, RoutedLlmClient, State as AlgorithmState};
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

    let content = if model == "model/classifier" {
        r#"{"recommended_route":"efficient","p_solve":0.9,"confidence":0.9,"abstain":false,"capability_boundary":"supported","primary_rule":"SUP-1","crux":"bounded task"}"#
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
            let target_set = LlmTargetSet::new(
                targets
                    .iter()
                    .map(|model| LlmTarget {
                        semantic_name: (*model).to_string(),
                        llm_client: Some(Arc::clone(&client)),
                    })
                    .collect(),
            );
            let algorithm: Arc<dyn Algorithm> = Arc::new(Random::new(target_set, None, None)?);
            Ok(((*route_model).to_string(), algorithm))
        })
        .collect::<TestResult<Vec<_>>>()?;
    Ok(ServerState::new(entries)?)
}

async fn test_app(routes: &[(&str, &[&str])]) -> TestResult<(MockUpstream, Router)> {
    let upstream = MockUpstream::start().await?;
    let app = build_switchyard_router(random_state(&upstream.base_url, routes)?);
    Ok((upstream, app))
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
    assert!(metric_line(
        metrics,
        "switchyard_routing_overhead_ms_bucket",
        &[("algorithm", "random"), ("le", "0.1")]
    )
    .is_some());
    assert!(metric_line(
        metrics,
        "switchyard_cache_creation_tokens_total",
        &[("model", MODEL)]
    )
    .is_none());
    assert!(metric_line(
        metrics,
        "switchyard_reasoning_tokens_total",
        &[("model", MODEL)]
    )
    .is_none());
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

[routes.random]
id = "switchyard/random"
type = "random"
targets = ["strong"]
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
    // The route's picked target is OpenAI, so count_tokens (Anthropic-only) is
    // unsupported for it.
    assert_eq!(
        response.json()?["error"]["code"],
        "count_tokens_unsupported"
    );
    Ok(())
}

// Build a stage_router (FallThrough) with an Anthropic `strong` tier and an
// OpenAI `weak` tier, both pointed at the mock upstream, and register it.
fn stage_router_state(upstream: &MockUpstream, mode: PickerMode) -> TestResult<ServerState> {
    let cfg = |url: &str| HttpBackendConfig {
        base_url: url.to_string(),
        api_key: Some("k".to_string()),
        extra_headers: BTreeMap::new(),
        extra_body: BTreeMap::new(),
        max_retries: 0,
    };
    let strong: Arc<dyn RoutedLlmClient> =
        Arc::new(TranslatingLlmClient::new(&[ModelConfig::new(
            "strong",
            Backend::Anthropic(cfg(&upstream.base_url)),
            None,
        )])?);
    let weak: Arc<dyn RoutedLlmClient> = Arc::new(TranslatingLlmClient::new(&[ModelConfig::new(
        "weak",
        Backend::OpenAiChat(cfg(&upstream.base_url)),
        None,
    )])?);
    let targets = LlmTargetSet::new(vec![
        LlmTarget {
            semantic_name: "strong".to_string(),
            llm_client: Some(strong),
        },
        LlmTarget {
            semantic_name: "weak".to_string(),
            llm_client: Some(weak),
        },
    ]);
    let stage: Arc<dyn Algorithm> = Arc::new(
        FallThrough::<AlgorithmState>::new_with_state(targets)
            .with_classifier(Arc::new(StageClassifier::new(mode, 0.5))),
    );
    Ok(ServerState::new([("switchyard/stage".to_string(), stage)])?)
}

/// `count_tokens` is a **direct passthrough**, not a routed call, so on a stage
/// router it goes straight to the Anthropic (`strong`) tier — bypassing the
/// classifier cascade. This is exactly what makes it work where a completion
/// can't: a signal-less request (as `count_tokens` always is) gives the
/// `StageClassifier` nothing to score, so a *completion* on this bare stage
/// router abstains — but `count_tokens` still succeeds.
#[tokio::test]
async fn count_tokens_on_a_stage_router_passes_through_to_the_anthropic_tier() -> TestResult {
    let upstream = MockUpstream::start().await?;
    let app = build_switchyard_router(stage_router_state(&upstream, PickerMode::EfficientFirst)?);
    let body = json!({
        "model": "switchyard/stage",
        "messages": [{"role": "user", "content": "hi"}]
    });

    // A completion routes → the bare StageClassifier abstains on a signal-less
    // request → error.
    let completion = send(&app, "POST", "/v1/messages", Some(body.clone())).await?;
    assert!(completion.text()?.contains("abstained"));

    // count_tokens does NOT route — it passes through to the strong (Anthropic)
    // tier and succeeds.
    let count = send(&app, "POST", "/v1/messages/count_tokens", Some(body)).await?;
    assert_eq!(count.status, StatusCode::OK);
    assert_eq!(count.json()?["input_tokens"], 7);

    // The forwarded call went to count_tokens with the strong tier's model id.
    let calls = upstream.calls.lock().await;
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["model"], "strong");
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
    Ok(())
}

#[tokio::test]
async fn streaming_error_records_neither_usage_nor_latency() -> TestResult {
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
