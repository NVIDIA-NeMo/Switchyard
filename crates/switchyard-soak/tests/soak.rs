// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests that drive the soak runner against a hermetic axum mock of a Switchyard server.

use std::error::Error;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::extract::Json;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use clap::Parser;
use parking_lot::Mutex;
use serde_json::{Value, json};

use switchyard_soak::client::{Endpoint, preflight, read_server_state, request_body, send_request};
use switchyard_soak::report::{ResultsWriter, invalid_request_canary};
use switchyard_soak::stats::RunStats;
use switchyard_soak::{Args, Stop, run};

type TestResult = Result<(), Box<dyn Error>>;

/// Bind an ephemeral port, serve *app*, and return its base URL. The listener is bound before the
/// server task starts, so client connections queue rather than being refused.
async fn serve(app: Router) -> Result<String, Box<dyn Error>> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    tokio::spawn(async move {
        let _ = axum::serve(listener, app.into_make_service()).await;
    });
    Ok(format!("http://{addr}"))
}

fn client() -> Result<reqwest::Client, Box<dyn Error>> {
    Ok(reqwest::Client::builder().no_proxy().build()?)
}

fn sse_body(body: &'static str) -> Response {
    ([(header::CONTENT_TYPE, "text/event-stream")], body).into_response()
}

fn is_stream(body: &Value) -> bool {
    body.get("stream").and_then(Value::as_bool).unwrap_or(false)
}

/// A well-behaved Switchyard: valid shapes, SSE for streaming, HTTP 400 for empty messages.
fn healthy_switchyard() -> Router {
    async fn health() -> Response {
        Json(json!({"status": "ok"})).into_response()
    }
    async fn metrics() -> Response {
        "switchyard_total_requests 1\nswitchyard_total_errors 0\n".into_response()
    }
    async fn models() -> Response {
        Json(json!({"data": [{"id": "soak-route"}]})).into_response()
    }
    async fn chat(Json(body): Json<Value>) -> Response {
        let empty = body
            .get("messages")
            .and_then(Value::as_array)
            .map(|messages| messages.is_empty())
            .unwrap_or(false);
        if empty {
            return (StatusCode::BAD_REQUEST, "messages must not be empty").into_response();
        }
        if is_stream(&body) {
            sse_body("data: {}\n\ndata: [DONE]\n\n")
        } else {
            Json(json!({"choices": []})).into_response()
        }
    }
    async fn messages(Json(body): Json<Value>) -> Response {
        if is_stream(&body) {
            sse_body("data: {}\n\ndata: [DONE]\n\n")
        } else {
            Json(json!({"content": []})).into_response()
        }
    }
    async fn responses(Json(body): Json<Value>) -> Response {
        if is_stream(&body) {
            sse_body("data: {}\n\ndata: [DONE]\n\n")
        } else {
            Json(json!({"output": []})).into_response()
        }
    }
    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/messages", post(messages))
        .route("/v1/responses", post(responses))
}

#[tokio::test]
async fn short_run_passes_against_live_switchyard_routes() -> TestResult {
    let base_url = serve(healthy_switchyard()).await?;
    let dir = tempfile::tempdir()?;
    let results_dir = dir.path().join("soak-results");

    let args = Args::try_parse_from([
        "switchyard-soak",
        "--base-url",
        &base_url,
        "--model",
        "soak-route",
        "--duration",
        "1s",
        "--concurrency",
        "2",
        "--stream-ratio",
        "0.5",
        "--report-interval",
        "0.2",
        "--invalid-canary-interval",
        "0.2",
        "--results-dir",
        results_dir.to_str().ok_or("non-utf8 results dir")?,
    ])?;
    args.validate()?;
    assert_eq!(run(args).await?, 0);

    let summary: Value =
        serde_json::from_str(&std::fs::read_to_string(results_dir.join("summary.json"))?)?;
    assert_eq!(summary["passed"], json!(true));
    for endpoint in ["chat", "messages", "responses"] {
        assert!(
            summary["endpoint_successes"][endpoint]
                .as_u64()
                .unwrap_or(0)
                > 0,
            "expected successes on {endpoint}: {summary}"
        );
    }
    assert!(summary["invalid_request_canaries"].as_u64().unwrap_or(0) > 0);
    Ok(())
}

#[tokio::test]
async fn preflight_selects_default_model() -> TestResult {
    async fn models() -> Response {
        Json(json!({"data": [{"id": "route-a"}, {"id": "route-b"}], "default_model": "route-b"}))
            .into_response()
    }
    let app = Router::new()
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/v1/models", get(models));
    let base_url = serve(app).await?;

    assert_eq!(preflight(&client()?, &base_url, None).await?, "route-b");
    Ok(())
}

#[tokio::test]
async fn preflight_rejects_empty_and_unknown_model() -> TestResult {
    let empty = Router::new()
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/v1/models", get(|| async { Json(json!({"data": []})) }));
    let base_url = serve(empty).await?;
    let error = preflight(&client()?, &base_url, None).await.unwrap_err();
    assert!(error.contains("no model to test"), "{error}");

    let one = Router::new()
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .route(
            "/v1/models",
            get(|| async { Json(json!({"data": [{"id": "route-a"}]})) }),
        );
    let base_url = serve(one).await?;
    let error = preflight(&client()?, &base_url, Some("missing-model"))
        .await
        .unwrap_err();
    assert!(error.contains("is not listed"), "{error}");
    Ok(())
}

#[tokio::test]
async fn send_request_accepts_json_and_streaming_success() -> TestResult {
    let base_url = serve(healthy_switchyard()).await?;
    let http = client()?;

    let (error, _) = send_request(
        &http,
        &base_url,
        Endpoint::Chat,
        &request_body(Endpoint::Chat, "route", "hi", 8, false),
    )
    .await;
    assert_eq!(error, None);
    let (error, _) = send_request(
        &http,
        &base_url,
        Endpoint::Chat,
        &request_body(Endpoint::Chat, "route", "hi", 8, true),
    )
    .await;
    assert_eq!(error, None);
    Ok(())
}

#[tokio::test]
async fn send_request_rejects_missing_fields_and_non_sse_stream() -> TestResult {
    async fn chat(Json(body): Json<Value>) -> Response {
        if is_stream(&body) {
            // 200 but application/json instead of an event stream.
            Json(json!({"choices": []})).into_response()
        } else {
            // 200 without the required "choices" field.
            Json(json!({})).into_response()
        }
    }
    let base_url = serve(Router::new().route("/v1/chat/completions", post(chat))).await?;
    let http = client()?;

    let (error, _) = send_request(
        &http,
        &base_url,
        Endpoint::Chat,
        &request_body(Endpoint::Chat, "route", "hi", 8, false),
    )
    .await;
    assert_eq!(error.as_deref(), Some("invalid_response"));
    let (error, _) = send_request(
        &http,
        &base_url,
        Endpoint::Chat,
        &request_body(Endpoint::Chat, "route", "hi", 8, true),
    )
    .await;
    assert_eq!(error.as_deref(), Some("invalid_stream"));
    Ok(())
}

#[tokio::test]
async fn send_request_reports_http_error_and_empty_stream() -> TestResult {
    async fn chat(Json(body): Json<Value>) -> Response {
        if is_stream(&body) {
            sse_body("\n\n") // event stream with no data
        } else {
            (StatusCode::BAD_REQUEST, "bad request").into_response()
        }
    }
    let base_url = serve(Router::new().route("/v1/chat/completions", post(chat))).await?;
    let http = client()?;

    let (error, _) = send_request(
        &http,
        &base_url,
        Endpoint::Chat,
        &request_body(Endpoint::Chat, "route", "hi", 8, false),
    )
    .await;
    assert_eq!(error.as_deref(), Some("http_400"));
    let (error, _) = send_request(
        &http,
        &base_url,
        Endpoint::Chat,
        &request_body(Endpoint::Chat, "route", "hi", 8, true),
    )
    .await;
    assert_eq!(error.as_deref(), Some("empty_stream"));
    Ok(())
}

#[tokio::test]
async fn read_server_state_treats_non_dict_health_as_unhealthy() -> TestResult {
    let app = Router::new()
        // Valid JSON, but a list rather than an object.
        .route("/health", get(|| async { Json(json!(["ok"])) }))
        .route(
            "/metrics",
            get(|| async { "switchyard_total_requests 5\nswitchyard_total_errors 0\n" }),
        );
    let base_url = serve(app).await?;

    let (healthy, metrics) = read_server_state(&client()?, &base_url).await;
    assert!(!healthy);
    assert_eq!(metrics.get("switchyard_total_requests"), Some(&5.0));
    Ok(())
}

#[tokio::test]
async fn invalid_request_canary_flags_non_400() -> TestResult {
    let app = Router::new()
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        // Should have been HTTP 400 for an invalid request.
        .route(
            "/v1/chat/completions",
            post(|| async { Json(json!({"choices": []})) }),
        );
    let base_url = serve(app).await?;

    let dir = tempfile::tempdir()?;
    let stats = Arc::new(Mutex::new(RunStats::new(0)));
    let writer = Arc::new(Mutex::new(ResultsWriter::new(&dir.path().join("results"))?));
    let stop = Arc::new(Stop::new());

    let task = tokio::spawn(invalid_request_canary(
        client()?,
        base_url,
        0.05,
        "soak-route".to_string(),
        stop.clone(),
        stats.clone(),
        writer.clone(),
    ));
    tokio::time::sleep(Duration::from_millis(120)).await;
    stop.set();
    task.await??;

    let stats = stats.lock();
    assert!(stats.canaries >= 1);
    assert_eq!(stats.canary_failures, stats.canaries);
    Ok(())
}
