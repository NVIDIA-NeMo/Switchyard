// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Migration tests proving components-v2 profiles cover legacy serving contracts.

use std::io::ErrorKind;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener as StdTcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use switchyard_components_v2::{
    parse_profile_config_str, profile_stats_accumulator, ProfileConfigFormat,
};
use switchyard_server::{build_switchyard_router, ServerState};
use tokio::sync::Mutex as TokioMutex;
use tower::ServiceExt;

static STATS_TEST_LOCK: TokioMutex<()> = TokioMutex::const_new(());

#[tokio::test]
async fn noop_profile_serves_all_inbound_formats_without_upstream() -> TestResult {
    let _stats_guard = STATS_TEST_LOCK.lock().await;
    reset_stats()?;
    let app = build_switchyard_router(state_from_yaml(
        r#"
profiles:
  bench:
    type: noop
"#,
    )?);

    let models = app
        .clone()
        .oneshot(request("GET", "/v1/models", None)?)
        .await?;
    assert_eq!(models.status(), StatusCode::OK);
    assert_eq!(json_body(models).await?["model_pool"], json!(["bench"]));

    let chat = app
        .clone()
        .oneshot(request(
            "POST",
            "/v1/chat/completions",
            Some(chat_body("bench")),
        )?)
        .await?;
    assert_eq!(chat.status(), StatusCode::OK);
    assert_eq!(
        json_body(chat).await?["choices"][0]["message"]["content"],
        "ok"
    );

    let messages = app
        .clone()
        .oneshot(request(
            "POST",
            "/v1/messages",
            Some(json!({
                "model": "bench",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            })),
        )?)
        .await?;
    assert_eq!(messages.status(), StatusCode::OK);
    assert_eq!(json_body(messages).await?["content"][0]["text"], "ok");

    let responses = app
        .oneshot(request(
            "POST",
            "/v1/responses",
            Some(json!({"model": "bench", "input": "hi"})),
        )?)
        .await?;
    assert_eq!(responses.status(), StatusCode::OK);
    assert_eq!(
        json_body(responses).await?["output"][0]["content"][0]["text"],
        "ok"
    );
    Ok(())
}

#[tokio::test]
async fn passthrough_profile_routes_all_inbound_formats_to_configured_target() -> TestResult {
    let _stats_guard = STATS_TEST_LOCK.lock().await;
    reset_stats()?;
    let Some(stub) = HttpStub::start(vec![
        StubResponse::ok(),
        StubResponse::ok(),
        StubResponse::ok(),
    ])?
    else {
        log_loopback_bind_skip();
        return Ok(());
    };
    let app = build_switchyard_router(state_from_yaml(&format!(
        r#"
targets:
  direct:
    model: provider/direct
    format: openai
    base_url: {base_url}
profiles:
  direct-profile:
    type: passthrough
    target: direct
"#,
        base_url = stub.base_url
    ))?);

    let chat = app
        .clone()
        .oneshot(request(
            "POST",
            "/v1/chat/completions",
            Some(chat_body("direct-profile")),
        )?)
        .await?;
    assert_eq!(chat.status(), StatusCode::OK);
    let body = json_body(chat).await?;
    assert_eq!(body["choices"][0]["message"]["content"], "stub-ok");

    let messages = app
        .clone()
        .oneshot(request(
            "POST",
            "/v1/messages",
            Some(json!({
                "model": "direct-profile",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            })),
        )?)
        .await?;
    assert_eq!(messages.status(), StatusCode::OK);
    assert_eq!(json_body(messages).await?["content"][0]["text"], "stub-ok");

    let responses = app
        .oneshot(request(
            "POST",
            "/v1/responses",
            Some(json!({"model": "direct-profile", "input": "hi"})),
        )?)
        .await?;
    assert_eq!(responses.status(), StatusCode::OK);
    assert_eq!(
        json_body(responses).await?["output"][0]["content"][0]["text"],
        "stub-ok"
    );

    let seen = stub.requests()?;
    assert_eq!(seen.len(), 3);
    for request in seen {
        assert_eq!(request["method"], "POST");
        assert_eq!(request["path"], "/v1/chat/completions");
        assert_eq!(request["body"]["model"], "provider/direct");
    }
    Ok(())
}

#[tokio::test]
async fn random_routing_profile_covers_strong_and_weak_selection() -> TestResult {
    let _stats_guard = STATS_TEST_LOCK.lock().await;
    reset_stats()?;
    let cases = [(1.0, "provider/strong"), (0.0, "provider/weak")];
    for (strong_probability, expected_model) in cases {
        let Some(stub) = HttpStub::start(vec![StubResponse::ok()])? else {
            log_loopback_bind_skip();
            return Ok(());
        };
        let app = build_switchyard_router(state_from_yaml(&format!(
            r#"
targets:
  strong:
    model: provider/strong
    format: openai
    base_url: {base_url}
  weak:
    model: provider/weak
    format: openai
    base_url: {base_url}
profiles:
  random-profile:
    type: random-routing
    strong: strong
    weak: weak
    strong_probability: {strong_probability}
    rng_seed: 7
"#,
            base_url = stub.base_url
        ))?);

        let response = app
            .oneshot(request(
                "POST",
                "/v1/chat/completions",
                Some(chat_body("random-profile")),
            )?)
            .await?;

        assert_eq!(response.status(), StatusCode::OK);
        let seen = stub.requests()?;
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0]["body"]["model"], expected_model);
    }
    Ok(())
}

#[tokio::test]
async fn stage_router_profile_threshold_zero_uses_dimensions_signal_path() -> TestResult {
    let _stats_guard = STATS_TEST_LOCK.lock().await;
    reset_stats()?;
    let Some(stub) = HttpStub::start(vec![StubResponse::ok()])? else {
        log_loopback_bind_skip();
        return Ok(());
    };
    let app = build_switchyard_router(state_from_yaml(&format!(
        r#"
targets:
  strong:
    model: provider/strong
    format: openai
    base_url: {base_url}
  weak:
    model: provider/weak
    format: openai
    base_url: {base_url}
profiles:
  smart-stage_router:
    type: stage_router
    strong: strong
    weak: weak
    fallback_target_on_evict: strong
    picker: stage_router_strong_default
    confidence_threshold: 0.0
"#,
        base_url = stub.base_url
    ))?);

    let response = app
        .clone()
        .oneshot(request(
            "POST",
            "/v1/chat/completions",
            Some(chat_body("smart-stage_router")),
        )?)
        .await?;
    assert_eq!(response.status(), StatusCode::OK);

    let seen = stub.requests()?;
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0]["body"]["model"], "provider/weak");

    let stats = app.oneshot(request("GET", "/v1/stats", None)?).await?;
    let stats = json_body(stats).await?;
    assert_eq!(stats["routing_decisions"]["stage_router"]["dimensions"], 1);
    assert_eq!(stats["classifier"]["total_requests"], 0);
    assert_eq!(stats["models"]["provider/weak"]["tier"], "weak");
    Ok(())
}

#[tokio::test]
async fn stage_router_profile_threshold_one_uses_llm_classifier_path() -> TestResult {
    let _stats_guard = STATS_TEST_LOCK.lock().await;
    reset_stats()?;
    let Some(stub) = HttpStub::start(vec![StubResponse::classifier("strong"), StubResponse::ok()])?
    else {
        log_loopback_bind_skip();
        return Ok(());
    };
    let app = build_switchyard_router(state_from_yaml(&format!(
        r#"
targets:
  strong:
    model: provider/strong
    format: openai
    base_url: {base_url}
  weak:
    model: provider/weak
    format: openai
    base_url: {base_url}
profiles:
  smart-stage_router:
    type: stage_router
    strong: strong
    weak: weak
    fallback_target_on_evict: strong
    picker: stage_router_strong_default
    confidence_threshold: 1.0
    classifier:
      model: classifier/model
      api_key: test-key
      base_url: {base_url}
      timeout_secs: 5.0
"#,
        base_url = stub.base_url
    ))?);

    let response = app
        .clone()
        .oneshot(request(
            "POST",
            "/v1/chat/completions",
            Some(chat_body("smart-stage_router")),
        )?)
        .await?;
    assert_eq!(response.status(), StatusCode::OK);

    let seen = stub.requests()?;
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0]["body"]["model"], "classifier/model");
    assert_eq!(seen[1]["body"]["model"], "provider/strong");

    let stats = app.oneshot(request("GET", "/v1/stats", None)?).await?;
    let stats = json_body(stats).await?;
    assert_eq!(stats["routing_decisions"]["stage_router"]["llm-classifier"], 1);
    assert_eq!(stats["classifier"]["total_requests"], 1);
    assert_eq!(stats["models"]["provider/strong"]["tier"], "strong");
    Ok(())
}

#[tokio::test]
async fn stage_router_profile_retries_fallback_after_context_overflow() -> TestResult {
    let _stats_guard = STATS_TEST_LOCK.lock().await;
    reset_stats()?;
    let Some(stub) = HttpStub::start(vec![StubResponse::context_overflow(), StubResponse::ok()])?
    else {
        log_loopback_bind_skip();
        return Ok(());
    };
    let app = build_switchyard_router(state_from_yaml(&format!(
        r#"
targets:
  strong:
    model: provider/strong
    format: openai
    base_url: {base_url}
  weak:
    model: provider/weak
    format: openai
    base_url: {base_url}
profiles:
  smart-stage_router:
    type: stage_router
    strong: strong
    weak: weak
    fallback_target_on_evict: strong
    picker: stage_router_weak_default
    confidence_threshold: 0.7
"#,
        base_url = stub.base_url
    ))?);

    let response = app
        .clone()
        .oneshot(request(
            "POST",
            "/v1/chat/completions",
            Some(chat_body("smart-stage_router")),
        )?)
        .await?;
    assert_eq!(response.status(), StatusCode::OK);

    let seen = stub.requests()?;
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0]["body"]["model"], "provider/weak");
    assert_eq!(seen[1]["body"]["model"], "provider/strong");

    let stats = app.oneshot(request("GET", "/v1/stats", None)?).await?;
    let stats = json_body(stats).await?;
    assert_eq!(stats["routing_decisions"]["stage_router"]["fall_open"], 1);
    assert_eq!(stats["models"]["provider/strong"]["tier"], "strong");
    Ok(())
}

#[tokio::test]
async fn latency_service_profile_uses_health_selection_and_retries_failed_target() -> TestResult {
    let _stats_guard = STATS_TEST_LOCK.lock().await;
    reset_stats()?;
    let Some(stub) = HttpStub::start(vec![
        StubResponse::latency_health(),
        StubResponse::error(503, json!({"error": {"message": "fast unavailable"}})),
        StubResponse::ok(),
    ])?
    else {
        log_loopback_bind_skip();
        return Ok(());
    };
    let app = build_switchyard_router(state_from_yaml(&latency_profile_yaml(&stub.origin, 1))?);

    let response = app
        .oneshot(request(
            "POST",
            "/v1/chat/completions",
            Some(chat_body("latency-profile")),
        )?)
        .await?;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        json_body(response).await?["choices"][0]["message"]["content"],
        "stub-ok"
    );

    let seen = stub.requests()?;
    assert_eq!(seen.len(), 3);
    assert_eq!(seen[0]["method"], "GET");
    assert!(seen[0]["path"]
        .as_str()
        .unwrap_or_default()
        .starts_with("/v1/endpoints/health?"));
    assert!(seen[0]["path"]
        .as_str()
        .unwrap_or_default()
        .contains("endpoint_ids=fast"));
    assert!(seen[0]["path"]
        .as_str()
        .unwrap_or_default()
        .contains("endpoint_ids=slow"));
    assert_eq!(seen[1]["path"], "/fast/v1/chat/completions");
    assert_eq!(seen[1]["body"]["model"], "provider/fast");
    assert_eq!(seen[2]["path"], "/slow/v1/chat/completions");
    assert_eq!(seen[2]["body"]["model"], "provider/slow");
    Ok(())
}

#[tokio::test]
async fn latency_service_profile_propagates_last_upstream_error_after_retries() -> TestResult {
    let _stats_guard = STATS_TEST_LOCK.lock().await;
    reset_stats()?;
    let Some(stub) = HttpStub::start(vec![
        StubResponse::latency_health(),
        StubResponse::error(503, json!({"error": {"message": "fast unavailable"}})),
        StubResponse::error(502, json!({"error": {"message": "slow unavailable"}})),
    ])?
    else {
        log_loopback_bind_skip();
        return Ok(());
    };
    let app = build_switchyard_router(state_from_yaml(&latency_profile_yaml(&stub.origin, 1))?);

    let response = app
        .oneshot(request(
            "POST",
            "/v1/chat/completions",
            Some(chat_body("latency-profile")),
        )?)
        .await?;
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let body = json_body(response).await?;
    assert_eq!(body["error"]["type"], "upstream_error");
    assert!(body["error"]["message"]
        .as_str()
        .unwrap_or_default()
        .contains("slow unavailable"));

    let seen = stub.requests()?;
    assert_eq!(seen.len(), 3);
    assert_eq!(seen[1]["body"]["model"], "provider/fast");
    assert_eq!(seen[2]["body"]["model"], "provider/slow");
    Ok(())
}

#[tokio::test]
async fn profile_config_negative_cases_fail_before_legacy_deletion() -> TestResult {
    let unknown_profile_type = state_from_yaml(
        r#"
profiles:
  bad:
    type: not-a-profile
"#,
    )
    .err()
    .ok_or("unknown profile type should fail")?
    .to_string();
    assert!(unknown_profile_type.contains("unknown profile type"));
    assert!(unknown_profile_type.contains("not-a-profile"));

    let unknown_target = state_from_yaml(
        r#"
targets: {}
profiles:
  bad:
    type: passthrough
    target: missing
"#,
    )
    .err()
    .ok_or("unknown target should fail")?
    .to_string();
    assert!(unknown_target.contains("unknown target missing"));

    let invalid_config = state_from_yaml(
        r#"
targets:
  strong:
    model: provider/strong
    format: openai
  weak:
    model: provider/weak
    format: openai
profiles:
  random-profile:
    type: random-routing
    strong: strong
    weak: weak
    strong_probability: 2.0
"#,
    )
    .err()
    .ok_or("invalid random-routing probability should fail")?
    .to_string();
    assert!(invalid_config.contains("strong_probability"));
    Ok(())
}

#[tokio::test]
async fn unknown_inbound_model_stays_a_client_error() -> TestResult {
    let app = build_switchyard_router(state_from_yaml(
        r#"
profiles:
  bench:
    type: noop
"#,
    )?);

    let response = app
        .oneshot(request(
            "POST",
            "/v1/chat/completions",
            Some(chat_body("missing")),
        )?)
        .await?;

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    assert_eq!(
        json_body(response).await?["error"]["code"],
        "model_not_found"
    );
    Ok(())
}

#[derive(Clone)]
struct StubResponse {
    status: u16,
    body: Value,
}

impl StubResponse {
    fn ok() -> Self {
        Self {
            status: 200,
            body: json!({
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "model": "stub",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "stub-ok"},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }),
        }
    }

    fn latency_health() -> Self {
        Self {
            status: 200,
            body: json!({
                "endpoint_health": {
                    "fast": {"status": "healthy", "last_latency_ms": 10.0},
                    "slow": {"status": "degraded", "last_latency_ms": 1.0},
                },
            }),
        }
    }

    fn error(status: u16, body: Value) -> Self {
        Self { status, body }
    }

    fn classifier(tier: &str) -> Self {
        Self {
            status: 200,
            body: json!({
                "id": "chatcmpl-classifier",
                "object": "chat.completion",
                "model": "classifier/model",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": json!({"tier": tier}).to_string()},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "total_tokens": 4,
                },
            }),
        }
    }

    fn context_overflow() -> Self {
        Self {
            status: 400,
            body: json!({
                "error": {
                    "code": "context_length_exceeded",
                    "message": "maximum context length exceeded",
                },
            }),
        }
    }
}

/// In-process HTTP stub that serves one response per accepted request.
struct HttpStub {
    /// Server origin used by latency-service health polling.
    origin: String,
    /// OpenAI-compatible base URL ending in `/v1`.
    base_url: String,
    /// Bound address used to wake blocked accepts during drop.
    addr: SocketAddr,
    /// Number of requests the stub thread expects before exiting.
    expected_requests: usize,
    /// Captured method, path, and JSON body for each real request.
    requests: Arc<Mutex<Vec<Value>>>,
    /// Background accept loop joined on drop after wake-up connects.
    handle: Option<thread::JoinHandle<()>>,
}

impl HttpStub {
    /// Binds an ephemeral port and returns `None` when loopback is sandbox-denied.
    fn start(responses: Vec<StubResponse>) -> TestResult<Option<Self>> {
        let listener = match StdTcpListener::bind("127.0.0.1:0") {
            Ok(listener) => listener,
            Err(error) if error.kind() == ErrorKind::PermissionDenied => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let addr = listener.local_addr()?;
        let expected_requests = responses.len();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let thread_requests = Arc::clone(&requests);
        let handle = thread::spawn(move || {
            for response in responses {
                let Ok((mut stream, _addr)) = listener.accept() else {
                    return;
                };
                if let Ok((method, path, body)) = read_http_request(&mut stream) {
                    if let Ok(mut requests) = thread_requests.lock() {
                        requests.push(json!({"method": method, "path": path, "body": body}));
                    }
                }
                let body = response.body.to_string();
                let reason = if response.status == 200 {
                    "OK"
                } else {
                    "ERROR"
                };
                let _ = write!(
                    stream,
                    "HTTP/1.1 {} {}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    response.status,
                    reason,
                    body.len(),
                    body
                );
            }
        });

        Ok(Some(Self {
            origin: format!("http://{addr}"),
            base_url: format!("http://{addr}/v1"),
            addr,
            expected_requests,
            requests,
            handle: Some(handle),
        }))
    }

    fn requests(&self) -> TestResult<Vec<Value>> {
        self.requests
            .lock()
            .map(|requests| requests.clone())
            .map_err(|_| "stub request mutex poisoned".into())
    }
}

impl Drop for HttpStub {
    fn drop(&mut self) {
        // Wake pending accepts before joining the stub thread.
        for _ in 0..self.expected_requests {
            let _ = TcpStream::connect(self.addr);
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Reads one small HTTP request, tolerating partial reads until headers and
/// the declared `content-length` body are available.
fn read_http_request(stream: &mut TcpStream) -> TestResult<(String, String, Value)> {
    let mut buffer = Vec::new();
    let mut header_end = None;
    while header_end.is_none() {
        let mut chunk = [0; 1024];
        let read = stream.read(&mut chunk)?;
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
        header_end = find_bytes(&buffer, b"\r\n\r\n").map(|index| index + 4);
    }

    let Some(body_start) = header_end else {
        return Err("HTTP request headers were incomplete".into());
    };
    let headers = std::str::from_utf8(&buffer[..body_start])?;
    let mut request_line = headers
        .lines()
        .next()
        .ok_or("HTTP request line was missing")?
        .split_whitespace();
    let method = request_line
        .next()
        .ok_or("HTTP request line was missing a method")?
        .to_string();
    let path = request_line
        .next()
        .ok_or("HTTP request line was missing a path")?
        .to_string();
    let content_length = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0);

    while buffer.len() < body_start + content_length {
        let mut chunk = [0; 1024];
        let read = stream.read(&mut chunk)?;
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
    }

    if buffer.len() < body_start + content_length {
        return Err("HTTP request body was incomplete".into());
    }
    let body = if content_length == 0 {
        Value::Null
    } else {
        serde_json::from_slice(&buffer[body_start..body_start + content_length])?
    };
    Ok((method, path, body))
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn state_from_yaml(input: &str) -> TestResult<ServerState> {
    let plan = parse_profile_config_str(input, ProfileConfigFormat::Yaml)?.resolve()?;
    Ok(ServerState::from_plan(&plan)?)
}

fn chat_body(model: &str) -> Value {
    json!({"model": model, "messages": [{"role": "user", "content": "hi"}]})
}

fn latency_profile_yaml(origin: &str, max_retries: usize) -> String {
    format!(
        r#"
targets:
  fast:
    model: provider/fast
    format: openai
    base_url: {origin}/fast/v1
  slow:
    model: provider/slow
    format: openai
    base_url: {origin}/slow/v1
profiles:
  latency-profile:
    type: latency-service
    latency_service_url: {origin}
    targets: [fast, slow]
    max_retries: {max_retries}
"#
    )
}

fn request(method: &str, uri: &str, body: Option<Value>) -> TestResult<Request<Body>> {
    let builder = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json");
    let body = body.map_or_else(Body::empty, |body| Body::from(body.to_string()));
    Ok(builder.body(body)?)
}

async fn json_body(response: axum::response::Response) -> TestResult<Value> {
    let bytes = response.into_body().collect().await?.to_bytes();
    Ok(serde_json::from_slice(&bytes)?)
}

fn reset_stats() -> switchyard_core::Result<()> {
    profile_stats_accumulator().reset()
}

fn log_loopback_bind_skip() {
    eprintln!("SKIP: permission denied binding loopback socket");
}

type TestResult<T = ()> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
