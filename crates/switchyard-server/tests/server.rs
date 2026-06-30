// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the components-v2 Rust profile server.

use std::io::ErrorKind;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener as StdTcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;

use async_trait::async_trait;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use switchyard_components_v2::{
    parse_profile_config_str, Profile, ProfileConfigFormat, ProfileInput, ProfileResponse,
    RoutingMetadata,
};
use switchyard_core::{
    ChatRequestType, ChatResponse, ModelId, Result, StreamEvent, SwitchyardError,
};
use switchyard_server::{build_switchyard_router, ProfileRegistry, ServerState};
use tower::ServiceExt;

#[tokio::test]
async fn minimal_noop_config_boots_and_serves_core_routes() -> TestResult {
    let app = build_switchyard_router(state_from_yaml(
        r#"
profiles:
  bench:
    type: noop
"#,
    )?);

    let health = app
        .clone()
        .oneshot(request("GET", "/health", None)?)
        .await?;
    assert_eq!(health.status(), StatusCode::OK);
    assert_eq!(json_body(health).await?, json!({"status": "ok"}));

    let models = app
        .clone()
        .oneshot(request("GET", "/v1/models", None)?)
        .await?;
    assert_eq!(models.status(), StatusCode::OK);
    let models = json_body(models).await?;
    assert_eq!(models["object"], "list");
    assert_eq!(models["data"][0]["id"], "bench");
    assert_eq!(models["default_model"], "bench");
    assert_eq!(models["model_pool"], json!(["bench"]));

    let chat = app
        .oneshot(request(
            "POST",
            "/v1/chat/completions",
            Some(json!({
                "model": "bench",
                "messages": [{"role": "user", "content": "hi"}],
            })),
        )?)
        .await?;
    assert_eq!(chat.status(), StatusCode::OK);
    assert_eq!(
        json_body(chat).await?["choices"][0]["message"]["content"],
        "ok"
    );
    Ok(())
}

#[tokio::test]
async fn all_request_endpoints_route_through_selected_profile() -> TestResult {
    let app = build_switchyard_router(state_from_yaml(
        r#"
profiles:
  bench:
    type: noop
"#,
    )?);

    let anthropic = app
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
    assert_eq!(anthropic.status(), StatusCode::OK);
    let anthropic = json_body(anthropic).await?;
    assert_eq!(anthropic["type"], "message");
    assert_eq!(anthropic["content"][0]["text"], "ok");

    let responses = app
        .oneshot(request(
            "POST",
            "/v1/responses",
            Some(json!({"model": "bench", "input": "hi"})),
        )?)
        .await?;
    assert_eq!(responses.status(), StatusCode::OK);
    let responses = json_body(responses).await?;
    assert_eq!(responses["object"], "response");
    assert_eq!(responses["output"][0]["content"][0]["text"], "ok");
    Ok(())
}

#[tokio::test]
async fn missing_and_unknown_models_return_client_errors() -> TestResult {
    let app = build_switchyard_router(state_from_yaml(
        r#"
profiles:
  bench:
    type: noop
"#,
    )?);

    let missing = app
        .clone()
        .oneshot(request(
            "POST",
            "/v1/chat/completions",
            Some(json!({"messages": [{"role": "user", "content": "hi"}]})),
        )?)
        .await?;
    assert_eq!(missing.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        json_body(missing).await?["error"]["type"],
        "invalid_request_error"
    );

    let unknown = app
        .oneshot(request(
            "POST",
            "/v1/chat/completions",
            Some(json!({
                "model": "missing-route",
                "messages": [{"role": "user", "content": "hi"}],
            })),
        )?)
        .await?;
    assert_eq!(unknown.status(), StatusCode::NOT_FOUND);
    assert_eq!(
        json_body(unknown).await?["error"]["type"],
        "model_not_found"
    );
    Ok(())
}

#[tokio::test]
async fn malformed_and_non_object_json_return_shared_client_errors() -> TestResult {
    let app = build_switchyard_router(state_from_yaml(
        r#"
profiles:
  bench:
    type: noop
"#,
    )?);

    for uri in ["/v1/chat/completions", "/v1/messages", "/v1/responses"] {
        let malformed = app.clone().oneshot(raw_request("POST", uri, "{")?).await?;
        assert_eq!(malformed.status(), StatusCode::BAD_REQUEST);
        assert_eq!(json_body(malformed).await?["error"]["code"], "invalid_body");

        let non_object = app.clone().oneshot(raw_request("POST", uri, "[]")?).await?;
        assert_eq!(non_object.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            json_body(non_object).await?["error"]["code"],
            "invalid_body"
        );
    }
    Ok(())
}

#[tokio::test]
async fn translation_errors_do_not_emit_routing_metadata_headers() -> TestResult {
    let app = build_switchyard_router(state_from_profile("bad", Arc::new(BadTranslationProfile))?);

    let response = app
        .oneshot(request(
            "POST",
            "/v1/messages",
            Some(json!({
                "model": "bad",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
            })),
        )?)
        .await?;

    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert!(!response
        .headers()
        .keys()
        .any(|name| name.as_str().starts_with("x-model-router-")));
    Ok(())
}

#[tokio::test]
async fn target_id_and_target_model_aliases_are_advertised_and_routable() -> TestResult {
    let Some(stub) = HttpStub::start(2)? else {
        log_loopback_bind_skip();
        return Ok(());
    };
    let app = build_switchyard_router(state_from_yaml(&format!(
        r#"
targets:
  direct:
    model: upstream-direct
    format: openai
    base_url: {base_url}
profiles:
  direct-profile:
    type: passthrough
    target: direct
"#,
        base_url = stub.base_url
    ))?);

    let models = app
        .clone()
        .oneshot(request("GET", "/v1/models", None)?)
        .await?;
    let model_ids = json_body(models).await?["model_pool"].clone();
    assert_eq!(
        model_ids,
        json!(["direct-profile", "direct", "upstream-direct"])
    );

    for public_model in ["direct", "upstream-direct"] {
        let response = app
            .clone()
            .oneshot(request(
                "POST",
                "/v1/chat/completions",
                Some(json!({
                    "model": public_model,
                    "messages": [{"role": "user", "content": "hi"}],
                })),
            )?)
            .await?;
        assert_eq!(response.status(), StatusCode::OK);
    }

    let seen = stub.requests()?;
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0]["model"], "upstream-direct");
    assert_eq!(seen[1]["model"], "upstream-direct");
    Ok(())
}

#[tokio::test]
async fn target_with_same_id_and_model_is_registered_once() -> TestResult {
    let Some(stub) = HttpStub::start(1)? else {
        log_loopback_bind_skip();
        return Ok(());
    };
    let app = build_switchyard_router(state_from_yaml(&format!(
        r#"
targets:
  upstream-direct:
    model: upstream-direct
    format: openai
    base_url: {base_url}
profiles:
  direct-profile:
    type: passthrough
    target: upstream-direct
"#,
        base_url = stub.base_url
    ))?);

    let models = app
        .clone()
        .oneshot(request("GET", "/v1/models", None)?)
        .await?;
    assert_eq!(
        json_body(models).await?["model_pool"],
        json!(["direct-profile", "upstream-direct"])
    );

    let response = app
        .oneshot(request(
            "POST",
            "/v1/chat/completions",
            Some(json!({
                "model": "upstream-direct",
                "messages": [{"role": "user", "content": "hi"}],
            })),
        )?)
        .await?;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(stub.requests()?[0]["model"], "upstream-direct");
    Ok(())
}

#[tokio::test]
async fn random_routing_profile_reaches_selected_backend_path() -> TestResult {
    let Some(stub) = HttpStub::start(1)? else {
        log_loopback_bind_skip();
        return Ok(());
    };
    let app = build_switchyard_router(state_from_yaml(&format!(
        r#"
targets:
  strong:
    model: upstream-strong
    format: openai
    base_url: {base_url}
  weak:
    model: upstream-weak
    format: openai
    base_url: {base_url}
profiles:
  random:
    type: random-routing
    strong: strong
    weak: weak
    strong_probability: 0.0000004
    rng_seed: 7
"#,
        base_url = stub.base_url
    ))?);

    let response = app
        .oneshot(request(
            "POST",
            "/v1/chat/completions",
            Some(json!({
                "model": "random",
                "messages": [{"role": "user", "content": "hi"}],
            })),
        )?)
        .await?;
    assert_eq!(response.status(), StatusCode::OK);
    for (name, expected) in [
        ("x-model-router-selected-model", "upstream-weak"),
        ("x-model-router-selected-tier", "weak"),
        ("x-model-router-version", "random-routing:v1"),
        ("x-model-router-tolerance", "0.0000004"),
    ] {
        assert_eq!(header(&response, name), Some(expected));
    }
    assert!(header(&response, "x-model-router-rationale")
        .is_some_and(|value| value.contains("strong_probability 0.0000004; selected weak")));

    let seen = stub.requests()?;
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0]["model"], "upstream-weak");
    Ok(())
}

#[tokio::test]
async fn latency_service_profile_reaches_configured_backend_path() -> TestResult {
    let Some(stub) = HttpStub::start(1)? else {
        log_loopback_bind_skip();
        return Ok(());
    };
    let app = build_switchyard_router(state_from_yaml(&format!(
        r#"
targets:
  fast:
    model: upstream-fast
    format: openai
    base_url: {base_url}
profiles:
  latency:
    type: latency-service
    latency_service_url: http://latency.local
    targets: [fast]
"#,
        base_url = stub.base_url
    ))?);

    let response = app
        .oneshot(request(
            "POST",
            "/v1/chat/completions",
            Some(json!({
                "model": "latency",
                "messages": [{"role": "user", "content": "hi"}],
            })),
        )?)
        .await?;
    assert_eq!(response.status(), StatusCode::OK);

    let seen = stub.requests()?;
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0]["model"], "upstream-fast");
    Ok(())
}

#[tokio::test]
async fn duplicate_public_model_ids_are_rejected() -> TestResult {
    let err = state_from_yaml(
        r#"
targets:
  direct:
    model: same
    format: openai
    base_url: http://127.0.0.1:9/v1
profiles:
  same:
    type: noop
"#,
    )
    .err()
    .ok_or("expected duplicate public model id failure")?;

    assert!(err.to_string().contains("same"));
    assert!(err.to_string().contains("already registered"));
    Ok(())
}

#[tokio::test]
async fn metrics_endpoint_serves_prometheus_exposition() -> TestResult {
    let app = build_switchyard_router(state_from_yaml(
        r#"
profiles:
  bench:
    type: noop
"#,
    )?);

    let response = app.oneshot(request("GET", "/metrics", None)?).await?;
    assert_eq!(response.status(), StatusCode::OK);
    let content_type = header(&response, "content-type")
        .ok_or_else(|| SwitchyardError::Other("missing content-type".into()))?;
    assert!(content_type.starts_with("text/plain"));
    let body = response.into_body().collect().await?.to_bytes();
    let text = String::from_utf8(body.to_vec())?;
    // `ensure_instruments` lights up pull-only series even before any request.
    assert!(text.contains("switchyard_build_info"));
    Ok(())
}

#[tokio::test]
async fn openai_streams_are_sse_framed_with_done() -> TestResult {
    let app = build_switchyard_router(state_from_profile(
        "stream",
        Arc::new(StreamProfile {
            kind: StreamKind::OpenAi,
            routing_metadata: None,
        }),
    )?);

    let response = app
        .oneshot(request(
            "POST",
            "/v1/chat/completions",
            Some(json!({
                "model": "stream",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true,
            })),
        )?)
        .await?;
    assert_eq!(response.status(), StatusCode::OK);
    let body = text_body(response).await?;
    assert!(body.contains("data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}"));
    assert!(body.contains("data: [DONE]"));
    Ok(())
}

#[tokio::test]
async fn openai_streams_include_routing_metadata_headers() -> TestResult {
    let app = build_switchyard_router(state_from_profile(
        "stream",
        Arc::new(StreamProfile {
            kind: StreamKind::OpenAi,
            routing_metadata: Some(RoutingMetadata {
                selected_model: Some("served-model".to_string()),
                selected_tier: Some("weak".to_string()),
                confidence: Some(0.0000004),
                router_version: Some("test-router:v1".to_string()),
                tolerance: Some(0.0000004),
                rationale: Some("line\nbreak".to_string()),
            }),
        }),
    )?);

    let response = app
        .oneshot(request(
            "POST",
            "/v1/chat/completions",
            Some(json!({
                "model": "stream",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true,
            })),
        )?)
        .await?;
    assert_eq!(response.status(), StatusCode::OK);
    for (name, expected) in [
        ("x-model-router-selected-model", "served-model"),
        ("x-model-router-confidence", "0.0000004"),
        ("x-model-router-rationale", "line break"),
    ] {
        assert_eq!(header(&response, name), Some(expected));
    }
    let body = text_body(response).await?;
    assert!(body.contains("data: [DONE]"));
    Ok(())
}

#[tokio::test]
async fn anthropic_streams_are_named_sse_without_done() -> TestResult {
    let app = build_switchyard_router(state_from_profile(
        "stream",
        Arc::new(StreamProfile {
            kind: StreamKind::Anthropic,
            routing_metadata: None,
        }),
    )?);

    let response = app
        .oneshot(request(
            "POST",
            "/v1/messages",
            Some(json!({
                "model": "stream",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true,
            })),
        )?)
        .await?;
    assert_eq!(response.status(), StatusCode::OK);
    let body = text_body(response).await?;
    assert!(body.contains("event: message_start"));
    assert!(body.contains("\"type\":\"message_start\""));
    assert!(!body.contains("[DONE]"));
    Ok(())
}

#[tokio::test]
async fn endpoint_metadata_is_passed_to_profiles() -> TestResult {
    let captured = Arc::new(Mutex::new(None));
    let app = build_switchyard_router(state_from_profile(
        "capture",
        Arc::new(CaptureProfile {
            captured: Arc::clone(&captured),
        }),
    )?);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .header("X-Request-ID", "req-123")
                .header("X-Switchyard-Trace", "trace-a")
                .body(Body::from(
                    json!({
                        "model": "capture",
                        "max_tokens": 16,
                        "messages": [{"role": "user", "content": "hi"}],
                    })
                    .to_string(),
                ))?,
        )
        .await?;
    assert_eq!(response.status(), StatusCode::OK);

    let input = captured
        .lock()
        .map_err(|_| "captured input mutex poisoned")?
        .clone()
        .ok_or("profile should have received input")?;
    assert_eq!(input.request.model(), Some("capture"));
    assert_eq!(
        input.metadata.request_id.as_ref().map(|id| id.as_str()),
        Some("req-123")
    );
    assert_eq!(
        input.metadata.inbound_format,
        Some(ChatRequestType::Anthropic)
    );
    assert_eq!(
        input
            .metadata
            .headers
            .get("x-switchyard-trace")
            .map(Vec::as_slice),
        Some(&["trace-a".to_string()][..])
    );
    Ok(())
}

/// Test profile that emits one deterministic stream event for SSE framing checks.
#[derive(Clone)]
struct StreamProfile {
    kind: StreamKind,
    routing_metadata: Option<RoutingMetadata>,
}

/// Stream format variant emitted by `StreamProfile`.
#[derive(Clone, Copy)]
enum StreamKind {
    OpenAi,
    Anthropic,
}

#[async_trait]
impl Profile for StreamProfile {
    async fn run(&self, _input: ProfileInput) -> Result<ProfileResponse> {
        let response = match self.kind {
            StreamKind::OpenAi => ChatResponse::OpenAiStream(Box::pin(futures_util::stream::iter(
                [Ok(StreamEvent::Json(json!({
                    "id": "chatcmpl-test",
                    "object": "chat.completion.chunk",
                    "model": "served-model",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": "hello"},
                        "finish_reason": null,
                    }],
                })))],
            ))),
            StreamKind::Anthropic => ChatResponse::AnthropicStream(Box::pin(
                futures_util::stream::iter([Ok(StreamEvent::Json(json!({
                    "type": "message_start",
                    "message": {
                        "id": "msg-test",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": "claude-test",
                        "stop_reason": null,
                        "stop_sequence": null,
                        "usage": {"input_tokens": 1, "output_tokens": 0},
                    },
                })))]),
            )),
        };
        Ok(match &self.routing_metadata {
            Some(metadata) => ProfileResponse::with_routing_metadata(response, metadata.clone()),
            None => ProfileResponse::from(response),
        })
    }
}

struct CaptureProfile {
    captured: Arc<Mutex<Option<ProfileInput>>>,
}

struct BadTranslationProfile;

#[async_trait]
impl Profile for BadTranslationProfile {
    async fn run(&self, _input: ProfileInput) -> Result<ProfileResponse> {
        Ok(ProfileResponse::with_routing_metadata(
            ChatResponse::openai_completion(json!("not an OpenAI response object")),
            RoutingMetadata {
                selected_model: Some("bad-upstream".to_string()),
                selected_tier: Some("weak".to_string()),
                router_version: Some("test-router:v1".to_string()),
                ..RoutingMetadata::default()
            },
        ))
    }
}

#[async_trait]
impl Profile for CaptureProfile {
    async fn run(&self, input: ProfileInput) -> Result<ProfileResponse> {
        {
            let mut captured = self
                .captured
                .lock()
                .map_err(|_| SwitchyardError::Other("captured input mutex poisoned".to_string()))?;
            *captured = Some(input);
        }
        Ok(ChatResponse::anthropic_completion(json!({
            "id": "msg-capture",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "captured"}],
            "model": "capture",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }))
        .into())
    }
}

/// In-process HTTP stub that records JSON request bodies from backend calls.
struct HttpStub {
    base_url: String,
    addr: SocketAddr,
    expected_requests: usize,
    requests: Arc<Mutex<Vec<Value>>>,
    handle: Option<thread::JoinHandle<()>>,
}

impl HttpStub {
    /// Binds an ephemeral port and accepts the expected number of stub requests.
    fn start(expected_requests: usize) -> TestResult<Option<Self>> {
        let listener = match StdTcpListener::bind("127.0.0.1:0") {
            Ok(listener) => listener,
            Err(error) if error.kind() == ErrorKind::PermissionDenied => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let addr = listener.local_addr()?;
        let requests = Arc::new(Mutex::new(Vec::new()));
        let thread_requests = Arc::clone(&requests);
        let handle = thread::spawn(move || {
            for _ in 0..expected_requests {
                let Ok((mut stream, _addr)) = listener.accept() else {
                    return;
                };
                if let Ok(body) = read_http_body(&mut stream) {
                    if let Ok(value) = serde_json::from_slice::<Value>(&body) {
                        if let Ok(mut requests) = thread_requests.lock() {
                            requests.push(value);
                        }
                    }
                }
                let response = json!({
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
                })
                .to_string();
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    response.len(),
                    response
                );
            }
        });

        Ok(Some(Self {
            base_url: format!("http://{addr}/v1"),
            addr,
            expected_requests,
            requests,
            handle: Some(handle),
        }))
    }

    /// Returns the JSON request bodies captured by the stub thread.
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

fn read_http_body(stream: &mut std::net::TcpStream) -> TestResult<Vec<u8>> {
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

    Ok(buffer[body_start..body_start + content_length].to_vec())
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

fn state_from_profile(model: &'static str, profile: Arc<dyn Profile>) -> TestResult<ServerState> {
    let registry = ProfileRegistry::from_profiles([(
        ModelId::from_static(model),
        profile,
        model.to_string(),
    )])?;
    Ok(ServerState::new(registry))
}

fn request(method: &str, uri: &str, body: Option<Value>) -> TestResult<Request<Body>> {
    let builder = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json");
    let body = body.map_or_else(Body::empty, |body| Body::from(body.to_string()));
    Ok(builder.body(body)?)
}

fn raw_request(method: &str, uri: &str, body: &str) -> TestResult<Request<Body>> {
    let builder = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json");
    Ok(builder.body(Body::from(body.to_string()))?)
}

async fn json_body(response: axum::response::Response) -> TestResult<Value> {
    let bytes = response.into_body().collect().await?.to_bytes();
    Ok(serde_json::from_slice(&bytes)?)
}

async fn text_body(response: axum::response::Response) -> TestResult<String> {
    let bytes = response.into_body().collect().await?.to_bytes();
    Ok(String::from_utf8(bytes.to_vec())?)
}

fn header<'a>(response: &'a axum::response::Response, name: &str) -> Option<&'a str> {
    response
        .headers()
        .get(name)
        .and_then(|value| value.to_str().ok())
}

fn log_loopback_bind_skip() {
    eprintln!("SKIP: permission denied binding loopback socket");
}

type TestResult<T = ()> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
