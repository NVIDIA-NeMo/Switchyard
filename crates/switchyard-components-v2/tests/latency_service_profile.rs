// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public API tests for the components-v2 latency-service profile config.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use serde_json::{json, Value};
use switchyard_components::otel_metrics;
use switchyard_components_v2::{
    LatencyServiceProfileConfig, Profile, ProfileConfig, ProfileInput, RequestMetadata,
};
use switchyard_core::{
    BackendFormat, ChatRequest, ChatResponse, EndpointConfig, LlmTarget, LlmTargetId, ModelId,
    Result, SwitchyardError,
};

fn target(id: &str, model: &str) -> Result<LlmTarget> {
    let mut target = LlmTarget::new(LlmTargetId::new(id)?, ModelId::new(model)?);
    target.format = BackendFormat::OpenAi;
    Ok(target)
}

fn config(targets: Vec<LlmTarget>) -> LatencyServiceProfileConfig {
    LatencyServiceProfileConfig {
        latency_service_url: "http://latency.test".to_string(),
        targets,
        poll_timeout_secs: 5.0,
        max_retries: 2,
    }
}

fn openai_target(id: &'static str, model: &'static str, base_url: &str) -> LlmTarget {
    LlmTarget {
        id: LlmTargetId::from_static(id),
        model: ModelId::from_static(model),
        format: BackendFormat::OpenAi,
        endpoint: EndpointConfig {
            base_url: Some(base_url.to_string()),
            api_key: Some("test-key".to_string()),
            timeout_secs: Some(5.0),
        },
        extra_body: None,
        extra_headers: BTreeMap::new(),
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
struct MockCounts {
    requests: Vec<HttpRequest>,
    health: u64,
    fast: u64,
    slow: u64,
    fast_body: Option<Value>,
    slow_body: Option<Value>,
}

#[derive(Clone, Debug, PartialEq)]
struct HttpRequest {
    method: String,
    path: String,
    body: Value,
}

struct MockLatencyOpenAiServer {
    base_url: String,
    address: SocketAddr,
    receiver: Receiver<Result<MockCounts>>,
    handle: Option<JoinHandle<()>>,
}

impl MockLatencyOpenAiServer {
    fn spawn(requests: usize) -> Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")
            .map_err(|error| SwitchyardError::Other(format!("bind failed: {error}")))?;
        let address = listener
            .local_addr()
            .map_err(|error| SwitchyardError::Other(format!("local addr failed: {error}")))?;
        let (sender, receiver) = mpsc::channel();
        let handle = thread::spawn(move || {
            let result = run_mock_server(listener, requests);
            let _ignored = sender.send(result);
        });
        Ok(Self {
            base_url: format!("http://{address}"),
            address,
            receiver,
            handle: Some(handle),
        })
    }

    fn base_url(&self) -> String {
        self.base_url.clone()
    }

    fn finish(&mut self) -> Result<MockCounts> {
        let result = match self.receiver.recv_timeout(Duration::from_secs(5)) {
            Ok(result) => result,
            Err(RecvTimeoutError::Timeout) => {
                self.wake();
                self.receiver
                    .recv_timeout(Duration::from_secs(1))
                    .map_err(|error| {
                        SwitchyardError::Other(format!("mock server did not finish: {error}"))
                    })?
            }
            Err(RecvTimeoutError::Disconnected) => {
                Err(SwitchyardError::Other("mock server disconnected".into()))
            }
        };
        if let Some(handle) = self.handle.take() {
            handle
                .join()
                .map_err(|_| SwitchyardError::Other("mock server thread panicked".into()))?;
        }
        result
    }

    fn wake(&self) {
        let _ignored = TcpStream::connect(self.address);
    }
}

impl Drop for MockLatencyOpenAiServer {
    fn drop(&mut self) {
        if self.handle.is_some() {
            self.wake();
        }
        if let Some(handle) = self.handle.take() {
            let _ignored = handle.join();
        }
    }
}

fn run_mock_server(listener: TcpListener, requests: usize) -> Result<MockCounts> {
    let mut counts = MockCounts::default();
    for _ in 0..requests {
        let (mut stream, _address) = listener
            .accept()
            .map_err(|error| SwitchyardError::Other(format!("accept failed: {error}")))?;
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .map_err(|error| SwitchyardError::Other(format!("set read timeout failed: {error}")))?;
        stream
            .set_write_timeout(Some(Duration::from_secs(5)))
            .map_err(|error| {
                SwitchyardError::Other(format!("set write timeout failed: {error}"))
            })?;
        let request = read_http_request(&mut stream)?;
        let body = response_for_request(request, &mut counts)?;
        write_json_response(&mut stream, 200, body)?;
    }
    Ok(counts)
}

fn read_http_request(stream: &mut TcpStream) -> Result<HttpRequest> {
    let mut bytes = Vec::new();
    let mut chunk = [0_u8; 4096];
    loop {
        let read = stream
            .read(&mut chunk)
            .map_err(|error| SwitchyardError::Other(format!("read request failed: {error}")))?;
        if read == 0 {
            break;
        }
        bytes.extend_from_slice(&chunk[..read]);
        if let Some((header_end, content_length)) = request_shape(&bytes) {
            let body_start = header_end + 4;
            if bytes.len() >= body_start + content_length {
                break;
            }
        }
    }

    let (header_end, content_length) = request_shape(&bytes)
        .ok_or_else(|| SwitchyardError::Other("request should contain headers".into()))?;
    let header_text = std::str::from_utf8(&bytes[..header_end])
        .map_err(|error| SwitchyardError::Other(format!("headers should be UTF-8: {error}")))?;
    let body_start = header_end + 4;
    let body_end = body_start + content_length;

    let mut lines = header_text.lines();
    let first_line = lines
        .next()
        .ok_or_else(|| SwitchyardError::Other("missing request line".into()))?;
    let mut parts = first_line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| SwitchyardError::Other("missing request method".into()))?
        .to_string();
    let path = parts
        .next()
        .ok_or_else(|| SwitchyardError::Other("missing request path".into()))?
        .to_string();

    let raw_body = &bytes[body_start..body_end];
    let body = if raw_body.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(raw_body)
            .map_err(|error| SwitchyardError::Other(format!("decode request body: {error}")))?
    };
    Ok(HttpRequest { method, path, body })
}

fn request_shape(bytes: &[u8]) -> Option<(usize, usize)> {
    let header_end = find_bytes(bytes, b"\r\n\r\n")?;
    let headers = std::str::from_utf8(&bytes[..header_end]).ok()?;
    let content_length = headers
        .lines()
        .filter_map(|line| line.split_once(':'))
        .find_map(|(name, value)| {
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0);
    Some((header_end, content_length))
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn response_for_request(request: HttpRequest, counts: &mut MockCounts) -> Result<Value> {
    counts.requests.push(request.clone());
    if request.method == "GET" && request.path.starts_with("/v1/endpoints/health") {
        counts.health = counts.health.saturating_add(1);
        return Ok(json!({
            "endpoint_health": {
                "fast": {"status": "healthy", "last_latency_ms": 10.0},
                "slow": {"status": "degraded", "last_latency_ms": 1.0}
            }
        }));
    }

    if request.method == "POST" && request.path == "/fast/v1/chat/completions" {
        counts.fast = counts.fast.saturating_add(1);
        counts.fast_body = Some(request.body.clone());
        return Ok(openai_completion("fast", request.body));
    }

    if request.method == "POST" && request.path == "/slow/v1/chat/completions" {
        counts.slow = counts.slow.saturating_add(1);
        counts.slow_body = Some(request.body.clone());
        return Ok(openai_completion("slow", request.body));
    }

    Err(SwitchyardError::Other(format!(
        "unexpected mock request {} {}",
        request.method, request.path
    )))
}

fn openai_completion(endpoint: &'static str, request_body: Value) -> Value {
    json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": request_body.get("model").cloned().unwrap_or(Value::Null),
        "mock_endpoint": endpoint,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 2,
            "completion_tokens": 1,
            "total_tokens": 3
        }
    })
}

fn write_json_response(stream: &mut std::net::TcpStream, status: u16, body: Value) -> Result<()> {
    let body = serde_json::to_string(&body)
        .map_err(|error| SwitchyardError::Other(format!("encode response body: {error}")))?;
    let response = format!(
        "HTTP/1.1 {status} OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    stream
        .write_all(response.as_bytes())
        .map_err(|error| SwitchyardError::Other(format!("write response failed: {error}")))
}

#[test]
fn profile_config_build_uses_existing_native_backend_stack() -> Result<()> {
    let config = config(vec![
        target("fast", "upstream-fast")?,
        target("slow", "upstream-slow")?,
    ]);

    let profile = config.build()?;

    assert_eq!(profile.health_snapshot().len(), 2);
    assert!(!profile.is_ready());
    Ok(())
}

#[tokio::test]
async fn profile_run_polls_and_routes_native_openai_call_to_healthy_target() -> Result<()> {
    let mut server = MockLatencyOpenAiServer::spawn(2)?;
    let fast = openai_target(
        "fast",
        "upstream-fast-native",
        &format!("{}/fast/v1", server.base_url()),
    );
    let slow = openai_target(
        "slow",
        "upstream-slow-native",
        &format!("{}/slow/v1", server.base_url()),
    );
    let config = LatencyServiceProfileConfig {
        latency_service_url: server.base_url(),
        targets: vec![fast, slow],
        poll_timeout_secs: 5.0,
        max_retries: 0,
    };
    let profile = config.build()?;

    let response = profile
        .run(ProfileInput {
            request: ChatRequest::openai_chat(json!({
                "model": "client-model",
                "messages": [{"role": "user", "content": "route me"}],
                "max_tokens": 8
            })),
            metadata: RequestMetadata::default(),
        })
        .await?;

    let routing_metadata = response
        .routing_metadata
        .as_ref()
        .ok_or_else(|| SwitchyardError::Other("routing metadata missing".into()))?;
    assert_eq!(
        routing_metadata.selected_model.as_deref(),
        Some("upstream-fast-native")
    );
    assert_eq!(routing_metadata.selected_tier.as_deref(), Some("healthy"));
    let response = response.response;

    match response {
        ChatResponse::OpenAiCompletion(body) => {
            assert_eq!(body.body()["mock_endpoint"], "fast");
            assert_eq!(body.body()["model"], "upstream-fast-native");
        }
        _ => return Err(SwitchyardError::Other("unexpected response shape".into())),
    }
    let counts = server.finish()?;
    assert_eq!(counts.health, 1);
    assert_eq!(counts.fast, 1);
    assert_eq!(counts.slow, 0);
    assert_eq!(counts.requests.len(), 2);
    assert_eq!(counts.requests[0].method, "GET");
    assert!(counts.requests[0].path.starts_with("/v1/endpoints/health?"));
    assert!(counts.requests[0].path.contains("endpoint_ids=fast"));
    assert!(counts.requests[0].path.contains("endpoint_ids=slow"));
    assert_eq!(counts.requests[1].method, "POST");
    assert_eq!(counts.requests[1].path, "/fast/v1/chat/completions");
    assert!(!counts
        .requests
        .iter()
        .any(|request| request.path == "/slow/v1/chat/completions"));
    assert_eq!(
        counts
            .fast_body
            .as_ref()
            .and_then(|body| body.get("model"))
            .and_then(Value::as_str),
        Some("upstream-fast-native")
    );

    // The process meter is a singleton; assert the metric surface mentions the
    // routed model and the latency_service router rather than exact counts.
    let metrics = otel_metrics::render_prometheus().map_err(SwitchyardError::Other)?;
    assert!(
        metrics.contains("upstream-fast-native"),
        "expected metrics for the routed model in:\n{metrics}"
    );
    assert!(metrics.contains("router=\"latency_service\""));
    Ok(())
}

#[test]
fn profile_config_macro_adds_type_metadata_and_strict_serde() -> Result<()> {
    let config = config(vec![target("fast", "upstream-fast")?]);

    assert_eq!(LatencyServiceProfileConfig::PROFILE_TYPE, "latency-service");
    assert_eq!(config.profile_type(), "latency-service");

    let old_poller_field = json!({
        "latency_service_url": "http://latency.test",
        "targets": config.targets,
        "poll_timeout_secs": 5.0,
        "max_retries": 2,
        "poll_interval_secs": 1.0,
    });
    let error = serde_json::from_value::<LatencyServiceProfileConfig>(old_poller_field)
        .err()
        .ok_or_else(|| SwitchyardError::Other("unknown profile field should fail".into()))?;
    assert!(error.to_string().contains("unknown field"));
    Ok(())
}

#[test]
fn invalid_profile_config_is_rejected_by_build() -> Result<()> {
    let fast = target("fast", "upstream-fast")?;
    let duplicate = config(vec![fast.clone(), fast]);

    match duplicate.build() {
        Err(SwitchyardError::InvalidConfig(message)) => {
            assert!(message.contains("duplicate target fast"));
        }
        Ok(_) => {
            return Err(SwitchyardError::Other(
                "duplicate targets should reject profile construction".into(),
            ));
        }
        Err(other) => {
            return Err(SwitchyardError::Other(format!(
                "expected InvalidConfig, got {other}"
            )));
        }
    }

    let mut bad_timeout = config(vec![target("fast", "upstream-fast")?]);
    bad_timeout.poll_timeout_secs = f64::INFINITY;
    match bad_timeout.build() {
        Err(SwitchyardError::InvalidConfig(message)) => {
            assert!(message.contains("poll_timeout_secs"));
        }
        Ok(_) => {
            return Err(SwitchyardError::Other(
                "invalid timeout should reject profile construction".into(),
            ));
        }
        Err(other) => {
            return Err(SwitchyardError::Other(format!(
                "expected InvalidConfig, got {other}"
            )));
        }
    }
    Ok(())
}
