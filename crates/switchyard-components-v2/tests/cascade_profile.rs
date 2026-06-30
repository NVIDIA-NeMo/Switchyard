// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public API tests for the components-v2 cascade profile config.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use serde_json::{json, Value};
use switchyard_components::otel_metrics;
use switchyard_components_v2::{
    CascadeClassifierConfig, CascadeDecisionSource, CascadePickerMode, CascadeProfileConfig,
    CascadeTier, Profile, ProfileConfig, ProfileHooks, ProfileInput, RequestMetadata,
};
use switchyard_core::{
    BackendFormat, ChatRequest, LlmTarget, LlmTargetId, ModelId, Result, SwitchyardError,
};

#[derive(Clone, Debug, PartialEq)]
struct ObservedRequest {
    path: String,
    body: Value,
}

enum MockResponse {
    Json(u16, Value),
    Raw(u16, &'static str),
}

const NON_JSON_CLASSIFIER_RESPONSE: &str = "__switchyard_non_json_classifier_response__";

struct MockOpenAiServer {
    addr: SocketAddr,
    requests: Arc<Mutex<Vec<ObservedRequest>>>,
    shutdown: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl MockOpenAiServer {
    fn spawn(classifier_content: Option<String>, backend_status: u16) -> Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")
            .map_err(|error| SwitchyardError::Other(format!("bind failed: {error}")))?;
        let addr = listener
            .local_addr()
            .map_err(|error| SwitchyardError::Other(format!("local_addr failed: {error}")))?;
        let requests = Arc::new(Mutex::new(Vec::new()));
        let shutdown = Arc::new(AtomicBool::new(false));
        let thread_requests = Arc::clone(&requests);
        let thread_shutdown = Arc::clone(&shutdown);
        let handle = thread::spawn(move || {
            for stream in listener.incoming() {
                if thread_shutdown.load(Ordering::SeqCst) {
                    break;
                }
                let Ok(mut stream) = stream else {
                    continue;
                };
                let Ok(request) = read_request(&mut stream) else {
                    continue;
                };
                if let Ok(mut requests) = thread_requests.lock() {
                    requests.push(request.clone());
                }
                let response =
                    response_for(&request, classifier_content.as_deref(), backend_status);
                let _ = write_response(&mut stream, response);
            }
        });
        Ok(Self {
            addr,
            requests,
            shutdown,
            handle: Some(handle),
        })
    }

    fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    fn requests(&self) -> Result<Vec<ObservedRequest>> {
        self.requests
            .lock()
            .map(|requests| requests.clone())
            .map_err(|_| SwitchyardError::Other("requests mutex poisoned".to_string()))
    }
}

impl Drop for MockOpenAiServer {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(self.addr);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn read_request(stream: &mut TcpStream) -> Result<ObservedRequest> {
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .map_err(|error| SwitchyardError::Other(format!("set timeout failed: {error}")))?;
    let mut bytes = Vec::new();
    let mut buf = [0_u8; 1024];
    let header_end = loop {
        let read = stream
            .read(&mut buf)
            .map_err(|error| SwitchyardError::Other(format!("read failed: {error}")))?;
        if read == 0 {
            return Err(SwitchyardError::Other(
                "connection closed early".to_string(),
            ));
        }
        bytes.extend_from_slice(&buf[..read]);
        if let Some(header_end) = find_header_end(&bytes) {
            break header_end;
        }
    };
    let headers = String::from_utf8_lossy(&bytes[..header_end]);
    let mut lines = headers.lines();
    let request_line = lines
        .next()
        .ok_or_else(|| SwitchyardError::Other("missing request line".to_string()))?;
    let path = request_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| SwitchyardError::Other("missing request path".to_string()))?
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
    let body_start = header_end + 4;
    while bytes.len().saturating_sub(body_start) < content_length {
        let read = stream
            .read(&mut buf)
            .map_err(|error| SwitchyardError::Other(format!("body read failed: {error}")))?;
        if read == 0 {
            break;
        }
        bytes.extend_from_slice(&buf[..read]);
    }
    let body_end = body_start + content_length;
    if bytes.len() < body_end {
        return Err(SwitchyardError::Other(
            "connection closed before full body was read".to_string(),
        ));
    }
    let body = if content_length == 0 {
        Value::Null
    } else {
        serde_json::from_slice(&bytes[body_start..body_end])
            .map_err(|error| SwitchyardError::Other(format!("decode request body: {error}")))?
    };
    Ok(ObservedRequest { path, body })
}

fn find_header_end(bytes: &[u8]) -> Option<usize> {
    bytes.windows(4).position(|window| window == b"\r\n\r\n")
}

fn response_for(
    request: &ObservedRequest,
    classifier_content: Option<&str>,
    backend_status: u16,
) -> MockResponse {
    if request.path.contains("/classifier/") {
        if classifier_content == Some(NON_JSON_CLASSIFIER_RESPONSE) {
            return MockResponse::Raw(200, "not json");
        }
        return MockResponse::Json(
            200,
            json!({
                "id": "chatcmpl-cascade-classifier",
                "object": "chat.completion",
                "model": request.body["model"],
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": classifier_content.unwrap_or("not json"),
                    },
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 11, "completion_tokens": 1, "total_tokens": 12},
            }),
        );
    }

    if backend_status >= 400 {
        return MockResponse::Json(
            backend_status,
            json!({"error": {"message": "selected backend failed", "code": "backend_failed"}}),
        );
    }
    MockResponse::Json(
        200,
        json!({
            "id": "chatcmpl-cascade-backend",
            "object": "chat.completion",
            "model": request.body["model"],
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }),
    )
}

fn write_response(stream: &mut TcpStream, response: MockResponse) -> Result<()> {
    match response {
        MockResponse::Json(status, body) => write_json_response(stream, status, body),
        MockResponse::Raw(status, body) => write_raw_response(stream, status, body),
    }
}

fn write_json_response(stream: &mut TcpStream, status: u16, body: Value) -> Result<()> {
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

fn write_raw_response(stream: &mut TcpStream, status: u16, body: &'static str) -> Result<()> {
    let response = format!(
        "HTTP/1.1 {status} OK\r\ncontent-type: text/plain\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    stream
        .write_all(response.as_bytes())
        .map_err(|error| SwitchyardError::Other(format!("write response failed: {error}")))
}

fn target(id: &str, model: &str, base_url: &str) -> Result<LlmTarget> {
    let mut target = LlmTarget::new(LlmTargetId::new(id)?, ModelId::new(model)?);
    target.format = BackendFormat::OpenAi;
    target.endpoint.base_url = Some(base_url.to_string());
    target.endpoint.api_key = Some("test-key".to_string());
    Ok(target)
}

fn config(base_url: &str) -> Result<CascadeProfileConfig> {
    Ok(CascadeProfileConfig {
        strong: target("strong", "frontier/model", &format!("{base_url}/strong/v1"))?,
        weak: target("weak", "cheap/model", &format!("{base_url}/weak/v1"))?,
        fallback_target_on_evict: LlmTargetId::new("strong")?,
        picker: CascadePickerMode::CascadeStrongDefault,
        confidence_threshold: 0.7,
        signal_recent_window: 3,
        classifier: Some(CascadeClassifierConfig {
            model: "classifier/model".to_string(),
            api_key: "test-key".to_string(),
            base_url: Some(format!("{base_url}/classifier/v1")),
            timeout_secs: 1.0,
            recent_turn_window: 3,
            max_tokens: 4096,
            system_prompt: None,
        }),
        enable_stats: true,
    })
}

fn profile_input(request: ChatRequest) -> ProfileInput {
    ProfileInput {
        request,
        metadata: RequestMetadata::default(),
    }
}

fn request() -> ProfileInput {
    profile_input(ChatRequest::openai_chat(json!({
        "model": "client/cascade",
        "messages": [{"role": "user", "content": "hello"}],
    })))
}

fn request_with_tool_result(tool_name: &str, content: &str) -> ProfileInput {
    profile_input(ChatRequest::openai_chat(json!({
        "model": "client/cascade",
        "messages": [
            {"role": "user", "content": "work on this"},
            {"role": "assistant", "tool_calls": [{"function": {"name": tool_name}}]},
            {"role": "tool", "tool_call_id": "1", "content": content},
        ],
    })))
}

fn tier_content(tier: &str) -> String {
    json!({"tier": tier}).to_string()
}

#[tokio::test]
async fn critical_severity_overrides_to_strong_without_classifier() -> Result<()> {
    let server = MockOpenAiServer::spawn(None, 200)?;
    let mut config = config(&server.base_url())?;
    config.classifier = None;
    let profile = config.build()?;

    let processed = profile
        .process(request_with_tool_result("Bash", "out of memory"))
        .await?;

    assert_eq!(
        processed.profile_input.request.model(),
        Some("frontier/model")
    );
    assert_eq!(processed.decision.tier, CascadeTier::Strong);
    assert_eq!(processed.decision.source, CascadeDecisionSource::Override);
    assert!(server.requests()?.is_empty());
    Ok(())
}

#[tokio::test]
async fn negative_score_routes_to_weak_without_classifier() -> Result<()> {
    let server = MockOpenAiServer::spawn(None, 200)?;
    let mut config = config(&server.base_url())?;
    config.classifier = None;
    config.confidence_threshold = 0.1;
    let profile = config.build()?;

    let processed = profile
        .process(request_with_tool_result("Write", "ok"))
        .await?;

    assert_eq!(processed.profile_input.request.model(), Some("cheap/model"));
    assert_eq!(processed.decision.tier, CascadeTier::Weak);
    assert_eq!(processed.decision.source, CascadeDecisionSource::Dimensions);
    Ok(())
}

#[tokio::test]
async fn low_confidence_uses_classifier_when_configured() -> Result<()> {
    let server = MockOpenAiServer::spawn(Some(tier_content("weak")), 200)?;
    let profile = config(&server.base_url())?.build()?;

    let processed = profile.process(request()).await?;

    assert_eq!(processed.profile_input.request.model(), Some("cheap/model"));
    assert_eq!(processed.decision.tier, CascadeTier::Weak);
    assert_eq!(
        processed.decision.source,
        CascadeDecisionSource::LlmClassifier
    );
    let requests = server.requests()?;
    assert_eq!(requests.len(), 1);
    let classifier_body = &requests[0].body;
    assert_eq!(classifier_body["max_tokens"], 4096);
    assert!(classifier_body["messages"][0]["content"]
        .as_str()
        .is_some_and(
            |content| content.contains("routing classifier inside an agentic coding cascade")
        ));
    assert_eq!(classifier_body["response_format"]["type"], "json_object");
    Ok(())
}

#[tokio::test]
async fn yaml_overrides_classifier_prompt_and_max_tokens() -> Result<()> {
    let server = MockOpenAiServer::spawn(Some(tier_content("weak")), 200)?;
    let mut config = config(&server.base_url())?;
    let classifier = config
        .classifier
        .as_mut()
        .ok_or_else(|| SwitchyardError::Other("classifier config missing".to_string()))?;
    classifier.max_tokens = 64;
    classifier.system_prompt = Some("Pick a cascade tier.".to_string());
    let profile = config.build()?;

    let processed = profile.process(request()).await?;

    assert_eq!(processed.profile_input.request.model(), Some("cheap/model"));
    let requests = server.requests()?;
    let classifier_body = &requests[0].body;
    assert_eq!(classifier_body["max_tokens"], 64);
    assert_eq!(
        classifier_body["messages"][0]["content"],
        "Pick a cascade tier."
    );
    Ok(())
}

#[test]
fn config_rejects_bad_classifier_options() -> Result<()> {
    let mut zero_tokens = config("http://127.0.0.1:9")?;
    zero_tokens
        .classifier
        .as_mut()
        .ok_or_else(|| SwitchyardError::Other("classifier config missing".to_string()))?
        .max_tokens = 0;
    let error = zero_tokens
        .build()
        .err()
        .ok_or_else(|| SwitchyardError::Other("expected max token validation failure".into()))?;
    assert!(error.to_string().contains("classifier.max_tokens"));

    let mut empty_prompt = config("http://127.0.0.1:9")?;
    empty_prompt
        .classifier
        .as_mut()
        .ok_or_else(|| SwitchyardError::Other("classifier config missing".to_string()))?
        .system_prompt = Some("   ".to_string());
    let error = empty_prompt
        .build()
        .err()
        .ok_or_else(|| SwitchyardError::Other("expected prompt validation failure".into()))?;
    assert!(error.to_string().contains("classifier.system_prompt"));
    Ok(())
}

#[tokio::test]
async fn malformed_classifier_falls_open_to_default() -> Result<()> {
    let server = MockOpenAiServer::spawn(None, 200)?;
    let profile = config(&server.base_url())?.build()?;

    let processed = profile.process(request()).await?;

    // A classifier failure is fail-open: routing falls back to the default tier.
    // Classifier failures are intentionally not counted as metric errors, matching
    // the Python classifier processor's fail-open behavior.
    assert_eq!(
        processed.profile_input.request.model(),
        Some("frontier/model")
    );
    assert_eq!(processed.decision.source, CascadeDecisionSource::FallOpen);
    Ok(())
}

#[tokio::test]
async fn non_json_classifier_falls_open_to_default() -> Result<()> {
    let server = MockOpenAiServer::spawn(Some(NON_JSON_CLASSIFIER_RESPONSE.to_string()), 200)?;
    let profile = config(&server.base_url())?.build()?;

    let processed = profile.process(request()).await?;

    assert_eq!(
        processed.profile_input.request.model(),
        Some("frontier/model")
    );
    assert_eq!(processed.decision.source, CascadeDecisionSource::FallOpen);
    Ok(())
}

#[tokio::test]
async fn run_emits_backend_and_classifier_metrics() -> Result<()> {
    let server = MockOpenAiServer::spawn(Some(tier_content("strong")), 200)?;
    let profile = config(&server.base_url())?.build()?;

    let response = profile.run(request()).await?;

    let routing_metadata = response
        .routing_metadata
        .as_ref()
        .ok_or_else(|| SwitchyardError::Other("routing metadata missing".to_string()))?;
    assert_eq!(
        routing_metadata.selected_model.as_deref(),
        Some("frontier/model")
    );
    assert_eq!(routing_metadata.selected_tier.as_deref(), Some("strong"));
    assert_eq!(routing_metadata.confidence, None);
    assert!(routing_metadata
        .rationale
        .as_deref()
        .is_some_and(|reason| reason.contains("source=llm-classifier")));
    assert_eq!(
        response.body().and_then(|body| body["model"].as_str()),
        Some("frontier/model")
    );
    let requests = server.requests()?;
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].path, "/classifier/v1/chat/completions");
    assert_eq!(requests[1].path, "/strong/v1/chat/completions");
    // The process meter is a singleton; assert the metric surface mentions the
    // routed model, the strong tier, and the classifier role.
    let metrics = otel_metrics::render_prometheus().map_err(SwitchyardError::Other)?;
    assert!(metrics.contains("frontier/model"));
    assert!(metrics.contains("tier=\"strong\""));
    // Classifier token spend is tagged with tier="classifier" so it never aliases
    // routed-tier buckets even on a shared model id.
    assert!(metrics.contains("tier=\"classifier\""));
    Ok(())
}

#[tokio::test]
async fn run_emits_metrics_on_selected_backend_failure() -> Result<()> {
    let server = MockOpenAiServer::spawn(Some(tier_content("strong")), 503)?;
    let profile = config(&server.base_url())?.build()?;

    let error =
        profile.run(request()).await.err().ok_or_else(|| {
            SwitchyardError::Other("expected selected backend failure".to_string())
        })?;

    assert!(error.to_string().contains("selected backend failed"));
    let metrics = otel_metrics::render_prometheus().map_err(SwitchyardError::Other)?;
    assert!(metrics.contains("switchyard_errors_total"));
    Ok(())
}
