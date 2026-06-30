// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public API tests for the components-v2 LLM-routing profile config.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use serde_json::{json, Value};
use switchyard_components::otel_metrics;
use switchyard_components_v2::{
    LlmRoutingProfileConfig, LlmRoutingTierMapping, Profile, ProfileConfig, ProfileHooks,
    ProfileInput, RequestMetadata,
};
use switchyard_core::{
    BackendFormat, ChatRequest, LlmTarget, LlmTargetId, ModelId, Result, SwitchyardError,
};

#[derive(Clone, Debug, PartialEq)]
struct ObservedRequest {
    path: String,
    headers: BTreeMap<String, String>,
    body: Value,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum BackendMode {
    FixedStatus(u16),
    WeakContextOverflowThenOk,
}

struct MockOpenAiServer {
    addr: SocketAddr,
    requests: Arc<Mutex<Vec<ObservedRequest>>>,
    shutdown: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl MockOpenAiServer {
    fn spawn(classifier_arguments: Option<Value>, backend_status: u16) -> Result<Self> {
        Self::spawn_with_backend_mode(
            classifier_arguments,
            BackendMode::FixedStatus(backend_status),
        )
    }

    fn spawn_with_backend_mode(
        classifier_arguments: Option<Value>,
        backend_mode: BackendMode,
    ) -> Result<Self> {
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
                let response = response_for(&request, classifier_arguments.as_ref(), backend_mode);
                let _ = write_json_response(&mut stream, response.0, response.1);
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
    let headers = headers
        .lines()
        .skip(1)
        .filter_map(|line| {
            let (name, value) = line.split_once(':')?;
            Some((name.to_ascii_lowercase(), value.trim().to_string()))
        })
        .collect::<BTreeMap<_, _>>();
    let content_length = headers
        .get("content-length")
        .and_then(|value| value.parse::<usize>().ok())
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
    let body = if content_length == 0 {
        Value::Null
    } else {
        serde_json::from_slice(&bytes[body_start..body_start + content_length])
            .map_err(|error| SwitchyardError::Other(format!("decode request body: {error}")))?
    };
    Ok(ObservedRequest {
        path,
        headers,
        body,
    })
}

fn find_header_end(bytes: &[u8]) -> Option<usize> {
    bytes.windows(4).position(|window| window == b"\r\n\r\n")
}

fn response_for(
    request: &ObservedRequest,
    classifier_arguments: Option<&Value>,
    backend_mode: BackendMode,
) -> (u16, Value) {
    if request.path.contains("/classifier/") {
        return match classifier_arguments {
            Some(arguments) => {
                let tool_name = request.body["tool_choice"]["function"]["name"]
                    .as_str()
                    .unwrap_or("select_route");
                (
                    200,
                    json!({
                        "id": "chatcmpl-llm-routing-classifier",
                        "object": "chat.completion",
                        "model": request.body["model"],
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "tool_calls": [{
                                    "id": "call_route",
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": arguments.to_string(),
                                    },
                                }],
                            },
                            "finish_reason": "tool_calls",
                        }],
                        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
                    }),
                )
            }
            None => (
                200,
                json!({
                    "id": "chatcmpl-llm-routing-classifier",
                    "object": "chat.completion",
                    "model": request.body["model"],
                    "choices": [{"message": {"role": "assistant", "content": "not a tool call"}}],
                }),
            ),
        };
    }

    if matches!(backend_mode, BackendMode::WeakContextOverflowThenOk)
        && request.path.contains("/weak/")
    {
        return (
            400,
            json!({
                "error": {
                    "message": "prompt is too long",
                    "code": "context_length_exceeded",
                    "type": "invalid_request_error"
                }
            }),
        );
    }

    if let BackendMode::FixedStatus(backend_status) = backend_mode {
        if backend_status >= 400 {
            return (
                backend_status,
                json!({"error": {"message": "selected backend failed", "code": "backend_failed"}}),
            );
        }
    }
    (
        200,
        json!({
            "id": "chatcmpl-llm-routing-backend",
            "object": "chat.completion",
            "model": request.body["model"],
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }),
    )
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

fn target(id: &str, model: &str, base_url: &str) -> Result<LlmTarget> {
    let mut target = LlmTarget::new(LlmTargetId::new(id)?, ModelId::new(model)?);
    target.format = BackendFormat::OpenAi;
    target.endpoint.base_url = Some(base_url.to_string());
    target.endpoint.api_key = Some("test-key".to_string());
    Ok(target)
}

fn config(base_url: &str) -> Result<LlmRoutingProfileConfig> {
    Ok(LlmRoutingProfileConfig {
        strong: target("strong", "frontier/model", &format!("{base_url}/strong/v1"))?,
        weak: target("weak", "cheap/model", &format!("{base_url}/weak/v1"))?,
        classifier: target(
            "classifier",
            "classifier/model",
            &format!("{base_url}/classifier/v1"),
        )?,
        fallback_target_on_evict: LlmTargetId::new("strong")?,
        profile_name: "coding_agent".to_string(),
        classifier_min_confidence: 0.0,
        classifier_fail_open: true,
        classifier_recent_turn_window: 4,
        classifier_max_tokens: 4096,
        alignment_min_confidence: 0.85,
        default_tier: None,
        tier_mapping: None,
        classifier_system_prompt: None,
        classifier_tool_name: None,
        classifier_tool_description: None,
        classifier_tool_parameters: None,
    })
}

fn request() -> ProfileInput {
    ProfileInput {
        request: ChatRequest::openai_chat(json!({
            "model": "client/model",
            "messages": [{"role": "user", "content": "list files"}],
        })),
        metadata: RequestMetadata::default(),
    }
}

fn medium_signals() -> Value {
    json!({
        "recommended_tier": "medium",
        "confidence": 0.9,
        "abstain": false,
        "turn_type": "exploration",
        "code_modification_scope": "none",
        "tool_call_count_estimate": 0,
        "requires_codebase_context": false
    })
}

fn complex_signals() -> Value {
    json!({
        "recommended_tier": "complex",
        "confidence": 0.9,
        "abstain": false,
        "turn_type": "debug",
        "code_modification_scope": "cross_module",
        "tool_call_count_estimate": 4,
        "requires_codebase_context": true
    })
}

#[test]
fn profile_config_macro_adds_type_metadata_and_strict_serde() -> Result<()> {
    let server = MockOpenAiServer::spawn(Some(medium_signals()), 200)?;
    let config = config(&server.base_url())?;

    assert_eq!(LlmRoutingProfileConfig::PROFILE_TYPE, "llm-routing");
    assert_eq!(config.profile_type(), "llm-routing");

    let old_stats_toggle = json!({
        "strong": config.strong,
        "weak": config.weak,
        "classifier": config.classifier,
        "stats": false,
    });
    let error = serde_json::from_value::<LlmRoutingProfileConfig>(old_stats_toggle)
        .err()
        .ok_or_else(|| SwitchyardError::Other("unknown profile field should fail".into()))?;
    assert!(error.to_string().contains("unknown field"));
    Ok(())
}

#[tokio::test]
async fn process_routes_medium_policy_to_weak_with_required_tool_call() -> Result<()> {
    let server = MockOpenAiServer::spawn(Some(medium_signals()), 200)?;
    let profile = config(&server.base_url())?.build()?;

    let processed = profile.process(request()).await?;

    assert_eq!(processed.profile_input.request.model(), Some("cheap/model"));
    assert_eq!(processed.decision.tier, "weak");
    assert_eq!(processed.decision.source, "policy_tier");
    let requests = server.requests()?;
    assert_eq!(requests.len(), 1);
    let classifier_body = &requests[0].body;
    assert_eq!(
        classifier_body["tools"][0]["function"]["name"],
        "select_route"
    );
    assert_eq!(
        classifier_body["tool_choice"]["function"]["name"],
        "select_route"
    );
    assert_eq!(classifier_body["tools"][0]["function"]["strict"], true);
    assert!(classifier_body.get("response_format").is_none());
    Ok(())
}

#[tokio::test]
async fn yaml_overrides_classifier_tool_contract_and_tier_mapping() -> Result<()> {
    let server = MockOpenAiServer::spawn(Some(medium_signals()), 200)?;
    let mut config = config(&server.base_url())?;
    config.classifier_max_tokens = 64;
    config.classifier_system_prompt = Some("Use the tool to pick a backend tier.".to_string());
    config.classifier_tool_name = Some("choose_model".to_string());
    config.classifier_tool_description = Some("Choose the target model tier.".to_string());
    config.classifier_tool_parameters = Some(json!({
        "type": "object",
        "properties": {"recommended_tier": {"type": "string"}},
        "required": ["recommended_tier"],
        "additionalProperties": false,
    }));
    config.tier_mapping = Some(LlmRoutingTierMapping {
        simple: "weak".to_string(),
        medium: "strong".to_string(),
        complex: "strong".to_string(),
        reasoning: "strong".to_string(),
    });
    let profile = config.build()?;

    let processed = profile.process(request()).await?;

    assert_eq!(
        processed.profile_input.request.model(),
        Some("frontier/model")
    );
    let requests = server.requests()?;
    let classifier_body = &requests[0].body;
    assert_eq!(classifier_body["max_tokens"], 64);
    assert_eq!(
        classifier_body["messages"][0]["content"],
        "Use the tool to pick a backend tier."
    );
    assert_eq!(
        classifier_body["tools"][0]["function"]["name"],
        "choose_model"
    );
    assert_eq!(
        classifier_body["tools"][0]["function"]["description"],
        "Choose the target model tier."
    );
    assert_eq!(
        classifier_body["tool_choice"]["function"]["name"],
        "choose_model"
    );
    Ok(())
}

#[test]
fn config_rejects_non_strict_classifier_tool_schema() -> Result<()> {
    let server = MockOpenAiServer::spawn(Some(medium_signals()), 200)?;
    let mut config = config(&server.base_url())?;
    config.classifier_tool_parameters = Some(json!({
        "type": "object",
        "properties": {"recommended_tier": {"type": "string"}},
        "required": ["recommended_tier"],
        "additionalProperties": true,
    }));

    let error = config
        .build()
        .err()
        .ok_or_else(|| SwitchyardError::Other("expected strict schema validation".to_string()))?;

    assert!(error.to_string().contains("additionalProperties"));
    assert!(error.to_string().contains("strict tool calls"));
    Ok(())
}

#[tokio::test]
async fn low_confidence_defaults_to_strong() -> Result<()> {
    let mut signals = medium_signals();
    signals["recommended_tier"] = json!("simple");
    signals["confidence"] = json!(0.2);
    let server = MockOpenAiServer::spawn(Some(signals), 200)?;
    let mut config = config(&server.base_url())?;
    config.classifier_min_confidence = 0.8;
    let profile = config.build()?;

    let processed = profile.process(request()).await?;

    assert_eq!(
        processed.profile_input.request.model(),
        Some("frontier/model")
    );
    assert_eq!(processed.decision.source, "low_confidence");
    Ok(())
}

#[tokio::test]
async fn classifier_failure_fails_open_with_visible_decision_source() -> Result<()> {
    let server = MockOpenAiServer::spawn(None, 200)?;
    let profile = config(&server.base_url())?.build()?;

    let processed = profile.process(request()).await?;

    assert_eq!(
        processed.profile_input.request.model(),
        Some("frontier/model")
    );
    // Classifier failures are fail-open with a visible decision source and are
    // intentionally not counted as metric errors, matching the Python path.
    assert_eq!(processed.decision.source, "classifier_error_fall_open");
    Ok(())
}

#[tokio::test]
async fn classifier_failure_fails_closed_when_fail_open_is_disabled() -> Result<()> {
    let server = MockOpenAiServer::spawn(None, 200)?;
    let mut config = config(&server.base_url())?;
    config.classifier_fail_open = false;
    let profile = config.build()?;

    let error = profile
        .process(request())
        .await
        .err()
        .ok_or_else(|| SwitchyardError::Other("expected classifier failure".to_string()))?;

    assert!(error.to_string().contains("LLM classifier failed"));
    Ok(())
}

#[tokio::test]
async fn run_emits_backend_and_classifier_metrics() -> Result<()> {
    let server = MockOpenAiServer::spawn(Some(complex_signals()), 200)?;
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
    assert_eq!(routing_metadata.confidence, Some(0.9));
    assert_eq!(
        routing_metadata.router_version.as_deref(),
        Some("llm-routing:coding_agent:v1")
    );
    assert_eq!(routing_metadata.tolerance, Some(0.0));
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
async fn run_retries_fallback_target_after_context_window_overflow() -> Result<()> {
    let server = MockOpenAiServer::spawn_with_backend_mode(
        Some(medium_signals()),
        BackendMode::WeakContextOverflowThenOk,
    )?;
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
    assert!(routing_metadata
        .rationale
        .as_deref()
        .is_some_and(|reason| reason.contains("context window")));
    assert_eq!(
        response.body().and_then(|body| body["model"].as_str()),
        Some("frontier/model")
    );
    let requests = server.requests()?;
    assert_eq!(
        requests
            .iter()
            .map(|request| request.path.as_str())
            .collect::<Vec<_>>(),
        vec![
            "/classifier/v1/chat/completions",
            "/weak/v1/chat/completions",
            "/strong/v1/chat/completions",
        ]
    );
    assert_eq!(requests[1].body["model"], "cheap/model");
    assert_eq!(requests[2].body["model"], "frontier/model");
    Ok(())
}

#[tokio::test]
async fn run_normalizes_reasoning_effort_and_applies_provider_defaults() -> Result<()> {
    let server = MockOpenAiServer::spawn(Some(medium_signals()), 200)?;
    let mut config = config(&server.base_url())?;
    config.classifier.model = ModelId::new("nvidia/deepseek-ai/deepseek-v4-flash")?;
    config.weak.model = ModelId::new("nvidia/deepseek-ai/evals-deepseek-v4-pro")?;
    let profile = config.build()?;
    let mut input = request();
    input.request = ChatRequest::openai_chat(json!({
        "model": "client/model",
        "messages": [{"role": "user", "content": "list files"}],
        "reasoning_effort": "xhigh",
    }));

    let response = profile.run(input).await?;

    assert_eq!(
        response.body().and_then(|body| body["model"].as_str()),
        Some("nvidia/deepseek-ai/evals-deepseek-v4-pro")
    );
    let requests = server.requests()?;
    let classifier = &requests[0];
    let backend = &requests[1];
    assert_eq!(
        classifier.body["chat_template_kwargs"],
        json!({"enable_thinking": false})
    );
    assert_eq!(
        classifier.headers.get("x-inference-priority"),
        Some(&"batch".to_string())
    );
    assert!(classifier.body["messages"][1]["content"]
        .as_str()
        .is_some_and(|content| content.contains("\"reasoning_effort\":\"high\"")));
    assert_eq!(backend.body["reasoning_effort"], "high");
    assert_eq!(
        backend.body["chat_template_kwargs"],
        json!({"enable_thinking": false})
    );
    assert_eq!(
        backend.headers.get("x-inference-priority"),
        Some(&"batch".to_string())
    );
    Ok(())
}

#[tokio::test]
async fn run_records_selected_backend_failure() -> Result<()> {
    let server = MockOpenAiServer::spawn(Some(complex_signals()), 503)?;
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
