// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the components-v2 latency-service profile internals.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use async_trait::async_trait;
use futures_core::Stream;
use parking_lot::RwLock;
use serde_json::{json, Value};
use switchyard_core::{BackendFormat, ChatRequest, LlmTargetId, ModelId, StreamEvent};

use super::polling::HealthPoller;
use super::selection::select_target;
use super::*;
use crate::backend::ProfileBackend;
use crate::{ProfileInput, RequestMetadata};

#[derive(Clone, Debug, PartialEq)]
struct ObservedCall {
    backend: String,
    body: Value,
}

struct TestBackend {
    name: String,
    calls: Arc<Mutex<Vec<ObservedCall>>>,
    error: Option<&'static str>,
    response_mode: ResponseMode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResponseMode {
    Completion,
    OpenAiStream,
}

struct OneEventStream {
    event: Option<Result<StreamEvent>>,
}

impl Stream for OneEventStream {
    type Item = Result<StreamEvent>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Ready(self.event.take())
    }
}

#[async_trait]
impl ProfileBackend for TestBackend {
    async fn call(&self, request: &ChatRequest) -> Result<ChatResponse> {
        self.calls
            .lock()
            .map_err(|_| SwitchyardError::Other("calls mutex poisoned".to_string()))?
            .push(ObservedCall {
                backend: self.name.clone(),
                body: request.body().clone(),
            });
        if let Some(error) = self.error {
            return Err(SwitchyardError::Backend(error.to_string()));
        }
        match self.response_mode {
            ResponseMode::Completion => Ok(ChatResponse::openai_completion(json!({
                "backend": self.name,
                "model": request.model(),
                "usage": {
                    "prompt_tokens": 13,
                    "completion_tokens": 5,
                },
            }))),
            ResponseMode::OpenAiStream => {
                Ok(ChatResponse::OpenAiStream(Box::pin(OneEventStream {
                    event: Some(Ok(StreamEvent::Json(json!({
                        "backend": self.name,
                        "model": request.model(),
                    })))),
                })))
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct CapturedRequest {
    path: String,
}

struct OneShotServer {
    base_url: String,
    captured: Arc<Mutex<Option<CapturedRequest>>>,
    handle: Option<JoinHandle<Result<()>>>,
}

impl OneShotServer {
    fn json(status: u16, body: Value) -> Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")
            .map_err(|error| SwitchyardError::Other(format!("bind failed: {error}")))?;
        let address = listener
            .local_addr()
            .map_err(|error| SwitchyardError::Other(format!("local addr failed: {error}")))?;
        let captured = Arc::new(Mutex::new(None));
        let captured_for_thread = Arc::clone(&captured);
        let handle = thread::spawn(move || serve_once(listener, status, body, captured_for_thread));
        Ok(Self {
            base_url: format!("http://{address}"),
            captured,
            handle: Some(handle),
        })
    }

    fn base_url(&self) -> String {
        self.base_url.clone()
    }

    fn captured(&mut self) -> Result<CapturedRequest> {
        if let Some(handle) = self.handle.take() {
            handle
                .join()
                .map_err(|_| SwitchyardError::Other("test server thread panicked".into()))??;
        }
        self.captured
            .lock()
            .map_err(|_| SwitchyardError::Other("captured mutex poisoned".into()))?
            .clone()
            .ok_or_else(|| SwitchyardError::Other("test server captured no request".into()))
    }
}

fn serve_once(
    listener: TcpListener,
    status: u16,
    body: Value,
    captured: Arc<Mutex<Option<CapturedRequest>>>,
) -> Result<()> {
    let (mut stream, _) = listener
        .accept()
        .map_err(|error| SwitchyardError::Other(format!("accept failed: {error}")))?;
    let mut reader = BufReader::new(
        stream
            .try_clone()
            .map_err(|error| SwitchyardError::Other(format!("clone stream failed: {error}")))?,
    );
    let mut first_line = String::new();
    reader
        .read_line(&mut first_line)
        .map_err(|error| SwitchyardError::Other(format!("read request line failed: {error}")))?;
    let path = first_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| SwitchyardError::Other("missing request path".into()))?
        .to_string();
    {
        let mut captured = captured
            .lock()
            .map_err(|_| SwitchyardError::Other("captured mutex poisoned".into()))?;
        *captured = Some(CapturedRequest { path });
    }

    loop {
        let mut line = String::new();
        let bytes = reader.read_line(&mut line).map_err(|error| {
            SwitchyardError::Other(format!("read request header failed: {error}"))
        })?;
        if bytes == 0 || line == "\r\n" {
            break;
        }
    }

    let body = serde_json::to_string(&body)
        .map_err(|error| SwitchyardError::Other(format!("json encode failed: {error}")))?;
    let reason = if status == 200 { "OK" } else { "ERROR" };
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    stream
        .write_all(response.as_bytes())
        .map_err(|error| SwitchyardError::Other(format!("write response failed: {error}")))?;
    Ok(())
}

fn target(id: &str, model: &str) -> Result<LlmTarget> {
    let mut target = LlmTarget::new(LlmTargetId::new(id)?, ModelId::new(model)?);
    target.format = BackendFormat::OpenAi;
    Ok(target)
}

fn profile_with_backends(
    targets: Vec<LlmTarget>,
    failures: BTreeMap<&'static str, &'static str>,
    max_retries: usize,
) -> Result<(LatencyServiceProfile, Arc<Mutex<Vec<ObservedCall>>>)> {
    profile_with_backend_modes(targets, failures, BTreeMap::new(), max_retries)
}

fn profile_with_backend_modes(
    targets: Vec<LlmTarget>,
    failures: BTreeMap<&'static str, &'static str>,
    response_modes: BTreeMap<&'static str, ResponseMode>,
    max_retries: usize,
) -> Result<(LatencyServiceProfile, Arc<Mutex<Vec<ObservedCall>>>)> {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut backends = BTreeMap::new();
    let mut health = BTreeMap::new();
    for target in &targets {
        health.insert(
            target.id.clone(),
            EndpointHealth::new(EndpointHealthStatus::Unknown),
        );
        let backend = TestBackend {
            name: format!("{}-backend", target.id),
            calls: Arc::clone(&calls),
            error: failures.get(target.id.as_str()).copied(),
            response_mode: response_modes
                .get(target.id.as_str())
                .copied()
                .unwrap_or(ResponseMode::Completion),
        };
        backends.insert(
            target.id.clone(),
            TargetBackend::new(target.clone(), Arc::new(backend)),
        );
    }

    let target_ids = targets.iter().map(|target| target.id.clone()).collect();
    let profile = LatencyServiceProfile {
        poller: HealthPoller::new("http://latency.test", target_ids, Duration::from_secs(5))?,
        backends,
        health: RwLock::new(health),
        poll_count: AtomicU64::new(0),
        initial_refresh_in_flight: AtomicBool::new(false),
        max_retries,
    };
    Ok((profile, calls))
}

fn observed(calls: &Arc<Mutex<Vec<ObservedCall>>>) -> Result<Vec<ObservedCall>> {
    calls
        .lock()
        .map(|calls| calls.clone())
        .map_err(|_| SwitchyardError::Other("calls mutex poisoned".to_string()))
}

fn profile_input(request: ChatRequest) -> ProfileInput {
    ProfileInput {
        request,
        metadata: RequestMetadata::default(),
    }
}

#[test]
fn profile_config_build_rejects_invalid_config() -> Result<()> {
    let fast = target("fast", "upstream-fast")?;
    let duplicate = LatencyServiceProfileConfig {
        latency_service_url: "http://latency.test".to_string(),
        targets: vec![fast.clone(), fast],
        poll_timeout_secs: 5.0,
        max_retries: 2,
    };
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

    let empty_url = LatencyServiceProfileConfig {
        latency_service_url: " ".to_string(),
        targets: vec![target("fast", "upstream-fast")?],
        poll_timeout_secs: 5.0,
        max_retries: 2,
    };
    match empty_url.build() {
        Err(SwitchyardError::InvalidConfig(message)) => {
            assert!(message.contains("requires latency_service_url"));
        }
        Ok(_) => {
            return Err(SwitchyardError::Other(
                "empty URL should reject profile construction".into(),
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

#[test]
fn profile_config_macro_adds_type_metadata_and_strict_serde() -> Result<()> {
    let config = LatencyServiceProfileConfig {
        latency_service_url: "http://latency.test".to_string(),
        targets: vec![target("fast", "upstream-fast")?],
        poll_timeout_secs: 5.0,
        max_retries: 2,
    };
    assert_eq!(LatencyServiceProfileConfig::PROFILE_TYPE, "latency-service");
    assert_eq!(config.profile_type(), "latency-service");

    let unknown_field = json!({
        "latency_service_url": "http://latency.test",
        "targets": config.targets,
        "poll_timeout_secs": 5.0,
        "max_retries": 2,
        "poll_interval_secs": 1.0,
    });
    let error = serde_json::from_value::<LatencyServiceProfileConfig>(unknown_field)
        .err()
        .ok_or_else(|| SwitchyardError::Other("unknown profile field should fail".into()))?;
    assert!(error.to_string().contains("unknown field"));
    Ok(())
}

#[tokio::test]
async fn healthy_endpoint_serves_request_and_records_stats() -> Result<()> {
    let (profile, calls) = profile_with_backends(
        vec![
            target("fast", "upstream-fast")?,
            target("slow", "upstream-slow")?,
        ],
        BTreeMap::new(),
        0,
    )?;
    profile.update_health(
        LlmTargetId::from_static("fast"),
        EndpointHealth::with_latency(EndpointHealthStatus::Healthy, 50.0),
    )?;
    profile.update_health(
        LlmTargetId::from_static("slow"),
        EndpointHealth::with_latency(EndpointHealthStatus::Degraded, 10.0),
    )?;

    let response = profile
        .run(profile_input(ChatRequest::openai_chat(json!({
            "model": "incoming-model",
            "messages": [{"role": "user", "content": "hi"}],
        }))))
        .await?;

    let routing_metadata = response
        .routing_metadata
        .as_ref()
        .ok_or_else(|| SwitchyardError::Other("routing metadata missing".into()))?;
    assert_eq!(
        routing_metadata.selected_model.as_deref(),
        Some("upstream-fast")
    );
    assert_eq!(routing_metadata.selected_tier.as_deref(), Some("healthy"));
    assert_eq!(
        routing_metadata.router_version.as_deref(),
        Some("latency-service:v1")
    );
    let response = response.response;

    let calls = observed(&calls)?;
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].backend, "fast-backend");
    assert_eq!(calls[0].body["model"], "upstream-fast");
    match response {
        ChatResponse::OpenAiCompletion(body) => {
            assert_eq!(body.body()["backend"], "fast-backend");
        }
        _ => return Err(SwitchyardError::Other("unexpected response shape".into())),
    }

    // The process meter is a singleton; assert the metric surface mentions the
    // selected model rather than exact counts another test may have moved.
    let metrics =
        switchyard_components::otel_metrics::render_prometheus().map_err(SwitchyardError::Other)?;
    assert!(
        metrics.contains("upstream-fast"),
        "expected metrics for the selected model in:\n{metrics}"
    );
    assert!(metrics.contains("router=\"latency_service\""));
    Ok(())
}

#[tokio::test]
async fn retry_avoids_failed_target_while_alternative_exists() -> Result<()> {
    let (profile, calls) = profile_with_backends(
        vec![
            target("failing", "upstream-failing")?,
            target("fallback", "upstream-fallback")?,
        ],
        BTreeMap::from([("failing", "down")]),
        1,
    )?;
    profile.update_health(
        LlmTargetId::from_static("failing"),
        EndpointHealth::new(EndpointHealthStatus::Healthy),
    )?;
    profile.update_health(
        LlmTargetId::from_static("fallback"),
        EndpointHealth::new(EndpointHealthStatus::Degraded),
    )?;

    let response = profile
        .run(profile_input(ChatRequest::openai_chat(json!({
            "model": "incoming-model",
            "messages": [],
        }))))
        .await?;

    let routing_metadata = response
        .routing_metadata
        .as_ref()
        .ok_or_else(|| SwitchyardError::Other("routing metadata missing".into()))?;
    assert_eq!(
        routing_metadata.selected_model.as_deref(),
        Some("upstream-fallback")
    );
    assert_eq!(routing_metadata.selected_tier.as_deref(), Some("degraded"));
    let response = response.response;

    let calls = observed(&calls)?;
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].backend, "failing-backend");
    assert_eq!(calls[1].backend, "fallback-backend");
    match response {
        ChatResponse::OpenAiCompletion(body) => {
            assert_eq!(body.body()["backend"], "fallback-backend");
        }
        _ => return Err(SwitchyardError::Other("unexpected response shape".into())),
    }
    Ok(())
}

#[tokio::test]
async fn exhausted_retries_return_last_target_error() -> Result<()> {
    let (profile, calls) = profile_with_backends(
        vec![
            target("first", "upstream-first")?,
            target("second", "upstream-second")?,
        ],
        BTreeMap::from([("first", "first down"), ("second", "second down")]),
        5,
    )?;
    profile.update_health(
        LlmTargetId::from_static("first"),
        EndpointHealth::new(EndpointHealthStatus::Healthy),
    )?;
    profile.update_health(
        LlmTargetId::from_static("second"),
        EndpointHealth::new(EndpointHealthStatus::Degraded),
    )?;

    let error = match profile
        .run(profile_input(ChatRequest::openai_chat(json!({
            "model": "incoming-model",
            "messages": [],
        }))))
        .await
    {
        Ok(_) => {
            return Err(SwitchyardError::Other(
                "all failing targets should return an error".into(),
            ));
        }
        Err(error) => error,
    };

    assert!(
        error.to_string().contains("second down"),
        "last target error should be returned, got {error:?}"
    );
    let calls = observed(&calls)?;
    assert_eq!(
        calls
            .iter()
            .map(|call| call.backend.as_str())
            .collect::<Vec<_>>(),
        vec!["first-backend", "second-backend"]
    );
    Ok(())
}

#[tokio::test]
async fn process_only_rewrites_request_and_does_not_call_backend() -> Result<()> {
    let (profile, calls) = profile_with_backends(
        vec![
            target("fast", "upstream-fast")?,
            target("slow", "upstream-slow")?,
        ],
        BTreeMap::new(),
        0,
    )?;
    profile.update_health(
        LlmTargetId::from_static("fast"),
        EndpointHealth::new(EndpointHealthStatus::Healthy),
    )?;

    let request = profile
        .process(profile_input(ChatRequest::openai_chat(json!({
            "model": "incoming-model",
            "messages": [],
        }))))
        .await?;

    assert_eq!(request.profile_input.request.model(), Some("upstream-fast"));
    assert_eq!(request.selected.target_id, LlmTargetId::from_static("fast"));
    assert!(observed(&calls)?.is_empty());
    Ok(())
}

#[tokio::test]
async fn streaming_response_is_passed_through_from_selected_target() -> Result<()> {
    let (profile, calls) = profile_with_backend_modes(
        vec![target("fast", "upstream-fast")?],
        BTreeMap::new(),
        BTreeMap::from([("fast", ResponseMode::OpenAiStream)]),
        0,
    )?;
    profile.update_health(
        LlmTargetId::from_static("fast"),
        EndpointHealth::new(EndpointHealthStatus::Healthy),
    )?;

    let response = profile
        .run(profile_input(ChatRequest::openai_chat(json!({
            "model": "incoming-model",
            "messages": [],
        }))))
        .await?;

    let routing_metadata = response
        .routing_metadata
        .as_ref()
        .ok_or_else(|| SwitchyardError::Other("routing metadata missing".into()))?;
    assert_eq!(
        routing_metadata.selected_model.as_deref(),
        Some("upstream-fast")
    );
    assert_eq!(routing_metadata.selected_tier.as_deref(), Some("healthy"));
    let response = response.response;

    match response {
        ChatResponse::OpenAiStream(_) => {}
        _ => {
            return Err(SwitchyardError::Other(
                "expected OpenAI stream response".into(),
            ))
        }
    }
    let calls = observed(&calls)?;
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].backend, "fast-backend");
    assert_eq!(calls[0].body["model"], "upstream-fast");
    Ok(())
}

#[test]
fn unknown_target_health_update_is_rejected() -> Result<()> {
    let (profile, _calls) =
        profile_with_backends(vec![target("known", "upstream-known")?], BTreeMap::new(), 0)?;

    let error = match profile.update_health(
        LlmTargetId::from_static("missing"),
        EndpointHealth::new(EndpointHealthStatus::Healthy),
    ) {
        Ok(_) => {
            return Err(SwitchyardError::Other(
                "unknown target health update should fail".into(),
            ));
        }
        Err(error) => error,
    };

    assert!(error
        .to_string()
        .contains("target missing is not configured"));
    Ok(())
}

#[tokio::test]
async fn poll_once_updates_cache_and_ignores_unknown_service_ids() -> Result<()> {
    let mut server = OneShotServer::json(
        200,
        json!({
            "endpoint_health": {
                "fast": {"status": "healthy", "last_latency_ms": 42.0},
                "unknown-service-id": {"status": "degraded", "last_latency_ms": 999.0}
            }
        }),
    )?;
    let config = LatencyServiceProfileConfig {
        latency_service_url: server.base_url(),
        targets: vec![target("fast", "upstream-fast")?],
        poll_timeout_secs: 5.0,
        max_retries: 0,
    };
    let profile = config.build()?;

    profile.poll_once().await?;
    let captured = server.captured()?;

    assert!(captured.path.starts_with("/v1/endpoints/health?"));
    assert!(captured.path.contains("endpoint_ids=fast"));
    assert_eq!(
        profile
            .health_snapshot()
            .get(&LlmTargetId::from_static("fast")),
        Some(&EndpointHealth::with_latency(
            EndpointHealthStatus::Healthy,
            42.0
        ))
    );
    assert!(profile.is_ready());
    Ok(())
}

#[tokio::test]
async fn poll_once_resets_omitted_known_targets_to_unknown() -> Result<()> {
    let mut server = OneShotServer::json(
        200,
        json!({
            "endpoint_health": {
                "fast": {"status": "healthy", "last_latency_ms": 42.0}
            }
        }),
    )?;
    let config = LatencyServiceProfileConfig {
        latency_service_url: server.base_url(),
        targets: vec![
            target("fast", "upstream-fast")?,
            target("slow", "upstream-slow")?,
        ],
        poll_timeout_secs: 5.0,
        max_retries: 0,
    };
    let profile = config.build()?;
    profile.update_health(
        LlmTargetId::from_static("slow"),
        EndpointHealth::with_latency(EndpointHealthStatus::Healthy, 7.0),
    )?;

    profile.poll_once().await?;
    let _captured = server.captured()?;

    assert_eq!(
        profile
            .health_snapshot()
            .get(&LlmTargetId::from_static("fast")),
        Some(&EndpointHealth::with_latency(
            EndpointHealthStatus::Healthy,
            42.0
        ))
    );
    assert_eq!(
        profile
            .health_snapshot()
            .get(&LlmTargetId::from_static("slow")),
        Some(&EndpointHealth::new(EndpointHealthStatus::Unknown))
    );
    Ok(())
}

#[tokio::test]
async fn poll_once_tolerates_additive_health_payload_fields() -> Result<()> {
    let mut server = OneShotServer::json(
        200,
        json!({
            "endpoint_health": {
                "fast": {
                    "status": "healthy",
                    "last_latency_ms": 42.0,
                    "reason": "warm"
                }
            },
            "generated_at": "2026-05-19T00:00:00Z"
        }),
    )?;
    let config = LatencyServiceProfileConfig {
        latency_service_url: server.base_url(),
        targets: vec![target("fast", "upstream-fast")?],
        poll_timeout_secs: 5.0,
        max_retries: 0,
    };
    let profile = config.build()?;

    profile.poll_once().await?;
    let _captured = server.captured()?;

    assert_eq!(
        profile
            .health_snapshot()
            .get(&LlmTargetId::from_static("fast")),
        Some(&EndpointHealth::with_latency(
            EndpointHealthStatus::Healthy,
            42.0
        ))
    );
    Ok(())
}

#[tokio::test]
async fn poll_failure_resets_stale_health_to_unknown_without_marking_ready() -> Result<()> {
    let mut server = OneShotServer::json(500, json!({"error": "down"}))?;
    let config = LatencyServiceProfileConfig {
        latency_service_url: server.base_url(),
        targets: vec![target("fast", "upstream-fast")?],
        poll_timeout_secs: 5.0,
        max_retries: 0,
    };
    let profile = config.build()?;
    profile.update_health(
        LlmTargetId::from_static("fast"),
        EndpointHealth::with_latency(EndpointHealthStatus::Healthy, 42.0),
    )?;

    let error = match profile.poll_once().await {
        Ok(_) => return Err(SwitchyardError::Other("poll should fail".into())),
        Err(error) => error,
    };

    assert!(error.to_string().contains("HTTP 500"));
    let _captured = server.captured()?;
    assert_eq!(
        profile
            .health_snapshot()
            .get(&LlmTargetId::from_static("fast")),
        Some(&EndpointHealth::new(EndpointHealthStatus::Unknown))
    );
    assert!(!profile.is_ready());
    Ok(())
}

#[tokio::test]
async fn malformed_health_status_resets_to_unknown() -> Result<()> {
    let mut server = OneShotServer::json(
        200,
        json!({
            "endpoint_health": {
                "fast": {"status": "on_fire", "last_latency_ms": 42.0}
            }
        }),
    )?;
    let config = LatencyServiceProfileConfig {
        latency_service_url: server.base_url(),
        targets: vec![target("fast", "upstream-fast")?],
        poll_timeout_secs: 5.0,
        max_retries: 0,
    };
    let profile = config.build()?;
    profile.update_health(
        LlmTargetId::from_static("fast"),
        EndpointHealth::with_latency(EndpointHealthStatus::Healthy, 42.0),
    )?;

    let error = match profile.poll_once().await {
        Ok(_) => {
            return Err(SwitchyardError::Other(
                "malformed health status should fail".into(),
            ));
        }
        Err(error) => error,
    };

    assert!(
        error.to_string().contains("invalid JSON"),
        "malformed status should surface as invalid response, got {error:?}"
    );
    let _captured = server.captured()?;
    assert_eq!(
        profile
            .health_snapshot()
            .get(&LlmTargetId::from_static("fast")),
        Some(&EndpointHealth::new(EndpointHealthStatus::Unknown))
    );
    Ok(())
}

#[test]
fn faster_endpoint_gets_more_traffic_within_same_tier() -> Result<()> {
    let snapshot = BTreeMap::from([
        (
            LlmTargetId::from_static("fast"),
            EndpointHealth::with_latency(EndpointHealthStatus::Healthy, 50.0),
        ),
        (
            LlmTargetId::from_static("slow"),
            EndpointHealth::with_latency(EndpointHealthStatus::Healthy, 500.0),
        ),
    ]);

    let mut fast = 0;
    let mut slow = 0;
    for _ in 0..2_000 {
        match select_target(&snapshot, &BTreeSet::new())?
            .target_id
            .as_str()
        {
            "fast" => fast += 1,
            "slow" => slow += 1,
            other => {
                return Err(SwitchyardError::Other(format!(
                    "unexpected target selected: {other}"
                )));
            }
        }
    }

    assert!(
        fast > slow * 5,
        "expected fast endpoint to dominate inverse-latency routing; fast={fast}, slow={slow}"
    );
    Ok(())
}

#[test]
fn missing_latency_falls_back_to_uniform_random() -> Result<()> {
    let snapshot = BTreeMap::from([
        (
            LlmTargetId::from_static("with-latency"),
            EndpointHealth::with_latency(EndpointHealthStatus::Healthy, 50.0),
        ),
        (
            LlmTargetId::from_static("without-latency"),
            EndpointHealth::new(EndpointHealthStatus::Healthy),
        ),
    ]);

    let mut first = 0;
    let mut second = 0;
    for _ in 0..2_000 {
        match select_target(&snapshot, &BTreeSet::new())?
            .target_id
            .as_str()
        {
            "with-latency" => first += 1,
            "without-latency" => second += 1,
            other => {
                return Err(SwitchyardError::Other(format!(
                    "unexpected target selected: {other}"
                )));
            }
        }
    }

    assert!(
        (850..=1150).contains(&first),
        "uniform fallback should stay near 50/50; first={first}, second={second}"
    );
    assert!(
        (850..=1150).contains(&second),
        "uniform fallback should stay near 50/50; first={first}, second={second}"
    );
    Ok(())
}

#[test]
fn non_positive_latency_selects_without_error_or_dividing_by_zero() -> Result<()> {
    let snapshot = BTreeMap::from([
        (
            LlmTargetId::from_static("zero"),
            EndpointHealth::with_latency(EndpointHealthStatus::Healthy, 0.0),
        ),
        (
            LlmTargetId::from_static("positive"),
            EndpointHealth::with_latency(EndpointHealthStatus::Healthy, 100.0),
        ),
    ]);

    for _ in 0..100 {
        let selected = select_target(&snapshot, &BTreeSet::new())?;
        assert!(
            selected.target_id == LlmTargetId::from_static("zero")
                || selected.target_id == LlmTargetId::from_static("positive")
        );
    }
    Ok(())
}
