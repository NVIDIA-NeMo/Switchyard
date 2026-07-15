// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end router tests: drive `build_router` with `tower::oneshot` against a
//! `wiremock` upstream, covering same-format and cross-format buffered/streaming
//! paths, error mapping, discovery endpoints, and credential fallback.

use std::collections::{BTreeMap, HashMap, HashSet};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use libsy_llm_client::{Backend, HttpBackendConfig, LlmModelClient};
use libsy_proxy::{build_router, ProxyState};
use serde_json::{json, Value};
use switchyard_translation::WireFormat;
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const UPSTREAM_MODEL: &str = "upstream-model";

fn backend_for(format: WireFormat, base_url: &str) -> Backend {
    let config = HttpBackendConfig {
        base_url: base_url.to_string(),
        api_key: Some("secret".to_string()),
        extra_headers: BTreeMap::new(),
    };
    match format {
        WireFormat::OpenAiChat => Backend::OpenAiChat(config),
        WireFormat::OpenAiResponses => Backend::OpenAiResponses(config),
        WireFormat::AnthropicMessages => Backend::Anthropic(config),
    }
}

// Builds proxy state serving the given upstream formats at base_url.
fn state(base_url: &str, formats: &[WireFormat], fallback: Option<WireFormat>) -> ProxyState {
    let backends: HashMap<WireFormat, Backend> = formats
        .iter()
        .map(|&format| (format, backend_for(format, base_url)))
        .collect();
    let available: HashSet<WireFormat> = backends.keys().copied().collect();
    let map = HashMap::from([(UPSTREAM_MODEL.to_string(), backends)]);
    let client = LlmModelClient::new(map).unwrap();
    ProxyState::new(client, UPSTREAM_MODEL, available, fallback)
}

async fn post(state: ProxyState, uri: &str, body: Value) -> (StatusCode, String) {
    let request = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    let response = build_router(state).oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

fn chat_completion_body() -> Value {
    json!({
        "id": "chatcmpl-1",
        "model": UPSTREAM_MODEL,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hi there"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    })
}

async fn mock_chat(server: &MockServer, template: ResponseTemplate) {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(template)
        .mount(server)
        .await;
}

#[tokio::test]
async fn buffered_same_format_chat() {
    let server = MockServer::start().await;
    mock_chat(
        &server,
        ResponseTemplate::new(200).set_body_json(chat_completion_body()),
    )
    .await;

    let state = state(
        &format!("{}/v1", server.uri()),
        &[WireFormat::OpenAiChat],
        None,
    );
    let (status, body) = post(
        state,
        "/v1/chat/completions",
        json!({"model": "switchyard", "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    let value: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(value["choices"][0]["message"]["content"], "Hi there");
    // The response echoes the requested model, not the upstream id.
    assert_eq!(value["model"], "switchyard");
}

#[tokio::test]
async fn buffered_cross_format_messages_to_chat() {
    // Inbound Anthropic, OpenAI-Chat upstream → response re-encoded to Anthropic.
    let server = MockServer::start().await;
    mock_chat(
        &server,
        ResponseTemplate::new(200).set_body_json(chat_completion_body()),
    )
    .await;

    // OpenAI-only credentials + fallback, so inbound Anthropic routes to the
    // OpenAI-Chat upstream and the response is re-encoded to Anthropic.
    let state = state_with_fallback(&format!("{}/v1", server.uri()));
    let (status, body) = post(
        state,
        "/v1/messages",
        json!({
            "model": "switchyard",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}]
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    let value: Value = serde_json::from_str(&body).unwrap();
    // Anthropic-shaped response, echoing the requested model.
    assert_eq!(value["type"], "message");
    assert_eq!(value["content"][0]["text"], "Hi there");
    assert_eq!(value["model"], "switchyard");
}

fn state_openai_only(base_url: &str) -> ProxyState {
    state(
        base_url,
        &[WireFormat::OpenAiChat, WireFormat::OpenAiResponses],
        None,
    )
}

// OpenAI-only credentials with an OpenAI-chat fallback, so inbound Anthropic
// routes to the OpenAI-chat upstream.
fn state_with_fallback(base_url: &str) -> ProxyState {
    state(
        base_url,
        &[WireFormat::OpenAiChat, WireFormat::OpenAiResponses],
        Some(WireFormat::OpenAiChat),
    )
}

#[tokio::test]
async fn streaming_messages_ends_with_message_stop() {
    let server = MockServer::start().await;
    let sse = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"Hi\"}}]}\n\n\
         data: {\"choices\":[{\"delta\":{\"content\":\" there\"}}]}\n\n\
         data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
         data: [DONE]\n\n";
    mock_chat(
        &server,
        ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"),
    )
    .await;

    let (status, body) = post(
        state_with_fallback(&format!("{}/v1", server.uri())),
        "/v1/messages",
        json!({
            "model": "switchyard",
            "max_tokens": 64,
            "stream": true,
            "messages": [{"role": "user", "content": "hi"}]
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("event: message_start"), "body: {body}");
    assert!(body.contains("event: message_stop"), "body: {body}");
    assert!(
        !body.contains("[DONE]"),
        "anthropic stream must not send [DONE]: {body}"
    );
}

#[tokio::test]
async fn streaming_chat_ends_with_done() {
    let server = MockServer::start().await;
    let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n\
         data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
         data: [DONE]\n\n";
    mock_chat(
        &server,
        ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"),
    )
    .await;

    let (status, body) = post(
        state(
            &format!("{}/v1", server.uri()),
            &[WireFormat::OpenAiChat],
            None,
        ),
        "/v1/chat/completions",
        json!({
            "model": "switchyard",
            "stream": true,
            "messages": [{"role": "user", "content": "hi"}]
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert!(
        body.contains("[DONE]"),
        "openai stream must send [DONE]: {body}"
    );
}

#[tokio::test]
async fn upstream_500_passes_status_through() {
    let server = MockServer::start().await;
    mock_chat(&server, ResponseTemplate::new(500).set_body_string("boom")).await;

    let (status, body) = post(
        state(
            &format!("{}/v1", server.uri()),
            &[WireFormat::OpenAiChat],
            None,
        ),
        "/v1/chat/completions",
        json!({"model": "switchyard", "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;

    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert!(body.contains("boom"), "body: {body}");
}

#[tokio::test]
async fn no_backend_for_inbound_without_fallback_is_400() {
    let server = MockServer::start().await;
    // Only OpenAI available, no fallback; inbound Anthropic has nowhere to go.
    let (status, body) = post(
        state_openai_only(&format!("{}/v1", server.uri())),
        "/v1/messages",
        json!({"model": "switchyard", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body.contains("no upstream backend"), "body: {body}");
}

#[tokio::test]
async fn models_lists_served_model() {
    let server = MockServer::start().await;
    let router = build_router(state(
        &format!("{}/v1", server.uri()),
        &[WireFormat::OpenAiChat],
        None,
    ));
    let response = router
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["data"][0]["id"], "switchyard");
}

#[tokio::test]
async fn health_ok() {
    let server = MockServer::start().await;
    let router = build_router(state(
        &format!("{}/v1", server.uri()),
        &[WireFormat::OpenAiChat],
        None,
    ));
    let response = router
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}
