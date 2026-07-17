// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end router tests: drive `build_router` with `tower::oneshot` against a
//! `wiremock` upstream, covering the buffered and streaming paths per wire format,
//! cross-format routing, error mapping, and discovery endpoints.
//!
//! Each request is decoded to the neutral IR, routed by the `RandomAlgo` to a
//! tier, translated to that tier's upstream format for the call, and translated
//! back to the inbound format for the client.

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use libsy::{Algorithm, LlmTarget, LlmTargetSet, RandomAlgo, RoutedLlmClient};
use libsy_server::{build_router, ProxyState};
use serde_json::{json, Value};
use switchyard_llm_client::{Backend, HttpBackendConfig, ModelConfig, TranslatingLlmClient};
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

// Builds server state whose single random-routing target serves `UPSTREAM_MODEL`
// over an upstream speaking `upstream_format` at `base_url`. Inbound requests are
// translated to `upstream_format` for the upstream call; when it matches the
// inbound format this is a same-format round trip.
fn state(base_url: &str, upstream_format: WireFormat) -> ProxyState {
    let config = ModelConfig::new(UPSTREAM_MODEL, backend_for(upstream_format, base_url), None);
    let client = Arc::new(TranslatingLlmClient::new(&[config]).unwrap());
    let targets = LlmTargetSet::new(vec![LlmTarget {
        semantic_name: UPSTREAM_MODEL.to_string(),
        llm_client: Some(client as Arc<dyn RoutedLlmClient>),
    }]);
    let algorithm: Arc<dyn Algorithm> = Arc::new(RandomAlgo::new(targets));
    ProxyState::new(algorithm, false)
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

async fn mock(server: &MockServer, endpoint: &str, template: ResponseTemplate) {
    Mock::given(method("POST"))
        .and(path(endpoint))
        .respond_with(template)
        .mount(server)
        .await;
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

fn anthropic_message_body() -> Value {
    json!({
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": UPSTREAM_MODEL,
        "content": [{"type": "text", "text": "Hi there"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 2}
    })
}

#[tokio::test]
async fn buffered_chat_round_trips() {
    let server = MockServer::start().await;
    mock(
        &server,
        "/v1/chat/completions",
        ResponseTemplate::new(200).set_body_json(chat_completion_body()),
    )
    .await;

    let (status, body) = post(
        state(&format!("{}/v1", server.uri()), WireFormat::OpenAiChat),
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
async fn buffered_messages_round_trips() {
    let server = MockServer::start().await;
    mock(
        &server,
        "/v1/messages",
        ResponseTemplate::new(200).set_body_json(anthropic_message_body()),
    )
    .await;

    let (status, body) = post(
        state(
            &format!("{}/v1", server.uri()),
            WireFormat::AnthropicMessages,
        ),
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
    assert_eq!(value["type"], "message");
    assert_eq!(value["content"][0]["text"], "Hi there");
    assert_eq!(value["model"], "switchyard");
}

#[tokio::test]
async fn streaming_chat_ends_with_done() {
    let server = MockServer::start().await;
    let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n\
         data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
         data: [DONE]\n\n";
    mock(
        &server,
        "/v1/chat/completions",
        ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"),
    )
    .await;

    let (status, body) = post(
        state(&format!("{}/v1", server.uri()), WireFormat::OpenAiChat),
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
async fn streaming_messages_ends_with_message_stop() {
    let server = MockServer::start().await;
    // A minimal Anthropic upstream stream: start, one text delta, then stop.
    let sse = "event: message_start\n\
         data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\
         \"role\":\"assistant\",\"model\":\"upstream-model\",\"content\":[],\
         \"stop_reason\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n\
         event: content_block_delta\n\
         data: {\"type\":\"content_block_delta\",\"index\":0,\
         \"delta\":{\"type\":\"text_delta\",\"text\":\"Hi there\"}}\n\n\
         event: message_delta\n\
         data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\
         \"usage\":{\"output_tokens\":2}}\n\n\
         event: message_stop\n\
         data: {\"type\":\"message_stop\"}\n\n";
    mock(
        &server,
        "/v1/messages",
        ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"),
    )
    .await;

    let (status, body) = post(
        state(
            &format!("{}/v1", server.uri()),
            WireFormat::AnthropicMessages,
        ),
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
    assert!(body.contains("event: message_stop"), "body: {body}");
    assert!(
        !body.contains("[DONE]"),
        "anthropic stream must not send [DONE]: {body}"
    );
}

#[tokio::test]
async fn upstream_500_passes_status_through() {
    let server = MockServer::start().await;
    mock(
        &server,
        "/v1/chat/completions",
        ResponseTemplate::new(500).set_body_string("boom"),
    )
    .await;

    let (status, body) = post(
        state(&format!("{}/v1", server.uri()), WireFormat::OpenAiChat),
        "/v1/chat/completions",
        json!({"model": "switchyard", "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;

    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert!(body.contains("boom"), "body: {body}");
}

#[tokio::test]
async fn routes_inbound_openai_to_anthropic_upstream() {
    let server = MockServer::start().await;
    // Inbound is OpenAI Chat, but the tier's upstream speaks Anthropic: the
    // request is translated to Anthropic Messages, and the Anthropic response is
    // translated back to the inbound OpenAI Chat shape.
    mock(
        &server,
        "/v1/messages",
        ResponseTemplate::new(200).set_body_json(anthropic_message_body()),
    )
    .await;

    let (status, body) = post(
        state(
            &format!("{}/v1", server.uri()),
            WireFormat::AnthropicMessages,
        ),
        "/v1/chat/completions",
        json!({"model": "switchyard", "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    let value: Value = serde_json::from_str(&body).unwrap();
    // Client-facing shape is OpenAI Chat, restamped to the served model id.
    assert_eq!(value["choices"][0]["message"]["content"], "Hi there");
    assert_eq!(value["model"], "switchyard");
}

#[tokio::test]
async fn non_object_body_is_400() {
    let server = MockServer::start().await;
    let (status, body) = post(
        state(&format!("{}/v1", server.uri()), WireFormat::OpenAiChat),
        "/v1/chat/completions",
        json!([1, 2, 3]),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body.contains("JSON object"), "body: {body}");
}

#[tokio::test]
async fn models_lists_served_model() {
    let server = MockServer::start().await;
    let router = build_router(state(
        &format!("{}/v1", server.uri()),
        WireFormat::OpenAiChat,
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
        WireFormat::OpenAiChat,
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
