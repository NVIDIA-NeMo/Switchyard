// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for the reusable mock LLM server.

use std::error::Error;

use reqwest::Client;
use serde_json::{json, Value};
use switchyard_protocol::completion_text;
use switchyard_test_server::{MockLlmServer, DEFAULT_RESPONSE_BANK};
use switchyard_translation::{decode_aggregated_response, WireFormat};

type TestResult = Result<(), Box<dyn Error + Send + Sync>>;

#[tokio::test]
async fn serves_all_buffered_provider_formats_and_captures_requests() -> TestResult {
    let server = MockLlmServer::builder()
        .default_response("ok")
        .model_response("model/special", "special")
        .start()
        .await?;
    let client = Client::new();
    let cases = [
        (
            "/v1/chat/completions",
            WireFormat::OpenAiChat,
            json!({"model": "model/default", "messages": []}),
            "ok",
        ),
        (
            "/v1/responses",
            WireFormat::OpenAiResponses,
            json!({"model": "model/special", "input": "hi"}),
            "special",
        ),
        (
            "/v1/messages",
            WireFormat::AnthropicMessages,
            json!({"model": "model/default", "max_tokens": 16, "messages": []}),
            "ok",
        ),
    ];

    for (path, format, request, expected) in cases {
        let body: Value = client
            .post(format!("{}{path}", server.url()))
            .bearer_auth("test-key")
            .json(&request)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        let response = decode_aggregated_response(&body, format)?;
        assert_eq!(completion_text(&response), expected);
        assert_eq!(response.usage.cached_input_tokens(), Some(7));
    }

    let requests = server.requests().await;
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[0].path, "/v1/chat/completions");
    assert_eq!(
        requests[0].headers.get("authorization").map(String::as_str),
        Some("Bearer test-key")
    );
    assert_eq!(requests[1].body["model"], "model/special");
    assert_eq!(requests[2].path, "/v1/messages");
    Ok(())
}

#[tokio::test]
async fn serves_streams_for_all_provider_formats() -> TestResult {
    let server = MockLlmServer::builder()
        .default_response("ok")
        .start()
        .await?;
    let client = Client::new();
    for (path, request, expected_event, has_done) in [
        (
            "/v1/chat/completions",
            json!({"model": "model/a", "messages": [], "stream": true}),
            None,
            true,
        ),
        (
            "/v1/responses",
            json!({"model": "model/a", "input": "hi", "stream": true}),
            Some("event: response."),
            true,
        ),
        (
            "/v1/messages",
            json!({"model": "model/a", "max_tokens": 16, "messages": [], "stream": true}),
            Some("event: message_"),
            false,
        ),
    ] {
        let body = client
            .post(format!("{}{path}", server.url()))
            .json(&request)
            .send()
            .await?
            .error_for_status()?
            .text()
            .await?;
        assert!(body.contains("ok"));
        assert_eq!(body.contains("data: [DONE]"), has_done);
        if let Some(expected_event) = expected_event {
            assert!(body.contains(expected_event));
        }
    }
    Ok(())
}

#[tokio::test]
async fn default_response_is_selected_from_the_built_in_bank() -> TestResult {
    let server = MockLlmServer::start().await?;
    let body: Value = Client::new()
        .post(format!("{}/v1/chat/completions", server.url()))
        .json(&json!({"model": "model/a", "messages": []}))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let response = decode_aggregated_response(&body, WireFormat::OpenAiChat)?;
    assert!(DEFAULT_RESPONSE_BANK.contains(&completion_text(&response).as_str()));
    Ok(())
}

#[tokio::test]
async fn injects_errors_by_model_or_header() -> TestResult {
    let server = MockLlmServer::builder()
        .model_error("model/fail", reqwest::StatusCode::IM_A_TEAPOT)
        .start()
        .await?;
    let client = Client::new();

    let by_model = client
        .post(format!("{}/v1/chat/completions", server.url()))
        .json(&json!({"model": "model/fail", "messages": []}))
        .send()
        .await?;
    assert_eq!(by_model.status(), reqwest::StatusCode::IM_A_TEAPOT);

    let by_header = client
        .post(format!("{}/v1/messages", server.url()))
        .header("x-switchyard-test-status", "429")
        .json(&json!({"model": "model/a", "messages": []}))
        .send()
        .await?;
    assert_eq!(by_header.status(), reqwest::StatusCode::TOO_MANY_REQUESTS);
    Ok(())
}
