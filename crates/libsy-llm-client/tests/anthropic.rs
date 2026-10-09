// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Anthropic fallback credit and MCP tests with process-level checks for credential leaks.

use std::collections::{BTreeMap, BTreeSet};
use std::process::Command;

use futures_util::StreamExt;
use serde_json::{Value, json};
use switchyard_llm_client::{
    Backend, HttpBackendConfig, ModelConfig, RawResponse, TranslatingLlmClient,
};
use switchyard_protocol::WireFormat;
use tracing_subscriber::fmt::format::FmtSpan;
use wiremock::matchers::{body_partial_json, header, headers, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

const FALLBACK_CREDIT_TOKEN: &str = "fallback-credit-secret-canary-8f651ed7";
const CREDIT_BETA: &str = "fallback-credit-2026-07-01";
const INITIAL_BETAS: &str = "fallback-credit-2026-07-01,server-side-fallback-2026-07-01";
const TRACE_MARKER: &str = "Anthropic credential trace capture active";
const CHILD_MODE_ENV: &str = "SWITCHYARD_ANTHROPIC_CREDENTIALS_TEST_CHILD";

const MCP_TOKEN: &str = "synthetic-mcp-secret-\"\\-canary";

async fn child_method() -> TestResult {
    let server = MockServer::start().await;
    let initial_body = json!({
        "model": "claude-fable-5",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": [{
            "type": "text",
            "text": "Review this code for security flaws.",
            "cache_control": {"type": "ephemeral"}
        }]}],
        "fallbacks": "default",
        "mcp_servers": [{"type": "url", "name": "inventory",
            "url": "https://example.invalid/mcp", "authorization_token": MCP_TOKEN}],
        "tools": [{"type": "mcp_toolset", "mcp_server_name": "inventory"}]
    });
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("anthropic-version", "2023-06-01"))
        .and(headers(
            "anthropic-beta",
            vec![CREDIT_BETA, "server-side-fallback-2026-07-01"],
        ))
        .and(body_partial_json(&initial_body))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_refused",
            "type": "message",
            "role": "assistant",
            "model": "claude-fable-5",
            "content": [],
            "stop_reason": "refusal",
            "stop_details": {
                "type": "refusal",
                "category": "cyber",
                "explanation": "The request was declined.",
                "recommended_model": "claude-opus-4-8",
                "fallback_credit_token": FALLBACK_CREDIT_TOKEN,
                "fallback_has_prefill_claim": false
            },
            "usage": {"input_tokens": 10, "output_tokens": 0}
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("anthropic-version", "2023-06-01"))
        .and(header("anthropic-beta", CREDIT_BETA))
        .and(body_partial_json(json!({
            "model": "claude-opus-4-8",
            "fallback_credit_token": FALLBACK_CREDIT_TOKEN,
            "messages": initial_body["messages"],
            "mcp_servers": initial_body["mcp_servers"],
            "tools": initial_body["tools"]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_retried",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-4-8",
            "content": [{"type": "text", "text": "Review completed."}],
            "stop_reason": "end_turn",
            "stop_details": null,
            "usage": {"input_tokens": 10, "output_tokens": 3}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let models = [
        ("claude-fable-5", INITIAL_BETAS),
        ("claude-opus-4-8", CREDIT_BETA),
    ]
    .map(|(model, beta)| {
        ModelConfig::new(
            model,
            Backend::Anthropic(HttpBackendConfig {
                base_url: server.uri(),
                api_key: None,
                forward_auth: false,
                extra_headers: BTreeMap::from([("anthropic-beta".to_string(), beta.to_string())]),
                extra_body: BTreeMap::new(),
                omit_body_fields: BTreeSet::new(),
                reasoning_effort: None,
                max_retries: 0,
                failure_cooldown: std::time::Duration::ZERO,
                timeout: None,
            }),
            None,
        )
    });
    let client = TranslatingLlmClient::new(&models)?;
    let RawResponse::Buffered(refusal) = client
        .call_rewrite_model_raw(
            initial_body.clone(),
            None,
            None,
            WireFormat::AnthropicMessages,
        )
        .await?
    else {
        return Err("expected a buffered refusal".into());
    };
    assert_eq!(refusal["stop_reason"], "refusal");
    let token = refusal["stop_details"]["fallback_credit_token"]
        .as_str()
        .ok_or("refusal did not preserve the credit token")?;
    assert!(
        token == FALLBACK_CREDIT_TOKEN,
        "refusal changed the credit token"
    );

    let mut retry_body = initial_body;
    retry_body
        .as_object_mut()
        .ok_or("expected an object")?
        .remove("fallbacks");
    retry_body["model"] = json!("claude-opus-4-8");
    retry_body["fallback_credit_token"] = json!(token);
    let RawResponse::Buffered(answer) = client
        .call_rewrite_model_raw(
            retry_body.clone(),
            None,
            None,
            WireFormat::AnthropicMessages,
        )
        .await?
    else {
        return Err("expected a buffered retry response".into());
    };
    assert_eq!(answer["stop_reason"], "end_turn");
    assert_eq!(answer["content"][0]["text"], "Review completed.");

    let requests = server
        .received_requests()
        .await
        .ok_or("missing request recording")?;
    let first: Value = serde_json::from_slice(&requests[0].body)?;
    assert!(first.get("fallback_credit_token").is_none());
    server.verify().await;

    let error = json!({"type": "error", "error": {
        "type": "invalid_request_error", "message": format!("prompt is too long: {MCP_TOKEN}")
    }});
    let request_echo = json!({"mcp_servers": retry_body["mcp_servers"]}).to_string();
    let echoed_request_error = json!({"type": "error", "error": {
        "type": "invalid_request_error", "message": format!("prompt is too long: {request_echo}")
    }});
    let plain_error = json!(format!("rejected MCP token: {MCP_TOKEN}"));
    for (status, streaming, prefix, error) in [
        (400, false, "", &echoed_request_error),
        (401, false, "", &plain_error),
        (200, false, "", &error),
        (
            200,
            true,
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude\",\"usage\":{}}}\n\n",
            &error,
        ),
    ] {
        server.reset().await;
        retry_body["stream"] = json!(streaming);
        let template = if streaming {
            ResponseTemplate::new(status)
                .set_body_raw(format!("{prefix}data: {error}\n\n"), "text/event-stream")
        } else if let Some(text) = error.as_str() {
            ResponseTemplate::new(status).set_body_string(text)
        } else {
            ResponseTemplate::new(status).set_body_json(error)
        };
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("anthropic-version", "2023-06-01"))
            .and(header("anthropic-beta", CREDIT_BETA))
            .and(body_partial_json(&retry_body))
            .respond_with(template)
            .expect(1)
            .mount(&server)
            .await;
        match client
            .call_rewrite_model_raw(
                retry_body.clone(),
                None,
                None,
                WireFormat::AnthropicMessages,
            )
            .await
        {
            Err(error) => tracing::warn!(%error, "upstream rejected request"),
            Ok(RawResponse::Stream(mut events)) => {
                while let Some(event) = events.next().await {
                    match event {
                        Err(error) => tracing::warn!(%error, "upstream stream failed"),
                        Ok(value) => tracing::debug!(%value, "upstream event"),
                    }
                }
            }
            Ok(RawResponse::Buffered(value)) => tracing::debug!(%value, "upstream response"),
        }
        server.verify().await;
    }
    Ok(())
}

#[tokio::test]
async fn ensure_credentials_are_not_leaked() -> TestResult {
    if std::env::var_os(CHILD_MODE_ENV).is_some() {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::TRACE)
            .with_span_events(FmtSpan::FULL)
            .with_ansi(false)
            .with_writer(std::io::stderr)
            .try_init()?;
        tracing::trace!("{TRACE_MARKER}");
        return child_method().await;
    }
    // A child process captures direct prints as well as logs from every async task.
    let output = Command::new(std::env::current_exe()?)
        .env(CHILD_MODE_ENV, "1")
        .args([
            "--exact",
            "ensure_credentials_are_not_leaked",
            "--nocapture",
        ])
        .output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    // Do not print captured output on failure: it could contain a credential.
    // Check the MCP token prefix too, since JSON escapes quotes and backslashes.
    for token in [FALLBACK_CREDIT_TOKEN, "synthetic-mcp-secret"] {
        assert!(!stdout.contains(token), "credential leaked to stdout");
        assert!(!stderr.contains(token), "credential leaked to stderr/logs");
    }
    assert!(
        output.status.success(),
        "Anthropic credential scenarios failed"
    );
    assert!(
        stderr.contains(TRACE_MARKER),
        "TRACE logging was not captured"
    );
    assert!(
        stderr.contains("libsy.upstream_attempt"),
        "client span fields were not captured"
    );
    assert!(
        stderr.contains("[REDACTED]"),
        "redacted errors were not logged"
    );
    Ok(())
}
