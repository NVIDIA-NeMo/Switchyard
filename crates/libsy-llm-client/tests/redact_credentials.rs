// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Check fallback credit and MCP credentials across responses, streams, and process logs.

use std::collections::{BTreeMap, BTreeSet};
use std::process::Command;

use futures_util::StreamExt;
use serde_json::{Value, json};
use switchyard_llm_client::{
    Backend, HttpBackendConfig, ModelConfig, RawResponse, TranslatingLlmClient,
};
use switchyard_protocol::WireFormat;
use switchyard_translation::{PreservationPolicy, TranslationEngine, TranslationPolicy};
use tracing_subscriber::fmt::format::FmtSpan;
use wiremock::matchers::{body_partial_json, header, headers, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

const MCP_TOKEN: &str = "synthetic-mcp-secret-canary-8f651ed7";
const MCP_HEADER_TOKEN: &str = "synthetic-mcp-header-canary-58a40c92";
const MCP_BEARER_TOKEN: &str = "synthetic-mcp-bearer-canary-1d9ef430";
const FALLBACK_CREDIT_TOKEN: &str = "fallback-credit-secret-canary-8f651ed7";
const CREDIT_BETA: &str = "fallback-credit-2026-07-01";
const INITIAL_BETAS: &str = "fallback-credit-2026-07-01,server-side-fallback-2026-07-01";
const TRACE_MARKER: &str = "credential trace capture active";
const CHILD_MODE_ENV: &str = "SWITCHYARD_REDACT_CREDENTIALS_TEST_CHILD";

#[tokio::test]
async fn credentials_do_not_leak() -> TestResult {
    if std::env::var_os(CHILD_MODE_ENV).is_some() {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::TRACE)
            .with_span_events(FmtSpan::FULL)
            .with_ansi(false)
            .with_writer(std::io::stderr)
            .try_init()?;
        tracing::trace!("{TRACE_MARKER}");
        fallback_credit_retry().await?;
        return mcp_errors().await;
    }
    // Capture direct prints and logs from every async task in a child process.
    let output = Command::new(std::env::current_exe()?)
        .env(CHILD_MODE_ENV, "1")
        .args(["--exact", "credentials_do_not_leak", "--nocapture"])
        .output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    // Never print captured output on failure: it could contain credentials.
    for credential in [
        FALLBACK_CREDIT_TOKEN,
        MCP_TOKEN,
        MCP_HEADER_TOKEN,
        MCP_BEARER_TOKEN,
    ] {
        assert!(!stdout.contains(credential), "credential leaked to stdout");
        assert!(
            !stderr.contains(credential),
            "credential leaked to stderr/logs"
        );
    }
    assert!(output.status.success(), "credential scenarios failed");
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

async fn fallback_credit_retry() -> TestResult {
    let server = MockServer::start().await;
    let initial_body = json!({
        "model": "claude-fable-5",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": [{
            "type": "text",
            "text": "Review this code for security flaws.",
            "cache_control": {"type": "ephemeral"}
        }]}],
        "fallbacks": "default"
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
            "messages": initial_body["messages"]
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
        .call_rewrite_model_raw(retry_body, None, None, WireFormat::AnthropicMessages)
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
    Ok(())
}

async fn mcp_errors() -> TestResult {
    let server = MockServer::start().await;
    for format in [WireFormat::AnthropicMessages, WireFormat::OpenAiResponses] {
        let message = if format == WireFormat::OpenAiResponses {
            format!("Rejected {MCP_TOKEN} {MCP_HEADER_TOKEN} {MCP_BEARER_TOKEN}")
        } else {
            format!("Rejected {MCP_TOKEN}")
        };
        let (fields, request, start, failure, stream_failure) = match format {
            WireFormat::AnthropicMessages => {
                let failure = json!({"type": "error", "error": {
                    "type": "invalid_request_error", "message": format!("Rejected {MCP_TOKEN}")
                }});
                (
                    json!({"mcp_servers": [{"type": "url", "name": "inventory",
                        "url": "https://example.invalid/mcp", "authorization_token": MCP_TOKEN}]}),
                    json!({"model": "route", "max_tokens": 64, "messages": [{"role": "user", "content": "Check stock"}]}),
                    json!({"type": "message_start", "message": {"id": "msg_1", "model": "route", "usage": {}}}),
                    failure.clone(),
                    failure,
                )
            }
            WireFormat::OpenAiResponses => {
                let failure = json!({"id": "resp_1", "status": "failed", "output": [],
                    "error": {"code": "mcp_error", "message": message}});
                (
                    json!({"tools": [{"type": "mcp", "server_label": "inventory",
                        "server_url": "https://example.invalid/mcp", "authorization": MCP_TOKEN,
                        "headers": {"X-Workspace-Context": MCP_HEADER_TOKEN,
                            "Authorization": format!("Bearer {MCP_BEARER_TOKEN}")}}]}),
                    json!({"model": "route", "input": "Check stock"}),
                    json!({"type": "response.created", "response": {"id": "resp_1", "model": "route", "output": []}}),
                    failure.clone(),
                    json!({"type": "response.failed", "response": failure}),
                )
            }
            WireFormat::OpenAiChat => unreachable!(),
        };
        let config = HttpBackendConfig {
            base_url: server.uri(),
            api_key: None,
            forward_auth: false,
            extra_headers: BTreeMap::new(),
            extra_body: if format == WireFormat::AnthropicMessages {
                serde_json::from_value(fields.clone())?
            } else {
                BTreeMap::new()
            },
            omit_body_fields: BTreeSet::new(),
            reasoning_effort: None,
            max_retries: 0,
            failure_cooldown: std::time::Duration::ZERO,
            timeout: None,
        };
        let backend = match format {
            WireFormat::AnthropicMessages => Backend::Anthropic(config),
            WireFormat::OpenAiResponses => Backend::OpenAiResponses(config),
            WireFormat::OpenAiChat => unreachable!(),
        };
        let client = TranslatingLlmClient::new(&[ModelConfig::new("route", backend, None)])?;
        let echoed = json!({"error": {"message": fields.to_string()}});
        let plain = json!(message);
        for (status, streaming, error) in [
            (400, false, &echoed),
            (401, false, &plain),
            (200, false, &failure),
            (200, true, &stream_failure),
        ] {
            server.reset().await;
            let template = if streaming {
                ResponseTemplate::new(status).set_body_raw(
                    format!("data: {start}\n\ndata: {error}\n\n"),
                    "text/event-stream",
                )
            } else if let Some(text) = error.as_str() {
                ResponseTemplate::new(status).set_body_string(text)
            } else {
                ResponseTemplate::new(status).set_body_json(error)
            };
            Mock::given(method("POST"))
                .and(body_partial_json(&fields))
                .respond_with(template)
                .expect(1)
                .mount(&server)
                .await;
            let mut request = request.clone();
            request["stream"] = json!(streaming);
            if format == WireFormat::OpenAiResponses {
                request["tools"] = fields["tools"].clone();
                // Force headers through McpToolset rather than replaying the original JSON.
                let engine = TranslationEngine::default();
                let policy = TranslationPolicy {
                    preservation: PreservationPolicy::Disabled,
                    ..TranslationPolicy::default()
                };
                let decoded = engine.decode_request(format, &request, &policy)?;
                tracing::debug!(request = ?decoded.request, "normalized MCP header request");
                request = engine
                    .encode_request(format, &decoded.request, &policy)?
                    .body;
            }
            let mut saw_redaction = false;
            match client
                .call_rewrite_model_raw(request, None, None, format)
                .await
            {
                Err(error) => {
                    saw_redaction = error.to_string().contains("[REDACTED]");
                    tracing::warn!(%error, "upstream rejected request");
                }
                Ok(RawResponse::Stream(mut events)) => {
                    while let Some(event) = events.next().await {
                        let text = match event {
                            Err(error) => error.to_string(),
                            Ok(value) => value.to_string(),
                        };
                        saw_redaction |= text.contains("[REDACTED]");
                        tracing::debug!(%text, "upstream event");
                    }
                }
                Ok(RawResponse::Buffered(value)) => {
                    saw_redaction = value.to_string().contains("[REDACTED]");
                    tracing::debug!(%value, "upstream response");
                }
            }
            assert!(
                saw_redaction,
                "missing redaction for {format}, status={status}, streaming={streaming}"
            );
            server.verify().await;
        }
    }
    Ok(())
}
