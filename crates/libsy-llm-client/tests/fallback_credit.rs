// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mock refusal and credit redemption with process-level checks for credential leaks.

use std::collections::{BTreeMap, BTreeSet};
use std::process::Command;

use serde_json::{Value, json};
use switchyard_llm_client::{
    Backend, HttpBackendConfig, ModelConfig, RawResponse, TranslatingLlmClient,
};
use switchyard_protocol::WireFormat;
use tracing_subscriber::fmt::format::FmtSpan;
use wiremock::matchers::{body_partial_json, header, headers, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

const TOKEN: &str = "fallback-credit-secret-canary-8f651ed7";
const CREDIT_BETA: &str = "fallback-credit-2026-07-01";
const INITIAL_BETAS: &str = "fallback-credit-2026-07-01,server-side-fallback-2026-07-01";
const TRACE_MARKER: &str = "fallback credit trace capture active";
const CHILD_MODE_ENV: &str = "SWITCHYARD_FALLBACK_CREDIT_TEST_CHILD";

#[tokio::test]
async fn fallback_credit_retry_does_not_leak_token() -> TestResult {
    if std::env::var_os(CHILD_MODE_ENV).is_some() {
        return fallback_credit_retry_child().await;
    }
    // A child process captures direct prints as well as logs from every async task.
    let output = Command::new(std::env::current_exe()?)
        .env(CHILD_MODE_ENV, "1")
        .args([
            "--exact",
            "fallback_credit_retry_does_not_leak_token",
            "--nocapture",
        ])
        .output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    // Do not print captured output on failure: it could contain the credential.
    assert!(
        !stdout.contains(TOKEN),
        "fallback credit token leaked to stdout"
    );
    assert!(
        !stderr.contains(TOKEN),
        "fallback credit token leaked to stderr/logs"
    );
    assert!(output.status.success(), "mock refusal and retry failed");
    assert!(
        stderr.contains(TRACE_MARKER),
        "TRACE logging was not captured"
    );
    assert!(
        stderr.contains("libsy.upstream_attempt"),
        "client span fields were not captured"
    );
    Ok(())
}

async fn fallback_credit_retry_child() -> TestResult {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_span_events(FmtSpan::FULL)
        .with_ansi(false)
        .with_writer(std::io::stderr)
        .try_init()?;
    tracing::trace!("{TRACE_MARKER}");

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
                "fallback_credit_token": TOKEN,
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
            "fallback_credit_token": TOKEN,
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
    assert!(token == TOKEN, "refusal changed the credit token");

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
