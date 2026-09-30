// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exercises the configured credential through the real runner and loopback HTTP client.

use std::process::Command;

use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::*;

const KEY_ENV: &str = "SWITCHYARD_RELAY_REDACTION_TEST_KEY";
const KEY: &str = "qa-only-synthetic-relay-provider-key";

// A child process owns the environment so parallel tests never race on set_var.
#[test]
fn configured_provider_credentials_are_redacted() {
    const CHILD: &str = "SWITCHYARD_RELAY_REDACTION_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "runtime::redaction_tests::configured_provider_credentials_are_redacted",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env(KEY_ENV, KEY)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        for format in [
            WireFormat::OpenAiChat,
            WireFormat::OpenAiResponses,
            WireFormat::AnthropicMessages,
        ] {
            for streaming in [false, true] {
                reflected_response(format, streaming, false).await;
                reflected_response(format, streaming, true).await;
            }
        }
    });
}

async fn reflected_response(format: WireFormat, streaming: bool, fail: bool) {
    let server = MockServer::start().await;
    let reflected = format!("ordinary diagnostics; Bearer {KEY}");
    let body = if streaming {
        let chunk = json!({
            "id": "chatcmpl-redaction", "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": reflected}, "finish_reason": null}]
        });
        if fail {
            format!(
                "data: {chunk}\n\nevent: error\ndata: {}\n\n",
                json!({"error": {"message": reflected, "type": "provider_error"}})
            )
        } else {
            format!(
                "data: {chunk}\n\ndata: {}\n\ndata: [DONE]\n\n",
                json!({"id": "chatcmpl-redaction", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            )
        }
    } else if fail {
        json!({"error": {"message": reflected, "type": "provider_error"}}).to_string()
    } else {
        json!({
            "id": "chatcmpl-redaction", "object": "chat.completion", "model": "target/model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": reflected}, "finish_reason": "stop"}]
        }).to_string()
    };
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", format!("Bearer {KEY}")))
        .respond_with(
            ResponseTemplate::new(if fail && !streaming { 400 } else { 200 })
                .insert_header("x-request-id", KEY)
                .insert_header(
                    "content-type",
                    if streaming {
                        "text/event-stream"
                    } else {
                        "application/json"
                    },
                )
                .set_body_string(body),
        )
        .expect(1)
        .mount(&server)
        .await;
    let deployment = json!({
        "schema_version": 1,
        "llm_clients": {"target": {"format": "openai_chat", "base_url": format!("{}/v1", server.uri()), "api_key_env": KEY_ENV, "max_retries": 0}},
        "targets": {"default": {"id": "target/model", "llm_client": "target"}},
        "routes": {"default": {"id": "switchyard/default", "type": "passthrough", "target": "default"}}
    });
    let runtime = SwitchyardRuntime::new(SwitchyardConfig {
        priority: 0,
        switchyard_config_path: None,
        switchyard_config: Some(deployment.as_object().unwrap().clone()),
    })
    .unwrap();
    let mut llm_request = switchyard_protocol::text_request(
        Some("switchyard/default".into()),
        "reflect the synthetic credential",
    );
    llm_request.stream = streaming;
    let request = Request {
        llm_request,
        raw_request: None,
        metadata: Some(Metadata {
            session_id: Some(KEY.into()),
            ..Default::default()
        }),
    };
    let captured = Arc::new(Mutex::new(Vec::new()));
    let output = if streaming {
        let observed = Arc::clone(&captured);
        let execution = runtime
            .execute_stream(
                format,
                request,
                Arc::new(move |event| {
                    observed.lock().unwrap().push(event);
                }),
            )
            .await;
        assert!(
            captured.lock().unwrap().is_empty(),
            "stream must remain unpolled"
        );
        captured.lock().unwrap().extend(execution.events);
        let events = execution.result.unwrap().collect::<Vec<_>>().await;
        assert_eq!(events.iter().any(Result::is_err), fail);
        format!("{events:?}")
    } else {
        let execution = runtime.execute_buffered(format, request).await;
        captured.lock().unwrap().extend(execution.events);
        assert_eq!(execution.result.is_err(), fail);
        format!("{:?}", execution.result)
    };
    assert!(
        !output.contains(KEY),
        "{format:?}, streaming={streaming}, fail={fail}: {output}"
    );
    if !fail || streaming {
        assert!(output.contains("ordinary diagnostics"), "{output}");
        assert!(output.contains("[REDACTED]"), "{output}");
    }
    let telemetry = format!("{:?}", captured.lock().unwrap());
    assert!(!telemetry.contains(KEY), "{telemetry}");
    assert!(telemetry.contains("[REDACTED]"), "{telemetry}");
}

#[test]
fn routing_data_and_metric_attributes_are_redacted() {
    let redactor = ProviderKeyRedactor::new(&[KEY.into()]);
    let events = [
        RoutingEvent::Mark(RoutingMark {
            name: "switchyard.routing.decision".into(),
            data: json!({"evidence": {"source": KEY}, "selected_model": KEY}),
            metadata: json!({"session_id": KEY}),
            severity: Some(LogSeverity::Info),
        }),
        counter_metric(
            "switchyard.routing.requests",
            "Requests managed by Switchyard routing.",
            json!({"target_model": KEY, KEY: [KEY, "ordinary"]}),
            json!({"session_id": KEY}),
        ),
    ];
    for event in events {
        let sanitized = sanitize_event(&redactor, event);
        let output = format!("{sanitized:?}");
        assert!(!output.contains(KEY), "{output}");
        assert!(output.contains("[REDACTED]"), "{output}");
    }
}

#[test]
fn client_visible_errors_preserve_only_nonsecret_diagnostics() {
    let runtime = SwitchyardRuntime {
        runner: Runner::new(Vec::new()),
        translation: TranslationEngine::default(),
        redactor: Arc::new(ProviderKeyRedactor::new(&[KEY.into()])),
    };
    let execution: Execution<()> = runtime.sanitize_execution(Execution {
        result: Err(format!("ordinary transport diagnostics: {KEY}")),
        events: Vec::new(),
    });
    assert_eq!(
        execution.result.unwrap_err(),
        "ordinary transport diagnostics: [REDACTED]"
    );
    let request = RelayRequest {
        headers: Map::new(),
        content: json!({"model": "switchyard/default", "messages": [{"role": KEY, "content": "hello"}]}),
    };
    let error = runtime
        .decode_request(WireFormat::OpenAiChat, request, false)
        .err()
        .expect("an invalid role should fail decoding");
    assert!(!error.contains(KEY), "{error}");
}
