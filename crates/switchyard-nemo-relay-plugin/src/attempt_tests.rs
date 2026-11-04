// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
    mpsc,
};

use futures_util::StreamExt;
use switchyard_protocol::{Request, WireFormat, text_request};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

use crate::{config::SwitchyardConfig, runtime::SwitchyardRuntime};

fn runtime(server: &MockServer, format: &str, retries: u32) -> SwitchyardRuntime {
    let config = serde_json::from_value::<SwitchyardConfig>(serde_json::json!({
        "switchyard_config": {
            "schema_version": 1,
            "llm_clients": {"local": {"format": format, "base_url": format!("{}/v1",server.uri()), "max_retries": retries, "failure_cooldown_ms": 0}},
            "targets": {"default": {"id": "test/model", "llm_client": "local"}},
            "routes": {"default": {"id": "switchyard/default", "type": "passthrough", "target": "default"}}
        }
    })).unwrap();
    SwitchyardRuntime::new(config).unwrap()
}

fn request(stream: bool) -> Request {
    let mut llm_request = text_request(Some("switchyard/default".into()), "test");
    llm_request.stream = stream;
    Request {
        llm_request,
        raw_request: None,
        metadata: None,
    }
}

fn recorder(payloads: bool) -> (SpanCapture, mpsc::Receiver<CapturedSpan>) {
    let (tx, rx) = mpsc::channel();
    (
        SpanCapture::new(payloads.then_some(8192), move |span| tx.send(span).unwrap()),
        rx,
    )
}

fn success() -> ResponseTemplate {
    ResponseTemplate::new(200).insert_header("x-litellm-response-cost", "1.0").set_body_json(serde_json::json!({
        "id":"response", "model":"test/model", "choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
        "usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9,"cost":0.03}
    }))
}

#[tokio::test]
async fn retry_usage_and_reported_charges_survive_failure_without_parent_charge() {
    let server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = calls.clone();
    Mock::given(method("POST")).respond_with(move |_: &wiremock::Request| {
        if seen.fetch_add(1, Ordering::SeqCst) == 0 {
            ResponseTemplate::new(503).insert_header("retry-after", "0").set_body_json(serde_json::json!({
                "error":{"message":"failed"}, "usage":{"prompt_tokens":11,"completion_tokens":3,"cost":0.02}
            }))
        } else { success() }
    }).mount(&server).await;
    let runtime = runtime(&server, "openai_chat", 1);
    let (capture, rx) = recorder(false);
    let result = capture
        .capture(runtime.execute_buffered(WireFormat::OpenAiChat, request(false)))
        .await;
    assert!(result.result.is_ok());
    let call = rx.try_recv().unwrap();
    assert_eq!(call.children.len(), 2);
    assert_eq!(call.fields["switchyard.accounting.attempts"], 2);
    let (kind, metadata, _, output) = call.projection();
    assert!(matches!(kind, ScopeType::Custom));
    assert!(output.is_none());
    assert_eq!(metadata["switchyard.accounting.attempt_records"], 2);
    for (attempt, outcome, input, cost) in [
        (&call.children[0], "error", 11, 0.02),
        (&call.children[1], "ok", 7, 0.03),
    ] {
        let (kind, metadata, input_data, output) = attempt.projection();
        assert!(matches!(kind, ScopeType::Llm));
        assert_eq!(metadata["outcome"], outcome);
        assert_eq!(metadata["switchyard.accounting.usage_status"], "complete");
        assert_eq!(input_data.unwrap()["model"], "test/model");
        let output = output.unwrap();
        assert_eq!(output["model"], "test/model");
        assert_eq!(output["usage"]["input_tokens"], input);
        assert_eq!(output["usage"]["cost"]["total"], cost);
    }
    assert_eq!(
        call.children[1].fields["switchyard.accounting.charge_source"],
        "provider"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn missing_retry_usage_is_not_zero() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(503).insert_header("retry-after", "0"))
        .mount(&server)
        .await;
    let runtime = runtime(&server, "openai_chat", 1);
    let (capture, rx) = recorder(false);
    assert!(
        capture
            .capture(runtime.execute_buffered(WireFormat::OpenAiChat, request(false)))
            .await
            .result
            .is_err()
    );
    let call = rx.try_recv().unwrap();
    assert_eq!(call.children.len(), 2);
    for attempt in &call.children {
        let (_, fields, _, output) = attempt.projection();
        assert_eq!(fields["switchyard.accounting.usage_status"], "missing");
        assert!(output.unwrap().get("usage").is_none());
    }
}

#[tokio::test]
async fn buffered_failed_generation_keeps_usage_without_retrying_content_failure() {
    let server = MockServer::start().await;
    Mock::given(method("POST")).respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
        "id":"failed", "model":"test/model", "status":"failed", "error":{"message":"generation failed"},
        "usage":{"input_tokens":11,"output_tokens":3,"total_tokens":14}
    }))).expect(1).mount(&server).await;
    let runtime = runtime(&server, "openai_responses", 1);
    let (capture, rx) = recorder(false);
    assert!(
        capture
            .capture(runtime.execute_buffered(WireFormat::OpenAiResponses, request(false)))
            .await
            .result
            .is_err()
    );
    let call = rx.try_recv().unwrap();
    let [attempt] = call.children.as_slice() else {
        panic!("expected one HTTP attempt")
    };
    assert_eq!(attempt.fields["outcome"], "error");
    assert_eq!(attempt.fields["gen_ai.usage.input_tokens"], 11);
    assert_eq!(
        attempt.fields["switchyard.accounting.usage_status"],
        "complete"
    );
}

#[tokio::test]
async fn failed_stream_terminal_frame_keeps_usage_and_error() {
    let server = MockServer::start().await;
    let body = concat!(
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"r1\",\"model\":\"test/model\"}}\n\n",
        "data: {\"type\":\"response.failed\",\"response\":{\"error\":{\"message\":\"failed\"},\"usage\":{\"input_tokens\":11,\"output_tokens\":3}}}\n\n"
    );
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .expect(1)
        .mount(&server)
        .await;
    let runtime = runtime(&server, "openai_responses", 1);
    let (capture, rx) = recorder(false);
    capture
        .capture(async {
            let mut events = runtime
                .execute_stream(WireFormat::OpenAiResponses, request(true), Arc::new(|_| {}))
                .await
                .result
                .unwrap();
            while events.next().await.is_some() {}
        })
        .await;
    let call = rx.try_recv().unwrap();
    let [attempt] = call.children.as_slice() else {
        panic!("expected one HTTP attempt")
    };
    assert_eq!(attempt.fields["outcome"], "error");
    assert_eq!(attempt.fields["gen_ai.usage.output_tokens"], 3);
    assert_eq!(
        attempt.fields["switchyard.accounting.usage_status"],
        "complete"
    );
}

#[tokio::test]
async fn abandoning_stream_preserves_partial_usage() {
    let server = MockServer::start().await;
    let body = concat!(
        "data: {\"id\":\"r1\",\"model\":\"test/model\",\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":3}}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}}]}\n\n"
    );
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .mount(&server)
        .await;
    let runtime = runtime(&server, "openai_chat", 0);
    let (capture, rx) = recorder(false);
    capture
        .capture(async {
            let mut events = runtime
                .execute_stream(WireFormat::OpenAiChat, request(true), Arc::new(|_| {}))
                .await
                .result
                .unwrap();
            while let Some(event) = events.next().await {
                if event.unwrap().to_string().contains("partial") {
                    break;
                }
            }
            drop(events);
        })
        .await;
    let call = rx.try_recv().unwrap();
    let attempt = &call.children[0];
    assert_eq!(attempt.fields["outcome"], "cancelled");
    assert_eq!(attempt.fields["gen_ai.usage.input_tokens"], 11);
    assert_eq!(
        attempt.fields["switchyard.accounting.usage_status"],
        "partial"
    );
}

#[test]
fn debug_payloads_cannot_replace_usage_and_summary_is_not_priceable() {
    let fields = serde_json::json!({"gen_ai.request.model":"model", "gen_ai.usage.input_tokens":12, "gen_ai.usage.output_tokens":0, "switchyard.accounting.unit":"completion_attempt"}).as_object().unwrap().clone();
    let mut attempt = CapturedSpan {
        name: UPSTREAM_ATTEMPT,
        fields,
        payloads: Payloads {
            request: Some(serde_json::json!({"body":"{}","truncated":false})),
            response: Some(serde_json::json!({"body":null,"truncated":true})),
        },
        started_at: SystemTime::now(),
        ended_at: SystemTime::now(),
        children: Vec::new(),
    };
    let (_, _, input, output) = attempt.projection();
    assert_eq!(input.unwrap()["model"], "model");
    let output = output.unwrap();
    assert_eq!(output["usage"]["output_tokens"], 0);
    assert_eq!(output["truncated"], true);
    attempt.fields.insert(
        "switchyard.accounting.unit".into(),
        Json::String("summary".into()),
    );
    let (kind, _, _, output) = attempt.projection();
    assert!(matches!(kind, ScopeType::Custom));
    assert!(output.is_none());
}

#[tokio::test]
async fn header_only_charge_is_preserved_on_an_http_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(400).insert_header("x-litellm-response-cost", "0"))
        .expect(1)
        .mount(&server)
        .await;
    let runtime = runtime(&server, "openai_chat", 0);
    let (capture, rx) = recorder(false);
    assert!(
        capture
            .capture(runtime.execute_buffered(WireFormat::OpenAiChat, request(false)))
            .await
            .result
            .is_err()
    );
    let call = rx.try_recv().unwrap();
    let [attempt] = call.children.as_slice() else {
        panic!("expected one attempt")
    };
    let (_, meta, _, output) = attempt.projection();
    assert_eq!(meta["outcome"], "error");
    assert_eq!(meta["switchyard.accounting.usage_status"], "missing");
    assert_eq!(meta["switchyard.accounting.charge_source"], "header");
    assert_eq!(output.unwrap()["usage"]["cost"]["total"], 0.0);
}

#[tokio::test]
async fn expired_deadline_does_not_create_an_attempt() {
    let server = MockServer::start().await;
    use switchyard_llm_client::{Backend, HttpBackendConfig, ModelConfig, TranslatingLlmClient};
    use tracing::Instrument;
    let client = TranslatingLlmClient::new(&[ModelConfig::new(
        "test/model",
        Backend::OpenAiChat(HttpBackendConfig {
            base_url: server.uri(),
            api_key: None,
            forward_auth: false,
            extra_headers: Default::default(),
            extra_body: Default::default(),
            omit_body_fields: Default::default(),
            reasoning_effort: None,
            max_retries: 0,
            failure_cooldown: std::time::Duration::ZERO,
            timeout: Some(std::time::Duration::ZERO),
        }),
        None,
    )])
    .unwrap();
    let (capture, rx) = recorder(false);
    let result = capture
        .capture(async {
            client
                .call_rewrite_model(request(false), Some(&"test/model".into()))
                .instrument(tracing::debug_span!(target: "libsy", "libsy.client_call",
                switchyard.accounting.unit = tracing::field::Empty,
                switchyard.accounting.attempts = tracing::field::Empty))
                .await
        })
        .await;
    assert!(result.is_err());
    let call = rx.try_recv().unwrap();
    assert!(call.children.is_empty());
    assert_eq!(call.fields["switchyard.accounting.attempts"], 0);
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn concurrent_calls_keep_their_own_attempts() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(success().set_delay(std::time::Duration::from_millis(10)))
        .expect(2)
        .mount(&server)
        .await;
    let runtime = runtime(&server, "openai_chat", 0);
    let (capture, rx) = recorder(false);
    let first = capture.capture(runtime.execute_buffered(WireFormat::OpenAiChat, request(false)));
    let second = capture.capture(runtime.execute_buffered(WireFormat::OpenAiChat, request(false)));
    let (first, second) = tokio::join!(first, second);
    assert!(first.result.is_ok());
    assert!(second.result.is_ok());
    let calls: Vec<_> = rx.try_iter().collect();
    assert_eq!(calls.len(), 2);
    for call in calls {
        assert_eq!(call.children.len(), 1);
        assert_eq!(call.fields["switchyard.accounting.attempts"], 1);
        assert_eq!(call.children[0].fields["gen_ai.usage.input_tokens"], 7);
    }
}

#[tokio::test]
async fn prefetched_context_rejection_keeps_usage_and_specific_error_type() {
    let server = MockServer::start().await;
    let body = "data: {\"type\":\"response.failed\",\"response\":{\"error\":{\"message\":\"maximum context length exceeded\"},\"usage\":{\"input_tokens\":11,\"output_tokens\":0}}}\n\n";
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
        .expect(1)
        .mount(&server)
        .await;
    let runtime = runtime(&server, "openai_responses", 1);
    let (capture, rx) = recorder(false);
    assert!(
        capture
            .capture(runtime.execute_stream(
                WireFormat::OpenAiResponses,
                request(true),
                Arc::new(|_| {})
            ))
            .await
            .result
            .is_err()
    );
    let call = rx.try_recv().unwrap();
    let [attempt] = call.children.as_slice() else {
        panic!("expected one attempt")
    };
    assert_eq!(attempt.fields["gen_ai.usage.input_tokens"], 11);
    assert_eq!(attempt.fields["error.type"], "context_window_exceeded");
    assert_eq!(
        attempt.fields["switchyard.accounting.usage_status"],
        "complete"
    );
}
