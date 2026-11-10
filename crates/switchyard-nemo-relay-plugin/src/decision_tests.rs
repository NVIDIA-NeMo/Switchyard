// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{config::SwitchyardConfig, runtime::SwitchyardRuntime};
use std::sync::{Once, mpsc};
use switchyard_protocol::{Metadata as RequestMetadata, Request, WireFormat, text_request};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

fn runtime(server: &MockServer, format: &str) -> SwitchyardRuntime {
    static KEY: Once = Once::new();
    KEY.call_once(|| unsafe {
        std::env::set_var("SWITCHYARD_REQ4_TEST_KEY", "synthetic-decision-key")
    });
    let config = serde_json::from_value::<SwitchyardConfig>(serde_json::json!({
        "switchyard_config": {
            "schema_version": 1,
            "llm_clients": {"local": {"format": "openai_chat", "base_url": format!("{}/v1",server.uri())}},
            "targets": {"answer": {"id": "answer/model", "llm_client": "local"}, "other": {"id":"other/model","llm_client":"local"}},
            "decision_clients": {"judge": {"format":format, "endpoint":format!("{}/decisions",server.uri()), "api_key_env":"SWITCHYARD_REQ4_TEST_KEY", "timeout_ms":1000}},
            "decision_targets": {"judge": {"id":"decision/model", "decision_client":"judge"}},
            "routes": {"default": {
                "id": "switchyard/default", "type": "llm_classifier", "classifier_target":"judge",
                "strong_target":"answer", "weak_target":"other", "fail_open":false,
                "decision": {"cutoff":0.5, "apply_cutoff_to":"capable_first", "choices":[{"id":"capable_first","description":"Capable"},{"id":"efficient_first","description":"Efficient"}], "instructions":"Choose"}
            }}
        }
    })).unwrap();
    SwitchyardRuntime::new(config).unwrap()
}

fn decision_response() -> Json {
    serde_json::json!({"model":"decision/model", "answers":[{"type":"choice","name":"route","choice":"capable_first","confidence":1.0,"probabilities":[{"value":"capable_first","probability":1.0},{"value":"efficient_first","probability":0.0}]}],
        "usage":{"input_tokens":42,"output_tokens":3,"total_tokens":45,"input_tokens_details":{"cached_tokens":10,"cache_write_tokens":2},"output_tokens_details":{"reasoning_tokens":1}}})
}

#[tokio::test]
async fn decision_usage_is_exported_once_without_payload_capture() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/decisions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(decision_response()))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST")).and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"model":"answer/model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]})))
        .expect(1).mount(&server).await;
    let runtime = runtime(&server, "openai_decisions");
    let (tx, rx) = mpsc::channel();
    let capture = SpanCapture::new(None, move |span| tx.send(span).unwrap());
    let result = capture
        .capture(runtime.execute_buffered(
            WireFormat::OpenAiChat,
            Request {
                llm_request: text_request(Some("switchyard/default".into()), "private context"),
                metadata: Some(RequestMetadata {
                    session_id: Some("session-1".into()),
                    task_id: Some("task-1".into()),
                    turn_id: Some("turn-1".into()),
                    correlation_id: Some("request-1".into()),
                    ..Default::default()
                }),
                ..Default::default()
            },
        ))
        .await;
    assert!(result.result.is_ok(), "{:?}", result.result);
    let spans: Vec<_> = rx.try_iter().collect();
    let decisions: Vec<_> = spans
        .iter()
        .filter(|s| s.name == "libsy.decision_client_call")
        .collect();
    assert_eq!(decisions.len(), 1, "{spans:?}");
    let (kind, metadata, input, output) = decisions[0].projection();
    assert!(matches!(kind, ScopeType::Llm));
    assert_eq!(metadata["switchyard.accounting.unit"], "decision_call");
    assert_eq!(metadata["switchyard.accounting.schema_version"], 1);
    assert_eq!(metadata["switchyard.accounting.usage_status"], "complete");
    assert_eq!(metadata["session_id"], "session-1");
    assert_eq!(metadata["task_id"], "task-1");
    assert_eq!(metadata["turn_id"], "turn-1");
    assert_eq!(metadata["correlation_id"], "request-1");
    assert_eq!(input.unwrap()["model"], "decision/model");
    let output = output.unwrap();
    assert_eq!(output["usage"]["input_tokens"], 30);
    assert_eq!(output["usage"]["cache_read_input_tokens"], 10);
    assert_eq!(output["usage"]["cache_creation_input_tokens"], 2);
    assert_eq!(output["usage"]["output_tokens"], 3);
    assert_eq!(output["usage"]["reasoning_tokens"], 1);
    assert!(!format!("{decisions:?}").contains("private context"));
}

fn request() -> Request {
    Request {
        llm_request: text_request(Some("switchyard/default".into()), "private context"),
        ..Default::default()
    }
}

async fn mount_answer(server: &MockServer) {
    Mock::given(method("POST")).and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"model":"answer/model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]})))
        .mount(server).await;
}

#[tokio::test]
async fn system_one_missing_partial_and_zero_usage_preserve_model_fallback() {
    for (usage, status) in [
        (serde_json::json!({"provider_cost":0.0}), "missing"),
        (serde_json::json!({"input_tokens":6}), "partial"),
        (
            serde_json::json!({"cached_input_tokens":3,"output_tokens":2}),
            "partial",
        ),
        (
            serde_json::json!({"input_tokens":0,"output_tokens":0}),
            "complete",
        ),
        (
            serde_json::json!({"input_tokens":6,"output_tokens":2,"cached_input_tokens":3,"cache_creation_input_tokens":2,"cache_creation_1h_input_tokens":1,"reasoning_tokens":1}),
            "complete",
        ),
    ] {
        let server = MockServer::start().await;
        Mock::given(method("POST")).and(path("/decisions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id":"decision-id", "answers":{"route":{"type":"choice","choice":"capable_first","confidence":1.0,"probabilities":{"capable_first":1.0,"efficient_first":0.0}}},"usage":usage
            }))).expect(1).mount(&server).await;
        mount_answer(&server).await;
        let runtime = runtime(&server, "system_one");
        let (tx, rx) = mpsc::channel();
        let capture = SpanCapture::new(None, move |span| tx.send(span).unwrap());
        assert!(
            capture
                .capture(runtime.execute_buffered(WireFormat::OpenAiChat, request()))
                .await
                .result
                .is_ok()
        );
        let spans: Vec<_> = rx.try_iter().filter(|s| s.name == DECISION_CALL).collect();
        assert_eq!(spans.len(), 1);
        let (_, meta, _, output) = spans[0].projection();
        assert_eq!(meta["switchyard.accounting.usage_status"], status);
        let output = output.unwrap();
        assert_eq!(output["model"], "decision/model");
        assert_eq!(output["id"], "decision-id");
        assert_eq!(
            output["usage"].get("input_tokens"),
            usage.get("input_tokens")
        );
        if status == "missing" {
            assert_eq!(output["usage"]["cost"]["total"], 0.0);
            assert!(output["usage"].get("input_tokens").is_none());
        }
        if usage.get("cached_input_tokens").is_some() {
            assert_eq!(output["usage"]["cache_read_input_tokens"], 3);
        }
        if usage.get("cache_creation_1h_input_tokens").is_some() {
            assert_eq!(
                output["usage"]["cache_creation"]["ephemeral_1h_input_tokens"],
                1
            );
        }
    }
}

#[tokio::test]
async fn decision_failures_keep_missing_usage_without_error_bodies() {
    for (reply, error_type) in [
        (ResponseTemplate::new(503).set_body_string("private provider body"), "503"),
        (ResponseTemplate::new(200).set_body_json(serde_json::json!({"model":"decision/model","answers":[],"usage":{"input_tokens":42}})), "invalid_response"),
    ] {
        let server=MockServer::start().await;
        Mock::given(method("POST")).and(path("/decisions")).respond_with(reply).expect(1).mount(&server).await;
        let runtime=runtime(&server,"openai_decisions");
        let (tx,rx)=mpsc::channel();
        let capture=SpanCapture::new(None,move |span|tx.send(span).unwrap());
        assert!(capture.capture(runtime.execute_buffered(WireFormat::OpenAiChat,request())).await.result.is_err());
        let spans:Vec<_>=rx.try_iter().collect();
        let [decision]=spans.as_slice() else {panic!("one decision span expected: {spans:?}")};
        let (_,meta,_,output)=decision.projection();
        assert_eq!(meta["outcome"],"error");
        assert_eq!(meta["error.type"],error_type);
        assert_eq!(meta["switchyard.accounting.usage_status"],"missing");
        assert!(output.unwrap().get("usage").is_none());
        assert!(!format!("{spans:?}").contains("private provider body"));
    }
}

#[tokio::test]
async fn dropping_in_flight_host_work_exports_one_cancelled_decision() {
    let server = MockServer::start().await;
    let started = std::sync::Arc::new(tokio::sync::Notify::new());
    let seen = started.clone();
    Mock::given(method("POST"))
        .and(path("/decisions"))
        .respond_with(move |_: &wiremock::Request| {
            seen.notify_one();
            ResponseTemplate::new(200)
                .set_body_json(decision_response())
                .set_delay(std::time::Duration::from_secs(5))
        })
        .mount(&server)
        .await;
    let runtime = runtime(&server, "openai_decisions");
    let (tx, rx) = mpsc::channel();
    let capture = SpanCapture::new(None, move |span| tx.send(span).unwrap());
    let mut execution =
        Box::pin(capture.capture(runtime.execute_buffered(WireFormat::OpenAiChat, request())));
    tokio::select! {
        _=started.notified()=>{},
        _=&mut execution=>panic!("decision should remain in flight"),
        _=tokio::time::sleep(std::time::Duration::from_secs(2))=>panic!("decision did not start"),
    }
    drop(execution);
    let spans: Vec<_> = rx.try_iter().collect();
    let [decision] = spans.as_slice() else {
        panic!("one cancelled span expected: {spans:?}")
    };
    assert_eq!(decision.name, DECISION_CALL);
    let (_, meta, _, _) = decision.projection();
    assert_eq!(meta["outcome"], "cancelled");
    assert_eq!(meta["switchyard.accounting.usage_status"], "missing");
    assert_eq!(meta["otel.status_code"], "ERROR");
}
