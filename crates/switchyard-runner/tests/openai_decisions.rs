// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, mpsc};

use serde_json::{Value, json};
use switchyard_llm_client::RunObservation;
use switchyard_protocol::{Metadata, Request, Usage, text_request};
use switchyard_runner::{AlgorithmFactory, Runner};
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn openai_decisions_routes_from_toml() -> Result<(), Box<dyn std::error::Error>> {
    let server = MockServer::start().await;
    let endpoint = server.uri();
    let source = format!(
        r#"
schema_version = 1
[llm_clients.answers]
format = "openai_chat"
base_url = "{endpoint}/v1"
[targets]
strong = {{ id = "strong/model", llm_client = "answers" }}
weak = {{ id = "weak/model", llm_client = "answers" }}
[decision_clients.judge]
format = "openai_decisions"
endpoint = "{endpoint}/v1/decisions"
api_key_env = "SWITCHYARD_OPENAI_DECISIONS_TEST_KEY"
timeout_ms = 2000
[decision_targets.judge]
id = "gpt-6-luna"
decision_client = "judge"
[routes.classifier]
id = "switchyard/classifier"
type = "llm_classifier"
classifier_target = "judge"
strong_target = "strong"
weak_target = "weak"
fail_open = false
[routes.classifier.decision]
cutoff = 0.92
candidates = {{ capable_first = "strong", efficient_first = "weak" }}
criteria = {{ capable_first = "Start capable", efficient_first = "Start efficient" }}
instructions = {{ task = "Choose a Stage profile." }}
evidence = {{ description = "Compare the two profiles" }}
"#
    );
    // This unique variable is only read while constructing this client.
    unsafe {
        std::env::set_var("SWITCHYARD_OPENAI_DECISIONS_TEST_KEY", "test-key");
    }
    let runner = Runner::from_toml(&source);
    unsafe {
        std::env::remove_var("SWITCHYARD_OPENAI_DECISIONS_TEST_KEY");
    }
    let runner = runner?;
    let question = json!({
        "name": "route", "type": "choice",
        "instructions": json!({"task": "Choose a Stage profile."}).to_string(),
        "choices": [{"value": "capable_first", "description": "Start capable"},
            {"value": "efficient_first", "description": "Start efficient"}]
    });
    Mock::given(method("POST")).and(path("/v1/decisions"))
        .and(header("authorization", "Bearer test-key"))
        .and(body_partial_json(json!({"model": "gpt-6-luna", "questions": [question]})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "gpt-6-luna", "answers": [{
                "type": "choice", "name": "route", "choice": "capable_first", "confidence": 0.99,
                "probabilities": [{"value": "capable_first", "probability": 0.91},
                    {"value": "efficient_first", "probability": 0.09}]
            }],
            "usage": {"input_tokens": 42, "input_tokens_details": {"cached_tokens": 10, "cache_write_tokens": 2},
                "output_tokens": 3, "output_tokens_details": {"reasoning_tokens": 1}, "total_tokens": 45}
        }))).expect(1).mount(&server).await;
    Mock::given(method("POST")).and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({"model": "weak/model"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "answer-1", "object": "chat.completion", "model": "weak/model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Answer"}, "finish_reason": "stop"}]
        }))).expect(1).mount(&server).await;
    let (sender, observations) = mpsc::channel();
    let output = runner
        .route("switchyard/classifier")
        .unwrap()
        .execute(
            Request {
                llm_request: text_request(None, "Route this request"),
                metadata: Some(Metadata {
                    session_id: Some("session-1".into()),
                    ..Metadata::default()
                }),
                ..Request::default()
            },
            Some(Arc::new(move |event| {
                sender.send(event).unwrap();
            })),
        )
        .await?;
    assert_eq!(output.selected_model, "weak/model");
    let usage = observations.try_iter().find_map(|event| match event {
        RunObservation::DecisionCall(call) => call.usage,
        _ => None,
    });
    assert_eq!(
        usage,
        Some(Usage {
            input_tokens: Some(30),
            cache: Usage::cache_details(Some(10), Some(2)),
            output_tokens: Some(3),
            total_tokens: Some(45),
            reasoning_tokens: Some(1),
            ..Usage::default()
        })
    );
    let requests = server.received_requests().await.unwrap();
    let decision: Value = serde_json::from_slice(&requests[0].body)?;
    let context: Value = serde_json::from_str(decision["input"].as_str().unwrap())?;
    assert_eq!(
        context["evidence"],
        json!({"description": "Compare the two profiles"})
    );
    assert_eq!(context["session"], "session-1");
    server.verify().await;
    Ok(())
}

fn direct_config(endpoint: &str) -> String {
    format!(
        r#"
schema_version = 1
[llm_clients.answers]
format = "openai_chat"
base_url = "{endpoint}/v1"
[targets]
fast = {{ id = "fast/model", llm_client = "answers" }}
balanced = {{ id = "balanced/model", llm_client = "answers" }}
premium = {{ id = "premium/model", llm_client = "answers" }}
[decision_clients.router]
format = "openai_decisions"
endpoint = "{endpoint}/v1/decisions"
api_key_env = "SWITCHYARD_DIRECT_DECISIONS_TEST_KEY"
timeout_ms = 2000
[decision_targets.router]
id = "router-model"
decision_client = "router"
[routes.direct]
id = "switchyard/direct"
type = "llm_classifier"
mode = "direct"
default_target = "fast"
[routes.direct.models]
judge = ["router"]
any = ["fast", "balanced", "premium"]
fast = ["fast"]
balanced = ["balanced"]
premium = ["premium"]
unused = []
"#
    )
}

#[tokio::test]
async fn direct_mode_serves_the_selected_group() -> Result<(), Box<dyn std::error::Error>> {
    let server = MockServer::start().await;
    // This unique variable is only read while constructing this client.
    unsafe {
        std::env::set_var("SWITCHYARD_DIRECT_DECISIONS_TEST_KEY", "test-key");
    }
    let runner = Runner::from_toml(&direct_config(&server.uri()));
    unsafe {
        std::env::remove_var("SWITCHYARD_DIRECT_DECISIONS_TEST_KEY");
    }
    let runner = runner?;
    // The choice routes even though another option has the higher probability.
    Mock::given(method("POST")).and(path("/v1/decisions"))
        .and(body_partial_json(json!({"model": "router-model", "questions": [{
            "name": "route", "type": "choice",
            "choices": [{"value": "balanced"}, {"value": "fast"}, {"value": "premium"}]
        }]})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "router-model", "answers": [{
                "type": "choice", "name": "route", "choice": "premium",
                "probabilities": [{"value": "balanced", "probability": 0},
                    {"value": "fast", "probability": 1}, {"value": "premium", "probability": 0}]
            }],
            "usage": {"input_tokens": 0, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens": 0, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 0}
        }))).expect(1).mount(&server).await;
    Mock::given(method("POST")).and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({"model": "premium/model"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "answer-1", "object": "chat.completion", "model": "premium/model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Answer"}, "finish_reason": "stop"}]
        }))).expect(1).mount(&server).await;
    let output = runner
        .route("switchyard/direct")
        .unwrap()
        .execute(
            Request {
                llm_request: text_request(None, "Route this request"),
                metadata: Some(Metadata {
                    session_id: Some("session-1".into()),
                    ..Metadata::default()
                }),
                ..Request::default()
            },
            None,
        )
        .await?;
    assert_eq!(output.selected_model, "premium/model");
    let requests = server.received_requests().await.unwrap();
    let decision: Value = serde_json::from_slice(&requests[0].body)?;
    let context: Value = serde_json::from_str(decision["input"].as_str().unwrap())?;
    assert_eq!(
        context,
        json!({
            "task": [{"role": "user", "content": [{"type": "text", "text": "Route this request"}]}],
            "session": "session-1",
        })
    );
    server.verify().await;
    Ok(())
}

#[test]
fn direct_mode_rejects_invalid_configuration() {
    for (from, to, expected) in [
        (
            r#"judge = ["router"]"#,
            r#"judge = ["fast"]"#,
            "requires target kind decision",
        ),
        (
            "balanced = [\"balanced\"]\npremium = [\"premium\"]\n",
            "",
            "at least two options",
        ),
        (
            r#"default_target = "fast""#,
            r#"default_target = "any""#,
            "must be one of the options",
        ),
    ] {
        let source = direct_config("http://127.0.0.1:1").replace(from, to);
        let error = AlgorithmFactory::from_toml(&source)
            .and_then(|factory| factory.build())
            .err()
            .unwrap_or_else(|| panic!("expected a load error for {to}"))
            .to_string();
        assert!(error.contains(expected), "{to}: {error}");
    }
}
