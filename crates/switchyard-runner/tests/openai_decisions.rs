// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, mpsc};

use serde_json::{Value, json};
use switchyard_llm_client::RunObservation;
use switchyard_protocol::{Request, Usage, text_request};
use switchyard_runner::Runner;
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
cutoff = 0.7
candidates = {{ a = "strong", b = "weak" }}
instructions = {{ task = "Choose advantage only when a succeeds and b fails." }}
evidence = {{ description = "Compare a and b" }}
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
        "instructions": json!({"task": "Choose advantage only when a succeeds and b fails."}).to_string(),
        "choices": [{"value": "advantage"}, {"value": "no_advantage"}]
    });
    Mock::given(method("POST")).and(path("/v1/decisions"))
        .and(header("authorization", "Bearer test-key"))
        .and(body_partial_json(json!({"model": "gpt-6-luna", "questions": [question]})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "gpt-6-luna", "answers": [{
                "type": "choice", "name": "route", "choice": "advantage", "confidence": 0.9,
                "probabilities": [{"value": "advantage", "probability": 0.6},
                    {"value": "no_advantage", "probability": 0.4}]
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
        json!({"description": "Compare a and b"})
    );
    server.verify().await;
    Ok(())
}
