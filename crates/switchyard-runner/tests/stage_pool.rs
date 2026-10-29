// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::{Value, json};
use switchyard_protocol::{Request, text_request};
use switchyard_runner::Runner;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn categorical_judge_selects_every_anchor_and_invalid_verdict_falls_open()
-> Result<(), Box<dyn std::error::Error>> {
    let server = MockServer::start().await;
    let endpoint = server.uri();
    let config = format!(
        r#"
schema_version = 1
[llm_clients.local]
format = "openai_chat"
base_url = "{endpoint}/v1"
[targets]
small = {{ id = "small", llm_client = "local" }}
medium = {{ id = "medium", llm_client = "local" }}
large = {{ id = "large", llm_client = "local" }}
largest = {{ id = "largest", llm_client = "local" }}
judge = {{ id = "judge", llm_client = "local" }}
[routes.stage]
id = "stage"
type = "stage_router"
efficient_target = "small"
capable_target = "largest"
picker = "efficient_first"
confidence_threshold = 1.0
capable_hold_turns = 0
[routes.stage.pool]
targets = ["small", "medium", "large", "largest"]
positions = [0.0, 0.2, 0.8, 1.0]
[routes.stage.classifier]
target = "judge"
"#
    );
    let runner = Runner::from_toml(&config)?;
    for (choice, expected) in [
        ("target_0", "small"),
        ("target_1", "medium"),
        ("target_2", "large"),
        ("target_3", "largest"),
        ("unknown", "small"),
    ] {
        server.reset().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_partial_json(json!({"model":"judge"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(completion(
                "judge",
                &json!({"target":choice,"reason":"fixture"}).to_string(),
            )))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_partial_json(json!({"model":expected})))
            .respond_with(ResponseTemplate::new(200).set_body_json(completion(expected, "answer")))
            .expect(1)
            .mount(&server)
            .await;
        let output = runner
            .route("stage")
            .unwrap()
            .execute(
                Request {
                    llm_request: text_request(None, "Implement and test this task"),
                    ..Request::default()
                },
                None,
            )
            .await?;
        assert_eq!(output.selected_model, expected);
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 2, "one judge call and one answer");
        let body: Value = serde_json::from_slice(&requests[0].body)?;
        assert_eq!(
            body["response_format"]["json_schema"]["schema"]["properties"]["target"]["enum"],
            json!(["target_0", "target_1", "target_2", "target_3"])
        );
        server.verify().await;
    }
    Ok(())
}

fn completion(model: &str, text: &str) -> Value {
    json!({"id":"test","object":"chat.completion","model":model,
        "choices":[{"index":0,"message":{"role":"assistant","content":text},"finish_reason":"stop"}]})
}
