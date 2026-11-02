// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::{Value, json};
use switchyard_protocol::{
    ContentBlock, Message, Metadata, Request, Role, ToolCall, ToolResult, text_request,
};
use switchyard_runner::{AlgorithmFactory, Runner};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const CONFIG: &str = include_str!("../../../examples/auto.toml");

#[test]
fn auto_validates_bindings_and_rejects_tuning() {
    let algorithms = AlgorithmFactory::from_toml(CONFIG)
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(algorithms["switchyard/auto"].0.name(), "composite");
    for (source, expected) in [
        (
            CONFIG.replace(
                "classifier_target = \"decider\"",
                "classifier_target = \"capable\"",
            ),
            "requires target kind decision",
        ),
        (
            CONFIG.replace(
                "capable_target = \"capable\"",
                "capable_target = \"missing\"",
            ),
            "unknown target missing",
        ),
        (
            format!("{CONFIG}\nconfidence_threshold = 0.5\n"),
            "unknown field",
        ),
    ] {
        let error = AlgorithmFactory::from_toml(&source)
            .and_then(|factory| factory.build())
            .err()
            .unwrap();
        assert!(error.to_string().contains(expected), "{error}");
    }
}

fn tool_step(request: &mut Request, name: &str, failed: bool) {
    let id = format!("call_{}", request.llm_request.messages.len());
    request.llm_request.messages.extend([
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: id.clone(),
                name: name.into(),
                arguments: json!({"command": "cargo test"}),
            })],
        },
        Message {
            role: Role::Tool,
            content: vec![ContentBlock::ToolResult(ToolResult {
                tool_call_id: id,
                content: vec![ContentBlock::Text {
                    text: if failed {
                        "fatal runtime error: out of memory"
                    } else {
                        "ok"
                    }
                    .into(),
                }],
                is_error: Some(failed),
            })],
        },
    ]);
}

#[tokio::test]
async fn auto_selects_once_and_preserves_each_profiles_behavior()
-> Result<(), Box<dyn std::error::Error>> {
    for (score, capable_first) in [(0.92, false), (0.93, true)] {
        let server = MockServer::start().await;
        let source = CONFIG
            .replace("http://localhost:8000", &server.uri())
            .replace("http://localhost:8001", &server.uri())
            .replace("api_key_env = \"MODEL_API_KEY\"", "")
            .replace("DECISION_API_KEY", "SWITCHYARD_AUTO_TEST_KEY");
        unsafe {
            std::env::set_var("SWITCHYARD_AUTO_TEST_KEY", "test-key");
        }
        let runner = Runner::from_toml(&source);
        unsafe {
            std::env::remove_var("SWITCHYARD_AUTO_TEST_KEY");
        }
        let runner = runner?;
        Mock::given(method("POST")).and(path("/v1/decisions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "model": "openai/openai/gpt-6-luna", "answers": [{
                    "type": "choice", "name": "route", "choice": "capable_first",
                    "probabilities": [{"value": "capable_first", "probability": score},
                        {"value": "efficient_first", "probability": 1.0 - score}]
                }],
                "usage": {"input_tokens": 42, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                    "output_tokens": 3, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 45}
            }))).expect(1).mount(&server).await;
        Mock::given(method("POST")).and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "answer", "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Answer"}, "finish_reason": "stop"}]
            }))).expect(4).mount(&server).await;
        let mut request = Request {
            llm_request: text_request(None, "fix the build"),
            metadata: Some(Metadata {
                session_id: Some("task".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        for step in 0..4 {
            match step {
                1 => tool_step(&mut request, "Write", false),
                2 => tool_step(&mut request, "Bash", true),
                3 => request
                    .llm_request
                    .messages
                    .push(Message::text(Role::User, "Continue fixing the build")),
                _ => {}
            }
            let output = runner
                .route("switchyard/auto")
                .unwrap()
                .execute(request.clone(), None)
                .await?;
            let capable = if capable_first { step == 0 } else { step >= 2 };
            assert_eq!(
                output.selected_model,
                if capable {
                    "openai/openai/gpt-6-astra"
                } else {
                    "openai/openai/gpt-6-luna"
                }
            );
        }
        let requests = server.received_requests().await.unwrap();
        let first: Value = serde_json::from_slice(&requests[1].body)?;
        let has_plan = first["messages"]
            .to_string()
            .contains("You are in the planning phase.");
        assert_eq!(has_plan, capable_first);
        server.verify().await;
    }
    Ok(())
}
