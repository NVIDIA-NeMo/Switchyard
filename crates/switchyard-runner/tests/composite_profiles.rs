// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::json;
use switchyard_protocol::{Request, text_request};
use switchyard_runner::Runner;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn profile_judges_and_fallback_prompts() -> Result<(), Box<dyn std::error::Error>> {
    for (profile, inner_score) in [
        ("efficient", None),
        ("efficient", Some(0.0)),
        ("capable", Some(1.0)),
    ] {
        let server = MockServer::start().await;
        let mut source = format!(
            r#"
schema_version = 1
[llm_clients.local]
format = "openai_chat"
base_url = "{}/v1"
max_retries = 0
[targets]
judge = {{ id = "judge", llm_client = "local" }}
strong = {{ id = "strong", llm_client = "local" }}
weak = {{ id = "weak", llm_client = "local" }}
inner = {{ id = "inner", llm_client = "local" }}
[routes.auto]
id = "auto"
type = "composite"
[routes.auto.classifier]
target = "judge"
base_threshold = 0.5
classify_trigger = "new_session"
[routes.auto.stage]
capable_target = "strong"
efficient_target = "weak"
[routes.auto.{profile}]
capable_system_prompt = "CAPABLE PROMPT"
efficient_system_prompt = "EFFICIENT PROMPT"
"#,
            server.uri()
        );
        if inner_score.is_some() {
            source.push_str(&format!(
                "\n[routes.auto.{profile}.classifier]\ntarget = 'inner'\nbase_threshold = 0.5\n"
            ));
        }
        let runner = Runner::from_toml(&source)?;
        let verdict = |score| {
            format!(
                r#"{{"crux":"task","primary_rule":"SUP-1","capability_boundary":"supported","p_solve":{score}}}"#
            )
        };
        let outer = verdict(if profile == "efficient" { 1.0 } else { 0.0 });
        let inner = verdict(inner_score.unwrap_or(0.0));
        for (model, status, content) in [
            ("judge", 200, outer.as_str()),
            ("inner", 200, inner.as_str()),
            ("weak", 503, "unavailable"),
            ("strong", 200, "answer"),
        ] {
            Mock::given(method("POST")).and(path("/v1/chat/completions"))
            .and(body_partial_json(json!({"model": model})))
            .and(move |request: &wiremock::Request| {
                let body: serde_json::Value = request.body_json().unwrap();
                match model {
                    "weak" => body["messages"][0]["content"] == "EFFICIENT PROMPT",
                    "strong" => body["messages"][0]["content"] == "CAPABLE PROMPT",
                    _ => true,
                }
            })
            .respond_with(ResponseTemplate::new(status).set_body_json(json!({
                "id": "response", "object": "chat.completion", "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]
            })))
            .expect(match model {
                "inner" => u64::from(inner_score.is_some()),
                "weak" => u64::from(inner_score != Some(0.0)),
                _ => 1,
            }).mount(&server).await;
        }
        runner
            .route("auto")
            .unwrap()
            .execute(
                Request {
                    llm_request: text_request(None, "Help with this task"),
                    ..Default::default()
                },
                None,
            )
            .await?;
        server.verify().await;
    }
    Ok(())
}
