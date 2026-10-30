// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::json;
use switchyard_protocol::{Request, text_request};
use switchyard_runner::Runner;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn profiles_use_their_own_judges_and_candidate_prompts()
-> Result<(), Box<dyn std::error::Error>> {
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
classify_trigger = "user_turn"
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
            .respond_with(ResponseTemplate::new(status).set_body_json(json!({
                "id": "response", "object": "chat.completion", "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]
            }))).mount(&server).await;
        }
        let result = runner
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
        assert_eq!(
            result.response.served_model().map(|model| model.as_str()),
            Some("strong")
        );
        let requests = server.received_requests().await.unwrap();
        let bodies: Vec<serde_json::Value> = requests
            .iter()
            .map(|request| serde_json::from_slice(&request.body))
            .collect::<Result<_, _>>()?;
        assert_eq!(
            bodies
                .iter()
                .filter(|body| body["model"] == "inner")
                .count(),
            usize::from(inner_score.is_some())
        );
        assert_eq!(
            bodies.iter().any(|body| body["model"] == "weak"),
            inner_score != Some(0.0)
        );
        for body in bodies {
            let expected = match body["model"].as_str() {
                Some("weak") => "EFFICIENT PROMPT",
                Some("strong") => "CAPABLE PROMPT",
                _ => continue,
            };
            assert_eq!(body["messages"][0]["content"], expected);
        }
    }
    Ok(())
}
