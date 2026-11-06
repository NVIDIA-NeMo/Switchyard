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

#[tokio::test]
async fn composite_defaults_select_profiles_and_preserve_overrides()
-> Result<(), Box<dyn std::error::Error>> {
    for (score, overrides, expected_model, expected_prompt) in [
        (0.92, "", "weak", None),
        (0.93, "", "strong", Some("You are in the planning phase.")),
        (
            0.8,
            "[routes.auto.classifier.decision]\ncutoff = 0.7",
            "strong",
            Some("You are in the planning phase."),
        ),
        (
            0.93,
            "[routes.auto.capable]\ncapable_system_prompt = 'Custom plan'",
            "strong",
            Some("Custom plan"),
        ),
        (
            0.93,
            "[routes.auto.stage]\ncapable_target = 'strong'\nefficient_target = 'weak'\ncapable_system_prompt = ''",
            "strong",
            None,
        ),
    ] {
        let server = MockServer::start().await;
        let mut source = format!(
            r#"
schema_version = 1
[llm_clients.local]
format = "openai_chat"
base_url = "{}/v1"
[targets]
strong = {{ id = "strong", llm_client = "local" }}
weak = {{ id = "weak", llm_client = "local" }}
[decision_clients.judge]
format = "openai_decisions"
endpoint = "{}/v1/decisions"
api_key_env = "SWITCHYARD_COMPOSITE_DEFAULTS_TEST_KEY"
timeout_ms = 2000
[decision_targets.judge]
id = "judge"
decision_client = "judge"
[routes.auto]
id = "auto"
type = "composite"
[routes.auto.classifier]
target = "judge"
classify_trigger = "user_turn"
"#,
            server.uri(),
            server.uri()
        );
        if !overrides.starts_with("[routes.auto.stage]") {
            source.push_str(
                "[routes.auto.stage]\ncapable_target = 'strong'\nefficient_target = 'weak'\n",
            );
        }
        source.push_str(overrides);
        unsafe {
            std::env::set_var("SWITCHYARD_COMPOSITE_DEFAULTS_TEST_KEY", "test-key");
        }
        let runner = Runner::from_toml(&source);
        unsafe {
            std::env::remove_var("SWITCHYARD_COMPOSITE_DEFAULTS_TEST_KEY");
        }
        let runner = runner?;
        Mock::given(method("POST"))
            .and(path("/v1/decisions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "model": "judge", "usage": {"input_tokens": 1, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0}, "output_tokens": 1, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 2}, "answers": [{
                    "type": "choice", "name": "route", "choice": "capable", "confidence": score,
                    "probabilities": [{"value": "capable", "probability": score},
                        {"value": "efficient", "probability": 1.0 - score}]
                }]
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST")).and(path("/v1/chat/completions"))
            .and(body_partial_json(json!({"model": expected_model})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "answer", "object": "chat.completion", "model": expected_model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Answer"}, "finish_reason": "stop"}]
            }))).expect(1).mount(&server).await;
        runner
            .route("auto")
            .unwrap()
            .execute(
                Request {
                    llm_request: text_request(None, "Implement the task"),
                    ..Default::default()
                },
                None,
            )
            .await?;
        let requests = server.received_requests().await.unwrap();
        let completion: serde_json::Value = requests
            .iter()
            .find(|r| r.url.path() == "/v1/chat/completions")
            .unwrap()
            .body_json()?;
        let messages = completion["messages"].as_array().unwrap();
        if let Some(prompt) = expected_prompt {
            assert_eq!(messages[0]["role"], "system");
            assert!(messages[0]["content"].as_str().unwrap().starts_with(prompt));
        } else {
            assert!(
                messages
                    .iter()
                    .all(|m| m["role"] != "system" || m["content"] == "")
            );
        }
        server.verify().await;
    }
    Ok(())
}
