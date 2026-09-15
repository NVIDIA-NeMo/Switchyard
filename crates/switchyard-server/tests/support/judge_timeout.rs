// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test judge deadlines with delayed replies, stalled streams, and HTTP retries.

use std::time::Duration;

use super::*;

fn deadline_config(base_url: &str, mode: &str, scenario: &str) -> String {
    let classifier = match mode {
        "stage" => {
            r#"
type = "stage_router"
capable_target = "strong"
efficient_target = "weak"
picker = "capable_first"
confidence_threshold = 1.0
[routes.deadline.classifier]
target = "judge"
"#
        }
        _ => {
            r#"
type = "llm_classifier"
classifier_target = "judge"
strong_target = "strong"
weak_target = "weak"
"#
        }
    };
    format!(
        r#"
schema_version = 1
[llm_clients.upstream]
format = "openai_chat"
base_url = "{base_url}"
max_retries = 2
[targets]
judge = {{ id = "judge/{mode}/{scenario}", llm_client = "upstream", extra_body = {{ scenario = "{scenario}", stream = {streaming} }} }}
strong = {{ id = "deadline/strong", llm_client = "upstream" }}
weak = {{ id = "deadline/weak", llm_client = "upstream" }}
[routes.deadline]
id = "deadline"
judge_timeout_ms = 75
{classifier}
base_threshold = 0.5
"#,
        streaming = scenario == "streaming",
    )
}

#[test]
fn judge_deadline_configuration_is_validated() -> TestResult {
    let config = deadline_config("http://127.0.0.1:9/v1", "classifier", "valid");
    for setting in ["", "judge_timeout_ms = 1", "judge_timeout_ms = 10000"] {
        load_test_config(&config.replace("judge_timeout_ms = 75", setting))?;
    }
    for setting in [
        "judge_timeout_ms = 0",
        "judge_timeout_ms = -1",
        "judge_timeout_ms = 1.5",
        "judge_timeout_ms = '75'",
        "judge_timeout_ms = 75\njudge_timeout_ms = 100",
        "judge_timeout_mss = 75",
        "timeout_ms = 75",
    ] {
        let result = load_test_config(&config.replace("judge_timeout_ms = 75", setting));
        assert!(result.is_err(), "{setting}");
        if setting == "judge_timeout_ms = 0" {
            assert!(result.is_err_and(|error| {
                error
                    .to_string()
                    .contains("judge_timeout_ms must be at least 1")
            }));
        }
    }
    Ok(())
}

#[tokio::test]
async fn configured_judge_deadline_covers_response_body_and_retries() -> TestResult {
    let upstream = MockUpstream::start().await?;
    for (mode, scenario, path) in [
        ("classifier", "streaming", "/v1/chat/completions"),
        ("classifier", "streaming", "/v1/messages"),
        ("classifier", "streaming", "/v1/responses"),
        ("stage", "streaming", "/v1/chat/completions"),
        ("classifier", "buffered", "/v1/chat/completions"),
        ("classifier", "retry", "/v1/chat/completions"),
        ("classifier", "valid", "/v1/chat/completions"),
        ("stage", "valid", "/v1/chat/completions"),
    ] {
        let app = build_switchyard_router(load_test_config(&deadline_config(
            &upstream.base_url,
            mode,
            scenario,
        ))?);
        let mut body = json!({"model": "deadline", "max_tokens": 20});
        if path == "/v1/responses" {
            body["input"] = json!("bounded task");
        } else {
            body["messages"] = json!([{"role": "user", "content": "bounded task"}]);
        }
        upstream.calls.lock().await.clear();
        let before = send(&app, "GET", "/metrics", None).await?;
        let response =
            tokio::time::timeout(Duration::from_secs(1), send(&app, "POST", path, Some(body)))
                .await??;
        let selected = if scenario == "valid" {
            "deadline/weak"
        } else {
            "deadline/strong"
        };
        assert_eq!(response.status, StatusCode::OK);
        assert_eq!(response.headers["x-model-router-selected-model"], selected);
        let content = match path {
            "/v1/messages" => "/content/0/text",
            "/v1/responses" => "/output/0/content/0/text",
            _ => "/choices/0/message/content",
        };
        assert_eq!(response.json()?.pointer(content), Some(&json!("ok")));
        assert_eq!(
            upstream.models().await,
            [format!("judge/{mode}/{scenario}"), selected.into()]
        );
        let after = send(&app, "GET", "/metrics", None).await?;
        let delta = metric_delta(
            before.text()?,
            after.text()?,
            "switchyard_classifier_fail_open_total",
            &[
                ("judge_model", &format!("judge/{mode}/{scenario}")),
                ("reason", "timeout"),
            ],
        )
        .unwrap_or_default();
        assert_eq!(delta, if scenario == "valid" { 0.0 } else { 1.0 });
    }
    Ok(())
}

#[tokio::test]
async fn judge_deadline_does_not_limit_an_answer_from_the_same_model() -> TestResult {
    let upstream = MockUpstream::start().await?;
    let config = format!(
        r#"
schema_version = 1
[llm_clients.upstream]
format = "openai_chat"
base_url = "{}"
max_retries = 0
[targets]
shared = {{ id = "deadline/shared", llm_client = "upstream" }}
weak = {{ id = "deadline/weak", llm_client = "upstream" }}
[routes.deadline]
id = "deadline"
type = "llm_classifier"
classifier_target = "shared"
strong_target = "shared"
weak_target = "weak"
base_threshold = 0.5
judge_timeout_ms = 75
"#,
        upstream.base_url,
    );
    let app = build_switchyard_router(load_test_config(&config)?);
    let response = tokio::time::timeout(
        Duration::from_secs(1),
        send(
            &app,
            "POST",
            "/v1/chat/completions",
            Some(json!({
                "model": "deadline",
                "messages": [{"role": "user", "content": "bounded task"}],
            })),
        ),
    )
    .await??;
    assert_eq!(response.status, StatusCode::OK);
    assert_eq!(response.json()?["choices"][0]["message"]["content"], "ok");
    assert_eq!(
        upstream.models().await,
        ["deadline/shared", "deadline/shared"]
    );
    Ok(())
}
