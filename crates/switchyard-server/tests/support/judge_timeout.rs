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
{classifier}
base_threshold = 0.5
timeout_ms = 75
"#,
        streaming = scenario == "streaming",
    )
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
async fn late_judge_response_does_not_abort_the_next_judge() -> TestResult {
    use libsy::{
        ClassifyTrigger, CompositeRouter, CompositeRouterConfig, LlmFallback, PickerMode,
        StageRouterConfig, TaskClassifierConfig,
    };
    use switchyard_protocol::{Request, text_request};

    let upstream = MockUpstream::start().await?;
    let mut stage = StageRouterConfig::new(PickerMode::CapableFirst, 1.0);
    stage.llm_fallback = Some(LlmFallback {
        config: TaskClassifierConfig {
            timeout_ms: 750,
            ..Default::default()
        },
    });
    let algorithm = Arc::new(CompositeRouter::new(CompositeRouterConfig {
        judge: TaskClassifierConfig {
            classify_trigger: ClassifyTrigger::UserTurn,
            timeout_ms: 75,
            ..Default::default()
        },
        stage,
    })?);
    let backend = Backend::OpenAiChat(HttpBackendConfig {
        base_url: upstream.base_url.clone(),
        api_key: None,
        forward_auth: false,
        extra_headers: BTreeMap::new(),
        extra_body: BTreeMap::new(),
        reasoning_effort: None,
        max_retries: 0,
    });
    let client = TranslatingLlmClient::new(&[
        ModelConfig::new("judge/late", backend.clone(), None),
        ModelConfig::new("late/weak", backend.clone(), None),
        ModelConfig::new("late/strong", backend, None),
    ])?;
    let models = RuntimeModels::new(
        [
            (Category::Judge, vec!["judge/late".into()]),
            (Category::Efficient, vec!["late/weak".into()]),
            (Category::Capable, vec!["late/strong".into()]),
            (
                Category::Any,
                vec!["late/strong".into(), "late/weak".into()],
            ),
        ]
        .into(),
    );
    let (selected, response) = tokio::time::timeout(
        Duration::from_secs(1),
        switchyard_llm_client::run(
            algorithm,
            ClientRouter::single(Arc::new(client)),
            Request {
                llm_request: text_request(Some("late".into()), "bounded task"),
                raw_request: None,
                metadata: None,
            },
            Arc::new(models),
            None,
        ),
    )
    .await??;
    assert_eq!(selected, "late/weak");
    assert_eq!(
        switchyard_protocol::completion_text(&response.llm_response.into_agg().await?),
        "ok"
    );
    assert_eq!(
        upstream.models().await,
        ["judge/late", "judge/late", "late/weak"]
    );
    Ok(())
}
