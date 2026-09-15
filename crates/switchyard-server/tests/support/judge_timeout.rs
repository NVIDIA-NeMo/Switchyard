// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test judge deadlines with delayed replies, stalled streams, and HTTP retries.

use std::time::Duration;

use super::*;

async fn judge_response(
    State(calls): State<Arc<Mutex<Vec<Value>>>>,
    Json(body): Json<Value>,
) -> HttpResponse {
    let call_count = {
        let mut calls = calls.lock().await;
        calls.push(body.clone());
        calls.len()
    };
    match body["scenario"].as_str() {
        Some("buffered") => tokio::time::sleep(Duration::from_secs(2)).await,
        Some("late") => {
            let delay = if call_count == 1 { 150 } else { 250 };
            tokio::time::sleep(Duration::from_millis(delay)).await;
        }
        Some("streaming") => {
            let stream = async_stream::stream! {
                yield Ok::<Event, Infallible>(Event::default().data(json!({
                    "id": "judge-stream",
                    "model": body["model"],
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": "{"}}]
                }).to_string()));
                tokio::time::sleep(Duration::from_secs(2)).await;
                yield Ok(Event::default().data("[DONE]"));
            };
            return Sse::new(stream).into_response();
        }
        Some("retry") => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                [("retry-after", "1")],
                Json(json!({"error": {"message": "unavailable"}})),
            )
                .into_response();
        }
        _ => {}
    }
    let content = if body.get("scenario").is_some() {
        r#"{"crux":"bounded task","primary_rule":"SUP-1","capability_boundary":"supported","p_solve":0.9}"#
    } else {
        "ok"
    };
    Json(json!({
        "id": "judge-timeout",
        "model": body["model"],
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    }))
    .into_response()
}

// Start a local judge provider. `MockUpstream::drop` aborts the server task,
// including when a test fails.
async fn deadline_upstream() -> TestResult<MockUpstream> {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .route("/v1/chat/completions", post(judge_response))
        .with_state(Arc::clone(&calls));
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let task = tokio::spawn(async move {
        let result = axum::serve(listener, app).await;
        assert!(result.is_ok(), "{result:?}");
    });
    Ok(MockUpstream {
        base_url: format!("http://{address}/v1"),
        calls,
        task,
    })
}

fn deadline_config(base_url: &str, mode: &str, scenario: &str, timeout_ms: &str) -> String {
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
[targets.judge]
id = "judge/{mode}/{scenario}"
llm_client = "upstream"
extra_body = {{ scenario = "{scenario}", stream = {streaming} }}
[targets.strong]
id = "deadline/strong"
llm_client = "upstream"
[targets.weak]
id = "deadline/weak"
llm_client = "upstream"
[routes.deadline]
id = "deadline"
{classifier}
base_threshold = 0.5
timeout_ms = {timeout_ms}
"#,
        streaming = scenario == "streaming",
    )
}

#[tokio::test]
async fn configured_judge_deadline_covers_response_body_and_retries() -> TestResult {
    let upstream = deadline_upstream().await?;
    for mode in ["classifier", "stage"] {
        for scenario in ["buffered", "streaming", "retry", "valid"] {
            let app = build_switchyard_router(load_test_config(&deadline_config(
                &upstream.base_url,
                mode,
                scenario,
                "75",
            ))?);
            for (path, mut body) in [
                (
                    "/v1/chat/completions",
                    json!({"messages": [{"role": "user", "content": "bounded task"}]}),
                ),
                (
                    "/v1/messages",
                    json!({"max_tokens": 20, "messages": [{"role": "user", "content": "bounded task"}]}),
                ),
                ("/v1/responses", json!({"input": "bounded task"})),
            ] {
                upstream.calls.lock().await.clear();
                let before = send(&app, "GET", "/metrics", None).await?;
                body["model"] = json!("deadline");
                let response = tokio::time::timeout(
                    Duration::from_secs(1),
                    send(&app, "POST", path, Some(body)),
                )
                .await??;
                assert_eq!(response.status, StatusCode::OK);
                let selected = if scenario == "valid" {
                    "deadline/weak"
                } else {
                    "deadline/strong"
                };
                assert_eq!(response.headers["x-model-router-selected-model"], selected);
                assert!(response.text()?.contains("ok"));
                let calls = upstream.calls.lock().await;
                assert_eq!(calls.len(), 2);
                assert_eq!(calls[0]["model"], format!("judge/{mode}/{scenario}"));
                assert_eq!(calls[0]["stream"], scenario == "streaming");
                assert_eq!(calls[1]["model"], selected);
                drop(calls);
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
        }
    }
    Ok(())
}

#[tokio::test]
async fn configured_judge_deadline_rejects_invalid_values() -> TestResult {
    for mode in ["classifier", "stage"] {
        for timeout_ms in [
            "0",
            "-1",
            "1.5",
            "\"75\"",
            "75\ntimeout_ms = 76",
            "75\ntimeout_mss = 76",
        ] {
            assert!(
                load_test_config(&deadline_config(
                    "http://127.0.0.1:1/v1",
                    mode,
                    "valid",
                    timeout_ms
                ))
                .is_err()
            );
        }
        load_test_config(&deadline_config(
            "http://127.0.0.1:1/v1",
            mode,
            "valid",
            "1",
        ))?;
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

    let upstream = deadline_upstream().await?;
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
    let backend = |scenario| {
        Backend::OpenAiChat(HttpBackendConfig {
            base_url: upstream.base_url.clone(),
            api_key: None,
            forward_auth: false,
            extra_headers: BTreeMap::new(),
            extra_body: if scenario {
                BTreeMap::from([("scenario".into(), json!("late"))])
            } else {
                BTreeMap::new()
            },
            reasoning_effort: None,
            max_retries: 0,
        })
    };
    let client = TranslatingLlmClient::new(&[
        ModelConfig::new("late/judge", backend(true), None),
        ModelConfig::new("late/weak", backend(false), None),
        ModelConfig::new("late/strong", backend(false), None),
    ])?;
    let models = RuntimeModels::new(
        [
            (Category::Judge, vec!["late/judge".into()]),
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
        ["late/judge", "late/judge", "late/weak"]
    );
    Ok(())
}
