// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Trusted requirements constrain routing roles, provider calls and retained tasks.

use serde_json::{Value, json};
use switchyard_llm_client::AuxiliaryOperation;
use switchyard_protocol::{
    ContentBlock, LlmClientError, Message, ModelId, Request, Role, ToolCall, ToolResult,
    text_request,
};
use switchyard_runner::{Runner, RunnerError, require_target_labels};
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const STAGE: &str = r#"
type = "stage_router"
efficient_target = "both_fast"
capable_target = "both_capable"
picker = "efficient_first"
confidence_threshold = 0.6
"#;
const ALTERNATIVES: &str = r#"
target_alternatives = { fast = ["private_fast", "both_fast"], capable = ["private_capable", "both_capable"] }
"#;

fn config(endpoint: &str, algorithm: &str, policy: &str) -> String {
    let targets = [
        ("fast", "cloud/fast", &["public"] as &[&str]),
        ("capable", "cloud/capable", &[]),
        ("private_fast", "private/fast", &["private"]),
        ("private_capable", "private/capable", &["private"]),
        ("both_fast", "approved/fast", &["private", "customer_data"]),
        (
            "both_capable",
            "approved/capable",
            &["private", "customer_data"],
        ),
        ("judge", "cloud/judge", &[]),
        (
            "approved_judge",
            "approved/judge",
            &["private", "customer_data"],
        ),
    ]
    .into_iter()
    .map(|(name, id, labels)| {
        format!(
            "[targets.{name}]\nid = {id:?}\nllm_client = \"primary\"\nlabels = {}\n",
            json!(labels)
        )
    })
    .collect::<String>();
    format!(
        r#"
schema_version = 1
[llm_clients.primary]
format = "openai_chat"
base_url = "{endpoint}/v1"
max_retries = 0
failure_cooldown_ms = 0
{targets}
[routes.chat]
id = "chat"
{algorithm}
[routes.chat.eligibility]
{policy}
"#
    )
}

fn request(task: &str, labels: Option<&[&str]>) -> Request {
    let mut request = Request {
        llm_request: text_request(None, "Summarize these synthetic notes."),
        ..Request::default()
    };
    request.metadata.get_or_insert_default().task_id = Some(task.into());
    if let Some(labels) = labels {
        require_target_labels(&mut request, labels.iter().copied()).expect("trusted requirements");
    }
    request
}

fn completion(model: &str, content: &str) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "id": "fixture", "object": "chat.completion", "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}]
    }))
}

async fn respond(server: &MockServer, model: &str, response: ResponseTemplate, count: u64) {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({"model": model})))
        .respond_with(response)
        .expect(count)
        .mount(server)
        .await;
}

async fn models_called(server: &MockServer) -> Vec<String> {
    server
        .received_requests()
        .await
        .expect("recorded requests")
        .iter()
        .map(|request| {
            let body: Value = serde_json::from_slice(&request.body).expect("provider JSON");
            body["model"].as_str().expect("provider model").to_owned()
        })
        .collect()
}

#[tokio::test]
async fn configured_stage_keeps_answer_fallbacks_approved() {
    let server = MockServer::start().await;
    let runner = Runner::from_toml(&config(&server.uri(), STAGE, "")).expect("approved Stage");
    let route = runner.route("chat").expect("route");
    let normal = route
        .decide(request("normal", None))
        .await
        .expect("normal routing");
    assert_eq!(normal.selected_model_ids, ["both_fast", "both_capable"]);
    respond(&server, "approved/fast", ResponseTemplate::new(503), 1).await;
    respond(
        &server,
        "approved/capable",
        completion("approved/capable", "approved answer"),
        1,
    )
    .await;
    let outcome = route
        .execute(
            request("private", Some(&["private", "customer_data"])),
            None,
        )
        .await
        .expect("approved fallback");
    assert_eq!(outcome.selected_model, "both_fast");
    assert_eq!(
        outcome.response.served_model().map(|model| model.as_str()),
        Some("both_capable")
    );
    assert_eq!(
        models_called(&server).await,
        ["approved/fast", "approved/capable"]
    );
    server.verify().await;
}

#[tokio::test]
async fn three_tier_stage_routes_moderate_recovery_and_keeps_fallbacks_approved() {
    let server = MockServer::start().await;
    let mut recovery = request("moderate", Some(&["private", "customer_data"]));
    recovery.llm_request.messages.extend([
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "call_1".into(),
                name: "Bash".into(),
                arguments: json!({"command": "cargo test"}),
            })],
        },
        Message {
            role: Role::Tool,
            content: vec![ContentBlock::ToolResult(ToolResult {
                tool_call_id: "call_1".into(),
                content: vec![ContentBlock::Text {
                    text: "connection refused".into(),
                }],
                is_error: Some(true),
            })],
        },
    ]);
    let algorithm = STAGE.replace(
        "confidence_threshold = 0.6",
        "confidence_threshold = 0.2\nbalanced_target = \"approved_judge\"",
    );
    let runner =
        Runner::from_toml(&config(&server.uri(), &algorithm, "")).expect("three-tier stage");
    let route = runner.route("chat").expect("route");
    let decision = route
        .decide(recovery.clone())
        .await
        .expect("moderate recovery");
    assert_eq!(
        decision.selected_model_ids,
        ["approved_judge", "both_capable", "both_fast"]
    );
    let quiet = route
        .decide(request("quiet", Some(&["private", "customer_data"])))
        .await
        .expect("picker default");
    assert_eq!(quiet.selected_model_ids[0], "both_fast");
    let mut critical = recovery.clone();
    critical.metadata.as_mut().expect("metadata").task_id = Some("critical".into());
    if let ContentBlock::ToolResult(result) = &mut critical.llm_request.messages[2].content[0] {
        result.content = vec![ContentBlock::Text {
            text: "out of memory".into(),
        }];
    }
    let severe = route.decide(critical).await.expect("hard override");
    assert_eq!(severe.selected_model_ids[0], "both_capable");
    respond(&server, "approved/judge", ResponseTemplate::new(503), 1).await;
    respond(
        &server,
        "approved/capable",
        completion("approved/capable", "recovered"),
        1,
    )
    .await;
    route
        .execute(recovery.clone(), None)
        .await
        .expect("approved fallback");
    assert_eq!(
        models_called(&server).await,
        ["approved/judge", "approved/capable"]
    );
    server.verify().await;

    let blocked = Runner::from_toml(&config(
        &server.uri(),
        &algorithm.replace(
            "balanced_target = \"approved_judge\"",
            "balanced_target = \"judge\"",
        ),
        "",
    ))
    .expect("valid route");
    assert!(
        blocked
            .route("chat")
            .expect("route")
            .decide(recovery)
            .await
            .is_err()
    );
    assert_eq!(
        models_called(&server).await.len(),
        2,
        "unapproved middle tier must fail before I/O"
    );
}

#[tokio::test]
async fn ordered_alternatives_rebind_clients_settings_prompts_and_fallbacks() {
    let primary = MockServer::start().await;
    let approved = MockServer::start().await;
    let algorithm = STAGE
        .replace("both_fast", "fast")
        .replace("both_capable", "capable");
    let mut source = config(&primary.uri(), &algorithm, ALTERNATIVES);
    let cases = [
        (
            &primary,
            "fast",
            "cloud/fast",
            "minimal",
            "PUBLIC PROMPT",
            200,
        ),
        (
            &approved,
            "private_fast",
            "private/fast",
            "low",
            "PRIVATE FAST PROMPT",
            503,
        ),
        (
            &approved,
            "private_capable",
            "private/capable",
            "high",
            "PRIVATE CAPABLE PROMPT",
            200,
        ),
        (
            &approved,
            "both_fast",
            "approved/fast",
            "medium",
            "APPROVED PROMPT",
            200,
        ),
    ];
    for (server, target, model, effort, prompt, status) in cases {
        let client = if target == "fast" {
            "primary"
        } else {
            "approved"
        };
        source = source.replace(
            &format!("[targets.{target}]\nid = {model:?}\nllm_client = \"primary\""),
            &format!(
                "[targets.{target}]\nid = {model:?}\nllm_client = {client:?}\nreasoning_effort = {effort:?}\nsystem_prompt = {prompt:?}"
            ),
        );
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_partial_json(json!({
                "model": model,
                "reasoning_effort": effort,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Summarize these synthetic notes."}
                ]
            })))
            .respond_with(if status == 200 {
                completion(model, "answer")
            } else {
                ResponseTemplate::new(status)
            })
            .expect(1)
            .mount(server)
            .await;
    }
    source.push_str(&format!(
        "\n[llm_clients.approved]\nformat = 'openai_chat'\nbase_url = '{}/v1'\nmax_retries = 0\nfailure_cooldown_ms = 0\n",
        approved.uri()
    ));
    let runner = Runner::from_toml(&source).expect("ordered alternatives");
    let route = runner.route("chat").expect("route");
    for (task, labels, selected, served) in [
        ("normal", None, "fast", "fast"),
        (
            "private",
            Some(&["private"] as &[&str]),
            "private_fast",
            "private_capable",
        ),
        (
            "pair",
            Some(&["private", "customer_data"] as &[&str]),
            "both_fast",
            "both_fast",
        ),
    ] {
        let outcome = route
            .execute(request(task, labels), None)
            .await
            .expect("resolved answer and fallback");
        assert_eq!(outcome.selected_model, selected);
        assert_eq!(
            outcome.response.served_model().map(|model| model.as_str()),
            Some(served)
        );
    }
    let alternative = ModelId::from("private_fast");
    assert_eq!(
        route
            .target(&alternative)
            .expect("alternative metadata")
            .reasoning_effort
            .as_deref(),
        Some("low")
    );
    assert_eq!(route.provider_model(&alternative), "private/fast");
    assert_eq!(models_called(&primary).await, ["cloud/fast"]);
    assert_eq!(
        models_called(&approved).await,
        ["private/fast", "private/capable", "approved/fast"]
    );
    primary.verify().await;
    approved.verify().await;
}

#[tokio::test]
async fn filtered_targets_sharing_a_model_keep_their_own_settings() {
    let server = MockServer::start().await;
    for (target, effort) in [("both_fast", "low"), ("both_capable", "high")] {
        let algorithm = format!("type = 'random'\ntargets = ['fast', '{target}']");
        let source = config(&server.uri(), &algorithm, "")
            .replace("id = \"approved/capable\"", "id = \"approved/fast\"")
            .replace(
                "[targets.both_fast]",
                "[targets.both_fast]\nreasoning_effort = 'low'",
            )
            .replace(
                "[targets.both_capable]",
                "[targets.both_capable]\nreasoning_effort = 'high'",
            );
        let runner = Runner::from_toml(&source).expect("shared provider model");
        let route = runner.route("chat").expect("route");
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_partial_json(
                json!({"model": "approved/fast", "reasoning_effort": effort}),
            ))
            .respond_with(completion("approved/fast", "approved answer"))
            .expect(1)
            .mount(&server)
            .await;
        let outcome = route
            .execute(request("shared", Some(&["private", "customer_data"])), None)
            .await
            .expect("approved target settings");
        assert_eq!(outcome.selected_model, target);
        assert_eq!(
            outcome.response.served_model().map(|id| id.as_str()),
            Some(target)
        );
        assert_eq!(
            route.provider_model(&outcome.selected_model),
            "approved/fast"
        );
        let metadata = route
            .target(&outcome.selected_model)
            .expect("target metadata");
        assert_eq!(metadata.reasoning_effort.as_deref(), Some(effort));
        server.verify().await;
        server.reset().await;
    }
}

#[tokio::test]
async fn every_label_is_required_and_subsets_need_no_catalogue() {
    let server = MockServer::start().await;
    let algorithm = "type = 'passthrough'\ntarget = 'both_fast'";
    let runner = Runner::from_toml(&config(&server.uri(), algorithm, "")).expect("route");
    let route = runner.route("chat").expect("route");
    for labels in [
        &[] as &[&str],
        &["private"],
        &["customer_data"],
        &["customer_data", "private", "private"],
    ] {
        let outcome = route
            .decide(request("independent", Some(labels)))
            .await
            .expect("every required label is approved");
        assert_eq!(outcome.selected_model_ids, ["both_fast"]);
    }
    assert!(
        route
            .decide(request("unknown", Some(&["unapproved"])))
            .await
            .is_err()
    );
    let runner = Runner::from_toml(&config(
        &server.uri(),
        "type = 'passthrough'\ntarget = 'private_fast'",
        "",
    ))
    .expect("single-label target");
    assert!(
        runner
            .route("chat")
            .expect("route")
            .decide(request("pair", Some(&["private", "customer_data"])))
            .await
            .is_err()
    );
    assert!(models_called(&server).await.is_empty());
}

#[tokio::test]
async fn filtering_random_targets_preserves_weight_positions_and_zero_weight_exclusions() {
    let server = MockServer::start().await;
    for (weights, labels, expected) in [
        (
            "[0.0, 0.0, 1.0]",
            &["private", "customer_data"] as &[&str],
            "both_fast",
        ),
        ("[0.0, 1.0, 0.0]", &["private"], "private_fast"),
    ] {
        let algorithm = format!(
            r#"
type = "random"
targets = ["fast", "private_fast", "both_fast"]
weights = {weights}
seed = 7
"#
        );
        let runner =
            Runner::from_toml(&config(&server.uri(), &algorithm, "")).expect("weighted route");
        let outcome = runner
            .route("chat")
            .expect("route")
            .decide(request("weights", Some(labels)))
            .await
            .expect("filtered weighted route");
        assert_eq!(outcome.selected_model_ids, [expected]);
    }
    assert!(models_called(&server).await.is_empty());
}

#[tokio::test]
async fn explicit_requirements_apply_only_to_the_current_request() {
    let server = MockServer::start().await;
    let algorithm = "type = 'random'\ntargets = ['fast', 'private_fast']\nseed = 7";
    let runner = Runner::from_toml(&config(
        &server.uri(),
        algorithm,
        "require_explicit_requirements = true",
    ))
    .expect("request-scoped route");
    let route = runner.route("chat").expect("route");
    let error = route
        .decide(request("shared", None))
        .await
        .err()
        .expect("missing requirements");
    assert!(error.to_string().contains("explicit"), "{error}");
    let unrestricted = route
        .decide(request("shared", Some(&[])))
        .await
        .expect("explicit empty requirements");
    assert_eq!(unrestricted.selected_model_ids.len(), 2);
    for (labels, expected) in [
        (&["private"] as &[&str], "private_fast"),
        (&["public"], "fast"),
    ] {
        let outcome = route
            .decide(request("shared", Some(labels)))
            .await
            .expect("independent requirements on the same task");
        assert_eq!(outcome.selected_model_ids, [expected]);
    }
    route
        .decide(request("shared", Some(&[])))
        .await
        .expect("empty requirements remain valid");
    assert!(route.decide(request("shared", None)).await.is_err());
    assert!(models_called(&server).await.is_empty());
}

#[tokio::test]
async fn explicit_empty_differs_from_missing_and_task_requirements_only_accumulate() {
    let server = MockServer::start().await;
    let algorithm = STAGE
        .replace("both_fast", "fast")
        .replace("both_capable", "capable");
    let policy = format!("require_explicit_requirements = true\nscope = 'task'\n{ALTERNATIVES}");
    let source = config(&server.uri(), &algorithm, &policy);
    let runner = Runner::from_toml(&source).expect("task-scoped route");
    let route = runner.route("chat").expect("route");
    assert!(route.decide(request("shared", None)).await.is_err());
    for (task, labels, expected) in [
        ("shared", &[] as &[&str], ["fast", "capable"]),
        ("shared", &["private"], ["private_fast", "private_capable"]),
        ("shared", &["customer_data"], ["both_fast", "both_capable"]),
        ("shared", &[], ["both_fast", "both_capable"]),
        ("other", &[], ["fast", "capable"]),
    ] {
        let outcome = route
            .decide(request(task, Some(labels)))
            .await
            .expect("retained task requirements");
        assert_eq!(outcome.selected_model_ids, expected);
    }
    assert!(route.decide(request("shared", None)).await.is_err());

    let unavailable = source.replace("[\"private\",\"customer_data\"]", "[\"private\"]");
    let runner =
        Runner::from_toml(&unavailable).expect("valid config with unavailable intersection");
    let route = runner.route("chat").expect("route");
    route
        .decide(request("blocked", Some(&["private"])))
        .await
        .expect("initial private task");
    for labels in [&["customer_data"] as &[&str], &["private"], &[]] {
        assert!(
            route
                .decide(request("blocked", Some(labels)))
                .await
                .is_err()
        );
    }
    let other = route
        .decide(request("other", Some(&["private"])))
        .await
        .expect("independent task requirements");
    assert_eq!(
        other.selected_model_ids,
        ["private_fast", "private_capable"]
    );
    assert!(models_called(&server).await.is_empty());
}

#[tokio::test]
async fn unapproved_primary_fallback_or_judge_fails_before_provider_io() {
    let server = MockServer::start().await;
    let judge = r#"
type = "llm_classifier"
classifier_target = "judge"
strong_target = "both_capable"
weak_target = "both_fast"
base_threshold = 0.5
fail_open = false
"#;
    for algorithm in [
        STAGE.replace("both_fast", "fast"),
        STAGE.replace("both_capable", "capable"),
        judge.to_owned(),
    ] {
        let runner = Runner::from_toml(&config(&server.uri(), &algorithm, ""))
            .expect("valid fixed-target configuration");
        let error = match runner
            .route("chat")
            .expect("route")
            .execute(
                request("blocked", Some(&["private", "customer_data"])),
                None,
            )
            .await
        {
            Ok(_) => panic!("unapproved model was admitted"),
            Err(error) => error,
        };
        assert!(
            matches!(
                error,
                RunnerError::Client(LlmClientError::InvalidRequest { .. })
            ),
            "{error}"
        );
    }
    assert!(models_called(&server).await.is_empty());
}

#[tokio::test]
async fn alternative_judges_and_auxiliary_calls_require_the_same_labels() {
    let server = MockServer::start().await;
    let algorithm = r#"
type = "llm_classifier"
classifier_target = "judge"
strong_target = "capable"
weak_target = "fast"
base_threshold = 0.5
fail_open = false
"#;
    let policy = r#"target_alternatives = { fast = ["both_fast"], capable = ["both_capable"], judge = ["approved_judge"] }"#;
    let runner = Runner::from_toml(&config(&server.uri(), algorithm, policy))
        .expect("approved judge alternatives");
    respond(&server, "approved/judge", completion("approved/judge", &json!({
        "crux": "bounded task", "primary_rule": "SUP-1", "capability_boundary": "supported", "p_solve": 0.9
    }).to_string()), 1).await;
    respond(
        &server,
        "approved/fast",
        completion("approved/fast", "approved answer"),
        1,
    )
    .await;
    let outcome = runner
        .route("chat")
        .expect("route")
        .execute(request("judge", Some(&["private", "customer_data"])), None)
        .await
        .expect("approved judge and answer");
    assert_eq!(outcome.selected_model, "both_fast");
    assert_eq!(
        models_called(&server).await,
        ["approved/judge", "approved/fast"]
    );
    server.verify().await;

    server.reset().await;
    let stage = STAGE
        .replace("both_fast", "fast")
        .replace("both_capable", "capable");
    for (format, endpoint, operation) in [
        (
            "openai_responses",
            "/v1/responses/input_tokens",
            AuxiliaryOperation::ResponsesInputTokens,
        ),
        (
            "anthropic_messages",
            "/v1/messages/count_tokens",
            AuxiliaryOperation::AnthropicCountTokens,
        ),
    ] {
        let source = config(&server.uri(), &stage, ALTERNATIVES)
            .replace("format = \"openai_chat\"", &format!("format = {format:?}"));
        let runner = Runner::from_toml(&source).expect("auxiliary route");
        let route = runner.route("chat").expect("route");
        Mock::given(method("POST"))
            .and(path(endpoint))
            .and(body_partial_json(json!({"model": "approved/capable"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"input_tokens": 42})))
            .expect(1)
            .mount(&server)
            .await;
        let result = route
            .call_auxiliary(
                request("aux", Some(&["private", "customer_data"])),
                operation,
            )
            .await
            .expect("approved auxiliary call");
        assert_eq!(result["input_tokens"], 42);
        assert!(
            route
                .call_auxiliary(request("blocked", Some(&["unapproved"])), operation)
                .await
                .is_err()
        );
        assert_eq!(models_called(&server).await, ["approved/capable"]);
        server.verify().await;
        server.reset().await;
    }
}

#[tokio::test]
async fn typed_decision_alternatives_rebind_the_judge_and_candidate_targets() {
    let primary = MockServer::start().await;
    let approved = MockServer::start().await;
    let algorithm = r#"
type = "llm_classifier"
classifier_target = "standard"
strong_target = "capable"
weak_target = "fast"
fail_open = false
"#;
    let policy = r#"target_alternatives = { fast = ["both_fast"], capable = ["both_capable"], standard = ["approved_decision"] }"#;
    let source = format!(
        r#"{}
[routes.chat.decision]
cutoff = 0.5
[decision_clients.standard]
format = "openai_decisions"
endpoint = "{}/v1/decisions"
api_key_env = "PATH"
timeout_ms = 2000
[decision_clients.approved]
format = "openai_decisions"
endpoint = "{}/v1/decisions"
api_key_env = "PATH"
timeout_ms = 2000
[decision_targets.standard]
id = "cloud/decision"
decision_client = "standard"
[decision_targets.approved_decision]
id = "approved/decision"
decision_client = "approved"
labels = ["private", "customer_data"]
"#,
        config(&primary.uri(), algorithm, policy),
        primary.uri(),
        approved.uri()
    );
    let runner = Runner::from_toml(&source).expect("typed judge alternatives");
    Mock::given(method("POST"))
        .and(path("/v1/decisions"))
        .and(body_partial_json(json!({"model": "approved/decision"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "approved/decision",
            "answers": [{
                "type": "choice", "name": "route", "choice": "efficient",
                "probabilities": [
                    {"value": "capable", "probability": 0.0},
                    {"value": "efficient", "probability": 1.0}
                ]
            }],
            "usage": {
                "input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0}
            }
        })))
        .expect(1)
        .mount(&approved)
        .await;
    respond(
        &primary,
        "approved/fast",
        completion("approved/fast", "approved answer"),
        1,
    )
    .await;
    let route = runner.route("chat").expect("route");
    let outcome = route
        .execute(request("typed", Some(&["private", "customer_data"])), None)
        .await
        .expect("approved typed judge and candidate");
    assert_eq!(outcome.selected_model, "both_fast");
    assert_eq!(models_called(&primary).await, ["approved/fast"]);
    assert_eq!(models_called(&approved).await, ["approved/decision"]);
    primary.verify().await;
    approved.verify().await;
}
