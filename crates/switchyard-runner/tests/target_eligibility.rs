// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Trusted request requirements constrain routing roles and provider calls.

use serde_json::{Value, json};
use switchyard_llm_client::AuxiliaryOperation;
use switchyard_protocol::{LlmClientError, Request, text_request};
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
fn config(endpoint: &str, algorithm: &str, policy: &str) -> String {
    let targets = [
        ("fast", "cloud/fast", &["public"] as &[&str]),
        ("capable", "cloud/capable", &[]),
        ("private_fast", "private/fast", &["private"]),
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
    assert_eq!(
        normal.selected_model_ids,
        ["approved/fast", "approved/capable"]
    );
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
    assert_eq!(outcome.selected_model, "approved/fast");
    assert_eq!(
        outcome.response.served_model().map(|model| model.as_str()),
        Some("approved/capable")
    );
    assert_eq!(
        models_called(&server).await,
        ["approved/fast", "approved/capable"]
    );
    server.verify().await;
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
        assert_eq!(outcome.selected_model_ids, ["approved/fast"]);
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
            "approved/fast",
        ),
        ("[0.0, 1.0, 0.0]", &["private"], "private/fast"),
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
    assert!(route.decide(request("shared", None)).await.is_err());
    route
        .decide(request("shared", Some(&[])))
        .await
        .expect("explicit empty requirements");
    for (labels, expected) in [
        (&["private"] as &[&str], "private/fast"),
        (&["public"], "cloud/fast"),
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
async fn configured_judges_and_auxiliary_calls_require_the_same_labels() {
    let server = MockServer::start().await;
    let algorithm = r#"
type = "llm_classifier"
classifier_target = "approved_judge"
strong_target = "both_capable"
weak_target = "both_fast"
base_threshold = 0.5
fail_open = false
"#;
    let runner =
        Runner::from_toml(&config(&server.uri(), algorithm, "")).expect("approved judge route");
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
    assert_eq!(outcome.selected_model, "approved/fast");
    assert_eq!(
        models_called(&server).await,
        ["approved/judge", "approved/fast"]
    );
    server.verify().await;

    server.reset().await;
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
        let source = config(&server.uri(), STAGE, "")
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

#[test]
fn conflicting_deployment_approvals_are_rejected() {
    let source = config(
        "http://127.0.0.1:1",
        "type = 'random'\ntargets = ['fast', 'both_fast']",
        "",
    );
    let alias = source.replace("id = \"approved/fast\"", "id = \"cloud/fast\"");
    let cross_client = format!(
        "{}\n[llm_clients.other]\nformat = \"openai_chat\"\nbase_url = \"http://127.0.0.1:2/v1\"\n",
        source.replace(
            "id = \"approved/fast\"\nllm_client = \"primary\"",
            "id = \"cloud/fast\"\nllm_client = \"other\""
        )
    );
    for (source, reason) in [(alias, "label"), (cross_client, "client")] {
        let error = match Runner::from_toml(&source) {
            Ok(_) => panic!("conflicting model approvals loaded"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains(reason), "{reason}: {error}");
    }
}
