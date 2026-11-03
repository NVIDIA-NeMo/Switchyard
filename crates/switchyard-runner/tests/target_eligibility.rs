// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Trusted request requirements constrain routing roles and provider calls.

use serde_json::{Value, json};
use switchyard_llm_client::AuxiliaryOperation;
use switchyard_protocol::{Request, text_request};
use switchyard_runner::{Runner, require_target_labels};
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const STAGE: &str = r#"
type = "stage_router"
efficient_target = "fast"
capable_target = "capable"
picker = "efficient_first"
confidence_threshold = 0.6
"#;
const ALTERNATIVES: &str = r#"
target_alternatives = { fast = ["private_fast", "both_fast"], capable = ["private_capable", "both_capable"] }
"#;

fn config(endpoint: &str, algorithm: &str, policy: &str) -> String {
    let targets = [
        ("fast", "cloud/fast", &[] as &[&str]),
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
async fn stage_routes_normally_and_keeps_answer_fallbacks_approved() {
    let server = MockServer::start().await;
    let runner = Runner::from_toml(&config(&server.uri(), STAGE, ALTERNATIVES))
        .expect("eligible Stage route");
    respond(
        &server,
        "cloud/fast",
        completion("cloud/fast", "public answer"),
        1,
    )
    .await;
    respond(&server, "private/fast", ResponseTemplate::new(503), 1).await;
    respond(
        &server,
        "private/capable",
        completion("private/capable", "approved answer"),
        1,
    )
    .await;
    let route = runner.route("chat").expect("route");
    let normal = route
        .execute(request("normal", None), None)
        .await
        .expect("normal answer");
    assert_eq!(normal.selected_model, "cloud/fast");
    let approved = route
        .execute(request("private", Some(&["private"])), None)
        .await
        .expect("approved fallback");
    assert_eq!(
        approved.response.served_model().map(|model| model.as_str()),
        Some("private/capable")
    );
    assert_eq!(
        models_called(&server).await,
        ["cloud/fast", "private/fast", "private/capable"]
    );
    let decision = route
        .decide(request("pair", Some(&["private", "customer_data"])))
        .await
        .expect("approved routing-only result");
    assert_eq!(
        decision.selected_model_ids,
        ["approved/fast", "approved/capable"]
    );
    assert_eq!(models_called(&server).await.len(), 3);
    server.verify().await;
}

#[tokio::test]
async fn every_label_is_required_and_subsets_need_no_catalogue() {
    let server = MockServer::start().await;
    let runner = Runner::from_toml(&config(&server.uri(), STAGE, ALTERNATIVES)).expect("route");
    let route = runner.route("chat").expect("route");
    for (labels, expected) in [
        (&[] as &[&str], "cloud/fast"),
        (&["private"], "private/fast"),
        (&["customer_data"], "approved/fast"),
        (&["customer_data", "private", "private"], "approved/fast"),
    ] {
        let outcome = route
            .decide(request("independent", Some(labels)))
            .await
            .expect("supported labels");
        assert_eq!(outcome.selected_model_ids[0], expected);
    }
    assert!(
        route
            .decide(request("unknown", Some(&["unapproved"])))
            .await
            .is_err()
    );
    assert!(models_called(&server).await.is_empty());
}

#[tokio::test]
async fn filtering_random_targets_preserves_weight_positions_and_zero_weight_exclusions() {
    let server = MockServer::start().await;
    for (weights, expected) in [
        ("[0.0, 0.0, 1.0]", "approved/fast"),
        ("[0.0, 1.0, 0.0]", "private/fast"),
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
            .decide(request("weights", Some(&["private"])))
            .await
            .expect("filtered weighted route");
        assert_eq!(outcome.selected_model_ids, [expected]);
    }
    assert!(models_called(&server).await.is_empty());
}

#[tokio::test]
async fn explicit_requirements_apply_only_to_the_current_request() {
    let server = MockServer::start().await;
    let policy = format!("require_explicit_requirements = true\n{ALTERNATIVES}");
    let source = config(&server.uri(), STAGE, &policy);
    let runner = Runner::from_toml(&source).expect("request-scoped route");
    let route = runner.route("chat").expect("route");
    assert!(route.decide(request("shared", None)).await.is_err());
    for (labels, expected) in [
        (&[] as &[&str], "cloud/fast"),
        (&["private"], "private/fast"),
        (&["customer_data"], "approved/fast"),
        (&[], "cloud/fast"),
    ] {
        let outcome = route
            .decide(request("shared", Some(labels)))
            .await
            .expect("explicit request requirements");
        assert_eq!(outcome.selected_model_ids[0], expected);
    }
    assert!(route.decide(request("shared", None)).await.is_err());

    let unavailable = source.replace("[\"private\",\"customer_data\"]", "[\"private\"]");
    let runner =
        Runner::from_toml(&unavailable).expect("valid config with unavailable intersection");
    let route = runner.route("chat").expect("route");
    assert!(
        route
            .decide(request("blocked", Some(&["private", "customer_data"])))
            .await
            .is_err()
    );
    let outcome = route
        .decide(request("blocked", Some(&["private"])))
        .await
        .expect("the next request has independent requirements");
    assert_eq!(outcome.selected_model_ids[0], "private/fast");
    assert!(models_called(&server).await.is_empty());
}

#[tokio::test]
async fn judges_and_auxiliary_calls_use_the_same_required_labels() {
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
    let source = config(&server.uri(), algorithm, policy);
    let runner = Runner::from_toml(&source).expect("approved judge route");
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
        let source = config(&server.uri(), STAGE, ALTERNATIVES)
            .replace("format = \"openai_chat\"", &format!("format = {format:?}"));
        let runner = Runner::from_toml(&source).expect("auxiliary route");
        Mock::given(method("POST"))
            .and(path(endpoint))
            .and(body_partial_json(json!({"model": "approved/capable"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"input_tokens": 42})))
            .expect(1)
            .mount(&server)
            .await;
        let result = runner
            .route("chat")
            .expect("route")
            .call_auxiliary(
                request("aux", Some(&["private", "customer_data"])),
                operation,
            )
            .await
            .expect("approved auxiliary call");
        assert_eq!(result["input_tokens"], 42);
        assert_eq!(models_called(&server).await, ["approved/capable"]);
        server.verify().await;
        server.reset().await;
    }
}

#[test]
fn invalid_alternatives_and_conflicting_deployment_approvals_are_rejected() {
    let source = config("http://127.0.0.1:1", STAGE, ALTERNATIVES);
    let alias = format!(
        "{source}\n[targets.alias]\nid = \"cloud/fast\"\nllm_client = \"primary\"\nlabels = [\"private\"]\n"
    );
    let cross_client = format!(
        "{}\n[llm_clients.other]\nformat = \"openai_chat\"\nbase_url = \"http://127.0.0.1:2/v1\"\n",
        source.replace(
            "id = \"approved/fast\"\nllm_client = \"primary\"",
            "id = \"cloud/fast\"\nllm_client = \"other\""
        )
    );
    for (source, reason) in [
        (
            source.replace("fast = [\"private_fast\"", "unused = [\"private_fast\""),
            "unused",
        ),
        (
            source.replace("fast = [\"private_fast\"", "fast = [\"missing\""),
            "missing",
        ),
        (alias, "label"),
        (cross_client, "client"),
    ] {
        let error = match Runner::from_toml(&source) {
            Ok(_) => panic!("invalid eligibility config loaded: {reason}"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains(reason), "{reason}: {error}");
    }
}
