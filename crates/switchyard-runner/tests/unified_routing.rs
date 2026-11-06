// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use switchyard_protocol::{Category, ModelId, Request, text_request};
use switchyard_runner::{AlgorithmFactory, Runner};

const DEPLOYMENT: &str = r#"
schema_version = 1
[llm_clients.local]
format = "openai_chat"
base_url = "https://example.test/v1"
[targets]
fast = { id = "provider/fast", llm_client = "local" }
strong = { id = "provider/strong", llm_client = "local" }
judge = { id = "provider/judge", llm_client = "local" }
reviewer = { id = "provider/reviewer", llm_client = "local" }
[pool]
models = [
  { target = "strong", provides = ["capable", "specialist"] },
  { target = "fast", provides = ["efficient"] },
]
default = "capable"
[routes.chat]
id = "chat"
type = "auto"
"#;

#[tokio::test]
async fn pool_only_route_uses_default_without_a_judge() {
    let mut routes = AlgorithmFactory::from_toml(DEPLOYMENT)
        .unwrap()
        .build()
        .unwrap();
    let (algorithm, models) = routes.remove("chat").unwrap();
    assert_eq!(algorithm.name(), "auto");
    assert_eq!(
        models.models_for(&Category::Any),
        [ModelId::from("strong"), ModelId::from("fast")]
    );
    let outcome = libsy::drive(
        algorithm,
        Request {
            llm_request: text_request(Some("auto".into()), "hello"),
            ..Request::default()
        },
        Arc::new(models),
        |_| async { panic!("a pool default does not call a judge") },
    )
    .await
    .unwrap();
    assert_eq!(
        outcome.selected_model_ids,
        [ModelId::from("strong"), ModelId::from("fast")]
    );
}

#[test]
fn native_and_factory_paths_share_the_pool_and_separate_judges() {
    let source = format!(
        "{DEPLOYMENT}\n[routes.chat.signals]\ntask = {{ judge = 'judge', trigger = 'on_demand', threshold = 0.9 }}\ntool_loop = {{ enabled = true, confidence = 0.5 }}\nreview = {{ judge = 'reviewer', max_reviews = 3, confirmations = 2 }}\n"
    );
    let routes = AlgorithmFactory::from_toml(&source)
        .unwrap()
        .build()
        .unwrap();
    let (_, models) = &routes["chat"];
    assert_eq!(
        models.models_for(&Category::Judge),
        [ModelId::from("judge"), ModelId::from("reviewer")]
    );
    assert_eq!(
        models.models_for(&Category::Any),
        [ModelId::from("strong"), ModelId::from("fast")]
    );
    assert!(Runner::from_toml(&source).is_ok());
}

#[test]
fn any_number_of_tags_can_share_the_same_catalog() {
    let source = format!(
        "{}\n[routes.second]\nid = 'second'\ntype = 'auto'\n",
        DEPLOYMENT.replace("default = \"capable\"", "default = \"specialist\"")
    );
    let routes = AlgorithmFactory::from_toml(&source)
        .unwrap()
        .build()
        .unwrap();
    for (_, models) in routes.values() {
        assert_eq!(
            models.models_for(&"specialist".parse().unwrap()),
            [ModelId::from("strong")]
        );
    }
}

#[test]
fn invalid_catalogs_and_mixed_migration_forms_are_rejected() {
    for source in [
        DEPLOYMENT.replace("default = \"capable\"", "default = \"unknown\""),
        DEPLOYMENT.replace("target = \"fast\"", "target = \"missing\""),
        DEPLOYMENT.replace("[pool]", "[unused_pool]"),
        format!(
            "{DEPLOYMENT}\ncapable_target = 'strong'\nefficient_target = 'fast'\n[routes.chat.signals.tool_loop]\nenabled = true\n"
        ),
        format!("{DEPLOYMENT}\n[routes.chat.signals.tool_loop]\nconfidence = 1.1\n"),
        format!("{DEPLOYMENT}\n[routes.chat.signals.task]\njudge = 'judge'\nthreshold = -0.1\n"),
        format!("{DEPLOYMENT}\n[routes.chat.signals.tool_loop]\ncapable_score_threshold = 0.55\n"),
        format!("{DEPLOYMENT}\n[routes.chat.signals.tool_loop]\nescalation_hold_turns = 3\n"),
        format!(
            "{DEPLOYMENT}\n[routes.chat.signals.review]\njudge = 'reviewer'\nmax_reviews = 3\nconfirmations = 0\n"
        ),
        format!(
            "{DEPLOYMENT}\n[routes.chat.signals.review]\njudge = 'reviewer'\nmax_reviews = 3\nconfirmations = 4\n"
        ),
    ] {
        assert!(
            AlgorithmFactory::from_toml(&source)
                .and_then(|factory| factory.build())
                .is_err(),
            "accepted {source}"
        );
    }
}

#[test]
fn legacy_auto_preset_keeps_loading() {
    let source = DEPLOYMENT.replace(
        "type = \"auto\"",
        "type = \"auto\"\ncapable_target = 'strong'\nefficient_target = 'fast'",
    );
    let routes = AlgorithmFactory::from_toml(&source)
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(routes["chat"].0.name(), "stage_router");
}

#[tokio::test]
async fn review_applies_the_selected_targets_prompt_without_leaking_it_to_the_judge() {
    use serde_json::{Value, json};
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    for (default, answer_model, prompt) in [
        ("capable", "provider/strong", "STRONG ANSWER PROMPT"),
        ("efficient", "provider/fast", "FAST ANSWER PROMPT"),
    ] {
        let server = MockServer::start().await;
        let source = DEPLOYMENT
            .replace("https://example.test/v1", &format!("{}/v1", server.uri()))
            .replace("default = \"capable\"", &format!("default = {default:?}"))
            .replace("id = \"provider/strong\", llm_client = \"local\"", "id = \"provider/strong\", llm_client = \"local\", system_prompt = \"STRONG ANSWER PROMPT\"")
            .replace("id = \"provider/fast\", llm_client = \"local\"", "id = \"provider/fast\", llm_client = \"local\", system_prompt = \"FAST ANSWER PROMPT\"");
        let source = format!("{source}\n[routes.chat.signals.review]\njudge = 'reviewer'\n");
        for (model, content) in [(answer_model, "done"), ("provider/reviewer", "APPROVE")] {
            Mock::given(method("POST"))
                .and(path("/v1/chat/completions"))
                .and(body_partial_json(json!({"model": model})))
                .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                    "id": "review-fixture", "object": "chat.completion", "model": model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]
                })))
                .expect(1)
                .mount(&server).await;
        }
        let runner = Runner::from_toml(&source).unwrap();
        let route = runner.route("chat").unwrap();
        route
            .execute(
                Request {
                    llm_request: text_request(None, "finish the task"),
                    ..Request::default()
                },
                None,
            )
            .await
            .unwrap();
        let calls = server.received_requests().await.unwrap();
        assert_eq!(calls.len(), 2);
        let answer: Value = serde_json::from_slice(&calls[0].body).unwrap();
        let review: Value = serde_json::from_slice(&calls[1].body).unwrap();
        assert_eq!(answer["model"], answer_model);
        assert!(answer["messages"].to_string().contains(prompt));
        assert!(!review["messages"].to_string().contains("ANSWER PROMPT"));
        server.verify().await;
    }
}

#[test]
fn reviewed_answer_prompts_cannot_also_be_judge_prompts() {
    let source = DEPLOYMENT.replace(
        "id = \"provider/strong\", llm_client = \"local\"",
        "id = \"provider/strong\", llm_client = \"local\", system_prompt = \"ANSWER ONLY\"",
    );
    let source = format!("{source}\n[routes.chat.signals.review]\njudge = 'strong'\n");
    let error = Runner::from_toml(&source)
        .err()
        .expect("ambiguous prompt role must fail");
    assert!(error.to_string().contains("also the routing-only target"));
}
