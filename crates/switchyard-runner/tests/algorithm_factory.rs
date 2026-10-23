// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public factory contracts for configured routes without provider clients.

use std::sync::Arc;

use switchyard_protocol::{Category, Metadata, ModelId, Request, text_request};
use switchyard_runner::{AlgorithmFactory, Runner};

const MISSING_KEY: &str = "SWITCHYARD_FACTORY_TEST_KEY_THAT_IS_NOT_SET";
const DEPLOYMENT: &str = r#"
schema_version = 1
[llm_clients.answers]
format = "openai_chat"
base_url = "https://example.test/v1"
api_key_env = "SWITCHYARD_FACTORY_TEST_KEY_THAT_IS_NOT_SET"
[targets]
fast = { id = "upstream/fast", llm_client = "answers" }
capable = { id = "upstream/capable", llm_client = "answers" }
[decision_clients.judge]
format = "openai_decisions"
endpoint = "https://example.test/v1/decisions"
api_key_env = "SWITCHYARD_FACTORY_TEST_KEY_THAT_IS_NOT_SET"
timeout_ms = 2000
[decision_targets.judge]
id = "upstream/judge"
decision_client = "judge"
[routes.parent]
id = "public/main"
type = "passthrough"
target = "fast"
[routes.parent.subagents]
type = "passthrough"
target = "capable"
[routes.classifier]
id = "public/classify"
type = "llm_classifier"
classifier_target = "judge"
strong_target = "capable"
weak_target = "fast"
[routes.classifier.decision]
cutoff = 0.7
candidates = { a = "capable", b = "fast" }
evidence = { description = "Compare both targets." }
"#;

#[test]
fn full_deployment_builds_without_provider_credentials() {
    assert!(std::env::var_os(MISSING_KEY).is_none());
    let factory = AlgorithmFactory::from_toml(DEPLOYMENT).expect("deployment should parse");
    let algorithms = factory.build().expect("algorithms need no provider keys");
    assert_eq!(
        algorithms.keys().map(String::as_str).collect::<Vec<_>>(),
        ["public/classify", "public/main"]
    );
    let configured = &algorithms["public/classify"];
    let models = &configured.models;
    assert_eq!(configured.algorithm.name(), "llm_task_classifier");
    assert_eq!(
        models.models_for(&Category::Judge),
        [ModelId::from("upstream/judge")]
    );
    assert_eq!(
        models.models_for(&Category::Capable),
        [ModelId::from("upstream/capable")]
    );
    assert_eq!(
        models.models_for(&Category::Efficient),
        [ModelId::from("upstream/fast")]
    );
    assert_eq!(
        models.models_for(&Category::Any),
        [
            ModelId::from("upstream/fast"),
            ModelId::from("upstream/capable")
        ]
    );

    let error = Runner::from_toml(DEPLOYMENT)
        .err()
        .expect("the native runner still requires configured credentials");
    assert!(error.to_string().contains(MISSING_KEY), "{error}");
}

#[tokio::test]
async fn configured_parent_and_subagent_targets_stay_separate() {
    let mut algorithms = AlgorithmFactory::from_toml(DEPLOYMENT)
        .expect("deployment should parse")
        .build()
        .expect("algorithms should build");
    let configured = algorithms.remove("public/main").expect("public route id");
    let algorithm = configured.algorithm;
    let models = configured.models;
    assert_eq!(
        models.models_for(&Category::Any),
        [ModelId::from("upstream/fast")]
    );
    assert_eq!(
        models.subagent_models_for(&Category::Any),
        [ModelId::from("upstream/capable")]
    );
    let models = Arc::new(models);
    for (delegated, selected) in [(false, "upstream/fast"), (true, "upstream/capable")] {
        let request = Request {
            llm_request: text_request(Some("public/main".to_string()), "hello"),
            metadata: delegated.then(|| Metadata {
                session_id: Some("task".to_string()),
                agent_id: Some("child".to_string()),
                is_subagent: true,
                is_delegated_work: true,
                ..Metadata::default()
            }),
            ..Request::default()
        };
        let outcome = libsy::drive(
            Arc::clone(&algorithm),
            request,
            Arc::clone(&models),
            |_| async { panic!("passthrough must not dispatch provider calls") },
        )
        .await
        .expect("configured route should resolve");
        assert_eq!(outcome.selected_model_ids, [ModelId::from(selected)]);
        assert!(outcome.response.is_none());
    }
}

#[test]
fn invalid_routing_configuration_is_rejected() {
    for (source, expected) in [
        (
            DEPLOYMENT.replacen("target = \"fast\"", "target = \"missing\"", 1),
            "unknown target missing",
        ),
        (
            DEPLOYMENT.replace("id = \"public/classify\"", "id = \"public/main\""),
            "route ids must be unique",
        ),
        (
            DEPLOYMENT.replace(
                "type = \"passthrough\"",
                "type = \"passthrough\"\nbogus = true",
            ),
            "bogus",
        ),
    ] {
        let error = AlgorithmFactory::from_toml(&source)
            .and_then(|factory| factory.build())
            .err()
            .expect("invalid deployment should fail");
        assert!(error.to_string().contains(expected), "{error}");
    }
}
