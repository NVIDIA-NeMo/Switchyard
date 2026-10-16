// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use async_trait::async_trait;
use std::sync::Arc;
use switchyard_protocol::{
    LlmClientError, LlmResponse, Request, Response, RoutedLlmClient, text_request, text_response,
};
use switchyard_runner::Runner;

const CONFIG: &str = r#"
schema_version = 1
[llm_clients.local]
format = "openai_chat"
base_url = "http://127.0.0.1:1/v1"
[targets.fast]
id = "fast-model"
llm_client = "local"
[targets.strong]
id = "strong-model"
llm_client = "local"
[routes.test]
id = "route-id"
type = "llm_classifier"
classifier_target = "strong"
strong_target = "strong"
weak_target = "fast"
base_threshold = 0.5
"#;

#[test]
fn accepts_per_request_classification_and_rejects_session_state() {
    assert!(Runner::from_toml_for_model_selection(CONFIG).is_ok());
    let stateful = format!("{CONFIG}\nclassify_trigger = 'new_session'");
    assert!(Runner::from_toml(&stateful).is_ok());
    assert!(Runner::from_toml_for_model_selection(&stateful).is_err());

    let invalid = format!("{CONFIG}\nmessage_hash_fallback = true");
    assert!(Runner::from_toml(&invalid).is_err());
    assert!(Runner::from_toml_for_model_selection(&invalid).is_err());
}

#[test]
fn rejects_answer_producing_algorithms() {
    let config = CONFIG.replace(
        "type = \"llm_classifier\"\nclassifier_target = \"strong\"\nstrong_target = \"strong\"\nweak_target = \"fast\"\nbase_threshold = 0.5",
        "type = 'advisor'\nexecutor_target = 'fast'\nadvisor_target = 'strong'",
    );
    assert!(Runner::from_toml(&config).is_ok());
    assert!(Runner::from_toml_for_model_selection(&config).is_err());
}

#[test]
fn rejects_completion_overrides_and_forwarded_auth() {
    for (table, setting) in [
        ("targets.fast", "system_prompt = 'override'"),
        ("llm_clients.local", "forward_auth = true"),
    ] {
        let config = CONFIG.replace(&format!("[{table}]"), &format!("[{table}]\n{setting}"));
        assert!(Runner::from_toml(&config).is_ok(), "{setting}");
        assert!(
            Runner::from_toml_for_model_selection(&config).is_err(),
            "{setting}"
        );
    }
}

#[tokio::test]
async fn composite_uses_host_client_and_only_changes_the_model() {
    struct Judge;
    #[async_trait]
    impl RoutedLlmClient for Judge {
        async fn call(&self, request: Request) -> Result<Response, LlmClientError> {
            assert_eq!(request.llm_request.model.as_deref(), Some("fast-model"));
            Ok(Response {
                llm_response: LlmResponse::Agg(text_response(
                    None,
                    r#"{"crux":"bounded test","primary_rule":"SUP-1","capability_boundary":"supported","p_solve":0.1}"#,
                )),
                metadata: None,
                upstream_headers: Default::default(),
            })
        }
    }
    let config = format!(
        r#"{CONFIG}
[routes.composed]
id = "composite-id"
type = "composite"
[routes.composed.classifier]
target = "fast"
base_threshold = 0.5
classify_trigger = "user_turn"
[routes.composed.stage]
efficient_target = "fast"
capable_target = "strong"
"#
    );
    let runner = Runner::from_toml(&config).unwrap();
    let request = Request {
        llm_request: text_request(Some("composite-id".into()), "Solve this task"),
        ..Default::default()
    };
    let outcome = runner
        .route("composite-id")
        .unwrap()
        .decide_with_client(request.clone(), Arc::new(Judge))
        .await
        .unwrap();
    let mut expected = request.llm_request;
    expected.model = Some("strong-model".into());
    assert_eq!(outcome.request.llm_request, expected);
    assert!(outcome.response.is_none());
}
