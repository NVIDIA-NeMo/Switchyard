// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configured factories enforce target requirements before host-owned calls.

use std::collections::BTreeMap;

use futures_util::StreamExt;
use libsy::{RoutingOutcome, State, Step, StepStream};
use serde_json::json;
use switchyard_protocol::{LlmResponse, Request, Response, text_request, text_response};
use switchyard_runner::{AlgorithmFactory, Runner, require_target_labels, route_targets};
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const MISSING_KEY: &str = "SWITCHYARD_FACTORY_TEST_KEY_THAT_IS_NOT_SET";
const STAGE: &str = "type = 'stage_router'\nefficient_target = 'fast'\ncapable_target = 'capable'\npicker = 'efficient_first'\nconfidence_threshold = 0.6";

fn deployment(scope: &str, algorithm: &str, judge_alternative: &str) -> String {
    let targets = [
        ("fast", "standard", &[] as &[&str]),
        ("capable", "standard", &[]),
        ("private_fast", "approved", &["private"]),
        ("private_capable", "approved", &["private"]),
        ("both_fast", "approved", &["private", "customer_data"]),
        ("both_capable", "approved", &["private", "customer_data"]),
        ("judge", "standard", &[]),
        ("approved_judge", "approved", &["private", "customer_data"]),
    ]
    .into_iter()
    .map(|(name, client, labels)| {
        let id = if matches!(name, "fast" | "private_fast") {
            "shared-model"
        } else {
            name
        };
        format!(
            "[targets.{name}]\nid = {id:?}\nllm_client = {client:?}\nlabels = {}\n",
            json!(labels)
        )
    })
    .collect::<String>();
    format!(
        r#"
schema_version = 1
[llm_clients.standard]
format = "openai_chat"
base_url = "http://127.0.0.1:1/v1"
api_key_env = "{MISSING_KEY}"
[llm_clients.approved]
format = "openai_chat"
base_url = "http://127.0.0.1:2/v1"
api_key_env = "{MISSING_KEY}"
{targets}
[decision_clients.scoring]
format = "openai_decisions"
endpoint = "http://127.0.0.1:1/v1/decisions"
api_key_env = "{MISSING_KEY}"
timeout_ms = 1000
[decision_targets.judge_decision]
id = "decision/standard"
decision_client = "scoring"
[decision_targets.approved_decision]
id = "decision/approved"
decision_client = "scoring"
labels = ["private", "customer_data"]
[routes.coding]
id = "switchyard/coding"
{algorithm}
[routes.coding.eligibility]
require_explicit_requirements = true
scope = "{scope}"
[routes.coding.eligibility.target_alternatives]
fast = ["private_fast", "both_fast"]
capable = ["private_capable", "both_capable"]
{judge_alternative}
"#
    )
}

fn request(task: Option<&str>, labels: Option<&[&str]>) -> Request {
    let mut request = Request {
        llm_request: text_request(None, "Summarize these notes."),
        ..Request::default()
    };
    request.metadata.get_or_insert_default().task_id = task.map(str::to_owned);
    if let Some(labels) = labels {
        require_target_labels(&mut request, labels).expect("trusted labels");
    }
    request
}

async fn outcome(mut stream: StepStream) -> RoutingOutcome {
    let Step::Done(result) = stream
        .next()
        .await
        .expect("routing step")
        .expect("successful routing")
    else {
        panic!("unexpected model call");
    };
    assert!(stream.next().await.is_none());
    *result
}

#[tokio::test]
async fn configured_factory_matches_runner_without_reading_credentials() {
    assert!(std::env::var_os(MISSING_KEY).is_none());
    let source = deployment("request", STAGE, "");
    let factory = AlgorithmFactory::from_toml(&source).expect("deployment");
    let configured = factory.build_configured().expect("no credentials needed");
    let native = Runner::from_toml(&source.replace(MISSING_KEY, "PATH")).expect("native route");
    let route = &configured["switchyard/coding"];
    let state: State = serde_json::from_value(json!({"host_marker": {"bool": true}})).unwrap();
    for (labels, expected) in [
        (&[] as &[&str], ["fast", "capable"]),
        (&["private", "customer_data"], ["both_fast", "both_capable"]),
    ] {
        let request = request(None, Some(labels));
        let native = native
            .route("switchyard/coding")
            .unwrap()
            .decide(request.clone())
            .await
            .unwrap();
        let selected = outcome(route.run_stream(request, state.clone()).unwrap()).await;
        assert_eq!(selected.selected_model_ids, expected);
        assert_eq!(selected.selected_model_ids, native.selected_model_ids);
        assert_eq!(
            selected.state.extra["host_marker"],
            state.extra["host_marker"]
        );
    }
}

#[tokio::test]
async fn host_resolves_alternatives_from_the_configured_handle() {
    let standard = MockServer::start().await;
    let approved = MockServer::start().await;
    for (server, client) in [(&standard, "standard"), (&approved, "approved")] {
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_partial_json(json!({"model": "shared-model"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"client": client})))
            .expect(1)
            .mount(server)
            .await;
    }
    let clients = BTreeMap::from([("standard", standard.uri()), ("approved", approved.uri())]);
    let source = deployment("request", STAGE, "");
    let configured = AlgorithmFactory::from_toml(&source)
        .unwrap()
        .build_configured()
        .unwrap();
    let route = &configured["switchyard/coding"];
    assert_eq!(route.target_models().len(), 6);
    let roles = route_targets(&source).unwrap();
    assert_eq!(
        roles[0]
            .answer_targets
            .iter()
            .map(|target| target.name.as_str())
            .collect::<Vec<_>>(),
        ["capable", "fast"]
    );
    assert!(roles[0].judge_targets.is_empty());

    let http = reqwest::Client::new();
    for (labels, expected) in [(&[] as &[&str], "standard"), (&["private"], "approved")] {
        let selected = outcome(
            route
                .run_stream(request(None, Some(labels)), State::default())
                .unwrap(),
        )
        .await;
        let target = &route.target_models()[&selected.selected_model_ids[0]];
        let response: serde_json::Value = http
            .post(format!(
                "{}/v1/chat/completions",
                clients[target.client.as_str()]
            ))
            .json(&json!({
                "model": target.id,
                "messages": [{"role": "user", "content": "Summarize these notes."}]
            }))
            .send()
            .await
            .unwrap()
            .error_for_status()
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(response["client"], expected);
    }
    standard.verify().await;
    approved.verify().await;
}

#[test]
fn invalid_requirements_fail_before_a_stream_is_created() {
    let configured = AlgorithmFactory::from_toml(&deployment("task", STAGE, ""))
        .unwrap()
        .build_configured()
        .unwrap();
    let route = &configured["switchyard/coding"];
    for request in [
        request(Some("task"), None),
        request(None, Some(&[])),
        request(Some("task"), Some(&["unapproved"])),
    ] {
        assert!(route.run_stream(request, State::default()).is_err());
    }
    let source = deployment("request", STAGE, "");
    let ungoverned = source.split("[routes.coding.eligibility]").next().unwrap();
    let configured = AlgorithmFactory::from_toml(ungoverned)
        .unwrap()
        .build_configured()
        .unwrap();
    assert!(
        configured["switchyard/coding"]
            .run_stream(request(None, Some(&[])), State::default())
            .is_err()
    );
}

#[tokio::test]
async fn task_requirements_accumulate_and_survive_unavailable_targets() {
    let source = deployment("task", STAGE, "");
    let configured = AlgorithmFactory::from_toml(&source)
        .unwrap()
        .build_configured()
        .unwrap();
    let route = &configured["switchyard/coding"];
    for (task, labels, expected) in [
        ("shared", &["private"] as &[&str], "private_fast"),
        ("shared", &["customer_data"], "both_fast"),
        ("shared", &[], "both_fast"),
        ("other", &[], "fast"),
    ] {
        let selected = outcome(
            route
                .run_stream(request(Some(task), Some(labels)), State::default())
                .unwrap(),
        )
        .await;
        assert_eq!(selected.selected_model_ids[0], expected);
    }
    for labels in [&["unapproved"] as &[&str], &[]] {
        assert!(
            route
                .run_stream(request(Some("shared"), Some(labels)), State::default())
                .is_err()
        );
    }
}

#[tokio::test]
async fn approved_judges_are_served_by_the_host() {
    for decision in [false, true] {
        let judge = if decision { "judge_decision" } else { "judge" };
        let settings = if decision {
            "[routes.coding.decision]\ncutoff = 0.7"
        } else {
            "base_threshold = 0.5"
        };
        let algorithm = format!(
            "type = 'llm_classifier'\nclassifier_target = '{judge}'\nstrong_target = 'capable'\nweak_target = 'fast'\nfail_open = false\n{settings}"
        );
        let alternative = if decision {
            "judge_decision = ['approved_decision']"
        } else {
            "judge = ['approved_judge']"
        };
        let configured =
            AlgorithmFactory::from_toml(&deployment("request", &algorithm, alternative))
                .unwrap()
                .build_configured()
                .unwrap();
        let route = &configured["switchyard/coding"];
        let mut stream = route
            .run_stream(
                request(None, Some(&["private", "customer_data"])),
                State::default(),
            )
            .unwrap();
        match stream.next().await.unwrap().unwrap() {
            Step::CallModel(call) => {
                assert!(!decision);
                assert_eq!(call.models, ["approved_judge"]);
                let target = &route.target_models()[&call.models[0]];
                assert_eq!(target.client, "approved");
                assert_eq!(target.id, "approved_judge");
                let response = Response {
                    llm_response: LlmResponse::Agg(text_response(
                        None,
                        r#"{"crux":"bounded task","primary_rule":"SUP-1","capability_boundary":"supported","p_solve":0.9}"#,
                    )),
                    metadata: None,
                    upstream_headers: Default::default(),
                };
                call.respond(std::future::ready(Ok(response)))
                    .await
                    .unwrap();
            }
            Step::CallDecision(call) => {
                assert!(decision);
                assert_eq!(call.model, "decision/approved");
                assert_eq!(call.request.model.as_ref(), Some(&call.model));
                let target = &route.target_models()[&call.model];
                assert_eq!(target.client, "scoring");
                assert_eq!(target.id, call.model);
                let response = serde_json::from_value(json!({
                    "answers": {
                        "route": {
                            "value": {
                                "type": "choice",
                                "data": {
                                    "selected": "efficient",
                                    "probabilities": {
                                        "efficient": 0.9,
                                        "capable": 0.1
                                    }
                                }
                            }
                        }
                    }
                }))
                .unwrap();
                call.respond(std::future::ready(Ok(response)))
                    .await
                    .unwrap();
            }
            Step::Done(_) => panic!("judge must be served by the host"),
        }
        assert_eq!(
            outcome(stream).await.selected_model_ids,
            ["both_fast", "both_capable"]
        );
    }
}
