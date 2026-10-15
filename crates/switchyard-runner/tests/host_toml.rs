// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TOML routes retain execution policy when their clients belong to an embedding host.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::{Value, json};
use switchyard_llm_client::{AuxiliaryOperation, RequestSettings};
use switchyard_protocol::{
    LlmClientError, LlmResponse, Metadata, ModelId, Request, Response, RoutedLlmClient, WireFormat,
    text_request, text_response,
};
use switchyard_runner::{Runner, RunnerError};

type Calls = Arc<Mutex<Vec<Value>>>;

struct HostClient {
    settings: BTreeMap<ModelId, RequestSettings>,
    calls: Calls,
}

#[async_trait]
impl RoutedLlmClient for HostClient {
    async fn call(&self, request: Request) -> Result<Response, LlmClientError> {
        let (model, settings) = self
            .settings
            .get_key_value(request.llm_request.model.as_deref().unwrap())
            .unwrap();
        let body = settings.prepare_request_body(&request.llm_request, model)?;
        let answer = if model.as_str() == "judge" {
            let score = if body.to_string().contains("hard follow-up") {
                0.1
            } else {
                0.9
            };
            json!({"crux":"task", "primary_rule":"SUP-1",
                "capability_boundary":"supported", "p_solve":score})
            .to_string()
        } else {
            "host answer".to_string()
        };
        let is_failure = body["fail"] == true;
        self.calls.lock().unwrap().push(body);
        if is_failure {
            return Err(LlmClientError::UpstreamHttp {
                status: reqwest::StatusCode::SERVICE_UNAVAILABLE,
                body: "host provider unavailable".to_string(),
            });
        }
        let mut response = text_response(Some(model.to_string()), answer);
        response.id = Some(format!("resp_{model}"));
        if settings.wire_format == WireFormat::OpenAiResponses {
            response.preservation.responses.insert(
                WireFormat::OpenAiResponses.into(),
                json!({"id": response.id, "model": model.as_str(), "store":true}),
            );
        }
        Ok(Response {
            llm_response: LlmResponse::Agg(response),
            metadata: request.metadata,
            upstream_headers: Default::default(),
        })
    }
}

fn source(format: &str, fast_settings: &str) -> String {
    format!(
        r#"
schema_version = 1
[llm_clients]
primary = {{ format = "{format}" }}
other = {{ format = "openai_chat" }}
[targets]
judge = {{ id = "judge", llm_client = "primary" }}
[targets.fast]
id = "fast"
llm_client = "primary"
system_prompt = "fast prompt"
{fast_settings}
[targets.capable]
id = "capable"
llm_client = "other"
system_prompt = "capable prompt"
reasoning_effort = "low"
extra_body = {{ seed = 7 }}
omit_body_fields = ["temperature"]
[routes.coding]
id = "coding"
type = "llm_classifier"
classifier_target = "judge"
weak_target = "fast"
strong_target = "capable"
base_threshold = 0.5
"#
    )
}

fn load(source: &str) -> (Runner, Calls) {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut clients = Vec::new();
    let runner = Runner::from_toml_with_clients(source, |name, settings| {
        clients.push(name.to_string());
        Ok(Arc::new(HostClient {
            settings,
            calls: Arc::clone(&calls),
        }))
    })
    .unwrap();
    assert_eq!(clients, ["other", "primary"]);
    (runner, calls)
}

fn request(text: &str) -> Request {
    Request {
        llm_request: text_request(Some("coding".to_string()), text),
        ..Request::default()
    }
}

fn models(calls: &[Value]) -> Vec<&str> {
    calls
        .iter()
        .map(|body| body["model"].as_str().unwrap())
        .collect()
}

#[tokio::test]
async fn host_clients_receive_judge_and_independently_prepared_fallbacks() {
    // Session affinity survives fallback; each answer candidate receives its own policy.
    let configured = format!(
        "{}\nclassify_trigger = 'new_session'",
        source("openai_chat", "extra_body = { fail = true }")
    );
    let (runner, calls) = load(&configured);
    let route = runner.route("coding").unwrap();
    for text in ["hello", "hard follow-up"] {
        let mut ingress = request(text);
        ingress.llm_request.sampling.temperature = Some(0.8);
        ingress.metadata = Some(Metadata {
            session_id: Some("host-session".to_string()),
            ..Default::default()
        });
        let output = route.execute(ingress, None).await.unwrap();
        assert_eq!(output.selected_model, "fast");
        assert_eq!(output.response.served_model().unwrap(), "capable");
    }
    let calls = calls.lock().unwrap();
    assert_eq!(
        models(&calls),
        ["judge", "fast", "capable", "fast", "capable"]
    );
    assert!(!calls[0].to_string().contains("fast prompt"));
    assert_eq!(calls[1]["messages"][0]["content"], "fast prompt");
    assert_eq!(calls[2]["messages"][0]["content"], "capable prompt");
    assert_eq!(calls[2]["messages"].as_array().unwrap().len(), 2);
    assert_eq!(calls[2]["seed"], 7);
    assert_eq!(calls[2]["reasoning_effort"], "low");
    assert!(calls[2].get("fail").is_none());
    assert!(calls[2].get("temperature").is_none());
}

#[tokio::test]
async fn host_client_groups_preserve_responses_continuations() {
    // Native state pins provider IDs or materializes cross-format history before routing.
    for format in ["openai_chat", "openai_responses"] {
        let (runner, calls) = load(&source(format, ""));
        let route = runner.route("coding").unwrap();
        assert!(matches!(
            route
                .call_auxiliary(request("hello"), AuxiliaryOperation::ResponsesInputTokens)
                .await,
            Err(RunnerError::AuxiliaryUnsupported)
        ));
        let outcome = libsy::RoutingOutcome::route_to("fast".into(), vec![], request("hello"));
        assert!(
            runner
                .describe_decision(&"coding".into(), &outcome)
                .is_none()
        );
        let mut seed = request("hello");
        seed.llm_request.preservation.requests.insert(
            WireFormat::OpenAiResponses.into(),
            json!({"model":"coding", "input":"hello", "store":true}),
        );
        route.execute(seed, None).await.unwrap();
        let mut follow = request("hard follow-up");
        follow
            .llm_request
            .extensions
            .fields
            .insert("previous_response_id".to_string(), json!("resp_fast"));
        let output = route.execute(follow, None).await.unwrap();
        assert_eq!(output.selected_model, "fast");
        let calls = calls.lock().unwrap();
        assert_eq!(models(&calls), ["judge", "fast", "fast"]);
        let body = &calls[2];
        if format == "openai_chat" {
            assert_eq!(body["messages"][1]["content"], "hello");
            assert_eq!(body["messages"][2]["content"], "host answer");
            assert_eq!(body["messages"][3]["content"], "hard follow-up");
        } else {
            assert_eq!(body["previous_response_id"], "resp_fast");
            assert!(body.to_string().contains("hard follow-up"));
        }
    }
}

#[test]
fn host_loading_preserves_validation_and_rejects_transport_settings() {
    // Host-owned transport needs no URL, while native HTTP loading still requires one.
    let source = source("openai_chat", "");
    assert!(
        Runner::from_toml(&source)
            .err()
            .unwrap()
            .to_string()
            .contains("base_url is required")
    );
    for setting in [
        "api_key_env = 'unused'",
        "forward_auth = false",
        "extra_headers = {}",
        "max_retries = 0",
        "failure_cooldown_ms = 5000",
        "timeout_ms = 1",
        "context_discovery = 'llama_cpp'",
    ] {
        let configured = source.replace("primary = {", &format!("primary = {{ {setting},"));
        let result = Runner::from_toml_with_clients(&configured, |_, _| {
            panic!("invalid host config reached factory")
        });
        assert!(
            result
                .err()
                .unwrap()
                .to_string()
                .contains("owned by the host")
        );
    }
    for configured in [
        source.replace(
            "schema_version = 1",
            "schema_version = 1\nfallback_client = 'primary'",
        ),
        format!(
            "{source}\n[decision_clients.unsupported]\nformat = 'system_one'\nendpoint = 'https://example.test'\napi_key_env = 'unused'\ntimeout_ms = 1"
        ),
        source.replace("primary = {", "primary = { base_url = 'invalid',"),
        source.replace("llm_client = \"other\"", "llm_client = \"missing\""),
    ] {
        assert!(
            Runner::from_toml_with_clients(&configured, |_, _| {
                panic!("invalid host config reached factory")
            })
            .is_err()
        );
    }
}
