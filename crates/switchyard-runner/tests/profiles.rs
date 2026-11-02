// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};

use serde_json::json;
use switchyard_llm_client::RunObservation;
use switchyard_protocol::{Metadata, Request, text_request};
use switchyard_runner::{AlgorithmFactory, Runner, route_targets};
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const PROFILES: &str = r#"
[profiles.adaptive]
strategy = "stage_router"
capable_target = "strong"
efficient_target = "weak"
picker = "efficient_first"
confidence_threshold = 0.4
[profiles.direct]
strategy = "passthrough"
target = "spare"
[routes.default]
id = "auto"
profile = "adaptive"
[routes.default.selector]
kind = "decision"
target = "judge"
choices = { adaptive = "adaptive", direct = "direct" }
positive_choice = "direct"
[routes.default.selector.decision]
cutoff = 0.92
candidates = { adaptive = "weak", direct = "strong" }
criteria = { adaptive = "Adapt during execution", direct = "Use the direct profile" }
instructions = { task = "Choose the execution strategy" }
evidence = {}
"#;

fn source(server: &MockServer) -> String {
    format!(
        r#"
schema_version = 2
[llm_clients.answers]
format = "openai_chat"
base_url = "{0}/v1"
max_retries = 0
failure_cooldown_ms = 0
[targets]
strong = {{ id = "strong", llm_client = "answers", system_prompt = "STRONG PROMPT" }}
weak = {{ id = "weak", llm_client = "answers", system_prompt = "WEAK PROMPT" }}
spare = {{ id = "spare", llm_client = "answers" }}
[decision_clients.judge]
format = "openai_decisions"
endpoint = "{0}/v1/decisions"
api_key_env = "PROFILE_TEST_KEY_{1}"
timeout_ms = 2000
[decision_targets.judge]
id = "judge"
decision_client = "judge"
{PROFILES}
"#,
        server.uri(),
        server.address().port()
    )
}

fn runner(server: &MockServer, source: &str) -> Runner {
    let variable = format!("PROFILE_TEST_KEY_{}", server.address().port());
    // Each mock server has its own credential name.
    unsafe { std::env::set_var(&variable, "test-key") }
    let runner = Runner::from_toml(source);
    unsafe { std::env::remove_var(&variable) }
    runner.unwrap()
}

fn request(session: &str) -> Request {
    Request {
        llm_request: text_request(None, "Help with this task"),
        metadata: Some(Metadata {
            session_id: Some(session.into()),
            ..Default::default()
        }),
        ..Default::default()
    }
}

async fn decision(server: &MockServer, score: f64, status: u16) {
    Mock::given(method("POST"))
        .and(path("/v1/decisions"))
        .respond_with(ResponseTemplate::new(status).set_body_json(json!({
            "model": "judge", "usage": {"input_tokens": 10, "output_tokens": 1, "total_tokens": 11, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}, "answers": [{"type": "choice", "name": "route", "choice": "direct",
                "probabilities": [{"value": "direct", "probability": score},
                    {"value": "adaptive", "probability": 1.0 - score}]}]
        })))
        .expect(1)
        .mount(server)
        .await;
}

async fn answer(server: &MockServer, model: &'static str, status: u16, count: u64) {
    Mock::given(method("POST")).and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({"model": model})))
        .respond_with(ResponseTemplate::new(status).set_body_json(json!({
            "id": "answer", "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Answer"}, "finish_reason": "stop"}]
        }))).expect(count).mount(server).await;
}

#[tokio::test]
async fn selector_retains_strategies_and_isolates_fallbacks() {
    for (score, status, profile, strategy, selected) in [
        (0.92, 200, "adaptive", "stage_router", "weak"),
        (0.93, 200, "direct", "passthrough", "spare"),
        (0.99, 503, "adaptive", "stage_router", "weak"),
    ] {
        let server = MockServer::start().await;
        let runner = runner(&server, &source(&server));
        decision(&server, score, status).await;
        let adaptive = profile == "adaptive";
        answer(&server, "weak", 503, if adaptive { 2 } else { 0 }).await;
        answer(&server, "strong", 200, if adaptive { 2 } else { 0 }).await;
        answer(&server, "spare", 200, if adaptive { 0 } else { 2 }).await;
        let events = Arc::new(Mutex::new(Vec::new()));
        let observe = {
            let events = events.clone();
            Arc::new(move |event| {
                if let RunObservation::Outcome(metadata) = event {
                    events.lock().unwrap().push(metadata);
                }
            })
        };
        let route = runner.route("auto").unwrap();
        let (a, b) = tokio::join!(
            route.execute(request("same-session"), Some(observe.clone())),
            route.execute(request("same-session"), Some(observe))
        );
        assert_eq!(a.unwrap().selected_model, selected);
        assert_eq!(b.unwrap().selected_model, selected);
        for metadata in events.lock().unwrap().iter() {
            assert_eq!(metadata.profile.as_deref(), Some(profile));
            assert_eq!(metadata.strategy.as_deref(), Some(strategy));
        }
        {
            let events = events.lock().unwrap();
            let judgments: Vec<_> = events.iter().flat_map(|event| &event.judgments).collect();
            assert_eq!(judgments.len(), 1);
            if status == 200 {
                assert_eq!(judgments[0].evidence.as_ref().unwrap()["score"], score);
            }
        }
        if adaptive {
            for request in server.received_requests().await.unwrap() {
                if request.url.path() == "/v1/chat/completions" {
                    let body: serde_json::Value = request.body_json().unwrap();
                    let prompt = if body["model"] == "weak" {
                        "WEAK PROMPT"
                    } else {
                        "STRONG PROMPT"
                    };
                    assert_eq!(body["messages"][0]["content"], prompt);
                }
            }
        }
        server.verify().await;
    }
}

#[tokio::test]
async fn fixed_profiles_preserve_legacy_execution_without_a_selector() {
    let server = MockServer::start().await;
    let base = source(&server);
    let fixed = base.split("[routes.default.selector]").next().unwrap();
    let named = runner(&server, fixed);
    let inline = fixed
        .split("[profiles.adaptive]")
        .next()
        .unwrap()
        .to_string()
        + "[routes.default]\nid = 'auto'\nstrategy = 'stage_router'\ncapable_target = 'strong'\nefficient_target = 'weak'\npicker = 'efficient_first'\nconfidence_threshold = 0.4\n";
    let legacy = inline
        .replace("schema_version = 2", "schema_version = 1")
        .replace("strategy =", "type =");
    answer(&server, "weak", 200, 3).await;
    for runner in [named, runner(&server, &inline), runner(&server, &legacy)] {
        let result = runner
            .route("auto")
            .unwrap()
            .execute(request("fixed"), None)
            .await
            .unwrap();
        assert_eq!(result.selected_model, "weak");
    }
    assert_eq!(server.received_requests().await.unwrap().len(), 3);
}

#[tokio::test]
async fn answer_producing_profile_reuses_response_and_its_prompt_policy() {
    let server = MockServer::start().await;
    let source = source(&server).replace(
        "strategy = \"passthrough\"\ntarget = \"spare\"",
        "strategy = 'advisor'\nexecutor_target = 'strong'\nadvisor_target = 'weak'\ngate_trigger = 'pattern'\ngate_trigger_pattern = '^NEVER$'"
    );
    let runner = runner(&server, &source);
    decision(&server, 0.99, 200).await;
    answer(&server, "strong", 200, 1).await;
    let result = runner
        .route("auto")
        .unwrap()
        .execute(request("advisor"), None)
        .await
        .unwrap();
    assert_eq!(result.selected_model, "strong");
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 2);
    let body: serde_json::Value = requests[1].body_json().unwrap();
    assert_eq!(body["messages"][0]["content"], "STRONG PROMPT");
}

#[tokio::test]
async fn strict_selector_error_does_not_execute_a_profile() {
    let server = MockServer::start().await;
    let source = source(&server).replace(
        "positive_choice = \"direct\"",
        "positive_choice = 'direct'\non_error = 'error'",
    );
    let runner = runner(&server, &source);
    decision(&server, 0.99, 503).await;
    assert!(
        runner
            .route("auto")
            .unwrap()
            .execute(request("error"), None)
            .await
            .is_err()
    );
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[test]
fn rejects_invalid_profile_references_and_overrides() {
    let valid = "schema_version = 2\n[targets]\n[profiles.fixed]\nstrategy = 'noop'\n[routes.default]\nid = 'auto'\nprofile = 'fixed'\n";
    assert!(route_targets(valid).is_ok());
    for (source, expected) in [
        (
            valid.replace("profile = 'fixed'", "profile = 'missing'"),
            "unknown profile",
        ),
        (
            valid.to_string() + "target = 'strong'\n",
            "must be configured in profile",
        ),
        (
            valid.replace("strategy = 'noop'", "strategy = 'noop'\ntype = 'noop'"),
            "not both",
        ),
        (
            valid.replace("strategy = 'noop'", "strategy = 'noop'\nmagic = true"),
            "unknown field",
        ),
        (
            valid.replace("schema_version = 2", "schema_version = 1"),
            "profile",
        ),
    ] {
        let error = route_targets(&source).unwrap_err().to_string();
        assert!(error.contains(expected), "{error}");
    }
}

#[test]
fn example_builds_and_rejects_unsupported_selector_settings() {
    let source = include_str!("../../../examples/profile-routing.toml");
    let algorithms = AlgorithmFactory::from_toml(source)
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(algorithms["switchyard/adaptive"].0.name(), "stage_router");
    assert_eq!(algorithms["switchyard/composite"].0.name(), "composite");
    for (source, expected) in [
        (
            source.replace(
                "classify_trigger = \"new_session\"",
                "classify_trigger = 'user_turn'",
            ),
            "new_session",
        ),
        (
            source.replace(
                "positive_choice = \"planning\"",
                "positive_choice = 'missing'",
            ),
            "positive_choice",
        ),
        (
            source.replace(
                "choices = { adaptive = \"adaptive\", planning = \"planning\" }",
                "choices = { adaptive = 'adaptive' }",
            ),
            "two choices",
        ),
    ] {
        let error = AlgorithmFactory::from_toml(&source)
            .unwrap()
            .build()
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains(expected), "{error}");
    }
}

#[tokio::test]
async fn selected_vgr_requires_explicit_context_with_discovery() {
    let server = MockServer::start().await;
    let source = source(&server)
        .replace(
            "failure_cooldown_ms = 0",
            "failure_cooldown_ms = 0\ncontext_discovery = 'llama_cpp'",
        )
        .replace(
            "strategy = \"passthrough\"\ntarget = \"spare\"",
            "strategy = 'vgr'\nefficient_target = 'weak'\ncapable_target = 'strong'\njudge_target = 'spare'",
        );
    let variable = format!("PROFILE_TEST_KEY_{}", server.address().port());
    unsafe { std::env::set_var(&variable, "test-key") }
    let error = Runner::from_toml(&source).err().unwrap().to_string();
    unsafe { std::env::remove_var(&variable) }
    assert!(error.contains("explicit context_window"), "{error}");
    runner(
        &server,
        &source.replace("id = \"auto\"", "id = 'auto'\ncontext_window = 128000"),
    );
    assert!(server.received_requests().await.unwrap().is_empty());
}
