// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use async_trait::async_trait;
use std::collections::BTreeMap;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use switchyard_llm_client::ClientRouter;
use switchyard_protocol::{
    LlmClientError, LlmResponse, Message, Metadata, Request, Response, Role, RoutedLlmClient,
    text_request, text_response,
};
use switchyard_runner::{AlgorithmSpec, ModelCapabilities, Route, Runner};

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
[routes.composed]
id = "composite-id"
type = "composite"
[routes.composed.classifier]
target = "fast"
mode = "custom"
prompt = 'Choose efficient or capable for the current task.'
response_schema = '{"type":"object","properties":{"decision":{"type":"object","properties":{"target":{"type":"string","enum":["efficient","capable"]}},"required":["target"],"additionalProperties":false}},"required":["decision"],"additionalProperties":false}'
policy = { type = "target_selector", selector = "/decision/target" }
classify_trigger = "user_turn"
max_output_tokens = 384
[routes.composed.stage]
efficient_target = "fast"
capable_target = "strong"
"#;

struct CustomJudge {
    reply: Mutex<Option<&'static str>>,
    calls: AtomicUsize,
}

#[async_trait]
impl RoutedLlmClient for CustomJudge {
    async fn call(&self, _request: Request) -> Result<Response, LlmClientError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let reply = self
            .reply
            .lock()
            .unwrap()
            .ok_or_else(|| LlmClientError::General("judge unavailable".into()))?;
        Ok(Response {
            llm_response: LlmResponse::Agg(text_response(None, reply)),
            metadata: None,
            upstream_headers: Default::default(),
        })
    }
}

fn custom_judge(reply: Option<&'static str>) -> Arc<CustomJudge> {
    Arc::new(CustomJudge {
        reply: Mutex::new(reply),
        calls: AtomicUsize::new(0),
    })
}

fn custom_route(config: &str, judge: Arc<CustomJudge>) -> Route {
    let mut config: toml::Value = toml::from_str(config).unwrap();
    config["routes"]["composed"]
        .as_table_mut()
        .unwrap()
        .remove("id");
    let spec: AlgorithmSpec = config["routes"]["composed"].clone().try_into().unwrap();
    let targets = BTreeMap::from([
        ("fast".into(), "fast-model".into()),
        ("strong".into(), "strong-model".into()),
    ]);
    let (algorithm, models) = spec
        .build_with_runtime_models("composed", &targets)
        .unwrap();
    Route::new(
        algorithm,
        ClientRouter::single(judge),
        None,
        ModelCapabilities::default(),
        None,
        None,
        Vec::new(),
        models,
    )
}

fn request() -> Request {
    Request {
        llm_request: text_request(None, "Analyze a difficult task"),
        metadata: Some(Metadata {
            session_id: Some("a".into()),
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[tokio::test]
async fn custom_composite_reclassifies_user_turns_and_retains_tier_on_failure() {
    let judge = custom_judge(Some(r#"{"decision":{"target":"capable"}}"#));
    let route = custom_route(CONFIG, judge.clone());
    let mut request = request();
    let first = route.decide(request.clone()).await.unwrap();
    assert_eq!(first.selected_model_id().unwrap().as_str(), "strong-model");
    for reply in [
        None,
        Some("not JSON"),
        Some("{}"),
        Some(r#"{"decision":{"target":"judge"}}"#),
        Some(r#"{"decision":{"target":"other"}}"#),
    ] {
        *judge.reply.lock().unwrap() = reply;
        let outcome = route.decide(request.clone()).await.unwrap();
        assert_eq!(
            outcome.selected_model_id().unwrap().as_str(),
            "strong-model"
        );
        assert_eq!(
            outcome.metadata.unwrap().evidence.unwrap()["source"],
            "fail_open"
        );
    }
    *judge.reply.lock().unwrap() = None;
    request.metadata.as_mut().unwrap().session_id = Some("b".into());
    let outcome = route.decide(request.clone()).await.unwrap();
    assert_eq!(outcome.selected_model_id().unwrap().as_str(), "fast-model");
    request.metadata.as_mut().unwrap().session_id = Some("a".into());
    *judge.reply.lock().unwrap() = Some(r#"{"decision":{"target":"efficient"}}"#);
    let outcome = route.decide(request.clone()).await.unwrap();
    assert_eq!(outcome.selected_model_id().unwrap().as_str(), "fast-model");
}

#[tokio::test]
async fn custom_composite_retains_first_verdict_or_fallback() {
    for trigger in ["user_turn", "new_session"] {
        for hash_keyed in [false, true] {
            for reply in [
                Some(r#"{"decision":{"target":"capable"}}"#),
                None,
                Some("not JSON"),
            ] {
                let config = CONFIG.replace("user_turn", trigger).replace(
                    "mode = \"custom\"",
                    &format!("mode = \"custom\"\nmessage_hash_fallback = {hash_keyed}"),
                );
                let judge = custom_judge(reply);
                let route = custom_route(&config, judge.clone());
                let mut request = request();
                if hash_keyed {
                    request.metadata = None;
                }
                let expected = if reply.is_some_and(|reply| reply.contains("capable")) {
                    "strong-model"
                } else {
                    "fast-model"
                };
                for _ in 0..2 {
                    let outcome = route.decide(request.clone()).await.unwrap();
                    assert_eq!(outcome.selected_model_id().unwrap().as_str(), expected);
                    request
                        .llm_request
                        .messages
                        .push(Message::text(Role::Assistant, "Working"));
                    // A recovered judge must not change a new-session assignment.
                    *judge.reply.lock().unwrap() = Some(r#"{"decision":{"target":"capable"}}"#);
                }
                assert_eq!(
                    judge.calls.load(Ordering::SeqCst),
                    1,
                    "{trigger}, hash_keyed={hash_keyed}, reply={reply:?}"
                );
            }
        }
    }
}

#[tokio::test]
async fn custom_composite_fail_closed_rejects_unusable_verdicts() {
    let config = CONFIG
        .replace("mode = \"custom\"", "mode = \"custom\"\nfail_open = false")
        .replace(
            r#"["efficient","capable"]"#,
            r#"["efficient","capable","judge"]"#,
        );
    for retained in [false, true] {
        for reply in [
            None,
            Some("not JSON"),
            Some("{}"),
            Some(r#"{"decision":{"target":"judge"}}"#),
            Some(r#"{"decision":{"target":"other"}}"#),
        ] {
            let judge = custom_judge(Some(r#"{"decision":{"target":"capable"}}"#));
            let route = custom_route(&config, judge.clone());
            if retained {
                route.decide(request()).await.unwrap();
            }
            *judge.reply.lock().unwrap() = reply;
            assert!(
                route.decide(request()).await.is_err(),
                "retained={retained}, reply={reply:?}"
            );
        }
    }
}

#[test]
fn custom_composite_rejects_incompatible_or_missing_settings() {
    let config = CONFIG;
    assert!(Runner::from_toml(config).is_ok());
    for invalid in [
        config.replace(
            "mode = \"custom\"",
            "mode = \"custom\"\nbase_threshold = 0.5",
        ),
        config.replace("mode = \"custom\"", "mode = \"escalation\""),
        config.replace("mode = \"custom\"", "mode = \"capability\""),
        config.replace(
            "prompt = 'Choose efficient or capable for the current task.'",
            "",
        ),
        config.replace(
            "selector = \"/decision/target\"",
            "selector = \"/decision/target~2\"",
        ),
        config.replace(
            "classify_trigger = \"user_turn\"",
            "classify_trigger = \"every_request\"",
        ),
        config.replace("response_schema = '", "response_schema = 'not-json"),
    ] {
        assert!(Runner::from_toml(&invalid).is_err());
    }
}
