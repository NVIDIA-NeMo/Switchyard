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
    LlmClientError, LlmRequest, LlmResponse, Message, Metadata, Request, Response, Role,
    RoutedLlmClient, text_response,
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
    async fn call(&self, request: Request) -> Result<Response, LlmClientError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        assert!(
            request.llm_request.instructions[0]
                .content
                .iter()
                .any(|block| {
                    matches!(block, switchyard_protocol::ContentBlock::Text { text }
                if text == "Choose efficient or capable for the current task.")
                })
        );
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
        llm_request: LlmRequest {
            messages: vec![Message::text(Role::User, "Analyze a difficult task")],
            ..Default::default()
        },
        metadata: Some(Metadata {
            session_id: Some("a".into()),
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[tokio::test]
async fn custom_composite_preserves_tier_across_continuations_and_judge_failure() {
    let judge = custom_judge(Some(r#"{"decision":{"target":"capable"}}"#));
    let route = custom_route(CONFIG, judge.clone());
    let mut request = request();
    let first = route.decide(request.clone()).await.unwrap();
    assert_eq!(first.selected_model_id().unwrap().as_str(), "strong-model");
    let mut continuation = request.clone();
    continuation
        .llm_request
        .messages
        .push(Message::text(Role::Assistant, "Working"));
    let outcome = route.decide(continuation).await.unwrap();
    assert_eq!(
        outcome.selected_model_id().unwrap().as_str(),
        "strong-model"
    );
    assert_eq!(judge.calls.load(Ordering::SeqCst), 1);

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
    let mut expected = request.llm_request;
    expected.model = Some("fast-model".into());
    assert_eq!(outcome.request.llm_request, expected);
    assert!(outcome.response.is_none());
}

#[tokio::test]
async fn custom_composite_respects_new_session_and_fail_closed() {
    let config = CONFIG.replace(
        "classify_trigger = \"user_turn\"",
        "classify_trigger = \"new_session\"",
    );
    let judge = custom_judge(Some(r#"{"decision":{"target":"capable"}}"#));
    let route = custom_route(&config, judge.clone());
    for _ in 0..2 {
        let outcome = route.decide(request()).await.unwrap();
        assert_eq!(
            outcome.selected_model_id().unwrap().as_str(),
            "strong-model"
        );
    }
    assert_eq!(judge.calls.load(Ordering::SeqCst), 1);
    let closed = CONFIG.replace("mode = \"custom\"", "mode = \"custom\"\nfail_open = false");
    let route = custom_route(&closed, custom_judge(None));
    assert!(route.decide(request()).await.is_err());

    let judge = custom_judge(Some(r#"{"decision":{"target":"capable"}}"#));
    let route = custom_route(&closed, judge.clone());
    route.decide(request()).await.unwrap();
    // Invalid verdicts abstain independently of client-error recovery.
    for reply in ["not JSON", "{}", r#"{"decision":{"target":"other"}}"#] {
        *judge.reply.lock().unwrap() = Some(reply);
        let outcome = route.decide(request()).await.unwrap();
        assert_eq!(
            outcome.selected_model_id().unwrap().as_str(),
            "strong-model"
        );
        assert_eq!(
            outcome.metadata.unwrap().evidence.unwrap()["source"],
            "retained"
        );
    }
}

#[tokio::test]
async fn custom_composite_retains_tier_with_message_hash_fallback() {
    for trigger in ["user_turn", "new_session"] {
        let config = CONFIG.replace("user_turn", trigger).replace(
            "mode = \"custom\"",
            "mode = \"custom\"\nmessage_hash_fallback = true",
        );
        let judge = custom_judge(Some(r#"{"decision":{"target":"capable"}}"#));
        let route = custom_route(&config, judge.clone());
        let mut request = request();
        request.metadata = None;
        for _ in 0..2 {
            let outcome = route.decide(request.clone()).await.unwrap();
            assert_eq!(
                outcome.selected_model_id().unwrap().as_str(),
                "strong-model"
            );
            request
                .llm_request
                .messages
                .push(Message::text(Role::Assistant, "Working"));
        }
        assert_eq!(judge.calls.load(Ordering::SeqCst), 1);
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
