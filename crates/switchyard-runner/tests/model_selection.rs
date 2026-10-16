// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use async_trait::async_trait;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use switchyard_protocol::{
    LlmClientError, LlmRequest, LlmResponse, Message, Metadata, Request, Response, Role,
    RoutedLlmClient, text_response,
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

fn composite_config() -> String {
    format!(
        r#"{CONFIG}
[routes.composed]
id = "composite-id"
type = "composite"
[routes.composed.classifier]
target = "fast"
base_threshold = 0.5
classify_trigger = "user_turn"
max_output_tokens = 384
[routes.composed.stage]
efficient_target = "fast"
capable_target = "strong"
"#
    )
}

#[test]
fn composite_selection_rejects_subagents_and_prompt_mutation() {
    let config = composite_config();
    assert!(Runner::from_toml_for_model_selection(&config).is_ok());
    let with_subagents =
        format!("{config}\n[routes.composed.subagents]\ntype = 'passthrough'\ntarget = 'strong'");
    assert!(Runner::from_toml(&with_subagents).is_ok());
    assert!(Runner::from_toml_for_model_selection(&with_subagents).is_err());
    let with_notes = format!(
        "{config}\n[routes.composed.stage.handoff_notes]\nescalation_note = 'Try a different approach'"
    );
    assert!(Runner::from_toml(&with_notes).is_ok());
    assert!(Runner::from_toml_for_model_selection(&with_notes).is_err());
}

#[tokio::test]
async fn composite_uses_host_judge_and_retains_tier_between_user_turns() {
    struct Judge(AtomicUsize);
    #[async_trait]
    impl RoutedLlmClient for Judge {
        async fn call(&self, request: Request) -> Result<Response, LlmClientError> {
            assert_eq!(request.llm_request.model.as_deref(), Some("fast-model"));
            assert_eq!(request.llm_request.output.max_output_tokens, Some(384));
            let probability = if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
                0.1
            } else {
                0.9
            };
            Ok(Response {
                llm_response: LlmResponse::Agg(text_response(None, serde_json::json!({
                    "crux": "bounded test", "primary_rule": "SUP-1", "capability_boundary": "supported", "p_solve": probability,
                }).to_string())),
                metadata: None,
                upstream_headers: Default::default(),
            })
        }
    }
    let runner = Runner::from_toml_for_model_selection(&composite_config()).unwrap();
    let route = runner.route("composite-id").unwrap();
    let judge = Arc::new(Judge(AtomicUsize::new(0)));
    let mut request = Request {
        llm_request: LlmRequest {
            model: Some("composite-id".into()),
            messages: vec![Message::text(Role::User, "Solve this task")],
            ..Default::default()
        },
        metadata: Some(Metadata {
            session_id: Some("host-session".into()),
            ..Default::default()
        }),
        ..Default::default()
    };
    for (tail, expected) in [
        (None, "strong-model"),
        (Some(Role::Assistant), "strong-model"),
        (Some(Role::User), "fast-model"),
    ] {
        if let Some(role) = tail {
            request
                .llm_request
                .messages
                .push(Message::text(role, "Follow-up"));
        }
        let outcome = route
            .decide_with_client(request.clone(), judge.clone())
            .await
            .unwrap();
        assert_eq!(outcome.selected_model_id().unwrap().as_str(), expected);
        assert!(outcome.response.is_none());
        let mut original = request.llm_request.clone();
        original.model = Some(expected.into());
        assert_eq!(outcome.request.llm_request, original);
    }
    assert_eq!(judge.0.load(Ordering::SeqCst), 2);
}

fn custom_composite_config() -> String {
    composite_config().replace(
        "base_threshold = 0.5\nclassify_trigger",
        r#"mode = "custom"
prompt = 'Choose efficient or capable for the current task.'
response_schema = '{"type":"object","properties":{"target":{"type":"string","enum":["efficient","capable"]}},"required":["target"],"additionalProperties":false}'
policy = { type = "target_selector", selector = "/target" }
classify_trigger"#,
    )
}

struct CustomJudge {
    reply: Option<&'static str>,
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
        reply,
        calls: AtomicUsize::new(0),
    })
}

#[tokio::test]
async fn custom_composite_preserves_tier_across_continuations_and_judge_failure() {
    let runner = Runner::from_toml_for_model_selection(&custom_composite_config()).unwrap();
    let route = runner.route("composite-id").unwrap();
    let mut request = Request {
        llm_request: LlmRequest {
            model: Some("composite-id".into()),
            messages: vec![Message::text(Role::User, "Analyze a difficult task")],
            ..Default::default()
        },
        metadata: Some(Metadata {
            session_id: Some("a".into()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let capable = custom_judge(Some(r#"{"target":"capable"}"#));
    assert_eq!(
        route
            .decide_with_client(request.clone(), capable.clone())
            .await
            .unwrap()
            .selected_model_id()
            .unwrap()
            .as_str(),
        "strong-model"
    );
    let mut continuation = request.clone();
    continuation
        .llm_request
        .messages
        .push(Message::text(Role::Assistant, "Working"));
    assert_eq!(
        route
            .decide_with_client(continuation, capable.clone())
            .await
            .unwrap()
            .selected_model_id()
            .unwrap()
            .as_str(),
        "strong-model"
    );
    assert_eq!(capable.calls.load(Ordering::SeqCst), 1);

    for reply in [
        None,
        Some("not JSON"),
        Some("{}"),
        Some(r#"{"target":"judge"}"#),
        Some(r#"{"target":"other"}"#),
    ] {
        let outcome = route
            .decide_with_client(request.clone(), custom_judge(reply))
            .await
            .unwrap();
        assert_eq!(
            outcome.selected_model_id().unwrap().as_str(),
            "strong-model"
        );
        assert_eq!(
            outcome.metadata.unwrap().evidence.unwrap()["source"],
            "fail_open"
        );
    }
    request.metadata.as_mut().unwrap().session_id = Some("b".into());
    assert_eq!(
        route
            .decide_with_client(request.clone(), custom_judge(None))
            .await
            .unwrap()
            .selected_model_id()
            .unwrap()
            .as_str(),
        "fast-model"
    );
    request.metadata.as_mut().unwrap().session_id = Some("a".into());
    let outcome = route
        .decide_with_client(
            request.clone(),
            custom_judge(Some(r#"{"target":"efficient"}"#)),
        )
        .await
        .unwrap();
    let mut expected = request.llm_request;
    expected.model = Some("fast-model".into());
    assert_eq!(outcome.request.llm_request, expected);
    assert!(outcome.response.is_none());
}

#[tokio::test]
async fn custom_composite_respects_new_session_and_fail_closed() {
    let config = custom_composite_config().replace(
        "classify_trigger = \"user_turn\"",
        "classify_trigger = \"new_session\"",
    );
    let runner = Runner::from_toml_for_model_selection(&config).unwrap();
    let route = runner.route("composite-id").unwrap();
    let request = Request {
        llm_request: LlmRequest {
            messages: vec![Message::text(Role::User, "Task")],
            ..Default::default()
        },
        metadata: Some(Metadata {
            session_id: Some("a".into()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let judge = custom_judge(Some(r#"{"target":"capable"}"#));
    for _ in 0..2 {
        route
            .decide_with_client(request.clone(), judge.clone())
            .await
            .unwrap();
    }
    assert_eq!(judge.calls.load(Ordering::SeqCst), 1);
    let closed = custom_composite_config()
        .replace("mode = \"custom\"", "mode = \"custom\"\nfail_open = false");
    let runner = Runner::from_toml_for_model_selection(&closed).unwrap();
    assert!(
        runner
            .route("composite-id")
            .unwrap()
            .decide_with_client(request, custom_judge(None))
            .await
            .is_err()
    );
}

#[test]
fn custom_composite_rejects_incompatible_or_missing_settings() {
    let config = custom_composite_config();
    assert!(Runner::from_toml_for_model_selection(&config).is_ok());
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
        config.replace("selector = \"/target\"", "selector = \"/target~2\""),
        config.replace(
            "classify_trigger = \"user_turn\"",
            "classify_trigger = \"every_request\"",
        ),
        config.replace("response_schema = '", "response_schema = 'not-json"),
    ] {
        assert!(Runner::from_toml_for_model_selection(&invalid).is_err());
    }
}
