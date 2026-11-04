// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! These tests check cache boundaries through the Stage router and HTTP client.
//! The ignored live test requires CACHE_TEST_BASE_URL, CACHE_TEST_API_KEY, CACHE_TEST_MODEL,
//! CACHE_TEST_FORMAT, and CACHE_TEST_PRICING_FILE. It makes billable provider calls.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};
use switchyard_costs::Pricing;
use switchyard_libsy::{
    HandoffNoteConfig, PickerMode, RuntimeModels, StageRouter, StageRouterConfig,
};
use switchyard_llm_client::{
    Backend, ClientRouter, HttpBackendConfig, ModelConfig, TranslatingLlmClient, run,
};
use switchyard_protocol::{Category, Metadata, ModelId, Request, Usage, WireFormat};
use switchyard_translation::{TranslationEngine, TranslationPolicy};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;
const NOTE: &str = "[router-guidance] The router escalated this turn from the efficient tier. Reassess the failure and verify the solution.";

fn backend(format: WireFormat, base_url: String, api_key: Option<String>) -> Backend {
    let config = HttpBackendConfig {
        base_url,
        api_key,
        forward_auth: false,
        extra_headers: BTreeMap::new(),
        extra_body: BTreeMap::new(),
        omit_body_fields: BTreeSet::new(),
        reasoning_effort: None,
        max_retries: 0,
        failure_cooldown: Duration::ZERO,
        timeout: Some(Duration::from_secs(120)),
    };
    match format {
        WireFormat::AnthropicMessages => Backend::Anthropic(config),
        WireFormat::OpenAiChat => Backend::OpenAiChat(config),
        WireFormat::OpenAiResponses => Backend::OpenAiResponses(config),
    }
}

fn router(notes: bool) -> Result<Arc<StageRouter>, switchyard_libsy::LibsyError> {
    let mut config = StageRouterConfig::new(PickerMode::EfficientFirst, 0.5);
    config.capable_hold_turns = 3;
    if notes {
        config.handoff_notes = Some(HandoffNoteConfig::new(NOTE, None, true));
    }
    Ok(Arc::new(StageRouter::new(config)?))
}

fn models(model: &str) -> Arc<RuntimeModels> {
    Arc::new(RuntimeModels::new(HashMap::from([
        (Category::Capable, vec![ModelId::from(model)]),
        (Category::Efficient, vec![ModelId::from("unused-efficient")]),
        (
            Category::Any,
            vec![ModelId::from(model), ModelId::from("unused-efficient")],
        ),
    ])))
}

fn body(format: WireFormat, model: &str, prefix: &str, stream: bool) -> Value {
    let failure = "fatal runtime error: out of memory";
    match format {
        WireFormat::AnthropicMessages => json!({
            "model":model,"max_tokens":16,"stream":stream,
            "system":[{"type":"text","text":"Inspect the failed command. Reply briefly."}],
            "tools":[{"name":"Bash","description":"Run a command","input_schema":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}],
            "messages":[{"role":"user","content":[{"type":"text","text":prefix}]},
                {"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"Bash","input":{"command":"cargo test"}}]},
                {"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":failure,"is_error":true}]}]
        }),
        WireFormat::OpenAiChat => json!({
            "model":model,"max_completion_tokens":16,"stream":stream,
            "messages":[{"role":"developer","content":"Inspect the failed command. Reply briefly."},
                {"role":"user","content":prefix},
                {"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"Bash","arguments":"{ \"command\": \"cargo test\" }"}}]},
                {"role":"tool","tool_call_id":"call_1","content":failure}],
            "tools":[{"type":"function","function":{"name":"Bash","description":"Run a command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}]
        }),
        WireFormat::OpenAiResponses => json!({
            "model":model,"max_output_tokens":16,"stream":stream,"store":false,
            "instructions":"Inspect the failed command. Reply briefly.",
            "input":[{"role":"user","content":prefix},
                {"type":"function_call","call_id":"call_1","name":"Bash","arguments":"{ \"command\": \"cargo test\" }"},
                {"type":"function_call_output","call_id":"call_1","output":failure}],
            "tools":[{"type":"function","name":"Bash","description":"Run a command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}]
        }),
    }
}

async fn turn(
    algorithm: Arc<StageRouter>,
    client: Arc<TranslatingLlmClient>,
    format: WireFormat,
    body: &Value,
    session: &str,
    model: &str,
) -> Result<Usage, Box<dyn std::error::Error + Send + Sync>> {
    let request = Request {
        llm_request: TranslationEngine::default()
            .decode_request(format, body, &TranslationPolicy::default())?
            .request,
        metadata: Some(Metadata {
            wire_format: Some(format),
            session_id: Some(session.to_string()),
            ..Default::default()
        }),
        raw_request: Some(body.clone()),
    };
    let (selected, response) = run(
        algorithm,
        ClientRouter::single(client),
        request,
        models(model),
        None,
    )
    .await?;
    assert_eq!(
        selected.as_str(),
        model,
        "Stage must select the capable model"
    );
    Ok(response.llm_response.into_agg().await?.usage)
}

// Captured HTTP requests distinguish preserved content from the added cache marker and note.
#[tokio::test]
async fn stage_escalation_and_hold_keep_the_native_request_prefix() -> TestResult {
    for format in [
        WireFormat::AnthropicMessages,
        WireFormat::OpenAiChat,
        WireFormat::OpenAiResponses,
    ] {
        let server = MockServer::start().await;
        let model = "gpt-6-luna";
        let (route, response) = match format {
            WireFormat::AnthropicMessages => (
                "/v1/messages",
                json!({
                    "id":"msg_test","type":"message","role":"assistant","model":model,
                    "content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn",
                    "usage":{"input_tokens":5,"output_tokens":1}
                }),
            ),
            WireFormat::OpenAiChat => (
                "/chat/completions",
                json!({
                    "id":"chat_test","object":"chat.completion","model":model,
                    "choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
                    "usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}
                }),
            ),
            WireFormat::OpenAiResponses => (
                "/responses",
                json!({
                    "id":"resp_test","object":"response","model":model,"status":"completed",
                    "output":[{"type":"message","id":"msg_test","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],
                    "usage":{"input_tokens":5,"output_tokens":1,"total_tokens":6}
                }),
            ),
        };
        Mock::given(method("POST"))
            .and(path(route))
            .respond_with(ResponseTemplate::new(200).set_body_json(response))
            .expect(2)
            .mount(&server)
            .await;
        let client = Arc::new(TranslatingLlmClient::new(&[ModelConfig::new(
            model,
            backend(format, server.uri(), None),
            None,
        )])?);
        let algorithm = router(true)?;
        let original = body(format, model, "Synthetic repository content", false);
        for _ in 0..2 {
            turn(
                algorithm.clone(),
                client.clone(),
                format,
                &original,
                "session",
                model,
            )
            .await?;
        }
        let requests = server
            .received_requests()
            .await
            .ok_or("missing captured requests")?;
        let escalation: Value = serde_json::from_slice(&requests[0].body)?;
        let held: Value = serde_json::from_slice(&requests[1].body)?;
        let key = if format == WireFormat::OpenAiResponses {
            "input"
        } else {
            "messages"
        };
        assert_eq!(escalation["tools"], original["tools"]);
        match format {
            WireFormat::AnthropicMessages => {
                assert_eq!(escalation["system"], original["system"]);
                assert_eq!(
                    escalation[key][2]["content"][0]["cache_control"],
                    json!({"type":"ephemeral"})
                );
                assert_eq!(escalation[key][2]["content"][1]["text"], NOTE);
                assert!(
                    escalation[key][2]["content"][1]
                        .get("cache_control")
                        .is_none()
                );
                assert_eq!(
                    held[key][2]["content"]
                        .as_array()
                        .ok_or("held content")?
                        .len(),
                    1
                );
                assert_eq!(escalation[key][2]["content"][0], held[key][2]["content"][0]);
            }
            WireFormat::OpenAiChat | WireFormat::OpenAiResponses => {
                let original_items = original[key].as_array().ok_or("original items")?;
                let prefix_len = original_items.len() - 1;
                let actual = escalation[key].as_array().ok_or("actual items")?;
                assert_eq!(&actual[..prefix_len], &original_items[..prefix_len]);
                let field = if format == WireFormat::OpenAiResponses {
                    "output"
                } else {
                    "content"
                };
                assert_eq!(
                    actual[prefix_len][field][0]["text"],
                    original_items[prefix_len][field]
                );
                assert_eq!(
                    actual[prefix_len][field][0]["prompt_cache_breakpoint"],
                    json!({"mode":"explicit"})
                );
                assert_eq!(actual.last().ok_or("note")?["content"][0]["text"], NOTE);
                assert_eq!(held[key], original[key]);
                assert_eq!(escalation["instructions"], original["instructions"]);
            }
        }
    }
    Ok(())
}

// Provider usage checks cache reuse after the note disappears; matching request JSON alone cannot.
#[tokio::test]
#[ignore = "requires provider credentials and makes billable cache-write/read calls"]
async fn live_stage_cache_writes_and_reads() -> TestResult {
    let base = std::env::var("CACHE_TEST_BASE_URL")?;
    let key = std::env::var("CACHE_TEST_API_KEY")?;
    let model = std::env::var("CACHE_TEST_MODEL")?;
    let format = serde_json::from_value::<WireFormat>(json!(std::env::var("CACHE_TEST_FORMAT")?))?;
    let pricing = Pricing::load(std::path::Path::new(&std::env::var(
        "CACHE_TEST_PRICING_FILE",
    )?))?;
    let client = Arc::new(TranslatingLlmClient::new(&[ModelConfig::new(
        model.as_str(),
        backend(format, base, Some(key)),
        None,
    )])?);
    let stream = matches!(
        std::env::var("CACHE_TEST_STREAM")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    );
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    for notes in [false, true] {
        let prefix = format!(
            "Cache experiment {nonce} notes={notes}.\n{}",
            "fn inspect_repository() { verify_tests_and_preserve_requirements(); }\n".repeat(1200)
        );
        let mut request = body(format, &model, &prefix, stream);
        let algorithm = router(notes)?;
        let session = format!("cache-{nonce}-{notes}");
        let mut cold_write = 0;
        for step in 0..3 {
            if step == 2 {
                let key = if format == WireFormat::OpenAiResponses {
                    "input"
                } else {
                    "messages"
                };
                request[key].as_array_mut().ok_or("conversation")?.extend([
                    json!({"role":"assistant","content":"I will inspect the failure."}),
                    json!({"role":"user","content":"Continue checking the repository."}),
                ]);
            }
            let usage = turn(
                algorithm.clone(),
                client.clone(),
                format,
                &request,
                &session,
                &model,
            )
            .await?;
            let cost = pricing
                .cost(&model, &usage)
                .ok_or("missing provider cost or configured rates")?;
            println!(
                "{}",
                json!({"notes":notes,"step":step,"model":model,"format":format,"stream":stream,"usage":usage,"cost":cost})
            );
            if step == 0 {
                cold_write = usage.cache_creation_input_tokens().unwrap_or(0);
                assert!(
                    usage.cache_creation_input_tokens().is_some_and(|n| n > 0),
                    "provider must report a cache write for this fresh prefix"
                );
            } else {
                assert!(
                    usage
                        .cached_input_tokens()
                        .is_some_and(|n| n >= cold_write * 95 / 100),
                    "provider must reuse at least 95% of the cold cache write after the note disappears"
                );
            }
        }
    }
    Ok(())
}
