// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicUsize, Ordering};

use switchyard_libsy::{
    ClassifierContractConfig, EscalationJudgeConfig, LlmClassifierConfig, LlmFallback,
    LlmTaskClassifier, Passthrough, PickerMode, StageRouter, StageRouterConfig,
    TaskClassifierConfig,
};
use switchyard_protocol::{
    LlmResponseChunk, LlmResponseStream, LlmResponseStreamEvent, ModelId, RoutedLlmClient, Usage,
    text_request, text_response,
};

use super::*;

enum ScriptedBehavior {
    Text(&'static str),
    EmptyBuffered,
    EmptyStream,
    FailingStream,
    PartialThenFailure,
    TransportFailure(&'static str),
}

struct ScriptedClient {
    behavior: ScriptedBehavior,
    calls: AtomicUsize,
}

fn scripted(behavior: ScriptedBehavior) -> Arc<ScriptedClient> {
    Arc::new(ScriptedClient {
        behavior,
        calls: AtomicUsize::new(0),
    })
}

#[async_trait::async_trait]
impl RoutedLlmClient for ScriptedClient {
    async fn call(&self, request: Request) -> Result<Response, LlmClientError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        match self.behavior {
            ScriptedBehavior::Text(text) => {
                let mut response = text_response(None, text);
                response.usage = Usage {
                    input_tokens: Some(11),
                    output_tokens: Some(7),
                    total_tokens: Some(18),
                    ..Usage::default()
                };
                Ok(Response {
                    llm_response: LlmResponse::Agg(response),
                    metadata: request.metadata,
                })
            }
            ScriptedBehavior::EmptyBuffered => Ok(Response {
                llm_response: LlmResponse::Agg(Default::default()),
                metadata: None,
            }),
            ScriptedBehavior::EmptyStream => Ok(Response {
                llm_response: LlmResponse::Stream(Box::pin(stream::empty())),
                metadata: None,
            }),
            ScriptedBehavior::FailingStream => {
                let stream: LlmResponseStream = Box::pin(stream::once(async {
                    Err(LlmClientError::Transport {
                        source: Box::new(std::io::Error::other("fallback stream failed")),
                    })
                }));
                Ok(Response {
                    llm_response: LlmResponse::Stream(stream),
                    metadata: None,
                })
            }
            ScriptedBehavior::PartialThenFailure => {
                let stream: LlmResponseStream = Box::pin(stream::iter(vec![
                    Ok(LlmResponseStreamEvent::from(LlmResponseChunk::TextDelta {
                        index: 0,
                        text: "partial".into(),
                    })),
                    Err(LlmClientError::Transport {
                        source: Box::new(std::io::Error::other("stream failed after a chunk")),
                    }),
                ]));
                Ok(Response {
                    llm_response: LlmResponse::Stream(stream),
                    metadata: None,
                })
            }
            ScriptedBehavior::TransportFailure(message) => Err(LlmClientError::Transport {
                source: Box::new(std::io::Error::other(message)),
            }),
        }
    }
}

fn fixed_target(name: &str) -> ModelId {
    ModelId::from(name)
}

fn runtime_with_algorithm(
    algorithm: Arc<dyn Algorithm>,
    fallback: Arc<ScriptedClient>,
    protocol: WireFormat,
) -> SwitchyardRuntime {
    runtime_with_algorithm_clients(algorithm, fallback, protocol, Vec::new())
}

fn runtime_with_algorithm_clients(
    algorithm: Arc<dyn Algorithm>,
    fallback: Arc<ScriptedClient>,
    protocol: WireFormat,
    clients: Vec<(&str, Arc<ScriptedClient>)>,
) -> SwitchyardRuntime {
    let mut targets = BTreeMap::from([(
        "fallback".into(),
        PreparedTargetBinding {
            client: fallback as Arc<dyn RoutedLlmClient>,
        },
    )]);
    for (name, client) in clients {
        targets.insert(
            name.to_string(),
            PreparedTargetBinding {
                client: client as Arc<dyn RoutedLlmClient>,
            },
        );
    }
    SwitchyardRuntime {
        algorithm,
        targets,
        default_targets: BTreeMap::from([(protocol, "fallback".into())]),
        translation: TranslationEngine::default(),
    }
}

fn request_with_session(protocol: WireFormat, session: Option<&str>) -> Request {
    Request {
        llm_request: text_request(Some("auto".into()), "fix the build"),
        raw_request: None,
        metadata: Some(Metadata {
            wire_format: Some(protocol),
            session_id: session.map(str::to_string),
            ..Metadata::default()
        }),
    }
}

fn stage_signal_relay_request(protocol: WireFormat) -> RelayRequest {
    let content = match protocol {
        WireFormat::OpenAiChat => json!({
            "model": "auto",
            "messages": [
                {"role": "user", "content": "fix the build"},
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": "{\"cmd\":\"cargo test\"}"
                        }
                    }]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "content": "fatal runtime error: out of memory"
                }
            ]
        }),
        WireFormat::OpenAiResponses => json!({
            "model": "auto",
            "input": [
                {"type": "message", "role": "user", "content": "fix the build"},
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "bash",
                    "arguments": "{\"cmd\":\"cargo test\"}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": "fatal runtime error: out of memory"
                }
            ]
        }),
        WireFormat::AnthropicMessages => json!({
            "model": "auto",
            "max_tokens": 128,
            "messages": [
                {"role": "user", "content": "fix the build"},
                {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": "call-1",
                        "name": "bash",
                        "input": {"cmd": "cargo test"}
                    }]
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": "call-1",
                        "content": "fatal runtime error: out of memory",
                        "is_error": true
                    }]
                }
            ]
        }),
    };

    RelayRequest {
        headers: Map::from_iter([(
            "x-switchyard-session-id".into(),
            json!(format!("stage-{}", protocol.as_str())),
        )]),
        content,
    }
}

#[test]
fn relay_gateway_placeholder_session_is_not_retained() {
    let fallback = scripted(ScriptedBehavior::Text("fallback"));
    let runtime = runtime_with_algorithm(
        Arc::new(Passthrough::new(ModelId::from("selected"))),
        fallback,
        WireFormat::OpenAiChat,
    );
    let request = RelayRequest {
        headers: Map::from_iter([
            ("x-nemo-relay-source".into(), json!("gateway")),
            ("x-nemo-relay-session-id".into(), json!("gateway-gateway")),
            ("x-dynamo-session-id".into(), json!("gateway-gateway")),
        ]),
        content: json!({
            "model": "router",
            "messages": [{"role": "user", "content": "hello"}]
        }),
    };

    let decoded = runtime
        .decode_request(WireFormat::OpenAiChat, &request, false)
        .unwrap();

    assert_eq!(decoded.metadata.unwrap().session_id, None);
}

#[test]
fn explicit_switchyard_session_overrides_relay_gateway_placeholder() {
    let fallback = scripted(ScriptedBehavior::Text("fallback"));
    let runtime = runtime_with_algorithm(
        Arc::new(Passthrough::new(ModelId::from("selected"))),
        fallback,
        WireFormat::OpenAiChat,
    );
    let request = RelayRequest {
        headers: Map::from_iter([
            ("x-switchyard-session-id".into(), json!("caller-session")),
            ("x-nemo-relay-source".into(), json!("gateway")),
            ("x-nemo-relay-session-id".into(), json!("gateway-gateway")),
        ]),
        content: json!({
            "model": "router",
            "messages": [{"role": "user", "content": "hello"}]
        }),
    };

    let decoded = runtime
        .decode_request(WireFormat::OpenAiChat, &request, false)
        .unwrap();

    assert_eq!(
        decoded.metadata.unwrap().session_id.as_deref(),
        Some("caller-session")
    );
}

#[tokio::test]
async fn buffered_finalization_failure_uses_fallback_once() {
    let selected = scripted(ScriptedBehavior::EmptyStream);
    let fallback = scripted(ScriptedBehavior::EmptyBuffered);
    let runtime = SwitchyardRuntime {
        algorithm: Arc::new(Passthrough::new(ModelId::from("selected"))),
        targets: BTreeMap::from([
            (
                "selected".into(),
                PreparedTargetBinding {
                    client: selected.clone(),
                },
            ),
            (
                "fallback".into(),
                PreparedTargetBinding {
                    client: fallback.clone(),
                },
            ),
        ]),
        default_targets: BTreeMap::from([(WireFormat::OpenAiChat, "fallback".into())]),
        translation: TranslationEngine::default(),
    };
    let mut marks = Vec::new();

    let response = runtime
        .execute_buffered(WireFormat::OpenAiChat, Request::default(), &mut marks)
        .await
        .expect("the buffered fallback response should be encoded");

    assert!(response.is_object());
    assert_eq!(selected.calls.load(Ordering::Relaxed), 1);
    assert_eq!(fallback.calls.load(Ordering::Relaxed), 1);
    let error = marks
        .iter()
        .find(|mark| mark.name == "switchyard.routing.error")
        .expect("finalization failure should emit an error mark");
    assert_eq!(error.data["non_http_kind"], "invalid_response");
    assert_eq!(
        marks
            .iter()
            .filter(|mark| mark.name == "switchyard.routing.fallback")
            .count(),
        1
    );
}

#[tokio::test]
async fn returned_events_replays_preserved_openai_chat_without_duplicate_terminal() {
    let content = json!({
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "model": "gpt-4o",
        "system_fingerprint": "fp_provider_specific",
        "choices": [{
            "index": 0,
            "delta": {"content": "Hi"},
            "finish_reason": null
        }]
    });
    let terminal = json!({
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    });
    let body = format!("data: {content}\n\ndata: {terminal}\n\ndata: [DONE]\n\n").into_bytes();
    let stream = switchyard_translation::decode_stream(
        stream::once(async move { Ok::<_, LlmClientError>(body) }),
        WireFormat::OpenAiChat,
    )
    .expect("provider SSE should decode");
    let response = Response {
        llm_response: LlmResponse::Stream(stream),
        metadata: None,
    };

    let replayed = returned_events(response, WireFormat::OpenAiChat)
        .await
        .expect("return stream should encode")
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("return stream should not fail");

    assert_eq!(replayed, vec![content, terminal]);
}

#[tokio::test]
async fn invalid_selected_stream_does_not_invoke_failing_fallback_twice() {
    let selected = scripted(ScriptedBehavior::EmptyStream);
    let fallback = scripted(ScriptedBehavior::FailingStream);
    let runtime = SwitchyardRuntime {
        algorithm: Arc::new(Passthrough::new(ModelId::from("selected"))),
        targets: BTreeMap::from([
            (
                "selected".into(),
                PreparedTargetBinding {
                    client: selected.clone(),
                },
            ),
            (
                "fallback".into(),
                PreparedTargetBinding {
                    client: fallback.clone(),
                },
            ),
        ]),
        default_targets: BTreeMap::from([(WireFormat::OpenAiChat, "fallback".into())]),
        translation: TranslationEngine::default(),
    };
    let (output, _messages) = async_channel::bounded(32);

    let error = runtime
        .execute_stream(WireFormat::OpenAiChat, Request::default(), &output)
        .await
        .expect_err("the failing fallback stream must fail the request");

    assert_eq!(error, "trusted fallback stream: provider transport failure");
    assert_eq!(selected.calls.load(Ordering::Relaxed), 1);
    assert_eq!(fallback.calls.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn failing_fallback_call_flushes_error_and_fallback_marks() {
    let selected = scripted(ScriptedBehavior::EmptyStream);
    let fallback = scripted(ScriptedBehavior::TransportFailure("fallback call failed"));
    let runtime = SwitchyardRuntime {
        algorithm: Arc::new(Passthrough::new(ModelId::from("selected"))),
        targets: BTreeMap::from([
            (
                "selected".into(),
                PreparedTargetBinding {
                    client: selected.clone(),
                },
            ),
            (
                "fallback".into(),
                PreparedTargetBinding {
                    client: fallback.clone(),
                },
            ),
        ]),
        default_targets: BTreeMap::from([(WireFormat::OpenAiChat, "fallback".into())]),
        translation: TranslationEngine::default(),
    };
    let (output, messages) = async_channel::bounded(32);

    let error = runtime
        .execute_stream(WireFormat::OpenAiChat, Request::default(), &output)
        .await
        .expect_err("the failing fallback call must fail the request");

    assert_eq!(error, "trusted fallback: provider transport failure");
    assert_eq!(selected.calls.load(Ordering::Relaxed), 1);
    assert_eq!(fallback.calls.load(Ordering::Relaxed), 1);
    let mut terminal_marks = Vec::new();
    while let Ok(message) = messages.try_recv() {
        if let StreamMessage::Mark(mark) = message
            && matches!(
                mark.name.as_str(),
                "switchyard.routing.error" | "switchyard.routing.fallback"
            )
        {
            terminal_marks.push(mark.name);
        }
    }
    assert_eq!(
        terminal_marks,
        ["switchyard.routing.error", "switchyard.routing.fallback"]
    );
}

#[tokio::test]
async fn committed_stream_failure_does_not_fallback() {
    let selected = scripted(ScriptedBehavior::PartialThenFailure);
    let fallback = scripted(ScriptedBehavior::Text("fallback"));
    let runtime = runtime_with_algorithm_clients(
        Arc::new(Passthrough::new(ModelId::from("selected"))),
        fallback.clone(),
        WireFormat::OpenAiChat,
        vec![("selected", selected.clone())],
    );
    let (output, messages) = async_channel::bounded(32);

    let error = runtime
        .execute_stream(WireFormat::OpenAiChat, Request::default(), &output)
        .await
        .expect_err("committed stream failures must reject the stream");

    assert_eq!(
        error,
        "Switchyard stream failed after response commitment: provider transport failure"
    );
    assert_eq!(selected.calls.load(Ordering::Relaxed), 1);
    assert_eq!(fallback.calls.load(Ordering::Relaxed), 0);
    let mut emitted_event = false;
    while let Ok(message) = messages.try_recv() {
        emitted_event |= matches!(message, StreamMessage::Event(_));
    }
    assert!(emitted_event);
}

#[tokio::test]
async fn capability_classifier_emits_judge_usage_without_serving_usage() {
    let weak = scripted(ScriptedBehavior::Text("weak answer"));
    let strong = scripted(ScriptedBehavior::Text("strong answer"));
    let judge = scripted(ScriptedBehavior::Text(
        r#"{"crux":"bounded","primary_rule":"SUP-1","capability_boundary":"supported","p_solve":0.9}"#,
    ));
    let fallback = scripted(ScriptedBehavior::Text("fallback"));
    let algorithm = LlmTaskClassifier::new(LlmClassifierConfig::Capability {
        judge_target: fixed_target("judge"),
        efficient_target: fixed_target("weak"),
        capable_target: fixed_target("strong"),
        config: TaskClassifierConfig {
            base_threshold: 0.5,
            ..TaskClassifierConfig::default()
        },
    })
    .unwrap();
    let runtime = runtime_with_algorithm_clients(
        Arc::new(algorithm),
        fallback,
        WireFormat::OpenAiChat,
        vec![
            ("weak", weak.clone()),
            ("strong", strong.clone()),
            ("judge", judge.clone()),
        ],
    );
    let mut marks = Vec::new();

    runtime
        .execute_buffered(
            WireFormat::OpenAiChat,
            request_with_session(WireFormat::OpenAiChat, Some("capability")),
            &mut marks,
        )
        .await
        .unwrap();

    assert_eq!(judge.calls.load(Ordering::Relaxed), 1);
    assert_eq!(weak.calls.load(Ordering::Relaxed), 1);
    assert_eq!(strong.calls.load(Ordering::Relaxed), 0);
    let routing_calls = marks
        .iter()
        .filter(|mark| mark.name == "switchyard.routing.llm_call")
        .collect::<Vec<_>>();
    assert_eq!(routing_calls.len(), 1);
    assert_eq!(routing_calls[0].data["selected_target"], "judge");
    assert_eq!(routing_calls[0].data["usage"]["total_tokens"], 18);
}

#[tokio::test]
async fn escalation_buffers_weak_stream_then_latches_the_session_to_strong() {
    let weak = scripted(ScriptedBehavior::Text("weak draft"));
    let strong = scripted(ScriptedBehavior::Text("strong answer"));
    let judge = scripted(ScriptedBehavior::Text(
        r#"{"escalate":true,"reason":"stuck"}"#,
    ));
    let fallback = scripted(ScriptedBehavior::Text("fallback"));
    let algorithm = LlmTaskClassifier::new(LlmClassifierConfig::Escalation {
        judge_target: fixed_target("judge"),
        efficient_target: fixed_target("weak"),
        capable_target: fixed_target("strong"),
        contract: ClassifierContractConfig::default(),
        config: EscalationJudgeConfig {
            confirmations: 1,
            ..EscalationJudgeConfig::default()
        },
        max_output_tokens: 128,
    })
    .unwrap();
    let runtime = runtime_with_algorithm_clients(
        Arc::new(algorithm),
        fallback.clone(),
        WireFormat::OpenAiChat,
        vec![
            ("weak", weak.clone()),
            ("strong", strong.clone()),
            ("judge", judge.clone()),
        ],
    );

    let mut first = request_with_session(WireFormat::OpenAiChat, Some("session-1"));
    first.llm_request.stream = true;
    let (output, messages) = async_channel::bounded(32);
    runtime
        .execute_stream(WireFormat::OpenAiChat, first, &output)
        .await
        .unwrap();
    let mut streamed = Vec::new();
    let mut routing_calls = Vec::new();
    while let Ok(message) = messages.try_recv() {
        match message {
            StreamMessage::Event(event) => streamed.push(event),
            StreamMessage::Mark(mark) if mark.name == "switchyard.routing.llm_call" => {
                routing_calls.push(mark.data)
            }
            StreamMessage::Mark(_) => {}
        }
    }
    assert!(!streamed.is_empty());
    assert!(
        streamed
            .iter()
            .any(|event| event.to_string().contains("strong answer"))
    );
    assert_eq!(routing_calls.len(), 2);
    assert_eq!(routing_calls[0]["selected_target"], "weak");
    assert_eq!(routing_calls[0]["call_role"], "routing");
    assert_eq!(routing_calls[0]["usage"]["total_tokens"], 18);
    assert_eq!(routing_calls[1]["selected_target"], "judge");
    assert_eq!(routing_calls[1]["call_role"], "routing");
    assert_eq!(routing_calls[1]["usage"]["total_tokens"], 18);
    assert!(
        routing_calls
            .iter()
            .all(|call| call["selected_target"] != "strong")
    );

    let mut marks = Vec::new();
    let response = runtime
        .execute_buffered(
            WireFormat::OpenAiChat,
            request_with_session(WireFormat::OpenAiChat, Some("session-1")),
            &mut marks,
        )
        .await
        .unwrap();
    assert!(response.to_string().contains("strong answer"));
    assert_eq!(weak.calls.load(Ordering::Relaxed), 1);
    assert_eq!(judge.calls.load(Ordering::Relaxed), 1);
    assert_eq!(strong.calls.load(Ordering::Relaxed), 2);
    assert_eq!(fallback.calls.load(Ordering::Relaxed), 0);
    assert!(
        !marks
            .iter()
            .any(|mark| mark.name == "switchyard.routing.llm_call")
    );
    assert!(marks.iter().any(|mark| {
        mark.name == "switchyard.routing.decision"
            && mark.data["selected_target"] == "strong"
            && mark.metadata["session_id"] == "session-1"
    }));
}

#[tokio::test]
async fn escalation_judge_failure_falls_open_to_the_buffered_weak_response() {
    let weak = scripted(ScriptedBehavior::Text("weak answer"));
    let strong = scripted(ScriptedBehavior::Text("strong answer"));
    let judge = scripted(ScriptedBehavior::TransportFailure("scripted failure"));
    let fallback = scripted(ScriptedBehavior::Text("fallback"));
    let algorithm = LlmTaskClassifier::new(LlmClassifierConfig::Escalation {
        judge_target: fixed_target("judge"),
        efficient_target: fixed_target("weak"),
        capable_target: fixed_target("strong"),
        contract: ClassifierContractConfig::default(),
        config: EscalationJudgeConfig::default(),
        max_output_tokens: 128,
    })
    .unwrap();
    let runtime = runtime_with_algorithm_clients(
        Arc::new(algorithm),
        fallback.clone(),
        WireFormat::OpenAiChat,
        vec![
            ("weak", weak.clone()),
            ("strong", strong.clone()),
            ("judge", judge.clone()),
        ],
    );
    let mut marks = Vec::new();

    let response = runtime
        .execute_buffered(
            WireFormat::OpenAiChat,
            request_with_session(WireFormat::OpenAiChat, Some("session-1")),
            &mut marks,
        )
        .await
        .unwrap();

    assert!(response.to_string().contains("weak answer"));
    assert_eq!(weak.calls.load(Ordering::Relaxed), 1);
    assert_eq!(judge.calls.load(Ordering::Relaxed), 1);
    assert_eq!(strong.calls.load(Ordering::Relaxed), 0);
    assert_eq!(fallback.calls.load(Ordering::Relaxed), 0);
    let routing_calls = marks
        .iter()
        .filter(|mark| mark.name == "switchyard.routing.llm_call")
        .collect::<Vec<_>>();
    assert_eq!(routing_calls.len(), 1);
    assert_eq!(routing_calls[0].data["selected_target"], "judge");
    assert_eq!(routing_calls[0].data["call_role"], "routing");
    assert_eq!(routing_calls[0].data["outcome"], "error");
    assert!(routing_calls[0].data["usage"].is_null());
}

#[tokio::test]
async fn escalation_without_session_identity_cannot_accumulate_confirmations() {
    let weak = scripted(ScriptedBehavior::Text("weak answer"));
    let strong = scripted(ScriptedBehavior::Text("strong answer"));
    let judge = scripted(ScriptedBehavior::Text(
        r#"{"escalate":true,"reason":"stuck"}"#,
    ));
    let fallback = scripted(ScriptedBehavior::Text("fallback"));
    let algorithm = LlmTaskClassifier::new(LlmClassifierConfig::Escalation {
        judge_target: fixed_target("judge"),
        efficient_target: fixed_target("weak"),
        capable_target: fixed_target("strong"),
        contract: ClassifierContractConfig::default(),
        config: EscalationJudgeConfig {
            confirmations: 2,
            ..EscalationJudgeConfig::default()
        },
        max_output_tokens: 128,
    })
    .unwrap();
    let runtime = runtime_with_algorithm_clients(
        Arc::new(algorithm),
        fallback.clone(),
        WireFormat::OpenAiChat,
        vec![
            ("weak", weak.clone()),
            ("strong", strong.clone()),
            ("judge", judge.clone()),
        ],
    );

    for _ in 0..2 {
        let mut marks = Vec::new();
        let response = runtime
            .execute_buffered(
                WireFormat::OpenAiChat,
                request_with_session(WireFormat::OpenAiChat, None),
                &mut marks,
            )
            .await
            .unwrap();
        assert!(response.to_string().contains("weak answer"));
    }
    assert_eq!(weak.calls.load(Ordering::Relaxed), 2);
    assert_eq!(judge.calls.load(Ordering::Relaxed), 2);
    assert_eq!(strong.calls.load(Ordering::Relaxed), 0);
    assert_eq!(fallback.calls.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn stage_router_uses_tool_signals_for_every_managed_protocol() {
    for protocol in [
        WireFormat::OpenAiChat,
        WireFormat::OpenAiResponses,
        WireFormat::AnthropicMessages,
    ] {
        let capable = scripted(ScriptedBehavior::Text("capable answer"));
        let efficient = scripted(ScriptedBehavior::Text("efficient answer"));
        let fallback = scripted(ScriptedBehavior::Text("fallback"));
        let algorithm = StageRouter::new(
            fixed_target("strong"),
            fixed_target("weak"),
            StageRouterConfig::new(PickerMode::EfficientFirst, 0.5),
        )
        .unwrap();
        let runtime = runtime_with_algorithm_clients(
            Arc::new(algorithm),
            fallback.clone(),
            protocol,
            vec![("strong", capable.clone()), ("weak", efficient.clone())],
        );
        let mut marks = Vec::new();
        let relay_request = stage_signal_relay_request(protocol);
        let request = runtime
            .decode_request(protocol, &relay_request, false)
            .unwrap();

        let response = runtime
            .execute_buffered(protocol, request, &mut marks)
            .await
            .unwrap();

        assert!(response.to_string().contains("capable answer"));
        assert_eq!(capable.calls.load(Ordering::Relaxed), 1);
        assert_eq!(efficient.calls.load(Ordering::Relaxed), 0);
        assert_eq!(fallback.calls.load(Ordering::Relaxed), 0);
        assert!(marks.iter().any(|mark| {
            mark.name == "switchyard.routing.decision"
                && mark.data["algorithm"] == "stage_router"
                && mark.data["attempt"] == 1
                && mark.data["selected_target"] == "strong"
                && mark.metadata["session_id"] == format!("stage-{}", protocol.as_str())
        }));
    }
}

#[tokio::test]
async fn stage_router_falls_open_to_each_picker_default_without_tool_history() {
    for (picker, expected) in [
        (PickerMode::CapableFirst, "strong"),
        (PickerMode::EfficientFirst, "weak"),
    ] {
        let capable = scripted(ScriptedBehavior::Text("strong"));
        let efficient = scripted(ScriptedBehavior::Text("weak"));
        let fallback = scripted(ScriptedBehavior::Text("fallback"));
        let algorithm = StageRouter::new(
            fixed_target("strong"),
            fixed_target("weak"),
            StageRouterConfig::new(picker, 0.5),
        )
        .unwrap();
        let runtime = runtime_with_algorithm_clients(
            Arc::new(algorithm),
            fallback,
            WireFormat::OpenAiChat,
            vec![("strong", capable), ("weak", efficient)],
        );
        let mut marks = Vec::new();

        runtime
            .execute_buffered(
                WireFormat::OpenAiChat,
                request_with_session(WireFormat::OpenAiChat, None),
                &mut marks,
            )
            .await
            .unwrap();

        assert!(marks.iter().any(|mark| {
            mark.name == "switchyard.routing.decision" && mark.data["selected_target"] == expected
        }));
    }
}

#[tokio::test]
async fn stage_router_classifier_resolves_an_ambiguous_turn() {
    let capable = scripted(ScriptedBehavior::Text("strong"));
    let efficient = scripted(ScriptedBehavior::Text("weak"));
    let judge = scripted(ScriptedBehavior::Text(
        r#"{"crux":"bounded","primary_rule":"SUP-1","capability_boundary":"supported","p_solve":0.9}"#,
    ));
    let fallback = scripted(ScriptedBehavior::Text("fallback"));
    let mut config = StageRouterConfig::new(PickerMode::CapableFirst, 0.5);
    config.llm_fallback = Some(LlmFallback {
        judge_target: fixed_target("judge"),
        config: TaskClassifierConfig {
            base_threshold: 0.5,
            ..TaskClassifierConfig::default()
        },
    });
    let algorithm = StageRouter::new(fixed_target("strong"), fixed_target("weak"), config).unwrap();
    let runtime = runtime_with_algorithm_clients(
        Arc::new(algorithm),
        fallback.clone(),
        WireFormat::OpenAiChat,
        vec![
            ("strong", capable.clone()),
            ("weak", efficient.clone()),
            ("judge", judge.clone()),
        ],
    );
    let mut marks = Vec::new();

    runtime
        .execute_buffered(
            WireFormat::OpenAiChat,
            request_with_session(WireFormat::OpenAiChat, Some("stage-classifier")),
            &mut marks,
        )
        .await
        .unwrap();

    assert_eq!(judge.calls.load(Ordering::Relaxed), 1);
    assert_eq!(efficient.calls.load(Ordering::Relaxed), 1);
    assert_eq!(capable.calls.load(Ordering::Relaxed), 0);
    assert_eq!(fallback.calls.load(Ordering::Relaxed), 0);
    let routing_calls = marks
        .iter()
        .filter(|mark| mark.name == "switchyard.routing.llm_call")
        .collect::<Vec<_>>();
    assert_eq!(routing_calls.len(), 1);
    assert_eq!(routing_calls[0].data["selected_target"], "judge");
    assert_eq!(routing_calls[0].data["call_role"], "routing");
    assert_eq!(routing_calls[0].data["outcome"], "ok");
    assert_eq!(routing_calls[0].data["usage"]["total_tokens"], 18);
    assert!(marks.iter().any(|mark| {
        mark.name == "switchyard.routing.decision" && mark.data["selected_target"] == "weak"
    }));
}
