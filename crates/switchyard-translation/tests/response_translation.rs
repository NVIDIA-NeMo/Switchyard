// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for buffered response translation between provider formats.

pub mod common;

use pretty_assertions::assert_eq;
use serde_json::json;
use switchyard_translation::{
    PreservationPolicy, TranslationEngine, TranslationPolicy, WireFormat,
};

use common::{
    REASONING_MODEL, normalized_policy, shell_tool_call, text_and_encrypted_reasoning_details,
};

type TestResult = std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>;

#[test]
fn responses_native_output_is_preserved_or_rejected() -> TestResult {
    use switchyard_translation::{StreamTranslationState, TranslationError};
    let engine = TranslationEngine::default();
    let source = WireFormat::OpenAiResponses;
    let items = [
        json!({"type": "apply_patch_call", "id": "ap_1", "call_id": "call_1", "status": "completed",
            "operation": {"type": "create_file", "path": "probe.txt", "diff": "+probe\n"}}),
        json!({"type": "shell_call", "id": "sh_1", "call_id": "call_1", "status": "completed",
            "action": {"commands": ["pwd"], "timeout_ms": 1000, "max_output_length": 1024}}),
        json!({"type": "computer_call", "id": "cu_1", "call_id": "call_1", "status": "completed",
            "action": {"type": "click", "x": 10, "y": 20, "button": "left"}, "pending_safety_checks": []}),
        json!({"type": "image_generation_call", "id": "ig_1", "status": "completed", "result": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l2QAAAAASUVORK5CYII="}),
    ];
    let text = json!({"type": "message", "role": "assistant", "content": [
        {"type": "output_text", "text": "Working."}
    ]});
    let mut failures = Vec::new();
    for item in items {
        let kind = item["type"].as_str().ok_or("missing item type")?;
        let body = json!({"id": "resp_native", "model": "provider", "status": "completed", "output": [item]});
        for policy in [TranslationPolicy::default(), normalized_policy()] {
            let decoded = engine.decode_response(source, &body, &policy)?.response;
            let replay = engine.encode_response(source, &decoded, &policy)?.body;
            if replay["output"] != body["output"] {
                failures.push(format!("{kind}: same-format buffered output lost"));
            }
            for target in [WireFormat::OpenAiChat, WireFormat::AnthropicMessages] {
                for output in [json!([item]), json!([text, item])] {
                    let mut body = body.clone();
                    body["output"] = output;
                    let result = engine.translate_response(source, target, &body, &policy);
                    if !matches!(result, Err(TranslationError::UnsupportedTranslation { .. })) {
                        failures.push(format!("{kind}: {target:?} buffered did not reject"));
                    }
                }
                if !matches!(
                    engine.encode_response(target, &decoded, &policy),
                    Err(TranslationError::UnsupportedTranslation { .. })
                ) {
                    failures.push(format!(
                        "{kind}: {target:?} separate buffered encode did not reject"
                    ));
                }
            }
        }
        let mut initial = item.clone();
        initial["status"] = json!("in_progress");
        let added =
            json!({"type": "response.output_item.added", "output_index": 0, "item": initial});
        let done = json!({"type": "response.output_item.done", "output_index": 0, "item": item});
        let completed = json!({"type": "response.completed", "response": body});
        let incomplete = json!({"type": "response.incomplete", "response": {
            "id": "resp_native", "status": "incomplete", "output": [item],
            "incomplete_details": {"reason": "max_output_tokens"}
        }});
        for frames in [
            vec![added, done.clone(), completed.clone()],
            vec![done, completed.clone()],
            vec![completed.clone()],
            vec![
                json!({"type": "response.output_text.delta", "output_index": 1, "delta": "Working."}),
                completed,
            ],
            vec![incomplete],
        ] {
            let mut decoder = StreamTranslationState::default();
            let mut replay_state = StreamTranslationState::default();
            let mut replay = Vec::new();
            for frame in &frames {
                let event = engine.decode_stream_event(&mut decoder, source, frame.clone())?;
                replay.extend(engine.encode_stream_event(&mut replay_state, source, event)?);
            }
            replay.extend(engine.finish_stream(&mut replay_state, source)?);
            assert_eq!(replay, frames, "{kind}: same-format stream");
            for target in [WireFormat::OpenAiChat, WireFormat::AnthropicMessages] {
                for is_preserved in [false, true] {
                    let mut decoder = StreamTranslationState::default();
                    let mut encoder = StreamTranslationState::default();
                    let mut events = Vec::new();
                    for frame in &frames {
                        let encoded = if is_preserved {
                            let event =
                                engine.decode_stream_event(&mut decoder, source, frame.clone())?;
                            engine.encode_stream_event(&mut encoder, target, event)?
                        } else {
                            engine.translate_event(&mut encoder, source, target, frame)?
                        };
                        events.extend(encoded);
                    }
                    events.extend(engine.finish_stream(&mut encoder, target)?);
                    let errors = events
                        .iter()
                        .filter(|event| event["error"].is_object())
                        .count();
                    let last = events.last().ok_or("missing terminal error")?;
                    if !encoder.errored
                        || errors != 1
                        || !last["error"]["message"]
                            .as_str()
                            .is_some_and(|message| message.contains("not supported"))
                        || events.iter().any(|event| {
                            event["type"] == "message_stop"
                                || event["choices"][0]["finish_reason"].is_string()
                        })
                    {
                        failures.push(format!("{kind}: {target:?} stream did not terminate with one unsupported-translation error"));
                    }
                    assert!(engine.finish_stream(&mut encoder, target)?.is_empty());
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

#[test]
fn url_citations_survive_chat_responses_translation() -> TestResult {
    let engine = TranslationEngine::default();
    let citation = json!({
        "start_index": 13, "end_index": 20,
        "url": "https://example.test/source", "title": "Example"
    });
    let chat_annotations = json!([{"type": "url_citation", "url_citation": citation}]);
    let mut flat_citation = citation;
    flat_citation["type"] = json!("url_citation");
    let responses_annotations = json!([flat_citation]);
    let chat = json!({
        "choices": [{"message": {
            "role": "assistant", "content": "According to Example.",
            "annotations": chat_annotations
        }, "finish_reason": "stop"}]
    });
    let responses = json!({
        "status": "completed",
        "output": [{"type": "message", "role": "assistant", "content": [{
            "type": "output_text", "text": "According to Example.",
            "annotations": responses_annotations
        }]}]
    });
    for policy in [TranslationPolicy::default(), normalized_policy()] {
        let output = engine
            .translate_response(
                WireFormat::OpenAiChat,
                WireFormat::OpenAiResponses,
                &chat,
                &policy,
            )?
            .body;
        assert_eq!(
            output["output"][0]["content"][0]["annotations"],
            responses_annotations
        );
        let output = engine
            .translate_response(
                WireFormat::OpenAiResponses,
                WireFormat::OpenAiChat,
                &responses,
                &policy,
            )?
            .body;
        assert_eq!(
            output["choices"][0]["message"]["annotations"],
            chat_annotations
        );

        // Responses text parts and messages are concatenated in Chat.
        let mut multipart = responses.clone();
        multipart["output"][0]["content"]
            .as_array_mut()
            .unwrap()
            .insert(0, json!({"type": "output_text", "text": "🌍 "}));
        multipart["output"].as_array_mut().unwrap().insert(
            0,
            json!({"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "é"}
            ]}),
        );
        let output = engine
            .translate_response(
                WireFormat::OpenAiResponses,
                WireFormat::OpenAiChat,
                &multipart,
                &policy,
            )?
            .body;
        let mut shifted = chat_annotations.clone();
        shifted[0]["url_citation"]["start_index"] = json!(16);
        shifted[0]["url_citation"]["end_index"] = json!(23);
        assert_eq!(
            output["choices"][0]["message"]["content"],
            "é🌍 According to Example."
        );
        assert_eq!(output["choices"][0]["message"]["annotations"], shifted);
    }
    Ok(())
}

// Verifies OpenAI Chat responses map to Anthropic message responses.
#[test]
fn openai_chat_response_translates_to_anthropic_message() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "id": "chatcmpl-test",
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello world"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    });

    let output = engine
        .translate_response(
            WireFormat::OpenAiChat,
            WireFormat::AnthropicMessages,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(output["type"], "message");
    assert_eq!(output["role"], "assistant");
    assert_eq!(output["model"], "gpt-4o");
    assert_eq!(
        output["content"],
        json!([{"type": "text", "text": "Hello world"}])
    );
    assert_eq!(output["stop_reason"], "end_turn");
    assert_eq!(
        output["usage"],
        json!({"input_tokens": 10, "output_tokens": 5})
    );
    Ok(())
}

// Verifies Anthropic message responses map to OpenAI Chat completions.
#[test]
fn anthropic_message_response_translates_to_openai_chat_completion() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet",
        "content": [{"type": "text", "text": "Hi there"}],
        "stop_reason": "max_tokens",
        "usage": {"input_tokens": 12, "output_tokens": 7}
    });

    let output = engine
        .translate_response(
            WireFormat::AnthropicMessages,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(output["object"], "chat.completion");
    assert_eq!(output["model"], "claude-sonnet");
    assert_eq!(output["choices"][0]["message"]["content"], "Hi there");
    assert_eq!(output["choices"][0]["finish_reason"], "length");
    assert_eq!(
        output["usage"],
        json!({"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19})
    );
    Ok(())
}

// Verifies Responses usage details survive when translating back to Chat Completions.
#[test]
fn responses_reasoning_usage_translates_to_openai_chat_usage_details() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "id": "resp_test",
        "object": "response",
        "model": "gpt-reasoning",
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Visible answer"}]
        }],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 2},
            "output_tokens_details": {"reasoning_tokens": 3}
        }
    });

    let output = engine
        .translate_response(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(output["usage"]["prompt_tokens"], 10);
    assert_eq!(output["usage"]["completion_tokens"], 5);
    assert_eq!(
        output["usage"]["completion_tokens_details"],
        json!({"reasoning_tokens": 3})
    );
    assert_eq!(
        output["usage"]["prompt_tokens_details"]["cache_creation_tokens"],
        2
    );
    let decoded = engine
        .decode_response(WireFormat::OpenAiResponses, &body, &normalized_policy())?
        .response;
    assert_eq!(decoded.usage.input_tokens, Some(4));
    assert_eq!(decoded.usage.cached_input_tokens(), Some(4));
    assert_eq!(decoded.usage.cache_creation_input_tokens(), Some(2));
    assert_eq!(decoded.usage.total_tokens, Some(15));
    let mut aliased = body.clone();
    aliased["usage"] = output["usage"].clone();
    assert_eq!(
        engine
            .decode_response(WireFormat::OpenAiResponses, &aliased, &normalized_policy())?
            .response
            .usage,
        decoded.usage
    );
    let encoded = engine
        .encode_response(WireFormat::OpenAiResponses, &decoded, &normalized_policy())?
        .body;
    assert_eq!(encoded["usage"], body["usage"]);

    let mut state = switchyard_translation::StreamTranslationState::new(
        WireFormat::OpenAiResponses,
        WireFormat::OpenAiChat,
    );
    let events = engine.translate_event(
        &mut state,
        WireFormat::OpenAiResponses,
        WireFormat::OpenAiChat,
        &json!({"type": "response.completed", "response": body}),
    )?;
    assert_eq!(state.usage, decoded.usage);
    let chat = events
        .iter()
        .find(|event| event.get("usage").is_some())
        .ok_or("missing usage")?;
    assert_eq!(chat["usage"], output["usage"]);
    let mut state = switchyard_translation::StreamTranslationState::new(
        WireFormat::OpenAiChat,
        WireFormat::OpenAiResponses,
    );
    let mut events = engine.translate_event(
        &mut state,
        WireFormat::OpenAiChat,
        WireFormat::OpenAiResponses,
        &json!({"id": "chat_usage", "model": "model", "usage": output["usage"],
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
    )?;
    events.extend(engine.finish_stream(&mut state, WireFormat::OpenAiResponses)?);
    let completed = events
        .iter()
        .find(|event| event["type"] == "response.completed")
        .ok_or("missing completion")?;
    assert_eq!(completed["response"]["usage"], body["usage"]);
    Ok(())
}

// Verifies Responses reasoning and its following message remain one semantic assistant output.
#[test]
fn responses_reasoning_and_message_preserve_the_final_answer() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "id": "resp_test",
        "object": "response",
        "model": "gpt-reasoning",
        "status": "completed",
        "output": [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "private reasoning"}]
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Visible answer"}]
            }
        ]
    });

    let anthropic = engine
        .translate_response(
            WireFormat::OpenAiResponses,
            WireFormat::AnthropicMessages,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;
    assert_eq!(
        anthropic["content"],
        json!([
            {"type": "thinking", "thinking": "private reasoning", "signature": ""},
            {"type": "text", "text": "Visible answer"}
        ])
    );

    let chat = engine
        .translate_response(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;
    assert_eq!(chat["choices"][0]["message"]["content"], "Visible answer");
    assert_eq!(
        chat["choices"][0]["message"]["reasoning_content"],
        "private reasoning"
    );
    Ok(())
}

// Verifies OpenAI cache usage survives the Chat-to-Responses translation used by Codex.
#[test]
fn openai_chat_cache_usage_translates_to_responses_usage_details() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "id": "chatcmpl-test",
        "model": "gpt-cached",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Cached answer"},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "total_tokens": 105,
            "prompt_tokens_details": {"cached_tokens": 80}
        }
    });

    let output = engine
        .translate_response(
            WireFormat::OpenAiChat,
            WireFormat::OpenAiResponses,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(output["usage"]["input_tokens"], 100);
    assert_eq!(
        output["usage"]["input_tokens_details"],
        json!({"cached_tokens": 80, "cache_write_tokens": 0})
    );
    Ok(())
}

// Verifies OpenRouter's cache-write field and the legacy alias normalize identically.
#[test]
fn openai_chat_cache_write_aliases_translate_to_anthropic_usage_fields() -> TestResult {
    let engine = TranslationEngine::default();
    for cache_write_field in ["cache_write_tokens", "cache_creation_tokens"] {
        let mut body = json!({
            "id": "chatcmpl-test",
            "model": "gpt-cached",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Cached answer"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 5,
                "total_tokens": 105,
                "prompt_tokens_details": {"cached_tokens": 70}
            }
        });
        body["usage"]["prompt_tokens_details"][cache_write_field] = json!(10);

        let output = engine
            .translate_response(
                WireFormat::OpenAiChat,
                WireFormat::AnthropicMessages,
                &body,
                &TranslationPolicy::default(),
            )?
            .body;

        assert_eq!(output["usage"]["input_tokens"], 20);
        assert_eq!(output["usage"]["cache_read_input_tokens"], 70);
        assert_eq!(output["usage"]["cache_creation_input_tokens"], 10);
        assert_eq!(output["usage"]["output_tokens"], 5);
    }
    Ok(())
}

// Verifies Anthropic thinking response blocks become OpenAI reasoning_content.
#[test]
fn anthropic_thinking_response_translates_to_openai_reasoning_content() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus",
        "content": [
            {"type": "thinking", "thinking": "private reasoning"},
            {"type": "text", "text": "Visible answer"}
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 7}
    });

    let output = engine
        .translate_response(
            WireFormat::AnthropicMessages,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    let message = &output["choices"][0]["message"];
    assert_eq!(message["content"], "Visible answer");
    assert_eq!(message["reasoning_content"], "private reasoning");
    for (details, thinking_tokens) in [
        (json!({"thinking_tokens": 0}), 0),
        (json!({"thinking_tokens": 5}), 5),
        (json!({"thinking_tokens": null, "reasoning_tokens": 5}), 5),
    ] {
        let mut body = body.clone();
        body["usage"]["output_tokens_details"] = details;
        let decoded = engine
            .decode_response(WireFormat::AnthropicMessages, &body, &normalized_policy())?
            .response;
        assert_eq!(decoded.usage.reasoning_tokens, Some(thinking_tokens));
        assert_eq!(decoded.usage.output_tokens, Some(7));
        assert_eq!(decoded.usage.total_tokens, Some(19));

        for (target, details, field) in [
            (
                WireFormat::OpenAiChat,
                "completion_tokens_details",
                "reasoning_tokens",
            ),
            (
                WireFormat::OpenAiResponses,
                "output_tokens_details",
                "reasoning_tokens",
            ),
            (
                WireFormat::AnthropicMessages,
                "output_tokens_details",
                "thinking_tokens",
            ),
        ] {
            let output = engine
                .translate_response(
                    WireFormat::AnthropicMessages,
                    target,
                    &body,
                    &normalized_policy(),
                )?
                .body;
            assert_eq!(output["usage"][details][field], thinking_tokens);
            let mut state = switchyard_translation::StreamTranslationState::new(
                WireFormat::AnthropicMessages,
                target,
            );
            engine.translate_event(
                &mut state,
                WireFormat::AnthropicMessages,
                target,
                &json!({"type": "message_start", "message": {
                    "id": "msg_test", "model": "claude-opus", "usage": {"input_tokens": 12}
                }}),
            )?;
            engine.translate_event(
                &mut state,
                WireFormat::AnthropicMessages,
                target,
                &json!({"type": "message_delta", "delta": {}, "usage": body["usage"]}),
            )?;
            let mut events = engine.translate_event(
                &mut state,
                WireFormat::AnthropicMessages,
                target,
                &json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 7}}),
            )?;
            assert_eq!(state.usage.reasoning_tokens, Some(thinking_tokens));
            assert_eq!(state.usage.output_tokens, Some(7));
            events.extend(engine.finish_stream(&mut state, target)?);
            let usage = events
                .iter()
                .find_map(|event| {
                    event
                        .get("usage")
                        .or_else(|| event.get("response")?.get("usage"))
                })
                .ok_or("missing terminal usage")?;
            assert_eq!(usage[details][field], thinking_tokens);
        }
    }
    Ok(())
}

// Verifies OpenAI reasoning_content becomes a separate Responses reasoning item.
#[test]
fn openai_reasoning_response_translates_to_responses_reasoning_item() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "id": "chatcmpl-test",
        "model": "gpt-reasoning",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "reasoning_content": "private reasoning",
                "content": "Visible answer"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": 3,
            "total_tokens": 7,
            "completion_tokens_details": {"reasoning_tokens": 2}
        }
    });

    let output = engine
        .translate_response(
            WireFormat::OpenAiChat,
            WireFormat::OpenAiResponses,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(output["output"][0]["type"], "reasoning");
    assert_eq!(
        output["output"][0]["summary"][0],
        json!({"type": "summary_text", "text": "private reasoning"})
    );
    assert_eq!(output["output"][1]["type"], "message");
    assert_eq!(output["output"][1]["content"][0]["text"], "Visible answer");
    assert_eq!(
        output["usage"]["output_tokens_details"],
        json!({"reasoning_tokens": 2})
    );
    Ok(())
}

#[test]
fn openai_reasoning_response_preserves_visible_content_presence() -> TestResult {
    let engine = TranslationEngine::default();
    for content in [json!(null), json!(""), json!("Answer")] {
        let mut body = json!({
            "id": "chatcmpl-test",
            "model": "gpt-reasoning",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "reasoning_content": "private reasoning"
                },
                "finish_reason": "length"
            }]
        });
        let mut expected = vec![json!({
            "type": "thinking",
            "thinking": "private reasoning",
            "signature": ""
        })];
        if let Some(text) = content.as_str() {
            expected.push(json!({"type": "text", "text": text}));
        }
        body["choices"][0]["message"]["content"] = content;

        let output = engine
            .translate_response(
                WireFormat::OpenAiChat,
                WireFormat::AnthropicMessages,
                &body,
                &normalized_policy(),
            )?
            .body;

        assert_eq!(output["content"], json!(expected), "input: {body}");
        assert_eq!(output["stop_reason"], "max_tokens");
    }
    Ok(())
}

#[test]
fn openai_chat_response_round_trips_reasoning_details() -> TestResult {
    let engine = TranslationEngine::default();
    // Exercise the normalized IR path instead of replaying the original JSON.
    let policy = normalized_policy();
    let details = text_and_encrypted_reasoning_details();
    let body = json!({
        "id": "chatcmpl-test",
        "model": REASONING_MODEL,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "reasoning": "fallback text",
                "reasoning_details": details,
                "tool_calls": [shell_tool_call()]
            },
            "finish_reason": "tool_calls"
        }]
    });

    let output = engine
        .translate_response(
            WireFormat::OpenAiChat,
            WireFormat::OpenAiChat,
            &body,
            &policy,
        )?
        .body;

    assert_eq!(
        output["choices"][0]["message"]["reasoning_details"],
        details
    );
    assert_eq!(
        output["choices"][0]["message"]["reasoning_content"],
        "Inspect the tool result."
    );
    Ok(())
}

// Verifies reasoning-only responses do not synthesize visible output text.
#[test]
fn openai_reasoning_only_response_translates_to_responses_reasoning_only() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "id": "chatcmpl-test",
        "model": "gpt-reasoning",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "reasoning_content": "private reasoning",
                "content": null
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}
    });

    let output = engine
        .translate_response(
            WireFormat::OpenAiChat,
            WireFormat::OpenAiResponses,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    let items = output["output"]
        .as_array()
        .ok_or("Responses output should be an array")?;
    assert_eq!(items.len(), 1);
    assert_eq!(items[0]["type"], "reasoning");
    assert_eq!(items[0]["summary"][0]["text"], "private reasoning");
    Ok(())
}

// Verifies OpenAI tool-call responses become Responses function-call output items.
#[test]
fn openai_chat_response_with_tool_call_translates_to_responses_output_item() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "id": "chatcmpl-test",
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{\"q\":\"rust\"}"}
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}
    });

    let output = engine
        .translate_response(
            WireFormat::OpenAiChat,
            WireFormat::OpenAiResponses,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(output["object"], "response");
    assert_eq!(output["output"][0]["type"], "function_call");
    assert_eq!(output["output"][0]["call_id"], "call_1");
    assert_eq!(output["output"][0]["name"], "lookup");
    assert_eq!(output["output"][0]["arguments"], "{\"q\": \"rust\"}");
    assert_eq!(
        output["usage"],
        json!({
            "input_tokens": 4,
            "output_tokens": 3,
            "total_tokens": 7,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0}
        })
    );
    Ok(())
}

// Verifies mixed assistant text and tool calls both survive into Responses output.
#[test]
fn openai_chat_response_with_text_and_tool_call_translates_both_to_responses() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "id": "chatcmpl-test",
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{\"q\":\"rust\"}"}
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}
    });

    let output = engine
        .translate_response(
            WireFormat::OpenAiChat,
            WireFormat::OpenAiResponses,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(output["output"][0]["type"], "message");
    assert_eq!(output["output"][0]["content"][0]["text"], "Let me check.");
    assert_eq!(output["output"][1]["type"], "function_call");
    assert_eq!(output["output"][1]["call_id"], "call_1");
    Ok(())
}

// Verifies both Responses usage detail objects are emitted even when the upstream reports no
// cache or reasoning breakdown. The Responses schema types them as required, so omitting them
// makes the payload unparseable by OpenAI-SDK clients.
#[test]
fn openai_chat_usage_without_breakdowns_still_emits_responses_usage_details() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "id": "chatcmpl-test",
        "model": "plain-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hi"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 41, "completion_tokens": 3, "total_tokens": 44}
    });

    let output = engine
        .translate_response(
            WireFormat::OpenAiChat,
            WireFormat::OpenAiResponses,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(
        output["usage"],
        json!({
            "input_tokens": 41,
            "output_tokens": 3,
            "total_tokens": 44,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0}
        })
    );
    Ok(())
}

// Verifies a partial breakdown does not suppress the other detail object: an upstream that
// reports cached tokens but no reasoning tokens must still carry both.
#[test]
fn openai_chat_cache_only_usage_still_emits_reasoning_details() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "id": "chatcmpl-test",
        "model": "cache-only-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hi"},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 41,
            "completion_tokens": 3,
            "total_tokens": 44,
            "prompt_tokens_details": {"cached_tokens": 32}
        }
    });

    let output = engine
        .translate_response(
            WireFormat::OpenAiChat,
            WireFormat::OpenAiResponses,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(
        output["usage"]["input_tokens_details"],
        json!({"cached_tokens": 32, "cache_write_tokens": 0})
    );
    assert_eq!(
        output["usage"]["output_tokens_details"],
        json!({"reasoning_tokens": 0})
    );
    Ok(())
}

// Verifies a token-limit stop is reported as an incomplete Responses result.
#[test]
fn openai_chat_length_finish_translates_to_incomplete_responses_status() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "id": "chatcmpl-test",
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Half an ans"},
            "finish_reason": "length"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    });

    let output = engine
        .translate_response(
            WireFormat::OpenAiChat,
            WireFormat::OpenAiResponses,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(output["status"], "incomplete");
    assert_eq!(
        output["incomplete_details"],
        json!({"reason": "max_output_tokens"})
    );
    assert_eq!(output["output"][0]["status"], "incomplete");
    Ok(())
}

// Verifies a truncated Responses source keeps its stop reason when re-encoded.
#[test]
fn incomplete_responses_source_survives_translation() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "id": "resp_1",
        "object": "response",
        "model": "gpt-4o",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Half an ans"}]
        }],
        "usage": {"input_tokens": 10, "output_tokens": 1, "total_tokens": 11}
    });

    let output = engine
        .translate_response(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiChat,
            &body,
            &TranslationPolicy::default(),
        )?
        .body;

    assert_eq!(output["choices"][0]["finish_reason"], "length");
    Ok(())
}

#[test]
fn failed_responses_return_upstream_failure_with_provider_message() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "id": "resp_failed",
        "object": "response",
        "model": "gpt-4o",
        "status": "failed",
        "error": {"code": "server_error", "message": "deterministic upstream failure"},
        "output": [],
        "usage": null
    });

    let target = WireFormat::OpenAiChat;
    let error = engine
        .translate_response(
            WireFormat::OpenAiResponses,
            target,
            &body,
            &TranslationPolicy::default(),
        )
        .err()
        .ok_or_else(|| format!("{target:?} accepted a failed response"))?;
    assert_eq!(error.kind(), "UpstreamFailure");
    assert!(error.to_string().contains("deterministic upstream failure"));
    Ok(())
}

// Verifies a moderation stop stays distinguishable from a normal turn in both
// directions, and that a named refusal category survives re-encoding.
#[test]
fn content_filter_and_refusal_translate_across_formats() -> TestResult {
    let engine = TranslationEngine::default();

    // An OpenAI moderation stop reaches Anthropic clients as `refusal`, not `end_turn`.
    // OpenAI reports no policy category, so the refusal carries the null form that
    // Anthropic documents for a refusal mapping to no named category.
    let openai = json!({
        "id": "chatcmpl-test",
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Partial answer"},
            "finish_reason": "content_filter"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    });
    let output = engine
        .translate_response(
            WireFormat::OpenAiChat,
            WireFormat::AnthropicMessages,
            &openai,
            &TranslationPolicy::default(),
        )?
        .body;
    assert_eq!(output["stop_reason"], "refusal");
    assert_eq!(
        output["stop_details"],
        json!({"type": "refusal", "category": null, "explanation": null})
    );

    // The distinction survives the other direction rather than being flattened.
    let anthropic = json!({
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5",
        "content": [{"type": "text", "text": "Partial answer"}],
        "stop_reason": "refusal",
        "stop_details": {
            "type": "refusal",
            "category": "cyber",
            "explanation": "This request was declined because it could enable cyber harm."
        },
        "usage": {"input_tokens": 10, "output_tokens": 5}
    });
    let output = engine
        .translate_response(
            WireFormat::AnthropicMessages,
            WireFormat::OpenAiChat,
            &anthropic,
            &TranslationPolicy::default(),
        )?
        .body;
    assert_eq!(output["choices"][0]["finish_reason"], "content_filter");

    // Re-encoding a refusal keeps the category the source named instead of
    // replacing it with the null form used for an unnamed refusal.
    let output = engine
        .translate_response(
            WireFormat::AnthropicMessages,
            WireFormat::AnthropicMessages,
            &anthropic,
            &TranslationPolicy {
                preservation: switchyard_translation::PreservationPolicy::Disabled,
                ..TranslationPolicy::default()
            },
        )?
        .body;
    assert_eq!(output["stop_reason"], "refusal");
    assert_eq!(output["stop_details"]["category"], "cyber");
    Ok(())
}

// A Responses reasoning item that carries only `encrypted_content` must survive a
// buffered decode/encode through the codec (preservation disabled so the same-format
// shortcut cannot mask a lossy codec), or a buffering caller loses the client's only
// replayable reasoning payload.
#[test]
fn responses_encrypted_reasoning_item_survives_buffered_round_trip() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "id": "resp_1",
        "object": "response",
        "status": "completed",
        "model": "kimi-k3",
        "output": [
            {
                "type": "reasoning",
                "id": "rs_upstream",
                "status": "completed",
                "summary": [],
                "encrypted_content": "opaque-encrypted-reasoning"
            },
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done", "annotations": []}]
            }
        ],
        "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7}
    });
    let policy = TranslationPolicy {
        preservation: PreservationPolicy::Disabled,
        ..TranslationPolicy::default()
    };

    let output = engine
        .translate_response(
            WireFormat::OpenAiResponses,
            WireFormat::OpenAiResponses,
            &body,
            &policy,
        )?
        .body;

    let reasoning = output["output"]
        .as_array()
        .ok_or("Responses output should be an array")?
        .iter()
        .find(|item| item["type"] == "reasoning")
        .ok_or("output should include the reasoning item")?;
    assert_eq!(reasoning["encrypted_content"], "opaque-encrypted-reasoning");
    // The payload only verifies upstream under the id it was issued with.
    assert_eq!(reasoning["id"], "rs_upstream");
    Ok(())
}

// A freeform tool call returned by the upstream must reach the client as a `custom_tool_call`
// again once the response is re-encoded with the request's extensions, and as a function-style
// call with an `input` argument when the client speaks chat.
#[test]
fn responses_custom_tool_call_output_round_trips_with_request_extensions() -> TestResult {
    let engine = TranslationEngine::default();
    let request = json!({
        "model": "gpt-5.6-luna",
        "input": "List files",
        "tools": [{
            "type": "custom",
            "name": "exec",
            "description": "Runs a shell command.",
            "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.*/"}
        }]
    });
    let decoded_request = engine.decode_request(
        WireFormat::OpenAiResponses,
        &request,
        &TranslationPolicy::default(),
    )?;
    let response = json!({
        "id": "resp_1",
        "object": "response",
        "status": "completed",
        "model": "gpt-5.6-luna",
        "output": [{
            "type": "custom_tool_call",
            "id": "ctc_1",
            "call_id": "call_1",
            "name": "exec",
            "input": "ls -la",
            "status": "completed"
        }],
        "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7}
    });
    let policy = TranslationPolicy {
        preservation: PreservationPolicy::Disabled,
        ..TranslationPolicy::default()
    };

    let ir = engine
        .decode_response(WireFormat::OpenAiResponses, &response, &policy)?
        .response;
    let encoded = engine
        .encode_response_with_extensions(
            WireFormat::OpenAiResponses,
            &ir,
            &decoded_request.request.extensions,
            &policy,
        )?
        .body;
    let item = encoded["output"]
        .as_array()
        .ok_or("output should be an array")?
        .iter()
        .find(|item| item["type"] == "custom_tool_call")
        .ok_or("the call must be re-emitted as custom_tool_call")?;
    assert_eq!(item["name"], "exec");
    assert_eq!(item["call_id"], "call_1");
    assert_eq!(item["input"], "ls -la");
    assert!(item.get("arguments").is_none(), "{item}");

    // Without the request extensions (e.g. a plain chat client) the call stays function-style.
    let chat = engine
        .encode_response(WireFormat::OpenAiChat, &ir, &policy)?
        .body;
    let call = &chat["choices"][0]["message"]["tool_calls"][0];
    assert_eq!(call["function"]["name"], "exec");
    assert_eq!(call["function"]["arguments"], "{\"input\":\"ls -la\"}");
    Ok(())
}
