// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for buffered response translation between provider formats.

pub mod common;

use pretty_assertions::assert_eq;
use serde_json::{Value, json};
use switchyard_translation::{
    ContentBlock, DeterministicIdPolicy, McpContent, PreservationPolicy, StopReason,
    TranslationEngine, TranslationError, TranslationPolicy, WireFormat,
};

use common::{
    REASONING_MODEL, mcp_call, mcp_definition, mcp_listing, mcp_request, mcp_response,
    normalized_policy, shell_tool_call, strict_normalized_policy,
    text_and_encrypted_reasoning_details,
};

type TestResult = std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>;

#[test]
fn anthropic_mcp_listings_round_trip_in_responses_and_history() -> TestResult {
    use switchyard_translation::{
        ContentBlock, LossyConversionPolicy, McpContent, StopReason, TranslationError,
    };

    let engine = TranslationEngine::default();
    let format = WireFormat::AnthropicMessages;
    let policy = TranslationPolicy {
        lossy_conversion_policy: LossyConversionPolicy::Reject,
        ..normalized_policy()
    };
    let content = json!([
        {"type": "mcp_tool_listing", "mcp_server_name": "inventory", "tools": [
            {"name": "lookup", "description": "Read stock", "input_schema": {
                "type": "object", "properties": {"sku": {"type": "string"}}, "required": ["sku"]
            }},
            {"name": "search", "input_schema": {"type": "object"}}
        ]},
        {"type": "mcp_tool_listing", "mcp_server_name": "empty", "tools": []},
        {"type": "text", "text": "Tools discovered."}
    ]);
    let body = json!({
        "id": "msg_listing", "type": "message", "role": "assistant", "model": "claude-test",
        "content": content, "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5}
    });
    let mut response = engine.decode_response(format, &body, &policy)?.response;
    assert!(matches!(
        &response.outputs[0].content[0],
        ContentBlock::Mcp(McpContent::Listing { server_name, id: None, tools, error: None })
            if server_name == "inventory" && tools.len() == 2 && tools[0].name == "lookup"
    ));
    assert_eq!(response.outputs[0].stop_reason, Some(StopReason::EndTurn));
    let output = engine.encode_response(format, &response, &policy)?;
    assert!(output.diagnostics.is_empty());
    assert_eq!(output.body["content"], content);

    let request_body = json!({
        "model": "claude-test", "max_tokens": 128,
        "messages": [{"role": "assistant", "content": content}, {"role": "user", "content": "Look up stock"}]
    });
    let mut request = engine
        .decode_request(format, &request_body, &policy)?
        .request;
    assert_eq!(request.messages[0].content, response.outputs[0].content);
    let output = engine.encode_request(format, &request, &policy)?;
    assert!(output.diagnostics.is_empty());
    assert_eq!(output.body["messages"][0]["content"], content);

    let ContentBlock::Mcp(McpContent::Listing {
        server_name, tools, ..
    }) = &mut response.outputs[0].content[0]
    else {
        return Err("expected MCP listing".into());
    };
    *server_name = "renamed".into();
    tools.remove(0);
    request.messages[0].content = response.outputs[0].content.clone();
    let expected = json!({"type": "mcp_tool_listing", "mcp_server_name": "renamed", "tools": [
        {"name": "search", "input_schema": {"type": "object"}}
    ]});
    assert_eq!(
        engine.encode_response(format, &response, &policy)?.body["content"][0],
        expected
    );
    assert_eq!(
        engine.encode_request(format, &request, &policy)?.body["messages"][0]["content"][0],
        expected
    );

    // Provider-specific listing data must not disappear during encoding.
    for field in ["id", "error", "annotations"] {
        let mut response = response.clone();
        let ContentBlock::Mcp(McpContent::Listing {
            id, error, tools, ..
        }) = &mut response.outputs[0].content[0]
        else {
            return Err("expected MCP listing".into());
        };
        match field {
            "id" => *id = Some("listing_1".into()),
            "error" => *error = Some("Server unavailable".into()),
            _ => tools[0].annotations = Some(json!({"readOnlyHint": true})),
        }
        request.messages[0].content = response.outputs[0].content.clone();
        assert!(matches!(
            engine.encode_response(format, &response, &policy),
            Err(TranslationError::LossyConversion(_))
        ));
        assert!(matches!(
            engine.encode_request(format, &request, &policy),
            Err(TranslationError::LossyConversion(_))
        ));
    }
    Ok(())
}

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

// Verifies a Chat Completions refusal survives both neutral decoding and Anthropic encoding.
#[test]
fn openai_chat_refusal_decodes_and_translates_to_anthropic() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({
        "id": "chatcmpl-refusal",
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "refusal": "I cannot help with that request."
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    });

    let decoded =
        engine.decode_response(WireFormat::OpenAiChat, &body, &TranslationPolicy::default())?;
    let output = decoded.response.first_output().ok_or("missing output")?;
    assert_eq!(
        output.content,
        vec![ContentBlock::Refusal {
            text: "I cannot help with that request.".to_string()
        }]
    );
    assert_eq!(output.stop_reason, Some(StopReason::ContentFilter));

    let translated = engine
        .encode_response(
            WireFormat::AnthropicMessages,
            &decoded.response,
            &TranslationPolicy::default(),
        )?
        .body;
    assert_eq!(
        translated["content"],
        json!([{"type": "text", "text": "I cannot help with that request."}])
    );
    assert_eq!(translated["stop_reason"], "refusal");
    assert_eq!(
        translated["stop_details"],
        json!({"type": "refusal", "category": null, "explanation": null})
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
    let mut mcp_body = body;
    mcp_body["content"] = json!([
        {"type": "mcp_tool_listing", "mcp_server_name": "inventory", "tools": []},
        {"type": "mcp_tool_use", "id": "mcp_1", "server_name": "inventory", "name": "lookup", "input": {}},
        {"type": "mcp_tool_result", "tool_use_id": "mcp_1", "content": [{"type": "text", "text": "in stock"}]},
        {"type": "text", "text": "Hi there"}
    ]);
    let source = WireFormat::AnthropicMessages;
    let strict = TranslationPolicy {
        lossy_conversion_policy: switchyard_translation::LossyConversionPolicy::Reject,
        ..normalized_policy()
    };
    assert_eq!(
        engine
            .translate_response(source, source, &mcp_body, &strict)?
            .body["content"],
        mcp_body["content"]
    );
    let target = WireFormat::OpenAiChat;
    assert!(matches!(
        engine.translate_response(source, target, &mcp_body, &strict),
        Err(switchyard_translation::TranslationError::LossyConversion(_))
    ));
    let output = engine.translate_response(
        source,
        target,
        &mcp_body,
        &TranslationPolicy {
            preservation: PreservationPolicy::Embed,
            ..TranslationPolicy::default()
        },
    )?;
    assert!(!output.diagnostics.is_empty());
    assert!(output.body.to_string().contains("Hi there"));
    assert!(!output.body.to_string().contains("mcp_tool"));
    let output =
        engine.translate_response(source, WireFormat::OpenAiResponses, &mcp_body, &strict)?;
    assert!(output.diagnostics.is_empty());
    assert_eq!(output.body["output"][0]["type"], "mcp_list_tools");
    assert_eq!(output.body["output"][1]["type"], "mcp_call");
    assert_eq!(output.body["output"][2]["type"], "message");

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

// Verifies Chat `abort`, `error` and `repetition` finish reasons fail the turn.
#[test]
fn failed_finish_reasons_return_upstream_failure() -> TestResult {
    let engine = TranslationEngine::default();
    for reason in ["abort", "error", "repetition"] {
        let body = json!({
            "id": "chatcmpl-test", "object": "chat.completion", "model": "gpt-4o",
            "choices": [{"index": 0, "finish_reason": reason,
                "message": {"role": "assistant", "content": "partial"}}]
        });
        let error = engine
            .translate_response(
                WireFormat::OpenAiChat,
                WireFormat::AnthropicMessages,
                &body,
                &TranslationPolicy::default(),
            )
            .err()
            .ok_or_else(|| format!("accepted finish_reason {reason}"))?;
        assert_eq!(error.kind(), "UpstreamFailure");
        assert!(error.to_string().contains(reason), "{error}");
    }
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

#[test]
fn typed_and_opaque_history_replay_as_top_level_items_in_order() -> TestResult {
    let engine = TranslationEngine::default();
    let mut error = mcp_call();
    error["id"] = json!("mcp_error");
    error["error"] =
        json!({"type": "mcp_protocol_error", "code": -32600, "message": "Invalid request"});
    let approval = json!({"type": "mcp_approval_request", "id": "approval_1", "server_label": "inventory", "name": "lookup", "arguments": "{}"});
    let approved = json!({"type": "mcp_approval_response", "approval_request_id": "approval_1", "approve": true});
    let items = json!([
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Before"}]},
        mcp_listing(), mcp_call(),
        {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "Check stock"}], "encrypted_content": "opaque"},
        error, approval, approved,
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "After"}]}
    ]);
    let mut body = mcp_request(mcp_definition());
    body["input"] = items.clone();
    let decoded = engine
        .decode_request(
            WireFormat::OpenAiResponses,
            &body,
            &strict_normalized_policy(),
        )?
        .request;
    let blocks: Vec<_> = decoded
        .messages
        .iter()
        .flat_map(|message| &message.content)
        .collect();
    assert!(
        blocks
            .iter()
            .any(|block| matches!(block, ContentBlock::Mcp(McpContent::Listing { .. })))
    );
    assert!(
        blocks
            .iter()
            .any(|block| matches!(block, ContentBlock::Mcp(McpContent::Call { .. })))
    );
    let input = engine
        .encode_request(
            WireFormat::OpenAiResponses,
            &decoded,
            &strict_normalized_policy(),
        )?
        .body["input"]
        .clone();
    let decoded = engine
        .decode_response(
            WireFormat::OpenAiResponses,
            &mcp_response(items.clone()),
            &strict_normalized_policy(),
        )?
        .response;
    assert_ne!(
        decoded.outputs[0].stop_reason,
        Some(switchyard_translation::StopReason::ToolUse)
    );
    let output = engine
        .encode_response(
            WireFormat::OpenAiResponses,
            &decoded,
            &strict_normalized_policy(),
        )?
        .body["output"]
        .clone();
    for encoded in [input, output] {
        assert_eq!(encoded.as_array().unwrap().len(), 8, "{encoded}");
        for index in [1, 2, 4, 5, 6] {
            assert_eq!(encoded[index], items[index]);
        }
        let text = |item: &Value| {
            item["content"]
                .as_str()
                .or_else(|| item["content"][0]["text"].as_str())
                .unwrap()
                .to_owned()
        };
        assert_eq!(text(&encoded[0]), "Before");
        assert_eq!(encoded[3]["type"], "reasoning");
        assert_eq!(text(&encoded[7]), "After");
    }
    Ok(())
}

#[test]
fn significant_call_fields_stay_opaque() -> TestResult {
    let engine = TranslationEngine::default();
    for (key, value) in [
        ("approval_request_id", json!("approval_1")),
        ("status", json!("in_progress")),
        ("arguments", json!("[]")),
        ("arguments", json!("invalid JSON")),
        ("output", Value::Null),
        ("future_field", json!({"important": true})),
    ] {
        let mut item = mcp_call();
        item[key] = value;
        let decoded = engine
            .decode_response(
                WireFormat::OpenAiResponses,
                &mcp_response(json!([item])),
                &strict_normalized_policy(),
            )?
            .response;
        assert!(
            matches!(&decoded.outputs[0].content[0], ContentBlock::Unknown { raw, .. } if raw == &item)
        );
        assert_eq!(
            engine
                .encode_response(
                    WireFormat::OpenAiResponses,
                    &decoded,
                    &strict_normalized_policy()
                )?
                .body["output"],
            json!([item])
        );
        assert!(
            engine
                .encode_response(
                    WireFormat::AnthropicMessages,
                    &decoded,
                    &strict_normalized_policy()
                )
                .is_err()
        );
    }
    Ok(())
}

#[test]
fn successful_calls_translate_through_anthropic_and_require_adjacent_results() -> TestResult {
    let engine = TranslationEngine::default();
    let decoded = engine
        .decode_response(
            WireFormat::OpenAiResponses,
            &mcp_response(json!([mcp_call()])),
            &strict_normalized_policy(),
        )?
        .response;
    let anthropic = engine.encode_response(
        WireFormat::AnthropicMessages,
        &decoded,
        &strict_normalized_policy(),
    )?;
    assert_eq!(anthropic.body["content"][0]["type"], "mcp_tool_use");
    assert_eq!(anthropic.body["content"][1]["type"], "mcp_tool_result");
    let back = engine.translate_response(
        WireFormat::AnthropicMessages,
        WireFormat::OpenAiResponses,
        &anthropic.body,
        &strict_normalized_policy(),
    )?;
    assert_eq!(back.body["output"], json!([mcp_call()]));
    for mutation in 0..4 {
        let mut changed = decoded.clone();
        let content = &mut changed.outputs[0].content;
        match mutation {
            0 => {
                content.pop();
            }
            1 => content.insert(
                1,
                ContentBlock::Text {
                    text: "Between".into(),
                },
            ),
            2 => content.extend(content.clone()),
            _ => {
                let ContentBlock::Mcp(McpContent::Result(result)) = &mut content[1] else {
                    unreachable!()
                };
                result.is_error = Some(true);
            }
        }
        assert!(matches!(
            engine.encode_response(
                WireFormat::OpenAiResponses,
                &changed,
                &strict_normalized_policy()
            ),
            Err(TranslationError::LossyConversion(_))
        ));
    }
    Ok(())
}

#[test]
fn listing_ids_are_stable_and_anthropic_rejects_unrepresentable_listing_fields() -> TestResult {
    let engine = TranslationEngine::default();
    let mut decoded = engine
        .decode_response(
            WireFormat::OpenAiResponses,
            &mcp_response(json!([mcp_listing()])),
            &strict_normalized_policy(),
        )?
        .response;
    assert!(
        engine
            .encode_response(
                WireFormat::AnthropicMessages,
                &decoded,
                &strict_normalized_policy()
            )
            .is_err()
    );
    let ContentBlock::Mcp(McpContent::Listing { id, tools, .. }) =
        &mut decoded.outputs[0].content[0]
    else {
        unreachable!()
    };
    *id = None;
    tools[0].annotations = None;
    assert!(
        engine
            .encode_response(
                WireFormat::AnthropicMessages,
                &decoded,
                &strict_normalized_policy()
            )
            .is_ok()
    );
    let first = engine.encode_response(
        WireFormat::OpenAiResponses,
        &decoded,
        &strict_normalized_policy(),
    )?;
    assert_eq!(
        first.body,
        engine
            .encode_response(
                WireFormat::OpenAiResponses,
                &decoded,
                &strict_normalized_policy()
            )?
            .body
    );
    let generated = first.body["output"][0]["id"].as_str().unwrap().to_owned();
    let mut existing = decoded.outputs[0].content[0].clone();
    let ContentBlock::Mcp(McpContent::Listing { id, .. }) = &mut existing else {
        unreachable!()
    };
    *id = Some(generated.clone());
    decoded.outputs[0].content.push(existing);
    let output = engine.encode_response(
        WireFormat::OpenAiResponses,
        &decoded,
        &strict_normalized_policy(),
    )?;
    assert_ne!(output.body["output"][0]["id"], generated);
    assert_eq!(output.body["output"][1]["id"], generated);
    let policy = TranslationPolicy {
        deterministic_ids: DeterministicIdPolicy::Preserve,
        ..strict_normalized_policy()
    };
    assert!(
        engine
            .encode_response(WireFormat::OpenAiResponses, &decoded, &policy)
            .is_err()
    );
    Ok(())
}

#[test]
fn mcp_response_ir_edits_do_not_restore_cached_history() -> TestResult {
    let engine = TranslationEngine::default();
    for preservation in [
        PreservationPolicy::Disabled,
        PreservationPolicy::InMemory,
        PreservationPolicy::Embed,
    ] {
        let policy = TranslationPolicy {
            preservation,
            ..strict_normalized_policy()
        };
        for source in [WireFormat::OpenAiResponses, WireFormat::AnthropicMessages] {
            let original = mcp_response(json!([mcp_call()]));
            let body = if source == WireFormat::OpenAiResponses {
                original
            } else {
                engine
                    .translate_response(
                        WireFormat::OpenAiResponses,
                        WireFormat::AnthropicMessages,
                        &original,
                        &strict_normalized_policy(),
                    )?
                    .body
            };
            let mut decoded = engine.decode_response(source, &body, &policy)?.response;
            decoded.outputs[0].content = vec![ContentBlock::Text {
                text: "Removed".into(),
            }];
            let encoded = engine.encode_response(source, &decoded, &policy)?;
            assert!(!encoded.body.to_string().contains("inventory"));
        }
    }
    Ok(())
}

#[test]
fn unrepresentable_anthropic_mcp_results_obey_loss_policy() -> TestResult {
    let engine = TranslationEngine::default();
    for (is_error, content) in [
        (true, json!([{"type": "text", "text": "Lookup failed"}])),
        (
            false,
            json!([{"type": "image", "source": {"type": "url", "url": "https://example.com/result.png"}}]),
        ),
    ] {
        let body = json!({"id": "msg_1", "type": "message", "role": "assistant", "model": "route", "stop_reason": "end_turn", "content": [
            {"type": "text", "text": "Before"},
            {"type": "mcp_tool_use", "id": "call_1", "server_name": "inventory", "name": "lookup", "input": {}},
            {"type": "mcp_tool_result", "tool_use_id": "call_1", "is_error": is_error, "content": content},
            {"type": "text", "text": "After"}
        ]});
        assert!(matches!(
            engine.translate_response(
                WireFormat::AnthropicMessages,
                WireFormat::OpenAiResponses,
                &body,
                &strict_normalized_policy()
            ),
            Err(TranslationError::LossyConversion(_))
        ));
        let translated = engine.translate_response(
            WireFormat::AnthropicMessages,
            WireFormat::OpenAiResponses,
            &body,
            &normalized_policy(),
        )?;
        assert!(
            translated
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message.contains("successful text result"))
        );
        let decoded = engine
            .decode_response(
                WireFormat::OpenAiResponses,
                &translated.body,
                &normalized_policy(),
            )?
            .response;
        let content: Vec<_> = decoded
            .outputs
            .into_iter()
            .flat_map(|output| output.content)
            .collect();
        assert_eq!(
            content,
            vec![ContentBlock::Text {
                text: "BeforeAfter".into()
            }]
        );
    }
    Ok(())
}

#[test]
fn anthropic_mcp_identifiers_must_be_nonempty_strings() -> TestResult {
    let engine = TranslationEngine::default();
    for (block, fields) in [
        (
            json!({"type": "mcp_tool_use", "id": "call_1", "name": "lookup", "server_name": "inventory", "input": {}}),
            vec!["id", "name", "server_name"],
        ),
        (
            json!({"type": "mcp_tool_result", "tool_use_id": "call_1", "content": "ok"}),
            vec!["tool_use_id"],
        ),
        (
            json!({"type": "mcp_tool_listing", "mcp_server_name": "inventory", "tools": []}),
            vec!["mcp_server_name"],
        ),
    ] {
        for field in fields {
            for invalid in [None, Some(Value::Null), Some(json!(5)), Some(json!(""))] {
                let mut malformed = block.clone();
                match invalid {
                    Some(value) => {
                        malformed[field] = value;
                    }
                    None => {
                        malformed.as_object_mut().unwrap().remove(field);
                    }
                }
                let body = json!({"type": "message", "role": "assistant", "content": [malformed]});
                assert!(
                    matches!(engine.decode_response(WireFormat::AnthropicMessages, &body, &normalized_policy()),
                    Err(TranslationError::InvalidValue { path, .. }) if path.ends_with(field))
                );
            }
        }
    }
    Ok(())
}

#[test]
fn citations_spanning_mcp_items_are_preserved_on_both_text_segments() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = strict_normalized_policy();
    let mut response = engine.decode_response(WireFormat::OpenAiResponses, &json!({
        "status": "completed", "output": [{"type": "message", "role": "assistant", "content": [{
            "type": "output_text", "text": "éA🌍B", "annotations": [{
                "type": "url_citation", "start_index": 1, "end_index": 3,
                "url": "https://source.example", "title": "Source"
            }]
        }]}]
    }), &policy)?.response;
    response.outputs[0].content = vec![
        ContentBlock::Text { text: "éA".into() },
        ContentBlock::Mcp(McpContent::Listing {
            server_name: "inventory".into(),
            id: Some("list_1".into()),
            tools: vec![],
            error: None,
        }),
        ContentBlock::Text {
            text: "🌍B".into()
        },
    ];
    let encoded = engine.encode_response(WireFormat::OpenAiResponses, &response, &policy)?;
    assert!(encoded.diagnostics.is_empty());
    assert_eq!(encoded.body["output"][1]["type"], "mcp_list_tools");
    for (index, start, end) in [(0, 1, 2), (2, 0, 1)] {
        assert_eq!(
            encoded.body["output"][index]["content"][0]["annotations"],
            json!([{
                "type": "url_citation", "start_index": start, "end_index": end,
                "url": "https://source.example", "title": "Source"
            }])
        );
    }
    assert_ne!(
        encoded.body["output"][0]["id"],
        encoded.body["output"][2]["id"]
    );
    Ok(())
}
