// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use futures::{StreamExt, executor::block_on, stream};
use serde_json::{Value, json};
use switchyard_protocol::{LlmClientError, LlmResponse, LlmResponseChunk, LlmResponseStreamEvent};
use switchyard_translation::{
    ContentBlock, LossyConversionPolicy, PreservationPolicy, StreamTranslationState, ToolChoice,
    TranslationEngine, TranslationPolicy, UnknownFieldPolicy, WireFormat, decode_event_stream,
    decode_stream, encode_aggregated_response, encode_stream,
};

type TestResult = Result<(), Box<dyn std::error::Error>>;
const BEDROCK: WireFormat = WireFormat::BedrockConverse;

fn normalized() -> TranslationPolicy {
    TranslationPolicy {
        preservation: PreservationPolicy::Disabled,
        ..TranslationPolicy::default()
    }
}

fn request() -> Value {
    json!({
        "system": [{"text": "Be brief."}],
        "messages": [
            {"role": "user", "content": [{"text": "weather?"}]},
            {"role": "assistant", "content": [{"toolUse": {"toolUseId": "call_1", "name": "weather", "input": {"city": "Taipei"}}}]},
            {"role": "user", "content": [{"toolResult": {"toolUseId": "call_1", "content": [{"text": "sunny"}], "status": "error"}}]}
        ],
        "inferenceConfig": {"maxTokens": 123, "temperature": 0.2, "topP": 0.8, "stopSequences": ["END"]},
        "toolConfig": {"tools": [{"toolSpec": {"name": "weather", "strict": true, "inputSchema": {"json": {"type": "object"}}}}], "toolChoice": {"tool": {"name": "weather"}}}
    })
}

fn usage() -> Value {
    json!({"inputTokens": 10, "outputTokens": 5, "totalTokens": 23,
        "cacheReadInputTokens": 6, "cacheWriteInputTokens": 2})
}

fn events() -> Vec<Value> {
    vec![
        json!({"messageStart": {"role": "assistant"}}),
        json!({"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hello"}}}),
        json!({"contentBlockStop": {"contentBlockIndex": 0}}),
        json!({"messageStop": {"stopReason": "end_turn"}}),
        json!({"metadata": {"usage": usage(), "metrics": {"latencyMs": 12}}}),
    ]
}

fn raw_events(
    values: Vec<Value>,
) -> Result<switchyard_protocol::LlmResponseStream, LlmClientError> {
    decode_event_stream(stream::iter(values.into_iter().map(Ok)), BEDROCK)
}

fn translate_events(
    values: &[Value],
    target: WireFormat,
) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
    let engine = TranslationEngine::default();
    let mut state = StreamTranslationState::new(BEDROCK, target);
    let mut result = Vec::new();
    for value in values {
        result.extend(engine.translate_event(&mut state, BEDROCK, target, value)?);
    }
    result.extend(engine.finish_stream(&mut state, target)?);
    Ok(result)
}

#[test]
fn bedrock_text_request_decodes() -> TestResult {
    let output = TranslationEngine::default().decode_request(
        "bedrock_converse",
        &json!({"messages": [{"role": "user", "content": [{"text": "hello"}]}]}),
        &TranslationPolicy::default(),
    )?;
    assert_eq!(
        output.request.messages[0].text_content(""),
        Some("hello".into())
    );
    assert_eq!(serde_json::to_value(BEDROCK)?, "bedrock_converse");
    assert!(output.request.model.is_none());
    Ok(())
}

#[test]
fn native_requests_and_cross_format_tool_histories() -> TestResult {
    let engine = TranslationEngine::default();
    let body = request();
    let decoded = engine
        .decode_request(BEDROCK, &body, &normalized())?
        .request;
    assert_eq!(
        engine
            .encode_request(BEDROCK, &decoded, &normalized())?
            .body,
        body
    );
    assert_eq!(decoded.output.max_output_tokens, Some(123));
    assert_eq!(decoded.tools[0].strict, Some(true));
    assert!(
        matches!(&decoded.messages[2].content[0], ContentBlock::ToolResult(result) if result.is_error == Some(true))
    );
    for target in [
        WireFormat::OpenAiChat,
        WireFormat::OpenAiResponses,
        WireFormat::AnthropicMessages,
    ] {
        let projected = engine.translate_request(BEDROCK, target, &body, &normalized())?;
        let back = engine
            .translate_request(target, BEDROCK, &projected.body, &normalized())?
            .body;
        assert_eq!(
            back["messages"][1]["content"][0]["toolUse"]["toolUseId"], "call_1",
            "{target}: {back}"
        );
        assert_eq!(
            back["messages"][1]["content"][0]["toolUse"]["input"],
            json!({"city": "Taipei"})
        );
        assert_eq!(back["inferenceConfig"]["maxTokens"], 123);
        // OpenAI Responses does not provide a stop-sequence setting.
        if target != WireFormat::OpenAiResponses {
            assert_eq!(back["inferenceConfig"]["stopSequences"], json!(["END"]));
        }
    }
    let mut rich = body.clone();
    rich["guardrailConfig"] = json!({"guardrailIdentifier": "guardrail", "guardrailVersion": "1"});
    rich["futureControl"] = json!({"enabled": true});
    assert_eq!(
        engine
            .translate_request(BEDROCK, BEDROCK, &rich, &TranslationPolicy::default())?
            .body,
        rich
    );
    let strict = TranslationPolicy {
        lossy_conversion_policy: LossyConversionPolicy::Reject,
        ..normalized()
    };
    assert!(
        engine
            .translate_request(BEDROCK, WireFormat::OpenAiChat, &rich, &strict)
            .is_err()
    );
    let mut json_result = body.clone();
    json_result["messages"][2]["content"][0]["toolResult"]["content"] =
        json!([{"json": {"answer": 42}}]);
    let chat = engine
        .translate_request(BEDROCK, WireFormat::OpenAiChat, &json_result, &normalized())?
        .body;
    assert_eq!(chat["messages"][3]["role"], "tool");
    assert_eq!(chat["messages"][3]["tool_call_id"], "call_1");
    assert_eq!(chat["messages"][3]["content"], "{\"answer\":42}");
    assert_eq!(chat["messages"].as_array().map(Vec::len), Some(4));
    let mut no_tools = decoded;
    no_tools.tool_choice = Some(ToolChoice::None);
    let output = engine.encode_request(BEDROCK, &no_tools, &normalized())?;
    assert!(output.body["toolConfig"]["tools"].is_array());
    assert!(
        output
            .diagnostics
            .iter()
            .any(|d| d.code == "lossy_conversion")
    );
    Ok(())
}

#[test]
fn rejects_malformed_requests_and_reports_unsupported_controls() -> TestResult {
    let engine = TranslationEngine::default();
    for body in [
        json!({"messages": "bad"}),
        json!({"messages": [{"role": "mystery", "content": [{"text": "hello"}]}]}),
        json!({"messages": [{"role": "user", "content": [{"text": "x", "image": {}}]}]}),
        json!({"messages": [], "inferenceConfig": {"maxTokens": -1}}),
        json!({"messages": [], "toolConfig": {"tools": [{"toolSpec": {"name": "t", "inputSchema": {"json": {}}}, "cachePoint": {"type": "default"}}]}}),
        json!({"messages": [{"role": "assistant", "content": [{"reasoningContent": {"reasoningText": {"text": "x"}, "redactedContent": "YQ=="}}]}]}),
    ] {
        assert!(
            engine
                .decode_request(BEDROCK, &body, &normalized())
                .is_err(),
            "{body}"
        );
    }
    let mut policy = normalized();
    policy.unknown_field_policy = UnknownFieldPolicy::Reject;
    assert!(
        engine
            .decode_request(BEDROCK, &json!({"messages": [], "typo": true}), &policy)
            .is_err()
    );
    let mut ir = engine
        .decode_request(BEDROCK, &request(), &normalized())?
        .request;
    ir.output.response_format = Some(json!({"type": "json_object"}));
    assert!(
        !engine
            .encode_request(BEDROCK, &ir, &normalized())?
            .diagnostics
            .is_empty()
    );
    policy.lossy_conversion_policy = LossyConversionPolicy::Reject;
    assert!(engine.encode_request(BEDROCK, &ir, &policy).is_err());
    Ok(())
}

#[test]
fn buffered_responses_preserve_reasoning_usage_and_native_fields() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({"output": {"message": {"role": "assistant", "content": [
        {"reasoningContent": {"reasoningText": {"text": "consider", "signature": "signed"}}},
        {"text": "answer"}]}}, "stopReason": "guardrail_intervened", "usage": usage(),
        "metrics": {"latencyMs": 12}, "trace": {"guardrail": {"action": "BLOCKED"}}});
    let mut malformed = body.clone();
    malformed["output"]["message"]["role"] = json!("user");
    assert!(
        engine
            .decode_response(BEDROCK, &malformed, &normalized())
            .is_err()
    );
    malformed = body.clone();
    malformed["usage"]
        .as_object_mut()
        .ok_or("usage object")?
        .remove("totalTokens");
    assert!(
        engine
            .decode_response(BEDROCK, &malformed, &normalized())
            .is_err()
    );
    let decoded = engine
        .decode_response(BEDROCK, &body, &normalized())?
        .response;
    assert_eq!(
        engine
            .encode_response(BEDROCK, &decoded, &normalized())?
            .body,
        body
    );
    let chat = engine
        .encode_response(WireFormat::OpenAiChat, &decoded, &normalized())?
        .body;
    let mut chat_with_reasoning = chat.clone();
    chat_with_reasoning["usage"] = json!({"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15,
        "completion_tokens_details": {"reasoning_tokens": 4}});
    let bedrock = engine
        .translate_response(
            WireFormat::OpenAiChat,
            BEDROCK,
            &chat_with_reasoning,
            &normalized(),
        )?
        .body;
    assert_eq!(
        bedrock["usage"],
        json!({"inputTokens": 5, "outputTokens": 10, "totalTokens": 15})
    );
    assert_eq!(chat["usage"]["prompt_tokens"], 18);
    assert_eq!(chat["usage"]["completion_tokens"], 5);
    assert_eq!(chat["choices"][0]["finish_reason"], "content_filter");
    assert_eq!(chat["choices"][0]["message"]["content"], "answer");
    assert_eq!(
        chat["choices"][0]["message"]["reasoning_content"],
        "consider"
    );
    assert!(
        encode_aggregated_response(&decoded, BEDROCK, Some("url-owned/model"))?
            .get("model")
            .is_none()
    );
    assert!(
        engine
            .decode_response(
                BEDROCK,
                &json!({"modelStreamErrorException": {"message": "failed"}}),
                &normalized()
            )
            .is_err()
    );
    Ok(())
}

#[test]
fn native_stream_replays_without_duplicate_terminal_events() -> TestResult {
    let values = events();
    let replayed = block_on(
        encode_stream(
            raw_events(values.clone())?,
            BEDROCK,
            Some("url-owned/model".into()),
        )?
        .collect::<Vec<_>>(),
    )
    .into_iter()
    .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(replayed, values);
    let engine = TranslationEngine::default();
    let mut decoder = StreamTranslationState::new(BEDROCK, BEDROCK);
    let mut encoder = StreamTranslationState::new(BEDROCK, BEDROCK);
    for raw in [
        values[0].clone(),
        json!({"contentBlockDelta": {"contentBlockIndex": 7, "delta": {"text": "partial"}}}),
    ] {
        let event = engine.decode_stream_event(&mut decoder, BEDROCK, raw.clone())?;
        assert_eq!(
            engine.encode_stream_event(&mut encoder, BEDROCK, event)?,
            vec![raw]
        );
    }
    engine.encode_stream_event(
        &mut encoder,
        BEDROCK,
        LlmResponseChunk::Usage(switchyard_protocol::Usage {
            input_tokens: Some(1),
            output_tokens: Some(2),
            ..Default::default()
        })
        .into(),
    )?;
    let finished = engine.finish_stream(&mut encoder, BEDROCK)?;
    assert_eq!(finished[0]["contentBlockStop"]["contentBlockIndex"], 7);
    assert_eq!(finished[2]["metadata"]["usage"]["totalTokens"], 3);
    Ok(())
}

#[test]
fn stream_text_and_usage_translate_to_existing_formats() -> TestResult {
    for target in [
        WireFormat::OpenAiChat,
        WireFormat::OpenAiResponses,
        WireFormat::AnthropicMessages,
    ] {
        let output = translate_events(&events(), target)?;
        let aggregate = block_on(
            LlmResponse::Stream(decode_event_stream(
                stream::iter(output.into_iter().map(Ok)),
                target,
            )?)
            .into_agg(),
        )?;
        assert_eq!(
            aggregate.outputs[0].content,
            vec![ContentBlock::Text {
                text: "Hello".into()
            }],
            "{target}"
        );
        assert_eq!(aggregate.usage.output_tokens, Some(5));
        assert_eq!(aggregate.usage.cached_input_tokens(), Some(6));
    }
    Ok(())
}

#[test]
fn streaming_tools_keep_arguments_that_arrive_before_identity() -> TestResult {
    let chunks = vec![
        LlmResponseChunk::TextDelta {
            index: 0,
            text: "calling tools".into(),
        },
        LlmResponseChunk::ToolCallDelta {
            index: 0,
            id: None,
            name: None,
            arguments_delta: Some("{\"city\":".into()),
        },
        LlmResponseChunk::ToolCallDelta {
            index: 1,
            id: Some("second".into()),
            name: Some("weather".into()),
            arguments_delta: Some("{}".into()),
        },
        LlmResponseChunk::ToolCallDelta {
            index: 0,
            id: Some("first".into()),
            name: Some("weather".into()),
            arguments_delta: Some("\"Taipei\"}".into()),
        },
        LlmResponseChunk::MessageStop {
            reason: Some("tool_calls".into()),
        },
        LlmResponseChunk::Usage(switchyard_protocol::Usage {
            input_tokens: Some(1),
            output_tokens: Some(2),
            total_tokens: Some(3),
            ..Default::default()
        }),
    ];
    let encoded = block_on(
        encode_stream(
            Box::pin(stream::iter(chunks.into_iter().map(|c| Ok(c.into())))),
            BEDROCK,
            None,
        )?
        .collect::<Vec<_>>(),
    )
    .into_iter()
    .collect::<Result<Vec<_>, _>>()?;
    let aggregate = block_on(LlmResponse::Stream(raw_events(encoded)?).into_agg())?;
    let calls = aggregate.outputs[0]
        .content
        .iter()
        .filter_map(|b| {
            if let ContentBlock::ToolCall(c) = b {
                Some(c)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(calls.len(), 2);
    assert!(
        calls
            .iter()
            .any(|c| c.id == "first" && c.arguments == json!({"city": "Taipei"}))
    );
    assert!(
        calls
            .iter()
            .any(|c| c.id == "second" && c.arguments == json!({}))
    );
    Ok(())
}

#[test]
fn reasoning_signatures_survive_stream_aggregation() -> TestResult {
    let values = vec![
        json!({"messageStart": {"role": "assistant"}}),
        json!({"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "think"}}}}),
        json!({"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"signature": "sig1"}}}}),
        json!({"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"signature": "sig2"}}}}),
        json!({"contentBlockStop": {"contentBlockIndex": 0}}),
        json!({"messageStop": {"stopReason": "end_turn"}}),
        json!({"metadata": {"usage": usage()}}),
    ];
    assert_eq!(translate_events(&values, BEDROCK)?, values);
    let mut redacted = values.clone();
    redacted.splice(1..4, [
        json!({"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"redactedContent": "YQ=="}}}}),
        json!({"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"redactedContent": "Yg=="}}}}),
    ]);
    let redacted_aggregate = block_on(LlmResponse::Stream(raw_events(redacted)?).into_agg())?;
    let body = TranslationEngine::default()
        .encode_response(BEDROCK, &redacted_aggregate, &normalized())?
        .body;
    assert_eq!(
        body["output"]["message"]["content"][0]["reasoningContent"]["redactedContent"],
        "YWI="
    );
    let aggregate = block_on(LlmResponse::Stream(raw_events(values)?).into_agg())?;
    let body = TranslationEngine::default()
        .encode_response(BEDROCK, &aggregate, &normalized())?
        .body;
    assert_eq!(
        body["output"]["message"]["content"][0]["reasoningContent"]["reasoningText"],
        json!({"text": "think", "signature": "sig1sig2"})
    );
    Ok(())
}

#[test]
fn event_stream_errors_and_truncation_fail_without_success_terminal() -> TestResult {
    for extra in [
        json!({"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "late"}}}),
        json!({"modelStreamErrorException": {"message": "late"}}),
    ] {
        let mut values = events();
        values.push(extra);
        let output =
            block_on(encode_stream(raw_events(values)?, BEDROCK, None)?.collect::<Vec<_>>());
        assert!(output.last().is_some_and(Result::is_err));
    }
    let mut incomplete = events();
    incomplete.pop();
    assert!(block_on(LlmResponse::Stream(raw_events(incomplete)?).into_agg()).is_err());
    for values in [
        vec![
            json!({"messageStart": {"role": "assistant"}}),
            json!({"throttlingException": {"message": "slow down"}}),
        ],
        vec![
            json!({"messageStart": {"role": "assistant"}}),
            json!({"contentBlockDelta": {"delta": {"text": "bad"}}}),
        ],
    ] {
        let output =
            block_on(encode_stream(raw_events(values)?, BEDROCK, None)?.collect::<Vec<_>>());
        assert!(output.last().is_some_and(Result::is_err));
        assert!(
            !output
                .iter()
                .filter_map(|v| v.as_ref().ok())
                .any(|v| v.get("messageStop").is_some())
        );
    }
    let incomplete_tool: Vec<Result<LlmResponseStreamEvent, LlmClientError>> =
        vec![Ok(LlmResponseChunk::ToolCallDelta {
            index: 0,
            id: None,
            name: None,
            arguments_delta: Some("{}".into()),
        }
        .into())];
    let output = block_on(
        encode_stream(Box::pin(stream::iter(incomplete_tool)), BEDROCK, None)?.collect::<Vec<_>>(),
    );
    assert!(output.last().is_some_and(Result::is_err));
    let unsupported = vec![LlmResponseChunk::ReasoningDetailsDelta {
        index: 0,
        text: String::new(),
        details: vec![json!({"type": "anthropic.signature_delta", "signature": "opaque"})],
    }];
    let output = block_on(
        encode_stream(
            Box::pin(stream::iter(unsupported.into_iter().map(|c| Ok(c.into())))),
            BEDROCK,
            None,
        )?
        .collect::<Vec<_>>(),
    );
    assert!(output.last().is_some_and(Result::is_err));
    let mut resume = StreamTranslationState::new(WireFormat::OpenAiChat, BEDROCK);
    let engine = TranslationEngine::default();
    for chunk in [
        LlmResponseChunk::ToolCallDelta {
            index: 0,
            id: Some("tool".into()),
            name: Some("t".into()),
            arguments_delta: Some("{".into()),
        },
        LlmResponseChunk::TextDelta {
            index: 1,
            text: "interleaved".into(),
        },
        LlmResponseChunk::ToolCallDelta {
            index: 0,
            id: None,
            name: None,
            arguments_delta: Some("}".into()),
        },
    ] {
        engine.encode_stream_event(&mut resume, BEDROCK, chunk.into())?;
    }
    assert!(resume.errored);
    let mut signature = StreamTranslationState::new(BEDROCK, WireFormat::AnthropicMessages);
    let output = engine.encode_stream_event(
        &mut signature,
        WireFormat::AnthropicMessages,
        LlmResponseChunk::ReasoningDetailsDelta {
            index: 0,
            text: String::new(),
            details: vec![json!({"type": "bedrock.redacted_content", "data": "YQ=="})],
        }
        .into(),
    )?;
    assert!(signature.errored);
    assert_eq!(output[0]["type"], "error");
    let bytes = stream::iter([Ok::<Vec<u8>, LlmClientError>(b"data: {}\n\n".to_vec())]);
    assert!(decode_stream(bytes, BEDROCK).is_err());
    Ok(())
}

#[test]
fn context_window_exhaustion_stays_incomplete_across_formats() -> TestResult {
    let engine = TranslationEngine::default();
    let body = json!({"output": {"message": {"role": "assistant", "content": [{"text": "partial"}]}},
        "stopReason": "model_context_window_exceeded", "usage": usage()});
    let response = engine
        .decode_response(BEDROCK, &body, &normalized())?
        .response;
    assert_eq!(
        response.outputs[0].stop_reason,
        Some(switchyard_protocol::StopReason::MaxTokens)
    );
    assert_eq!(
        engine
            .encode_response(BEDROCK, &response, &normalized())?
            .body,
        body
    );
    let mut values = events();
    values[3] = json!({"messageStop": {"stopReason": "model_context_window_exceeded"}});
    let aggregate = block_on(LlmResponse::Stream(raw_events(values.clone())?).into_agg())?;
    assert_eq!(
        aggregate.outputs[0].stop_reason,
        Some(switchyard_protocol::StopReason::MaxTokens)
    );
    let replayed =
        block_on(encode_stream(raw_events(values.clone())?, BEDROCK, None)?.collect::<Vec<_>>())
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(replayed, values);
    for target in [
        WireFormat::OpenAiChat,
        WireFormat::OpenAiResponses,
        WireFormat::AnthropicMessages,
    ] {
        let encoded = engine
            .encode_response(target, &response, &normalized())?
            .body;
        match target {
            WireFormat::OpenAiChat => assert_eq!(encoded["choices"][0]["finish_reason"], "length"),
            WireFormat::OpenAiResponses => assert_eq!(encoded["status"], "incomplete"),
            WireFormat::AnthropicMessages => assert_eq!(encoded["stop_reason"], "max_tokens"),
            WireFormat::BedrockConverse => unreachable!(),
        }
        let projected = translate_events(&values, target)?;
        let aggregate = block_on(
            LlmResponse::Stream(decode_event_stream(
                stream::iter(projected.into_iter().map(Ok)),
                target,
            )?)
            .into_agg(),
        )?;
        assert_eq!(
            aggregate.outputs[0].stop_reason,
            Some(switchyard_protocol::StopReason::MaxTokens),
            "{target}"
        );
    }
    let mut direct = StreamTranslationState::new(BEDROCK, BEDROCK);
    engine.encode_stream_event(
        &mut direct,
        BEDROCK,
        LlmResponseChunk::MessageStop {
            reason: Some("model_context_window_exceeded".into()),
        }
        .into(),
    )?;
    engine.encode_stream_event(
        &mut direct,
        BEDROCK,
        LlmResponseChunk::Usage(response.usage).into(),
    )?;
    assert!(
        engine
            .finish_stream(&mut direct, BEDROCK)?
            .iter()
            .any(|e| e["messageStop"]["stopReason"] == "model_context_window_exceeded")
    );
    Ok(())
}

#[test]
fn consecutive_tool_results_and_user_turns_merge_without_reordering() -> TestResult {
    let body = json!({"messages": [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "weather", "arguments": "{}"}},
            {"id": "b", "type": "function", "function": {"name": "weather", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "a", "content": "sunny"},
        {"role": "tool", "tool_call_id": "b", "content": "rainy"},
        {"role": "user", "content": "also tomorrow?"},
        {"role": "assistant", "content": "forecast"},
        {"role": "assistant", "content": "continued"},
        {"role": "user", "content": "thanks"}
    ], "tools": [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}]});
    let engine = TranslationEngine::default();
    let ir = engine
        .decode_request(WireFormat::OpenAiChat, &body, &normalized())?
        .request;
    let encoded = engine.encode_request(BEDROCK, &ir, &normalized())?.body;
    assert_eq!(encoded["messages"].as_array().map(Vec::len), Some(5));
    assert_eq!(encoded["messages"][2]["role"], "user");
    let content = encoded["messages"][2]["content"]
        .as_array()
        .ok_or("content array")?;
    assert_eq!(content.len(), 3);
    assert_eq!(content[0]["toolResult"]["toolUseId"], "a");
    assert_eq!(content[1]["toolResult"]["toolUseId"], "b");
    assert_eq!(content[2], json!({"text": "also tomorrow?"}));
    assert_eq!(
        encoded["messages"][3]["content"],
        json!([{"text": "forecast"}, {"text": "continued"}])
    );
    let responses = json!({"input": [
        {"role": "user", "content": "weather?"},
        {"type": "function_call", "call_id": "a", "name": "weather", "arguments": "{}"},
        {"type": "function_call", "call_id": "b", "name": "weather", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "a", "output": "sunny"},
        {"type": "function_call_output", "call_id": "b", "output": "rainy"},
        {"role": "user", "content": "also tomorrow?"},
        {"role": "assistant", "content": "forecast"},
        {"role": "assistant", "content": "continued"},
        {"role": "user", "content": "thanks"}
    ], "tools": [{"type": "function", "name": "weather", "parameters": {"type": "object"}}]});
    let translated = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            BEDROCK,
            &responses,
            &normalized(),
        )?
        .body;
    assert_eq!(translated["messages"], encoded["messages"]);
    Ok(())
}

#[test]
fn disabled_tools_keep_required_history_configuration_or_fail() -> TestResult {
    let engine = TranslationEngine::default();
    let mut ir = engine
        .decode_request(BEDROCK, &request(), &normalized())?
        .request;
    ir.tool_choice = Some(ToolChoice::None);
    let output = engine.encode_request(BEDROCK, &ir, &normalized())?;
    assert!(output.body["toolConfig"]["tools"].is_array());
    assert!(output.body["toolConfig"].get("toolChoice").is_none());
    assert!(
        output
            .diagnostics
            .iter()
            .any(|d| d.code == "lossy_conversion")
    );
    let strict = TranslationPolicy {
        lossy_conversion_policy: LossyConversionPolicy::Reject,
        ..normalized()
    };
    assert!(engine.encode_request(BEDROCK, &ir, &strict).is_err());
    let mut no_history = ir.clone();
    no_history.messages.truncate(1);
    assert!(
        engine
            .encode_request(BEDROCK, &no_history, &strict)?
            .body
            .get("toolConfig")
            .is_none()
    );
    ir.tools.clear();
    for choice in [ToolChoice::None, ToolChoice::Auto] {
        ir.tool_choice = Some(choice);
        assert!(engine.encode_request(BEDROCK, &ir, &normalized()).is_err());
    }
    ir.messages = vec![ir.messages[0].clone()];
    ir.tool_choice = Some(ToolChoice::None);
    assert!(
        engine
            .encode_request(BEDROCK, &ir, &strict)?
            .body
            .get("toolConfig")
            .is_none()
    );
    let mut prompt_body = request();
    for field in ["toolConfig", "system", "inferenceConfig"] {
        prompt_body
            .as_object_mut()
            .ok_or("request object")?
            .remove(field);
    }
    prompt_body["promptVariables"] = json!({"city": {"text": "Taipei"}});
    let preserved_policy = TranslationPolicy {
        preservation: PreservationPolicy::InMemory,
        ..strict
    };
    let preserved = engine
        .decode_request(BEDROCK, &prompt_body, &preserved_policy)?
        .request;
    assert!(preserved.tools.is_empty());
    assert_eq!(
        engine
            .encode_request(BEDROCK, &preserved, &preserved_policy)?
            .body,
        prompt_body
    );
    Ok(())
}

#[test]
fn signed_bedrock_reasoning_keeps_anthropic_answer_but_redacted_fails() -> TestResult {
    let values = vec![
        json!({"messageStart": {"role": "assistant"}}),
        json!({"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "think"}}}}),
        json!({"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"signature": "opaque"}}}}),
        json!({"contentBlockStop": {"contentBlockIndex": 0}}),
        json!({"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"text": "answer"}}}),
        json!({"contentBlockStop": {"contentBlockIndex": 1}}),
        json!({"messageStop": {"stopReason": "end_turn"}}),
        json!({"metadata": {"usage": usage()}}),
    ];
    let output = translate_events(&values, WireFormat::AnthropicMessages)?;
    assert!(!output.iter().any(|event| event["type"] == "error"));
    let aggregate = block_on(
        LlmResponse::Stream(decode_event_stream(
            stream::iter(output.into_iter().map(Ok)),
            WireFormat::AnthropicMessages,
        )?)
        .into_agg(),
    )?;
    assert!(
        aggregate.outputs[0]
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::Reasoning { text, .. } if text == "think"))
    );
    assert!(
        aggregate.outputs[0]
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::Text { text } if text == "answer"))
    );
    let engine = TranslationEngine::default();
    let mut state = StreamTranslationState::new(BEDROCK, WireFormat::AnthropicMessages);
    let output = engine.encode_stream_event(
        &mut state,
        WireFormat::AnthropicMessages,
        LlmResponseChunk::ReasoningDetailsDelta {
            index: 0,
            text: "visible".into(),
            details: vec![
                json!({"type": "bedrock.signature_delta", "signature": "opaque"}),
                json!({"type": "bedrock.redacted_content", "data": "YQ=="}),
            ],
        }
        .into(),
    )?;
    assert!(state.errored);
    assert_eq!(output[0]["type"], "error");
    assert!(
        output[0]["error"]["message"]
            .as_str()
            .is_some_and(|s| s.contains("redacted"))
    );
    assert!(
        engine
            .finish_stream(&mut state, WireFormat::AnthropicMessages)?
            .is_empty()
    );
    Ok(())
}
