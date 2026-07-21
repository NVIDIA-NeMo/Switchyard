// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for translating streaming provider events through the stream IR.

use pretty_assertions::assert_eq;
use serde_json::json;
use switchyard_protocol::{ResponseAccumulator, StopReason};
use switchyard_translation::{
    decode_stream_event, StreamTranslationState, TranslationEngine, WireFormat,
};

type TestResult = std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>;

// Verifies an OpenAI text delta opens the expected Anthropic message and content blocks.
#[test]
fn openai_chat_stream_event_translates_to_anthropic_message_events() -> TestResult {
    let engine = TranslationEngine::default();
    let mut state =
        StreamTranslationState::new(WireFormat::OpenAiChat, WireFormat::AnthropicMessages);
    let chunk = json!({
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "delta": {"content": "Hi"},
            "finish_reason": null
        }]
    });

    let events = engine.translate_event(
        &mut state,
        WireFormat::OpenAiChat,
        WireFormat::AnthropicMessages,
        &chunk,
    )?;

    assert_eq!(events[0]["type"], "message_start");
    assert_eq!(events[0]["message"]["model"], "gpt-4o");
    assert_eq!(
        events[1],
        json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""}
        })
    );
    assert_eq!(
        events[2],
        json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hi"}
        })
    );
    Ok(())
}

// Verifies Anthropic usage and stop events become terminal OpenAI chunks.
#[test]
fn anthropic_stream_usage_and_stop_translate_to_openai_chunks() -> TestResult {
    let engine = TranslationEngine::default();
    let mut state =
        StreamTranslationState::new(WireFormat::AnthropicMessages, WireFormat::OpenAiChat);
    let start = json!({
        "type": "message_start",
        "message": {"id": "msg_1", "model": "claude", "role": "assistant", "content": []}
    });
    engine.translate_event(
        &mut state,
        WireFormat::AnthropicMessages,
        WireFormat::OpenAiChat,
        &start,
    )?;

    let usage = json!({
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": 42}
    });
    let events = engine.translate_event(
        &mut state,
        WireFormat::AnthropicMessages,
        WireFormat::OpenAiChat,
        &usage,
    )?;

    assert_eq!(events[0]["usage"]["completion_tokens"], 42);
    assert_eq!(events[0]["choices"][0]["finish_reason"], "stop");
    Ok(())
}

// Verifies Chat target streams expose upstream model identity, not the client request alias.
#[test]
fn anthropic_to_openai_chat_uses_source_model_even_with_target_override() -> TestResult {
    let engine = TranslationEngine::default();
    let mut state =
        StreamTranslationState::new(WireFormat::AnthropicMessages, WireFormat::OpenAiChat);
    state.target_model = Some("gpt-client".to_string());
    let start = json!({
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "model": "claude-upstream",
            "role": "assistant",
            "content": []
        }
    });
    engine.translate_event(
        &mut state,
        WireFormat::AnthropicMessages,
        WireFormat::OpenAiChat,
        &start,
    )?;

    let delta = json!({
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "hello"}
    });
    let events = engine.translate_event(
        &mut state,
        WireFormat::AnthropicMessages,
        WireFormat::OpenAiChat,
        &delta,
    )?;

    assert_eq!(state.model.as_deref(), Some("claude-upstream"));
    assert_eq!(state.target_model.as_deref(), Some("gpt-client"));
    assert_eq!(events[0]["model"], "claude-upstream");
    Ok(())
}

// Verifies Anthropic target streams can expose a client model without losing source identity.
#[test]
fn responses_to_anthropic_uses_target_model_without_losing_source_model() -> TestResult {
    let engine = TranslationEngine::default();
    let mut state =
        StreamTranslationState::new(WireFormat::OpenAiResponses, WireFormat::AnthropicMessages);
    state.target_model = Some("claude-client".to_string());
    let created = json!({
        "type": "response.created",
        "response": {"id": "resp_1", "model": "responses-upstream"}
    });

    let events = engine.translate_event(
        &mut state,
        WireFormat::OpenAiResponses,
        WireFormat::AnthropicMessages,
        &created,
    )?;

    assert_eq!(state.model.as_deref(), Some("responses-upstream"));
    assert_eq!(state.target_model.as_deref(), Some("claude-client"));
    assert_eq!(events[0]["message"]["model"], "claude-client");
    Ok(())
}

// Verifies Responses target streams can expose a client model while retaining source identity.
#[test]
fn openai_chat_to_responses_uses_target_model_without_losing_source_model() -> TestResult {
    let engine = TranslationEngine::default();
    let mut state =
        StreamTranslationState::new(WireFormat::OpenAiChat, WireFormat::OpenAiResponses);
    state.target_model = Some("responses-client".to_string());
    let chunk = json!({
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "model": "gpt-upstream",
        "choices": [{
            "index": 0,
            "delta": {"content": "hello"},
            "finish_reason": null
        }]
    });

    let events = engine.translate_event(
        &mut state,
        WireFormat::OpenAiChat,
        WireFormat::OpenAiResponses,
        &chunk,
    )?;

    assert_eq!(state.model.as_deref(), Some("gpt-upstream"));
    assert_eq!(state.target_model.as_deref(), Some("responses-client"));
    assert_eq!(events[0]["response"]["model"], "responses-client");
    Ok(())
}

// Verifies OpenAI Chat finish emits a terminal chunk when the source closes without one.
#[test]
fn openai_chat_finish_synthesizes_terminal_chunk_after_incomplete_source() -> TestResult {
    let engine = TranslationEngine::default();
    let mut state = StreamTranslationState::new(WireFormat::OpenAiChat, WireFormat::OpenAiChat);
    let chunk = json!({
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "delta": {"content": "hello"},
            "finish_reason": null
        }]
    });

    let mut events = engine.translate_event(
        &mut state,
        WireFormat::OpenAiChat,
        WireFormat::OpenAiChat,
        &chunk,
    )?;
    events.extend(engine.finish_stream(&mut state, WireFormat::OpenAiChat)?);

    let Some(terminal) = events.last() else {
        return Err("finish should emit a terminal OpenAI Chat chunk".into());
    };
    assert_eq!(terminal["choices"][0]["delta"], json!({}));
    assert_eq!(terminal["choices"][0]["finish_reason"], "stop");
    assert_eq!(terminal["usage"]["total_tokens"], 0);
    Ok(())
}

// Verifies OpenAI Chat streaming usage preserves reasoning details for Responses clients.
#[test]
fn openai_chat_stream_reasoning_usage_translates_to_responses_usage_details() -> TestResult {
    let engine = TranslationEngine::default();
    let mut state =
        StreamTranslationState::new(WireFormat::OpenAiChat, WireFormat::OpenAiResponses);
    let usage = json!({
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "model": "gpt-reasoning",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "completion_tokens_details": {"reasoning_tokens": 3}
        }
    });

    let mut events = engine.translate_event(
        &mut state,
        WireFormat::OpenAiChat,
        WireFormat::OpenAiResponses,
        &usage,
    )?;
    events.extend(engine.finish_stream(&mut state, WireFormat::OpenAiResponses)?);

    let Some(completed) = events
        .iter()
        .find(|event| event["type"] == "response.completed")
    else {
        return Err("expected final Responses completion event".into());
    };
    assert_eq!(
        completed["response"]["usage"]["output_tokens_details"],
        json!({"reasoning_tokens": 3})
    );
    Ok(())
}

// Verifies Responses text deltas become OpenAI Chat content chunks.
#[test]
fn responses_stream_delta_translates_to_openai_chat_chunk() -> TestResult {
    let engine = TranslationEngine::default();
    let mut state =
        StreamTranslationState::new(WireFormat::OpenAiResponses, WireFormat::OpenAiChat);
    let created = json!({
        "type": "response.created",
        "response": {"id": "resp_1", "model": "gpt-4o"}
    });
    engine.translate_event(
        &mut state,
        WireFormat::OpenAiResponses,
        WireFormat::OpenAiChat,
        &created,
    )?;

    let delta = json!({
        "type": "response.output_text.delta",
        "output_index": 0,
        "delta": "hello"
    });
    let events = engine.translate_event(
        &mut state,
        WireFormat::OpenAiResponses,
        WireFormat::OpenAiChat,
        &delta,
    )?;

    assert_eq!(events[0]["model"], "gpt-4o");
    assert_eq!(events[0]["choices"][0]["delta"]["content"], "hello");
    Ok(())
}

// Verifies OpenAI-compatible reasoning deltas become Anthropic thinking, not text content.
#[test]
fn openai_chat_reasoning_stream_fields_do_not_become_anthropic_text() -> TestResult {
    let engine = TranslationEngine::default();
    let mut state =
        StreamTranslationState::new(WireFormat::OpenAiChat, WireFormat::AnthropicMessages);
    let chunk = json!({
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "model": "nvidia/nemotron",
        "choices": [{
            "index": 0,
            "delta": {
                "reasoning": "private reasoning",
                "reasoning_content": "private reasoning content"
            },
            "finish_reason": null
        }]
    });

    let mut events = engine.translate_event(
        &mut state,
        WireFormat::OpenAiChat,
        WireFormat::AnthropicMessages,
        &chunk,
    )?;
    events.extend(engine.finish_stream(&mut state, WireFormat::AnthropicMessages)?);

    let serialized = serde_json::to_string(&events)?;
    assert!(serialized.contains("private reasoning"));
    assert!(serialized.contains("private reasoning content"));
    assert!(!serialized.contains("reasoning_content"));
    assert!(events.iter().any(|event| {
        event["type"] == "content_block_start"
            && event["content_block"]["type"] == "thinking"
            && event["content_block"]["signature"] == ""
    }));
    assert!(events.iter().any(|event| {
        event["type"] == "content_block_delta"
            && event["delta"]["type"] == "thinking_delta"
            && event["delta"]["thinking"] == "private reasoning content"
    }));
    assert!(events.iter().any(|event| {
        event["type"] == "content_block_delta"
            && event["delta"]["type"] == "signature_delta"
            && event["delta"]["signature"] == ""
    }));
    assert!(!events.iter().any(|event| {
        event["type"] == "content_block_delta"
            && event["delta"]["type"] == "text_delta"
            && event["delta"]["text"]
                .as_str()
                .is_some_and(|text| text.contains("private reasoning"))
    }));
    Ok(())
}

// Verifies Anthropic thinking deltas become OpenAI reasoning_content, not content.
#[test]
fn anthropic_thinking_stream_deltas_do_not_become_openai_chat_content() -> TestResult {
    let engine = TranslationEngine::default();
    let mut state =
        StreamTranslationState::new(WireFormat::AnthropicMessages, WireFormat::OpenAiChat);
    let start = json!({
        "type": "message_start",
        "message": {"id": "msg_1", "model": "claude", "role": "assistant", "content": []}
    });
    let thinking_start = json!({
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "thinking", "thinking": ""}
    });
    let thinking_delta = json!({
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "thinking_delta", "thinking": "private chain of thought"}
    });
    let signature_delta = json!({
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "signature_delta", "signature": "opaque-signature"}
    });

    let mut events = Vec::new();
    events.extend(engine.translate_event(
        &mut state,
        WireFormat::AnthropicMessages,
        WireFormat::OpenAiChat,
        &start,
    )?);
    events.extend(engine.translate_event(
        &mut state,
        WireFormat::AnthropicMessages,
        WireFormat::OpenAiChat,
        &thinking_start,
    )?);
    events.extend(engine.translate_event(
        &mut state,
        WireFormat::AnthropicMessages,
        WireFormat::OpenAiChat,
        &thinking_delta,
    )?);
    events.extend(engine.translate_event(
        &mut state,
        WireFormat::AnthropicMessages,
        WireFormat::OpenAiChat,
        &signature_delta,
    )?);

    let serialized = serde_json::to_string(&events)?;
    assert!(serialized.contains("private chain of thought"));
    assert!(!serialized.contains("opaque-signature"));
    assert!(events.iter().any(|event| {
        event["choices"][0]["delta"]["reasoning_content"] == "private chain of thought"
    }));
    assert!(!events
        .iter()
        .any(|event| { event["choices"][0]["delta"]["content"] == "private chain of thought" }));
    Ok(())
}

// A real Anthropic stream carries its stop reason on `message_delta` and then always
// sends a reasonless `message_stop`. Decoding both and folding the chunks must preserve
// the provider reason (here `max_tokens`) — the terminal `message_stop` must not emit a
// second, reasonless `MessageStop` that the accumulator would fold to `EndTurn`.
#[test]
fn anthropic_message_stop_does_not_overwrite_max_tokens_stop_reason() -> TestResult {
    let mut state =
        StreamTranslationState::new(WireFormat::AnthropicMessages, WireFormat::AnthropicMessages);
    let events = vec![
        json!({"type": "message_start", "message": {"id": "msg_1", "model": "claude", "usage": {"input_tokens": 3}}}),
        json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}),
        json!({"type": "message_delta", "delta": {"stop_reason": "max_tokens"}, "usage": {"output_tokens": 5}}),
        json!({"type": "message_stop"}),
    ];

    let mut accumulator = ResponseAccumulator::new();
    for event in &events {
        for chunk in decode_stream_event(&mut state, WireFormat::AnthropicMessages, event) {
            accumulator.push(chunk);
        }
    }
    let aggregate = accumulator.finish();

    assert_eq!(
        aggregate.outputs[0].stop_reason,
        Some(StopReason::MaxTokens),
        "the reasonless message_stop must not overwrite the max_tokens stop reason"
    );
    Ok(())
}
