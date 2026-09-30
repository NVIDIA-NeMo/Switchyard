// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for translating interleaved Chat tool calls to Anthropic blocks.

use serde_json::{Value, json};
use switchyard_translation::{StreamTranslationState, TranslationEngine, WireFormat};

type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

fn translate(
    engine: &TranslationEngine,
    state: &mut StreamTranslationState,
    delta: Value,
) -> Result<Vec<Value>, switchyard_translation::TranslationError> {
    engine.translate_event(
        state,
        WireFormat::OpenAiChat,
        WireFormat::AnthropicMessages,
        &json!({
            "id": "chatcmpl-parallel",
            "object": "chat.completion.chunk",
            "model": "test-model",
            "choices": [{"index": 0, "delta": delta, "finish_reason": null}]
        }),
    )
}

#[test]
fn parallel_chat_tools_stream_as_ordered_nonoverlapping_anthropic_blocks() -> TestResult {
    // Source tool indexes need not match target content indexes or arrival order.
    for [first, second, third] in [[0, 1, 2], [7, 2, 1]] {
        let engine = TranslationEngine::default();
        let target = WireFormat::AnthropicMessages;
        let mut state = StreamTranslationState::new(WireFormat::OpenAiChat, target);
        let mut events = translate(&engine, &mut state, json!({"content": "Checking weather."}))?;
        events.extend(translate(
            &engine,
            &mut state,
            json!({"tool_calls": [
                {"index": first, "id": "call_paris", "type": "function",
                 "function": {"name": "weather", "arguments": ""}},
                {"index": second, "id": "call_tokyo", "type": "function",
                 "function": {"name": "weather", "arguments": ""}},
                {"index": third, "id": "call_clock", "type": "function",
                 "function": {"name": "clock", "arguments": "{}"}}
            ]}),
        )?);
        let first_fragments = translate(
            &engine,
            &mut state,
            json!({"tool_calls": [
                {"index": first, "function": {"arguments": "{\"city\":\"Pa"}},
                {"index": second, "function": {"arguments": "{\"city\":\"To"}}
            ]}),
        )?;
        assert!(
            first_fragments.is_empty(),
            "tool names are not complete yet"
        );
        events.extend(first_fragments);
        events.extend(translate(
            &engine,
            &mut state,
            json!({"tool_calls": [
                {"index": second, "function": {"arguments": "kyo\"}"}},
                {"index": first, "function": {"arguments": "ris\"}"}}
            ]}),
        )?);
        events.extend(engine.translate_event(
            &mut state,
            WireFormat::OpenAiChat,
            target,
            &json!({
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}
            }),
        )?);
        events.extend(engine.finish_stream(&mut state, target)?);

        let mut active = None;
        let mut block = Value::Null;
        let mut arguments = String::new();
        let mut completed_tools = Vec::new();
        let mut next_index = 0;
        for event in &events {
            match event["type"].as_str() {
                Some("content_block_start") => {
                    assert!(active.is_none(), "overlapping content blocks: {event}");
                    assert_eq!(event["index"], next_index);
                    next_index += 1;
                    active = Some(event["index"].clone());
                    block = event["content_block"].clone();
                    arguments.clear();
                }
                Some("content_block_delta") => {
                    assert_eq!(active.as_ref(), Some(&event["index"]));
                    if let Some(delta) = event["delta"]["partial_json"].as_str() {
                        arguments.push_str(delta);
                    }
                }
                Some("content_block_stop") => {
                    assert_eq!(active.take(), Some(event["index"].clone()));
                    if block["type"] == "tool_use" {
                        block["input"] = serde_json::from_str(&arguments)?;
                        completed_tools.push(block.clone());
                    }
                }
                Some("message_delta" | "message_stop") => assert!(active.is_none()),
                _ => {}
            }
        }
        assert!(active.is_none());
        assert_eq!(
            completed_tools,
            vec![
                json!({"type": "tool_use", "id": "call_paris", "name": "weather", "input": {"city": "Paris"}}),
                json!({"type": "tool_use", "id": "call_tokyo", "name": "weather", "input": {"city": "Tokyo"}}),
                json!({"type": "tool_use", "id": "call_clock", "name": "clock", "input": {}})
            ]
        );
        let terminal = events
            .iter()
            .find(|event| event["type"] == "message_delta")
            .ok_or("missing terminal delta")?;
        assert_eq!(terminal["delta"]["stop_reason"], "tool_use");
        assert_eq!(terminal["usage"]["output_tokens"], 6);
        assert_eq!(
            events.last().ok_or("missing terminal event")?["type"],
            "message_stop"
        );
        assert!(engine.finish_stream(&mut state, target)?.is_empty());
    }
    Ok(())
}
