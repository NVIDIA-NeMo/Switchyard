// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Intake response processor tests for buffered, streaming, and fail-open paths.

mod support;

use std::sync::Arc;

use serde_json::json;
use switchyard_components::{
    IntakeRequestState, IntakeResponseProcessor, IntakeSinkConfig, SubModelCall, SubModelCalls,
};
use switchyard_core::{
    ChatRequestType, ChatResponse, ModelId, ProxyContext, Result, StreamEvent, SwitchyardError,
};

use support::intake::{
    completion, drain_stream, opted_in_context, record_backend_selection, RecordingSink,
};

// Skip metadata should preserve the response and avoid sink writes.
#[tokio::test]
async fn response_processor_skip_metadata_leaves_response_untouched() -> Result<()> {
    let sink = Arc::new(RecordingSink::default());
    let processor = IntakeResponseProcessor::new(IntakeSinkConfig::default(), sink.clone());
    let mut ctx = ProxyContext::new();
    ctx.insert(IntakeRequestState {
        started_at_ms: 1,
        inbound_format: ChatRequestType::OpenAiChat,
        session_id: None,
        skip: true,
        request_snapshot: None,
    });

    let returned = processor
        .process(&mut ctx, completion("chatcmpl-test", "world"))
        .await?;

    assert!(matches!(returned, ChatResponse::OpenAiCompletion(_)));
    assert!(sink.payloads()?.is_empty());
    Ok(())
}

// Buffered responses should enqueue a single chat-completions ingest payload.
#[tokio::test]
async fn response_processor_enqueues_non_streaming_payload() -> Result<()> {
    let sink = Arc::new(RecordingSink::default());
    let processor = IntakeResponseProcessor::new(
        IntakeSinkConfig {
            workspace: Some("default".to_string()),
            capture_content: true,
            ..IntakeSinkConfig::default()
        },
        sink.clone(),
    );
    let mut ctx = opted_in_context();

    let returned = processor
        .process(&mut ctx, completion("chatcmpl-test", "world"))
        .await?;

    assert!(matches!(returned, ChatResponse::OpenAiCompletion(_)));
    let payloads = sink.payloads()?;
    assert_eq!(payloads.len(), 1);
    assert_eq!(
        payloads[0]["response"]["choices"][0]["message"]["content"],
        "world"
    );
    assert!(payloads[0]["request"]["switchyard"].get("app").is_none());
    assert_eq!(payloads[0]["session_id"], "session-123");
    assert_eq!(payloads[0]["response"]["id"], "chatcmpl-test");
    Ok(())
}

// Routing sub-model calls stashed on context each emit their own record next to
// the primary one, all through the same sink.
#[tokio::test]
async fn response_processor_emits_a_record_per_submodel_call() -> Result<()> {
    let sink = Arc::new(RecordingSink::default());
    let processor = IntakeResponseProcessor::new(
        IntakeSinkConfig {
            workspace: Some("default".to_string()),
            ..IntakeSinkConfig::default()
        },
        sink.clone(),
    );
    let mut ctx = opted_in_context();
    ctx.insert(SubModelCalls(vec![
        SubModelCall {
            model: "router-clf".to_string(),
            prompt_tokens: 100,
            completion_tokens: 10,
            cached_tokens: 0,
            router_type: "deterministic".to_string(),
            routed_to: "classifier".to_string(),
        },
        SubModelCall {
            model: "planner".to_string(),
            prompt_tokens: 50,
            completion_tokens: 5,
            cached_tokens: 0,
            router_type: "plan_execute".to_string(),
            routed_to: "planner".to_string(),
        },
    ]));

    processor
        .process(&mut ctx, completion("chatcmpl-test", "world"))
        .await?;

    // One primary record plus one per routing call.
    let payloads = sink.payloads()?;
    assert_eq!(payloads.len(), 3);
    let routed: Vec<&str> = payloads
        .iter()
        .filter_map(|payload| payload["request"]["switchyard"]["routing"]["routed_to"].as_str())
        .collect();
    assert!(routed.contains(&"classifier"));
    assert!(routed.contains(&"planner"));
    Ok(())
}

// The opt-in gate covers sub-model records too: a skipped turn emits nothing,
// even with routing calls stashed on context.
#[tokio::test]
async fn response_processor_skips_submodel_records_when_opted_out() -> Result<()> {
    let sink = Arc::new(RecordingSink::default());
    let processor = IntakeResponseProcessor::new(IntakeSinkConfig::default(), sink.clone());
    let mut ctx = ProxyContext::new();
    ctx.insert(IntakeRequestState {
        started_at_ms: 1,
        inbound_format: ChatRequestType::OpenAiChat,
        session_id: None,
        skip: true,
        request_snapshot: None,
    });
    ctx.insert(SubModelCalls(vec![SubModelCall {
        model: "router-clf".to_string(),
        prompt_tokens: 100,
        completion_tokens: 10,
        cached_tokens: 0,
        router_type: "deterministic".to_string(),
        routed_to: "classifier".to_string(),
    }]));

    processor
        .process(&mut ctx, completion("chatcmpl-test", "world"))
        .await?;

    assert!(sink.payloads()?.is_empty());
    Ok(())
}

// Intake failures must be fail-open so user responses are not replaced.
#[tokio::test]
async fn response_processor_fail_open_when_payload_build_or_sink_enqueue_fails() -> Result<()> {
    let missing_snapshot_sink = Arc::new(RecordingSink::default());
    let processor =
        IntakeResponseProcessor::new(IntakeSinkConfig::default(), missing_snapshot_sink.clone());
    let mut missing_snapshot_ctx = ProxyContext::new();

    let returned = processor
        .process(
            &mut missing_snapshot_ctx,
            completion("chatcmpl-test", "world"),
        )
        .await?;
    assert!(matches!(returned, ChatResponse::OpenAiCompletion(_)));
    assert!(missing_snapshot_sink.payloads()?.is_empty());

    let failing_sink = Arc::new(RecordingSink::with_error(SwitchyardError::Upstream(
        "intake down".to_string(),
    )));
    let processor = IntakeResponseProcessor::new(IntakeSinkConfig::default(), failing_sink.clone());
    let mut ctx = opted_in_context();
    let returned = processor
        .process(&mut ctx, completion("chatcmpl-test", "world"))
        .await?;
    assert!(matches!(returned, ChatResponse::OpenAiCompletion(_)));
    assert!(failing_sink.payloads()?.is_empty());
    Ok(())
}

// OpenAI streams should pass through unchanged and enqueue after stream completion.
#[tokio::test]
async fn response_processor_wraps_openai_stream_and_enqueues_after_stream_end() -> Result<()> {
    let sink = Arc::new(RecordingSink::default());
    let processor = IntakeResponseProcessor::new(
        IntakeSinkConfig {
            workspace: Some("default".to_string()),
            capture_content: true,
            ..IntakeSinkConfig::default()
        },
        sink.clone(),
    );
    let mut ctx = opted_in_context();
    let events = vec![
        StreamEvent::Json(json!({
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "model": "gpt-4o",
            "choices": [{"delta": {"content": "hel"}, "finish_reason": null}]
        })),
        StreamEvent::Json(json!({
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "model": "gpt-4o",
            "choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        })),
    ];
    let stream = futures_util::stream::iter(events.clone().into_iter().map(Ok));

    let returned = processor
        .process(&mut ctx, ChatResponse::OpenAiStream(Box::pin(stream)))
        .await?;
    let drained = drain_stream(returned).await?;

    assert_eq!(drained, events);
    let payloads = sink.payloads()?;
    assert_eq!(payloads.len(), 1);
    assert_eq!(
        payloads[0]["response"]["choices"][0]["message"]["content"],
        "hello"
    );
    assert_eq!(payloads[0]["response"]["usage"]["total_tokens"], 15);
    Ok(())
}

// OpenAI streams without usage still produce a valid response payload.
#[tokio::test]
async fn response_processor_wraps_openai_stream_without_usage() -> Result<()> {
    let sink = Arc::new(RecordingSink::default());
    let processor = IntakeResponseProcessor::new(
        IntakeSinkConfig {
            workspace: Some("default".to_string()),
            capture_content: true,
            ..IntakeSinkConfig::default()
        },
        sink.clone(),
    );
    let mut ctx = opted_in_context();
    let events = vec![
        StreamEvent::Json(json!({
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "model": "gpt-4o",
            "choices": [{"delta": {"content": "hel"}, "finish_reason": null}]
        })),
        StreamEvent::Json(json!({
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "model": "gpt-4o",
            "choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]
        })),
    ];
    let stream = futures_util::stream::iter(events.clone().into_iter().map(Ok));

    let returned = processor
        .process(&mut ctx, ChatResponse::OpenAiStream(Box::pin(stream)))
        .await?;
    let drained = drain_stream(returned).await?;

    assert_eq!(drained, events);
    let payloads = sink.payloads()?;
    assert_eq!(payloads.len(), 1);
    assert_eq!(
        payloads[0]["response"]["choices"][0]["message"]["content"],
        "hello"
    );
    assert!(payloads[0]["response"].get("usage").is_none());
    Ok(())
}

// OpenAI stream capture must preserve tool-call deltas and reasoning fields.
#[tokio::test]
async fn response_processor_wraps_openai_stream_tool_calls_and_reasoning() -> Result<()> {
    let sink = Arc::new(RecordingSink::default());
    let processor = IntakeResponseProcessor::new(
        IntakeSinkConfig {
            capture_content: true,
            ..IntakeSinkConfig::default()
        },
        sink.clone(),
    );
    let mut ctx = opted_in_context();
    let events = vec![
        StreamEvent::Json(json!({
            "id": "chatcmpl-tools",
            "object": "chat.completion.chunk",
            "model": "gpt-4o",
            "choices": [{
                "delta": {
                    "reasoning_content": "private ",
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{\"q\":"}
                    }]
                },
                "finish_reason": null
            }]
        })),
        StreamEvent::Json(json!({
            "id": "chatcmpl-tools",
            "object": "chat.completion.chunk",
            "model": "gpt-4o",
            "choices": [{
                "delta": {
                    "reasoning_content": "reasoning",
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": "\"x\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        })),
    ];
    let stream = futures_util::stream::iter(events.clone().into_iter().map(Ok));

    let returned = processor
        .process(&mut ctx, ChatResponse::OpenAiStream(Box::pin(stream)))
        .await?;
    let drained = drain_stream(returned).await?;

    assert_eq!(drained, events);
    let payloads = sink.payloads()?;
    let message = &payloads[0]["response"]["choices"][0]["message"];
    assert_eq!(message["reasoning_content"], "private reasoning");
    assert_eq!(message["tool_calls"][0]["id"], "call_1");
    assert_eq!(
        message["tool_calls"][0]["function"]["arguments"],
        "{\"q\":\"x\"}"
    );
    assert_eq!(
        payloads[0]["response"]["choices"][0]["finish_reason"],
        "tool_calls"
    );
    Ok(())
}

// Anthropic stream usage should become OpenAI-style prompt/completion usage.
#[tokio::test]
async fn response_processor_wraps_anthropic_stream_usage() -> Result<()> {
    let sink = Arc::new(RecordingSink::default());
    let processor = IntakeResponseProcessor::new(
        IntakeSinkConfig {
            capture_content: true,
            ..IntakeSinkConfig::default()
        },
        sink.clone(),
    );
    let mut ctx = opted_in_context();
    record_backend_selection(&mut ctx, ModelId::from_static("claude-opus-4-6"));
    let events = vec![
        StreamEvent::Json(json!({
            "type": "message_start",
            "message": {
                "id": "msg_123",
                "model": "claude-opus-4-6",
                "usage": {"input_tokens": 7}
            }
        })),
        StreamEvent::Json(json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hello"}
        })),
        StreamEvent::Json(json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 3}
        })),
        StreamEvent::Json(json!({"type": "message_stop"})),
    ];
    let stream = futures_util::stream::iter(events.clone().into_iter().map(Ok));

    let returned = processor
        .process(&mut ctx, ChatResponse::AnthropicStream(Box::pin(stream)))
        .await?;
    let drained = drain_stream(returned).await?;

    assert_eq!(drained, events);
    let payloads = sink.payloads()?;
    assert_eq!(
        payloads[0]["response"]["choices"][0]["message"]["content"],
        "hello"
    );
    assert_eq!(payloads[0]["response"]["usage"]["prompt_tokens"], 7);
    assert_eq!(payloads[0]["response"]["usage"]["completion_tokens"], 3);
    Ok(())
}

// Anthropic tool-use events should become OpenAI Chat tool calls in intake.
#[tokio::test]
async fn response_processor_wraps_anthropic_stream_tool_use() -> Result<()> {
    let sink = Arc::new(RecordingSink::default());
    let processor = IntakeResponseProcessor::new(
        IntakeSinkConfig {
            capture_content: true,
            ..IntakeSinkConfig::default()
        },
        sink.clone(),
    );
    let mut ctx = opted_in_context();
    record_backend_selection(&mut ctx, ModelId::from_static("claude-opus-4-6"));
    let events = vec![
        StreamEvent::Json(json!({
            "type": "message_start",
            "message": {"id": "msg_tools", "model": "claude-opus-4-6"}
        })),
        StreamEvent::Json(json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "search"}
        })),
        StreamEvent::Json(json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": "{\"q\":\"x\"}"}
        })),
        StreamEvent::Json(json!({
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"}
        })),
        StreamEvent::Json(json!({"type": "message_stop"})),
    ];
    let stream = futures_util::stream::iter(events.clone().into_iter().map(Ok));

    let returned = processor
        .process(&mut ctx, ChatResponse::AnthropicStream(Box::pin(stream)))
        .await?;
    drain_stream(returned).await?;

    let payloads = sink.payloads()?;
    let response = &payloads[0]["response"];
    assert_eq!(response["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(
        response["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "search"
    );
    assert_eq!(
        response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
        "{\"q\":\"x\"}"
    );
    Ok(())
}

// Responses streams with completed response events should use the canonical response.
#[tokio::test]
async fn response_processor_wraps_responses_stream_completed_response() -> Result<()> {
    let sink = Arc::new(RecordingSink::default());
    let processor = IntakeResponseProcessor::new(
        IntakeSinkConfig {
            capture_content: true,
            ..IntakeSinkConfig::default()
        },
        sink.clone(),
    );
    let mut ctx = opted_in_context();
    record_backend_selection(&mut ctx, ModelId::from_static("gpt-4o"));
    let response_body = json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 1_700_000_000,
        "model": "gpt-4o",
        "output": [{
            "type": "message",
            "id": "msg_123",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": "hello",
                "annotations": []
            }]
        }],
        "status": "completed",
        "parallel_tool_calls": true,
        "tool_choice": "auto",
        "tools": [],
        "text": {"format": {"type": "text"}},
        "usage": {
            "input_tokens": 11,
            "output_tokens": 4,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0}
        }
    });
    let events = vec![
        StreamEvent::Json(json!({"type": "response.output_text.delta", "delta": "hello"})),
        StreamEvent::Json(json!({
            "type": "response.completed",
            "response": response_body
        })),
    ];
    let stream = futures_util::stream::iter(events.clone().into_iter().map(Ok));

    let returned = processor
        .process(
            &mut ctx,
            ChatResponse::OpenAiResponsesStream(Box::pin(stream)),
        )
        .await?;
    let drained = drain_stream(returned).await?;

    assert_eq!(drained, events);
    let payloads = sink.payloads()?;
    assert_eq!(
        payloads[0]["response"]["choices"][0]["message"]["content"],
        "hello"
    );
    assert_eq!(payloads[0]["response"]["usage"]["prompt_tokens"], 11);
    assert_eq!(payloads[0]["response"]["usage"]["completion_tokens"], 4);
    Ok(())
}

// Responses streams without a completed event should synthesize the final response.
#[tokio::test]
async fn response_processor_wraps_responses_stream_function_call_without_completed_event(
) -> Result<()> {
    let sink = Arc::new(RecordingSink::default());
    let processor = IntakeResponseProcessor::new(
        IntakeSinkConfig {
            capture_content: true,
            ..IntakeSinkConfig::default()
        },
        sink.clone(),
    );
    let mut ctx = opted_in_context();
    record_backend_selection(&mut ctx, ModelId::from_static("gpt-4o"));
    let events = vec![
        StreamEvent::Json(json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "search",
                "arguments": ""
            }
        })),
        StreamEvent::Json(json!({
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "delta": "{\"q\":\"x\"}"
        })),
    ];
    let stream = futures_util::stream::iter(events.clone().into_iter().map(Ok));

    let returned = processor
        .process(
            &mut ctx,
            ChatResponse::OpenAiResponsesStream(Box::pin(stream)),
        )
        .await?;
    drain_stream(returned).await?;

    let payloads = sink.payloads()?;
    let message = &payloads[0]["response"]["choices"][0]["message"];
    assert_eq!(message["tool_calls"][0]["id"], "call_1");
    assert_eq!(message["tool_calls"][0]["function"]["name"], "search");
    assert_eq!(
        message["tool_calls"][0]["function"]["arguments"],
        "{\"q\":\"x\"}"
    );
    Ok(())
}

// Shutdown should delegate to the configured sink exactly once.
#[tokio::test]
async fn response_processor_delegates_shutdown_to_sink() -> Result<()> {
    let sink = Arc::new(RecordingSink::default());
    let processor = IntakeResponseProcessor::new(IntakeSinkConfig::default(), sink.clone());

    processor.shutdown().await?;

    assert_eq!(sink.shutdowns()?, 1);
    Ok(())
}
