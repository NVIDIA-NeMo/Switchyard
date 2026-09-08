// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end check of the escalation serving path for freeform tools: a Codex request with a
//! `custom` tool is decoded, a buffered reply that calls the tool is re-streamed with the
//! request's extensions, and the client must receive a `custom_tool_call` item.

use futures::StreamExt;
use serde_json::json;
use switchyard_protocol::{
    AggLlmResponse, ContentBlock, PreservationMetadata, ProviderExtensions, ResponseOutput, Role,
    StopReason, ToolCall, Usage,
};
use switchyard_translation::{WireFormat, decode_request, encode_stream_with_extensions};

#[test]
fn escalation_restream_rewrites_custom_tool_calls_for_the_client() {
    futures::executor::block_on(run());
}

async fn run() {
    let request = json!({
        "model": "gpt-5.6-luna-switchyard",
        "instructions": "You are Codex.",
        "input": [{"type": "message", "role": "user", "content": "List files"}],
        "tools": [
            {"type": "custom", "name": "exec", "description": "Run JS.",
             "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.*/"}},
            {"type": "function", "name": "update_plan", "description": "Plan", "parameters": {"type": "object"}}
        ],
        "stream": true
    });
    let decoded = decode_request(WireFormat::OpenAiResponses, &request).expect("decode");
    assert!(
        decoded
            .extensions
            .fields
            .contains_key("switchyard_codex_custom_tools"),
        "custom tools must be recorded on the request extensions: {:?}",
        decoded.extensions.fields.keys().collect::<Vec<_>>()
    );

    let agg = AggLlmResponse {
        id: Some("resp_luna".to_string()),
        model: Some("gpt-5.6-luna".to_string()),
        outputs: vec![ResponseOutput {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "call_1".to_string(),
                name: "exec".to_string(),
                arguments: json!({"input": "const p = await tools.update_plan({});"}),
            })],
            stop_reason: Some(StopReason::ToolUse),
        }],
        usage: Usage::default(),
        extensions: ProviderExtensions::default(),
        preservation: PreservationMetadata::default(),
    };
    let mut events = encode_stream_with_extensions(
        agg.into_stream(),
        WireFormat::OpenAiResponses,
        Some("gpt-5.6-luna-switchyard".to_string()),
        &decoded.extensions,
    )
    .expect("encode");
    let mut collected = Vec::new();
    while let Some(event) = events.next().await {
        collected.push(event.expect("event"));
    }
    let done = collected
        .iter()
        .find(|event| {
            event["type"] == "response.output_item.done" && event["item"]["name"] == "exec"
        })
        .expect("tool item done event");
    assert_eq!(done["item"]["type"], "custom_tool_call", "{done}");
    assert_eq!(
        done["item"]["input"], "const p = await tools.update_plan({});",
        "{done}"
    );
    assert!(
        !collected
            .iter()
            .any(|event| event["type"] == "response.function_call_arguments.delta"),
        "argument deltas for a custom tool must be dropped"
    );
    let completed = collected
        .iter()
        .find(|event| event["type"] == "response.completed")
        .expect("completed");
    assert_eq!(
        completed["response"]["output"][0]["type"],
        "custom_tool_call"
    );
}
