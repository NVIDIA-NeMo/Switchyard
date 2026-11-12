// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use futures::{StreamExt, stream};
use serde_json::json;
use switchyard_translation::{LlmResponseChunk, WireFormat, decode_stream, response_usage};

#[test]
fn failure_usage_reuses_provider_normalization_and_preserves_unknowns() {
    let body = json!({"status":"failed", "error":{"message":"failed"}, "usage":{"input_tokens":20,"input_tokens_details":{"cached_tokens":3,"cache_write_tokens":2},"output_tokens":4,"cost":0.0}});
    let usage = response_usage(&body, WireFormat::OpenAiResponses);
    assert_eq!(usage.input_tokens, Some(15));
    assert_eq!(usage.cached_input_tokens(), Some(3));
    assert_eq!(usage.cache_creation_input_tokens(), Some(2));
    assert_eq!(usage.total_tokens, Some(24));
    assert_eq!(usage.provider_cost, Some(0.0));
    assert_eq!(
        response_usage(
            &json!({"error":{"usage":{"input_tokens":10}}}),
            WireFormat::OpenAiResponses
        ),
        Default::default()
    );
    assert_eq!(
        response_usage(&json!({"usage":{}}), WireFormat::AnthropicMessages),
        Default::default()
    );
}

#[test]
fn responses_terminal_failures_preserve_usage_and_original_event() {
    futures::executor::block_on(async {
        for kind in ["response.failed", "response.incomplete"] {
            let raw = json!({"type":kind,"response":{"id":"r1","status":kind.strip_prefix("response.").unwrap(),"error":{"message":"failed"},"usage":{"input_tokens":11,"output_tokens":3}}});
            let bytes = format!("data: {raw}\n\n").into_bytes();
            let mut events = decode_stream(
                stream::iter(vec![Ok(bytes)]).boxed(),
                WireFormat::OpenAiResponses,
            )
            .unwrap();
            let event = events.next().await.unwrap().unwrap();
            assert_eq!(event.preservation().unwrap().raw(), &raw);
            let [LlmResponseChunk::Usage(usage), terminal] = event.normalized() else {
                panic!("usage must precede terminal bookkeeping: {event:?}")
            };
            assert_eq!(usage.input_tokens, Some(11));
            assert_eq!(usage.output_tokens, Some(3));
            assert!(matches!(
                (kind, terminal),
                ("response.failed", LlmResponseChunk::StreamError { .. })
                    | ("response.incomplete", LlmResponseChunk::MessageStop { .. })
            ));
        }
    });
}
