// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mutation-aware request preservation tests.

use serde_json::{Value, json};
use switchyard_translation::{
    ContentBlock, InstructionBlock, Role, TranslationEngine, TranslationPolicy, WireFormat,
};

type TestResult = std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>;

fn mutate_and_encode(format: WireFormat, body: Value) -> TestResult {
    let engine = TranslationEngine::default();
    let policy = TranslationPolicy::default();
    let mut request = engine.decode_request(format, &body, &policy)?.request;
    request.instructions.insert(
        0,
        InstructionBlock {
            role: Role::System,
            content: vec![ContentBlock::Text {
                text: "route with the capable tier".to_string(),
            }],
        },
    );
    request.invalidate_preserved_requests();

    let encoded = engine.encode_request(format, &request, &policy)?.body;
    assert_eq!(encoded["provider_extension"], json!({"preserved": true}));
    match format {
        WireFormat::OpenAiChat => {
            assert_eq!(encoded["messages"][0]["role"], "system");
            assert_eq!(
                encoded["messages"][0]["content"],
                "route with the capable tier"
            );
        }
        WireFormat::OpenAiResponses => {
            assert_eq!(encoded["instructions"], "route with the capable tier");
        }
        WireFormat::AnthropicMessages => {
            assert_eq!(encoded["system"], "route with the capable tier");
        }
    }
    Ok(())
}

#[test]
fn mutated_openai_chat_request_keeps_unknown_fields_without_replaying_stale_content() -> TestResult
{
    mutate_and_encode(
        WireFormat::OpenAiChat,
        json!({
            "model": "caller/chat",
            "messages": [{"role": "user", "content": "hello"}],
            "provider_extension": {"preserved": true}
        }),
    )
}

#[test]
fn mutated_openai_responses_request_keeps_unknown_fields_without_replaying_stale_content()
-> TestResult {
    mutate_and_encode(
        WireFormat::OpenAiResponses,
        json!({
            "model": "caller/responses",
            "input": "hello",
            "provider_extension": {"preserved": true}
        }),
    )
}

#[test]
fn mutated_anthropic_request_keeps_unknown_fields_without_replaying_stale_content() -> TestResult {
    mutate_and_encode(
        WireFormat::AnthropicMessages,
        json!({
            "model": "caller/anthropic",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hello"}],
            "provider_extension": {"preserved": true}
        }),
    )
}
