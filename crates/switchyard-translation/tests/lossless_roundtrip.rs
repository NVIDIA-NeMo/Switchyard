// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lossless round-trip tests for preservation metadata across all wire formats.

use pretty_assertions::assert_eq;
use serde_json::{json, Value};
use switchyard_translation::{
    PreservationPolicy, TranslationEngine, TranslationPolicy, WireFormat, PRESERVATION_METADATA_KEY,
};

const FORMATS: [WireFormat; 4] = [
    WireFormat::OpenAiChat,
    WireFormat::AnthropicMessages,
    WireFormat::OpenAiResponses,
    WireFormat::GeminiGenerateContent,
];

type TestResult = std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>;

// Verifies a concrete Responses -> Anthropic -> Chat -> Responses request cycle is exact.
#[test]
fn embedded_preservation_makes_responses_to_anthropic_to_chat_to_responses_exact() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = embed_policy();
    let original = request_fixture(WireFormat::OpenAiResponses);

    let anthropic = engine
        .translate_request(
            WireFormat::OpenAiResponses,
            WireFormat::AnthropicMessages,
            &original,
            &policy,
        )?
        .body;
    assert!(anthropic["metadata"]["_switchyard_translation"].is_object());

    let chat = engine
        .translate_request(
            WireFormat::AnthropicMessages,
            WireFormat::OpenAiChat,
            &anthropic,
            &policy,
        )?
        .body;
    assert!(chat["metadata"]["_switchyard_translation"].is_object());

    let roundtripped = engine
        .translate_request(
            WireFormat::OpenAiChat,
            WireFormat::OpenAiResponses,
            &chat,
            &policy,
        )?
        .body;

    assert_eq!(roundtripped, original);
    Ok(())
}

// Verifies every request source-target pair round-trips exactly with embedded preservation.
#[test]
fn embedded_preservation_roundtrips_requests_for_every_source_target_pair_exactly() {
    let engine = TranslationEngine::default();
    let policy = embed_policy();

    for source in FORMATS {
        let original = request_fixture(source);
        for target in FORMATS {
            let translated = engine
                .translate_request(source, target, &original, &policy)
                .unwrap_or_else(|error| {
                    panic!("request {source:?} -> {target:?} should translate: {error}")
                })
                .body;

            if source == target {
                assert_eq!(translated, original, "request {source:?} self-translation");
            } else {
                assert_embeds_original(&translated, "requests", source, &original);
            }

            let roundtripped = engine
                .translate_request(target, source, &translated, &policy)
                .unwrap_or_else(|error| {
                    panic!(
                        "request {source:?} -> {target:?} -> {source:?} should translate: {error}"
                    )
                })
                .body;

            assert_eq!(
                roundtripped, original,
                "request {source:?} -> {target:?} -> {source:?}"
            );
        }
    }
}

// Verifies every response source-target pair round-trips exactly with embedded preservation.
#[test]
fn embedded_preservation_roundtrips_responses_for_every_source_target_pair_exactly() {
    let engine = TranslationEngine::default();
    let policy = embed_policy();

    for source in FORMATS {
        let original = response_fixture(source);
        for target in FORMATS {
            let translated = engine
                .translate_response(source, target, &original, &policy)
                .unwrap_or_else(|error| {
                    panic!("response {source:?} -> {target:?} should translate: {error}")
                })
                .body;

            if source == target {
                assert_eq!(translated, original, "response {source:?} self-translation");
            } else {
                assert_embeds_original(&translated, "responses", source, &original);
            }

            let roundtripped = engine
                .translate_response(target, source, &translated, &policy)
                .unwrap_or_else(|error| {
                    panic!(
                        "response {source:?} -> {target:?} -> {source:?} should translate: {error}"
                    )
                })
                .body;

            assert_eq!(
                roundtripped, original,
                "response {source:?} -> {target:?} -> {source:?}"
            );
        }
    }
}

// Verifies request preservation survives every distinct three-format hop order.
#[test]
fn embedded_preservation_survives_every_distinct_three_format_request_cycle() {
    let engine = TranslationEngine::default();
    let policy = embed_policy();

    for source in FORMATS {
        let original = request_fixture(source);
        for (first, second) in distinct_hop_orders(source) {
            let first_body = engine
                .translate_request(source, first, &original, &policy)
                .unwrap_or_else(|error| {
                    panic!("request {source:?} -> {first:?} should translate: {error}")
                })
                .body;
            let second_body = engine
                .translate_request(first, second, &first_body, &policy)
                .unwrap_or_else(|error| {
                    panic!(
                        "request {source:?} -> {first:?} -> {second:?} should translate: {error}"
                    )
                })
                .body;
            assert_embeds_original(&second_body, "requests", source, &original);
            assert_embeds_original(&second_body, "requests", first, &first_body);

            let roundtripped = engine
                .translate_request(second, source, &second_body, &policy)
                .unwrap_or_else(|error| {
                    panic!(
                        "request {source:?} -> {first:?} -> {second:?} -> {source:?} should translate: {error}"
                    )
                })
                .body;

            assert_eq!(
                roundtripped, original,
                "request {source:?} -> {first:?} -> {second:?} -> {source:?}"
            );
        }
    }
}

// Verifies response preservation survives every distinct three-format hop order.
#[test]
fn embedded_preservation_survives_every_distinct_three_format_response_cycle() {
    let engine = TranslationEngine::default();
    let policy = embed_policy();

    for source in FORMATS {
        let original = response_fixture(source);
        for (first, second) in distinct_hop_orders(source) {
            let first_body = engine
                .translate_response(source, first, &original, &policy)
                .unwrap_or_else(|error| {
                    panic!("response {source:?} -> {first:?} should translate: {error}")
                })
                .body;
            let second_body = engine
                .translate_response(first, second, &first_body, &policy)
                .unwrap_or_else(|error| {
                    panic!(
                        "response {source:?} -> {first:?} -> {second:?} should translate: {error}"
                    )
                })
                .body;
            assert_embeds_original(&second_body, "responses", source, &original);
            assert_embeds_original(&second_body, "responses", first, &first_body);

            let roundtripped = engine
                .translate_response(second, source, &second_body, &policy)
                .unwrap_or_else(|error| {
                    panic!(
                        "response {source:?} -> {first:?} -> {second:?} -> {source:?} should translate: {error}"
                    )
                })
                .body;

            assert_eq!(
                roundtripped, original,
                "response {source:?} -> {first:?} -> {second:?} -> {source:?}"
            );
        }
    }
}

// Verifies in-memory preservation can replay the original body without metadata embedding.
#[test]
fn in_memory_preservation_replays_exact_original_when_encoding_from_the_same_ir() -> TestResult {
    let engine = TranslationEngine::default();
    let policy = TranslationPolicy::default();
    let original = json!({
        "model": "gpt-4o",
        "input": "Hello",
        "metadata": {"trace": "keep-me"},
        "store": false
    });

    let decoded = engine.decode_request(WireFormat::OpenAiResponses, &original, &policy)?;
    let encoded = engine.encode_request(WireFormat::OpenAiResponses, &decoded.request, &policy)?;

    assert_eq!(encoded.body, original);
    Ok(())
}

// Builds the policy used by exact wire round-trip tests.
fn embed_policy() -> TranslationPolicy {
    TranslationPolicy {
        preservation: PreservationPolicy::Embed,
        ..TranslationPolicy::default()
    }
}

// Returns both possible two-hop orders through the non-source formats.
fn distinct_hop_orders(source: WireFormat) -> [(WireFormat, WireFormat); 2] {
    let others = FORMATS
        .iter()
        .copied()
        .filter(|format| *format != source)
        .collect::<Vec<_>>();
    [(others[0], others[1]), (others[1], others[0])]
}

// Asserts a translated body carries the exact original body in preservation metadata.
fn assert_embeds_original(body: &Value, group: &str, source: WireFormat, original: &Value) {
    let embedded = body
        .get("metadata")
        .and_then(Value::as_object)
        .and_then(|metadata| metadata.get(PRESERVATION_METADATA_KEY))
        .unwrap_or_else(|| {
            panic!("translated {group} body should contain {PRESERVATION_METADATA_KEY}: {body}")
        });
    assert_eq!(
        embedded
            .get(group)
            .and_then(Value::as_object)
            .and_then(|requests| requests.get(format_key(source))),
        Some(original),
        "embedded preservation should contain exact {source:?} {group} body"
    );
}

// Returns the JSON metadata key for a built-in wire format.
fn format_key(format: WireFormat) -> &'static str {
    match format {
        WireFormat::OpenAiChat => "openai_chat",
        WireFormat::AnthropicMessages => "anthropic_messages",
        WireFormat::OpenAiResponses => "openai_responses",
        WireFormat::GeminiGenerateContent => "gemini_generate_content",
    }
}

// Builds an intentionally broad request fixture for a provider format.
fn request_fixture(format: WireFormat) -> Value {
    match format {
        WireFormat::OpenAiChat => json!({
            "model": "gpt-5.2",
            "messages": [
                {"role": "system", "content": "Follow exact instructions."},
                {"role": "developer", "content": "Preserve developer constraints."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Inspect this payload."},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.test/image.png", "detail": "high"}
                        },
                        {
                            "type": "file",
                            "file": {"file_id": "file_123"}
                        },
                        {
                            "type": "vendor_block",
                            "nested": {"array": [1, true, null]},
                            "text": "unknown user content"
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {
                            "id": "call_lookup",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": "{\"query\":\"rust\",\"limit\":2}"
                            }
                        },
                        {
                            "id": "call_raw",
                            "type": "function",
                            "function": {
                                "name": "raw_args",
                                "arguments": "not-json"
                            }
                        }
                    ]
                },
                {"role": "tool", "tool_call_id": "call_lookup", "content": "{\"ok\":true}"},
                {"role": "user", "content": "Finish."}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup data",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {"type": "integer"}
                        },
                        "required": ["query"]
                    },
                    "strict": true
                }
            }],
            "tool_choice": {"type": "function", "function": {"name": "lookup"}},
            "max_tokens": 777,
            "max_completion_tokens": 888,
            "temperature": 0.2,
            "top_p": 0.91,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {"type": "object"}
                }
            },
            "reasoning_effort": "high",
            "stream": true,
            "metadata": {"trace": "chat-request", "kept": {"nested": true}},
            "seed": 1234,
            "user": "nachiketb"
        }),
        WireFormat::AnthropicMessages => json!({
            "model": "claude-opus-4-7",
            "system": [
                {"type": "text", "text": "Follow exact instructions."},
                {"type": "text", "text": "Preserve Anthropic system arrays."}
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Inspect this payload."},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "iVBORw0KGgo="
                            }
                        },
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "data": "ZmlsZQ==",
                                "filename": "notes.txt"
                            }
                        },
                        {"type": "vendor_block", "payload": {"keep": ["this"]}}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Calling the tool."},
                        {
                            "type": "tool_use",
                            "id": "toolu_lookup",
                            "name": "lookup",
                            "input": {"query": "rust", "limit": 2}
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": "toolu_lookup",
                        "content": [
                            {"type": "text", "text": "found"},
                            {"type": "json", "value": {"ok": true}}
                        ],
                        "is_error": false
                    }]
                }
            ],
            "tools": [{
                "name": "lookup",
                "description": "Lookup data",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }],
            "tool_choice": {"type": "tool", "name": "lookup"},
            "max_tokens": 888,
            "temperature": 0.2,
            "top_p": 0.91,
            "top_k": 40,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "output_config": {"effort": "high"},
            "stream": true,
            "metadata": {"trace": "anthropic-request", "kept": {"nested": true}},
            "container": "claude-container"
        }),
        WireFormat::OpenAiResponses => json!({
            "model": "gpt-5.2",
            "instructions": "Follow exact instructions.",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": "Preserve developer input roles."
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Inspect this payload."},
                        {
                            "type": "input_image",
                            "image_url": {
                                "url": "https://example.test/image.png",
                                "detail": "high"
                            }
                        },
                        {
                            "type": "input_file",
                            "file": {"file_id": "file_123"}
                        },
                        {
                            "type": "vendor_block",
                            "payload": {"keep": ["this"]},
                            "text": "unknown responses content"
                        }
                    ]
                },
                {
                    "type": "function_call",
                    "call_id": "call_lookup",
                    "name": "lookup",
                    "arguments": {"query": "rust", "limit": 2}
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_lookup",
                    "output": {"ok": true, "items": [1, 2]}
                },
                {
                    "type": "local_shell_call",
                    "call_id": "call_shell",
                    "action": {"command": "pwd"}
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "Lookup data",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"]
                    },
                    "strict": true
                },
                {
                    "id": "exec_command",
                    "description": "Runs a command.",
                    "inputSchema": {
                        "jsonSchema": {
                            "type": "object",
                            "properties": {"cmd": {"type": "string"}}
                        }
                    }
                }
            ],
            "tool_choice": {"type": "function", "name": "lookup"},
            "max_output_tokens": 888,
            "reasoning": {"effort": "high", "summary": "auto"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "schema": {"type": "object"}
                }
            },
            "temperature": 0.2,
            "top_p": 0.91,
            "stream": true,
            "metadata": {"trace": "responses-request", "kept": {"nested": true}},
            "parallel_tool_calls": false,
            "store": false,
            "truncation": "auto"
        }),
        WireFormat::GeminiGenerateContent => json!({
            "model": "gemini-2.5-flash",
            "stream": true,
            "systemInstruction": {
                "parts": [{"text": "Follow exact instructions."}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": "Inspect this payload."},
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": "iVBORw0KGgo="
                            }
                        },
                        {
                            "fileData": {
                                "mimeType": "video/mp4",
                                "fileUri": "https://example.test/clip.mp4"
                            }
                        }
                    ]
                },
                {
                    "role": "model",
                    "parts": [
                        {"text": "Planning the lookup.", "thought": true},
                        {
                            "functionCall": {
                                "name": "lookup",
                                "args": {"query": "rust", "limit": 2}
                            },
                            "thoughtSignature": "c2lnbmF0dXJl"
                        }
                    ]
                },
                {
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            "name": "lookup",
                            "response": {"parts": [{"text": "found"}]}
                        }
                    }]
                }
            ],
            "tools": [{
                "functionDeclarations": [{
                    "name": "lookup",
                    "description": "Lookup data",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {"query": {"type": "STRING"}},
                        "required": ["query"]
                    }
                }]
            }],
            "toolConfig": {
                "functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["lookup"]}
            },
            "generationConfig": {
                "maxOutputTokens": 888,
                "temperature": 0.2,
                "topP": 0.91,
                "topK": 40,
                "stopSequences": ["END"],
                "responseMimeType": "application/json",
                "responseSchema": {"type": "OBJECT"},
                "thinkingConfig": {"thinkingBudget": 1024, "includeThoughts": true}
            },
            "safetySettings": [{
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            }],
            "cachedContent": "cachedContents/example",
            "labels": {"trace": "gemini-request"}
        }),
    }
}

// Builds an intentionally broad response fixture for a provider format.
fn response_fixture(format: WireFormat) -> Value {
    match format {
        WireFormat::OpenAiChat => json!({
            "id": "chatcmpl_adversarial",
            "object": "chat.completion",
            "created": 1780000000,
            "model": "gpt-5.2",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Here is the answer."},
                            {
                                "type": "refusal",
                                "refusal": "Cannot reveal hidden chain."
                            },
                            {
                                "type": "vendor_block",
                                "payload": {"preserve": true}
                            }
                        ],
                        "tool_calls": [{
                            "id": "call_lookup",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": "{\"query\":\"rust\"}"
                            }
                        }]
                    },
                    "finish_reason": "tool_calls",
                    "logprobs": {"content": []}
                },
                {
                    "index": 1,
                    "message": {"role": "assistant", "content": "alternate"},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "completion_tokens_details": {"reasoning_tokens": 2}
            },
            "metadata": {"trace": "chat-response", "kept": {"nested": true}},
            "system_fingerprint": "fp_test"
        }),
        WireFormat::AnthropicMessages => json!({
            "id": "msg_adversarial",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-4-7",
            "content": [
                {"type": "text", "text": "Here is the answer."},
                {
                    "type": "tool_use",
                    "id": "toolu_lookup",
                    "name": "lookup",
                    "input": {"query": "rust"}
                },
                {
                    "type": "server_tool_use",
                    "id": "srv_1",
                    "name": "web_search",
                    "input": {"query": "rust"}
                }
            ],
            "stop_reason": "tool_use",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_input_tokens": 3,
                "cache_read_input_tokens": 4
            },
            "metadata": {"trace": "anthropic-response", "kept": {"nested": true}},
            "container": {"id": "container_123"}
        }),
        WireFormat::OpenAiResponses => json!({
            "id": "resp_adversarial",
            "object": "response",
            "created_at": 1780000000,
            "model": "gpt-5.2",
            "status": "completed",
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Here is the answer.",
                            "annotations": [
                                {"type": "url_citation", "url": "https://example.test"}
                            ]
                        },
                        {
                            "type": "refusal",
                            "refusal": "Cannot reveal hidden chain."
                        },
                        {
                            "type": "vendor_output",
                            "payload": {"preserve": true}
                        }
                    ]
                },
                {
                    "type": "function_call",
                    "call_id": "call_lookup",
                    "name": "lookup",
                    "arguments": {"query": "rust"}
                },
                {
                    "type": "web_search_call",
                    "id": "ws_1",
                    "status": "completed"
                }
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "output_tokens_details": {"reasoning_tokens": 2}
            },
            "parallel_tool_calls": false,
            "tool_choice": "auto",
            "tools": [{"type": "web_search_preview"}],
            "metadata": {"trace": "responses-response", "kept": {"nested": true}},
            "service_tier": "default"
        }),
        WireFormat::GeminiGenerateContent => json!({
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "Planning the lookup.", "thought": true},
                        {"text": "Here is the answer."},
                        {
                            "functionCall": {
                                "name": "lookup",
                                "args": {"query": "rust"}
                            },
                            "thoughtSignature": "c2lnbmF0dXJl"
                        }
                    ],
                    "role": "model"
                },
                "finishReason": "STOP",
                "index": 0,
                "finishMessage": "Model generated function call(s)."
            }],
            "promptFeedback": {"safetyRatings": []},
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "thoughtsTokenCount": 2,
                "totalTokenCount": 17,
                "cachedContentTokenCount": 4,
                "serviceTier": "standard"
            },
            "modelVersion": "gemini-2.5-flash",
            "responseId": "resp_adversarial"
        }),
    }
}
