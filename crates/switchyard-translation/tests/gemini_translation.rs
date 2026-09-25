// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::json;
use switchyard_translation::codecs::gemini::GEMINI_GENERATE_CONTENT as GEMINI;
use switchyard_translation::{
    ContentBlock, LossyConversionPolicy, PreservationPolicy, StopReason, TranslationEngine,
    TranslationPolicy, WireFormat,
};

fn reconstruct() -> TranslationPolicy {
    TranslationPolicy {
        preservation: PreservationPolicy::Disabled,
        ..Default::default()
    }
}

#[test]
fn native_request_preserves_all_fields_and_translates_tools() {
    let engine = TranslationEngine::default();
    let body = json!({
        "systemInstruction":{"parts":[{"text":"Be precise"}]},
        "contents":[
            {"role":"user","parts":[{"text":"Weather?"}]},
            {"role":"model","parts":[{"functionCall":{"name":"weather","args":{"city":"Paris"}}}]},
            {"role":"user","parts":[{"functionResponse":{"name":"weather","response":{"temperature":21}}}]}
        ],
        "tools":[{"functionDeclarations":[{"name":"weather","description":"Weather lookup","parameters":{"type":"OBJECT","properties":{"city":{"type":"STRING"}}}}]}],
        "toolConfig":{"functionCallingConfig":{"mode":"ANY","allowedFunctionNames":["weather"]}},
        "generationConfig":{"temperature":0.2,"topP":0.9,"topK":10,"maxOutputTokens":100},
        "safetySettings":[{"category":"HARM_CATEGORY_DANGEROUS_CONTENT","threshold":"BLOCK_ONLY_HIGH"}]
    });
    assert_eq!(
        engine
            .translate_request(GEMINI, GEMINI, &body, &TranslationPolicy::default())
            .unwrap()
            .body,
        body
    );
    let output = engine
        .translate_request(GEMINI, WireFormat::OpenAiChat, &body, &reconstruct())
        .unwrap();
    assert_eq!(output.body["messages"][0]["content"], "Be precise");
    assert_eq!(
        output.body["messages"][2]["tool_calls"][0]["function"]["name"],
        "weather"
    );
    assert_eq!(output.body["messages"][3]["role"], "tool");
    assert_eq!(
        output.body["messages"][3]["tool_call_id"],
        output.body["messages"][2]["tool_calls"][0]["id"]
    );
    assert_eq!(
        output.body["tools"][0]["function"]["parameters"]["properties"]["city"]["type"],
        "string"
    );
    assert!(!output.diagnostics.is_empty());
}

#[test]
fn openai_request_encodes_native_function_results_and_json_schema() {
    let engine = TranslationEngine::default();
    let body = json!({"model":"routed-model","messages":[
        {"role":"system","content":"Instructions"},
        {"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},
        {"role":"tool","tool_call_id":"call_1","content":"{\"ok\":true}"}],
        "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],
        "tool_choice":"required","max_completion_tokens":42,
        "response_format":{"type":"json_schema","json_schema":{"name":"answer","schema":{"type":"object"}}}
    });
    let output = engine
        .translate_request(WireFormat::OpenAiChat, GEMINI, &body, &reconstruct())
        .unwrap();
    assert!(output.body.get("model").is_none());
    assert_eq!(
        output.body["contents"][1]["parts"][0]["functionResponse"],
        json!({"id":"call_1","name":"lookup","response":{"ok":true}})
    );
    assert_eq!(
        output.body["generationConfig"]["responseJsonSchema"],
        json!({"type":"object"})
    );
    assert_eq!(
        output.body["toolConfig"]["functionCallingConfig"]["mode"],
        "ANY"
    );
    let back = engine
        .translate_request(GEMINI, WireFormat::OpenAiChat, &output.body, &reconstruct())
        .unwrap();
    assert_eq!(back.body["messages"][2]["tool_call_id"], "call_1");
}

#[test]
fn media_and_thoughts_reach_neutral_ir_without_visible_reasoning() {
    let engine = TranslationEngine::default();
    let body = json!({"contents":[{"role":"user","parts":[
        {"text":"Describe"},{"inlineData":{"mimeType":"image/png","data":"AA=="}},
        {"inlineData":{"mimeType":"audio/wav","data":"AA=="}},
        {"fileData":{"mimeType":"video/mp4","fileUri":"gs://bucket/video"}}]}]});
    let ir = engine
        .decode_request(GEMINI, &body, &reconstruct())
        .unwrap()
        .request;
    assert!(matches!(
        ir.messages[0].content[1],
        ContentBlock::Image { .. }
    ));
    assert!(matches!(
        ir.messages[0].content[2],
        ContentBlock::Audio { .. }
    ));
    assert!(matches!(
        ir.messages[0].content[3],
        ContentBlock::Video { .. }
    ));
    assert_eq!(
        engine
            .encode_request(GEMINI, &ir, &reconstruct())
            .unwrap()
            .body,
        body
    );
    let response = json!({"candidates":[{"content":{"role":"model","parts":[{"text":"Private","thought":true,"thoughtSignature":"opaque"},{"text":"Public"}]},"finishReason":"STOP"}]});
    let ir = engine
        .decode_response(GEMINI, &response, &reconstruct())
        .unwrap()
        .response;
    assert!(matches!(
        ir.outputs[0].content[0],
        ContentBlock::Reasoning { .. }
    ));
    let ContentBlock::Reasoning { signature, .. } = &ir.outputs[0].content[0] else {
        panic!("missing reasoning")
    };
    assert!(
        signature.is_none(),
        "native signatures must not cross providers"
    );
    let chat = engine
        .encode_response(WireFormat::OpenAiChat, &ir, &reconstruct())
        .unwrap();
    assert_eq!(chat.body["choices"][0]["message"]["content"], "Public");
    assert_eq!(
        engine
            .translate_response(GEMINI, GEMINI, &response, &TranslationPolicy::default())
            .unwrap()
            .body,
        response
    );
}

#[test]
fn response_usage_normalizes_cache_and_keeps_reasoning_separate() {
    let engine = TranslationEngine::default();
    let body = json!({"responseId":"r1","modelVersion":"gemini-test","candidates":[
        {"index":0,"content":{"role":"model","parts":[{"text":"done"}]},"finishReason":"MAX_TOKENS"},
        {"index":1,"content":{"role":"model","parts":[]},"finishReason":"SAFETY"}],
        "usageMetadata":{"promptTokenCount":100,"cachedContentTokenCount":70,"candidatesTokenCount":20,"thoughtsTokenCount":10,"totalTokenCount":130}});
    let ir = engine
        .decode_response(GEMINI, &body, &reconstruct())
        .unwrap()
        .response;
    assert_eq!(ir.usage.input_tokens, Some(30));
    assert_eq!(ir.usage.output_tokens, Some(20));
    assert_eq!(ir.usage.reasoning_tokens, Some(10));
    assert_eq!(ir.outputs[1].stop_reason, Some(StopReason::ContentFilter));
    assert_eq!(
        engine
            .encode_response(GEMINI, &ir, &reconstruct())
            .unwrap()
            .body,
        body
    );
}

#[test]
fn strict_policy_rejects_unrepresentable_controls_and_malformed_payloads() {
    let engine = TranslationEngine::default();
    let strict = TranslationPolicy {
        lossy_conversion_policy: LossyConversionPolicy::Reject,
        ..reconstruct()
    };
    assert!(
        engine
            .decode_request(
                GEMINI,
                &json!({"contents":[],"tools":[{"googleSearch":{}}]}),
                &strict
            )
            .is_err()
    );
    assert!(
        engine
            .decode_request(
                GEMINI,
                &json!({"contents":[],"generationConfig":{"candidateCount":2}}),
                &strict
            )
            .is_err()
    );
    assert!(
        engine
            .decode_request(
                GEMINI,
                &json!({"contents":[{"role":"developer","parts":[]}]}),
                &strict
            )
            .is_err()
    );
    assert!(
        engine
            .decode_request(GEMINI, &json!({"contents":"invalid"}), &strict)
            .is_err()
    );
    assert!(engine.decode_request(GEMINI,&json!({"contents":[{"parts":[{"functionResponse":{"name":"missing","response":{}}}]}]}),&strict).is_err());
    assert!(
        engine
            .decode_response(GEMINI, &json!({"error":{"code":403}}), &strict)
            .is_err()
    );
    let ir = engine
        .decode_response(
            GEMINI,
            &json!({"promptFeedback":{"blockReason":"SAFETY"}}),
            &strict,
        )
        .unwrap()
        .response;
    assert_eq!(ir.outputs[0].stop_reason, Some(StopReason::ContentFilter));
}

#[test]
fn unknown_and_lossy_policies_cover_nested_fields_and_native_constraints() {
    use switchyard_translation::{InstructionBlock, LlmRequest, Role, UnknownFieldPolicy};
    let engine = TranslationEngine::default();
    let unknown = TranslationPolicy {
        unknown_field_policy: UnknownFieldPolicy::Reject,
        ..reconstruct()
    };
    for body in [
        json!({"contents":[],"futureControl":true}),
        json!({"contents":[{"parts":[{"text":"hi","futureControl":true}]}]}),
        json!({"contents":[{"parts":[{"functionCall":{"name":"f","args":{},"futureControl":true}}]}]}),
    ] {
        assert!(engine.decode_request(GEMINI, &body, &unknown).is_err());
    }
    assert!(engine.decode_request(GEMINI,&json!({"contents":[{"parts":[{"text":"hi","functionCall":{"name":"f","args":{}}}]}]}),&reconstruct()).is_err());
    let strict = TranslationPolicy {
        lossy_conversion_policy: LossyConversionPolicy::Reject,
        ..reconstruct()
    };
    let request = LlmRequest {
        stream: true,
        ..Default::default()
    };
    assert!(engine.encode_request(GEMINI, &request, &strict).is_err());
    let mut request = LlmRequest::default();
    request
        .extensions
        .fields
        .insert("logit_bias".into(), json!({"1":2}));
    assert!(engine.encode_request(GEMINI, &request, &strict).is_err());
    let request = LlmRequest {
        instructions: vec![InstructionBlock {
            role: Role::System,
            content: vec![ContentBlock::Reasoning {
                text: "hidden".into(),
                signature: None,
                details: vec![],
            }],
        }],
        ..Default::default()
    };
    assert!(
        engine
            .encode_request(GEMINI, &request, &reconstruct())
            .is_err()
    );
}

#[test]
fn generated_ids_avoid_explicit_ids_and_duplicate_ids_fail() {
    let engine = TranslationEngine::default();
    let body = json!({"contents":[{"role":"model","parts":[
        {"functionCall":{"name":"first","args":{}}},
        {"functionCall":{"id":"sw_gemini_1","name":"second","args":{}}}]},
        {"role":"user","parts":[{"functionResponse":{"name":"first","response":{}}},{"functionResponse":{"name":"second","response":{}}}]}]});
    let ir = engine
        .decode_request(GEMINI, &body, &reconstruct())
        .unwrap()
        .request;
    let ContentBlock::ToolCall(first) = &ir.messages[0].content[0] else {
        panic!("missing call")
    };
    let ContentBlock::ToolCall(second) = &ir.messages[0].content[1] else {
        panic!("missing call")
    };
    assert_ne!(first.id, second.id);
    let ContentBlock::ToolResult(result) = &ir.messages[1].content[0] else {
        panic!("missing result")
    };
    assert_eq!(result.tool_call_id, first.id);
    let mut duplicate = body;
    duplicate["contents"][0]["parts"][0]["functionCall"]["id"] = json!("sw_gemini_1");
    assert!(
        engine
            .decode_request(GEMINI, &duplicate, &reconstruct())
            .is_err()
    );
}

#[test]
fn native_nullable_schema_keeps_null_as_a_valid_value() {
    let engine = TranslationEngine::default();
    let body = json!({"contents":[],"tools":[{"functionDeclarations":[{"name":"f","parameters":{"type":"OBJECT","properties":{"type":{"type":"STRING","nullable":true,"enum":["x"]}},"propertyOrdering":["type"]}}]}]});
    let ir = engine
        .decode_request(GEMINI, &body, &reconstruct())
        .unwrap()
        .request;
    assert_eq!(
        ir.tools[0].parameters["properties"]["type"],
        json!({"anyOf":[{"type":"string","enum":["x"]},{"type":"null"}]})
    );
}

#[test]
fn anthropic_tool_history_and_openai_response_cross_the_public_engine() {
    let engine = TranslationEngine::default();
    let body = json!({"model":"claude-test","max_tokens":50,"messages":[
        {"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"lookup","input":{"key":"value"}}]},
        {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"found"}]}]});
    let native = engine
        .translate_request(WireFormat::AnthropicMessages, GEMINI, &body, &reconstruct())
        .unwrap();
    assert_eq!(
        native.body["contents"][1]["parts"][0]["functionResponse"]["name"],
        "lookup"
    );
    let restored = engine
        .translate_request(
            GEMINI,
            WireFormat::AnthropicMessages,
            &native.body,
            &reconstruct(),
        )
        .unwrap();
    assert_eq!(restored.body["messages"][0]["content"][0]["id"], "toolu_1");
    assert_eq!(
        restored.body["messages"][1]["content"][0]["tool_use_id"],
        "toolu_1"
    );
    let response = json!({"id":"chat1","model":"test","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}});
    let native = engine
        .translate_response(WireFormat::OpenAiChat, GEMINI, &response, &reconstruct())
        .unwrap();
    assert_eq!(native.body["candidates"][0]["finishReason"], "STOP");
    assert_eq!(
        native.body["candidates"][0]["content"]["parts"][0]["functionCall"]["id"],
        "call1"
    );
    let restored = engine
        .translate_response(GEMINI, WireFormat::OpenAiChat, &native.body, &reconstruct())
        .unwrap();
    assert_eq!(restored.body["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(restored.body["usage"]["total_tokens"], 15);
}

#[test]
fn mismatched_tool_result_and_invalid_generation_numbers_fail() {
    let engine = TranslationEngine::default();
    let body = json!({"contents":[{"role":"model","parts":[{"functionCall":{"name":"first","id":"id1","args":{}}}]},{"role":"user","parts":[{"functionResponse":{"name":"second","id":"id1","response":{}}}]}]});
    assert!(
        engine
            .decode_request(GEMINI, &body, &reconstruct())
            .is_err()
    );
    for config in [
        json!({"temperature":"hot"}),
        json!({"maxOutputTokens":-1}),
        json!({"topK":1.5}),
    ] {
        assert!(
            engine
                .decode_request(
                    GEMINI,
                    &json!({"contents":[],"generationConfig":config}),
                    &reconstruct()
                )
                .is_err()
        );
    }
}

#[test]
fn nested_tool_configuration_and_enum_output_are_not_silently_dropped() {
    let engine = TranslationEngine::default();
    let strict = TranslationPolicy {
        lossy_conversion_policy: LossyConversionPolicy::Reject,
        ..reconstruct()
    };
    for extra in [
        json!({"toolConfig":{"functionCallingConfig":{"mode":"AUTO","streamFunctionCallArguments":true}}}),
        json!({"toolConfig":{"retrievalConfig":{}}}),
        json!({"generationConfig":{"responseMimeType":"text/x.enum","responseSchema":{"type":"STRING","enum":["yes","no"]}}}),
    ] {
        let mut body = extra;
        body["contents"] = json!([]);
        assert!(engine.decode_request(GEMINI, &body, &strict).is_err());
        assert!(
            !engine
                .decode_request(GEMINI, &body, &reconstruct())
                .unwrap()
                .diagnostics
                .is_empty()
        );
    }
    let body = json!({"model":"test","max_tokens":10,"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"toolu1","name":"f","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu1","content":"{\"message\":\"failed\"}","is_error":true}]}]});
    let native = engine
        .translate_request(WireFormat::AnthropicMessages, GEMINI, &body, &reconstruct())
        .unwrap();
    assert_eq!(
        native.body["contents"][1]["parts"][0]["functionResponse"]["response"],
        json!({"error":{"message":"failed"}})
    );
}

#[test]
fn nontext_thought_is_not_projected_as_visible_media() {
    let engine = TranslationEngine::default();
    let body = json!({"candidates":[{"content":{"parts":[{"thought":true,"inlineData":{"mimeType":"image/png","data":"AA=="}}]},"finishReason":"STOP"}]});
    let ir = engine
        .decode_response(GEMINI, &body, &reconstruct())
        .unwrap();
    assert!(matches!(
        ir.response.outputs[0].content[0],
        ContentBlock::Unknown { .. }
    ));
    assert!(!ir.diagnostics.is_empty());
}

#[test]
fn lossy_request_omits_messages_with_no_representable_parts() {
    let engine = TranslationEngine::default();
    let body = json!({"messages":[
        {"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/image.png"}}]},
        {"role":"user","content":"Keep this text"}
    ]});
    let output = engine
        .translate_request(WireFormat::OpenAiChat, GEMINI, &body, &reconstruct())
        .unwrap();
    assert_eq!(
        output.body["contents"],
        json!([{"role":"user","parts":[{"text":"Keep this text"}]}])
    );
    assert!(!output.diagnostics.is_empty());
    let all_dropped = json!({"messages":[body["messages"][0].clone()]});
    let error = engine
        .translate_request(WireFormat::OpenAiChat, GEMINI, &all_dropped, &reconstruct())
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("no representable conversation content")
    );
    let strict = TranslationPolicy {
        lossy_conversion_policy: LossyConversionPolicy::Reject,
        ..reconstruct()
    };
    assert!(
        engine
            .translate_request(WireFormat::OpenAiChat, GEMINI, &body, &strict)
            .is_err()
    );
}
