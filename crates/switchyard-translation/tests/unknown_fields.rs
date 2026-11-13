// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::{Value, json};
use switchyard_translation::{
    DiagnosticSeverity, PreservationPolicy, TranslationEngine, TranslationError, TranslationPolicy,
    UnknownFieldPolicy, WireFormat,
};

#[test]
fn buffered_decoders_apply_unknown_field_policy() {
    let engine = TranslationEngine::default();
    let fixtures = [
        (
            WireFormat::OpenAiChat,
            false,
            json!({
                "model": "qa/model", "messages": [{"role": "user", "content": "hello"}]
            }),
        ),
        (
            WireFormat::OpenAiChat,
            true,
            json!({
                "id": "reply", "model": "qa/model", "choices": [{"index": 0,
                    "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}]
            }),
        ),
        (
            WireFormat::OpenAiResponses,
            false,
            json!({
                "model": "qa/model", "input": [{"role": "user", "content": "hello"}]
            }),
        ),
        (
            WireFormat::OpenAiResponses,
            true,
            json!({
                "id": "reply", "model": "qa/model", "status": "completed", "output": [{
                    "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hello"}]}]
            }),
        ),
        (
            WireFormat::AnthropicMessages,
            false,
            json!({
                "model": "qa/model", "max_tokens": 32, "messages": [{"role": "user", "content": "hello"}]
            }),
        ),
        (
            WireFormat::AnthropicMessages,
            true,
            json!({
                "id": "reply", "model": "qa/model", "type": "message", "role": "assistant",
                "content": [{"type": "text", "text": "hello"}], "stop_reason": "end_turn"
            }),
        ),
    ];
    for (format, response, known) in fixtures {
        let decode = |body: &Value, policy: &TranslationPolicy| {
            if response {
                engine.decode_response(format, body, policy).map(|output| {
                    (
                        serde_json::to_value(output.response).unwrap(),
                        output.diagnostics,
                    )
                })
            } else {
                engine.decode_request(format, body, policy).map(|output| {
                    (
                        serde_json::to_value(output.request).unwrap(),
                        output.diagnostics,
                    )
                })
            }
        };
        let (mut expected_ir, _) = decode(&known, &TranslationPolicy::default()).unwrap();
        expected_ir.as_object_mut().unwrap().remove("preservation");
        for preservation in [PreservationPolicy::InMemory, PreservationPolicy::Disabled] {
            for unknown_field_policy in [
                UnknownFieldPolicy::Preserve,
                UnknownFieldPolicy::DropWithWarning,
                UnknownFieldPolicy::Reject,
            ] {
                for has_unknown in [false, true] {
                    let mut body = known.clone();
                    let extension = json!({"number": 17, "label": "unknown-field-control"});
                    if has_unknown {
                        body["qa_future"] = extension.clone();
                    }
                    let policy = TranslationPolicy {
                        preservation,
                        unknown_field_policy,
                        ..TranslationPolicy::default()
                    };
                    let result = decode(&body, &policy);
                    if has_unknown && unknown_field_policy == UnknownFieldPolicy::Reject {
                        assert!(
                            matches!(result, Err(TranslationError::UnknownField { path }) if path == "$.qa_future")
                        );
                        continue;
                    }
                    let (mut ir, diagnostics) = result.unwrap();
                    let fields = ir["extensions"]["fields"].as_object_mut().unwrap();
                    if has_unknown && unknown_field_policy == UnknownFieldPolicy::Preserve {
                        assert_eq!(fields.remove("qa_future"), Some(extension));
                    } else {
                        assert!(!fields.contains_key("qa_future"));
                    }
                    if has_unknown && unknown_field_policy == UnknownFieldPolicy::DropWithWarning {
                        assert_eq!(diagnostics.len(), 1);
                        assert_eq!(diagnostics[0].severity, DiagnosticSeverity::Warning);
                        assert_eq!(diagnostics[0].code, "unknown_field_dropped");
                        assert_eq!(diagnostics[0].path.as_deref(), Some("$.qa_future"));
                    } else {
                        assert!(diagnostics.is_empty());
                    }
                    ir.as_object_mut().unwrap().remove("preservation");
                    assert_eq!(
                        ir, expected_ir,
                        "{format:?}, response={response}, policy={policy:?}"
                    );
                }
            }
        }
    }
}
