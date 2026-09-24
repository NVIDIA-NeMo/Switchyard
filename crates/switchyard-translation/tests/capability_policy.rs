// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::json;
use switchyard_translation::util::validate_request_capabilities;
use switchyard_translation::{
    ContentBlock, DiagnosticSeverity, FileSource, ImageSource, LlmRequest, LossyConversionPolicy,
    MediaSource, Message, Role, TargetCapabilities, ToolCall, TranslationError, TranslationPolicy,
};

#[derive(Clone, Copy, Debug)]
enum CapabilityCase {
    Tools,
    Images,
    Audio,
    Video,
    Files,
    ReasoningEffort,
    StructuredOutput,
}

const CASES: [CapabilityCase; 7] = [
    CapabilityCase::Tools,
    CapabilityCase::Images,
    CapabilityCase::Audio,
    CapabilityCase::Video,
    CapabilityCase::Files,
    CapabilityCase::ReasoningEffort,
    CapabilityCase::StructuredOutput,
];

fn request_for(case: CapabilityCase) -> LlmRequest {
    let mut request = LlmRequest::default();
    let content = match case {
        CapabilityCase::Tools => Some(ContentBlock::ToolCall(ToolCall {
            id: "call_1".to_string(),
            name: "lookup".to_string(),
            arguments: json!({}),
        })),
        CapabilityCase::Images => Some(ContentBlock::Image {
            source: ImageSource::Url {
                url: "https://example.test/image.png".to_string(),
                detail: None,
            },
        }),
        CapabilityCase::Audio => Some(ContentBlock::Audio {
            source: MediaSource::Base64 {
                media_type: Some("audio/wav".to_string()),
                data: "UklGRg==".to_string(),
            },
        }),
        CapabilityCase::Video => Some(ContentBlock::Video {
            source: MediaSource::Url {
                url: "https://example.test/video.mp4".to_string(),
                media_type: Some("video/mp4".to_string()),
            },
        }),
        CapabilityCase::Files => Some(ContentBlock::File {
            source: FileSource::FileData {
                data: "aGVsbG8=".to_string(),
                filename: Some("notes.txt".to_string()),
            },
        }),
        CapabilityCase::ReasoningEffort => {
            request.reasoning.effort = Some("medium".to_string());
            None
        }
        CapabilityCase::StructuredOutput => {
            request.output.response_format = Some(json!({"type": "json_object"}));
            None
        }
    };
    if let Some(content) = content {
        request.messages.push(Message {
            role: Role::User,
            content: vec![content],
        });
    }
    request
}

fn policy_for(
    case: CapabilityCase,
    is_supported: bool,
    lossy_conversion_policy: LossyConversionPolicy,
) -> TranslationPolicy {
    let mut target_capabilities = TargetCapabilities::default();
    match case {
        CapabilityCase::Tools => target_capabilities.supports_tools = Some(is_supported),
        CapabilityCase::Images => target_capabilities.supports_images = Some(is_supported),
        CapabilityCase::Audio => target_capabilities.supports_audio = Some(is_supported),
        CapabilityCase::Video => target_capabilities.supports_video = Some(is_supported),
        CapabilityCase::Files => target_capabilities.supports_files = Some(is_supported),
        CapabilityCase::ReasoningEffort => {
            target_capabilities.supports_reasoning_effort = Some(is_supported);
        }
        CapabilityCase::StructuredOutput => {
            target_capabilities.supports_json_schema_response_format = Some(is_supported);
        }
    }
    TranslationPolicy {
        lossy_conversion_policy,
        target_capabilities,
        ..TranslationPolicy::default()
    }
}

fn expected_message(case: CapabilityCase) -> &'static str {
    match case {
        CapabilityCase::Tools => "target format/profile does not support tools",
        CapabilityCase::Images => "target format/profile does not support images",
        CapabilityCase::Audio => "target format/profile does not support audio",
        CapabilityCase::Video => "target format/profile does not support video",
        CapabilityCase::Files => "target format/profile does not support files",
        CapabilityCase::ReasoningEffort => {
            "target format/profile does not support reasoning effort"
        }
        CapabilityCase::StructuredOutput => {
            "target format/profile does not support structured response formats"
        }
    }
}

#[test]
fn unsupported_request_capabilities_allow() {
    for case in CASES {
        let request = request_for(case);
        let policy = policy_for(case, false, LossyConversionPolicy::AllowWithDiagnostics);
        let mut diagnostics = Vec::new();

        let result = validate_request_capabilities(&request, &mut diagnostics, &policy);

        assert!(result.is_ok(), "{case:?}: {result:?}");
        assert_eq!(diagnostics.len(), 1, "{case:?}");
        let diagnostic = &diagnostics[0];
        assert_eq!(diagnostic.severity, DiagnosticSeverity::Warning, "{case:?}");
        assert_eq!(diagnostic.code, "lossy_conversion", "{case:?}");
        assert_eq!(diagnostic.message, expected_message(case), "{case:?}");
        assert_eq!(diagnostic.source, None, "{case:?}");
        assert_eq!(diagnostic.target, None, "{case:?}");
        assert_eq!(diagnostic.path, None, "{case:?}");
    }
}

#[test]
fn unsupported_request_capabilities_reject() {
    for case in CASES {
        let request = request_for(case);
        let policy = policy_for(case, false, LossyConversionPolicy::Reject);
        let mut diagnostics = Vec::new();

        let error = validate_request_capabilities(&request, &mut diagnostics, &policy)
            .expect_err("unsupported capability should be rejected");

        match error {
            TranslationError::LossyConversion(message) => {
                assert_eq!(message, expected_message(case), "{case:?}");
            }
            other => panic!("{case:?}: expected LossyConversion, got {other:?}"),
        }
        assert!(diagnostics.is_empty(), "{case:?}: {diagnostics:?}");
    }
}

#[test]
fn supported_request_capabilities() {
    for lossy_conversion_policy in [
        LossyConversionPolicy::AllowWithDiagnostics,
        LossyConversionPolicy::Reject,
    ] {
        for case in CASES {
            let request = request_for(case);
            let policy = policy_for(case, true, lossy_conversion_policy);
            let mut diagnostics = Vec::new();

            let result = validate_request_capabilities(&request, &mut diagnostics, &policy);

            assert!(
                result.is_ok(),
                "{case:?}, {lossy_conversion_policy:?}: {result:?}"
            );
            assert!(
                diagnostics.is_empty(),
                "{case:?}, {lossy_conversion_policy:?}: {diagnostics:?}"
            );
        }
    }
}
