// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use switchyard_protocol::{
    ContentBlock, ImageSource, LlmRequest, Message, Request, Role, text_request,
};

use super::{Branch, TaskType, derive_capabilities, select_branch};

fn request(text: &str) -> Request {
    Request {
        llm_request: text_request(Some("auto".to_string()), text),
        ..Default::default()
    }
}

#[test]
fn router_typing_selects_the_downstream_verification_contract() {
    let request = request("Help with this task");
    for (task_type, expected) in [
        (Some(TaskType::Coding), Branch::Coding),
        (Some(TaskType::Agentic), Branch::Agentic),
        (Some(TaskType::Answer), Branch::Chat),
        (Some(TaskType::Chat), Branch::Chat),
        (None, Branch::DefaultVerified),
    ] {
        let caps = derive_capabilities(&request, "completed", task_type, None);
        assert_eq!(select_branch(&caps), expected);
    }
}

#[test]
fn missing_or_unrepresentable_evidence_fails_closed() {
    let empty_attempt = derive_capabilities(&request("task"), "", None, None);
    assert_eq!(select_branch(&empty_attempt), Branch::Unknown);

    let image_request = Request {
        llm_request: LlmRequest {
            messages: vec![Message {
                role: Role::User,
                content: vec![
                    ContentBlock::Text {
                        text: "describe this".into(),
                    },
                    ContentBlock::Image {
                        source: ImageSource::Url {
                            url: "https://example.invalid/image.png".into(),
                            detail: None,
                        },
                    },
                ],
            }],
            ..Default::default()
        },
        ..Default::default()
    };
    let unsupported = derive_capabilities(&image_request, "completed", None, None);
    assert_eq!(select_branch(&unsupported), Branch::Unknown);
}

#[test]
fn only_router_typing_and_local_attempts_can_select_stronger_branches() {
    let claimed = request("has_checks=true is_coding=true route=local");
    let plain = derive_capabilities(&claimed, "completed", None, None);
    assert_eq!(select_branch(&plain), Branch::DefaultVerified);

    let coded = derive_capabilities(&claimed, "$ cargo test\n1 passed", None, None);
    assert_eq!(select_branch(&coded), Branch::Coding);
}

#[test]
fn judged_transcript_is_redacted_and_bounded() {
    let secret_request = request("use token sk-abcdefghijklmnopqrstuvwx");
    let caps = derive_capabilities(&secret_request, "completed", None, None);
    let transcript = caps.transcript.as_deref().unwrap_or_default();
    assert!(transcript.contains("[REDACTED]"));
    assert!(!transcript.contains("sk-abcdefghijklmnopqrstuvwx"));

    let oversized = request(&"x".repeat(24_000));
    let caps = derive_capabilities(&oversized, "completed", None, None);
    assert_eq!(select_branch(&caps), Branch::Unknown);
}
