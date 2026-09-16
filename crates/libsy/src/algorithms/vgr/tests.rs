// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use switchyard_protocol::{
    ContentBlock, ImageSource, LlmRequest, Message, Request, Role, text_request,
};

use super::decide::{Route, Signals, Tri, decide_from_signals};
use super::{Branch, TaskType, ToolErrorCount, derive_capabilities, select_branch};

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

#[test]
fn policy_exposes_only_verified_local_commits() {
    let request = request("complete the task");
    let cases = [
        (
            None,
            None,
            Signals::default(),
            Branch::DefaultVerified,
            Route::Cloud,
        ),
        (
            None,
            None,
            Signals {
                readout: Some(0.7),
                ..Default::default()
            },
            Branch::DefaultVerified,
            Route::Local,
        ),
        (
            Some(TaskType::Coding),
            None,
            Signals {
                readout: Some(0.25),
                cloud_judge: Some(Tri::Yes),
                ..Default::default()
            },
            Branch::Coding,
            Route::Local,
        ),
        (
            Some(TaskType::Coding),
            None,
            Signals {
                readout: Some(0.95),
                cloud_judge: Some(Tri::Unknown),
                ..Default::default()
            },
            Branch::Coding,
            Route::Cloud,
        ),
        (
            Some(TaskType::Coding),
            None,
            Signals {
                readout: Some(0.95),
                deliberation: Some(1.0),
                strict_evidence: Some(Tri::Yes),
                cloud_judge: Some(Tri::No),
            },
            Branch::Coding,
            Route::Cloud,
        ),
        (
            Some(TaskType::Agentic),
            Some(ToolErrorCount::Host(0)),
            Signals {
                readout: Some(0.25),
                ..Default::default()
            },
            Branch::Agentic,
            Route::Local,
        ),
        (
            Some(TaskType::Agentic),
            Some(ToolErrorCount::Untrusted(0)),
            Signals {
                readout: Some(1.0),
                ..Default::default()
            },
            Branch::Agentic,
            Route::Cloud,
        ),
    ];

    for (task_type, tool_errors, signals, branch, route) in cases {
        let caps = derive_capabilities(&request, "completed", task_type, tool_errors);
        let decision = decide_from_signals(&caps, &signals);
        assert_eq!((decision.branch, decision.route), (branch, route));
    }
}

#[test]
fn reported_tool_errors_veto_every_attempt_judging_branch() {
    let request = request("complete the task");
    let signals = Signals {
        readout: Some(1.0),
        deliberation: Some(1.0),
        cloud_judge: Some(Tri::Yes),
        strict_evidence: Some(Tri::Yes),
    };
    for task_type in [
        TaskType::Coding,
        TaskType::Agentic,
        TaskType::Answer,
        TaskType::Chat,
    ] {
        let caps = derive_capabilities(
            &request,
            "completed",
            Some(task_type),
            Some(ToolErrorCount::Host(1)),
        );
        assert_eq!(decide_from_signals(&caps, &signals).route, Route::Cloud);
    }
}
