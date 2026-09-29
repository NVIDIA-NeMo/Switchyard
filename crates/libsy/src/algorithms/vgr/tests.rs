// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use switchyard_protocol::{
    ContentBlock, ImageSource, LlmRequest, Message, Request, Role, ToolCall, ToolResult,
};

use super::text::ToolRecord;
use super::{Branch, TaskType, derive_capabilities};

pub(super) fn request(messages: Vec<Message>) -> Request {
    Request {
        llm_request: LlmRequest {
            messages,
            ..LlmRequest::default()
        },
        ..Default::default()
    }
}

pub(super) fn call(id: &str) -> Message {
    Message {
        role: Role::Assistant,
        content: vec![ContentBlock::ToolCall(ToolCall {
            id: id.into(),
            name: "bash".into(),
            arguments: serde_json::json!({"command": "ls"}),
        })],
    }
}

pub(super) fn result(id: &str, text: &str) -> Message {
    Message {
        role: Role::Tool,
        content: vec![ContentBlock::ToolResult(ToolResult {
            tool_call_id: id.into(),
            content: vec![ContentBlock::Text { text: text.into() }],
            is_error: None,
        })],
    }
}

fn branch(request: &Request, attempt: &str, task_type: Option<TaskType>) -> Branch {
    derive_capabilities(request, attempt, task_type).branch
}

#[test]
fn router_typing_selects_the_verification_regime() {
    let single = request(vec![Message::text(Role::User, "Help with this task")]);
    for (task_type, expected) in [
        (Some(TaskType::Coding), Branch::Coding),
        (Some(TaskType::Agentic), Branch::Agentic),
        (Some(TaskType::Answer), Branch::Answer),
        (Some(TaskType::Chat), Branch::Chat),
        (None, Branch::DefaultVerified),
    ] {
        assert_eq!(branch(&single, "done", task_type), expected);
    }

    // A tool trajectory outranks a coding type; only conversation types keep it off.
    let session = request(vec![
        Message::text(Role::User, "fix the build"),
        call("c1"),
        result("c1", "ok"),
    ]);
    assert_eq!(
        branch(&session, "done", Some(TaskType::Coding)),
        Branch::Agentic
    );
    assert_eq!(branch(&session, "done", Some(TaskType::Chat)), Branch::Chat);
    assert_eq!(branch(&single, "```\nx\n```", None), Branch::Coding);
}

#[test]
fn missing_or_unrepresentable_evidence_fails_closed() {
    let task = request(vec![Message::text(Role::User, "task")]);
    assert_eq!(branch(&task, " ", None), Branch::Unknown);
    assert_eq!(branch(&request(Vec::new()), "done", None), Branch::Unknown);

    let image = request(vec![Message {
        role: Role::User,
        content: vec![ContentBlock::Image {
            source: ImageSource::Url {
                url: "https://example.test/a.png".into(),
                detail: None,
            },
        }],
    }]);
    assert_eq!(branch(&image, "done", None), Branch::Unknown);
}

#[test]
fn a_zero_exit_code_overrules_every_error_inference() {
    let record = ToolRecord::from_request(&request(vec![
        result(
            "a",
            r#"{"output": "cat: HEAD: No such file", "exit_code": 0, "error": null}"#,
        ),
        result(
            "b",
            r#"{"output": "boom", "exit_code": 127, "error": "spawn failed"}"#,
        ),
        result("c", r#"{"error": "record not found"}"#),
        result("d", "Traceback (most recent call last):\n  ValueError"),
        result("e", "done"),
    ]));
    assert_eq!(
        record,
        ToolRecord {
            errors: 3,
            results: 5,
            tail_clean: true,
            clean_tail: 1,
        }
    );
}

#[test]
fn the_agentic_view_keeps_the_task_and_redacts_the_trajectory() {
    let session = request(vec![
        Message::text(Role::System, "framework boilerplate"),
        Message::text(Role::User, "rotate the key"),
        call("c1"),
        result("c1", "Authorization: Bearer abcdefghijklmnopqrstuvwxyz"),
    ]);
    let caps = derive_capabilities(&session, "rotated", None);
    let view = caps.transcript.unwrap_or_default();
    assert_eq!(caps.branch, Branch::Agentic);
    assert!(view.starts_with("USER TASK AND UPDATES:\n[user task] rotate the key"));
    assert!(view.contains("[tool call] bash"));
    assert!(view.contains("[REDACTED]") && !view.contains("abcdefghij"));
    assert!(!view.contains("boilerplate"));
    assert!(view.ends_with("CURRENT ATTEMPT:\nrotated"));
}
