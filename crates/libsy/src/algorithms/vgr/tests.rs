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
// Router typing selects a branch; tool history takes precedence over coding.
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
// Missing text and unsupported content cannot license a local answer.
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
// Executor success takes precedence over inferred tool errors.
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
            clean_tail: 1,
        }
    );
}

#[test]
// Verifiers see user requirements and redacted tool history without framework instructions.
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

#[test]
fn redaction_covers_keys_tokens_and_email_and_is_idempotent() {
    for (input, expected) in [
        ("sk-abcdefghijklmnop", "[REDACTED]"),
        ("rk-abcdefghijklmnop", "[REDACTED]"),
        ("pk-abcdefghijklmnop", "[REDACTED]"),
        ("AKIA0123456789ABCDEF", "[REDACTED]"),
        (
            "Authorization: Bearer abcdefghijklmnop",
            "Authorization: [REDACTED]",
        ),
        ("BEARER abcdefghijklmnop", "[REDACTED]"),
        ("contact dev+test@example.com", "contact [REDACTED]"),
        ("ordinary task text", "ordinary task text"),
        ("sk-abcdefghijklmno", "sk-abcdefghijklmno"),
        ("Bearer abcdefghijklmno", "Bearer abcdefghijklmno"),
    ] {
        let redacted = super::text::redact(input);
        assert_eq!(redacted, expected, "{input}");
        assert_eq!(super::text::redact(&redacted), expected, "{input}");
    }
}

#[test]
fn clipping_preserves_character_budgets_and_utf8_boundaries() {
    for (input, budget, fraction, expected) in [
        ("abcdef", 6, 0.5, "abcdef"),
        ("abcdef", 4, 0.5, "ab\n...[2 chars omitted]...\nef"),
        ("aé🙂中z", 4, 0.5, "aé\n...[1 chars omitted]...\n中z"),
        ("aé🙂中z", 3, 1.0 / 3.0, "a\n...[2 chars omitted]...\n中z"),
        ("aé🙂中z", 2, 1.0, "aé\n...[3 chars omitted]...\n"),
        ("aé🙂中z", 1, 0.5, "a\n...[4 chars omitted]...\n"),
    ] {
        assert_eq!(super::text::clip_mid(input, budget, fraction), expected);
    }
}

#[test]
fn session_rendering_preserves_roles_and_moves_the_latest_user_turn() {
    let session = request(vec![
        Message::text(Role::System, "system rule"),
        Message::text(Role::Developer, "developer rule"),
        Message::text(Role::User, "first task"),
        Message::text(Role::Assistant, "first reply"),
        Message::text(Role::User, "latest task"),
        Message::text(Role::Tool, "tool output"),
        Message::text(Role::User, " "),
    ]);
    let (turns, _) = super::text::turns(&session);
    assert_eq!(
        super::render::render_session(&turns, "done"),
        "[system] system rule\n[developer] developer rule\n[user] first task\n[assistant] first reply\n[tool] tool output\n[user (latest)] latest task\n[assistant attempt] done"
    );
}

#[test]
fn agentic_rendering_preserves_long_user_updates_and_clips_redacted_attempts() {
    let task = "é".repeat(2500);
    let session = request(vec![
        Message::text(Role::System, "framework"),
        Message::text(Role::User, &task),
        Message::text(Role::Assistant, "previous reply"),
        Message::text(Role::User, " "),
        Message::text(Role::User, "update sk-abcdefghijklmnop"),
    ]);
    let (turns, _) = super::text::turns(&session);
    // A large secret must be redacted before clipping can expose fragments of it.
    let attempt = format!("Bearer {}", "x".repeat(25_000));
    let view = super::render::render_agentic_view(&session, &turns, &attempt);
    assert!(view.starts_with(&format!(
        "USER TASK AND UPDATES:\n[user task] {task}\n[user update 1] update [REDACTED]"
    )));
    assert!(view.contains("TOOL TRAJECTORY:\n[assistant] previous reply"));
    assert!(!view.contains("framework"));
    assert!(view.ends_with("CURRENT ATTEMPT:\n[REDACTED]"));
    assert!(!view.contains("chars omitted"));
}

mod decide {
    use super::super::decide::{AgenticRun, Route, Signals, Tri, agentic_run, decide};
    use super::super::text::ToolRecord;
    use super::super::{Branch, Capabilities};

    fn agentic(errors: i32, results: i32, clean_tail: i32) -> Capabilities {
        Capabilities {
            branch: Branch::Agentic,
            transcript: Some("view".into()),
            tools: Some(ToolRecord {
                errors,
                results,
                clean_tail,
            }),
        }
    }

    fn readout(score: f64) -> Signals {
        Signals {
            readout: Some(score),
            ..Signals::default()
        }
    }

    #[test]
    fn agentic_runs_commit_only_on_a_clean_or_recovered_record() {
        assert_eq!(decide(&agentic(0, 3, 3), &readout(0.2), None), Route::Local);
        assert_eq!(
            decide(&agentic(0, 3, 3), &readout(0.19), None),
            Route::Cloud
        );

        let short = agentic(2, 100, 1);
        assert_eq!(agentic_run(&short, None), AgenticRun::ShortRecovered);
        assert_eq!(decide(&short, &readout(0.9), None), Route::Local);
        let unrecovered = agentic(2, 100, 0);
        assert_eq!(decide(&unrecovered, &readout(1.0), Some(1)), Route::Cloud);

        let long = agentic(2, 101, 1);
        assert_eq!(agentic_run(&long, None), AgenticRun::Vetoed);
        assert_eq!(agentic_run(&long, Some(1)), AgenticRun::ConfirmedRecovery);
        let mut confirmed = readout(0.9);
        assert_eq!(decide(&long, &confirmed, Some(1)), Route::Cloud);
        confirmed.cloud_judge = Some(Tri::Yes);
        assert_eq!(decide(&long, &confirmed, Some(1)), Route::Local);
    }

    #[test]
    fn indeterminate_or_unconsulted_evidence_never_commits() {
        let mut chat = agentic(0, 0, 0);
        chat.branch = Branch::Chat;
        let mut signals = Signals {
            cloud_judge: Some(Tri::Yes),
            ..Signals::default()
        };
        assert_eq!(decide(&chat, &signals, None), Route::Cloud);
        signals.evidence_confirm = Some(Tri::Unknown);
        assert_eq!(decide(&chat, &signals, None), Route::Cloud);
        signals.evidence_confirm = Some(Tri::Yes);
        assert_eq!(decide(&chat, &signals, None), Route::Local);

        chat.tools = Some(ToolRecord {
            errors: 1,
            ..ToolRecord::default()
        });
        assert_eq!(decide(&chat, &signals, None), Route::Cloud);
        chat.branch = Branch::Coding;
        assert_eq!(decide(&chat, &readout(1.0), None), Route::Cloud);
    }

    #[test]
    fn deliberation_commits_only_on_an_affirmative_verdict() {
        for branch in [Branch::Chat, Branch::Agentic, Branch::DefaultVerified] {
            let mut caps = agentic(0, 1, 1);
            caps.branch = branch;
            for (verdict, expected) in [
                (Some(Tri::Yes), Route::Local),
                (Some(Tri::No), Route::Cloud),
                (Some(Tri::Unknown), Route::Cloud),
                (None, Route::Cloud),
            ] {
                let signals = Signals {
                    deliberation: verdict,
                    ..Signals::default()
                };
                assert_eq!(
                    decide(&caps, &signals, None),
                    expected,
                    "{branch:?}: {verdict:?}"
                );
            }
        }
    }
}
