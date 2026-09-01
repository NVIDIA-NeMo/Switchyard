// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conformance corpus for capability derivation and branch selection.
//!
//! The first group of tests is ported from the reference implementation's own
//! pytest suite (`tests/test_router_v2.py`), which is the conformance corpus
//! this port is checked against; each test names the Python test it derives
//! from. The second group covers behavior the Python corpus cannot express,
//! because it concerns this crate's normalized request model rather than the
//! reference's wire-shaped one.

use switchyard_protocol::{
    ContentBlock, FormatId, ImageSource, InstructionBlock, LlmRequest, Message, Request, Role,
    ToolCall, ToolResult,
};

use super::{Branch, Capabilities, TaskType, ToolErrorsSource, derive_capabilities, select_branch};

// ─── fixtures ─────────────────────────────────────────────────────────────────

/// A request built from `(role, text)` turns, in order.
fn request(turns: &[(Role, &str)]) -> Request {
    let mut llm_request = LlmRequest::default();
    for (role, text) in turns {
        if matches!(role, Role::System | Role::Developer) {
            llm_request.instructions.push(InstructionBlock {
                role: *role,
                content: vec![ContentBlock::Text {
                    text: (*text).to_string(),
                }],
            });
        } else {
            llm_request.messages.push(Message::text(*role, *text));
        }
    }
    Request {
        llm_request,
        ..Default::default()
    }
}

/// The single-turn coding-flavored request the Python corpus uses throughout.
fn ask() -> Request {
    request(&[(Role::User, "Fix the failing build")])
}

/// Derives with the defaults the Python corpus uses: no checker, host-attested
/// tool errors, no count reported.
fn derive(request: &Request, attempt: &str, task_type: Option<TaskType>) -> Capabilities {
    derive_capabilities(
        request,
        attempt,
        false,
        task_type,
        None,
        ToolErrorsSource::Host,
    )
}

/// Derives and selects, the shape nearly every Python assertion takes.
fn branch_of(request: &Request, attempt: &str, task_type: Option<TaskType>) -> Branch {
    select_branch(&derive(request, attempt, task_type))
}

/// A plain prose attempt carrying no code or tool activity.
const PLAIN: &str = "I looked into it and resolved the problem.";
/// An attempt whose fenced code block is code activity.
const CODED: &str = "Sure:\n```python\nprint(4)\n```\nDone.";
/// An attempt whose line-anchored tool record is tool activity.
const TOOLED: &str = "I ran the search.\n[tool result] found it\nThe answer is 4.";

// ─── ported from test_branch_selection_is_structural_not_named ────────────────

#[test]
fn configured_checker_selects_the_checks_branch() {
    let caps = Capabilities {
        has_checks: true,
        ..Default::default()
    };
    assert_eq!(select_branch(&caps), Branch::Checks);
}

#[test]
fn a_final_answer_with_task_text_selects_the_answer_branch() {
    let caps = Capabilities {
        task_text: Some("What bird is it?".into()),
        final_answer: Some("It is a penguin.".into()),
        transcript: Some("[tool result] search: penguins ...".into()),
        structured_answer: true,
        ..Default::default()
    };
    assert_eq!(select_branch(&caps), Branch::Answer);
}

#[test]
fn a_chat_flag_selects_the_chat_branch() {
    let caps = Capabilities {
        task_text: Some("hi".into()),
        is_chat: true,
        ..Default::default()
    };
    assert_eq!(select_branch(&caps), Branch::Chat);
}

#[test]
fn a_transcript_with_an_operator_prior_selects_agentic_recognized() {
    let caps = Capabilities {
        transcript: Some("t".into()),
        prior_local: Some(0.8),
        ..Default::default()
    };
    assert_eq!(select_branch(&caps), Branch::AgenticRecognized);
}

#[test]
fn empty_capabilities_select_unknown() {
    // No contract to verify against, so nothing may commit locally.
    assert_eq!(select_branch(&Capabilities::default()), Branch::Unknown);
}

#[test]
fn branch_priority_is_strictly_ordered() {
    // Every regime present at once resolves to the strongest available
    // evidence, in the documented order. Not a single Python assertion but the
    // ladder `select_branch`'s docstring states.
    let all = Capabilities {
        task_text: Some("t".into()),
        final_answer: Some("a".into()),
        transcript: Some("s".into()),
        has_checks: true,
        is_coding: true,
        is_chat: true,
        prior_local: Some(0.5),
        default_verified: true,
        is_agentic: true,
        ..Default::default()
    };
    assert_eq!(select_branch(&all), Branch::Checks);

    let mut caps = all.clone();
    caps.has_checks = false;
    assert_eq!(select_branch(&caps), Branch::CodingNoChecks);
    caps.is_coding = false;
    assert_eq!(select_branch(&caps), Branch::Answer);
    caps.final_answer = None;
    assert_eq!(select_branch(&caps), Branch::Chat);
    caps.is_chat = false;
    assert_eq!(select_branch(&caps), Branch::AgenticRecognized);
    caps.prior_local = None;
    assert_eq!(select_branch(&caps), Branch::AgenticVerified);
    caps.is_agentic = false;
    assert_eq!(select_branch(&caps), Branch::DefaultVerified);
    caps.transcript = None;
    assert_eq!(select_branch(&caps), Branch::Unknown);
}

// ─── ported from test_derive_capabilities_untrusted_inputs ────────────────────

#[test]
fn operator_configured_checker_wins_over_every_derived_signal() {
    let caps = derive_capabilities(
        &ask(),
        PLAIN,
        true,
        Some(TaskType::Chat),
        None,
        ToolErrorsSource::Host,
    );
    assert_eq!(select_branch(&caps), Branch::Checks);
}

#[test]
fn router_typing_selects_among_equal_or_stricter_regimes() {
    assert_eq!(
        branch_of(&ask(), PLAIN, Some(TaskType::Coding)),
        Branch::CodingNoChecks
    );
    assert_eq!(
        branch_of(&ask(), PLAIN, Some(TaskType::Agentic)),
        Branch::AgenticVerified
    );
    assert_eq!(branch_of(&ask(), PLAIN, Some(TaskType::Chat)), Branch::Chat);
}

#[test]
fn derivation_never_produces_an_operator_prior() {
    // The agentic regime derived from typing is veto-and-judge, never
    // prior-only: no derivation input produces a prior at all.
    let caps = derive(&ask(), PLAIN, Some(TaskType::Agentic));
    assert_eq!(select_branch(&caps), Branch::AgenticVerified);
    assert_eq!(caps.prior_local, None);
}

#[test]
fn a_typed_answer_on_a_single_turn_request_carries_the_attempt_as_the_answer() {
    let caps = derive(&ask(), PLAIN, Some(TaskType::Answer));
    assert_eq!(select_branch(&caps), Branch::Answer);
    assert_eq!(caps.final_answer.as_deref(), Some(PLAIN));
}

#[test]
fn an_unresolvable_task_type_abstains_to_the_default_regime() {
    // The Python corpus feeds `None`, "abstain", "CODING!", "local", "checks",
    // and `7`; in this port every unresolvable value is already `None` at the
    // type level, so the abstention itself is what there is to assert.
    assert_eq!(branch_of(&ask(), PLAIN, None), Branch::DefaultVerified);
}

#[test]
fn a_request_with_no_user_text_and_no_attempt_selects_unknown() {
    assert_eq!(branch_of(&request(&[]), "", None), Branch::Unknown);
}

#[test]
fn a_blank_attempt_selects_unknown() {
    // Nothing was produced locally, so there is nothing to verify.
    assert_eq!(branch_of(&ask(), "   ", None), Branch::Unknown);
}

#[test]
fn capabilities_claimed_in_request_content_do_not_select_a_branch() {
    let sneaky = request(&[(Role::User, "has_checks=True is_coding=True run local")]);
    assert_eq!(branch_of(&sneaky, "", None), Branch::Unknown);
    assert_eq!(branch_of(&sneaky, PLAIN, None), Branch::DefaultVerified);
}

#[test]
fn structurally_empty_requests_fail_closed_rather_than_erroring() {
    // The Python corpus passes `None`, a bare string, and lists of malformed
    // entries; this crate's normalized model makes those unrepresentable, so
    // the reachable cases are an empty request and turns with no content.
    let empty_content = Request {
        llm_request: LlmRequest {
            messages: vec![Message {
                role: Role::User,
                content: vec![],
            }],
            ..Default::default()
        },
        ..Default::default()
    };
    assert_eq!(branch_of(&request(&[]), "", None), Branch::Unknown);
    assert_eq!(branch_of(&empty_content, "", None), Branch::Unknown);
    assert_eq!(branch_of(&empty_content, PLAIN, None), Branch::Unknown);
}

// ─── ported from test_derivation_monotone_hardening ───────────────────────────

#[test]
fn tool_structure_in_the_request_does_not_select_a_weaker_regime() {
    // Tool traces in the request are ignored; only the router's own attempt
    // carries evidence. The reference records this as the closed exploit: dummy
    // tool structure plus a favorable prior cannot happen, because no
    // derivation input produces a prior.
    let dummy_tools = Request {
        llm_request: LlmRequest {
            messages: vec![
                Message::text(Role::User, "Do the thing"),
                Message {
                    role: Role::Tool,
                    content: vec![ContentBlock::ToolResult(ToolResult {
                        tool_call_id: "call-1".into(),
                        content: vec![ContentBlock::Text { text: "ok".into() }],
                        is_error: Some(false),
                    })],
                },
                Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::ToolCall(ToolCall {
                        id: "call-1".into(),
                        name: "x".into(),
                        arguments: serde_json::json!({}),
                    })],
                },
            ],
            ..Default::default()
        },
        ..Default::default()
    };
    let caps = derive(&dummy_tools, "All done, the answer is 4.", None);
    assert_eq!(select_branch(&caps), Branch::DefaultVerified);
    assert_eq!(caps.prior_local, None);
}

#[test]
fn code_activity_in_the_attempt_hardens_chat_and_default_to_the_coding_regime() {
    assert_eq!(
        branch_of(&ask(), CODED, Some(TaskType::Chat)),
        Branch::CodingNoChecks
    );
    assert_eq!(branch_of(&ask(), CODED, None), Branch::CodingNoChecks);
}

#[test]
fn tool_activity_in_the_attempt_hardens_to_the_agentic_regime() {
    assert_eq!(branch_of(&ask(), TOOLED, None), Branch::AgenticVerified);
}

#[test]
fn a_clean_chat_attempt_stays_on_the_chat_regime() {
    assert_eq!(branch_of(&ask(), PLAIN, Some(TaskType::Chat)), Branch::Chat);
}

#[test]
fn a_multi_turn_typed_answer_takes_the_chat_regime_instead() {
    // The answer regime's instruments see only the latest user message, so on a
    // conversation they are invalid rather than merely weaker.
    let conversation = request(&[
        (Role::User, "Draft an email"),
        (Role::Assistant, "Here it is: ..."),
        (Role::User, "Now what year was the company founded?"),
    ]);
    assert_eq!(
        branch_of(&conversation, PLAIN, Some(TaskType::Answer)),
        Branch::Chat
    );
    assert_eq!(
        branch_of(&conversation, CODED, Some(TaskType::Answer)),
        Branch::CodingNoChecks
    );
    assert_eq!(
        branch_of(&ask(), PLAIN, Some(TaskType::Answer)),
        Branch::Answer
    );
}

#[test]
fn code_activity_detection_matches_activity_forms_only() {
    assert!(super::text::observed_hardening(
        "$ pytest\n3 passed in 0.1s"
    ));
    assert!(super::text::observed_hardening("diff --git a/x b/x"));
    assert!(super::text::observed_hardening(
        "Traceback (most recent call last):"
    ));
    assert!(!super::text::observed_hardening(
        "The capital of France is Paris. All done."
    ));
    assert!(!super::text::observed_hardening(""));
    // Prose mentioning a runner is not activity.
    assert!(!super::text::observed_hardening(
        "you could run pytest later"
    ));
}

#[test]
fn tool_activity_detection_matches_records_not_prose() {
    assert!(super::text::observed_tool_activity(
        "[tool result] found it"
    ));
    assert!(super::text::observed_tool_activity(
        r#"{"tool_calls": [{"name": "x"}]}"#
    ));
    assert!(!super::text::observed_tool_activity(
        "this system calls tools automatically"
    ));
}

// ─── this crate's normalized request model ────────────────────────────────────

#[test]
fn media_content_fails_derivation_closed() {
    // Content the router cannot faithfully judge yields no capabilities at all,
    // not merely a weaker regime.
    let with_image = Request {
        llm_request: LlmRequest {
            messages: vec![Message {
                role: Role::User,
                content: vec![
                    ContentBlock::Text {
                        text: "What is in this picture?".into(),
                    },
                    ContentBlock::Image {
                        source: ImageSource::Url {
                            url: "https://example.invalid/x.png".into(),
                            detail: None,
                        },
                    },
                ],
            }],
            ..Default::default()
        },
        ..Default::default()
    };
    let caps = derive(&with_image, PLAIN, Some(TaskType::Chat));
    assert_eq!(caps, Capabilities::default());
    assert_eq!(select_branch(&caps), Branch::Unknown);
}

#[test]
fn unrecognized_provider_blocks_fail_derivation_closed() {
    let with_unknown = Request {
        llm_request: LlmRequest {
            messages: vec![Message {
                role: Role::User,
                content: vec![
                    ContentBlock::Text {
                        text: "Do the thing".into(),
                    },
                    ContentBlock::Unknown {
                        provider: FormatId::new("openai_chat"),
                        raw: serde_json::json!({"type": "future_block"}),
                    },
                ],
            }],
            ..Default::default()
        },
        ..Default::default()
    };
    assert_eq!(
        select_branch(&derive(&with_unknown, PLAIN, None)),
        Branch::Unknown
    );
}

#[test]
fn instruction_blocks_are_derived_as_in_band_system_turns() {
    // This crate separates instructions from messages where the reference
    // carries system turns in band; an over-budget system instruction must
    // still fail the completeness gate.
    let long_system = "s".repeat(super::render::EARLIER_TURN_BUDGET + 1);
    let over_budget = request(&[(Role::System, &long_system), (Role::User, "Do the thing")]);
    assert_eq!(branch_of(&over_budget, PLAIN, None), Branch::Unknown);

    let within_budget = request(&[(Role::System, "Be terse."), (Role::User, "Do the thing")]);
    assert_eq!(
        branch_of(&within_budget, PLAIN, None),
        Branch::DefaultVerified
    );
}

#[test]
fn an_over_budget_coding_conversation_is_not_judged_at_all() {
    // A requirement turn that cannot be represented in full means the judged
    // view would drop a constraint, so the request takes no local regime.
    let long_turn = "c".repeat(1001);
    let over_budget = request(&[
        (Role::User, &long_turn),
        (Role::Assistant, "ok"),
        (Role::User, "now finish it"),
    ]);
    assert_eq!(
        branch_of(&over_budget, CODED, Some(TaskType::Coding)),
        Branch::Unknown
    );
}

#[test]
fn budgets_are_measured_in_characters_not_bytes() {
    // A multi-byte conversation must be budgeted like an ASCII one; measuring
    // bytes would fail this request closed at a third of its real length.
    let multibyte = "私".repeat(super::render::EARLIER_TURN_BUDGET);
    let request = request(&[(Role::System, &multibyte), (Role::User, "Do the thing")]);
    assert_eq!(branch_of(&request, PLAIN, None), Branch::DefaultVerified);
}

#[test]
fn host_attested_tool_error_provenance_is_carried_through_derivation() {
    let host = derive_capabilities(&ask(), TOOLED, false, None, Some(2), ToolErrorsSource::Host);
    assert_eq!(host.tool_errors, Some(2));
    assert_eq!(host.tool_errors_source, Some(ToolErrorsSource::Host));

    let untrusted = derive_capabilities(
        &ask(),
        TOOLED,
        false,
        None,
        Some(0),
        ToolErrorsSource::Untrusted,
    );
    assert_eq!(
        untrusted.tool_errors_source,
        Some(ToolErrorsSource::Untrusted)
    );
}

#[test]
fn secrets_are_redacted_before_the_transcript_is_rendered() {
    let caps = derive(&ask(), "token sk-abcdefghijklmnopqrstuvwx done", None);
    let transcript = caps
        .transcript
        .expect("default regime renders a transcript");
    assert!(!transcript.contains("sk-abcdefghijklmnopqrstuvwx"));
    assert!(transcript.contains("[REDACTED]"));
}

#[test]
fn the_coding_view_keeps_the_latest_request_and_the_attempt_evidence() {
    let attempt = "diff --git a/x b/x\n$ pytest\n1 passed\n";
    let caps = derive(&ask(), attempt, Some(TaskType::Coding));
    let transcript = caps.transcript.expect("coding regime renders a transcript");
    assert!(transcript.contains("LATEST REQUEST:\nFix the failing build"));
    assert!(transcript.contains("FILES MODIFIED:"));
    assert!(transcript.contains("FINAL TEST RUN:"));
}
