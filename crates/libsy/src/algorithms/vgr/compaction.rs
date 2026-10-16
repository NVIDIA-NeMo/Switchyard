// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Condenses local history into a recorded handoff digest.
//!
//! Later requests reuse the digest only when their original history prefix
//! matches. User updates and tool content are clipped, so the digest is lossy.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use async_trait::async_trait;
use switchyard_protocol::{ContentBlock, Message, Request, Response, Role, ToolCall, ToolResult};

use super::text::clip_mid;
use crate::Result;
use crate::algorithms::util::affinity::has_new_user_turn;
use crate::algorithms::util::prompts::drop_exact_replay;
use crate::core::algorithm::Driver;
use crate::core::classifier::{Classification, Classifier};
use crate::core::state::{State, StateValue};
use serde::{Deserialize, Serialize};

const RECORD_KEY: &str = "vgr_compaction";

#[derive(Serialize, Deserialize)]
struct Recorded {
    head: u32,
    cut: u32,
    anchor: String,
    summary: String,
}

/// Tool steps shown in detail; earlier steps receive short entries.
const RECENT_STEPS: usize = 3;
const RECENT_ARGUMENT_BUDGET: usize = 600;
const RECENT_RESULT_BUDGET: usize = 800;
const EARLY_ARGUMENT_BUDGET: usize = 100;
const EARLY_RESULT_BUDGET: usize = 80;
const EARLY_SECTION_BUDGET: usize = 2_500;
const NARRATION_BUDGET: usize = 200;
const RECENT_USER_TEXTS: usize = 2;

/// Condenses everything after the task message and records how, so later
/// requests of this user turn can be rewritten identically.
///
/// Returns `false`, leaving the request untouched, when there is no task
/// message with local work after it.
pub(super) fn compact(request: &mut Request, state: &mut State, note: &str) -> bool {
    let messages = &request.llm_request.messages;
    let Some(task) = messages.iter().position(is_task_message) else {
        return false;
    };
    let head = task + 1;
    let cut = messages.len();
    if cut <= head {
        return false;
    }
    let mut summary = digest(&messages[head..cut]);
    summary.push_str(note);
    let anchor = fingerprint(&messages[..cut]);
    let (Ok(head_count), Ok(cut_count)) = (u32::try_from(head), u32::try_from(cut)) else {
        return false;
    };
    let record = Recorded {
        head: head_count,
        cut: cut_count,
        anchor,
        summary,
    };
    let Ok(serialized) = serde_json::to_string(&record) else {
        return false;
    };
    rewrite(request, head, cut, &record.summary);
    state
        .extra
        .insert(RECORD_KEY.into(), StateValue::String(serialized));
    true
}

/// Reapplies a recorded compaction to a later request of the same user turn.
///
/// Registered ahead of every other classifier and never votes.
pub(super) struct Compactor;

#[async_trait]
impl Classifier for Compactor {
    async fn score(
        &self,
        state: &mut State,
        request: &mut Request,
        _driver: &Driver,
    ) -> Result<(Classification, Option<Response>)> {
        if has_new_user_turn(&request.llm_request.messages) {
            state.extra.remove(RECORD_KEY);
        } else if let Some((head, cut, summary)) = recorded(state, request) {
            rewrite(request, head, cut, &summary);
        }
        Ok((Classification::Scores(Vec::new()), None))
    }
}

/// The recorded compaction, when the request still begins with the condensed history.
fn recorded(state: &State, request: &Request) -> Option<(usize, usize, String)> {
    let StateValue::String(serialized) = state.extra.get(RECORD_KEY)? else {
        return None;
    };
    let record: Recorded = serde_json::from_str(serialized).ok()?;
    let head = usize::try_from(record.head).ok()?;
    let cut = usize::try_from(record.cut).ok()?;
    let messages = &request.llm_request.messages;
    (head > 0
        && head < cut
        && cut <= messages.len()
        && fingerprint(&messages[..cut]) == record.anchor)
        .then(|| (head, cut, record.summary.clone()))
}

fn rewrite(request: &mut Request, head: usize, cut: usize, summary: &str) {
    let messages = &mut request.llm_request.messages;
    messages[head - 1].content.push(ContentBlock::Text {
        text: summary.to_string(),
    });
    messages.drain(head..cut);
    drop_exact_replay(request);
}

fn is_task_message(message: &Message) -> bool {
    message.role == Role::User
        && message
            .content
            .iter()
            .any(|block| matches!(block, ContentBlock::Text { .. }))
}

fn fingerprint(messages: &[Message]) -> String {
    let mut hasher = DefaultHasher::new();
    serde_json::to_string(messages)
        .unwrap_or_default()
        .hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// One tool call and what came back, in conversation order.
struct Step<'a> {
    narration: Vec<&'a str>,
    call: &'a ToolCall,
    result: Option<&'a ToolResult>,
}

fn result_text(result: &ToolResult) -> String {
    result
        .content
        .iter()
        .map(|part| match part {
            ContentBlock::Text { text } => text.as_str(),
            _ => "[non-text content]",
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn digest(messages: &[Message]) -> String {
    let mut steps = Vec::<Step<'_>>::new();
    let mut narration = Vec::new();
    let mut user_text = Vec::new();
    for message in messages {
        for block in &message.content {
            match block {
                ContentBlock::Text { text } if message.role == Role::Assistant => {
                    narration.push(text.as_str());
                }
                ContentBlock::Text { text } if message.role == Role::User => {
                    user_text.push(clip_mid(text, RECENT_RESULT_BUDGET, 0.5));
                }
                ContentBlock::ToolCall(call) => steps.push(Step {
                    narration: std::mem::take(&mut narration),
                    call,
                    result: None,
                }),
                ContentBlock::ToolResult(result) => {
                    if let Some(step) = steps
                        .iter_mut()
                        .rev()
                        .find(|step| step.call.id == result.tool_call_id && step.result.is_none())
                    {
                        step.result = Some(result);
                    }
                }
                _ => {}
            }
        }
    }

    let recent_from = steps.len().saturating_sub(RECENT_STEPS);
    let mut early = String::new();
    for (index, step) in steps[..recent_from].iter().enumerate() {
        let body = step.result.map(result_text).unwrap_or_default();
        let status = match step.result {
            Some(result) if result.is_error == Some(true) => "error",
            Some(_) => "ok",
            None => "no result",
        };
        let first_line = body
            .lines()
            .find(|line| !line.trim().is_empty())
            .unwrap_or("");
        early.push_str(&format!(
            "[{}] {} {} -> {status}: {}\n",
            index + 1,
            step.call.name,
            clip_mid(&step.call.arguments.to_string(), EARLY_ARGUMENT_BUDGET, 0.7),
            clip_mid(first_line, EARLY_RESULT_BUDGET, 0.7),
        ));
    }
    let mut out = String::from(
        "\n\n[Context note from the serving infrastructure, not from the user] The tool-using \
         work that followed this message was condensed into the digest below. Earlier steps \
         are one line each; the most recent steps are shown in detail.\n<condensed_history>\n",
    );
    out.push_str(&clip_mid(&early, EARLY_SECTION_BUDGET, 0.3));
    for (offset, step) in steps[recent_from..].iter().enumerate() {
        let narration = step.narration.concat();
        if !narration.trim().is_empty() {
            out.push_str(&format!(
                "assistant: {}\n",
                clip_mid(narration.trim(), NARRATION_BUDGET, 0.5)
            ));
        }
        out.push_str(&format!(
            "[{}] {} {}\n",
            recent_from + offset + 1,
            step.call.name,
            clip_mid(
                &step.call.arguments.to_string(),
                RECENT_ARGUMENT_BUDGET,
                0.5
            )
        ));
        match step.result {
            Some(result) => out.push_str(&format!(
                "result{}:\n{}\n",
                if result.is_error == Some(true) {
                    " (error)"
                } else {
                    ""
                },
                clip_mid(&result_text(result), RECENT_RESULT_BUDGET, 0.3)
            )),
            None => out.push_str("result: none\n"),
        }
    }
    let narration = narration.concat();
    if !narration.trim().is_empty() {
        out.push_str(&format!(
            "assistant: {}\n",
            clip_mid(narration.trim(), NARRATION_BUDGET, 0.5)
        ));
    }
    for text in &user_text[user_text.len().saturating_sub(RECENT_USER_TEXTS)..] {
        out.push_str(&format!("user: {text}\n"));
    }
    out.push_str("</condensed_history>");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use switchyard_protocol::{LlmRequest, ToolCall, ToolResult};

    fn call(id: &str, command: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: id.into(),
                name: "bash".into(),
                arguments: serde_json::json!({ "command": command }),
            })],
        }
    }

    fn result(id: &str, text: &str) -> Message {
        Message {
            role: Role::Tool,
            content: vec![ContentBlock::ToolResult(ToolResult {
                tool_call_id: id.into(),
                content: vec![ContentBlock::Text { text: text.into() }],
                is_error: None,
            })],
        }
    }

    fn session(steps: usize) -> Request {
        let mut messages = vec![Message::text(Role::User, "build the thing")];
        for index in 0..steps {
            let id = format!("call-{index}");
            messages.push(call(&id, &format!("step {index}")));
            messages.push(result(&id, &format!("output {index}")));
        }
        Request {
            llm_request: LlmRequest {
                messages,
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: None,
        }
    }

    async fn reapply(state: &mut State, request: &mut Request) {
        let driver = Driver::new(
            "test",
            std::sync::Arc::new(crate::core::algorithm::RuntimeModels::new(
                std::collections::HashMap::new(),
            )),
        )
        .0;
        Compactor
            .score(state, request, &driver)
            .await
            .expect("the compactor never fails");
    }

    #[tokio::test]
    async fn later_requests_see_the_same_condensed_prefix() {
        let mut state = State::default();
        let mut handoff = session(12);
        assert!(compact(&mut handoff, &mut state, "NOTE"));
        assert_eq!(handoff.llm_request.messages.len(), 1);
        let condensed = handoff.llm_request.messages[0].clone();
        let digest = condensed.text_content("").expect("task text");
        assert!(digest.contains("step 11") && digest.ends_with("NOTE"));

        let saved = serde_json::to_string(&state).expect("serializable routing state");
        let mut state: State = serde_json::from_str(&saved).expect("restored routing state");

        let mut follow_up = session(12);
        follow_up.llm_request.messages.push(call("cloud-1", "ls"));
        follow_up
            .llm_request
            .messages
            .push(result("cloud-1", "a b"));
        reapply(&mut state, &mut follow_up).await;

        let messages = &follow_up.llm_request.messages;
        assert_eq!(
            messages.len(),
            3,
            "task, then only the capable tier's own work"
        );
        assert_eq!(
            messages[0], condensed,
            "the cached prefix is byte-identical"
        );
    }

    #[tokio::test]
    async fn a_rewritten_history_is_sent_untouched() {
        let mut state = State::default();
        let mut handoff = session(4);
        assert!(compact(&mut handoff, &mut state, "NOTE"));

        let mut edited = session(4);
        edited.llm_request.messages[2] = result("call-0", "the client rewrote this");
        let before = edited.llm_request.messages.clone();
        reapply(&mut state, &mut edited).await;
        assert_eq!(edited.llm_request.messages, before);
    }

    #[tokio::test]
    async fn a_new_user_turn_forgets_the_compaction() {
        let mut state = State::default();
        let mut handoff = session(4);
        assert!(compact(&mut handoff, &mut state, "NOTE"));

        let mut next_turn = session(4);
        next_turn
            .llm_request
            .messages
            .push(Message::text(Role::User, "now do another thing"));
        let before = next_turn.llm_request.messages.clone();
        reapply(&mut state, &mut next_turn).await;
        assert_eq!(next_turn.llm_request.messages, before);
        assert!(!state.extra.contains_key(RECORD_KEY));
    }

    #[test]
    fn batched_results_are_paired_by_tool_call_id() {
        let calls = Message {
            role: Role::Assistant,
            content: ["a", "b"]
                .into_iter()
                .map(|id| {
                    ContentBlock::ToolCall(ToolCall {
                        id: id.into(),
                        name: format!("tool_{id}"),
                        arguments: serde_json::json!({}),
                    })
                })
                .collect(),
        };
        for order in [["a", "b"], ["b", "a"]] {
            let results = Message {
                role: Role::Tool,
                content: order
                    .into_iter()
                    .map(|id| {
                        ContentBlock::ToolResult(ToolResult {
                            tool_call_id: id.into(),
                            content: vec![ContentBlock::Text {
                                text: format!("output_{id}"),
                            }],
                            is_error: Some(id == "b"),
                        })
                    })
                    .collect(),
            };
            let view = digest(&[calls.clone(), results]);
            assert!(view.contains("tool_a {}\nresult:\noutput_a"), "{view}");
            assert!(
                view.contains("tool_b {}\nresult (error):\noutput_b"),
                "{view}"
            );
        }
    }
}
