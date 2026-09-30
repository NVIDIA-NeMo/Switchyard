// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Condensing the local tier's history when the capable tier takes over.
//!
//! An escalated session otherwise hands the capable tier every token the local
//! tier produced, uncached, and then re-sends that history on every later call.
//! Compaction replaces the local work with a digest appended to the task
//! message, once, at handoff.
//!
//! The client keeps sending its full history, so the same rewrite must be
//! reapplied to every later request of the user turn. It is reapplied only when
//! the client's history still starts with exactly the messages that were
//! condensed; anything else is sent untouched, which is always safe. The digest
//! is stored rather than recomputed so the rewritten prefix is byte-identical
//! across calls and stays in the capable tier's prompt cache.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use async_trait::async_trait;
use switchyard_protocol::{ContentBlock, Message, Request, Response, Role};

use super::text::clip_mid;
use crate::Result;
use crate::algorithms::util::affinity::has_new_user_turn;
use crate::algorithms::util::prompts::drop_exact_replay;
use crate::core::algorithm::Driver;
use crate::core::classifier::{Classification, Classifier};
use crate::core::state::{State, StateValue};

const HEAD_KEY: &str = "vgr.compaction.head";
const CUT_KEY: &str = "vgr.compaction.cut";
const ANCHOR_KEY: &str = "vgr.compaction.anchor";
const SUMMARY_KEY: &str = "vgr.compaction.summary";

/// Steps rendered in full at the end of the digest; earlier ones get one line.
///
/// The digest is re-sent on every later capable-tier call, so its size is paid
/// once per step, not once per handoff. About 2k tokens at most.
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
    state
        .extra
        .insert(HEAD_KEY.into(), StateValue::Count(head_count));
    state
        .extra
        .insert(CUT_KEY.into(), StateValue::Count(cut_count));
    state
        .extra
        .insert(ANCHOR_KEY.into(), StateValue::String(anchor));
    rewrite(request, head, cut, &summary);
    state
        .extra
        .insert(SUMMARY_KEY.into(), StateValue::String(summary));
    true
}

/// Reapplies a recorded compaction to a later request of the same user turn.
///
/// Registered ahead of every other classifier and never votes.
pub(super) struct Compactor;

#[async_trait]
impl Classifier<State> for Compactor {
    async fn score(
        &self,
        state: &mut State,
        request: &mut Request,
        _driver: &Driver,
    ) -> Result<(Classification, Option<Response>)> {
        if has_new_user_turn(&request.llm_request.messages) {
            for key in [HEAD_KEY, CUT_KEY, ANCHOR_KEY, SUMMARY_KEY] {
                state.extra.remove(key);
            }
        } else if let Some((head, cut, summary)) = recorded(state, request) {
            rewrite(request, head, cut, &summary);
        }
        Ok((Classification::Scores(Vec::new()), None))
    }
}

/// The recorded compaction, when the request still begins with the condensed history.
fn recorded(state: &State, request: &Request) -> Option<(usize, usize, String)> {
    let count = |key| match state.extra.get(key) {
        Some(StateValue::Count(value)) => usize::try_from(*value).ok(),
        _ => None,
    };
    let text = |key| match state.extra.get(key) {
        Some(StateValue::String(value)) => Some(value.clone()),
        _ => None,
    };
    let (head, cut) = (count(HEAD_KEY)?, count(CUT_KEY)?);
    let messages = &request.llm_request.messages;
    (head < cut && cut <= messages.len() && fingerprint(&messages[..cut]) == text(ANCHOR_KEY)?)
        .then(|| text(SUMMARY_KEY))
        .flatten()
        .map(|summary| (head, cut, summary))
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
struct Step {
    narration: String,
    name: String,
    arguments: String,
    result: Option<(bool, String)>,
}

fn digest(messages: &[Message]) -> String {
    let mut steps: Vec<Step> = Vec::new();
    let mut narration = String::new();
    let mut user_text = Vec::new();
    for message in messages {
        for block in &message.content {
            match block {
                ContentBlock::Text { text } if message.role == Role::Assistant => {
                    narration.push_str(text);
                }
                ContentBlock::Text { text } if message.role == Role::User => {
                    user_text.push(clip_mid(text, RECENT_RESULT_BUDGET, 0.5));
                }
                ContentBlock::ToolCall(call) => steps.push(Step {
                    narration: std::mem::take(&mut narration),
                    name: call.name.clone(),
                    arguments: call.arguments.to_string(),
                    result: None,
                }),
                ContentBlock::ToolResult(result) => {
                    let body = result
                        .content
                        .iter()
                        .map(|part| match part {
                            ContentBlock::Text { text } => text.as_str(),
                            _ => "[non-text content]",
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    let failed = result.is_error == Some(true);
                    if let Some(step) = steps.iter_mut().rev().find(|step| step.result.is_none()) {
                        step.result = Some((failed, body));
                    }
                }
                _ => {}
            }
        }
    }

    let recent_from = steps.len().saturating_sub(RECENT_STEPS);
    let mut early = String::new();
    for (index, step) in steps[..recent_from].iter().enumerate() {
        let (status, first_line) = match &step.result {
            Some((failed, body)) => (
                if *failed { "error" } else { "ok" },
                body.lines()
                    .find(|line| !line.trim().is_empty())
                    .unwrap_or(""),
            ),
            None => ("no result", ""),
        };
        early.push_str(&format!(
            "[{}] {} {} -> {status}: {}\n",
            index + 1,
            step.name,
            clip_mid(&step.arguments, EARLY_ARGUMENT_BUDGET, 0.7),
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
        if !step.narration.trim().is_empty() {
            out.push_str(&format!(
                "assistant: {}\n",
                clip_mid(step.narration.trim(), NARRATION_BUDGET, 0.5)
            ));
        }
        out.push_str(&format!(
            "[{}] {} {}\n",
            recent_from + offset + 1,
            step.name,
            clip_mid(&step.arguments, RECENT_ARGUMENT_BUDGET, 0.5)
        ));
        match &step.result {
            Some((failed, body)) => out.push_str(&format!(
                "result{}:\n{}\n",
                if *failed { " (error)" } else { "" },
                clip_mid(body, RECENT_RESULT_BUDGET, 0.3)
            )),
            None => out.push_str("result: none\n"),
        }
    }
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
        assert!(!state.extra.contains_key(SUMMARY_KEY));
    }
}
