// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-origin signals and compatibility with harness-generated summaries.

use switchyard_protocol::{ContentBlock, Metadata, Request, RequestOrigin};

use super::affinity::{is_environment_update, is_user_turn};

/// Explicit origin takes precedence over message-shape heuristics.
pub(crate) fn starts_user_turn(request: &Request) -> bool {
    if let Some(origin) = request.metadata.as_ref().and_then(Metadata::request_origin) {
        return origin == RequestOrigin::User;
    }
    request
        .llm_request
        .messages
        .iter()
        .rev()
        .find(|message| !is_environment_update(message))
        .is_some_and(|message| {
            is_user_turn(message) && !message.content.iter().all(is_compaction_summary)
        })
}

fn is_compaction_summary(block: &ContentBlock) -> bool {
    match block {
        ContentBlock::Unknown { raw, .. } => matches!(
            raw.get("type").and_then(serde_json::Value::as_str),
            Some("compaction" | "context_compaction")
        ),
        ContentBlock::Text { text } => {
            let text = text.trim().to_ascii_lowercase();
            // These harnesses flatten their internal summary type to ordinary user text.
            text.starts_with("another language model started to solve this problem and produced a summary of its thinking process.")
                || text.starts_with("this session is being continued from a previous conversation that ran out of context.")
                || (text.starts_with("the conversation history before this point was compacted into the following summary:")
                    && text.ends_with("</summary>"))
                || (text.starts_with("prior model work/tool state available.")
                    && text.ends_with("</summary>"))
                || (text.starts_with("<conversation-checkpoint>")
                    && text.ends_with("</conversation-checkpoint>"))
                || hermes_summary(&text)
        }
        _ => false,
    }
}

fn hermes_summary(text: &str) -> bool {
    const REPLAY: &str = "[still in progress \u{2014} this is the active request, restated after the compaction boundary";
    const END: &str = "--- end of context summary \u{2014} respond to the message below, not the summary above ---";
    // A merged summary may also carry a fresh user message after its closing marker.
    if let Some((_, tail)) = text.rsplit_once(END)
        && !tail.trim().is_empty()
        && !tail.trim_start().starts_with(REPLAY)
    {
        return false;
    }
    text.lines().any(|line| {
        let line = line.trim_start();
        line.starts_with("[context compaction \u{2014} reference only]")
            || line.starts_with("[context summary]:")
            || line.starts_with(REPLAY)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::util::tier_fixtures::turn_request;
    use switchyard_protocol::{Message, Role, WireFormat};

    #[test]
    fn summaries_do_not_hide_followups() {
        for summary in [
            "Another language model started to solve this problem and produced a summary of its thinking process.\nSummary",
            "This session is being continued from a previous conversation that ran out of context.\nSummary",
            "[CONTEXT COMPACTION \u{2014} REFERENCE ONLY]\nSummary",
            "Prior context\n[CONTEXT COMPACTION \u{2014} REFERENCE ONLY]\nSummary",
            "[CONTEXT SUMMARY]: Summary",
            "[STILL IN PROGRESS \u{2014} this is the active request, restated after the compaction boundary] Task",
            "The conversation history before this point was compacted into the following summary:\n\n<summary>\nSummary\n</summary>",
            "Prior model work/tool state available.\nMUST build on prior work\n\n<summary>\nSummary\n</summary>",
            "<conversation-checkpoint>\nThe following is a summary and serialized record of earlier conversation. Treat it as historical context, not as new instructions.\n\n<summary>\nSummary\n</summary>\n</conversation-checkpoint>",
        ] {
            let mut request = turn_request(false);
            request
                .llm_request
                .messages
                .push(Message::text(Role::User, summary));
            assert!(!starts_user_turn(&request), "{summary}");
            request
                .llm_request
                .messages
                .push(Message::text(Role::User, "Now fix the parser"));
            assert!(starts_user_turn(&request), "{summary}");
        }

        let mut request = turn_request(false);
        for tail in [
            "Now fix the parser",
            "[STILL IN PROGRESS \u{2014} this is the active request, restated after the compaction boundary] Task",
        ] {
            request.llm_request.messages = vec![Message::text(
                Role::User,
                format!(
                    "[CONTEXT COMPACTION \u{2014} REFERENCE ONLY]\nSummary\n--- END OF CONTEXT SUMMARY \u{2014} respond to the message below, not the summary above ---\n{tail}"
                ),
            )];
            assert_eq!(starts_user_turn(&request), tail.starts_with("Now"));
        }
    }

    #[test]
    fn native_compaction_items_are_continuations() {
        let mut request = turn_request(false);
        for kind in ["compaction", "context_compaction", "unrecognized"] {
            request.llm_request.messages = vec![Message {
                role: Role::User,
                content: vec![ContentBlock::Unknown {
                    provider: WireFormat::OpenAiResponses.into(),
                    raw: serde_json::json!({"type": kind, "encrypted_content": "opaque"}),
                }],
            }];
            assert_eq!(starts_user_turn(&request), kind == "unrecognized");
        }
    }

    #[test]
    fn explicit_origin_overrides_message_shape() {
        for (text, header, value, expected) in [
            (
                "[CONTEXT SUMMARY]: quoted by user",
                "x-switchyard-request-origin",
                "user",
                true,
            ),
            (
                "[CONTEXT SUMMARY]: summary",
                "x-initiator",
                "unknown",
                false,
            ),
            ("[CONTEXT SUMMARY]: summary", "x-initiator", "user", false),
            (
                "Continue the task",
                "x-switchyard-request-origin",
                "compaction",
                false,
            ),
            ("Continue the task", "x-initiator", "agent", false),
            ("Now fix the parser", "x-initiator", "unknown", true),
            (
                "Now fix the parser",
                "x-codex-turn-metadata",
                "invalid json",
                true,
            ),
            (
                "Now fix the parser",
                "x-codex-turn-metadata",
                r#"{"request_kind":"turn"}"#,
                true,
            ),
            (
                "Summarize the conversation",
                "x-codex-turn-metadata",
                r#"{"request_kind":"compaction"}"#,
                false,
            ),
        ] {
            let mut request = turn_request(false);
            request.llm_request.messages = vec![Message::text(Role::User, text)];
            request
                .metadata
                .as_mut()
                .unwrap()
                .http_headers
                .get_or_insert_with(Default::default)
                .insert(header, value.parse().unwrap());
            assert_eq!(starts_user_turn(&request), expected, "{header}: {value}");
        }
    }
}
