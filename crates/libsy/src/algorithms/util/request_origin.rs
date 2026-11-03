// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-origin signals and compatibility with harness-generated summaries.

use switchyard_protocol::{ContentBlock, Message, Request};

use super::affinity::{initiated_by_user, is_environment_update, is_user_turn};

pub(crate) fn is_new_user_turn(request: &Request) -> bool {
    let codex_compaction = request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.http_headers.as_ref())
        .and_then(|headers| headers.get("x-codex-turn-metadata"))
        .and_then(|value| serde_json::from_slice::<serde_json::Value>(value.as_bytes()).ok())
        .is_some_and(|metadata| metadata["request_kind"] == "compaction");
    if codex_compaction || initiated_by_user(request) == Some(false) {
        return false;
    }
    request
        .llm_request
        .messages
        .iter()
        .rev()
        .find(|message| !is_environment_update(message))
        .is_some_and(|message| is_user_turn(message) && !is_compaction_message(message))
}

fn is_compaction_message(message: &Message) -> bool {
    let mut has_summary = false;
    for block in &message.content {
        match block {
            ContentBlock::Image { .. } => {}
            _ if is_compaction_summary(block) => has_summary = true,
            _ => return false,
        }
    }
    has_summary
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
                || is_hermes_summary(&text)
        }
        _ => false,
    }
}

fn is_hermes_summary(text: &str) -> bool {
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
    use switchyard_protocol::{ImageSource, Role, WireFormat};

    #[test]
    fn structured_compaction_is_not_user_input() {
        let mut request = turn_request(false);
        for kind in ["compaction", "context_compaction", "unrecognized"] {
            request.llm_request.messages = vec![Message {
                role: Role::User,
                content: vec![
                    ContentBlock::Unknown {
                        provider: WireFormat::OpenAiResponses.into(),
                        raw: serde_json::json!({"type": kind, "encrypted_content": "opaque"}),
                    },
                    ContentBlock::Image {
                        source: ImageSource::Url {
                            url: "image.png".into(),
                            detail: None,
                        },
                    },
                ],
            }];
            assert_eq!(is_new_user_turn(&request), kind == "unrecognized");
        }
        request.llm_request.messages =
            vec![Message::text(Role::User, "Summarize the conversation")];
        for (header, value, expected) in [
            (
                "x-codex-turn-metadata",
                r#"{"request_kind":"compaction"}"#,
                false,
            ),
            ("x-initiator", "agent", false),
            ("x-initiator", "user", true),
        ] {
            let metadata = request.metadata.as_mut().unwrap();
            let headers = metadata.http_headers.get_or_insert_default();
            headers.clear();
            headers.insert(header, value.parse().unwrap());
            assert_eq!(is_new_user_turn(&request), expected);
        }
        request.llm_request.messages =
            vec![Message::text(Role::User, "[CONTEXT SUMMARY]: summary")];
        assert!(!is_new_user_turn(&request));
    }

    #[test]
    fn merged_hermes_summary_preserves_fresh_input() {
        let summary = "prior context\n[context compaction \u{2014} reference only] summary\n--- end of context summary \u{2014} respond to the message below, not the summary above ---";
        assert!(is_hermes_summary(summary));
        assert!(!is_hermes_summary(&format!(
            "{summary}\nNow fix the parser"
        )));
        assert!(is_hermes_summary(&format!(
            "{summary}\n[still in progress \u{2014} this is the active request, restated after the compaction boundary] Task"
        )));
    }
}
