// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use http::HeaderMap;
use switchyard_protocol::{ContentBlock, LlmRequest, Message, Role};

/// This function detects current and historical compaction while preserving explicit annotations.
pub fn normalize_compaction(request: &mut LlmRequest, headers: Option<&HeaderMap>) {
    let is_compaction = headers
        .and_then(|headers| headers.get("x-codex-turn-metadata"))
        .and_then(|value| serde_json::from_slice::<serde_json::Value>(value.as_bytes()).ok())
        .is_some_and(|metadata| metadata["request_kind"] == "compaction");
    // Both scans skip inline instructions so they cannot hide a trigger or count as a summary.
    let messages = request
        .messages
        .iter()
        .filter(|message| !matches!(message.role, Role::System | Role::Developer));
    request.has_compaction |= messages
        .clone()
        .flat_map(|message| &message.content)
        .any(|block| match block {
            ContentBlock::Text { text, .. } => {
                let text = text.to_lowercase();
                text.contains("session is being continued") || has_hermes_marker(&text)
            }
            _ => false,
        });
    request.is_compaction |= is_compaction
        || messages
            .rev()
            .find(|message| !is_environment_update(message))
            .is_some_and(|message| message.role == Role::User && is_compaction_message(message));
}

fn is_environment_update(message: &Message) -> bool {
    message.role == Role::User
        && !message.content.is_empty()
        && message.content.iter().all(|block| {
            matches!(block, ContentBlock::Text { text, .. } if text.trim()
            .strip_prefix("<environment_context>")
            .and_then(|context| context.split_once("</environment_context>"))
            .is_some_and(|(_, rest)| rest.trim().is_empty()))
        })
}

fn is_compaction_message(message: &Message) -> bool {
    let mut has_summary = false;
    for block in &message.content {
        match block {
            ContentBlock::Image { .. } => {}
            _ if is_compaction_block(block) => has_summary = true,
            _ => return false,
        }
    }
    has_summary
}

fn is_compaction_block(block: &ContentBlock) -> bool {
    match block {
        ContentBlock::Unknown { raw, .. } => matches!(
            raw.get("type").and_then(serde_json::Value::as_str),
            Some("compaction" | "context_compaction" | "compaction_trigger")
        ),
        ContentBlock::Text { text, .. } => {
            let text = text.trim().to_ascii_lowercase();
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

const HERMES_REPLAY: &str = "[still in progress \u{2014} this is the active request, restated after the compaction boundary";

fn is_hermes_summary(text: &str) -> bool {
    const END: &str = "--- end of context summary \u{2014} respond to the message below, not the summary above ---";
    if let Some((_, tail)) = text.rsplit_once(END)
        && !tail.trim().is_empty()
        && !tail.trim_start().starts_with(HERMES_REPLAY)
    {
        return false;
    }
    has_hermes_marker(text)
        || text
            .lines()
            .any(|line| line.trim_start().starts_with("[context summary]:"))
}

fn has_hermes_marker(text: &str) -> bool {
    text.lines().any(|line| {
        let line = line.trim_start();
        line.starts_with("[context compaction \u{2014} reference only]")
            || line.starts_with(HERMES_REPLAY)
    })
}

#[cfg(test)]
mod tests {
    use crate::{WireFormat, decode_request};
    use serde_json::json;

    #[test]
    fn summary_continues_but_fresh_user_input_does_not() {
        let mut body = json!({"messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "prior context\n[CONTEXT COMPACTION \u{2014} REFERENCE ONLY] summary"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
            ]},
            {"role": "user", "content": "<environment_context>updated</environment_context>"}
        ]});
        let request = decode_request(WireFormat::OpenAiChat, &body).unwrap();
        assert!(request.is_compaction && request.has_compaction);
        body["messages"]
            .as_array_mut()
            .unwrap()
            .push(json!({"role": "user", "content": "Now fix the parser"}));
        let followup = decode_request(WireFormat::OpenAiChat, &body).unwrap();
        assert!(!followup.is_compaction && followup.has_compaction);
    }
}
