// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use http::HeaderMap;
use switchyard_protocol::{ContentBlock, LlmRequest, Message, Role};

/// Marks harness-generated continuations, preserving an explicit IR annotation.
pub fn normalize_request_origin(request: &mut LlmRequest, headers: Option<&HeaderMap>) {
    let is_continuation = headers.is_some_and(|headers| {
        headers
            .get("x-initiator")
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.trim().eq_ignore_ascii_case("agent"))
            || headers
                .get("x-codex-turn-metadata")
                .and_then(|value| {
                    serde_json::from_slice::<serde_json::Value>(value.as_bytes()).ok()
                })
                .is_some_and(|metadata| metadata["request_kind"] == "compaction")
    });
    request.is_continuation |= is_continuation
        || request
            .messages
            .iter()
            .rev()
            .find(|message| !is_environment_update(message))
            .is_some_and(|message| message.role == Role::User && is_compaction_message(message));
}

fn is_environment_update(message: &Message) -> bool {
    message.role == Role::User
        && !message.content.is_empty()
        && message.content.iter().all(|block| {
            matches!(block, ContentBlock::Text { text } if text.trim()
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
    use crate::{WireFormat, decode_request, encode_request};
    use serde_json::json;

    #[test]
    fn normalizes_continuations_without_hiding_user_input() {
        for summary in [
            "Another language model started to solve this problem and produced a summary of its thinking process.",
            "This session is being continued from a previous conversation that ran out of context.",
            "prior context\n[CONTEXT COMPACTION \u{2014} REFERENCE ONLY] summary",
            "The conversation history before this point was compacted into the following summary:\n<summary>Summary</summary>",
            "Prior model work/tool state available.\n<summary>Summary</summary>",
            "<conversation-checkpoint>Summary</conversation-checkpoint>",
        ] {
            let mut body = json!({"messages": [
                {"role": "user", "content": [{"type": "text", "text": summary}, {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}]},
                {"role": "user", "content": "<environment_context>updated</environment_context>"}
            ]});
            let request = decode_request(WireFormat::OpenAiChat, &body).unwrap();
            assert!(request.is_continuation, "{summary}");
            let encoded = encode_request(&request, WireFormat::OpenAiChat).unwrap();
            assert!(encoded.get("is_continuation").is_none());
            body["messages"]
                .as_array_mut()
                .unwrap()
                .push(json!({"role": "user", "content": "Now fix the parser"}));
            let followup = decode_request(WireFormat::OpenAiChat, &body).unwrap();
            assert!(!followup.is_continuation);
        }
        let summary = "prior context\n[context summary]: summary\n--- end of context summary \u{2014} respond to the message below, not the summary above ---";
        assert!(!is_hermes_summary(&format!(
            "{summary}\nNow fix the parser"
        )));
        assert!(is_hermes_summary(&format!(
            "{summary}\n[still in progress \u{2014} this is the active request, restated after the compaction boundary] Task"
        )));
        let body = json!({"input": [{"type": "compaction", "encrypted_content": "opaque"}]});
        let request = decode_request(WireFormat::OpenAiResponses, &body).unwrap();
        assert!(request.is_continuation);
        let mut request = switchyard_protocol::text_request(None, "Summarize this");
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-codex-turn-metadata",
            r#"{"request_kind":"compaction"}"#.parse().unwrap(),
        );
        normalize_request_origin(&mut request, Some(&headers));
        assert!(request.is_continuation);
    }
}
