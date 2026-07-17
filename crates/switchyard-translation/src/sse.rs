// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal SSE frame parser for decoding streamed provider responses.
//!
//! One copy backs all neutral-IR stream decoding ([`decode_stream`](crate::decode_stream)).
//! Errors are boxed `std::error::Error`s — the item error type of a streamed
//! response — so this module stays free of any HTTP client or server types.

use serde_json::Value;

use crate::WireFormat;

/// Boxed, thread-safe error carried by a streamed item.
pub(crate) type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// The terminal SSE marker for `format`, if any. OpenAI Chat and Responses use
/// `[DONE]`; Anthropic ends on its `message_stop` event with no marker.
#[inline]
pub(crate) fn done_marker(format: WireFormat) -> Option<&'static str> {
    match format {
        WireFormat::OpenAiChat | WireFormat::OpenAiResponses => Some("[DONE]"),
        WireFormat::AnthropicMessages => None,
    }
}

pub(crate) fn parse_json_sse_frame(
    frame: &str,
    done_marker: Option<&str>,
) -> Result<Option<Value>, BoxError> {
    let data = frame
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with(':'))
        .filter_map(|line| line.strip_prefix("data: ").map(|l| l.to_string()))
        .fold(String::new(), |mut a, b| {
            a.reserve(b.len() + 1);
            a.push_str(&b);
            a.push('\n');
            a
        });
    let data = data.trim_end();

    // No data payload, or the terminal marker: nothing to decode.
    if data.is_empty() || done_marker.is_some_and(|marker| data == marker) {
        return Ok(None);
    }
    let value = serde_json::from_str::<Value>(data)?;
    Ok(Some(value))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    const DONE: Option<&str> = Some("[DONE]");

    #[test]
    fn parses_a_data_line_as_json() -> Result<(), BoxError> {
        let value =
            parse_json_sse_frame("data: {\"text\":\"hi\"}\n", DONE)?.ok_or("expected a payload")?;
        assert_eq!(value, json!({"text": "hi"}));
        Ok(())
    }

    #[test]
    fn ignores_comment_and_non_data_fields() -> Result<(), BoxError> {
        // Only `data:` fields contribute; comments (`:`) and other fields (`event:`) are dropped.
        let frame = ": keep-alive\nevent: message\ndata: {\"n\":1}\n";
        let value = parse_json_sse_frame(frame, DONE)?.ok_or("expected a payload")?;
        assert_eq!(value, json!({"n": 1}));
        Ok(())
    }

    #[test]
    fn done_marker_yields_no_payload() -> Result<(), BoxError> {
        assert!(parse_json_sse_frame("data: [DONE]\n", DONE)?.is_none());
        Ok(())
    }

    #[test]
    fn frame_without_data_yields_no_payload() -> Result<(), BoxError> {
        // A comment-only frame carries no data payload.
        assert!(parse_json_sse_frame(": keep-alive\n", DONE)?.is_none());
        // An empty frame likewise decodes to nothing.
        assert!(parse_json_sse_frame("", DONE)?.is_none());
        Ok(())
    }

    #[test]
    fn marker_is_only_terminal_when_configured() {
        // Without a configured marker, `[DONE]` is treated as (invalid) JSON data.
        assert!(parse_json_sse_frame("data: [DONE]\n", None).is_err());
    }

    #[test]
    fn invalid_json_is_an_error() {
        assert!(parse_json_sse_frame("data: {not json}\n", DONE).is_err());
    }

    #[test]
    fn openai_formats_terminate_on_done() {
        assert_eq!(done_marker(WireFormat::OpenAiChat), Some("[DONE]"));
        assert_eq!(done_marker(WireFormat::OpenAiResponses), Some("[DONE]"));
    }

    #[test]
    fn anthropic_has_no_done_marker() {
        assert_eq!(done_marker(WireFormat::AnthropicMessages), None);
    }
}
