// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal SSE frame parser for decoding streamed provider responses.
//!
//! One copy backs all neutral-IR stream decoding ([`decode_stream`](crate::decode_stream)).
//! Errors are boxed `std::error::Error`s — the item error type of a streamed
//! response — so this module stays free of any HTTP client or server types.

use serde_json::Value;

/// Boxed, thread-safe error carried by a streamed item.
pub(crate) type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Parsed contents of one SSE frame.
pub(crate) enum SseFrame {
    Empty,
    Done,
    Data(Value),
}

/// The optional terminal SSE marker accepted for all supported wire formats.
/// Native Anthropic streams can end at `message_stop`, while compatible
/// providers may additionally send `[DONE]`.
pub(crate) const fn done_marker() -> &'static str {
    "[DONE]"
}

/// Value of a `data` field line; the space after the colon is optional framing.
fn data_field_value(line: &str) -> Option<String> {
    let value = match line.split_once(':') {
        Some(("data", value)) => value,
        None if line == "data" => "",
        _ => return None,
    };
    Some(value.strip_prefix(' ').unwrap_or(value).to_string())
}

pub(crate) fn parse_json_sse_frame(frame: &str, done_marker: &str) -> Result<SseFrame, BoxError> {
    let data = frame
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with(':'))
        .filter_map(data_field_value)
        .fold(String::new(), |mut a, b| {
            a.reserve(b.len() + 1);
            a.push_str(&b);
            a.push('\n');
            a
        });
    let data = data.trim_end();

    if data.is_empty() {
        return Ok(SseFrame::Empty);
    }
    if data == done_marker {
        return Ok(SseFrame::Done);
    }
    let value = serde_json::from_str::<Value>(data)?;
    Ok(SseFrame::Data(value))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    const DONE: &str = done_marker();

    #[test]
    fn parses_a_data_line_as_json() -> Result<(), BoxError> {
        let SseFrame::Data(value) = parse_json_sse_frame("data: {\"text\":\"hi\"}\n", DONE)? else {
            return Err("expected a payload".into());
        };
        assert_eq!(value, json!({"text": "hi"}));
        Ok(())
    }

    #[test]
    fn ignores_comment_and_non_data_fields() -> Result<(), BoxError> {
        // Only `data:` fields contribute; comments (`:`) and other fields (`event:`) are dropped.
        let frame = ": keep-alive\nevent: message\ndata: {\"n\":1}\n";
        let SseFrame::Data(value) = parse_json_sse_frame(frame, DONE)? else {
            return Err("expected a payload".into());
        };
        assert_eq!(value, json!({"n": 1}));
        Ok(())
    }

    #[test]
    fn parses_a_data_line_without_a_space_after_the_colon() -> Result<(), BoxError> {
        // The space after `data:` is optional framing, not part of the value.
        let SseFrame::Data(value) = parse_json_sse_frame("data:{\"text\":\"hi\"}\n", DONE)? else {
            return Err("expected a payload".into());
        };
        assert_eq!(value, json!({"text": "hi"}));
        Ok(())
    }

    #[test]
    fn done_marker_is_recognised_without_a_space() -> Result<(), BoxError> {
        assert!(matches!(
            parse_json_sse_frame("data:[DONE]\n", DONE)?,
            SseFrame::Done
        ));
        Ok(())
    }

    #[test]
    fn field_names_are_matched_exactly() -> Result<(), BoxError> {
        // `database:` must not be read as a `data` field.
        assert!(matches!(
            parse_json_sse_frame("database: {\"n\":1}\n", DONE)?,
            SseFrame::Empty
        ));
        Ok(())
    }

    #[test]
    fn done_marker_yields_no_payload() -> Result<(), BoxError> {
        assert!(matches!(
            parse_json_sse_frame("data: [DONE]\n", DONE)?,
            SseFrame::Done
        ));
        Ok(())
    }

    #[test]
    fn frame_without_data_yields_no_payload() -> Result<(), BoxError> {
        // A comment-only frame carries no data payload.
        assert!(matches!(
            parse_json_sse_frame(": keep-alive\n", DONE)?,
            SseFrame::Empty
        ));
        // An empty frame likewise decodes to nothing.
        assert!(matches!(parse_json_sse_frame("", DONE)?, SseFrame::Empty));
        Ok(())
    }

    #[test]
    fn invalid_json_is_an_error() {
        assert!(parse_json_sse_frame("data: {not json}\n", DONE).is_err());
    }
}
