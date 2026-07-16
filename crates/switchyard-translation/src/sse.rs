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

/// The outcome of parsing one SSE frame's data lines.
pub(crate) enum ParsedSseFrame {
    /// Frame contained a JSON payload.
    Json(Value),
    /// Frame contained the provider's terminal marker (e.g. `[DONE]`).
    Done,
    /// Frame had no data payload.
    Empty,
}

/// The terminal SSE marker for `format`, if any. OpenAI Chat and Responses use
/// `[DONE]`; Anthropic ends on its `message_stop` event with no marker.
pub(crate) fn done_marker(format: WireFormat) -> Option<&'static str> {
    match format {
        WireFormat::OpenAiChat | WireFormat::OpenAiResponses => Some("[DONE]"),
        WireFormat::AnthropicMessages => None,
    }
}

/// Drains one complete SSE frame from the buffer when a boundary is present.
pub(crate) fn drain_next_sse_frame(buffer: &mut Vec<u8>) -> Result<Option<String>, BoxError> {
    let Some((index, separator_len)) = next_sse_boundary(buffer) else {
        return Ok(None);
    };
    let frame = decode_sse_frame(&buffer[..index])?;
    buffer.drain(..index + separator_len);
    Ok(Some(frame))
}

/// Decodes one raw SSE frame as UTF-8.
pub(crate) fn decode_sse_frame(frame: &[u8]) -> Result<String, BoxError> {
    std::str::from_utf8(frame)
        .map(str::to_string)
        .map_err(|error| format!("stream emitted invalid UTF-8 frame: {error}").into())
}

/// Returns whether the buffer has any non-whitespace bytes.
pub(crate) fn has_non_whitespace_bytes(buffer: &[u8]) -> bool {
    buffer.iter().any(|byte| !byte.is_ascii_whitespace())
}

/// Parses data lines from one SSE frame into JSON, terminal, or empty states.
pub(crate) fn parse_json_sse_frame(
    frame: &str,
    done_marker: Option<&str>,
) -> Result<ParsedSseFrame, BoxError> {
    let mut data_lines = Vec::new();
    for line in frame.lines() {
        // SSE comments and blank lines do not contribute data.
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        if let Some(data) = line.strip_prefix("data:") {
            data_lines.push(data.trim_start().to_string());
        }
    }

    if data_lines.is_empty() {
        return Ok(ParsedSseFrame::Empty);
    }

    let data = data_lines.join("\n");
    if done_marker.is_some_and(|marker| data.trim() == marker) {
        return Ok(ParsedSseFrame::Done);
    }

    let value = serde_json::from_str::<Value>(&data)
        .map_err(|error| format!("stream emitted invalid JSON frame: {error}"))?;
    Ok(ParsedSseFrame::Json(value))
}

/// Finds the next CRLF or LF SSE frame boundary.
fn next_sse_boundary(buffer: &[u8]) -> Option<(usize, usize)> {
    match (find_bytes(buffer, b"\r\n\r\n"), find_bytes(buffer, b"\n\n")) {
        (Some(crlf), Some(lf)) if crlf < lf => Some((crlf, 4)),
        (Some(_), Some(lf)) => Some((lf, 2)),
        (Some(crlf), None) => Some((crlf, 4)),
        (None, Some(lf)) => Some((lf, 2)),
        (None, None) => None,
    }
}

/// Finds a byte needle inside a byte haystack.
fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    // Multi-byte UTF-8 split across network chunks should wait for a full frame.
    #[test]
    fn buffers_incomplete_utf8_until_a_complete_sse_frame_arrives() -> Result<(), BoxError> {
        let mut buffer = b"data: {\"text\":\"".to_vec();
        let multibyte = "é".as_bytes();
        buffer.extend_from_slice(&multibyte[..1]);
        assert!(drain_next_sse_frame(&mut buffer)?.is_none());

        buffer.extend_from_slice(&multibyte[1..]);
        buffer.extend_from_slice(b"\"}\n\n");

        let Some(frame) = drain_next_sse_frame(&mut buffer)? else {
            return Err("complete SSE frame should be drained".into());
        };
        let ParsedSseFrame::Json(value) = parse_json_sse_frame(&frame, None)? else {
            return Err("SSE frame should parse as JSON".into());
        };
        assert_eq!(value, json!({"text": "é"}));
        assert!(buffer.is_empty());
        Ok(())
    }

    #[test]
    fn recognizes_done_marker() -> Result<(), BoxError> {
        let ParsedSseFrame::Done = parse_json_sse_frame("data: [DONE]\n", Some("[DONE]"))? else {
            return Err("DONE frame should stop".into());
        };
        Ok(())
    }
}
