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
    let frame: String = String::from_utf8_lossy(&buffer[..index]).to_string();
    buffer.drain(..index + separator_len);
    Ok(Some(frame))
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
        let Some(value) = parse_json_sse_frame(&frame, None)? else {
            return Err("SSE frame should parse as JSON".into());
        };
        assert_eq!(value, json!({"text": "é"}));
        assert!(buffer.is_empty());
        Ok(())
    }

    #[test]
    fn recognizes_done_marker() -> Result<(), BoxError> {
        // The terminal marker carries no JSON payload.
        if parse_json_sse_frame("data: [DONE]\n", Some("[DONE]"))?.is_some() {
            return Err("DONE frame should not yield a JSON value".into());
        }
        Ok(())
    }
}
