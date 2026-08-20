// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SSE framing helpers for OpenAI, Anthropic, and Responses endpoints.

use std::convert::Infallible;

use axum::response::sse::{Event, Sse};
use futures_util::Stream;
use serde_json::{Value, json};
use switchyard_translation::{RawEventStream, StreamOutcome, WireFormat};

/// Boxed stream type accepted by Axum's SSE response wrapper.
pub(crate) type SseFrameStream =
    std::pin::Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>>;

/// Converts translated JSON events into endpoint-specific SSE frames.
///
/// `outcome` reports whether the translated stream ended on an in-band upstream
/// error. A failed stream must not be closed with a success sentinel, so the
/// OpenAI Chat `[DONE]` marker is emitted only for a clean finish.
pub(crate) fn frame_stream(
    stream: RawEventStream,
    target_format: WireFormat,
    outcome: StreamOutcome,
) -> Sse<SseFrameStream> {
    let framed = async_stream::stream! {
        let mut stream = stream;
        let mut failed = false;
        while let Some(item) = futures_util::StreamExt::next(&mut stream).await {
            let event = match item {
                Ok(value) => match frame_event(target_format, value) {
                    Ok(event) => event,
                    Err(error) => {
                        failed = true;
                        error_event(target_format, error.to_string())
                    }
                },
                Err(error) => {
                    tracing::warn!(error = %error, "stream iteration failed");
                    failed = true;
                    error_event(target_format, error.to_string())
                }
            };
            yield Ok(event);
            if failed {
                break;
            }
        }

        // An upstream in-band error terminates the stream inside the translation
        // layer, which reports it through `outcome` rather than a framing error.
        let upstream_errored = outcome.load(std::sync::atomic::Ordering::Acquire);
        if !failed && !upstream_errored && target_format == WireFormat::OpenAiChat {
            yield Ok(Event::default().data("[DONE]"));
        }
    };

    Sse::new(Box::pin(framed) as SseFrameStream)
}

fn frame_event(target_format: WireFormat, value: Value) -> Result<Event, axum::Error> {
    match target_format {
        WireFormat::OpenAiChat => Event::default().json_data(value),
        WireFormat::AnthropicMessages | WireFormat::OpenAiResponses => {
            let event_type = value
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("message")
                .to_string();
            Event::default().event(event_type).json_data(value)
        }
    }
}

fn error_event(target_format: WireFormat, message: String) -> Event {
    match target_format {
        WireFormat::OpenAiChat => Event::default().data(
            json!({
                "error": {
                    "message": message,
                    "type": "SwitchyardError",
                }
            })
            .to_string(),
        ),
        WireFormat::AnthropicMessages | WireFormat::OpenAiResponses => {
            Event::default().event("error").data(
                json!({
                    "type": "error",
                    "error": {
                        "message": message,
                        "type": "SwitchyardError",
                    }
                })
                .to_string(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{error::Error, io};

    use axum::{body::to_bytes, response::IntoResponse};
    use futures_util::stream;

    use super::*;

    type TestResult = Result<(), Box<dyn Error + Send + Sync>>;

    #[tokio::test]
    async fn stream_error_terminates_without_done_marker() -> TestResult {
        let failure: Box<dyn Error + Send + Sync> = Box::new(io::Error::other("boom"));
        let stream: RawEventStream = Box::pin(stream::iter(vec![
            Ok(json!({"id": "before"})),
            Err(failure),
            Ok(json!({"id": "after"})),
        ]));

        let response = frame_stream(
            stream,
            WireFormat::OpenAiChat,
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        )
        .into_response();
        let body = String::from_utf8(to_bytes(response.into_body(), usize::MAX).await?.to_vec())?;

        // A stream error is terminal: later chunks and success markers must not be emitted.
        assert!(body.contains("before"));
        assert!(body.contains("boom"));
        assert!(!body.contains("after"));
        assert!(!body.contains("[DONE]"));
        Ok(())
    }

    // An upstream in-band error ends the translated stream without a framing error,
    // so the outcome flag is the only signal that the turn failed. `[DONE]` would
    // otherwise tell the client a failed generation completed successfully.
    #[tokio::test]
    async fn upstream_in_band_error_suppresses_the_done_marker() -> TestResult {
        let stream: RawEventStream = Box::pin(stream::iter(vec![
            Ok(json!({"choices": [{"delta": {"content": "partial"}}]})),
            Ok(json!({"error": {"message": "upstream failed after stream start"}})),
        ]));
        let outcome = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));

        let response = frame_stream(stream, WireFormat::OpenAiChat, outcome).into_response();
        let body = String::from_utf8(to_bytes(response.into_body(), usize::MAX).await?.to_vec())?;

        assert!(body.contains("partial"));
        assert!(body.contains("upstream failed after stream start"));
        assert!(
            !body.contains("[DONE]"),
            "failed stream must not be closed with [DONE]:\n{body}"
        );
        Ok(())
    }

    // A clean stream still ends with the sentinel the OpenAI Chat contract requires.
    #[tokio::test]
    async fn clean_stream_still_emits_the_done_marker() -> TestResult {
        let stream: RawEventStream = Box::pin(stream::iter(vec![Ok(
            json!({"choices": [{"delta": {"content": "hello"}}]}),
        )]));
        let outcome = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

        let response = frame_stream(stream, WireFormat::OpenAiChat, outcome).into_response();
        let body = String::from_utf8(to_bytes(response.into_body(), usize::MAX).await?.to_vec())?;

        assert!(body.contains("hello"));
        assert!(
            body.contains("[DONE]"),
            "clean stream must end with [DONE]:\n{body}"
        );
        Ok(())
    }
}
