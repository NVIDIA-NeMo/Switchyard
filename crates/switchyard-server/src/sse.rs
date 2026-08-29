// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SSE framing helpers for OpenAI, Anthropic, and Responses endpoints.

use std::convert::Infallible;

use axum::response::sse::{Event, Sse};
use futures_util::Stream;
use serde_json::{Value, json};
use switchyard_protocol::LlmClientError;
use switchyard_translation::{RawEventStream, WireFormat};

/// Boxed stream type accepted by Axum's SSE response wrapper.
pub(crate) type SseFrameStream =
    std::pin::Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>>;

/// Converts translated JSON events into endpoint-specific SSE frames.
pub(crate) fn frame_stream(
    stream: RawEventStream,
    target_format: WireFormat,
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
                    // The full text stays in the log; only the client-facing copy is redacted.
                    tracing::warn!(error = %error, "stream iteration failed");
                    failed = true;
                    error_event(target_format, client_visible_stream_error(error.as_ref()))
                }
            };
            yield Ok(event);
            if failed {
                break;
            }
        }

        if !failed && target_format == WireFormat::OpenAiChat {
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

// A stream can fail after its response has begun, which is past the buffered `client_error`
// boundary. Transport and timeout sources are reqwest's, and they render the full request URL,
// including any credentials configured in its query string, so they get the same fixed messages
// the buffered path uses.
fn client_visible_stream_error(error: &(dyn std::error::Error + 'static)) -> String {
    match error.downcast_ref::<LlmClientError>() {
        Some(LlmClientError::Transport { .. }) => "upstream transport error".to_string(),
        Some(LlmClientError::Timeout { .. }) => "upstream request timed out".to_string(),
        _ => error.to_string(),
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

        let response = frame_stream(stream, WireFormat::OpenAiChat).into_response();
        let body = String::from_utf8(to_bytes(response.into_body(), usize::MAX).await?.to_vec())?;

        // A stream error is terminal: later chunks and success markers must not be emitted.
        assert!(body.contains("before"));
        assert!(body.contains("boom"));
        assert!(!body.contains("after"));
        assert!(!body.contains("[DONE]"));
        Ok(())
    }

    // A stream that fails after its response has begun is past the buffered `client_error`
    // boundary, so the redaction has to be repeated here.
    #[tokio::test]
    async fn stream_transport_error_hides_credential_bearing_upstream_url() -> TestResult {
        const UPSTREAM_URL: &str = "http://upstream.invalid/v1?key=CANARY_ADMIN_QUERY_KEY";
        let failure: Box<dyn Error + Send + Sync> = Box::new(LlmClientError::Transport {
            source: Box::new(io::Error::other(format!(
                "error sending request for url ({UPSTREAM_URL})"
            ))),
        });
        let stream: RawEventStream = Box::pin(stream::iter(vec![
            Ok(json!({"id": "before"})),
            Err(failure),
        ]));

        let response = frame_stream(stream, WireFormat::OpenAiChat).into_response();
        let body = String::from_utf8(to_bytes(response.into_body(), usize::MAX).await?.to_vec())?;

        assert!(body.contains("upstream transport error"));
        assert!(
            !body.contains("CANARY_ADMIN_QUERY_KEY"),
            "credential leaked in {body:?}"
        );
        assert!(
            !body.contains(UPSTREAM_URL),
            "upstream URL leaked in {body:?}"
        );
        Ok(())
    }

    // Errors that do not describe the request target keep their text.
    #[tokio::test]
    async fn stream_translation_error_keeps_its_message() -> TestResult {
        let failure: Box<dyn Error + Send + Sync> = Box::new(LlmClientError::ResponseTranslation(
            "unrecognized event type".to_string(),
        ));
        let stream: RawEventStream = Box::pin(stream::iter(vec![Err(failure)]));

        let response = frame_stream(stream, WireFormat::OpenAiChat).into_response();
        let body = String::from_utf8(to_bytes(response.into_body(), usize::MAX).await?.to_vec())?;

        assert!(body.contains("unrecognized event type"));
        Ok(())
    }
}
