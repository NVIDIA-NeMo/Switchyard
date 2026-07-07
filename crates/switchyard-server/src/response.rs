// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Response translation glue for Rust server endpoints.

use std::sync::Arc;

use axum::response::sse::Sse;
use serde_json::Value;
use switchyard_core::{BoxResponseStream, ChatResponse, Result, StreamEvent, SwitchyardError};
use switchyard_translation::{
    StreamTranslationState, TranslationEngine, TranslationPolicy, WireFormat,
};

use crate::sse::{frame_stream, SseFrameStream};

pub(crate) enum TranslatedResponse {
    /// Complete JSON response body ready for an Axum JSON response.
    Buffered(Value),
    /// Framed SSE response ready for Axum streaming.
    Stream(Sse<SseFrameStream>),
}

/// Translates a profile response into the endpoint's expected response shape.
pub(crate) fn translate_chain_response(
    response: ChatResponse,
    target_format: WireFormat,
    translation: Arc<TranslationEngine>,
    policy: TranslationPolicy,
) -> Result<TranslatedResponse> {
    match response {
        ChatResponse::OpenAiCompletion(response) => translate_buffered_response(
            response.into_body(),
            WireFormat::OpenAiChat,
            target_format,
            translation,
            &policy,
        ),
        ChatResponse::OpenAiResponsesCompletion(response) => translate_buffered_response(
            response.into_body(),
            WireFormat::OpenAiResponses,
            target_format,
            translation,
            &policy,
        ),
        ChatResponse::AnthropicCompletion(response) => translate_buffered_response(
            response.into_body(),
            WireFormat::AnthropicMessages,
            target_format,
            translation,
            &policy,
        ),
        ChatResponse::GeminiCompletion(response) => translate_buffered_response(
            response.into_body(),
            WireFormat::GeminiGenerateContent,
            target_format,
            translation,
            &policy,
        ),
        ChatResponse::OpenAiStream(stream) => Ok(TranslatedResponse::Stream(
            translate_stream_response(stream, WireFormat::OpenAiChat, target_format, translation),
        )),
        ChatResponse::OpenAiResponsesStream(stream) => {
            Ok(TranslatedResponse::Stream(translate_stream_response(
                stream,
                WireFormat::OpenAiResponses,
                target_format,
                translation,
            )))
        }
        ChatResponse::AnthropicStream(stream) => {
            Ok(TranslatedResponse::Stream(translate_stream_response(
                stream,
                WireFormat::AnthropicMessages,
                target_format,
                translation,
            )))
        }
        ChatResponse::GeminiStream(stream) => {
            Ok(TranslatedResponse::Stream(translate_stream_response(
                stream,
                WireFormat::GeminiGenerateContent,
                target_format,
                translation,
            )))
        }
    }
}

fn translate_buffered_response(
    body: Value,
    source_format: WireFormat,
    target_format: WireFormat,
    translation: Arc<TranslationEngine>,
    policy: &TranslationPolicy,
) -> Result<TranslatedResponse> {
    if source_format == target_format {
        return Ok(TranslatedResponse::Buffered(body));
    }

    let output = translation
        .translate_response(source_format, target_format, &body, policy)
        .map_err(|error| {
            SwitchyardError::Other(format!(
                "failed to translate {source_format} response to {target_format}: {error}"
            ))
        })?;
    Ok(TranslatedResponse::Buffered(output.body))
}

fn translate_stream_response(
    mut stream: BoxResponseStream,
    source_format: WireFormat,
    target_format: WireFormat,
    translation: Arc<TranslationEngine>,
) -> Sse<SseFrameStream> {
    let events = async_stream::try_stream! {
        let mut translation_state = StreamTranslationState::new(source_format, target_format);
        while let Some(event) = futures_util::StreamExt::next(&mut stream).await {
            let event = event?;
            match event {
                StreamEvent::Json(value) => {
                    if source_format == target_format {
                        yield value;
                    } else {
                        for translated in translation.translate_event(
                            &mut translation_state,
                            source_format,
                            target_format,
                            &value,
                        ).map_err(|error| {
                            SwitchyardError::Other(format!(
                                "failed to translate {source_format} stream event to {target_format}: {error}"
                            ))
                        })? {
                            yield translated;
                        }
                    }
                }
                StreamEvent::Text(text) => {
                    yield Value::String(text);
                }
            }
        }

        if source_format != target_format {
            for translated in translation.finish_stream(&mut translation_state, target_format)
                .map_err(|error| {
                    SwitchyardError::Other(format!(
                        "failed to finish {target_format} stream translation: {error}"
                    ))
                })? {
                yield translated;
            }
        }
    };

    frame_stream(events, target_format)
}
