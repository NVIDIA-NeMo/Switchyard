// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Encodes a neutral IR [`Response`] back into the inbound wire format —
//! buffered JSON or a framed SSE stream.

use async_stream::try_stream;
use axum::response::sse::Sse;
use futures_util::StreamExt;
use libsy_protocol::{AggLlmResponse, LlmResponseStream};
use serde_json::Value;
use switchyard_translation::{
    StreamCodecRegistry, StreamTranslationState, TranslationEngine, TranslationError,
    TranslationPolicy, WireFormat,
};

use crate::sse::{frame_stream, BoxError, SseFrameStream};

/// Encodes a buffered aggregate response into the target wire format's JSON body.
pub(crate) fn encode_buffered(
    engine: &TranslationEngine,
    policy: &TranslationPolicy,
    agg: &AggLlmResponse,
    target: WireFormat,
) -> Result<Value, TranslationError> {
    Ok(engine.encode_response(target, agg, policy)?.body)
}

/// Encodes a stream of IR chunks into a framed SSE response in the target format.
///
/// `requested_model` is exposed to the client as the response model (via the
/// stream state's `target_model`) so it sees the model it asked for rather than
/// the upstream id. The target stream codec is resolved once and reused per
/// chunk; terminal events (`message_stop` / `response.completed`) come from
/// `finish`.
pub(crate) fn encode_stream(
    chunks: LlmResponseStream,
    target: WireFormat,
    requested_model: Option<String>,
) -> Sse<SseFrameStream> {
    // Target is always a built-in wire format, so this lookup cannot fail; a
    // failure would surface as a single error frame rather than a panic.
    let codec = match StreamCodecRegistry::with_builtins().codec(target) {
        Ok(codec) => codec,
        Err(error) => {
            let message = error.to_string();
            let events = futures_util::stream::once(async move {
                Err::<Value, BoxError>(Box::new(std::io::Error::other(message)))
            });
            return frame_stream(events, target);
        }
    };

    let events = try_stream! {
        let mut state = StreamTranslationState::default();
        state.target = Some(target.into());
        state.target_model = requested_model;

        let mut chunks = chunks;
        while let Some(item) = chunks.next().await {
            let chunk = item?;
            for value in codec.encode_event(&mut state, chunk) {
                yield value;
            }
        }
        for value in codec.finish(&mut state) {
            yield value;
        }
    };

    frame_stream(events, target)
}
