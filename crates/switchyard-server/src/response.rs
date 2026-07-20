// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Response encoding glue for libsy server endpoints.

use std::error::Error;

use axum::response::sse::Sse;
use libsy::{LlmResponse, Response};
use serde_json::Value;
use switchyard_translation::{encode_aggregated_response, encode_stream, WireFormat};

use crate::sse::{frame_stream, SseFrameStream};

type BoxError = Box<dyn Error + Send + Sync>;

pub(crate) enum TranslatedResponse {
    /// Complete JSON response body ready for an Axum JSON response.
    Buffered(Value),
    /// Framed SSE response ready for Axum streaming.
    Stream(Sse<SseFrameStream>),
}

/// Encodes a libsy response into the endpoint's wire format.
pub(crate) fn translate_response(
    response: Response,
    target_format: WireFormat,
    requested_model: Option<String>,
) -> Result<TranslatedResponse, BoxError> {
    match response.llm_response {
        LlmResponse::Agg(response) => Ok(TranslatedResponse::Buffered(encode_aggregated_response(
            &response,
            target_format,
            requested_model.as_deref(),
        )?)),
        LlmResponse::Stream(stream) => Ok(TranslatedResponse::Stream(frame_stream(
            encode_stream(stream, target_format, requested_model)?,
            target_format,
        ))),
    }
}
