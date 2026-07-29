// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Response encoding glue for libsy server endpoints.

use std::error::Error;

use axum::response::{IntoResponse, Response as HttpResponse};
use axum::Json;
use libsy::{LlmResponse, Response as AlgorithmResponse};
use switchyard_translation::{encode_aggregated_response, encode_stream, WireFormat};

use crate::sse::frame_stream;

type BoxError = Box<dyn Error + Send + Sync>;

/// Encodes a libsy response into the endpoint's wire format.
pub(crate) fn into_http_response(
    response: AlgorithmResponse,
    target_format: WireFormat,
    requested_model: Option<String>,
) -> Result<HttpResponse, BoxError> {
    match response.llm_response {
        LlmResponse::Agg(response) => Ok(Json(encode_aggregated_response(
            &response,
            target_format,
            requested_model.as_deref(),
        )?)
        .into_response()),
        LlmResponse::Stream(stream) => Ok(frame_stream(
            encode_stream(stream, target_format, requested_model)?,
            target_format,
        )
        .into_response()),
    }
}
