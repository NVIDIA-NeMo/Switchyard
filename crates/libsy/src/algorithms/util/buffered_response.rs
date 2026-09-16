// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Buffer a model response for inspection and replay its original stream events.

use futures::StreamExt;
use switchyard_protocol::{
    AggLlmResponse, LlmClientError, LlmResponse, LlmResponseChunk, LlmResponseStreamEvent,
    Metadata, Response, ResponseAccumulator,
};

use crate::{LibsyError, Result};

/// A completed model response held while an algorithm inspects it.
pub(crate) struct BufferedResponse {
    /// Original stream events, including response IDs, signed thinking, and provider
    /// extensions. Replaying these preserves data lost when synthesizing events from `agg`.
    events: Option<Vec<LlmResponseStreamEvent>>,
    /// Complete response for inspection and usage accounting. For a non-streaming reply,
    /// this is the original response with its preserved provider data.
    pub(crate) agg: AggLlmResponse,
    metadata: Option<Metadata>,
    upstream_headers: http::HeaderMap,
}

impl BufferedResponse {
    /// Replay the buffered stream events or return the original non-streaming response.
    pub(crate) fn into_response(self) -> Response {
        let llm_response = match self.events {
            Some(events) => {
                LlmResponse::Stream(Box::pin(futures::stream::iter(events.into_iter().map(Ok))))
            }
            None => LlmResponse::Agg(self.agg),
        };
        Response {
            llm_response,
            metadata: self.metadata,
            upstream_headers: self.upstream_headers,
        }
    }
}

/// Consume the response to completion before replaying any events. Stream item errors
/// and error chunks become typed client-call errors, as in [`LlmResponse::into_agg`].
pub(crate) async fn buffer_response(
    executor: &str,
    response: Response,
) -> Result<BufferedResponse> {
    let metadata = response.metadata;
    let upstream_headers = response.upstream_headers;
    match response.llm_response {
        LlmResponse::Agg(agg) => Ok(BufferedResponse {
            events: None,
            agg,
            metadata,
            upstream_headers,
        }),
        LlmResponse::Stream(mut stream) => {
            let mut events = Vec::new();
            let mut accumulator = ResponseAccumulator::new();
            while let Some(item) = stream.next().await {
                let event =
                    item.map_err(|source| LibsyError::client_call(executor.to_string(), source))?;
                for chunk in event.normalized() {
                    let failure = match chunk {
                        LlmResponseChunk::DecodeError { message } => {
                            Some(LlmClientError::ResponseTranslation(message.clone()))
                        }
                        LlmResponseChunk::StreamError { message } => {
                            Some(LlmClientError::UpstreamHttp {
                                status: http::StatusCode::BAD_GATEWAY,
                                body: message.clone(),
                            })
                        }
                        chunk => {
                            accumulator.push(chunk.clone());
                            None
                        }
                    };
                    if let Some(source) = failure {
                        return Err(LibsyError::client_call(executor.to_string(), source));
                    }
                }
                events.push(event);
            }
            Ok(BufferedResponse {
                events: Some(events),
                agg: accumulator.finish(),
                metadata,
                upstream_headers,
            })
        }
    }
}
