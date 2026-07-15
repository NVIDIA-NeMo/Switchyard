// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Response-side intake processor.

use std::sync::Arc;

use async_stream::try_stream;
use futures_util::StreamExt;
use switchyard_core::{BoxResponseStream, ChatResponse, ProxyContext, Result};

use crate::intake::{
    now_millis, HttpIntakeSink, IntakePayloadBuilder, IntakePayloadContext, IntakeRequestState,
    IntakeSink, IntakeSinkConfig, IntakeStreamCapture, IntakeStreamFormat, SubModelCalls,
};

/// Response processor that converts completed responses into intake payloads.
#[derive(Clone)]
pub struct IntakeResponseProcessor {
    /// Payload builder shared by buffered and streaming responses.
    builder: IntakePayloadBuilder,
    /// Sink used to deliver payloads without blocking response correctness.
    sink: Arc<dyn IntakeSink>,
}

impl IntakeResponseProcessor {
    /// Creates an intake response processor with an injected sink.
    pub fn new(config: IntakeSinkConfig, sink: Arc<dyn IntakeSink>) -> Self {
        Self {
            builder: IntakePayloadBuilder::new(config),
            sink,
        }
    }

    /// Creates an intake response processor backed by the default HTTP sink.
    pub fn with_http_sink(config: IntakeSinkConfig) -> Result<Self> {
        let sink = Arc::new(HttpIntakeSink::new(config.clone())?);
        Ok(Self::new(config, sink))
    }

    /// Returns the payload builder for tests and diagnostics.
    pub fn builder(&self) -> &IntakePayloadBuilder {
        &self.builder
    }

    /// Emits intake payloads for buffered responses or wraps streams for deferred emission.
    pub async fn process(
        &self,
        ctx: &mut ProxyContext,
        response: ChatResponse,
    ) -> Result<ChatResponse> {
        if ctx
            .get::<IntakeRequestState>()
            .map(IntakeRequestState::skipped)
            == Some(true)
        {
            return Ok(response);
        }

        // Routing sub-model records don't depend on the response body — the
        // router already made its call at request time — so emit them here,
        // gated by the same opt-in, instead of threading through the
        // buffered/streaming split below.
        self.emit_submodel_records(ctx, Arc::clone(&self.sink)).await;

        match response {
            ChatResponse::OpenAiCompletion(_)
            | ChatResponse::OpenAiResponsesCompletion(_)
            | ChatResponse::AnthropicCompletion(_) => {
                let payload = self.buffered_payload(ctx, &response);
                enqueue_payload(Arc::clone(&self.sink), payload).await;
                Ok(response)
            }
            ChatResponse::OpenAiStream(stream) => Ok(ChatResponse::OpenAiStream(wrap_stream(
                stream,
                self.builder.clone(),
                Arc::clone(&self.sink),
                IntakePayloadContext::from_proxy_context(ctx, None),
                IntakeStreamFormat::OpenAiChat,
            ))),
            ChatResponse::OpenAiResponsesStream(stream) => {
                Ok(ChatResponse::OpenAiResponsesStream(wrap_stream(
                    stream,
                    self.builder.clone(),
                    Arc::clone(&self.sink),
                    IntakePayloadContext::from_proxy_context(ctx, None),
                    IntakeStreamFormat::OpenAiResponses,
                )))
            }
            ChatResponse::AnthropicStream(stream) => {
                Ok(ChatResponse::AnthropicStream(wrap_stream(
                    stream,
                    self.builder.clone(),
                    Arc::clone(&self.sink),
                    IntakePayloadContext::from_proxy_context(ctx, None),
                    IntakeStreamFormat::Anthropic,
                )))
            }
        }
    }

    /// Flushes and shuts down the configured intake sink.
    pub async fn shutdown(&self) -> Result<()> {
        self.sink.shutdown().await
    }
}

impl IntakeResponseProcessor {
    /// Emits one anonymous record per routing sub-model call stashed on context.
    async fn emit_submodel_records(&self, ctx: &ProxyContext, sink: Arc<dyn IntakeSink>) {
        // Clone the calls out before the await so no context borrow is held
        // across it.
        let calls = match ctx.get::<SubModelCalls>() {
            Some(calls) if !calls.0.is_empty() => calls.0.clone(),
            _ => return,
        };
        let payload_ctx = IntakePayloadContext::from_proxy_context(ctx, Some(now_millis()));
        for (index, call) in calls.iter().enumerate() {
            let payload = self.builder.build_submodel_record(&payload_ctx, call, index);
            enqueue_payload(Arc::clone(&sink), payload).await;
        }
    }

    /// Builds a payload for a buffered response using request state from context.
    fn buffered_payload(
        &self,
        ctx: &ProxyContext,
        response: &ChatResponse,
    ) -> Result<serde_json::Value> {
        let payload_ctx = IntakePayloadContext::from_proxy_context(ctx, Some(now_millis()));
        self.builder
            .request_from_state(&payload_ctx)
            .and_then(|request| self.builder.build(&payload_ctx, request, response, false))
    }
}

// Wraps a response stream, mirrors all events to the caller, and emits intake
// after the upstream stream finishes.
fn wrap_stream(
    mut stream: BoxResponseStream,
    builder: IntakePayloadBuilder,
    sink: Arc<dyn IntakeSink>,
    mut payload_ctx: IntakePayloadContext,
    format: IntakeStreamFormat,
) -> BoxResponseStream {
    Box::pin(try_stream! {
        let mut capture = IntakeStreamCapture::new(format, payload_ctx.served_model.as_deref());
        while let Some(event) = stream.next().await {
            let event = event?;
            capture.observe(&event);
            yield event;
        }

        payload_ctx.ended_at_ms = Some(now_millis());
        let payload = capture
            .finish()
            .and_then(|openai_response| {
                builder
                    .request_from_state(&payload_ctx)
                    .and_then(|request| {
                        builder.build_from_openai_response_body(
                            &payload_ctx,
                            request,
                            openai_response,
                            true,
                        )
                    })
            });
        enqueue_payload(sink, payload).await;
    })
}

// Intake is deliberately fail-open: build and enqueue failures are logged but
// never replace the user-visible LLM response.
async fn enqueue_payload(sink: Arc<dyn IntakeSink>, payload: Result<serde_json::Value>) {
    match payload {
        Ok(payload) => {
            if let Err(error) = sink.enqueue(payload).await {
                tracing::warn!(
                    error = %error,
                    "failed to enqueue intake payload"
                );
            }
        }
        Err(error) => {
            tracing::warn!(
                error = %error,
                "failed to build intake payload"
            );
        }
    }
}

impl std::fmt::Debug for IntakeResponseProcessor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("IntakeResponseProcessor")
            .field("config", self.builder.config())
            .finish_non_exhaustive()
    }
}
