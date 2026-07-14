// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Records provider token usage for buffered and streaming responses.
//!
//! Buffered responses carry usage in the body and can be recorded synchronously.
//! Streaming responses only emit usage inside a later event, so the response
//! stream is wrapped and usage is recorded when the first usage-bearing event
//! flows through — no buffering, no added latency, and the client still sees
//! every event.

use std::time::Instant;

use async_stream::try_stream;
use futures_util::StreamExt;
use switchyard_components::stats::{
    openai_chat_usage_from_stream_event, openai_responses_usage_from_stream_event, usage_from_body,
    AnthropicStreamUsage, StatsAccumulator, TokenUsage,
};
use switchyard_core::{BoxResponseStream, ChatResponse, Result};

/// Records usage from a buffered response, or wraps a streaming response so
/// usage is recorded when its usage-bearing event arrives.
pub(crate) fn record_usage_or_wrap_stream(
    stats: &StatsAccumulator,
    model: &str,
    tier: Option<&str>,
    profile_started_at: Instant,
    backend_latency_ms: f64,
    response: ChatResponse,
) -> Result<ChatResponse> {
    match response {
        ChatResponse::OpenAiCompletion(_)
        | ChatResponse::OpenAiResponsesCompletion(_)
        | ChatResponse::AnthropicCompletion(_) => {
            let usage = response.body().map(usage_from_body).unwrap_or_default();
            record_usage(
                stats,
                model,
                tier,
                profile_started_at,
                backend_latency_ms,
                usage,
            )?;
            Ok(response)
        }
        ChatResponse::OpenAiStream(stream) => Ok(ChatResponse::OpenAiStream(wrap_openai_chat(
            stream,
            stats.clone(),
            model.to_string(),
            tier.map(str::to_string),
            profile_started_at,
            backend_latency_ms,
        ))),
        ChatResponse::OpenAiResponsesStream(stream) => {
            Ok(ChatResponse::OpenAiResponsesStream(wrap_openai_responses(
                stream,
                stats.clone(),
                model.to_string(),
                tier.map(str::to_string),
                profile_started_at,
                backend_latency_ms,
            )))
        }
        ChatResponse::AnthropicStream(stream) => Ok(ChatResponse::AnthropicStream(wrap_anthropic(
            stream,
            stats.clone(),
            model.to_string(),
            tier.map(str::to_string),
            profile_started_at,
            backend_latency_ms,
        ))),
    }
}

fn wrap_openai_chat(
    mut stream: BoxResponseStream,
    stats: StatsAccumulator,
    model: String,
    tier: Option<String>,
    profile_started_at: Instant,
    backend_latency_ms: f64,
) -> BoxResponseStream {
    Box::pin(try_stream! {
        let mut committed = false;
        while let Some(event) = stream.next().await {
            let event = event?;
            if !committed {
                if let Some(usage) = openai_chat_usage_from_stream_event(&event) {
                    log_stream_record_result(
                        record_usage(
                            &stats,
                            &model,
                            tier.as_deref(),
                            profile_started_at,
                            backend_latency_ms,
                            usage,
                        ),
                        &model,
                    );
                    committed = true;
                }
            }
            yield event;
        }
    })
}

fn wrap_openai_responses(
    mut stream: BoxResponseStream,
    stats: StatsAccumulator,
    model: String,
    tier: Option<String>,
    profile_started_at: Instant,
    backend_latency_ms: f64,
) -> BoxResponseStream {
    Box::pin(try_stream! {
        let mut committed = false;
        while let Some(event) = stream.next().await {
            let event = event?;
            if !committed {
                if let Some(usage) = openai_responses_usage_from_stream_event(&event) {
                    log_stream_record_result(
                        record_usage(
                            &stats,
                            &model,
                            tier.as_deref(),
                            profile_started_at,
                            backend_latency_ms,
                            usage,
                        ),
                        &model,
                    );
                    committed = true;
                }
            }
            yield event;
        }
    })
}

fn wrap_anthropic(
    mut stream: BoxResponseStream,
    stats: StatsAccumulator,
    model: String,
    tier: Option<String>,
    profile_started_at: Instant,
    backend_latency_ms: f64,
) -> BoxResponseStream {
    Box::pin(try_stream! {
        let mut stream_usage = AnthropicStreamUsage::default();
        while let Some(event) = stream.next().await {
            let event = event?;
            if let Some(usage) = stream_usage.observe(&event) {
                log_stream_record_result(
                    record_usage(
                        &stats,
                        &model,
                        tier.as_deref(),
                        profile_started_at,
                        backend_latency_ms,
                        usage,
                    ),
                    &model,
                );
            }
            yield event;
        }
    })
}

fn record_usage(
    stats: &StatsAccumulator,
    model: &str,
    tier: Option<&str>,
    profile_started_at: Instant,
    backend_latency_ms: f64,
    usage: TokenUsage,
) -> Result<()> {
    let total_latency_ms = profile_started_at.elapsed().as_secs_f64() * 1000.0;
    let routing_overhead_ms = (total_latency_ms - backend_latency_ms).max(0.0);
    stats.record_usage_after_success_attribution(
        model.to_string(),
        usage,
        Some(total_latency_ms),
        Some(routing_overhead_ms),
        tier,
    )
}

fn log_stream_record_result(result: Result<()>, model: &str) {
    if let Err(error) = result {
        tracing::warn!(
            error = %error,
            model = %model,
            "failed to record stream usage",
        );
    }
}
