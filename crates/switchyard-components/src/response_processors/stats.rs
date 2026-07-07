// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Response-side stats processor.

use async_stream::try_stream;
use futures_util::StreamExt;
use switchyard_core::{BoxResponseStream, ChatResponse, ProxyContext, Result};

use crate::stats::{
    gemini_usage_from_body, openai_chat_usage_from_stream_event,
    openai_responses_usage_from_stream_event, selected_stats_model, selected_stats_tier,
    usage_from_body, AnthropicStreamUsage, GeminiStreamUsage, PrefixProbe, StatsAccumulator,
    StatsBackendLatency, StatsRequestStart, TokenUsage,
};

/// Records token usage, total latency, and routing overhead.
#[derive(Clone, Debug)]
pub struct StatsResponseProcessor {
    accumulator: StatsAccumulator,
}

impl StatsResponseProcessor {
    /// Creates a response processor sharing the supplied accumulator.
    pub fn new(accumulator: StatsAccumulator) -> Self {
        Self { accumulator }
    }

    /// Returns the shared accumulator.
    pub fn accumulator(&self) -> &StatsAccumulator {
        &self.accumulator
    }

    /// Records response usage and wraps streams so usage is captured on completion.
    pub async fn process(
        &self,
        ctx: &mut ProxyContext,
        response: ChatResponse,
    ) -> Result<ChatResponse> {
        let model = selected_stats_model(ctx, None);
        let tier = selected_stats_tier(ctx);
        let started_at = ctx.get::<StatsRequestStart>().copied();
        let backend_latency = ctx.get::<StatsBackendLatency>().copied();
        // Switch-aware: eligible only against the prefixes this model has already seen.
        let cache_eligible = ctx
            .get::<PrefixProbe>()
            .map(|probe| self.accumulator.prefix_eligibility(&model, probe))
            .unwrap_or(0.0);

        match response {
            ChatResponse::OpenAiCompletion(response) => {
                record_usage(
                    &self.accumulator,
                    &model,
                    usage_from_body(response.body()),
                    started_at,
                    backend_latency,
                    tier.as_deref(),
                    cache_eligible,
                )?;
                Ok(ChatResponse::OpenAiCompletion(response))
            }
            ChatResponse::OpenAiResponsesCompletion(response) => {
                record_usage(
                    &self.accumulator,
                    &model,
                    usage_from_body(response.body()),
                    started_at,
                    backend_latency,
                    tier.as_deref(),
                    cache_eligible,
                )?;
                Ok(ChatResponse::OpenAiResponsesCompletion(response))
            }
            ChatResponse::AnthropicCompletion(response) => {
                record_usage(
                    &self.accumulator,
                    &model,
                    usage_from_body(response.body()),
                    started_at,
                    backend_latency,
                    tier.as_deref(),
                    cache_eligible,
                )?;
                Ok(ChatResponse::AnthropicCompletion(response))
            }
            ChatResponse::GeminiCompletion(response) => {
                record_usage(
                    &self.accumulator,
                    &model,
                    gemini_usage_from_body(response.body()),
                    started_at,
                    backend_latency,
                    tier.as_deref(),
                    cache_eligible,
                )?;
                Ok(ChatResponse::GeminiCompletion(response))
            }
            ChatResponse::OpenAiStream(stream) => {
                Ok(ChatResponse::OpenAiStream(wrap_openai_chat_stream(
                    stream,
                    self.accumulator.clone(),
                    model,
                    started_at,
                    backend_latency,
                    tier,
                    cache_eligible,
                )))
            }
            ChatResponse::OpenAiResponsesStream(stream) => Ok(ChatResponse::OpenAiResponsesStream(
                wrap_openai_responses_stream(
                    stream,
                    self.accumulator.clone(),
                    model,
                    started_at,
                    backend_latency,
                    tier,
                    cache_eligible,
                ),
            )),
            ChatResponse::AnthropicStream(stream) => {
                Ok(ChatResponse::AnthropicStream(wrap_anthropic_stream(
                    stream,
                    self.accumulator.clone(),
                    model,
                    started_at,
                    backend_latency,
                    tier,
                    cache_eligible,
                )))
            }
            ChatResponse::GeminiStream(stream) => {
                Ok(ChatResponse::GeminiStream(wrap_gemini_stream(
                    stream,
                    self.accumulator.clone(),
                    model,
                    started_at,
                    backend_latency,
                    tier,
                    cache_eligible,
                )))
            }
        }
    }
}

fn wrap_openai_chat_stream(
    mut stream: BoxResponseStream,
    accumulator: StatsAccumulator,
    model: String,
    started_at: Option<StatsRequestStart>,
    backend_latency: Option<StatsBackendLatency>,
    tier: Option<String>,
    cache_eligible: f64,
) -> BoxResponseStream {
    Box::pin(try_stream! {
        let mut committed = false;
        while let Some(event) = stream.next().await {
            let event = event?;
            if !committed {
                if let Some(usage) = openai_chat_usage_from_stream_event(&event) {
                    log_stream_record_result(record_usage(
                        &accumulator,
                        &model,
                        usage,
                        started_at,
                        backend_latency,
                        tier.as_deref(),
                        cache_eligible,
                    ), &model);
                    committed = true;
                }
            }
            yield event;
        }
    })
}

fn wrap_openai_responses_stream(
    mut stream: BoxResponseStream,
    accumulator: StatsAccumulator,
    model: String,
    started_at: Option<StatsRequestStart>,
    backend_latency: Option<StatsBackendLatency>,
    tier: Option<String>,
    cache_eligible: f64,
) -> BoxResponseStream {
    Box::pin(try_stream! {
        let mut committed = false;
        while let Some(event) = stream.next().await {
            let event = event?;
            if !committed {
                if let Some(usage) = openai_responses_usage_from_stream_event(&event) {
                    log_stream_record_result(record_usage(
                        &accumulator,
                        &model,
                        usage,
                        started_at,
                        backend_latency,
                        tier.as_deref(),
                        cache_eligible,
                    ), &model);
                    committed = true;
                }
            }
            yield event;
        }
    })
}

fn wrap_anthropic_stream(
    mut stream: BoxResponseStream,
    accumulator: StatsAccumulator,
    model: String,
    started_at: Option<StatsRequestStart>,
    backend_latency: Option<StatsBackendLatency>,
    tier: Option<String>,
    cache_eligible: f64,
) -> BoxResponseStream {
    Box::pin(try_stream! {
        let mut stream_usage = AnthropicStreamUsage::default();
        while let Some(event) = stream.next().await {
            let event = event?;
            if let Some(usage) = stream_usage.observe(&event) {
                log_stream_record_result(record_usage(
                    &accumulator,
                    &model,
                    usage,
                    started_at,
                    backend_latency,
                    tier.as_deref(),
                    cache_eligible,
                ), &model);
            }
            yield event;
        }
    })
}

fn wrap_gemini_stream(
    mut stream: BoxResponseStream,
    accumulator: StatsAccumulator,
    model: String,
    started_at: Option<StatsRequestStart>,
    backend_latency: Option<StatsBackendLatency>,
    tier: Option<String>,
    cache_eligible: f64,
) -> BoxResponseStream {
    Box::pin(try_stream! {
        let mut stream_usage = GeminiStreamUsage::default();
        while let Some(event) = stream.next().await {
            let event = event?;
            if let Some(usage) = stream_usage.observe(&event) {
                log_stream_record_result(record_usage(
                    &accumulator,
                    &model,
                    usage,
                    started_at,
                    backend_latency,
                    tier.as_deref(),
                    cache_eligible,
                ), &model);
            }
            yield event;
        }
    })
}

fn log_stream_record_result(result: Result<()>, model: &str) {
    if let Err(error) = result {
        tracing::warn!(
            error = %error,
            model = %model,
            "failed to record stream usage"
        );
    }
}

fn record_usage(
    accumulator: &StatsAccumulator,
    model: &str,
    mut usage: TokenUsage,
    started_at: Option<StatsRequestStart>,
    backend_latency: Option<StatsBackendLatency>,
    tier: Option<&str>,
    cache_eligible: f64,
) -> Result<()> {
    usage.cacheable_prompt_tokens = (usage.prompt_tokens as f64 * cache_eligible).round() as u64;
    let total_latency_ms = started_at.map(StatsRequestStart::elapsed_ms);
    let backend_latency_ms = backend_latency.map(StatsBackendLatency::as_millis_f64);
    let routing_overhead_ms =
        total_latency_ms
            .zip(backend_latency_ms)
            .map(|(total_latency_ms, backend_latency_ms)| {
                (total_latency_ms - backend_latency_ms).max(0.0)
            });
    accumulator.record_usage_after_success_attribution(
        model.to_string(),
        usage,
        total_latency_ms,
        routing_overhead_ms,
        tier,
    )
}
