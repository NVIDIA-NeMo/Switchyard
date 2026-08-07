// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Response usage and full-turn latency metrics for routed requests.

use std::time::{Duration, Instant};

use futures_util::StreamExt;
use opentelemetry::{KeyValue, global};
use switchyard_protocol::{LlmResponse, LlmResponseChunk, Response, Usage};

use crate::SharedRoutingLog;
use crate::routing_log::RoutingLogContext;
use crate::stats::{StatsAccumulator, TokenUsage};

/// Observes a routed response without changing its aggregate or streaming contents.
pub(crate) fn observe(
    response: Response,
    model: &str,
    tier: Option<&str>,
    started: Instant,
    stats: StatsAccumulator,
    cache_eligible: f64,
    routing_log: Option<(SharedRoutingLog, RoutingLogContext)>,
) -> Response {
    let Response {
        llm_response,
        metadata,
    } = response;
    let model = model.to_string();
    let tier = tier
        .map(str::trim)
        .filter(|tier| !tier.is_empty())
        .map(str::to_string);

    let llm_response = match llm_response {
        LlmResponse::Agg(agg) => {
            record_terminal(
                &stats,
                &agg.usage,
                &model,
                tier.as_deref(),
                started,
                cache_eligible,
            );
            if let Some((log, context)) = routing_log {
                log.append(context, &model, tier.as_deref(), &agg.usage);
            }
            LlmResponse::Agg(agg)
        }
        LlmResponse::Stream(mut stream) => {
            let wrapped = async_stream::stream! {
                let mut latest_usage = None;
                while let Some(item) = stream.next().await {
                    let failed = match &item {
                        Err(_) => true,
                        Ok(event) => event.normalized().iter().any(|chunk| {
                            matches!(
                                chunk,
                                LlmResponseChunk::StreamError { .. }
                                    | LlmResponseChunk::DecodeError { .. }
                            )
                        }),
                    };
                    if let Ok(event) = &item {
                        for chunk in event.normalized() {
                            if let LlmResponseChunk::Usage(usage) = chunk {
                                latest_usage = Some(usage.clone());
                            }
                        }
                    }
                    if failed {
                        record_stream_error(&stats, &model, tier.as_deref());
                    }
                    yield item;
                    if failed {
                        return;
                    }
                }
                let usage = latest_usage.unwrap_or_default();
                record_terminal(
                    &stats,
                    &usage,
                    &model,
                    tier.as_deref(),
                    started,
                    cache_eligible,
                );
                if let Some((log, context)) = routing_log {
                    log.append(context, &model, tier.as_deref(), &usage);
                }
            };
            LlmResponse::Stream(Box::pin(wrapped))
        }
    };

    Response {
        llm_response,
        metadata,
    }
}

// Records a terminal stream failure after the routed call was already counted.
fn record_stream_error(stats: &StatsAccumulator, model: &str, tier: Option<&str>) {
    stats.record_stream_error(model, tier);
    global::meter("switchyard")
        .u64_counter("switchyard.errors")
        .build()
        .add(1, &attributes(model, tier));
}

pub(crate) fn token_usage(usage: &Usage) -> TokenUsage {
    let cached_tokens = usage.cached_input_tokens().unwrap_or(0);
    let cache_creation_tokens = usage.cache_creation_input_tokens().unwrap_or(0);
    TokenUsage {
        prompt_tokens: usage
            .input_tokens
            .unwrap_or(0)
            .saturating_add(cached_tokens)
            .saturating_add(cache_creation_tokens),
        completion_tokens: usage.output_tokens.unwrap_or(0),
        cached_tokens,
        cache_creation_tokens,
        cacheable_prompt_tokens: 0,
        reasoning_tokens: usage.reasoning_tokens.unwrap_or(0),
    }
}

/// Records final usage and latency in both OpenTelemetry metrics and JSON stats.
fn record_terminal(
    stats: &StatsAccumulator,
    usage: &Usage,
    model: &str,
    tier: Option<&str>,
    started: Instant,
    cache_eligible: f64,
) {
    let total_latency = started.elapsed();
    record_usage(usage, model, tier);
    record_latency(model, tier, total_latency);
    let mut token_usage = token_usage(usage);
    token_usage.cacheable_prompt_tokens =
        (token_usage.prompt_tokens as f64 * cache_eligible).round() as u64;
    stats.record_usage(
        model,
        token_usage,
        total_latency.as_secs_f64() * 1_000.0,
        tier,
    );
}

fn attributes(model: &str, tier: Option<&str>) -> Vec<KeyValue> {
    let mut attributes = vec![KeyValue::new("model", model.to_string())];
    if let Some(tier) = tier {
        attributes.push(KeyValue::new("tier", tier.to_string()));
    }
    attributes
}

fn record_usage(usage: &Usage, model: &str, tier: Option<&str>) {
    let attributes = attributes(model, tier);
    let meter = global::meter("switchyard");
    let cached = usage.cached_input_tokens();
    let cache_creation = usage.cache_creation_input_tokens();

    if usage.input_tokens.is_some() || cached.is_some() || cache_creation.is_some() {
        let prompt =
            usage.input_tokens.unwrap_or(0) + cached.unwrap_or(0) + cache_creation.unwrap_or(0);
        meter
            .u64_counter("switchyard.prompt_tokens")
            .build()
            .add(prompt, &attributes);
    }
    for (name, value) in [
        ("switchyard.completion_tokens", usage.output_tokens),
        ("switchyard.cached_tokens", cached),
        ("switchyard.cache_creation_tokens", cache_creation),
        ("switchyard.reasoning_tokens", usage.reasoning_tokens),
    ] {
        if let Some(value) = value {
            meter.u64_counter(name).build().add(value, &attributes);
        }
    }
}

fn record_latency(model: &str, tier: Option<&str>, latency: Duration) {
    global::meter("switchyard")
        .f64_histogram("switchyard.total_latency_ms")
        .build()
        .record(latency.as_secs_f64() * 1000.0, &attributes(model, tier));
}
