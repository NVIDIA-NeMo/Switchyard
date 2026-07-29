// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Response usage and full-turn latency metrics for routed requests.

use std::time::Instant;

use futures_util::StreamExt;
use libsy::{LlmResponse, LlmResponseChunk, Response, Usage};
use opentelemetry::{global, KeyValue};

/// Observes a routed response without changing its aggregate or streaming contents.
pub(crate) fn observe(
    response: Response,
    model: &str,
    tier: Option<&str>,
    started: Instant,
) -> Response {
    let Response {
        llm_response,
        metadata,
    } = response;
    let model = model.to_string();
    let tier = tier.map(str::to_string);

    let llm_response = match llm_response {
        LlmResponse::Agg(agg) => {
            record_usage(&agg.usage, &model, tier.as_deref());
            record_latency(&model, tier.as_deref(), started);
            LlmResponse::Agg(agg)
        }
        LlmResponse::Stream(mut stream) => {
            let wrapped = async_stream::stream! {
                let mut latest_usage = None;
                while let Some(item) = stream.next().await {
                    let failed = matches!(
                        &item,
                        Err(_)
                            | Ok(
                                LlmResponseChunk::StreamError { .. }
                                    | LlmResponseChunk::DecodeError { .. }
                            )
                    );
                    if let Ok(LlmResponseChunk::Usage(usage)) = &item {
                        latest_usage = Some(usage.clone());
                    }
                    yield item;
                    if failed {
                        return;
                    }
                }
                if let Some(usage) = latest_usage {
                    record_usage(&usage, &model, tier.as_deref());
                }
                record_latency(&model, tier.as_deref(), started);
            };
            LlmResponse::Stream(Box::pin(wrapped))
        }
    };

    Response {
        llm_response,
        metadata,
    }
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

fn record_latency(model: &str, tier: Option<&str>, started: Instant) {
    global::meter("switchyard")
        .f64_histogram("switchyard.total_latency_ms")
        .build()
        .record(
            started.elapsed().as_secs_f64() * 1000.0,
            &attributes(model, tier),
        );
}
