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

    let llm_response = match llm_response {
        LlmResponse::Agg(agg) => {
            record_terminal(&stats, &agg.usage, &model, started, cache_eligible);
            if let Some((log, context)) = routing_log {
                log.append(context, &model, None, &agg.usage);
            }
            LlmResponse::Agg(agg)
        }
        LlmResponse::Stream(mut stream) => {
            let wrapped = async_stream::stream! {
                let mut latest_usage = None;
                let mut terminal_seen = false;
                let mut recorded = false;
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
                            match chunk {
                                LlmResponseChunk::Usage(usage) => {
                                    latest_usage = Some(usage.clone());
                                }
                                LlmResponseChunk::MessageStop { .. } => {
                                    terminal_seen = true;
                                }
                                _ => {}
                            }
                        }
                    }
                    if failed {
                        record_stream_error(&stats, &model);
                    }
                    // Responses clients may stop polling immediately after the terminal event.
                    // Commit first when that event already carries the final usage.
                    if !failed && !recorded && terminal_seen
                        && let Some(usage) = latest_usage.as_ref()
                    {
                        record_terminal(&stats, usage, &model, started, cache_eligible);
                        if let Some((log, context)) = routing_log.as_ref() {
                            log.append(context.clone(), &model, None, usage);
                        }
                        recorded = true;
                    }
                    yield item;
                    if failed {
                        return;
                    }
                }
                if !recorded {
                    let usage = latest_usage.unwrap_or_default();
                    record_terminal(&stats, &usage, &model, started, cache_eligible);
                    if let Some((log, context)) = routing_log {
                        log.append(context, &model, None, &usage);
                    }
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
fn record_stream_error(stats: &StatsAccumulator, model: &str) {
    stats.record_stream_error(model);
    global::meter("switchyard")
        .u64_counter("switchyard.errors")
        .build()
        .add(1, &attributes(model));
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
    started: Instant,
    cache_eligible: f64,
) {
    let total_latency = started.elapsed();
    record_usage(usage, model);
    record_latency(model, total_latency);
    let mut token_usage = token_usage(usage);
    token_usage.cacheable_prompt_tokens =
        (token_usage.prompt_tokens as f64 * cache_eligible).round() as u64;
    stats.record_usage(model, token_usage, total_latency.as_secs_f64() * 1_000.0);
}

fn attributes(model: &str) -> [KeyValue; 1] {
    [KeyValue::new("model", model.to_string())]
}

fn record_usage(usage: &Usage, model: &str) {
    let attributes = attributes(model);
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

fn record_latency(model: &str, latency: Duration) {
    global::meter("switchyard")
        .f64_histogram("switchyard.total_latency_ms")
        .build()
        .record(latency.as_secs_f64() * 1000.0, &attributes(model));
}

#[cfg(test)]
mod tests {
    use futures_util::{StreamExt, stream};
    use switchyard_protocol::{LlmResponseChunk, LlmResponseStreamEvent, Metadata, Response};

    use super::*;

    /// An OpenAI Responses client may stop polling immediately after receiving the
    /// terminal `response.completed` event. Switchyard must record usage and routing
    /// data before returning that event because the stream wrapper will not resume
    /// after the client drops it.
    #[tokio::test]
    async fn terminal_event_is_recorded_before_the_client_drops_the_stream() {
        let dir = tempfile::tempdir().expect("temp dir");
        let log = SharedRoutingLog::new(dir.path().join("routing.jsonl")).expect("routing log");
        let context = RoutingLogContext::from_metadata(&Metadata {
            session_id: Some("streaming-session".to_string()),
            ..Metadata::default()
        });
        let usage = Usage {
            input_tokens: Some(10),
            output_tokens: Some(3),
            ..Usage::default()
        };
        let source = stream::iter([Ok(LlmResponseStreamEvent::new(vec![
            LlmResponseChunk::Usage(usage),
            LlmResponseChunk::MessageStop { reason: None },
        ]))]);
        let response = Response {
            llm_response: LlmResponse::Stream(Box::pin(source)),
            metadata: None,
        };
        let stats = StatsAccumulator::default();
        let observed = observe(
            response,
            "model/worker",
            Instant::now(),
            stats.clone(),
            0.0,
            Some((log.clone(), context)),
        );

        let LlmResponse::Stream(mut observed) = observed.llm_response else {
            panic!("expected stream");
        };
        assert!(observed.next().await.is_some());
        drop(observed);

        let routing = log
            .snapshot_session("streaming-session")
            .expect("read routing log")
            .expect("terminal event was recorded");
        let routing = serde_json::to_value(routing).expect("serialize routing stats");
        assert_eq!(routing["models"]["model/worker"]["calls"], 1);
        assert_eq!(routing["models"]["model/worker"]["prompt_tokens"], 10);
        assert_eq!(routing["models"]["model/worker"]["completion_tokens"], 3);

        let process = stats.snapshot();
        assert_eq!(process.models["model/worker"].prompt_tokens, 10);
        assert_eq!(process.models["model/worker"].completion_tokens, 3);
    }
}
