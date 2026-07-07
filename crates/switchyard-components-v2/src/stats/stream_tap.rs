// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native v2 stream tap that records token usage once a stream completes.
//!
//! v2 profiles record success synchronously, then read usage from the buffered
//! response body. Streaming responses carry no buffered body, so their usage is
//! only available while the stream is consumed. This module reuses the existing
//! `switchyard-components` usage parsers and commits usage exactly once per
//! stream, without depending on the v1 `ProxyContext` that
//! `StatsResponseProcessor` needs.
//!
//! OpenAI-format taps keep the **last** usage frame observed and commit when
//! the stream completes: spec-compliant upstreams send usage only on the final
//! chunk, while cumulative emitters (e.g. vLLM with
//! `stream_options.continuous_usage_stats`) update every chunk and must not be
//! recorded from the first, partial frame. Streams that never carry a usage
//! frame still commit zero usage so total-latency samples are not lost.

use std::time::Instant;

use async_stream::try_stream;
use futures_util::StreamExt;
use switchyard_components::stats::{
    openai_chat_usage_from_stream_event, openai_responses_usage_from_stream_event, usage_from_body,
    AnthropicStreamUsage, TokenUsage,
};
use switchyard_components::StatsAccumulator;
use switchyard_core::{BoxResponseStream, ChatResponse, StreamEvent};

/// Attribution captured before a response stream is consumed.
///
/// The matching `record_success` call has already counted the request against
/// the model and tier, so usage is recorded with
/// [`StatsAccumulator::record_usage_after_success_attribution`], which adds
/// tokens and latency without double-counting the call.
pub(crate) struct UsageAttribution {
    accumulator: StatsAccumulator,
    model: String,
    tier: Option<String>,
    started_at: Instant,
    backend_latency_ms: f64,
}

impl UsageAttribution {
    /// Captures the state needed to record usage once it becomes available.
    pub(crate) fn new(
        accumulator: StatsAccumulator,
        model: impl Into<String>,
        tier: Option<String>,
        started_at: Instant,
        backend_latency_ms: f64,
    ) -> Self {
        Self {
            accumulator,
            model: model.into(),
            tier,
            started_at,
            backend_latency_ms,
        }
    }

    /// Records the resolved usage, deriving total latency and routing overhead.
    ///
    /// For streams this runs on completion, so total latency spans the whole
    /// response, matching the v1 stats processor semantics.
    fn commit(&self, usage: TokenUsage) {
        let total_latency_ms = self.started_at.elapsed().as_secs_f64() * 1000.0;
        let routing_overhead_ms = (total_latency_ms - self.backend_latency_ms).max(0.0);
        if let Err(error) = self.accumulator.record_usage_after_success_attribution(
            self.model.clone(),
            usage,
            Some(total_latency_ms),
            Some(routing_overhead_ms),
            self.tier.as_deref(),
        ) {
            tracing::warn!(
                error = %error,
                model = %self.model,
                "failed to record stream usage",
            );
        }
    }
}

/// Records usage from a buffered body immediately, or taps a stream so usage is
/// recorded once the stream completes.
pub(crate) fn record_usage_or_tap_stream(
    response: ChatResponse,
    attribution: UsageAttribution,
) -> ChatResponse {
    match response {
        ChatResponse::OpenAiStream(stream) => ChatResponse::OpenAiStream(tap_last_usage(
            stream,
            attribution,
            openai_chat_usage_from_stream_event,
        )),
        ChatResponse::OpenAiResponsesStream(stream) => {
            ChatResponse::OpenAiResponsesStream(tap_last_usage(
                stream,
                attribution,
                openai_responses_usage_from_stream_event,
            ))
        }
        ChatResponse::AnthropicStream(stream) => {
            ChatResponse::AnthropicStream(tap_anthropic(stream, attribution))
        }
        buffered => {
            let usage = buffered.body().map(usage_from_body).unwrap_or_default();
            attribution.commit(usage);
            buffered
        }
    }
}

/// Taps an OpenAI-format stream, committing the last usage frame on completion.
///
/// Keeping the last frame handles both spec-compliant upstreams (usage only on
/// the final chunk) and cumulative emitters that update usage on every chunk.
/// Committing on completion also records a latency sample for streams that
/// never carry a usage frame. Abandoned streams commit nothing, matching the
/// v1 stream taps.
fn tap_last_usage(
    mut stream: BoxResponseStream,
    attribution: UsageAttribution,
    extract: fn(&StreamEvent) -> Option<TokenUsage>,
) -> BoxResponseStream {
    Box::pin(try_stream! {
        let mut latest = None;
        while let Some(event) = stream.next().await {
            let event = event?;
            if let Some(usage) = extract(&event) {
                latest = Some(usage);
            }
            yield event;
        }
        attribution.commit(latest.unwrap_or_default());
    })
}

/// Taps an Anthropic stream, committing accumulated usage at `message_stop`.
///
/// If the stream ends without the observer committing (no usage frames seen),
/// zero usage is committed so the total-latency sample is still recorded.
fn tap_anthropic(
    mut stream: BoxResponseStream,
    attribution: UsageAttribution,
) -> BoxResponseStream {
    Box::pin(try_stream! {
        let mut stream_usage = AnthropicStreamUsage::default();
        let mut committed = false;
        while let Some(event) = stream.next().await {
            let event = event?;
            if let Some(usage) = stream_usage.observe(&event) {
                attribution.commit(usage);
                committed = true;
            }
            yield event;
        }
        if !committed {
            attribution.commit(TokenUsage::default());
        }
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use switchyard_core::{Result, SwitchyardError};

    use super::*;

    fn attribution(accumulator: &StatsAccumulator) -> UsageAttribution {
        UsageAttribution::new(
            accumulator.clone(),
            "provider/model",
            Some("weak".to_string()),
            Instant::now(),
            5.0,
        )
    }

    fn boxed_stream(events: Vec<StreamEvent>) -> BoxResponseStream {
        let items: Vec<Result<StreamEvent>> = events.into_iter().map(Ok).collect();
        Box::pin(futures_util::stream::iter(items))
    }

    async fn drain(response: ChatResponse) -> Result<usize> {
        let mut stream = match response {
            ChatResponse::OpenAiStream(stream)
            | ChatResponse::OpenAiResponsesStream(stream)
            | ChatResponse::AnthropicStream(stream) => stream,
            other => {
                return Err(SwitchyardError::Other(format!(
                    "expected a stream response, got {other:?}"
                )))
            }
        };
        let mut count = 0;
        while let Some(event) = stream.next().await {
            event?;
            count += 1;
        }
        Ok(count)
    }

    #[tokio::test]
    async fn buffered_response_records_usage_immediately() -> Result<()> {
        let accumulator = StatsAccumulator::new();
        let response = record_usage_or_tap_stream(
            ChatResponse::openai_completion(json!({
                "usage": {"prompt_tokens": 12, "completion_tokens": 4},
            })),
            attribution(&accumulator),
        );

        assert!(matches!(response, ChatResponse::OpenAiCompletion(_)));
        let snapshot = accumulator.snapshot()?;
        assert_eq!(snapshot.total_tokens.prompt, 12);
        assert_eq!(snapshot.total_tokens.completion, 4);
        Ok(())
    }

    #[tokio::test]
    async fn openai_chat_stream_records_usage_only_after_consumption() -> Result<()> {
        let accumulator = StatsAccumulator::new();
        let response = record_usage_or_tap_stream(
            ChatResponse::OpenAiStream(boxed_stream(vec![
                StreamEvent::Json(json!({"choices": [{"delta": {"content": "hi"}}]})),
                StreamEvent::Json(
                    json!({"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 8}}),
                ),
            ])),
            attribution(&accumulator),
        );

        // The usage frame has not been consumed yet, so nothing is recorded.
        assert_eq!(accumulator.snapshot()?.total_tokens.prompt, 0);

        let forwarded = drain(response).await?;
        assert_eq!(forwarded, 2, "every event must still reach the client");

        let snapshot = accumulator.snapshot()?;
        assert_eq!(snapshot.total_tokens.prompt, 20);
        assert_eq!(snapshot.total_tokens.completion, 8);
        Ok(())
    }

    #[tokio::test]
    async fn openai_chat_cumulative_usage_frames_record_the_last_value() -> Result<()> {
        // Cumulative emitters (e.g. vLLM continuous_usage_stats) put usage on
        // every chunk; only the final cumulative value may be recorded.
        let accumulator = StatsAccumulator::new();
        let response = record_usage_or_tap_stream(
            ChatResponse::OpenAiStream(boxed_stream(vec![
                StreamEvent::Json(json!({
                    "choices": [{"delta": {"content": "h"}}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 1},
                })),
                StreamEvent::Json(json!({
                    "choices": [{"delta": {"content": "i"}}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 2},
                })),
                StreamEvent::Json(json!({
                    "choices": [],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 8},
                })),
            ])),
            attribution(&accumulator),
        );

        drain(response).await?;

        let snapshot = accumulator.snapshot()?;
        assert_eq!(snapshot.total_tokens.prompt, 20);
        assert_eq!(
            snapshot.total_tokens.completion, 8,
            "the last cumulative frame must win, not the first"
        );
        Ok(())
    }

    #[tokio::test]
    async fn stream_without_usage_frame_still_records_a_latency_sample() -> Result<()> {
        // Clients that stream without stream_options.include_usage get no usage
        // frame; the request already counted a call and must still contribute a
        // total-latency sample (with zero tokens), as buffered paths do.
        let accumulator = StatsAccumulator::new();
        let response = record_usage_or_tap_stream(
            ChatResponse::OpenAiStream(boxed_stream(vec![StreamEvent::Json(
                json!({"choices": [{"delta": {"content": "hi"}}]}),
            )])),
            attribution(&accumulator),
        );

        drain(response).await?;

        let snapshot = accumulator.snapshot()?;
        assert_eq!(snapshot.total_tokens.prompt, 0);
        assert_eq!(snapshot.total_tokens.completion, 0);
        let model = snapshot
            .models
            .get("provider/model")
            .ok_or_else(|| SwitchyardError::Other("model stats should be present".into()))?;
        assert_eq!(
            model.total_latency.count, 1,
            "usage-less streams must still record a total-latency sample"
        );
        Ok(())
    }

    #[tokio::test]
    async fn openai_responses_stream_records_usage_after_consumption() -> Result<()> {
        let accumulator = StatsAccumulator::new();
        let response = record_usage_or_tap_stream(
            ChatResponse::OpenAiResponsesStream(boxed_stream(vec![StreamEvent::Json(json!({
                "type": "response.completed",
                "response": {"usage": {"input_tokens": 15, "output_tokens": 6}},
            }))])),
            attribution(&accumulator),
        );

        assert_eq!(accumulator.snapshot()?.total_tokens.prompt, 0);
        drain(response).await?;

        let snapshot = accumulator.snapshot()?;
        assert_eq!(snapshot.total_tokens.prompt, 15);
        assert_eq!(snapshot.total_tokens.completion, 6);
        Ok(())
    }

    #[tokio::test]
    async fn anthropic_stream_records_usage_at_message_stop() -> Result<()> {
        let accumulator = StatsAccumulator::new();
        let response = record_usage_or_tap_stream(
            ChatResponse::AnthropicStream(boxed_stream(vec![
                StreamEvent::Json(json!({
                    "type": "message_start",
                    "message": {"usage": {"input_tokens": 30, "output_tokens": 0}},
                })),
                StreamEvent::Json(json!({
                    "type": "message_delta",
                    "usage": {"output_tokens": 9},
                })),
                StreamEvent::Json(json!({"type": "message_stop"})),
            ])),
            attribution(&accumulator),
        );

        assert_eq!(accumulator.snapshot()?.total_tokens.prompt, 0);
        drain(response).await?;

        let snapshot = accumulator.snapshot()?;
        assert_eq!(snapshot.total_tokens.prompt, 30);
        assert_eq!(snapshot.total_tokens.completion, 9);
        Ok(())
    }

    #[tokio::test]
    async fn anthropic_stream_without_usage_still_records_a_latency_sample() -> Result<()> {
        let accumulator = StatsAccumulator::new();
        let response = record_usage_or_tap_stream(
            ChatResponse::AnthropicStream(boxed_stream(vec![StreamEvent::Json(
                json!({"type": "content_block_delta", "delta": {"text": "hi"}}),
            )])),
            attribution(&accumulator),
        );

        drain(response).await?;

        let snapshot = accumulator.snapshot()?;
        assert_eq!(snapshot.total_tokens.completion, 0);
        let model = snapshot
            .models
            .get("provider/model")
            .ok_or_else(|| SwitchyardError::Other("model stats should be present".into()))?;
        assert_eq!(model.total_latency.count, 1);
        Ok(())
    }
}
