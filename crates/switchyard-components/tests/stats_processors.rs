// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stats processor tests covering request stamps, backend wrappers, and stream usage.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures_util::StreamExt;
use serde_json::json;
use switchyard_components::{
    BackendSelection, BackendSelectionReason, RandomRoutingDecision, RandomRoutingTier,
    StatsAccumulator, StatsBackendLatency, StatsLlmBackend, StatsRequestProcessor,
    StatsRequestStart, StatsResponseProcessor, StatsRouteLabel,
};
use switchyard_core::{
    ChatRequest, ChatRequestType, ChatResponse, LlmBackend, LlmTargetId, ModelId, ProxyContext,
    Result, StreamEvent, SwitchyardError,
};

static SUPPORTED_OPENAI_CHAT: [ChatRequestType; 1] = [ChatRequestType::OpenAiChat];

// Stamps a served model into context the same way native backends do.
fn record_backend_selection(ctx: &mut ProxyContext, model: ModelId) {
    let _ = ctx.insert(BackendSelection::for_model(
        model,
        None,
        BackendSelectionReason::PassthroughModel,
    ));
}

// Fake backend records lifecycle and request observations for stats tests.
struct FakeBackend {
    response: Mutex<Option<Result<ChatResponse>>>,
    calls: Mutex<Vec<Option<String>>>,
    selected_model: Option<ModelId>,
    tier: Option<String>,
    startup_count: AtomicUsize,
    shutdown_count: AtomicUsize,
}

impl FakeBackend {
    // Creates a fake backend that returns one successful response.
    fn success(response: ChatResponse) -> Self {
        Self {
            response: Mutex::new(Some(Ok(response))),
            calls: Mutex::new(Vec::new()),
            selected_model: None,
            tier: None,
            startup_count: AtomicUsize::new(0),
            shutdown_count: AtomicUsize::new(0),
        }
    }

    // Creates a fake backend that returns one error.
    fn error(error: SwitchyardError) -> Self {
        Self {
            response: Mutex::new(Some(Err(error))),
            calls: Mutex::new(Vec::new()),
            selected_model: None,
            tier: None,
            startup_count: AtomicUsize::new(0),
            shutdown_count: AtomicUsize::new(0),
        }
    }

    // Configures the model that the fake backend records in context.
    fn with_selected_model(mut self, model: ModelId) -> Self {
        self.selected_model = Some(model);
        self
    }

    // Configures the route label that the fake backend records in context.
    fn with_tier(mut self, tier: impl Into<String>) -> Self {
        self.tier = Some(tier.into());
        self
    }

    // Returns every request model observed by the fake backend.
    fn calls(&self) -> Result<Vec<Option<String>>> {
        Ok(self
            .calls
            .lock()
            .map_err(|_| SwitchyardError::Other("fake calls mutex poisoned".to_string()))?
            .clone())
    }
}

#[async_trait]
impl LlmBackend for FakeBackend {
    fn supported_request_types(&self) -> &[ChatRequestType] {
        &SUPPORTED_OPENAI_CHAT
    }

    async fn call(&self, ctx: &mut ProxyContext, request: &ChatRequest) -> Result<ChatResponse> {
        self.calls
            .lock()
            .map_err(|_| SwitchyardError::Other("fake calls mutex poisoned".to_string()))?
            .push(request.model().map(str::to_string));
        if let Some(model) = &self.selected_model {
            record_backend_selection(ctx, model.clone());
        }
        if let Some(tier) = &self.tier {
            ctx.insert(StatsRouteLabel::new(tier.clone()));
        }
        self.response
            .lock()
            .map_err(|_| SwitchyardError::Other("fake response mutex poisoned".to_string()))?
            .take()
            .ok_or_else(|| SwitchyardError::Other("fake response already consumed".to_string()))?
    }

    async fn startup(&self) -> Result<()> {
        self.startup_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn shutdown(&self) -> Result<()> {
        self.shutdown_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

// Request stats should add timing state without changing the request.
#[tokio::test]
async fn request_processor_stamps_start_without_mutating_request() -> Result<()> {
    let processor = StatsRequestProcessor::default();
    let request = ChatRequest::openai_chat(json!({"model": "client", "messages": []}));
    let mut ctx = ProxyContext::new();

    let processed = processor.process(&mut ctx, request.clone()).await?;

    assert_eq!(processed, request);
    if ctx.get::<StatsRequestStart>().is_none() {
        return Err(SwitchyardError::Other(
            "stats request start should be stamped".to_string(),
        ));
    }
    Ok(())
}

// End-to-end stats components should share one accumulator across the chain.
#[tokio::test]
async fn full_stats_chain_shares_one_accumulator_across_all_components() -> Result<()> {
    let accumulator = StatsAccumulator::new();
    let request_processor = StatsRequestProcessor::default();
    let backend = StatsLlmBackend::new(
        Arc::new(
            FakeBackend::success(ChatResponse::openai_completion(json!({
                "usage": {"prompt_tokens": 12, "completion_tokens": 8}
            })))
            .with_selected_model(ModelId::new("served-chain-model")?)
            .with_tier("strong"),
        ),
        accumulator.clone(),
    );
    let response_processor = StatsResponseProcessor::new(accumulator.clone());
    let mut ctx = ProxyContext::new();

    let request = request_processor
        .process(
            &mut ctx,
            ChatRequest::openai_chat(json!({"model": "client-model", "messages": []})),
        )
        .await?;
    let response = backend.call(&mut ctx, &request).await?;
    response_processor.process(&mut ctx, response).await?;

    let snapshot = accumulator.snapshot()?;
    assert_eq!(snapshot.total_requests, 1);
    assert_eq!(snapshot.total_tokens.prompt, 12);
    assert_eq!(snapshot.total_tokens.completion, 8);
    assert_eq!(snapshot.routing_overhead.count, 1);
    let model = model_stats(&snapshot, "served-chain-model")?;
    assert_eq!(model.calls, 1);
    assert_eq!(model.model_call_latency.count, 1);
    assert_eq!(model.total_latency.count, 1);
    Ok(())
}

// Theoretical flows request -> ctx -> response -> snapshot, and is switch-aware:
// a model only credits a prefix it has already been sent.
#[tokio::test]
async fn theoretical_cache_hit_rate_flows_through_the_chain() -> Result<()> {
    let accumulator = StatsAccumulator::new();
    let request_processor = StatsRequestProcessor::new(true);
    let response_processor = StatsResponseProcessor::new(accumulator.clone());
    // FakeBackend serves one response, so build a fresh one per turn.
    let make_backend = || -> Result<StatsLlmBackend> {
        Ok(StatsLlmBackend::new(
            Arc::new(
                FakeBackend::success(ChatResponse::openai_completion(json!({
                    "usage": {"prompt_tokens": 100, "completion_tokens": 4}
                })))
                .with_selected_model(ModelId::new("served-model")?)
                .with_tier("strong"),
            ),
            accumulator.clone(),
        ))
    };

    // Turn 1: first sight of the conversation, nothing cached yet.
    let mut ctx = ProxyContext::new();
    let request = request_processor
        .process(
            &mut ctx,
            ChatRequest::openai_chat(json!({
                "model": "m",
                "messages": [{"role": "user", "content": "aaaa"}],
            })),
        )
        .await?;
    let response = make_backend()?.call(&mut ctx, &request).await?;
    response_processor.process(&mut ctx, response).await?;

    // Turn 2: same model, the prior turn is re-presented, so half is eligible.
    let mut ctx = ProxyContext::new();
    let request = request_processor
        .process(
            &mut ctx,
            ChatRequest::openai_chat(json!({
                "model": "m",
                "messages": [
                    {"role": "user", "content": "aaaa"},
                    {"role": "user", "content": "bbbb"},
                ],
            })),
        )
        .await?;
    let response = make_backend()?.call(&mut ctx, &request).await?;
    response_processor.process(&mut ctx, response).await?;

    let snapshot = accumulator.snapshot()?;
    let model = model_stats(&snapshot, "served-model")?;
    assert_eq!(model.cache_hit_rate, 0.0);
    // Turn 1 cold (0/100) + turn 2 half (50/100) = 50 over 200 prompt tokens.
    assert_eq!(model.theoretical_cache_hit_rate, 0.25);
    Ok(())
}

// Backend wrapper should record served model latency and delegate lifecycle.
#[tokio::test]
async fn backend_wrapper_records_success_using_served_model_and_delegates_lifecycle() -> Result<()>
{
    let accumulator = StatsAccumulator::new();
    let inner = Arc::new(
        FakeBackend::success(ChatResponse::openai_completion(json!({"id": "ok"})))
            .with_selected_model(ModelId::new("served-model")?)
            .with_tier("strong"),
    );
    let backend = StatsLlmBackend::new(inner.clone(), accumulator.clone());
    let request = ChatRequest::openai_chat(json!({"model": "client-model", "messages": []}));
    let mut ctx = ProxyContext::new();

    assert_eq!(
        backend.supported_request_types(),
        &[ChatRequestType::OpenAiChat]
    );
    backend.startup().await?;
    let response = backend.call(&mut ctx, &request).await?;
    backend.shutdown().await?;

    assert!(matches!(response, ChatResponse::OpenAiCompletion(_)));
    assert_eq!(inner.startup_count.load(Ordering::SeqCst), 1);
    assert_eq!(inner.shutdown_count.load(Ordering::SeqCst), 1);
    assert_eq!(inner.calls()?, vec![Some("client-model".to_string())]);
    if ctx.get::<StatsBackendLatency>().is_none() {
        return Err(SwitchyardError::Other(
            "backend latency should be stamped".to_string(),
        ));
    }

    let snapshot = accumulator.snapshot()?;
    assert_eq!(snapshot.total_requests, 1);
    let model = model_stats(&snapshot, "served-model")?;
    assert_eq!(model.calls, 1);
    assert_eq!(model.errors, 0);
    assert_eq!(model.tier.as_deref(), Some("strong"));
    assert_eq!(model.model_call_latency.count, 1);
    Ok(())
}

// If a backend does not stamp a served model, request model is the fallback.
#[tokio::test]
async fn backend_wrapper_falls_back_to_request_model_when_backend_does_not_stamp_model(
) -> Result<()> {
    let accumulator = StatsAccumulator::new();
    let backend = StatsLlmBackend::new(
        Arc::new(FakeBackend::success(ChatResponse::openai_completion(
            json!({"id": "ok"}),
        ))),
        accumulator.clone(),
    );
    let request = ChatRequest::openai_chat(json!({"model": "client-fallback", "messages": []}));
    let mut ctx = ProxyContext::new();

    backend.call(&mut ctx, &request).await?;

    let snapshot = accumulator.snapshot()?;
    let model = model_stats(&snapshot, "client-fallback")?;
    assert_eq!(model.calls, 1);
    Ok(())
}

// Backend wrapper should preserve the original error while counting it.
#[tokio::test]
async fn backend_wrapper_records_errors_and_preserves_original_error() -> Result<()> {
    let accumulator = StatsAccumulator::new();
    let inner = Arc::new(
        FakeBackend::error(SwitchyardError::Upstream("boom".to_string()))
            .with_selected_model(ModelId::new("served-error-model")?)
            .with_tier("weak"),
    );
    let backend = StatsLlmBackend::new(inner, accumulator.clone());
    let request = ChatRequest::openai_chat(json!({"model": "client-model", "messages": []}));
    let mut ctx = ProxyContext::new();

    let Err(error) = backend.call(&mut ctx, &request).await else {
        return Err(SwitchyardError::Other(
            "backend wrapper should return inner error".to_string(),
        ));
    };

    assert!(matches!(error, SwitchyardError::Upstream(_)));
    assert!(error.to_string().contains("boom"));
    let snapshot = accumulator.snapshot()?;
    assert_eq!(snapshot.total_requests, 1);
    assert_eq!(snapshot.total_errors, 1);
    let model = model_stats(&snapshot, "served-error-model")?;
    assert_eq!(model.calls, 0);
    assert_eq!(model.errors, 1);
    assert_eq!(model.tier.as_deref(), Some("weak"));
    Ok(())
}

// Response stats should clamp impossible negative overhead to zero.
#[tokio::test]
async fn response_processor_records_openai_usage_latency_and_clamps_negative_overhead() -> Result<()>
{
    let accumulator = StatsAccumulator::new();
    let processor = StatsResponseProcessor::new(accumulator.clone());
    let mut ctx = ProxyContext::new();
    record_backend_selection(&mut ctx, ModelId::new("openai-model")?);
    ctx.insert(StatsRequestStart::now());
    ctx.insert(StatsBackendLatency(Duration::from_secs(60)));

    let response = ChatResponse::openai_completion(json!({
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 5,
            "prompt_tokens_details": {
                "cached_tokens": 3,
                "cache_creation_tokens": 2
            },
            "completion_tokens_details": {
                "reasoning_tokens": 4
            }
        }
    }));
    let processed = processor.process(&mut ctx, response).await?;

    assert!(matches!(processed, ChatResponse::OpenAiCompletion(_)));
    let snapshot = accumulator.snapshot()?;
    assert_eq!(snapshot.total_requests, 0);
    assert_eq!(snapshot.total_tokens.prompt, 11);
    assert_eq!(snapshot.total_tokens.completion, 5);
    assert_eq!(snapshot.total_tokens.cached, 3);
    assert_eq!(snapshot.total_tokens.cache_creation, 2);
    assert_eq!(snapshot.total_tokens.reasoning, 4);
    assert_eq!(snapshot.routing_overhead.count, 1);
    assert_eq!(snapshot.routing_overhead.max_ms, 0.0);
    let model = model_stats(&snapshot, "openai-model")?;
    assert_eq!(model.total_latency.count, 1);
    Ok(())
}

// Anthropic cache counters should contribute to prompt token accounting.
#[tokio::test]
async fn response_processor_sums_anthropic_cache_buckets_into_prompt_tokens() -> Result<()> {
    let accumulator = StatsAccumulator::new();
    let processor = StatsResponseProcessor::new(accumulator.clone());
    let mut ctx = ProxyContext::new();
    record_backend_selection(&mut ctx, ModelId::new("claude-model")?);

    let response = ChatResponse::anthropic_completion(json!({
        "usage": {
            "input_tokens": 10,
            "output_tokens": 6,
            "cache_read_input_tokens": 3,
            "cache_creation_input_tokens": 2,
            "output_tokens_details": {"reasoning_tokens": 1}
        }
    }));
    processor.process(&mut ctx, response).await?;

    let snapshot = accumulator.snapshot()?;
    let model = model_stats(&snapshot, "claude-model")?;
    assert_eq!(model.prompt_tokens, 15);
    assert_eq!(model.completion_tokens, 6);
    assert_eq!(model.cached_tokens, 3);
    assert_eq!(model.cache_creation_tokens, 2);
    assert_eq!(model.reasoning_tokens, 1);
    Ok(())
}

// Responses streams should pass through unchanged while capturing final usage.
#[tokio::test]
async fn streaming_response_is_forwarded_and_records_nested_responses_usage() -> Result<()> {
    let accumulator = StatsAccumulator::new();
    let processor = StatsResponseProcessor::new(accumulator.clone());
    let mut ctx = ProxyContext::new();
    record_backend_selection(&mut ctx, ModelId::new("responses-model")?);
    ctx.insert(StatsRequestStart::now());

    let events = vec![
        StreamEvent::Json(json!({"type": "response.output_text.delta", "delta": "hi"})),
        StreamEvent::Json(json!({
            "type": "response.in_progress",
            "response": {"usage": null}
        })),
        StreamEvent::Json(json!({
            "type": "response.completed",
            "response": {
                "usage": {
                    "input_tokens": 8,
                    "output_tokens": 13,
                    "input_tokens_details": {"cached_tokens": 5}
                }
            }
        })),
    ];
    let stream_events = events.clone();
    let stream = futures_util::stream::iter(stream_events.into_iter().map(Ok));
    let response = ChatResponse::OpenAiResponsesStream(Box::pin(stream));

    let processed = processor.process(&mut ctx, response).await?;
    let drained = drain_responses_stream(processed).await?;

    assert_eq!(drained, events);
    let snapshot = accumulator.snapshot()?;
    let model = model_stats(&snapshot, "responses-model")?;
    assert_eq!(model.prompt_tokens, 8);
    assert_eq!(model.completion_tokens, 13);
    assert_eq!(model.cached_tokens, 5);
    assert_eq!(model.total_latency.count, 1);
    Ok(())
}

// Raw SSE frame strings (verbatim Responses passthrough) still record usage.
#[tokio::test]
async fn raw_sse_frame_responses_stream_passes_through_and_records_usage() -> Result<()> {
    let accumulator = StatsAccumulator::new();
    let processor = StatsResponseProcessor::new(accumulator.clone());
    let mut ctx = ProxyContext::new();
    record_backend_selection(&mut ctx, ModelId::new("raw-frame-model")?);
    ctx.insert(StatsRequestStart::now());

    let events = vec![
        StreamEvent::Text(
            "event: response.output_text.delta\n\
             data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n"
                .to_string(),
        ),
        StreamEvent::Text(": keep-alive\n\n".to_string()),
        StreamEvent::Text(
            "event: response.completed\n\
             data: {\"type\":\"response.completed\",\"response\":{\"usage\":\
             {\"input_tokens\":8,\"output_tokens\":13,\
             \"input_tokens_details\":{\"cached_tokens\":5}}}}\n\n"
                .to_string(),
        ),
    ];
    let stream_events = events.clone();
    let stream = futures_util::stream::iter(stream_events.into_iter().map(Ok));
    let response = ChatResponse::OpenAiResponsesStream(Box::pin(stream));

    let processed = processor.process(&mut ctx, response).await?;
    let drained = drain_responses_stream(processed).await?;

    // Frames pass through byte-identical; usage still lands in the snapshot.
    assert_eq!(drained, events);
    let snapshot = accumulator.snapshot()?;
    let model = model_stats(&snapshot, "raw-frame-model")?;
    assert_eq!(model.prompt_tokens, 8);
    assert_eq!(model.completion_tokens, 13);
    assert_eq!(model.cached_tokens, 5);
    Ok(())
}

// OpenAI streams should record the first real usage block only.
#[tokio::test]
async fn openai_chat_stream_records_first_usage_chunk_only_with_details() -> Result<()> {
    let accumulator = StatsAccumulator::new();
    let processor = StatsResponseProcessor::new(accumulator.clone());
    let mut ctx = ProxyContext::new();
    record_backend_selection(&mut ctx, ModelId::new("chat-stream-model")?);

    let events = vec![
        StreamEvent::Json(json!({"usage": null})),
        StreamEvent::Json(json!({"choices": [{"delta": {"content": "hi"}}]})),
        StreamEvent::Json(json!({
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 7,
                "prompt_tokens_details": {"cached_tokens": 4},
                "completion_tokens_details": {"reasoning_tokens": 2}
            }
        })),
        StreamEvent::Json(json!({
            "usage": {"prompt_tokens": 200, "completion_tokens": 70}
        })),
    ];
    let stream = futures_util::stream::iter(events.clone().into_iter().map(Ok));
    let processed = processor
        .process(&mut ctx, ChatResponse::OpenAiStream(Box::pin(stream)))
        .await?;
    let drained = drain_openai_stream(processed).await?;

    assert_eq!(drained, events);
    let snapshot = accumulator.snapshot()?;
    let model = model_stats(&snapshot, "chat-stream-model")?;
    assert_eq!(model.prompt_tokens, 20);
    assert_eq!(model.completion_tokens, 7);
    assert_eq!(model.cached_tokens, 4);
    assert_eq!(model.reasoning_tokens, 2);
    assert_eq!(model.total_latency.count, 0);
    Ok(())
}

// Anthropic streaming usage can arrive in start and delta events.
#[tokio::test]
async fn anthropic_stream_merges_start_and_delta_usage_and_commits_once_at_stop() -> Result<()> {
    let accumulator = StatsAccumulator::new();
    let processor = StatsResponseProcessor::new(accumulator.clone());
    let mut ctx = ProxyContext::new();
    record_backend_selection(&mut ctx, ModelId::new("claude-stream-model")?);
    ctx.insert(StatsRequestStart::now());

    let events = vec![
        StreamEvent::Json(json!({
            "type": "message_start",
            "message": {
                "usage": {
                    "input_tokens": 50,
                    "cache_read_input_tokens": 10,
                    "cache_creation_input_tokens": 5
                }
            }
        })),
        StreamEvent::Json(json!({
            "type": "message_delta",
            "usage": {"output_tokens": 20}
        })),
        StreamEvent::Json(json!({"type": "message_stop"})),
        StreamEvent::Json(json!({"type": "message_stop"})),
    ];
    let stream = futures_util::stream::iter(events.clone().into_iter().map(Ok));
    let processed = processor
        .process(&mut ctx, ChatResponse::AnthropicStream(Box::pin(stream)))
        .await?;
    let drained = drain_anthropic_stream(processed).await?;

    assert_eq!(drained, events);
    let snapshot = accumulator.snapshot()?;
    let model = model_stats(&snapshot, "claude-stream-model")?;
    assert_eq!(model.prompt_tokens, 65);
    assert_eq!(model.completion_tokens, 20);
    assert_eq!(model.cached_tokens, 10);
    assert_eq!(model.cache_creation_tokens, 5);
    assert_eq!(model.total_latency.count, 1);
    Ok(())
}

// Later Anthropic input-token deltas should override a zero start value.
#[tokio::test]
async fn anthropic_stream_delta_input_tokens_override_zero_start_tokens() -> Result<()> {
    let accumulator = StatsAccumulator::new();
    let processor = StatsResponseProcessor::new(accumulator.clone());
    let mut ctx = ProxyContext::new();
    record_backend_selection(&mut ctx, ModelId::new("claude-delta-input")?);

    let events = vec![
        StreamEvent::Json(json!({
            "type": "message_start",
            "message": {"usage": {"input_tokens": 0}}
        })),
        StreamEvent::Json(json!({
            "type": "message_delta",
            "usage": {"input_tokens": 75, "output_tokens": 30}
        })),
        StreamEvent::Json(json!({"type": "message_stop"})),
    ];
    let stream = futures_util::stream::iter(events.into_iter().map(Ok));
    let processed = processor
        .process(&mut ctx, ChatResponse::AnthropicStream(Box::pin(stream)))
        .await?;
    let _ = drain_anthropic_stream(processed).await?;

    let snapshot = accumulator.snapshot()?;
    let model = model_stats(&snapshot, "claude-delta-input")?;
    assert_eq!(model.prompt_tokens, 75);
    assert_eq!(model.completion_tokens, 30);
    Ok(())
}

// Streams with no usage should remain transparent and avoid fake stats entries.
#[tokio::test]
async fn stream_without_usage_passes_through_without_creating_stats_entry() -> Result<()> {
    let accumulator = StatsAccumulator::new();
    let processor = StatsResponseProcessor::new(accumulator.clone());
    let mut ctx = ProxyContext::new();
    record_backend_selection(&mut ctx, ModelId::new("no-usage-stream")?);

    let events = vec![
        StreamEvent::Json(json!({"choices": [{"delta": {"content": "a"}}]})),
        StreamEvent::Text("plain text ignored".to_string()),
    ];
    let stream = futures_util::stream::iter(events.clone().into_iter().map(Ok));
    let processed = processor
        .process(&mut ctx, ChatResponse::OpenAiStream(Box::pin(stream)))
        .await?;

    assert_eq!(drain_openai_stream(processed).await?, events);
    assert!(accumulator.snapshot()?.models.is_empty());
    Ok(())
}

// Buffered responses without usage still create a zero-token model entry.
#[tokio::test]
async fn buffered_response_without_usage_records_zero_token_model_entry() -> Result<()> {
    let accumulator = StatsAccumulator::new();
    let processor = StatsResponseProcessor::new(accumulator.clone());
    let mut ctx = ProxyContext::new();
    record_backend_selection(&mut ctx, ModelId::new("missing-usage-model")?);

    processor
        .process(
            &mut ctx,
            ChatResponse::openai_completion(json!({"id": "no-usage"})),
        )
        .await?;

    let snapshot = accumulator.snapshot()?;
    let model = model_stats(&snapshot, "missing-usage-model")?;
    assert_eq!(model.prompt_tokens, 0);
    assert_eq!(model.completion_tokens, 0);
    assert_eq!(snapshot.total_tokens.total, 0);
    Ok(())
}

// Malformed usage fields should be ignored without losing valid alternatives.
#[tokio::test]
async fn malformed_usage_values_are_ignored_without_wrapping_or_panicking() -> Result<()> {
    let accumulator = StatsAccumulator::new();
    let processor = StatsResponseProcessor::new(accumulator.clone());
    let mut ctx = ProxyContext::new();
    record_backend_selection(&mut ctx, ModelId::new("malformed-usage-model")?);

    processor
        .process(
            &mut ctx,
            ChatResponse::openai_completion(json!({
                "usage": {
                    "prompt_tokens": -1,
                    "completion_tokens": "bad",
                    "input_tokens": 9,
                    "output_tokens": true,
                    "prompt_tokens_details": {"cached_tokens": null},
                    "completion_tokens_details": {"reasoning_tokens": 3.5}
                }
            })),
        )
        .await?;

    let snapshot = accumulator.snapshot()?;
    let model = model_stats(&snapshot, "malformed-usage-model")?;
    assert_eq!(model.prompt_tokens, 9);
    assert_eq!(model.completion_tokens, 0);
    assert_eq!(model.cached_tokens, 0);
    assert_eq!(model.reasoning_tokens, 0);
    Ok(())
}

// Random-routing context should populate tier rollups in response stats.
#[tokio::test]
async fn response_processor_uses_random_routing_decision_for_tier_rollups() -> Result<()> {
    let accumulator = StatsAccumulator::new();
    let processor = StatsResponseProcessor::new(accumulator.clone());
    let mut ctx = ProxyContext::new();
    record_backend_selection(&mut ctx, ModelId::new("weak-model")?);
    ctx.insert(RandomRoutingDecision {
        tier: RandomRoutingTier::Weak,
        selected_target: LlmTargetId::from_static("weak"),
        selected_model: ModelId::new("weak-model")?,
        original_model: Some("client-model".to_string()),
        strong_probability: 0.25,
        draw: 0.75,
    });

    processor
        .process(
            &mut ctx,
            ChatResponse::openai_completion(json!({
                "usage": {"prompt_tokens": 4, "completion_tokens": 9}
            })),
        )
        .await?;

    let snapshot = accumulator.snapshot()?;
    let tier = snapshot
        .tiers
        .get("weak")
        .ok_or_else(|| SwitchyardError::Other("weak tier should be present".to_string()))?;
    assert_eq!(tier.model, "weak-model");
    assert_eq!(tier.prompt_tokens, 4);
    assert_eq!(tier.completion_tokens, 9);
    assert_eq!(tier.total_tokens, 13);
    Ok(())
}

// Context-selected targets should be a generic tier fallback when no richer
// router-specific label exists.
#[tokio::test]
async fn response_processor_uses_selected_target_as_tier_fallback() -> Result<()> {
    let accumulator = StatsAccumulator::new();
    let processor = StatsResponseProcessor::new(accumulator.clone());
    let mut ctx = ProxyContext::new();
    record_backend_selection(&mut ctx, ModelId::new("executor-model")?);
    ctx.set_selected_target(LlmTargetId::from_static("executor"));

    processor
        .process(
            &mut ctx,
            ChatResponse::openai_completion(json!({
                "usage": {"prompt_tokens": 3, "completion_tokens": 5}
            })),
        )
        .await?;

    let snapshot = accumulator.snapshot()?;
    let tier = snapshot
        .tiers
        .get("executor")
        .ok_or_else(|| SwitchyardError::Other("executor tier should be present".to_string()))?;
    assert_eq!(tier.model, "executor-model");
    assert_eq!(tier.prompt_tokens, 3);
    assert_eq!(tier.completion_tokens, 5);
    Ok(())
}

// Explicit stats route labels intentionally override random-routing labels.
#[tokio::test]
async fn explicit_stats_route_label_takes_precedence_over_random_routing_decision() -> Result<()> {
    let accumulator = StatsAccumulator::new();
    let processor = StatsResponseProcessor::new(accumulator.clone());
    let mut ctx = ProxyContext::new();
    record_backend_selection(&mut ctx, ModelId::new("labeled-model")?);
    ctx.insert(StatsRouteLabel::new("plugin"));
    ctx.insert(RandomRoutingDecision {
        tier: RandomRoutingTier::Weak,
        selected_target: LlmTargetId::from_static("weak"),
        selected_model: ModelId::new("labeled-model")?,
        original_model: None,
        strong_probability: 0.25,
        draw: 0.75,
    });

    processor
        .process(
            &mut ctx,
            ChatResponse::openai_completion(json!({
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}
            })),
        )
        .await?;

    let snapshot = accumulator.snapshot()?;
    let model = model_stats(&snapshot, "labeled-model")?;
    assert_eq!(model.tier.as_deref(), Some("plugin"));
    let plugin = snapshot
        .tiers
        .get("plugin")
        .ok_or_else(|| SwitchyardError::Other("plugin tier should be present".to_string()))?;
    assert_eq!(plugin.model, "labeled-model");
    assert_eq!(plugin.calls, 0);
    assert_eq!(plugin.prompt_tokens, 1);
    assert_eq!(plugin.completion_tokens, 1);
    assert!(!snapshot.tiers.contains_key("weak"));
    Ok(())
}

// Drains an OpenAI Responses stream after the stats wrapper has observed it.
async fn drain_responses_stream(response: ChatResponse) -> Result<Vec<StreamEvent>> {
    let ChatResponse::OpenAiResponsesStream(mut stream) = response else {
        return Err(SwitchyardError::Other(
            "response should remain an OpenAI Responses stream".to_string(),
        ));
    };
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event?);
    }
    Ok(events)
}

// Drains an OpenAI Chat stream after the stats wrapper has observed it.
async fn drain_openai_stream(response: ChatResponse) -> Result<Vec<StreamEvent>> {
    let ChatResponse::OpenAiStream(mut stream) = response else {
        return Err(SwitchyardError::Other(
            "response should remain an OpenAI stream".to_string(),
        ));
    };
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event?);
    }
    Ok(events)
}

// Drains an Anthropic stream after the stats wrapper has observed it.
async fn drain_anthropic_stream(response: ChatResponse) -> Result<Vec<StreamEvent>> {
    let ChatResponse::AnthropicStream(mut stream) = response else {
        return Err(SwitchyardError::Other(
            "response should remain an Anthropic stream".to_string(),
        ));
    };
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event?);
    }
    Ok(events)
}

// Fetches one model stats block with an explicit test error on absence.
fn model_stats<'a>(
    snapshot: &'a switchyard_components::StatsSnapshot,
    model: &str,
) -> Result<&'a switchyard_components::ModelStatsSnapshot> {
    snapshot
        .models
        .get(model)
        .ok_or_else(|| SwitchyardError::Other(format!("model stats missing for {model}")))
}
