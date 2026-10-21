// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-process metrics work with fake clients and no HTTP server or observer.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use opentelemetry::{KeyValue, global};
use switchyard_libsy::metrics::{MetricFilter, MetricValue, Metrics};
use switchyard_libsy::{Algorithm, Driver, RoutingOutcome, RuntimeModels};
use switchyard_llm_client::ClientRouter;
use switchyard_protocol::{
    DecisionRequest, DecisionResponse, LlmClientError, LlmResponse, LlmResponseChunk, Message,
    Metadata, Request, Response, Role, RoutedDecisionClient, RoutedLlmClient, Usage, text_request,
    text_response,
};

type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

struct Clients;

fn usage(input: u64, output: u64) -> Usage {
    Usage {
        input_tokens: Some(input),
        total_tokens: Some(input + output + 7),
        output_tokens: Some(output),
        cache: Usage::cache_details(Some(3), Some(4)),
        reasoning_tokens: Some(1),
        ..Usage::default()
    }
}

#[async_trait]
impl RoutedLlmClient for Clients {
    async fn call(&self, request: Request) -> Result<Response, LlmClientError> {
        let model = request.llm_request.model.as_deref().unwrap();
        if model == "failed" {
            return Err(LlmClientError::Timeout {
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "test timeout",
                )),
            });
        }
        let mut response = text_response(Some(model.into()), "answer");
        response.usage = if model == "judge" {
            usage(10, 2)
        } else {
            usage(20, 4)
        };
        let llm_response = if model == "partial" {
            LlmResponse::Stream(Box::pin(futures::stream::iter([
                Ok(LlmResponseChunk::Usage(usage(20, 4)).into()),
                Err(LlmClientError::Timeout {
                    source: Box::new(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "test timeout",
                    )),
                }),
            ])))
        } else if request.llm_request.stream {
            LlmResponse::Stream(response.into_stream())
        } else {
            LlmResponse::Agg(response)
        };
        Ok(Response {
            llm_response,
            metadata: None,
            upstream_headers: http::HeaderMap::new(),
        })
    }
}

#[async_trait]
impl RoutedDecisionClient for Clients {
    async fn call(&self, _: DecisionRequest) -> Result<DecisionResponse, LlmClientError> {
        Ok(DecisionResponse {
            id: None,
            model: None,
            answers: BTreeMap::new(),
            usage: usage(5, 1),
        })
    }
}

struct Router {
    answer: &'static str,
}

#[async_trait]
impl Algorithm for Router {
    fn name(&self) -> &str {
        "metrics-query-router"
    }

    async fn route(
        self: Arc<Self>,
        driver: Driver,
        request: Request,
    ) -> switchyard_libsy::Result<RoutingOutcome> {
        let response = driver
            .call_model(request.clone(), vec!["judge".into()])
            .await?;
        if self.answer == "reuse" {
            return Ok(RoutingOutcome::answered("judge".into(), request, response));
        }
        // A discarded routing response still contributes its billed usage.
        drop(response);
        driver
            .call_decision(
                DecisionRequest {
                    model: None,
                    context: serde_json::json!({}),
                    questions: BTreeMap::new(),
                },
                "decision".into(),
            )
            .await?;
        global::meter("switchyard")
            .u64_counter("switchyard.custom_decisions")
            .build()
            .add(
                1,
                &switchyard_libsy::metric_context::attributes(&[KeyValue::new("route", "custom")]),
            );
        Ok(RoutingOutcome::route_to(
            self.answer.into(),
            vec![],
            request,
        ))
    }
}

fn request(session: &str, agent: &str) -> Request {
    Request {
        llm_request: text_request(Some("auto".into()), "hello"),
        raw_request: None,
        metadata: Some(Metadata {
            session_id: Some(session.into()),
            agent_id: Some(agent.into()),
            ..Metadata::default()
        }),
    }
}

async fn run(request: Request, answer: &'static str) -> switchyard_libsy::Result<Response> {
    let clients = ClientRouter::single_with_decision_clients(
        Arc::new(Clients),
        HashMap::from([(
            "decision".into(),
            Arc::new(Clients) as Arc<dyn RoutedDecisionClient>,
        )]),
    );
    switchyard_llm_client::run(
        Arc::new(Router { answer }),
        clients,
        request,
        Arc::new(RuntimeModels::default()),
        None,
    )
    .await
    .map(|(_, response)| response)
}

#[tokio::test]
async fn queries_all_metrics_or_one_session_agent_and_turn_without_a_server() -> TestResult {
    let metrics = Metrics::install()?;
    let mut first = request("one", "worker");
    run(first.clone(), "answer").await?;
    first
        .llm_request
        .messages
        .push(Message::text(Role::Tool, "result"));
    first.llm_request.stream = true;
    let streamed = run(first.clone(), "answer").await?;
    // Another agent completes while the worker's stream is still open.
    run(request("one", "child"), "answer").await?;
    streamed.llm_response.into_agg().await?;
    first
        .llm_request
        .messages
        .push(Message::text(Role::User, "next"));
    first.llm_request.stream = false;
    run(first, "answer").await?;
    run(request("two", "worker"), "answer").await?;

    for (session, agent, agent_idx, user_idx) in [
        ("one", "worker", 0, 0),
        ("one", "worker", 1, 0),
        ("one", "worker", 2, 1),
        ("one", "child", 0, 0),
        ("two", "worker", 0, 0),
    ] {
        let filter = MetricFilter {
            session_id: Some(session),
            agent_id: Some(agent),
            agent_turn_idx: Some(agent_idx),
            user_turn_idx: Some(user_idx),
        };
        let stats = metrics.stats(filter);
        assert_eq!(
            stats.counter_total("switchyard_llm_call_prompt_tokens_total"),
            Some(56.0)
        );
        assert_eq!(
            stats.counter_total("switchyard_llm_call_input_tokens_total"),
            Some(35.0)
        );
        assert_eq!(
            stats.counter_total("switchyard_llm_call_total_tokens_total"),
            Some(63.0)
        );
        assert_eq!(
            stats.counter_total("switchyard_llm_call_completion_tokens_total"),
            Some(7.0)
        );
        assert_eq!(
            stats.counter_total("switchyard_llm_call_cached_tokens_total"),
            Some(9.0)
        );
        assert_eq!(
            stats.counter_total("switchyard_llm_call_cache_creation_tokens_total"),
            Some(12.0)
        );
        assert_eq!(
            stats.counter_total("switchyard_llm_call_reasoning_tokens_total"),
            Some(3.0)
        );
        // Compatibility counters contain only the answer, without routing or duplicate server accounting.
        assert_eq!(
            stats.counter_total("switchyard_prompt_tokens_total"),
            Some(27.0)
        );
        assert_eq!(
            stats.counter_total("switchyard_custom_decisions_total"),
            Some(1.0)
        );
        for name in [
            "switchyard_inference_duration_ms",
            "switchyard_routing_overhead_ms",
            "switchyard_model_call_latency_ms",
        ] {
            assert!(
                stats.series.iter().any(|series| series.name == name
                    && matches!(series.value, MetricValue::Histogram { count: 1, .. })),
                "missing {name}"
            );
        }
        let text = metrics.prometheus(filter)?;
        assert!(text.contains("switchyard_llm_call_prompt_tokens_total"));
        assert!(text.contains(&format!("agent_turn_idx=\"{agent_idx}\"")));
        assert!(!text.contains("switchyard_total_requests{"));
        serde_json::to_value(stats)?;
    }
    let user_turn = metrics.stats(MetricFilter {
        session_id: Some("one"),
        agent_id: Some("worker"),
        user_turn_idx: Some(0),
        ..MetricFilter::default()
    });
    assert_eq!(
        user_turn.counter_total("switchyard_llm_call_prompt_tokens_total"),
        Some(112.0)
    );
    let all = metrics.stats(MetricFilter::default());
    assert_eq!(
        all.counter_total("switchyard_inference_requests_total"),
        Some(5.0)
    );
    assert_eq!(
        all.counter_total("switchyard_llm_call_prompt_tokens_total"),
        Some(280.0)
    );
    assert!(
        all.series
            .iter()
            .any(|series| series.name == "switchyard_total_requests")
    );
    assert!(
        metrics
            .stats(MetricFilter {
                session_id: Some("absent"),
                ..MetricFilter::default()
            })
            .series
            .is_empty()
    );

    // A failed answer still includes paid routing usage and failure timing.
    assert!(run(request("failure", "worker"), "failed").await.is_err());
    let failed = metrics.stats(MetricFilter {
        session_id: Some("failure"),
        ..MetricFilter::default()
    });
    assert_eq!(
        failed.counter_total("switchyard_llm_call_prompt_tokens_total"),
        Some(29.0)
    );
    assert_eq!(
        failed.counter_total("switchyard_inference_requests_total"),
        Some(1.0)
    );
    assert_eq!(failed.counter_total("switchyard_prompt_tokens_total"), None);

    // Usage reported before a stream error is retained, but never counted as a successful answer.
    let response = run(request("partial", "worker"), "partial").await?;
    let LlmResponse::Stream(mut stream) = response.llm_response else {
        panic!("expected stream")
    };
    assert!(stream.next().await.unwrap().is_ok());
    assert!(stream.next().await.unwrap().is_err());
    drop(stream);
    let partial = metrics.stats(MetricFilter {
        session_id: Some("partial"),
        ..MetricFilter::default()
    });
    assert_eq!(
        partial.counter_total("switchyard_llm_call_prompt_tokens_total"),
        Some(56.0)
    );
    assert_eq!(
        partial.counter_total("switchyard_prompt_tokens_total"),
        None
    );

    // Reusing a routing response as the answer counts its usage once.
    let mut reused = request("reuse", "worker");
    reused.llm_request.stream = true;
    run(reused, "reuse").await?.llm_response.into_agg().await?;
    let reused = metrics.stats(MetricFilter {
        session_id: Some("reuse"),
        ..MetricFilter::default()
    });
    assert_eq!(
        reused.counter_total("switchyard_llm_call_prompt_tokens_total"),
        Some(17.0)
    );
    assert_eq!(
        reused.counter_total("switchyard_prompt_tokens_total"),
        Some(17.0)
    );
    Ok(())
}
