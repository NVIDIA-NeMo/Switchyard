// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metric labelling inherited from Python

use std::time::{Duration, Instant};

use futures::{StreamExt, stream};
use std::{
    sync::OnceLock,
    sync::atomic::{AtomicU64, Ordering},
};

use opentelemetry::metrics::ObservableGauge;
use opentelemetry::{KeyValue, global};
use switchyard_libsy::Result;
use switchyard_protocol::{LlmResponse, LlmResponseChunk, ModelId, Response};

static TOTAL_REQUESTS: AtomicU64 = AtomicU64::new(0);
static TOTAL_ERRORS: AtomicU64 = AtomicU64::new(0);
static TOTAL_GAUGES: OnceLock<(ObservableGauge<u64>, ObservableGauge<u64>)> = OnceLock::new();

/// Registers process-wide compatibility gauges with the installed global meter provider.
pub fn initialize() {
    TOTAL_GAUGES.get_or_init(|| {
        let meter = global::meter("switchyard");
        let requests = meter
            .u64_observable_gauge("switchyard.total_requests")
            .with_callback(|observer| {
                observer.observe(TOTAL_REQUESTS.load(Ordering::Relaxed), &[]);
            })
            .build();
        let errors = meter
            .u64_observable_gauge("switchyard.total_errors")
            .with_callback(|observer| {
                observer.observe(TOTAL_ERRORS.load(Ordering::Relaxed), &[]);
            })
            .build();
        (requests, errors)
    });
}

pub(crate) const fn is_retryable_http_status(status: u16) -> bool {
    status == 408 || status == 429 || (status >= 500 && status <= 599)
}

/// Maps an HTTP status code (or lack thereof) to a normalized outcome metric label:
/// - `"ok"` for HTTP 2xx success responses.
/// - `"retryable_error"` for retryable HTTP statuses (408, 429, 5xx) or non-HTTP failures (`None`).
/// - `"other_error"` for all other client or non-retryable statuses.
pub const fn http_outcome_label(status: Option<u16>) -> &'static str {
    match status {
        Some(200..=299) => "ok",
        Some(status) if is_retryable_http_status(status) => "retryable_error",
        None => "retryable_error",
        Some(_) => "other_error",
    }
}

/// Limit the cardinality of the HTTP status code.
/// This matches Python, but likely we should log the status code directly. Most of these never
/// appear.
pub const fn http_status_code_label(status: Option<u16>) -> &'static str {
    match status {
        None => "none",
        Some(200) => "200",
        Some(400) => "400",
        Some(401) => "401",
        Some(403) => "403",
        Some(404) => "404",
        Some(408) => "408",
        Some(409) => "409",
        Some(422) => "422",
        Some(429) => "429",
        Some(500) => "500",
        Some(502) => "502",
        Some(503) => "503",
        Some(504) => "504",
        Some(100..=199) => "1xx",
        Some(200..=299) => "2xx",
        Some(300..=399) => "3xx",
        Some(400..=499) => "4xx",
        Some(500..=599) => "5xx",
        Some(_) => "other",
    }
}

pub(crate) fn record_upstream_attempt(status: Option<u16>) {
    global::meter("switchyard")
        .u64_counter("switchyard.upstream_attempts")
        .build()
        .add(
            1,
            &[
                KeyValue::new("outcome", http_outcome_label(status)),
                KeyValue::new("code", http_status_code_label(status)),
            ],
        );
}

/// Records an upstream operation that succeeded after at least one retry.
pub(crate) fn record_retry_recovered() {
    global::meter("switchyard")
        .u64_counter("switchyard.router_retry_recovered")
        .build()
        .add(1, &[]);
}

/// Records the time needed to produce the routing outcome, including classifier calls,
/// target resolution, request rewrites, and decision publishing.
pub(crate) fn record_routing_overhead(algorithm: &str, overhead: Duration) {
    global::meter("switchyard")
        .f64_histogram("switchyard.routing_overhead_ms")
        .build()
        .record(
            overhead.as_secs_f64() * 1000.0,
            &[KeyValue::new("algorithm", algorithm.to_string())],
        );
}

/// Records the final outcome, retaining ownership until a streamed answer ends or is dropped.
pub(crate) fn observe_routed_request(
    algorithm: &str,
    selected_model: &ModelId,
    answer_duration: Option<Duration>,
    result: Result<Response>,
) -> Result<Response> {
    let mut call = RoutedCallMetrics {
        algorithm: algorithm.to_owned(),
        model: selected_model.clone(),
        duration: answer_duration,
        started: Instant::now(),
        outcome: Some(result.is_ok()),
    };
    result.map(|mut response| {
        if let LlmResponse::Stream(stream) = response.llm_response {
            call.outcome = None;
            response.llm_response = LlmResponse::Stream(Box::pin(stream::unfold(
                (stream, call),
                |(mut stream, mut call)| async move {
                    let Some(item) = stream.next().await else {
                        call.outcome.get_or_insert(true);
                        return None;
                    };
                    match &item {
                        Err(_) => call.outcome = Some(false),
                        Ok(event) => {
                            for chunk in event.normalized() {
                                match chunk {
                                    LlmResponseChunk::StreamError { .. }
                                    | LlmResponseChunk::DecodeError { .. } => {
                                        call.outcome = Some(false)
                                    }
                                    LlmResponseChunk::MessageStop { .. } => {
                                        call.outcome.get_or_insert(true);
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                    Some((item, (stream, call)))
                },
            )));
        }
        response
    })
}

struct RoutedCallMetrics {
    algorithm: String,
    model: ModelId,
    duration: Option<Duration>,
    started: Instant,
    // Dropping an open stream is a failure; a terminal message permits a successful drop.
    outcome: Option<bool>,
}

impl Drop for RoutedCallMetrics {
    fn drop(&mut self) {
        let is_success = self.outcome.unwrap_or(false);
        let duration = self
            .duration
            .map(|duration| duration + self.started.elapsed());
        if let Some(duration) = duration {
            record_answer_call(&self.algorithm, &self.model, duration, is_success);
        }
        record_routed_request(&self.model, duration, is_success);
    }
}

/// Records one terminal model call made after routing, preserving the libsy call metric surface.
pub(crate) fn record_answer_call(
    algorithm: &str,
    selected_model: &ModelId,
    duration: Duration,
    is_success: bool,
) {
    let attributes = [
        KeyValue::new("algorithm", algorithm.to_string()),
        KeyValue::new("selected_model", selected_model.to_string()),
        KeyValue::new("outcome", if is_success { "ok" } else { "error" }),
    ];
    let meter = global::meter("switchyard");
    meter
        .u64_counter("switchyard.llm_calls")
        .build()
        .add(1, &attributes);
    meter
        .f64_histogram("switchyard.llm_call_duration_ms")
        .build()
        .record(duration.as_secs_f64() * 1000.0, &attributes);
}

/// Records one terminal routed request after the algorithm has produced an outcome.
pub(crate) fn record_routed_request(
    selected_model: &ModelId,
    answer_duration: Option<Duration>,
    is_success: bool,
) {
    TOTAL_REQUESTS.fetch_add(1, Ordering::Relaxed);
    let attributes = [KeyValue::new("model", selected_model.to_string())];
    let meter = global::meter("switchyard");
    if is_success {
        meter
            .u64_counter("switchyard.requests")
            .build()
            .add(1, &attributes);
        if let Some(duration) = answer_duration {
            meter
                .f64_histogram("switchyard.model_call_latency_ms")
                .build()
                .record(duration.as_secs_f64() * 1000.0, &attributes);
        }
    } else {
        TOTAL_ERRORS.fetch_add(1, Ordering::Relaxed);
        meter
            .u64_counter("switchyard.errors")
            .build()
            .add(1, &attributes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outcome_labels_match_the_retry_policy() {
        assert_eq!(http_outcome_label(Some(200)), "ok");
        for status in [408, 429, 500, 502, 599] {
            assert!(is_retryable_http_status(status));
            assert_eq!(http_outcome_label(Some(status)), "retryable_error");
        }
        for status in [400, 409, 499, 600] {
            assert!(!is_retryable_http_status(status));
            assert_eq!(http_outcome_label(Some(status)), "other_error");
        }
        assert_eq!(http_outcome_label(None), "retryable_error");
    }
}
