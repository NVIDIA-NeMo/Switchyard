// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metric labelling inherited from Python

use std::time::Duration;
use std::{
    sync::OnceLock,
    sync::atomic::{AtomicU64, Ordering},
};

use opentelemetry::metrics::ObservableGauge;
use opentelemetry::{KeyValue, global};
use switchyard_libsy::Result;
use switchyard_protocol::{ModelId, Response};

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

pub const fn http_outcome_label(status: Option<u16>) -> &'static str {
    match status {
        Some(200..=299) => "success",
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

/// Records one terminal model call made after routing, preserving the libsy call metric surface.
pub(crate) fn record_answer_call(
    algorithm: &str,
    selected_model: &ModelId,
    duration: Duration,
    result: &Result<Response>,
) {
    let attributes = [
        KeyValue::new("algorithm", algorithm.to_string()),
        KeyValue::new("selected_model", selected_model.to_string()),
        KeyValue::new("outcome", if result.is_ok() { "ok" } else { "error" }),
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
    result: &Result<Response>,
) {
    TOTAL_REQUESTS.fetch_add(1, Ordering::Relaxed);
    let attributes = [KeyValue::new("model", selected_model.to_string())];
    let meter = global::meter("switchyard");
    if result.is_ok() {
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
