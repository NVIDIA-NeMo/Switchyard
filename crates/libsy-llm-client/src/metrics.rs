// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metric labelling inherited from Python

use opentelemetry::{global, KeyValue};

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
