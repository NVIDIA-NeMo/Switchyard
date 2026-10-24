// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Anthropic prices cache writes at separate rates for five-minute and one-hour lifetimes.

use switchyard_protocol::Usage;

/// The caller divides this weighted token sum by one million to obtain USD.
pub(super) fn cache_write_cost(
    usage: &Usage,
    short_rate: f64,
    long_rate: Option<f64>,
) -> Option<f64> {
    let written = usage.cache_creation_input_tokens().unwrap_or(0);
    let one_hour = usage
        .cache
        .as_ref()
        .and_then(|cache| cache.cache_creation_1h_input_tokens)
        .unwrap_or(0);
    let five_minute = written.checked_sub(one_hour)?;
    let long_cost = if one_hour > 0 {
        one_hour as f64 * long_rate?
    } else {
        0.0
    };
    Some(five_minute as f64 * short_rate + long_cost)
}
