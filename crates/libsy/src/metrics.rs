// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Model-call token accounting and in-process metric queries.

use opentelemetry::{KeyValue, global};
use switchyard_protocol::Usage;

#[cfg(feature = "metrics")]
mod collector;
#[cfg(feature = "metrics")]
pub use collector::{
    HistogramBucket, MetricFilter, MetricSeries, MetricValue, Metrics, MetricsSnapshot,
};

/// Records provider-reported usage for one model call, including routing work.
///
/// The LLM client driver calls this automatically for LLM and Decision Model calls.
/// Hosts serving libsy calls themselves can record usage here once per call.
/// `call_kind` is `routing`, `answer`, or `decision`. Prompt includes cache reads and
/// writes; separate counters retain those details. Unknown counts are omitted.
pub fn record_usage(algorithm: &str, model: &str, call_kind: &str, usage: &Usage) {
    let attributes = crate::metric_context::attributes(&[
        KeyValue::new("algorithm", algorithm.to_owned()),
        KeyValue::new("selected_model", model.to_owned()),
        KeyValue::new("call_kind", call_kind.to_owned()),
    ]);
    let cached = usage.cached_input_tokens();
    let creation = usage.cache_creation_input_tokens();
    let prompt =
        (usage.input_tokens.is_some() || cached.is_some() || creation.is_some()).then(|| {
            usage
                .input_tokens
                .unwrap_or(0)
                .saturating_add(cached.unwrap_or(0))
                .saturating_add(creation.unwrap_or(0))
        });
    let meter = global::meter("switchyard");
    for (name, value) in [
        ("switchyard.llm_call.input_tokens", usage.input_tokens),
        ("switchyard.llm_call.prompt_tokens", prompt),
        ("switchyard.llm_call.total_tokens", usage.total_tokens),
        ("switchyard.llm_call.completion_tokens", usage.output_tokens),
        ("switchyard.llm_call.cached_tokens", cached),
        ("switchyard.llm_call.cache_creation_tokens", creation),
        (
            "switchyard.llm_call.reasoning_tokens",
            usage.reasoning_tokens,
        ),
    ] {
        if let Some(value) = value {
            meter.u64_counter(name).build().add(value, &attributes);
        }
    }
}
