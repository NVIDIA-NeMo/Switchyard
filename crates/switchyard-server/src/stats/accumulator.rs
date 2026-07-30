// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thread-safe stats accumulator and serializable snapshot schema.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::Arc;

use parking_lot::{Mutex, MutexGuard};
use serde::Serialize;

use super::cache_eligibility::PrefixProbe;

const MAX_LATENCY_SAMPLES: usize = 10_000;

/// Normalized token counters recorded by the server.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct TokenUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cached_tokens: u64,
    pub cache_creation_tokens: u64,
    pub cacheable_prompt_tokens: u64,
    pub reasoning_tokens: u64,
}

/// Thread-safe process-local stats store.
#[derive(Clone, Debug, Default)]
pub(crate) struct StatsAccumulator {
    inner: Arc<Mutex<StatsAccumulatorInner>>,
}

impl StatsAccumulator {
    /// Records one successful routed backend call.
    pub(crate) fn record_success(
        &self,
        model: impl Into<String>,
        backend_latency_ms: f64,
        tier: Option<&str>,
    ) {
        let mut inner = self.lock();
        inner.total_requests = inner.total_requests.saturating_add(1);
        let model = model.into();
        let tier = normalized_tier(tier);
        let stats = inner.model_stats_mut(model.clone());
        stats.calls = stats.calls.saturating_add(1);
        stats.model_call_latency.record(backend_latency_ms);
        if let Some(tier) = tier {
            stats.tiers.insert(tier.to_string());
            let tier_stats = inner.tier_stats_mut(tier, &model);
            tier_stats.calls = tier_stats.calls.saturating_add(1);
        }
    }

    /// Records one failed routed backend call.
    pub(crate) fn record_error(&self, model: impl Into<String>, tier: Option<&str>) {
        let mut inner = self.lock();
        inner.total_requests = inner.total_requests.saturating_add(1);
        inner.total_errors = inner.total_errors.saturating_add(1);
        let model = model.into();
        let stats = inner.model_stats_mut(model.clone());
        stats.errors = stats.errors.saturating_add(1);
        if let Some(tier) = normalized_tier(tier) {
            stats.tiers.insert(tier.to_string());
            inner.tier_stats_mut(tier, &model);
        }
    }

    /// Records usage and terminal latency after a successful routed call.
    pub(crate) fn record_usage(
        &self,
        model: impl Into<String>,
        usage: TokenUsage,
        total_latency_ms: f64,
        tier: Option<&str>,
    ) {
        let mut inner = self.lock();
        let model = model.into();
        let stats = inner.model_stats_mut(model.clone());
        stats.add_usage(usage);
        stats.total_latency.record(total_latency_ms);
        if let Some(tier) = normalized_tier(tier) {
            stats.tiers.insert(tier.to_string());
            inner.tier_stats_mut(tier, &model).add_usage(usage);
        }
    }

    /// Records routing time for one completed algorithm run.
    pub(crate) fn record_routing_overhead(&self, routing_overhead_ms: f64) {
        self.lock().routing_overhead.record(routing_overhead_ms);
    }

    /// Records one successful classifier or judge call.
    pub(crate) fn record_classifier_success(
        &self,
        model: impl Into<String>,
        usage: Option<TokenUsage>,
        latency_ms: f64,
    ) {
        let mut inner = self.lock();
        inner.classifier_requests = inner.classifier_requests.saturating_add(1);
        let stats = inner.classifier_stats_mut(model.into());
        stats.calls = stats.calls.saturating_add(1);
        if let Some(usage) = usage {
            stats.add_usage(usage);
        }
        stats.model_call_latency.record(latency_ms);
        stats.total_latency.record(latency_ms);
    }

    /// Records one failed classifier or judge call.
    pub(crate) fn record_classifier_error(&self, model: impl Into<String>) {
        let mut inner = self.lock();
        inner.classifier_requests = inner.classifier_requests.saturating_add(1);
        inner.classifier_errors = inner.classifier_errors.saturating_add(1);
        let stats = inner.classifier_stats_mut(model.into());
        stats.errors = stats.errors.saturating_add(1);
    }

    /// Returns the cache-eligible fraction for `model` and records the prefix as seen.
    pub(crate) fn prefix_eligibility(&self, model: &str, probe: &PrefixProbe) -> f64 {
        let mut inner = self.lock();
        let stats = inner.model_stats_mut(model.to_string());
        let fraction = probe.eligible_fraction(&stats.seen_prefixes);
        if let Some(hash) = probe.full_hash() {
            stats.seen_prefixes.insert(hash);
        }
        fraction
    }

    /// Returns a serializable point-in-time snapshot.
    pub(crate) fn snapshot(&self) -> StatsSnapshot {
        let inner = self.lock().clone();
        inner.snapshot()
    }

    fn lock(&self) -> MutexGuard<'_, StatsAccumulatorInner> {
        self.inner.lock()
    }
}

fn normalized_tier(tier: Option<&str>) -> Option<&str> {
    tier.map(str::trim).filter(|tier| !tier.is_empty())
}

#[derive(Clone, Debug, Default)]
struct StatsAccumulatorInner {
    by_model: BTreeMap<String, ModelStats>,
    by_tier: BTreeMap<String, TierStats>,
    total_requests: u64,
    total_errors: u64,
    routing_overhead: LatencyHistogram,
    by_classifier: BTreeMap<String, ModelStats>,
    classifier_requests: u64,
    classifier_errors: u64,
}

impl StatsAccumulatorInner {
    fn model_stats_mut(&mut self, model: String) -> &mut ModelStats {
        self.by_model.entry(model).or_default()
    }

    fn classifier_stats_mut(&mut self, model: String) -> &mut ModelStats {
        self.by_classifier.entry(model).or_default()
    }

    fn tier_stats_mut(&mut self, tier: &str, model: &str) -> &mut TierStats {
        let stats = self.by_tier.entry(tier.to_string()).or_default();
        stats.models.insert(model.to_string());
        stats
    }

    fn snapshot(&self) -> StatsSnapshot {
        let (models, total_tokens) = build_model_snapshots(&self.by_model, self.total_requests);
        let classifier = build_classifier_snapshot(
            &self.by_classifier,
            self.classifier_requests,
            self.classifier_errors,
        );
        StatsSnapshot {
            total_requests: self.total_requests,
            total_errors: self.total_errors,
            total_tokens,
            models,
            tiers: tier_snapshots(&self.by_tier, total_tokens.total, self.total_requests),
            routing_overhead: self.routing_overhead.snapshot(),
            classifier,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct ModelStats {
    calls: u64,
    errors: u64,
    prompt_tokens: u64,
    max_observed_context_tokens: u64,
    completion_tokens: u64,
    cached_tokens: u64,
    cache_creation_tokens: u64,
    cacheable_prompt_tokens: u64,
    reasoning_tokens: u64,
    seen_prefixes: HashSet<u64>,
    model_call_latency: LatencyHistogram,
    total_latency: LatencyHistogram,
    tiers: BTreeSet<String>,
}

impl ModelStats {
    fn add_usage(&mut self, usage: TokenUsage) {
        self.prompt_tokens = self.prompt_tokens.saturating_add(usage.prompt_tokens);
        self.completion_tokens = self
            .completion_tokens
            .saturating_add(usage.completion_tokens);
        self.cached_tokens = self.cached_tokens.saturating_add(usage.cached_tokens);
        self.cache_creation_tokens = self
            .cache_creation_tokens
            .saturating_add(usage.cache_creation_tokens);
        self.cacheable_prompt_tokens = self
            .cacheable_prompt_tokens
            .saturating_add(usage.cacheable_prompt_tokens);
        self.reasoning_tokens = self.reasoning_tokens.saturating_add(usage.reasoning_tokens);
        self.max_observed_context_tokens = self
            .max_observed_context_tokens
            .max(usage.prompt_tokens.saturating_add(usage.completion_tokens));
    }
}

#[derive(Clone, Debug, Default)]
struct TierStats {
    models: BTreeSet<String>,
    calls: u64,
    prompt_tokens: u64,
    completion_tokens: u64,
}

impl TierStats {
    fn add_usage(&mut self, usage: TokenUsage) {
        self.prompt_tokens = self.prompt_tokens.saturating_add(usage.prompt_tokens);
        self.completion_tokens = self
            .completion_tokens
            .saturating_add(usage.completion_tokens);
    }
}

#[derive(Clone, Debug)]
struct LatencyHistogram {
    count: u64,
    total_ms: f64,
    min_ms: f64,
    max_ms: f64,
    samples: Vec<f64>,
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self {
            count: 0,
            total_ms: 0.0,
            min_ms: f64::INFINITY,
            max_ms: 0.0,
            samples: Vec::new(),
        }
    }
}

impl LatencyHistogram {
    fn record(&mut self, latency_ms: f64) {
        if !latency_ms.is_finite() {
            tracing::debug!(latency_ms, "dropping non-finite latency sample");
            return;
        }
        let latency_ms = latency_ms.max(0.0);
        self.count = self.count.saturating_add(1);
        self.total_ms += latency_ms;
        self.min_ms = self.min_ms.min(latency_ms);
        self.max_ms = self.max_ms.max(latency_ms);
        if self.samples.len() < MAX_LATENCY_SAMPLES {
            self.samples.push(latency_ms);
        } else {
            let index = self.count.saturating_sub(1) as usize % MAX_LATENCY_SAMPLES;
            self.samples[index] = latency_ms;
        }
    }

    fn snapshot(&self) -> LatencyHistogramSnapshot {
        if self.count == 0 {
            return LatencyHistogramSnapshot::default();
        }
        let mut samples = self.samples.to_vec();
        samples.sort_by(f64::total_cmp);
        let count = samples.len();
        let p99_index = count.saturating_sub(1).min((count as f64 * 0.99) as usize);
        LatencyHistogramSnapshot {
            count: self.count,
            total_ms: round2(self.total_ms),
            min_ms: round2(self.min_ms),
            max_ms: round2(self.max_ms),
            avg_ms: round2(self.total_ms / self.count as f64),
            p50_ms: samples.get(count / 2).copied().map(round2).unwrap_or(0.0),
            p99_ms: samples.get(p99_index).copied().map(round2).unwrap_or(0.0),
        }
    }
}

/// Full JSON stats snapshot.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub(crate) struct StatsSnapshot {
    pub total_requests: u64,
    pub total_errors: u64,
    pub total_tokens: TokenTotals,
    pub models: BTreeMap<String, ModelStatsSnapshot>,
    pub tiers: BTreeMap<String, TierStatsSnapshot>,
    pub routing_overhead: LatencyHistogramSnapshot,
    pub classifier: ClassifierStatsSnapshot,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub(crate) struct ClassifierStatsSnapshot {
    pub total_requests: u64,
    pub total_errors: u64,
    pub total_tokens: TokenTotals,
    pub models: BTreeMap<String, ModelStatsSnapshot>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub(crate) struct TokenTotals {
    pub prompt: u64,
    pub completion: u64,
    pub cached: u64,
    pub cache_creation: u64,
    pub reasoning: u64,
    pub total: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct ModelStatsSnapshot {
    pub calls: u64,
    pub errors: u64,
    pub request_pct: f64,
    pub prompt_tokens: u64,
    pub max_observed_context_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub token_pct: f64,
    pub cached_tokens: u64,
    pub cache_creation_tokens: u64,
    pub reasoning_tokens: u64,
    pub avg_prompt_tokens: f64,
    pub avg_completion_tokens: f64,
    pub cache_hit_rate: f64,
    pub theoretical_cache_hit_rate: f64,
    pub model_call_latency: LatencyHistogramSnapshot,
    pub total_latency: LatencyHistogramSnapshot,
    pub tiers: BTreeSet<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct TierStatsSnapshot {
    pub models: BTreeSet<String>,
    pub calls: u64,
    pub request_pct: f64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub token_pct: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub(crate) struct LatencyHistogramSnapshot {
    pub count: u64,
    pub total_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub avg_ms: f64,
    pub p50_ms: f64,
    pub p99_ms: f64,
}

fn build_model_snapshots(
    by_model: &BTreeMap<String, ModelStats>,
    total_requests: u64,
) -> (BTreeMap<String, ModelStatsSnapshot>, TokenTotals) {
    let mut totals = TokenTotals::default();
    for stats in by_model.values() {
        totals.prompt = totals.prompt.saturating_add(stats.prompt_tokens);
        totals.completion = totals.completion.saturating_add(stats.completion_tokens);
        totals.cached = totals.cached.saturating_add(stats.cached_tokens);
        totals.cache_creation = totals
            .cache_creation
            .saturating_add(stats.cache_creation_tokens);
        totals.reasoning = totals.reasoning.saturating_add(stats.reasoning_tokens);
    }
    totals.total = totals.prompt.saturating_add(totals.completion);

    let models = by_model
        .iter()
        .map(|(model, stats)| {
            let total_tokens = stats.prompt_tokens.saturating_add(stats.completion_tokens);
            let snapshot = ModelStatsSnapshot {
                calls: stats.calls,
                errors: stats.errors,
                request_pct: percentage(stats.calls, total_requests),
                prompt_tokens: stats.prompt_tokens,
                max_observed_context_tokens: stats.max_observed_context_tokens,
                completion_tokens: stats.completion_tokens,
                total_tokens,
                token_pct: percentage(total_tokens, totals.total),
                cached_tokens: stats.cached_tokens,
                cache_creation_tokens: stats.cache_creation_tokens,
                reasoning_tokens: stats.reasoning_tokens,
                avg_prompt_tokens: average(stats.prompt_tokens, stats.calls),
                avg_completion_tokens: average(stats.completion_tokens, stats.calls),
                cache_hit_rate: ratio4(stats.cached_tokens, stats.prompt_tokens),
                theoretical_cache_hit_rate: ratio4(
                    stats.cacheable_prompt_tokens,
                    stats.prompt_tokens,
                ),
                model_call_latency: stats.model_call_latency.snapshot(),
                total_latency: stats.total_latency.snapshot(),
                tiers: stats.tiers.clone(),
            };
            (model.clone(), snapshot)
        })
        .collect();
    (models, totals)
}

fn build_classifier_snapshot(
    models: &BTreeMap<String, ModelStats>,
    total_requests: u64,
    total_errors: u64,
) -> ClassifierStatsSnapshot {
    let (models, total_tokens) = build_model_snapshots(models, total_requests);
    ClassifierStatsSnapshot {
        total_requests,
        total_errors,
        total_tokens,
        models,
    }
}

fn tier_snapshots(
    tiers: &BTreeMap<String, TierStats>,
    total_tokens: u64,
    total_requests: u64,
) -> BTreeMap<String, TierStatsSnapshot> {
    tiers
        .iter()
        .map(|(tier, stats)| {
            let tier_tokens = stats.prompt_tokens.saturating_add(stats.completion_tokens);
            (
                tier.clone(),
                TierStatsSnapshot {
                    models: stats.models.clone(),
                    calls: stats.calls,
                    request_pct: percentage(stats.calls, total_requests),
                    prompt_tokens: stats.prompt_tokens,
                    completion_tokens: stats.completion_tokens,
                    total_tokens: tier_tokens,
                    token_pct: percentage(tier_tokens, total_tokens),
                },
            )
        })
        .collect()
}

fn percentage(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        round2(numerator as f64 / denominator as f64 * 100.0)
    }
}

fn average(total: u64, count: u64) -> f64 {
    if count == 0 {
        0.0
    } else {
        round2(total as f64 / count as f64)
    }
}

fn ratio4(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        round4(numerator as f64 / denominator as f64)
    }
}

fn round2(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

fn round4(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::stats::cache_eligibility::prefix_probe;

    fn usage(prompt: u64, completion: u64) -> TokenUsage {
        TokenUsage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            ..TokenUsage::default()
        }
    }

    #[test]
    fn snapshot_aggregates_backend_and_classifier_stats() {
        let stats = StatsAccumulator::default();
        stats.record_success("model/strong", 10.0, Some("strong"));
        stats.record_usage(
            "model/strong",
            TokenUsage {
                cached_tokens: 4,
                cache_creation_tokens: 2,
                reasoning_tokens: 3,
                ..usage(10, 5)
            },
            15.0,
            Some("strong"),
        );
        stats.record_routing_overhead(5.0);
        stats.record_success("model/weak", 20.0, Some("weak"));
        stats.record_usage("model/weak", usage(20, 10), 30.0, Some("weak"));
        stats.record_routing_overhead(10.0);
        stats.record_classifier_success("gemini-3.5-flash", Some(usage(1_000_000, 0)), 8.0);
        let snapshot = stats.snapshot();
        assert_eq!(snapshot.total_tokens.prompt, 30);
        assert_eq!(snapshot.total_tokens.cached, 4);
        assert_eq!(
            snapshot.models["model/strong"].max_observed_context_tokens,
            15
        );
        assert_eq!(
            snapshot.models["model/strong"].theoretical_cache_hit_rate,
            0.0
        );
        assert_eq!(snapshot.tiers["strong"].request_pct, 50.0);
        assert_eq!(snapshot.routing_overhead.p50_ms, 10.0);
        assert_eq!(snapshot.routing_overhead.p99_ms, 10.0);
        assert_eq!(snapshot.classifier.total_requests, 1);
        assert_eq!(snapshot.classifier.total_tokens.prompt, 1_000_000);

        stats.record_success("model/strong", 1.0, Some("weak"));
        stats.record_success("model/other", 1.0, Some("strong"));
        let snapshot = stats.snapshot();
        assert_eq!(
            snapshot.models["model/strong"].tiers,
            BTreeSet::from(["strong".to_string(), "weak".to_string()])
        );
        assert_eq!(
            snapshot.tiers["strong"].models,
            BTreeSet::from(["model/other".to_string(), "model/strong".to_string()])
        );
    }

    #[test]
    fn theoretical_cache_hit_rate_is_switch_aware() {
        let stats = StatsAccumulator::default();
        let first = prefix_probe(&json!({
            "messages": [{"role": "user", "content": "aaaa"}],
        }));
        let first_eligible = stats.prefix_eligibility("model/a", &first);
        stats.record_usage(
            "model/a",
            TokenUsage {
                cacheable_prompt_tokens: (100.0 * first_eligible).round() as u64,
                ..usage(100, 4)
            },
            1.0,
            None,
        );

        let second = prefix_probe(&json!({
            "messages": [
                {"role": "user", "content": "aaaa"},
                {"role": "user", "content": "bbbb"},
            ],
        }));
        let second_eligible = stats.prefix_eligibility("model/a", &second);
        stats.record_usage(
            "model/a",
            TokenUsage {
                cacheable_prompt_tokens: (100.0 * second_eligible).round() as u64,
                ..usage(100, 4)
            },
            1.0,
            None,
        );

        assert_eq!(first_eligible, 0.0);
        assert_eq!(second_eligible, 0.5);
        assert_eq!(
            stats.snapshot().models["model/a"].theoretical_cache_hit_rate,
            0.25
        );
        assert_eq!(stats.prefix_eligibility("model/b", &second), 0.0);
    }
}
