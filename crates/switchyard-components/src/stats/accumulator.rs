// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thread-safe stats accumulator and serializable snapshot schema.

use std::cmp::{Ordering, Reverse};
use std::collections::{btree_map::Entry, BTreeMap, BinaryHeap, HashSet};
use std::sync::Arc;

use parking_lot::{Mutex, MutexGuard};
use serde::{Deserialize, Serialize};
use switchyard_core::Result;

use super::cost::{estimate_cost, CostEstimate};
use super::{PrefixProbe, TokenUsage};

const MAX_LATENCY_SAMPLES: usize = 10_000;

/// Thread-safe stats store shared by stats processors and backend wrappers.
#[derive(Clone, Debug)]
pub struct StatsAccumulator {
    inner: Arc<Mutex<StatsAccumulatorInner>>,
}

impl Default for StatsAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl StatsAccumulator {
    /// Creates an empty stats accumulator.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(StatsAccumulatorInner::default())),
        }
    }

    /// Records a successful backend call.
    pub fn record_success(
        &self,
        model: impl Into<String>,
        backend_latency_ms: Option<f64>,
        tier: Option<&str>,
    ) -> Result<()> {
        let mut inner = self.lock();
        inner.total_requests = inner.total_requests.saturating_add(1);
        let model = model.into();
        let tier = tier.map(str::trim).filter(|tier| !tier.is_empty());
        {
            let stats = inner.model_stats_mut(model.clone());
            stats.calls = stats.calls.saturating_add(1);
            if let Some(tier) = tier {
                stats.tier = Some(tier.to_string());
            }
            if let Some(latency) = backend_latency_ms {
                stats.model_call_latency.record(latency);
            }
        }
        if let Some(tier) = tier {
            if let Some(tier_stats) = inner.tier_stats_mut(tier, &model) {
                tier_stats.calls = tier_stats.calls.saturating_add(1);
            }
        } else {
            inner.record_untiered_success(&model);
        }
        Ok(())
    }

    /// Records a backend error.
    pub fn record_error(&self, model: impl Into<String>, tier: Option<&str>) -> Result<()> {
        let mut inner = self.lock();
        inner.total_requests = inner.total_requests.saturating_add(1);
        inner.total_errors = inner.total_errors.saturating_add(1);
        let stats = inner.model_stats_mut(model.into());
        stats.errors = stats.errors.saturating_add(1);
        if let Some(tier) = tier {
            stats.tier = Some(tier.to_string());
        }
        Ok(())
    }

    /// Returns the cache-eligible fraction for `model` and records the prefix as seen.
    ///
    /// Switch-aware: a prefix counts only if this model was previously sent it, so a
    /// switch to a cold model yields 0 and a return credits only the shared prefix.
    pub fn prefix_eligibility(&self, model: &str, probe: &PrefixProbe) -> f64 {
        let mut inner = self.lock();
        let stats = inner.model_stats_mut(model.to_string());
        let fraction = probe.eligible_fraction(&stats.seen_prefixes);
        if let Some(hash) = probe.full_hash() {
            stats.seen_prefixes.insert(hash);
        }
        fraction
    }

    /// Records token usage and end-to-end latency for a completed response.
    pub fn record_usage(
        &self,
        model: impl Into<String>,
        usage: TokenUsage,
        total_latency_ms: Option<f64>,
        routing_overhead_ms: Option<f64>,
        tier: Option<&str>,
    ) -> Result<()> {
        self.record_usage_inner(
            model,
            usage,
            total_latency_ms,
            routing_overhead_ms,
            tier,
            TierCallAttribution::LegacyPendingUntiered,
        )
    }

    /// Records usage and attaches one previously untiered success to `tier`.
    ///
    /// This is only for compatibility paths that recorded success before the
    /// route label was available. Normal stats backends record success with the
    /// tier already present, so their usage events must leave this flag false.
    pub fn record_usage_with_success_was_untiered(
        &self,
        model: impl Into<String>,
        usage: TokenUsage,
        total_latency_ms: Option<f64>,
        routing_overhead_ms: Option<f64>,
        tier: Option<&str>,
    ) -> Result<()> {
        self.record_usage_inner(
            model,
            usage,
            total_latency_ms,
            routing_overhead_ms,
            tier,
            TierCallAttribution::ExplicitUntiered,
        )
    }

    /// Records usage after the corresponding success call was already attributed.
    ///
    /// `StatsLlmBackend` and profile runtimes record success before the response
    /// processor records tokens. Those internal paths must not consume a legacy
    /// pending untiered success that belongs to some other direct accumulator caller.
    pub fn record_usage_after_success_attribution(
        &self,
        model: impl Into<String>,
        usage: TokenUsage,
        total_latency_ms: Option<f64>,
        routing_overhead_ms: Option<f64>,
        tier: Option<&str>,
    ) -> Result<()> {
        self.record_usage_inner(
            model,
            usage,
            total_latency_ms,
            routing_overhead_ms,
            tier,
            TierCallAttribution::AlreadyRecorded,
        )
    }

    fn record_usage_inner(
        &self,
        model: impl Into<String>,
        usage: TokenUsage,
        total_latency_ms: Option<f64>,
        routing_overhead_ms: Option<f64>,
        tier: Option<&str>,
        tier_call_attribution: TierCallAttribution,
    ) -> Result<()> {
        let mut inner = self.lock();
        let model = model.into();
        {
            let stats = inner.model_stats_mut(model.clone());
            stats.prompt_tokens = stats.prompt_tokens.saturating_add(usage.prompt_tokens);
            stats.completion_tokens = stats
                .completion_tokens
                .saturating_add(usage.completion_tokens);
            stats.cached_tokens = stats.cached_tokens.saturating_add(usage.cached_tokens);
            stats.cache_creation_tokens = stats
                .cache_creation_tokens
                .saturating_add(usage.cache_creation_tokens);
            stats.cacheable_prompt_tokens = stats
                .cacheable_prompt_tokens
                .saturating_add(usage.cacheable_prompt_tokens);
            stats.reasoning_tokens = stats
                .reasoning_tokens
                .saturating_add(usage.reasoning_tokens);
            if let Some(tier) = tier {
                stats.tier = Some(tier.to_string());
            }
            if let Some(latency) = total_latency_ms {
                stats.total_latency.record(latency);
            }
        }
        if let Some(tier) = tier {
            let tier = tier.trim();
            if !tier.is_empty() {
                let should_attribute_call = match tier_call_attribution {
                    TierCallAttribution::LegacyPendingUntiered => {
                        inner.consume_untiered_success(&model)
                    }
                    TierCallAttribution::ExplicitUntiered => {
                        inner.consume_untiered_success(&model);
                        true
                    }
                    TierCallAttribution::AlreadyRecorded => false,
                };
                if let Some(tier_stats) = inner.tier_stats_mut(tier, &model) {
                    if should_attribute_call {
                        tier_stats.calls = tier_stats.calls.saturating_add(1);
                    }
                    tier_stats.prompt_tokens =
                        tier_stats.prompt_tokens.saturating_add(usage.prompt_tokens);
                    tier_stats.completion_tokens = tier_stats
                        .completion_tokens
                        .saturating_add(usage.completion_tokens);
                }
            }
        }
        if let Some(overhead) = routing_overhead_ms {
            inner.routing_overhead.record(overhead);
        }
        Ok(())
    }

    /// Records one LLM-classifier overhead call.
    ///
    /// The classifier's per-request call is not part of the routed-backend
    /// chain and must stay out of `by_model` — otherwise the default TB-lite
    /// config (classifier model == weak-tier model) double-counts the spend.
    /// `record_classifier_usage` writes to a dedicated bucket; the snapshot
    /// exposes it under `classifier.models` with its own `cost_estimate`.
    pub fn record_classifier_usage(
        &self,
        model: impl Into<String>,
        usage: TokenUsage,
        latency_ms: Option<f64>,
    ) -> Result<()> {
        let mut inner = self.lock();
        inner.classifier_requests = inner.classifier_requests.saturating_add(1);
        let stats = inner.classifier_stats_mut(model.into());
        stats.calls = stats.calls.saturating_add(1);
        stats.prompt_tokens = stats.prompt_tokens.saturating_add(usage.prompt_tokens);
        stats.completion_tokens = stats
            .completion_tokens
            .saturating_add(usage.completion_tokens);
        stats.cached_tokens = stats.cached_tokens.saturating_add(usage.cached_tokens);
        stats.cache_creation_tokens = stats
            .cache_creation_tokens
            .saturating_add(usage.cache_creation_tokens);
        stats.reasoning_tokens = stats
            .reasoning_tokens
            .saturating_add(usage.reasoning_tokens);
        if let Some(latency) = latency_ms {
            stats.model_call_latency.record(latency);
            stats.total_latency.record(latency);
        }
        Ok(())
    }

    /// Records a classifier-call failure.
    ///
    /// Bumps the classifier `total_requests` (so the failure shows up in
    /// the per-request distribution) and `total_errors`, plus the
    /// per-model `errors` counter.  Does **not** bump `calls` — that
    /// field counts completed (token-bearing) calls only, so the
    /// `errors / (calls + errors)` ratio is the failure rate.
    pub fn record_classifier_error(&self, model: impl Into<String>) -> Result<()> {
        let mut inner = self.lock();
        inner.classifier_requests = inner.classifier_requests.saturating_add(1);
        inner.classifier_errors = inner.classifier_errors.saturating_add(1);
        let stats = inner.classifier_stats_mut(model.into());
        stats.errors = stats.errors.saturating_add(1);
        Ok(())
    }

    /// Records one routing decision source for a profile family.
    ///
    /// This is intentionally separate from model/tier accounting: a stage-router can
    /// choose `weak` because of an override, a dimensions score, an LLM-classifier
    /// verdict, or a fail-open default, and those explanations are useful even
    /// when they all land on the same backend model.
    pub fn record_routing_decision(
        &self,
        profile_type: impl Into<String>,
        source: impl Into<String>,
    ) -> Result<()> {
        let mut inner = self.lock();
        let sources = inner
            .routing_decisions
            .entry(profile_type.into())
            .or_default();
        let count = sources.entry(source.into()).or_insert(0);
        *count = count.saturating_add(1);
        Ok(())
    }

    /// Records one planner-overhead call's token usage.
    ///
    /// Parallel to :meth:`record_classifier_usage`: the planner runs
    /// out-of-band from the routed-backend chain and lives in its own
    /// bucket so the planner-model spend never aliases the executor
    /// model when both happen to share a model id.  Snapshot exposes
    /// the bucket under :attr:`StatsSnapshot.planner` and rolls
    /// :attr:`CostEstimate.planner_cost` into the headline
    /// :attr:`CostEstimate.total_cost`.
    pub fn record_planner_usage(
        &self,
        model: impl Into<String>,
        usage: TokenUsage,
        latency_ms: Option<f64>,
    ) -> Result<()> {
        let mut inner = self.lock();
        inner.planner_requests = inner.planner_requests.saturating_add(1);
        let stats = inner.planner_stats_mut(model.into());
        stats.calls = stats.calls.saturating_add(1);
        stats.prompt_tokens = stats.prompt_tokens.saturating_add(usage.prompt_tokens);
        stats.completion_tokens = stats
            .completion_tokens
            .saturating_add(usage.completion_tokens);
        stats.cached_tokens = stats.cached_tokens.saturating_add(usage.cached_tokens);
        stats.cache_creation_tokens = stats
            .cache_creation_tokens
            .saturating_add(usage.cache_creation_tokens);
        stats.reasoning_tokens = stats
            .reasoning_tokens
            .saturating_add(usage.reasoning_tokens);
        if let Some(latency) = latency_ms {
            stats.model_call_latency.record(latency);
            stats.total_latency.record(latency);
        }
        Ok(())
    }

    /// Records a planner-call failure.
    ///
    /// Mirror of :meth:`record_classifier_error`.  Used by
    /// :class:`switchyard.lib.processors.plan_execute.PlanningRequestProcessor`
    /// in its fail-open branch so silent planner failures still count
    /// against the planner bucket on :attr:`StatsSnapshot.planner`.
    pub fn record_planner_error(&self, model: impl Into<String>) -> Result<()> {
        let mut inner = self.lock();
        inner.planner_requests = inner.planner_requests.saturating_add(1);
        inner.planner_errors = inner.planner_errors.saturating_add(1);
        let stats = inner.planner_stats_mut(model.into());
        stats.errors = stats.errors.saturating_add(1);
        Ok(())
    }

    /// Returns a computed snapshot suitable for JSON serialization.
    pub fn snapshot(&self) -> Result<StatsSnapshot> {
        let inner = self.lock().clone();
        Ok(inner.snapshot())
    }

    /// Clears all counters.
    pub fn reset(&self) -> Result<()> {
        let mut inner = self.lock();
        *inner = StatsAccumulatorInner::default();
        Ok(())
    }

    fn lock(&self) -> MutexGuard<'_, StatsAccumulatorInner> {
        self.inner.lock()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TierCallAttribution {
    LegacyPendingUntiered,
    ExplicitUntiered,
    AlreadyRecorded,
}

#[derive(Clone, Debug, Default)]
struct StatsAccumulatorInner {
    by_model: BTreeMap<String, ModelStats>,
    model_order: Vec<String>,
    untiered_successes_by_model: BTreeMap<String, u64>,
    by_tier: BTreeMap<String, TierStats>,
    total_requests: u64,
    total_errors: u64,
    routing_overhead: LatencyHistogram,
    by_classifier: BTreeMap<String, ModelStats>,
    classifier_model_order: Vec<String>,
    classifier_requests: u64,
    classifier_errors: u64,
    by_planner: BTreeMap<String, ModelStats>,
    planner_model_order: Vec<String>,
    planner_requests: u64,
    planner_errors: u64,
    routing_decisions: BTreeMap<String, BTreeMap<String, u64>>,
}

impl StatsAccumulatorInner {
    fn model_stats_mut(&mut self, model: String) -> &mut ModelStats {
        match self.by_model.entry(model) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                self.model_order.push(entry.key().clone());
                entry.insert(ModelStats::default())
            }
        }
    }

    fn classifier_stats_mut(&mut self, model: String) -> &mut ModelStats {
        match self.by_classifier.entry(model) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                self.classifier_model_order.push(entry.key().clone());
                entry.insert(ModelStats::default())
            }
        }
    }

    fn planner_stats_mut(&mut self, model: String) -> &mut ModelStats {
        match self.by_planner.entry(model) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                self.planner_model_order.push(entry.key().clone());
                entry.insert(ModelStats::default())
            }
        }
    }

    fn tier_stats_mut(&mut self, tier: &str, model: &str) -> Option<&mut TierStats> {
        let tier = tier.trim();
        if tier.is_empty() {
            return None;
        }
        Some(match self.by_tier.entry(tier.to_string()) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(TierStats {
                model: model.to_string(),
                ..TierStats::default()
            }),
        })
    }

    fn record_untiered_success(&mut self, model: &str) {
        let count = self
            .untiered_successes_by_model
            .entry(model.to_string())
            .or_insert(0);
        *count = count.saturating_add(1);
    }

    fn consume_untiered_success(&mut self, model: &str) -> bool {
        match self.untiered_successes_by_model.entry(model.to_string()) {
            Entry::Occupied(mut entry) => {
                let count = entry.get_mut();
                if *count > 1 {
                    *count -= 1;
                } else {
                    entry.remove();
                }
                true
            }
            Entry::Vacant(_) => false,
        }
    }

    fn snapshot(&self) -> StatsSnapshot {
        let (models, totals) = build_model_snapshots(&self.by_model, self.total_requests);
        let total_tokens = totals.total;
        let mut cost_estimate = estimate_cost(&models);

        let classifier = build_classifier_snapshot(
            &self.by_classifier,
            self.classifier_requests,
            self.classifier_errors,
        );
        cost_estimate.classifier_cost = classifier.cost_estimate.total_cost;

        let planner =
            build_planner_snapshot(&self.by_planner, self.planner_requests, self.planner_errors);
        cost_estimate.planner_cost = planner.cost_estimate.total_cost;

        cost_estimate.total_cost = round6(
            cost_estimate.backend_cost + cost_estimate.classifier_cost + cost_estimate.planner_cost,
        );

        StatsSnapshot {
            total_requests: self.total_requests,
            total_errors: self.total_errors,
            total_tokens: totals,
            tiers: tier_snapshots(&self.by_tier, total_tokens, self.total_requests),
            cost_estimate,
            models,
            routing_overhead: self.routing_overhead.snapshot(),
            classifier,
            planner,
            routing_decisions: self.routing_decisions.clone(),
        }
    }
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

    let mut models = BTreeMap::new();
    for (model, stats) in by_model {
        let token_total = stats.prompt_tokens.saturating_add(stats.completion_tokens);
        let request_pct = if total_requests == 0 {
            0.0
        } else {
            round2(stats.calls as f64 / total_requests as f64 * 100.0)
        };
        let token_pct = if totals.total == 0 {
            0.0
        } else {
            round2(token_total as f64 / totals.total as f64 * 100.0)
        };
        let avg_prompt_tokens = if stats.calls == 0 {
            0.0
        } else {
            round2(stats.prompt_tokens as f64 / stats.calls as f64)
        };
        let avg_completion_tokens = if stats.calls == 0 {
            0.0
        } else {
            round2(stats.completion_tokens as f64 / stats.calls as f64)
        };
        let cache_hit_rate = if stats.prompt_tokens == 0 {
            0.0
        } else {
            round4(stats.cached_tokens as f64 / stats.prompt_tokens as f64)
        };
        // Switch-aware ceiling: prefix this model had already seen; gap to actual is the backend.
        let theoretical_cache_hit_rate = if stats.prompt_tokens == 0 {
            0.0
        } else {
            round4(stats.cacheable_prompt_tokens as f64 / stats.prompt_tokens as f64)
        };

        models.insert(
            model.clone(),
            ModelStatsSnapshot {
                calls: stats.calls,
                errors: stats.errors,
                request_pct,
                prompt_tokens: stats.prompt_tokens,
                completion_tokens: stats.completion_tokens,
                total_tokens: token_total,
                token_pct,
                cached_tokens: stats.cached_tokens,
                cache_creation_tokens: stats.cache_creation_tokens,
                reasoning_tokens: stats.reasoning_tokens,
                avg_prompt_tokens,
                avg_completion_tokens,
                cache_hit_rate,
                theoretical_cache_hit_rate,
                model_call_latency: stats.model_call_latency.snapshot(),
                total_latency: stats.total_latency.snapshot(),
                tier: stats.tier.clone(),
            },
        );
    }
    (models, totals)
}

fn build_classifier_snapshot(
    by_classifier: &BTreeMap<String, ModelStats>,
    total_requests: u64,
    total_errors: u64,
) -> ClassifierStatsSnapshot {
    let (models, totals) = build_model_snapshots(by_classifier, total_requests);
    let cost_estimate = estimate_cost(&models);
    ClassifierStatsSnapshot {
        total_requests,
        total_errors,
        total_tokens: totals,
        models,
        cost_estimate,
    }
}

fn build_planner_snapshot(
    by_planner: &BTreeMap<String, ModelStats>,
    total_requests: u64,
    total_errors: u64,
) -> PlannerStatsSnapshot {
    let (models, totals) = build_model_snapshots(by_planner, total_requests);
    let cost_estimate = estimate_cost(&models);
    PlannerStatsSnapshot {
        total_requests,
        total_errors,
        total_tokens: totals,
        models,
        cost_estimate,
    }
}

#[derive(Clone, Debug, Default)]
struct ModelStats {
    calls: u64,
    errors: u64,
    prompt_tokens: u64,
    completion_tokens: u64,
    cached_tokens: u64,
    cache_creation_tokens: u64,
    cacheable_prompt_tokens: u64,
    reasoning_tokens: u64,
    /// Prefix fingerprints this model has been sent; basis for switch-aware theoretical.
    seen_prefixes: HashSet<u64>,
    model_call_latency: LatencyHistogram,
    total_latency: LatencyHistogram,
    tier: Option<String>,
}

#[derive(Clone, Debug, Default)]
struct TierStats {
    model: String,
    calls: u64,
    prompt_tokens: u64,
    completion_tokens: u64,
}

#[derive(Clone, Debug)]
struct LatencyHistogram {
    count: u64,
    total_ms: f64,
    min_ms: f64,
    max_ms: f64,
    samples: BinaryHeap<Reverse<LatencySample>>,
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self {
            count: 0,
            total_ms: 0.0,
            min_ms: f64::INFINITY,
            max_ms: 0.0,
            samples: BinaryHeap::new(),
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
        let sample = Reverse(LatencySample(latency_ms));
        if self.samples.len() < MAX_LATENCY_SAMPLES {
            self.samples.push(sample);
        } else if let Some(smallest_sample) = self.samples.peek() {
            if latency_ms > smallest_sample.0.value() {
                if let Some(mut smallest_sample) = self.samples.peek_mut() {
                    *smallest_sample = sample;
                }
            }
        }
    }

    fn snapshot(&self) -> LatencyHistogramSnapshot {
        if self.count == 0 {
            return LatencyHistogramSnapshot::default();
        }
        let mut samples = self
            .samples
            .iter()
            .map(|sample| sample.0.value())
            .collect::<Vec<_>>();
        samples.sort_by(f64::total_cmp);
        let sample_count = samples.len();
        let p50_ms = samples
            .get(sample_count / 2)
            .copied()
            .map(round2)
            .unwrap_or(0.0);
        let p99_index = sample_count
            .saturating_sub(1)
            .min((sample_count as f64 * 0.99) as usize);
        let p99_ms = samples.get(p99_index).copied().map(round2).unwrap_or(0.0);

        LatencyHistogramSnapshot {
            count: self.count,
            total_ms: round2(self.total_ms),
            min_ms: round2(self.min_ms),
            max_ms: round2(self.max_ms),
            avg_ms: round2(self.total_ms / self.count as f64),
            p50_ms,
            p99_ms,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct LatencySample(f64);

impl LatencySample {
    fn value(self) -> f64 {
        self.0
    }
}

impl PartialEq for LatencySample {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0) == Ordering::Equal
    }
}

impl Eq for LatencySample {}

impl PartialOrd for LatencySample {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LatencySample {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Full stats snapshot.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct StatsSnapshot {
    pub total_requests: u64,
    pub total_errors: u64,
    pub total_tokens: TokenTotals,
    pub models: BTreeMap<String, ModelStatsSnapshot>,
    pub tiers: BTreeMap<String, TierStatsSnapshot>,
    pub routing_overhead: LatencyHistogramSnapshot,
    pub cost_estimate: CostEstimate,
    pub classifier: ClassifierStatsSnapshot,
    pub planner: PlannerStatsSnapshot,
    pub routing_decisions: BTreeMap<String, BTreeMap<String, u64>>,
}

/// LLM-classifier overhead stats, recorded out-of-band from routed-backend
/// traffic so the two never alias on a shared model id.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ClassifierStatsSnapshot {
    pub total_requests: u64,
    pub total_errors: u64,
    pub total_tokens: TokenTotals,
    pub models: BTreeMap<String, ModelStatsSnapshot>,
    pub cost_estimate: CostEstimate,
}

/// Planner-overhead stats, recorded out-of-band from routed-backend
/// traffic.  Parallel to :class:`ClassifierStatsSnapshot`; the
/// :class:`switchyard.lib.processors.plan_execute.PlanningRequestProcessor`
/// emits one record per planner call (usage on success,
/// error-only on fail-open) so the planner's spend and failure rate
/// are first-class on the stats endpoint instead of being inferable
/// only from the per-request audit log.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PlannerStatsSnapshot {
    pub total_requests: u64,
    pub total_errors: u64,
    pub total_tokens: TokenTotals,
    pub models: BTreeMap<String, ModelStatsSnapshot>,
    pub cost_estimate: CostEstimate,
}

/// Aggregate token totals.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct TokenTotals {
    pub prompt: u64,
    pub completion: u64,
    pub cached: u64,
    pub cache_creation: u64,
    pub reasoning: u64,
    pub total: u64,
}

/// Per-model stats in a snapshot.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelStatsSnapshot {
    pub calls: u64,
    pub errors: u64,
    pub request_pct: f64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub token_pct: f64,
    pub cached_tokens: u64,
    pub cache_creation_tokens: u64,
    pub reasoning_tokens: u64,
    pub avg_prompt_tokens: f64,
    pub avg_completion_tokens: f64,
    pub cache_hit_rate: f64,
    /// Switch-aware ceiling: fraction of the prompt this model had already been sent.
    pub theoretical_cache_hit_rate: f64,
    pub model_call_latency: LatencyHistogramSnapshot,
    pub total_latency: LatencyHistogramSnapshot,
    pub tier: Option<String>,
}

/// Per-tier stats in a snapshot.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TierStatsSnapshot {
    pub model: String,
    pub calls: u64,
    pub request_pct: f64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub token_pct: f64,
}

/// Latency histogram summary in milliseconds.
///
/// Non-finite latency samples are ignored and logged at debug level.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LatencyHistogramSnapshot {
    pub count: u64,
    pub total_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub avg_ms: f64,
    pub p50_ms: f64,
    pub p99_ms: f64,
}

fn tier_snapshots(
    by_tier: &BTreeMap<String, TierStats>,
    total_tokens: u64,
    total_requests: u64,
) -> BTreeMap<String, TierStatsSnapshot> {
    let mut tiers = BTreeMap::new();
    for (tier, stats) in by_tier {
        tiers.insert(
            tier.clone(),
            TierStatsSnapshot {
                model: stats.model.clone(),
                calls: stats.calls,
                request_pct: 0.0,
                prompt_tokens: stats.prompt_tokens,
                completion_tokens: stats.completion_tokens,
                total_tokens: 0,
                token_pct: 0.0,
            },
        );
    }

    for tier in tiers.values_mut() {
        tier.total_tokens = tier.prompt_tokens.saturating_add(tier.completion_tokens);
        tier.request_pct = if total_requests == 0 {
            0.0
        } else {
            round2(tier.calls as f64 / total_requests as f64 * 100.0)
        };
        tier.token_pct = if total_tokens == 0 {
            0.0
        } else {
            round2(tier.total_tokens as f64 / total_tokens as f64 * 100.0)
        };
    }
    tiers
}

fn round2(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

fn round4(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

fn round6(value: f64) -> f64 {
    (value * 1_000_000.0).round() / 1_000_000.0
}
