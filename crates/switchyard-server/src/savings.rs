// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cost and savings accounting derived from the stats snapshot.
//!
//! Pricing is configured per model in the deployment TOML (USD per 1M
//! tokens). Savings compare the actual routed spend against a baseline:
//! what the same traffic would have cost if every request had been served
//! by the baseline (typically the most capable / expensive) model.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::stats::StatsSnapshot;

/// Per-model pricing in USD per 1 million tokens.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelPrice {
    /// Base input price (fresh prompt tokens).
    pub input: f64,
    /// Output / completion price.
    pub output: f64,
    /// Cache-read price (prompt cache hit).
    pub cached: f64,
    /// Cache-write price (prompt cache creation).
    pub cache_write: f64,
}

/// Savings configuration: a pricing table plus an optional explicit baseline.
#[derive(Clone, Debug, Default)]
pub struct SavingsConfig {
    pricing: BTreeMap<String, ModelPrice>,
    baseline: Option<String>,
}

impl SavingsConfig {
    /// Creates a savings config from per-model prices and an optional
    /// baseline model. When `baseline` is `None`, the most expensive
    /// priced model (by input + output rate) is used.
    pub fn new(pricing: BTreeMap<String, ModelPrice>, baseline: Option<String>) -> Self {
        Self { pricing, baseline }
    }

    fn price_for(&self, model: &str) -> Option<ModelPrice> {
        self.pricing.get(model).copied()
    }

    fn baseline_model(&self) -> Option<(&str, ModelPrice)> {
        if let Some(name) = &self.baseline {
            return self.price_for(name).map(|price| (name.as_str(), price));
        }
        self.pricing
            .iter()
            .max_by(|a, b| {
                let ka = a.1.input + a.1.output;
                let kb = b.1.input + b.1.output;
                ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(name, price)| (name.as_str(), *price))
    }

    /// Computes a savings snapshot from the current stats snapshot.
    pub(crate) fn compute(&self, stats: &StatsSnapshot) -> SavingsSnapshot {
        let baseline = self.baseline_model();
        let mut models = BTreeMap::new();
        let mut actual_cost = 0.0;
        let mut baseline_cost = 0.0;
        let mut unpriced_models = Vec::new();

        for (model, m) in &stats.models {
            let tokens = TokenBuckets {
                prompt: m.prompt_tokens,
                completion: m.completion_tokens,
                cached: m.cached_tokens,
                cache_creation: m.cache_creation_tokens,
            };
            let price = self.price_for(model);
            if price.is_none() {
                unpriced_models.push(model.to_string());
            }
            let cost = price.map(|p| tokens.cost(p)).unwrap_or(0.0);
            let would_be = baseline.map(|(_, p)| tokens.cost(p)).unwrap_or(0.0);
            actual_cost += cost;
            baseline_cost += would_be;
            models.insert(
                model.to_string(),
                ModelSavings {
                    calls: m.calls,
                    prompt_tokens: m.prompt_tokens,
                    completion_tokens: m.completion_tokens,
                    cached_tokens: m.cached_tokens,
                    cost: round6(cost),
                    baseline_cost: round6(would_be),
                    priced: price.is_some(),
                },
            );
        }

        // Classifier / judge calls are pure routing overhead: they add to the
        // actual spend but a baseline deployment would not make them at all.
        let mut classifier_cost = 0.0;
        for (model, m) in &stats.classifier.models {
            let tokens = TokenBuckets {
                prompt: m.prompt_tokens,
                completion: m.completion_tokens,
                cached: m.cached_tokens,
                cache_creation: m.cache_creation_tokens,
            };
            match self.price_for(model) {
                Some(price) => classifier_cost += tokens.cost(price),
                // An unpriced judge is under-counted the same way as an
                // unpriced serving model; surface it rather than hide it.
                None => {
                    let model = model.to_string();
                    if !unpriced_models.contains(&model) {
                        unpriced_models.push(model);
                    }
                }
            }
        }
        actual_cost += classifier_cost;

        let saved = baseline_cost - actual_cost;
        let saved_pct = if baseline_cost > 0.0 {
            (saved / baseline_cost) * 100.0
        } else {
            0.0
        };

        SavingsSnapshot {
            total_requests: stats.total_requests,
            actual_cost: round6(actual_cost),
            baseline_cost: round6(baseline_cost),
            classifier_cost: round6(classifier_cost),
            saved: round6(saved),
            saved_pct: round2(saved_pct),
            baseline_model: baseline.map(|(name, _)| name.to_string()),
            models,
            unpriced_models,
        }
    }
}

/// Token counters priced by [`TokenBuckets::cost`].
#[derive(Clone, Copy, Debug, Default)]
struct TokenBuckets {
    prompt: u64,
    completion: u64,
    cached: u64,
    cache_creation: u64,
}

impl TokenBuckets {
    /// Splits prompt tokens into base / cached / cache-write buckets and
    /// prices each, matching the Python `cost_estimator` semantics.
    fn cost(&self, price: ModelPrice) -> f64 {
        let base_input = self
            .prompt
            .saturating_sub(self.cached)
            .saturating_sub(self.cache_creation);
        (base_input as f64 / 1e6) * price.input
            + (self.cached as f64 / 1e6) * price.cached
            + (self.cache_creation as f64 / 1e6) * price.cache_write
            + (self.completion as f64 / 1e6) * price.output
    }
}

/// Serialized savings response returned by `GET /v1/savings`.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct SavingsSnapshot {
    pub total_requests: u64,
    /// USD actually spent (routed calls + classifier overhead).
    pub actual_cost: f64,
    /// USD the same routed traffic would have cost on the baseline model.
    pub baseline_cost: f64,
    /// USD spent on classifier / judge routing calls.
    pub classifier_cost: f64,
    /// `baseline_cost - actual_cost`.
    pub saved: f64,
    /// Percentage of baseline cost saved.
    pub saved_pct: f64,
    /// Model the baseline comparison is priced against.
    pub baseline_model: Option<String>,
    pub models: BTreeMap<String, ModelSavings>,
    /// Models that served traffic but have no configured price (costed at 0).
    pub unpriced_models: Vec<String>,
}

/// Per-model cost breakdown inside [`SavingsSnapshot`].
#[derive(Clone, Debug, Serialize)]
pub(crate) struct ModelSavings {
    pub calls: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cached_tokens: u64,
    pub cost: f64,
    pub baseline_cost: f64,
    pub priced: bool,
}

fn round6(v: f64) -> f64 {
    (v * 1e6).round() / 1e6
}

fn round2(v: f64) -> f64 {
    (v * 1e2).round() / 1e2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::StatsAccumulator;
    use crate::stats::TokenUsage;

    fn price(input: f64, output: f64) -> ModelPrice {
        ModelPrice {
            input,
            output,
            cached: input * 0.1,
            cache_write: input,
        }
    }

    fn usage(prompt: u64, completion: u64) -> TokenUsage {
        TokenUsage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            ..TokenUsage::default()
        }
    }

    #[test]
    fn savings_compare_cheap_traffic_against_expensive_baseline() {
        let stats = StatsAccumulator::default();
        stats.record_success("cheap", 10.0);
        stats.record_usage("cheap", usage(1_000_000, 1_000_000), 10.0);

        let mut pricing = BTreeMap::new();
        pricing.insert("cheap".to_string(), price(0.1, 0.4));
        pricing.insert("expensive".to_string(), price(3.0, 15.0));
        let config = SavingsConfig::new(pricing, None);

        let snapshot = config.compute(&stats.snapshot());
        assert_eq!(snapshot.baseline_model.as_deref(), Some("expensive"));
        assert!((snapshot.actual_cost - 0.5).abs() < 1e-9);
        assert!((snapshot.baseline_cost - 18.0).abs() < 1e-9);
        assert!((snapshot.saved - 17.5).abs() < 1e-9);
        assert!((snapshot.saved_pct - 97.22).abs() < 0.01);
    }

    #[test]
    fn explicit_baseline_wins_over_most_expensive() {
        let stats = StatsAccumulator::default();
        stats.record_success("cheap", 1.0);
        stats.record_usage("cheap", usage(1_000_000, 0), 1.0);

        let mut pricing = BTreeMap::new();
        pricing.insert("cheap".to_string(), price(0.1, 0.4));
        pricing.insert("mid".to_string(), price(1.0, 2.0));
        pricing.insert("expensive".to_string(), price(3.0, 15.0));
        let config = SavingsConfig::new(pricing, Some("mid".to_string()));

        let snapshot = config.compute(&stats.snapshot());
        assert_eq!(snapshot.baseline_model.as_deref(), Some("mid"));
        assert!((snapshot.baseline_cost - 1.0).abs() < 1e-9);
    }

    #[test]
    fn unpriced_models_are_reported_not_priced() {
        let stats = StatsAccumulator::default();
        stats.record_success("mystery", 1.0);
        stats.record_usage("mystery", usage(500_000, 0), 1.0);

        let mut pricing = BTreeMap::new();
        pricing.insert("expensive".to_string(), price(3.0, 15.0));
        let config = SavingsConfig::new(pricing, None);

        let snapshot = config.compute(&stats.snapshot());
        assert_eq!(snapshot.unpriced_models, vec!["mystery".to_string()]);
        assert!((snapshot.actual_cost - 0.0).abs() < 1e-9);
        // Baseline still counts what that traffic would have cost.
        assert!((snapshot.baseline_cost - 1.5).abs() < 1e-9);
    }

    #[test]
    fn classifier_calls_count_as_overhead_cost_only() {
        let stats = StatsAccumulator::default();
        stats.record_success("cheap", 1.0);
        stats.record_usage("cheap", usage(1_000_000, 0), 1.0);
        stats.record_classifier_success("cheap", Some(usage(1_000_000, 0)), 1.0);

        let mut pricing = BTreeMap::new();
        pricing.insert("cheap".to_string(), price(1.0, 1.0));
        pricing.insert("expensive".to_string(), price(2.0, 2.0));
        let config = SavingsConfig::new(pricing, None);

        let snapshot = config.compute(&stats.snapshot());
        // 1.0 routed + 1.0 classifier overhead.
        assert!((snapshot.actual_cost - 2.0).abs() < 1e-9);
        assert!((snapshot.classifier_cost - 1.0).abs() < 1e-9);
        // Baseline only reprices the routed traffic.
        assert!((snapshot.baseline_cost - 2.0).abs() < 1e-9);
    }
}
