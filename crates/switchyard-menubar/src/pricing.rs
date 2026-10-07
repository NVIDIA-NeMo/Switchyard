// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Counterfactual cost: what routed traffic cost against what it would have
//! cost had every call gone to one capable model.

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::rollup::{ModelTokens, Totals};

const PER_MILLION: f64 = 1_000_000.0;

/// Per-million-token rates for one model.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
pub struct ModelPrice {
    pub input_per_mtok: f64,
    /// Rate for cache reads. Falls back to the full input rate.
    #[serde(default)]
    pub cached_input_per_mtok: Option<f64>,
    /// Rate for generated tokens, reasoning included.
    pub output_per_mtok: f64,
}

impl ModelPrice {
    fn cost(&self, tokens: &ModelTokens) -> f64 {
        let cached_rate = self.cached_input_per_mtok.unwrap_or(self.input_per_mtok);
        (tokens.input as f64 * self.input_per_mtok
            + tokens.cached_input as f64 * cached_rate
            + tokens.output as f64 * self.output_per_mtok)
            / PER_MILLION
    }
}

/// Rates keyed by the model id the server records in the routing log.
pub type PriceTable = BTreeMap<String, ModelPrice>;

/// What the traffic cost, and what it would have cost without routing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Savings {
    /// Cost of the calls that actually ran, classifier overhead included.
    pub actual: f64,
    /// Cost of the same caller-facing calls had they all used the baseline model.
    pub baseline: f64,
}

impl Savings {
    /// Dollars kept. Negative when routing overhead outweighed the cheaper tier.
    pub fn saved(&self) -> f64 {
        self.baseline - self.actual
    }

    /// Fraction of the baseline bill avoided, or `None` with nothing to compare.
    pub fn percent(&self) -> Option<f64> {
        (self.baseline > 0.0).then(|| self.saved() / self.baseline * 100.0)
    }
}

/// Prices a period against the baseline model.
///
/// Returns `None` when any model seen has no entry in the table, since a
/// partial bill would understate cost and overstate savings.
pub fn estimate(totals: &Totals, prices: &PriceTable, baseline_model: &str) -> Option<Savings> {
    let baseline_price = prices.get(baseline_model)?;

    let mut actual = 0.0;
    let mut baseline = 0.0;
    for (model, tokens) in &totals.routed {
        actual += prices.get(model)?.cost(tokens);
        baseline += baseline_price.cost(tokens);
    }
    // Classifier calls are Switchyard's own cost. They have no baseline
    // counterpart, so they only ever reduce the savings figure.
    for (model, tokens) in &totals.classifier {
        actual += prices.get(model)?.cost(tokens);
    }

    Some(Savings { actual, baseline })
}

/// Returns the models in `models` that have no entry in `prices`, sorted and
/// each once. Such a model makes [`estimate`] return `None`.
pub fn unpriced<'a>(
    models: impl IntoIterator<Item = &'a str>,
    prices: &PriceTable,
) -> Vec<&'a str> {
    let mut missing: Vec<&str> = models
        .into_iter()
        .filter(|model| !prices.contains_key(*model))
        .collect();
    missing.sort_unstable();
    missing.dedup();
    missing
}

#[cfg(test)]
mod tests {
    use super::*;

    fn price(input: f64, output: f64) -> ModelPrice {
        ModelPrice {
            input_per_mtok: input,
            cached_input_per_mtok: None,
            output_per_mtok: output,
        }
    }

    fn table() -> PriceTable {
        PriceTable::from([
            ("sol".to_string(), price(1.25, 10.0)),
            ("luna".to_string(), price(0.25, 2.0)),
            ("terra".to_string(), price(0.05, 0.4)),
        ])
    }

    fn tokens(input: u64, output: u64) -> ModelTokens {
        ModelTokens {
            requests: 1,
            input,
            cached_input: 0,
            output,
        }
    }

    #[test]
    fn prices_the_cheaper_tier_against_the_capable_baseline() {
        let mut totals = Totals::default();
        totals
            .routed
            .insert("luna".to_string(), tokens(1_000_000, 100_000));

        let savings = estimate(&totals, &table(), "sol").expect("priced");

        assert!((savings.actual - 0.45).abs() < 1e-9);
        assert!((savings.baseline - 2.25).abs() < 1e-9);
        assert!((savings.percent().expect("percent") - 80.0).abs() < 1e-9);
    }

    #[test]
    fn classifier_overhead_counts_against_savings() {
        let mut totals = Totals::default();
        totals
            .routed
            .insert("luna".to_string(), tokens(1_000_000, 100_000));
        let before = estimate(&totals, &table(), "sol").expect("priced");

        totals
            .classifier
            .insert("terra".to_string(), tokens(1_000_000, 10_000));
        let after = estimate(&totals, &table(), "sol").expect("priced");

        assert_eq!(after.baseline, before.baseline);
        assert!(after.saved() < before.saved());
    }

    #[test]
    fn cache_reads_use_the_cached_rate() {
        let prices = PriceTable::from([(
            "sol".to_string(),
            ModelPrice {
                input_per_mtok: 1.0,
                cached_input_per_mtok: Some(0.1),
                output_per_mtok: 0.0,
            },
        )]);
        let mut totals = Totals::default();
        totals.routed.insert(
            "sol".to_string(),
            ModelTokens {
                requests: 1,
                input: 1_000_000,
                cached_input: 1_000_000,
                output: 0,
            },
        );

        let savings = estimate(&totals, &prices, "sol").expect("priced");

        assert!((savings.actual - 1.1).abs() < 1e-9);
    }

    #[test]
    fn an_unpriced_model_suppresses_the_estimate() {
        let mut totals = Totals::default();
        totals
            .routed
            .insert("unknown".to_string(), tokens(1_000, 100));

        assert!(estimate(&totals, &table(), "sol").is_none());
        assert!(estimate(&Totals::default(), &table(), "absent").is_none());
    }

    #[test]
    fn an_empty_period_has_no_percentage() {
        let savings = estimate(&Totals::default(), &table(), "sol").expect("priced");

        assert_eq!(savings.saved(), 0.0);
        assert!(savings.percent().is_none());
    }

    #[test]
    fn names_each_model_that_has_no_price_once() {
        let models = ["sol", "mystery", "luna", "other", "mystery"];

        assert_eq!(unpriced(models, &table()), ["mystery", "other"]);
    }
}
