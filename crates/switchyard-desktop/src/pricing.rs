// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module compares recorded calls at configured rates with the baseline model estimate.

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::rollup::{ModelTokens, Totals};

const PER_MILLION: f64 = 1_000_000.0;

/// ModelPrice stores rates per million tokens for one model.
#[derive(Clone, Copy, Debug, Deserialize, serde::Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelPrice {
    pub input_per_mtok: f64,
    /// This field sets the cache-read rate and defaults to the full input rate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_input_per_mtok: Option<f64>,
    /// This field sets the rate for generated tokens, including reasoning.
    pub output_per_mtok: f64,
}

impl ModelPrice {
    pub fn validate(&self) -> Result<(), String> {
        for rate in [
            Some(self.input_per_mtok),
            self.cached_input_per_mtok,
            Some(self.output_per_mtok),
        ]
        .into_iter()
        .flatten()
        {
            if !rate.is_finite() || rate < 0.0 {
                return Err(
                    "Model prices must be finite, nonnegative dollars per million tokens.".into(),
                );
            }
        }
        Ok(())
    }

    fn cost(&self, tokens: &ModelTokens) -> f64 {
        let cached_rate = self.cached_input_per_mtok.unwrap_or(self.input_per_mtok);
        (tokens.input as f64 * self.input_per_mtok
            + tokens.cached_input as f64 * cached_rate
            + tokens.output as f64 * self.output_per_mtok)
            / PER_MILLION
    }
}

/// PriceTable stores rates by the model ID recorded in the routing log.
pub type PriceTable = BTreeMap<String, ModelPrice>;

/// Savings stores estimated costs with and without routing.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct Savings {
    /// This field stores the estimated cost of recorded calls, including classifier calls.
    pub actual: f64,
    /// This field estimates caller-facing calls at the baseline model rates.
    pub baseline: f64,
}

impl Savings {
    /// This function returns estimated dollars saved; routing overhead can make the value negative.
    pub fn saved(&self) -> f64 {
        self.baseline - self.actual
    }

    /// This function returns the percentage saved, or `None` when the baseline cost is zero.
    pub fn percent(&self) -> Option<f64> {
        (self.baseline > 0.0).then(|| self.saved() / self.baseline * 100.0)
    }
}

/// This function prices a period against the baseline model.
///
/// This function returns `None` when any model seen has no entry in the table, since a
/// partial bill would understate cost and overstate savings.
pub fn estimate(totals: &Totals, prices: &PriceTable, baseline_model: &str) -> Option<Savings> {
    if prices.values().any(|price| price.validate().is_err()) {
        return None;
    }
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

/// This function returns the models in `models` that have no entry in `prices`, sorted and
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

    #[test]
    fn rates_reject_negative_nonfinite_and_unknown_fields() {
        for rate in [-1.0, f64::INFINITY, f64::NAN] {
            for field in 0..3 {
                let mut price = ModelPrice {
                    input_per_mtok: 0.0,
                    cached_input_per_mtok: Some(0.0),
                    output_per_mtok: 0.0,
                };
                match field {
                    0 => price.input_per_mtok = rate,
                    1 => price.cached_input_per_mtok = Some(rate),
                    _ => price.output_per_mtok = rate,
                }
                assert!(price.validate().is_err());
            }
        }
        assert!(
            toml::from_str::<ModelPrice>(
                "input_per_mtok=0.0\noutput_per_mtok=1.0\ncached_input_per_mtok_typo=0.0"
            )
            .is_err()
        );
    }

    #[test]
    fn estimates_cost_without_hiding_overhead_or_missing_prices() {
        let price = ModelPrice {
            input_per_mtok: 1.0,
            cached_input_per_mtok: Some(0.1),
            output_per_mtok: 2.0,
        };
        let prices = PriceTable::from([
            ("baseline".into(), price),
            (
                "cheap".into(),
                ModelPrice {
                    input_per_mtok: 0.2,
                    cached_input_per_mtok: None,
                    output_per_mtok: 0.4,
                },
            ),
        ]);
        let mut totals = Totals::default();
        let empty = estimate(&totals, &prices, "baseline").expect("empty");
        assert_eq!(empty.saved(), 0.0);
        assert!(empty.percent().is_none());
        totals.routed.insert(
            "cheap".into(),
            ModelTokens {
                requests: 1,
                input: 1_000_000,
                cached_input: 1_000_000,
                output: 100_000,
            },
        );
        let routed = estimate(&totals, &prices, "baseline").expect("priced");
        assert!((routed.actual - 0.44).abs() < 1e-9);
        assert!((routed.baseline - 1.3).abs() < 1e-9);
        assert!((routed.percent().expect("percent") - (1.3 - 0.44) / 1.3 * 100.0).abs() < 1e-9);
        totals.classifier.insert(
            "baseline".into(),
            ModelTokens {
                requests: 1,
                input: 2_000_000,
                ..ModelTokens::default()
            },
        );
        let overhead = estimate(&totals, &prices, "baseline").expect("overhead");
        assert_eq!(overhead.baseline, routed.baseline);
        assert!((overhead.actual - 2.44).abs() < 1e-9);
        assert!(overhead.saved() < 0.0);
        totals
            .routed
            .insert("missing".into(), ModelTokens::default());
        assert!(estimate(&totals, &prices, "baseline").is_none());
        assert!(estimate(&Totals::default(), &prices, "missing").is_none());
        assert_eq!(
            unpriced(["cheap", "missing", "other", "missing"], &prices),
            ["missing", "other"]
        );
    }
}
