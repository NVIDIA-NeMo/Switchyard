// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Counterfactual cost: what routed traffic cost against what it would have
//! cost had every call gone to one capable model.

use switchyard_costs::{CostTotals, Pricing};
use switchyard_protocol::{InputCacheUsage, Usage};

use crate::rollup::Totals;

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
/// Returns `None` when recorded spending or the baseline estimate is incomplete.
pub fn estimate(totals: &Totals, prices: &Pricing, baseline_model: &str) -> Option<Savings> {
    if totals.baseline_unknown {
        return None;
    }
    let mut baseline = CostTotals::default();
    for tokens in totals.routed.values() {
        let usage = Usage {
            input_tokens: Some(tokens.input),
            output_tokens: Some(tokens.output),
            cache: Some(Box::new(InputCacheUsage {
                cached_input_tokens: Some(tokens.cached_input),
                cache_creation_input_tokens: Some(tokens.cache_write),
                cache_creation_1h_input_tokens: Some(tokens.cache_write_1h),
            })),
            ..Usage::default()
        };
        baseline.add(prices.estimate(baseline_model, &usage));
    }
    Some(Savings {
        actual: totals.cost.total_usd?,
        baseline: baseline.total_usd?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rollup::ModelTokens;

    fn table() -> Pricing {
        Pricing::parse(
            br#"{
          "sol":{"input":1.25,"cache_read":1.25,"cache_write":1.25,"output":10},
          "luna":{"input":0.25,"cache_read":0.25,"cache_write":0.25,"output":2},
          "terra":{"input":0.05,"cache_read":0.05,"cache_write":0.05,"output":0.4}
        }"#,
        )
        .expect("rates")
    }

    fn record_cost(totals: &mut Totals, usd: f64) {
        totals.cost.add(Some(switchyard_costs::RecordedCost {
            usd,
            source: switchyard_costs::CostSource::Estimated,
        }));
    }

    fn tokens(input: u64, output: u64) -> ModelTokens {
        ModelTokens {
            requests: 1,
            input,
            cached_input: 0,
            output,
            ..ModelTokens::default()
        }
    }

    #[test]
    fn prices_the_cheaper_tier_against_the_capable_baseline() {
        let mut totals = Totals::default();
        totals
            .routed
            .insert("luna".to_string(), tokens(1_000_000, 100_000));

        record_cost(&mut totals, 0.45);
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
        record_cost(&mut totals, 0.45);
        let before = estimate(&totals, &table(), "sol").expect("priced");

        totals
            .classifier
            .insert("terra".to_string(), tokens(1_000_000, 10_000));
        record_cost(&mut totals, 0.054);
        let after = estimate(&totals, &table(), "sol").expect("priced");

        assert_eq!(after.baseline, before.baseline);
        assert!(after.saved() < before.saved());
    }

    #[test]
    fn cache_reads_use_the_cached_rate() {
        let prices =
            Pricing::parse(br#"{"sol":{"input":1,"cache_read":0.1,"cache_write":1,"output":0}}"#)
                .expect("rates");
        let mut totals = Totals::default();
        totals.routed.insert(
            "sol".to_string(),
            ModelTokens {
                requests: 1,
                input: 1_000_000,
                cached_input: 1_000_000,
                output: 0,
                ..ModelTokens::default()
            },
        );

        record_cost(&mut totals, 1.1);
        let savings = estimate(&totals, &prices, "sol").expect("priced");

        assert!((savings.actual - 1.1).abs() < 1e-9);
    }

    #[test]
    fn an_unknown_cost_suppresses_savings() {
        let mut totals = Totals::default();
        totals
            .routed
            .insert("unknown".to_string(), tokens(1_000, 100));

        totals.cost.add(None);
        assert!(estimate(&totals, &table(), "sol").is_none());
        assert!(estimate(&totals, &table(), "absent").is_none());
    }

    #[test]
    fn an_empty_period_has_no_percentage() {
        let savings = estimate(&Totals::default(), &table(), "sol").expect("priced");

        assert_eq!(savings.saved(), 0.0);
        assert!(savings.percent().is_none());
    }
}
