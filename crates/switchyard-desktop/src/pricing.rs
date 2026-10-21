// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module compares the cost that the server recorded with the cost of the same answers on
//! the baseline model. The server prices each call when it finishes. The app uses bundled
//! rates or a configured override only to price the baseline.

use std::path::Path;

use serde::Serialize;
use switchyard_costs::Pricing;

use crate::rollup::Totals;

/// This function loads bundled rates when no override is selected.
/// An explicit file replaces the entire table; this function returns an error if it cannot load it.
pub fn load(path: Option<&Path>) -> Result<Pricing, String> {
    match path {
        Some(path) => Pricing::load(path),
        None => Pricing::bundled(),
    }
}

/// Estimate stores one period's recorded cost and the baseline model's cost for its answers.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Estimate {
    /// This field stores the recorded cost of every call, including routing calls. It is
    /// `None` when any call has no recorded cost.
    pub actual: Option<f64>,
    /// This field adds up the calls that have a recorded cost.
    pub known: f64,
    /// This field counts calls that the server recorded without a cost.
    pub unknown_calls: u64,
    /// This field prices the answer calls at the baseline model's rates. It is `None` when
    /// the selected price table has no rates for the baseline model.
    pub baseline: Option<f64>,
    /// This field counts answer calls recorded without token counts. The baseline cannot
    /// price them, so they hide the savings.
    pub uncounted_answers: u64,
}

impl Estimate {
    pub fn new(totals: &Totals, rates: &Pricing, baseline_model: &str) -> Self {
        let cost = totals.cost();
        // Routing calls would not exist without Switchyard, so the baseline prices only answers.
        let baseline = rates
            .contains(baseline_model)
            .then(|| {
                totals.routed.values().try_fold(0.0, |sum, tokens| {
                    Some(sum + rates.estimate(baseline_model, &tokens.usage())?.usd)
                })
            })
            .flatten();
        Self {
            actual: cost.total_usd,
            known: cost.known_usd,
            unknown_calls: cost.unknown_calls,
            baseline,
            uncounted_answers: totals.routed.values().map(|tokens| tokens.uncounted).sum(),
        }
    }

    /// This function returns the savings when every call has a cost, every answer has token
    /// counts, and the baseline has rates.
    pub fn savings(&self) -> Option<Savings> {
        if self.uncounted_answers > 0 {
            return None;
        }
        Some(Savings {
            actual: self.actual?,
            baseline: self.baseline?,
        })
    }
}

/// Savings stores a period's recorded cost and the baseline cost of the same answers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Savings {
    pub actual: f64,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rollup::ModelTokens;
    use switchyard_costs::{CostSource, RecordedCost};

    fn tokens(input: u64, cached: u64, write: u64, output: u64, usd: Option<f64>) -> ModelTokens {
        let mut tokens = ModelTokens {
            requests: 1,
            input,
            cached_input: cached,
            cache_write: write,
            output,
            ..ModelTokens::default()
        };
        tokens.cost.add(usd.map(|usd| RecordedCost {
            usd,
            source: CostSource::ProviderReported,
        }));
        tokens
    }

    #[test]
    fn savings_use_recorded_costs_and_the_baseline_rates() {
        let rates = Pricing::parse(
            br#"{"baseline":{"input":1,"cache_read":0.1,"cache_write":2,"output":10}}"#,
        )
        .expect("rates");
        let mut totals = Totals::default();
        let empty = Estimate::new(&totals, &rates, "baseline");
        assert_eq!(empty.savings().map(|s| s.saved()), Some(0.0));
        assert!(empty.savings().and_then(|s| s.percent()).is_none());

        // The answer's recorded cost is used as is; the baseline prices its tokens:
        // 1M input at $1, 1M cache reads at $0.10, 1M cache writes at $2, 0.1M output at $10.
        totals.routed.insert(
            "cheap".into(),
            tokens(1_000_000, 1_000_000, 1_000_000, 100_000, Some(0.5)),
        );
        let routed = Estimate::new(&totals, &rates, "baseline")
            .savings()
            .expect("savings");
        assert_eq!(routed.actual, 0.5);
        assert!((routed.baseline - 4.1).abs() < 1e-9);

        // Routing calls add to the recorded cost but not to the baseline.
        totals
            .classifier
            .insert("judge".into(), tokens(10, 0, 0, 1, Some(5.0)));
        let overhead = Estimate::new(&totals, &rates, "baseline")
            .savings()
            .expect("savings");
        assert_eq!(overhead.baseline, routed.baseline);
        assert!(overhead.saved() < 0.0);

        // An answer with a charge but no token counts cannot be priced on the baseline.
        let mut uncounted = tokens(0, 0, 0, 0, Some(1.0));
        uncounted.uncounted = 1;
        totals.routed.insert("uncounted".into(), uncounted);
        let hidden = Estimate::new(&totals, &rates, "baseline");
        assert_eq!((hidden.actual, hidden.uncounted_answers), (Some(6.5), 1));
        assert!(hidden.savings().is_none());
        totals.routed.remove("uncounted");

        // A call without a recorded cost hides the savings but keeps the known subtotal.
        totals.routed.insert("old".into(), tokens(1, 0, 0, 1, None));
        let partial = Estimate::new(&totals, &rates, "baseline");
        assert_eq!(
            (partial.actual, partial.known, partial.unknown_calls),
            (None, 5.5, 1)
        );
        assert!(partial.savings().is_none());
        assert!(
            Estimate::new(&Totals::default(), &rates, "missing")
                .baseline
                .is_none()
        );

        // One-hour cache writes need the baseline's one-hour rate.
        let mut long = tokens(0, 0, 1_000_000, 0, Some(1.0));
        long.cache_write_1h = 1_000_000;
        let long = Totals {
            routed: [("long".into(), long)].into(),
            ..Totals::default()
        };
        assert!(Estimate::new(&long, &rates, "baseline").baseline.is_none());
        let with_1h = Pricing::parse(
            br#"{"baseline":{"input":1,"cache_read":0.1,"cache_write":2,"cache_write_1h":4,"output":10}}"#,
        )
        .expect("rates");
        assert_eq!(
            Estimate::new(&long, &with_1h, "baseline").baseline,
            Some(4.0)
        );
    }
}
