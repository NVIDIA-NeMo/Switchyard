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
    /// This flag distinguishes a recorded zero cost from missing costs.
    pub has_known_cost: bool,
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
    /// This function totals recorded costs and prices only answer tokens for the baseline.
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
            has_known_cost: cost.estimated_calls > 0 || cost.provider_reported_calls > 0,
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
