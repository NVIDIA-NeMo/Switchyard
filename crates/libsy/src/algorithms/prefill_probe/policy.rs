// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cost-aware selection between the learned router's weak and strong heads.

use crate::{LibsyError, Result};

/// Completion tier selected by the learned utility policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PrefillTier {
    Weak,
    Strong,
}

/// Validated utility policy for two completion targets.
#[derive(Clone, Copy, Debug)]
pub(super) struct CostAwareRoutingPolicy {
    lambda: f64,
    normalized_weak_cost: f64,
    normalized_strong_cost: f64,
}

impl CostAwareRoutingPolicy {
    /// Validates the policy and min-max normalizes costs across weak and strong.
    ///
    /// With two targets, normalization makes routing depend on cost ordering,
    /// not the magnitude of the difference between the configured costs.
    pub(super) fn new(lambda: f64, weak_cost: f64, strong_cost: f64) -> Result<Self> {
        if !lambda.is_finite() || !(0.0..=1.0).contains(&lambda) {
            return Err(policy_error(
                "routing policy lambda must be finite and in [0.0, 1.0]",
            ));
        }
        validate_cost("weak_cost", weak_cost)?;
        validate_cost("strong_cost", strong_cost)?;

        let (normalized_weak_cost, normalized_strong_cost) = if weak_cost == strong_cost {
            (0.0, 0.0)
        } else {
            let minimum = weak_cost.min(strong_cost);
            let range = weak_cost.max(strong_cost) - minimum;
            (
                (weak_cost - minimum) / range,
                (strong_cost - minimum) / range,
            )
        };

        Ok(Self {
            lambda,
            normalized_weak_cost,
            normalized_strong_cost,
        })
    }

    /// Selects the tier with the greater cost-adjusted correctness utility.
    ///
    /// Equal utilities deterministically select weak.
    pub(super) fn select(
        &self,
        weak_probability: f64,
        strong_probability: f64,
    ) -> Result<PrefillTier> {
        validate_probability("weak", weak_probability)?;
        validate_probability("strong", strong_probability)?;

        let cost_weight = 1.0 - self.lambda;
        let weak_utility = self.lambda * weak_probability - cost_weight * self.normalized_weak_cost;
        let strong_utility =
            self.lambda * strong_probability - cost_weight * self.normalized_strong_cost;
        Ok(if weak_utility >= strong_utility {
            PrefillTier::Weak
        } else {
            PrefillTier::Strong
        })
    }
}

fn validate_cost(field: &str, cost: f64) -> Result<()> {
    if !cost.is_finite() || cost < 0.0 {
        return Err(policy_error(format!(
            "routing policy {field} must be finite and non-negative"
        )));
    }
    Ok(())
}

fn validate_probability(head: &str, probability: f64) -> Result<()> {
    if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return Err(policy_error(format!(
            "{head} checkpoint probability must be finite and in [0.0, 1.0]"
        )));
    }
    Ok(())
}

fn policy_error(message: impl Into<String>) -> LibsyError {
    LibsyError::AlgorithmError {
        message: format!("prefill-probe policy error: {}", message.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lambda_zero_uses_cost_and_weak_wins_equal_utility() -> Result<()> {
        let weak_cheaper = CostAwareRoutingPolicy::new(0.0, 1.0, 10.0)?;
        let strong_cheaper = CostAwareRoutingPolicy::new(0.0, 10.0, 1.0)?;
        let equal_cost = CostAwareRoutingPolicy::new(0.0, 3.0, 3.0)?;

        assert_eq!(weak_cheaper.select(0.0, 1.0)?, PrefillTier::Weak);
        assert_eq!(strong_cheaper.select(1.0, 0.0)?, PrefillTier::Strong);
        assert_eq!(equal_cost.select(0.0, 1.0)?, PrefillTier::Weak);
        Ok(())
    }

    #[test]
    fn lambda_one_uses_correctness_probabilities() -> Result<()> {
        let policy = CostAwareRoutingPolicy::new(1.0, 100.0, 1.0)?;

        assert_eq!(policy.select(0.8, 0.6)?, PrefillTier::Weak);
        assert_eq!(policy.select(0.4, 0.6)?, PrefillTier::Strong);
        assert_eq!(policy.select(0.6, 0.6)?, PrefillTier::Weak);
        Ok(())
    }

    #[test]
    fn invalid_values_are_rejected() -> Result<()> {
        for (lambda, weak_cost, strong_cost, expected) in [
            (f64::NAN, 0.0, 1.0, "lambda"),
            (-0.1, 0.0, 1.0, "lambda"),
            (1.1, 0.0, 1.0, "lambda"),
            (0.5, -1.0, 1.0, "weak_cost"),
            (0.5, 1.0, f64::INFINITY, "strong_cost"),
        ] {
            let error = CostAwareRoutingPolicy::new(lambda, weak_cost, strong_cost)
                .err()
                .ok_or_else(|| policy_error("invalid policy value should fail"))?;
            assert!(error.to_string().contains(expected));
        }

        let policy = CostAwareRoutingPolicy::new(0.5, 0.0, 1.0)?;
        let error = policy
            .select(f64::NAN, 0.5)
            .err()
            .ok_or_else(|| policy_error("invalid probability should fail"))?;
        assert!(error.to_string().contains("weak"));
        Ok(())
    }
}
