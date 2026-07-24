// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Private cost-aware policy for converting learned correctness into a binary routing score.

use switchyard_core::{Result, SwitchyardError};

/// Validated utility policy for the two configured completion targets.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CostAwareRoutingPolicy {
    lambda: f64,
    normalized_weak_cost: f64,
    normalized_strong_cost: f64,
}

impl CostAwareRoutingPolicy {
    /// Validates the policy and min-max normalizes costs across weak and strong.
    pub(crate) fn new(lambda: f64, weak_cost: f64, strong_cost: f64) -> Result<Self> {
        if !lambda.is_finite() || !(0.0..=1.0).contains(&lambda) {
            return Err(SwitchyardError::InvalidConfig(
                "routing_policy.lambda must be finite and in [0.0, 1.0]".into(),
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

    /// Returns `1.0` for weak and `0.0` for strong from two mapped probabilities.
    pub(crate) fn score(&self, weak_probability: f64, strong_probability: f64) -> Result<f64> {
        validate_probability("weak", weak_probability)?;
        validate_probability("strong", strong_probability)?;

        let cost_weight = 1.0 - self.lambda;
        let weak_utility = self.lambda * weak_probability - cost_weight * self.normalized_weak_cost;
        let strong_utility =
            self.lambda * strong_probability - cost_weight * self.normalized_strong_cost;
        let margin = weak_utility - strong_utility;
        Ok(if margin >= 0.0 { 1.0 } else { 0.0 })
    }
}

fn validate_cost(field: &str, cost: f64) -> Result<()> {
    if !cost.is_finite() || cost < 0.0 {
        return Err(SwitchyardError::InvalidConfig(format!(
            "routing_policy.{field} must be finite and non-negative"
        )));
    }
    Ok(())
}

fn validate_probability(head: &str, probability: f64) -> Result<()> {
    if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return Err(SwitchyardError::Other(format!(
            "{head} checkpoint probability must be finite and in [0.0, 1.0]"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lambda_zero_uses_only_cost_and_selects_weak_on_tie() -> Result<()> {
        let weak_cheaper = CostAwareRoutingPolicy::new(0.0, 1.0, 10.0)?;
        let strong_cheaper = CostAwareRoutingPolicy::new(0.0, 10.0, 1.0)?;
        let equal_cost = CostAwareRoutingPolicy::new(0.0, 3.0, 3.0)?;

        assert_eq!(weak_cheaper.score(0.0, 1.0)?, 1.0);
        assert_eq!(strong_cheaper.score(1.0, 0.0)?, 0.0);
        assert_eq!(equal_cost.score(0.0, 1.0)?, 1.0);
        assert_eq!(equal_cost.normalized_weak_cost, 0.0);
        assert_eq!(equal_cost.normalized_strong_cost, 0.0);
        Ok(())
    }

    #[test]
    fn lambda_one_uses_only_correctness_probabilities() -> Result<()> {
        let policy = CostAwareRoutingPolicy::new(1.0, 100.0, 1.0)?;

        assert_eq!(policy.score(0.8, 0.6)?, 1.0);
        assert_eq!(policy.score(0.4, 0.6)?, 0.0);
        assert_eq!(policy.score(0.6, 0.6)?, 1.0);
        Ok(())
    }

    #[test]
    fn lambda_changes_the_route_without_another_threshold() -> Result<()> {
        let balanced = CostAwareRoutingPolicy::new(0.5, 0.0, 1.0)?;
        let correctness_favoring = CostAwareRoutingPolicy::new(0.9, 0.0, 1.0)?;

        assert_eq!(balanced.score(0.4, 0.8)?, 1.0);
        assert_eq!(correctness_favoring.score(0.4, 0.8)?, 0.0);
        Ok(())
    }

    #[test]
    fn unmapped_checkpoint_outputs_do_not_affect_the_score() -> Result<()> {
        let first_outputs = [0.99, 0.7, 0.4, 0.01];
        let second_outputs = [0.01, 0.7, 0.4, 0.99];
        let policy = CostAwareRoutingPolicy::new(1.0, 0.0, 0.0)?;

        assert_eq!(
            policy.score(first_outputs[1], first_outputs[2])?,
            policy.score(second_outputs[1], second_outputs[2])?,
        );
        Ok(())
    }

    #[test]
    fn invalid_policy_values_are_rejected() -> Result<()> {
        let cases = [
            (f64::NAN, 0.0, 1.0, "lambda"),
            (-0.1, 0.0, 1.0, "lambda"),
            (1.1, 0.0, 1.0, "lambda"),
            (0.5, -1.0, 1.0, "weak_cost"),
            (0.5, 1.0, f64::INFINITY, "strong_cost"),
        ];
        for (lambda, weak_cost, strong_cost, expected) in cases {
            let error = CostAwareRoutingPolicy::new(lambda, weak_cost, strong_cost)
                .err()
                .ok_or_else(|| {
                    SwitchyardError::Other(format!(
                        "invalid policy value for {expected} should fail"
                    ))
                })?;
            assert!(format!("{error}").contains(expected));
        }
        Ok(())
    }

    #[test]
    fn invalid_checkpoint_probabilities_are_rejected() -> Result<()> {
        let policy = CostAwareRoutingPolicy::new(0.5, 0.0, 1.0)?;
        let cases = [
            (f64::NAN, 0.5, "weak"),
            (-0.1, 0.5, "weak"),
            (0.5, 1.1, "strong"),
            (0.5, f64::INFINITY, "strong"),
        ];
        for (weak, strong, expected) in cases {
            let error = policy.score(weak, strong).err().ok_or_else(|| {
                SwitchyardError::Other(format!(
                    "invalid checkpoint probability for {expected} should fail"
                ))
            })?;
            assert!(format!("{error}").contains(expected));
        }
        Ok(())
    }
}
