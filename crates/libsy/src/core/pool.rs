// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A fixed score line shared by classifiers and the final target resolver.

use crate::{LibsyError, Result};
use switchyard_protocol::ModelId;

/// Ordered target anchors and the boundaries halfway between them.
#[derive(Clone, Debug)]
pub struct TargetPool {
    targets: Vec<ModelId>,
    positions: Vec<f64>,
}

impl TargetPool {
    /// Targets in increasing score order. An empty line uses evenly spaced anchors.
    pub fn new(targets: Vec<ModelId>, positions: Vec<f64>) -> Result<Self> {
        if targets.is_empty() {
            return Err(LibsyError::NoTargets);
        }
        if targets
            .iter()
            .enumerate()
            .any(|(i, target)| targets[..i].contains(target))
        {
            return Err(invalid("pool targets must be distinct"));
        }
        let positions = if positions.is_empty() {
            (0..targets.len())
                .map(|i| i as f64 / targets.len().saturating_sub(1).max(1) as f64)
                .collect()
        } else {
            positions
        };
        if positions.len() != targets.len()
            || positions
                .iter()
                .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
            || positions.windows(2).any(|pair| {
                let midpoint = (pair[0] + pair[1]) / 2.0;
                pair[0] >= midpoint || midpoint >= pair[1]
            })
            || positions[0] != 0.0
            || (positions.len() > 1 && positions.last() != Some(&1.0))
        {
            return Err(invalid(
                "pool positions must match targets, increase with distinct midpoints, and span 0 to 1",
            ));
        }
        Ok(Self { targets, positions })
    }

    /// Number of targets on the score line.
    pub fn target_count(&self) -> usize {
        self.targets.len()
    }

    pub(crate) fn validate_targets(&self, targets: &[ModelId]) -> Result<()> {
        for target in &self.targets {
            super::algorithm::ensure_model_is_target(targets, target)?;
        }
        Ok(())
    }

    pub(crate) fn position(&self, target: &ModelId) -> Result<f64> {
        self.targets
            .iter()
            .position(|candidate| candidate == target)
            .map(|index| self.positions[index])
            .ok_or_else(|| {
                invalid(format!(
                    "classifier selected target {target} outside the score pool"
                ))
            })
    }

    /// Distance from the nearest midpoint, normalized within its adjacent interval.
    /// Zero is ambiguous; one is an anchor. This reduces to abs(2*p - 1) for two targets.
    pub fn confidence(&self, value: f64) -> Result<f64> {
        validate_value(value)?;
        Ok(self
            .positions
            .windows(2)
            .find(|pair| value >= pair[0] && value <= pair[1])
            .map_or(1.0, |pair| {
                (2.0 * (value - (pair[0] + pair[1]) / 2.0).abs() / (pair[1] - pair[0])).min(1.0)
            }))
    }

    /// Continuous scores on a closed ambiguity band fall through to the next classifier.
    pub fn is_clear(&self, value: f64, threshold: f64) -> Result<bool> {
        validate_value(value)?;
        validate_value(threshold)?;
        // At one the bands cover the line; midpoint rounding must not open gaps.
        if threshold == 1.0 {
            return Ok(false);
        }
        Ok(!self.positions.windows(2).any(|pair| {
            let midpoint = (pair[0] + pair[1]) / 2.0;
            let width = threshold * (pair[1] - pair[0]) / 2.0;
            value >= midpoint - width && value <= midpoint + width
        }))
    }

    /// Resolve once, after the cascade accepts a clear score. Midpoint ties go left.
    pub fn resolve(&self, value: f64) -> Result<&ModelId> {
        validate_value(value)?;
        let index = self
            .positions
            .windows(2)
            .take_while(|pair| value > (pair[0] + pair[1]) / 2.0)
            .count();
        Ok(&self.targets[index])
    }
}

pub(crate) fn validate_value(value: f64) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(invalid(format!(
            "classifier score must be finite and between 0 and 1, got {value}"
        )));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> LibsyError {
    LibsyError::AlgorithmError {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn line_and_relative_bands() -> Result<()> {
        let pool = TargetPool::new(
            vec!["small".into(), "middle".into(), "large".into()],
            vec![0.0, 0.4, 1.0],
        )?;
        for (score, target, confidence) in [
            (0.0, "small", 1.0),
            (0.2, "small", 0.0),
            (0.4, "middle", 1.0),
            (0.7, "middle", 0.0),
            (1.0, "large", 1.0),
        ] {
            assert_eq!(pool.resolve(score)?.as_str(), target);
            assert!((pool.confidence(score)? - confidence).abs() < 1e-12);
            assert!((0.0..=1.0).contains(&pool.confidence(score)?));
        }
        Ok(())
    }
    #[test]
    fn rejects_invalid_lines_and_scores() {
        for line in [
            vec![0.0],
            vec![0.1, 1.0],
            vec![0.0, 0.8],
            vec![0.0, f64::NAN],
        ] {
            assert!(TargetPool::new(vec!["a".into(), "b".into()], line).is_err());
        }
        assert!(TargetPool::new(vec!["a".into(), "a".into()], vec![]).is_err());
        let pool = TargetPool::new(vec!["a".into()], vec![]).unwrap();
        for score in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
            assert!(pool.resolve(score).is_err());
        }
    }
    #[test]
    fn binary_bands_and_multi_target_boundaries_are_closed() -> Result<()> {
        let binary = TargetPool::new(vec!["a".into(), "b".into()], vec![])?;
        for threshold in [0.0, 0.2, 0.5, 1.0] {
            for i in 0..=1000 {
                let p = i as f64 / 1000.0;
                assert_eq!(
                    binary.is_clear(p, threshold)?,
                    p < 0.5 - threshold / 2.0 || p > 0.5 + threshold / 2.0
                );
            }
        }
        let pool = TargetPool::new(vec!["a".into(), "b".into(), "c".into()], vec![])?;
        for p in [0.25, 0.75] {
            assert!(!pool.is_clear(p, 0.0)?);
        }
        for p in [0.0, 0.5, 1.0] {
            assert!(pool.is_clear(p, 0.5)?);
        }
        for p in [0.125, 0.375, 0.625, 0.875] {
            assert!(!pool.is_clear(p, 0.5)?);
        }
        let uneven = TargetPool::new(
            vec!["a".into(), "b".into(), "c".into(), "d".into(), "e".into()],
            vec![
                0.0,
                0.34284326542215604,
                0.471240086114993,
                0.5346217948794222,
                1.0,
            ],
        )?;
        for value in &uneven.positions {
            assert!(
                !uneven.is_clear(*value, 1.0)?,
                "anchor {value} escaped the closed band"
            );
        }
        Ok(())
    }
    #[test]
    fn runtime_pool_rejects_targets_outside_the_completion_group() -> Result<()> {
        use switchyard_protocol::Category;
        let models = crate::RuntimeModels::new(
            [
                (Category::Any, vec!["small".into()]),
                (Category::Judge, vec!["judge".into()]),
            ]
            .into(),
        );
        for target in ["missing", "judge"] {
            let error = models
                .clone()
                .with_pool(TargetPool::new(
                    vec!["small".into(), target.into()],
                    vec![],
                )?)
                .expect_err("pool targets must be completion targets");
            assert!(matches!(error, LibsyError::TargetNotFound { target: name } if name == target));
        }
        Ok(())
    }

    #[test]
    fn parent_line_does_not_leak_into_subagent_scope() -> Result<()> {
        use switchyard_protocol::Category;
        let models = crate::RuntimeModels::new(
            [(
                Category::Any,
                vec!["small".into(), "middle".into(), "large".into()],
            )]
            .into(),
        )
        .with_pool(TargetPool::new(
            vec!["small".into(), "middle".into(), "large".into()],
            vec![0.0, 0.4, 1.0],
        )?)?
        .with_subagent(
            [
                (Category::Efficient, vec!["worker-small".into()]),
                (Category::Capable, vec!["worker-large".into()]),
            ]
            .into(),
        );
        let (driver, _) = crate::Driver::new("test", std::sync::Arc::new(models));
        assert_eq!(driver.pool().unwrap().resolve(0.4)?.as_str(), "middle");
        let worker = driver.for_subagent()?;
        assert!(worker.pool().is_none());
        Ok(())
    }
}
