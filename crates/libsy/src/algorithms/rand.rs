// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Random routing as a stateless [`FallThrough`] composition.
//!
//! [`RandomClassifier`] selects one target; [`FallThrough`] owns the common
//! processor/classifier/target-call orchestration.

use std::collections::BTreeSet;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use rand::SeedableRng;
use rand::distr::{Distribution, weighted::WeightedIndex};
use rand::rngs::StdRng;

use crate::algorithms::fall_through::{FallThrough, FallThroughDecision};
use crate::core::algorithm::{Algorithm, Driver, LlmTargetSet};
use crate::core::classifier::{Classification, Classifier, Score};
use crate::{LibsyError, Result};
use switchyard_protocol::{Context, Request, Response};

/// Compatibility name for the decision produced by [`Random`].
pub type RandomDecision = FallThroughDecision;

/// Stateless weighted classifier used by random fall-through routing.
pub struct RandomClassifier {
    targets: Vec<String>,
    distribution: WeightedIndex<f64>,
    rng: Mutex<StdRng>,
}

impl RandomClassifier {
    /// Creates a classifier over ordered target names.
    ///
    /// Missing weights default to one per target. Explicit weights are relative,
    /// follow target order, and need not sum to one. Zero disables a target.
    /// Missing `seed` uses entropy-backed randomness.
    ///
    /// # Errors
    ///
    /// Returns an error when targets are empty or duplicated, or when explicit
    /// weights have the wrong length, are negative or non-finite, or contain no
    /// positive value.
    pub fn new(targets: Vec<String>, weights: Option<Vec<f64>>, seed: Option<u64>) -> Result<Self> {
        let target_count = targets.len();
        if target_count == 0 {
            return Err(LibsyError::NoTargets);
        }
        let unique_targets = targets.iter().map(String::as_str).collect::<BTreeSet<_>>();
        if unique_targets.len() != target_count {
            return Err(LibsyError::AlgorithmError {
                message: "random targets must be unique".to_string(),
            });
        }

        let weights = weights.unwrap_or_else(|| vec![1.0; target_count]);
        if weights.len() != target_count {
            return Err(invalid_weights(format!(
                "expected {target_count} weights, got {}",
                weights.len()
            )));
        }
        if weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return Err(invalid_weights(
                "weights must be finite and nonnegative".to_string(),
            ));
        }
        if !weights.iter().any(|weight| *weight > 0.0) {
            return Err(invalid_weights(
                "at least one weight must be positive".to_string(),
            ));
        }
        let distribution =
            WeightedIndex::new(weights).map_err(|error| invalid_weights(error.to_string()))?;
        let rng = match seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => rand::make_rng(),
        };
        Ok(Self {
            targets,
            distribution,
            rng: Mutex::new(rng),
        })
    }

    fn select_target(&self) -> String {
        let mut rng = self.rng.lock();
        let index = self.distribution.sample(&mut *rng);
        self.targets[index].clone()
    }
}

fn invalid_weights(message: String) -> LibsyError {
    LibsyError::AlgorithmError {
        message: format!("invalid random weights: {message}"),
    }
}

#[async_trait]
impl<S> Classifier<S> for RandomClassifier
where
    S: Send + 'static,
{
    async fn score(
        &self,
        _state: &mut S,
        _request: &mut Request,
        _driver: Option<&Driver>,
    ) -> Result<(Classification, Option<Response>)> {
        Ok((
            Classification::Scores(vec![Score {
                confidence: 1.0,
                target: self.select_target(),
            }]),
            None,
        ))
    }
}

/// Random router implemented as a stateless fall-through composition.
pub struct Random {
    inner: FallThrough<()>,
}

impl Random {
    /// Creates a router over `target_set`.
    ///
    /// # Errors
    ///
    /// Returns an error when targets or weights are invalid for [`RandomClassifier`].
    pub fn new(
        target_set: LlmTargetSet,
        weights: Option<Vec<f64>>,
        seed: Option<u64>,
    ) -> Result<Self> {
        let target_names = target_set
            .targets()
            .iter()
            .map(|target| target.semantic_name.clone())
            .collect();
        let classifier = Arc::new(RandomClassifier::new(target_names, weights, seed)?);
        let inner = FallThrough::<()>::new(target_set)
            .with_name("random")
            .with_decision_reason(random_decision_reason)
            .with_classifier(classifier);
        Ok(Self { inner })
    }
}

fn random_decision_reason(_name: &str, winner: &Score) -> String {
    format!("random routing selected target '{}'", winner.target)
}

#[async_trait]
impl Algorithm for Random {
    fn name(&self) -> &str {
        "random"
    }

    async fn create_run_task(
        self: Arc<Self>,
        ctx: Context,
        driver: Driver,
        request: Request,
    ) -> Result<Response> {
        self.inner.execute(ctx, driver, request).await
    }
}

#[cfg(test)]
#[path = "rand_tests.rs"]
mod tests;
