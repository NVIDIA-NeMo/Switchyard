// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Random routing as a stateless [`FallThrough`] composition.
//!
//! [`RandomClassifier`] selects one target; [`FallThrough`] owns the common
//! processor/classifier/target-call orchestration.

use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use rand::RngExt as _;
use rand::SeedableRng;
use rand::distr::{Distribution, weighted::WeightedIndex};
use rand::rngs::StdRng;

use crate::algorithms::fall_through::FallThrough;
use crate::core::algorithm::{Algorithm, Driver};
use crate::core::classifier::{Classification, Classifier, Score};
use crate::{LibsyError, Result};
use switchyard_protocol::{Category, Request, Response};

/// Stateless weighted classifier used by random fall-through routing.
pub struct RandomClassifier {
    distribution: Option<WeightedIndex<f64>>,
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
    /// Returns an error explicit weights are negative or non-finite, or contain no
    /// positive value.
    pub fn new(weights: Option<Vec<f64>>, seed: Option<u64>) -> Result<Self> {
        let distribution = if let Some(weights) = weights {
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
            Some(WeightedIndex::new(weights).map_err(|error| invalid_weights(error.to_string()))?)
        } else {
            None
        };
        let rng = match seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => rand::make_rng(),
        };
        Ok(Self {
            distribution,
            rng: Mutex::new(rng),
        })
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
        driver: Option<&Driver>,
    ) -> Result<(Classification, Option<Response>)> {
        let Some(driver) = driver else {
            // Temp until can remove Option from driver
            return Err(LibsyError::NoTargets);
        };
        // All the available models
        let options = driver.models_for(Category::Any);
        if options.is_empty() {
            return Err(LibsyError::NoTargets);
        }
        let mut rng = self.rng.lock();
        let index = if let Some(distribution) = self.distribution.as_ref() {
            // The user gave us weights
            distribution.sample(&mut *rng)
        } else {
            // No weights, assume equal probability
            rng.random_range(..options.len())
        };
        let target = options[index].clone();
        Ok((
            Classification::Scores(vec![Score {
                confidence: 1.0,
                target,
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
    /// Creates a random router. The models themselves will be passed at runtime.
    pub fn new(weights: Option<Vec<f64>>, seed: Option<u64>) -> Result<Self> {
        let classifier = Arc::new(RandomClassifier::new(weights, seed)?);
        let inner = FallThrough::<()>::new(vec![])
            .with_name("random")
            .with_classifier(classifier);
        Ok(Self { inner })
    }
}

#[async_trait]
impl Algorithm for Random {
    fn name(&self) -> &str {
        "random"
    }

    async fn route(
        self: Arc<Self>,
        driver: Driver,
        request: Request,
    ) -> Result<crate::RoutingOutcome> {
        self.inner.execute(driver, request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    use switchyard_protocol::{Metadata, ModelId, completion_text, text_request};

    use crate::algorithms::util::affinity::AffinityRouter;
    use crate::core::testing::{echo, test_drive_with_models};
    use switchyard_protocol::Request;

    fn request() -> Request {
        Request {
            llm_request: text_request(Some("auto".to_string()), "hi"),
            raw_request: None,
            metadata: None,
        }
    }

    fn request_for_session(session_id: &str) -> Request {
        Request {
            metadata: Some(Metadata {
                session_id: Some(session_id.to_string()),
                ..Metadata::default()
            }),
            ..request()
        }
    }

    /*
    fn target_set(names: &[&str]) -> Vec<ModelId> {
        names.iter().map(|name| ModelId::from(*name)).collect()
    }
    */

    fn algorithm(weights: Option<Vec<f64>>, seed: Option<u64>) -> Result<Random> {
        Random::new(weights, seed)
    }

    fn shared_algorithm() -> Result<Arc<dyn Algorithm>> {
        Ok(Arc::new(algorithm(None, None)?))
    }

    async fn selected_models(
        algorithm: Arc<dyn Algorithm>,
        count: usize,
        models: HashMap<Category, Vec<ModelId>>,
    ) -> Result<Vec<String>> {
        let mut selected = Vec::with_capacity(count);
        for _ in 0..count {
            let (_, response) =
                test_drive_with_models(algorithm.clone(), request(), models.clone(), echo())
                    .await?;
            selected.push(
                response
                    .llm_response
                    .as_agg()
                    .map(completion_text)
                    .unwrap_or_default(),
            );
        }
        Ok(selected)
    }

    fn to_category_map(names: &[&str]) -> HashMap<Category, Vec<ModelId>> {
        Category::to_map(Category::Any, names)
    }

    #[tokio::test]
    async fn single_target_is_always_selected_and_called() -> Result<()> {
        let algorithm = shared_algorithm()?;
        let models = to_category_map(&["only/model"]);
        let (selected_model, response) =
            test_drive_with_models(algorithm, request(), models, echo()).await?;

        assert_eq!(
            response
                .llm_response
                .as_agg()
                .map(completion_text)
                .unwrap_or_default(),
            "only/model"
        );
        assert_eq!(selected_model, "only/model");
        Ok(())
    }

    #[tokio::test]
    async fn selection_covers_all_targets_over_many_runs() -> Result<()> {
        let algorithm = shared_algorithm()?;
        let models = to_category_map(&["a/model", "b/model"]);
        let mut seen = HashSet::new();

        for _ in 0..100 {
            let (selected_model, response) =
                test_drive_with_models(algorithm.clone(), request(), models.clone(), echo())
                    .await?;
            let served_model = response
                .llm_response
                .as_agg()
                .map(completion_text)
                .unwrap_or_default();
            assert_eq!(selected_model, served_model.as_str());
            seen.insert(served_model);
        }

        // Missing either target after 100 uniform draws has probability about 2^-99.
        assert_eq!(
            seen.len(),
            2,
            "expected both targets to be selected, saw {seen:?}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn weighted_seeded_selection_is_reproducible() -> Result<()> {
        let models = to_category_map(&["a/model", "b/model"]);
        let first: Arc<dyn Algorithm> = Arc::new(algorithm(Some(vec![1.0, 3.0]), Some(42))?);
        let second: Arc<dyn Algorithm> = Arc::new(algorithm(Some(vec![1.0, 3.0]), Some(42))?);

        let first_selections = selected_models(first, 1_000, models.clone()).await?;
        let second_selections = selected_models(second, 1_000, models).await?;
        assert_eq!(first_selections, second_selections);

        let second_count = first_selections
            .iter()
            .filter(|model| model.as_str() == "b/model")
            .count();
        assert!(
            (700..=800).contains(&second_count),
            "expected a roughly 25/75 split, selected b/model {second_count} times"
        );
        Ok(())
    }

    #[tokio::test]
    async fn affinity_reuses_the_initial_random_selection() -> Result<()> {
        let names = ["a/model", "b/model"];
        let models = to_category_map(&names);
        let affinity = Arc::new(AffinityRouter::new());
        let random = Arc::new(RandomClassifier::new(None, Some(42))?);
        let algorithm: Arc<dyn Algorithm> = Arc::new(
            FallThrough::<()>::new(vec![])
                .with_name("affinity_random")
                .with_processor(affinity.clone())
                .with_classifier(affinity.clone())
                .with_classifier(random),
        );

        let (_, first) = test_drive_with_models(
            algorithm.clone(),
            request_for_session("session-1"),
            models.clone(),
            echo(),
        )
        .await?;
        let selected = first
            .llm_response
            .as_agg()
            .map(completion_text)
            .unwrap_or_default();

        let mut state = ();
        let mut request = request_for_session("session-1");
        let retained = affinity
            .score(&mut state, &mut request, None)
            .await?
            .0
            .argmax(false)?;
        assert_eq!(
            retained.map(|score| score.target),
            Some(ModelId::from(selected.clone()))
        );

        let (_, second) =
            test_drive_with_models(algorithm, request_for_session("session-1"), models, echo())
                .await?;
        assert_eq!(
            second
                .llm_response
                .as_agg()
                .map(completion_text)
                .unwrap_or_default(),
            selected
        );
        Ok(())
    }

    #[test]
    fn rejects_invalid_weights() {
        let cases = [
            (vec![1.0, -1.0], "finite and nonnegative"),
            (vec![0.0, 0.0], "at least one weight must be positive"),
            (vec![1.0, f64::INFINITY], "finite and nonnegative"),
        ];

        for (weights, expected) in cases {
            let error = algorithm(Some(weights), None)
                .err()
                .map(|error| error.to_string())
                .unwrap_or_default();
            assert!(error.contains(expected), "unexpected error: {error}");
        }
    }

    #[tokio::test]
    async fn decision_is_inspectable() -> Result<()> {
        let algorithm = shared_algorithm()?;
        let models = to_category_map(&["only/model"]);
        let (selected_model, _) =
            test_drive_with_models(algorithm, request(), models, echo()).await?;
        assert_eq!(selected_model, "only/model");
        Ok(())
    }
}
