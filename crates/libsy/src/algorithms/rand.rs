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

use crate::algorithms::fall_through::FallThrough;
use crate::core::algorithm::{Algorithm, Driver};
use crate::core::classifier::{Classification, Classifier, Score};
use crate::{LibsyError, Result, TargetModalities};
use switchyard_protocol::{ModelId, Request, Response};

/// Stateless weighted classifier used by random fall-through routing.
pub struct RandomClassifier {
    targets: Vec<ModelId>,
    weights: Vec<f64>,
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
    pub fn new(
        targets: Vec<ModelId>,
        weights: Option<Vec<f64>>,
        seed: Option<u64>,
    ) -> Result<Self> {
        let target_count = targets.len();
        if target_count == 0 {
            return Err(LibsyError::NoTargets);
        }
        let unique_targets = targets.iter().map(ModelId::as_str).collect::<BTreeSet<_>>();
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
        let distribution = WeightedIndex::new(weights.clone())
            .map_err(|error| invalid_weights(error.to_string()))?;
        let rng = match seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => rand::make_rng(),
        };
        Ok(Self {
            targets,
            weights,
            distribution,
            rng: Mutex::new(rng),
        })
    }

    fn select_target(&self) -> ModelId {
        let mut rng = self.rng.lock();
        let index = self.distribution.sample(&mut *rng);
        self.targets[index].clone()
    }

    /// Samples from compatible targets using their original relative weights.
    fn select_eligible_target(
        &self,
        eligible_targets: &BTreeSet<ModelId>,
    ) -> Result<Option<ModelId>> {
        let eligible = self
            .targets
            .iter()
            .zip(&self.weights)
            .filter(|(target, _)| eligible_targets.contains(*target))
            .collect::<Vec<_>>();
        let weights = eligible
            .iter()
            .map(|(_, weight)| **weight)
            .collect::<Vec<_>>();
        if !weights.iter().any(|weight| *weight > 0.0) {
            return Ok(None);
        }
        let distribution =
            WeightedIndex::new(weights).map_err(|error| invalid_weights(error.to_string()))?;
        let mut rng = self.rng.lock();
        let index = distribution.sample(&mut *rng);
        Ok(Some(eligible[index].0.clone()))
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

    async fn score_with_eligible_targets(
        &self,
        _state: &mut S,
        _request: &mut Request,
        _driver: Option<&Driver>,
        eligible_targets: &BTreeSet<ModelId>,
    ) -> Result<(Classification, Option<Response>)> {
        let scores = self
            .select_eligible_target(eligible_targets)?
            .map(|target| Score {
                confidence: 1.0,
                target,
            })
            .into_iter()
            .collect();
        Ok((Classification::Scores(scores), None))
    }
}

/// Random router implemented as a stateless fall-through composition.
pub struct Random {
    inner: FallThrough<()>,
}

impl Random {
    /// Creates a router over `targets`.
    ///
    /// # Errors
    ///
    /// Returns an error when targets or weights are invalid for [`RandomClassifier`].
    pub fn new(
        targets: Vec<ModelId>,
        weights: Option<Vec<f64>>,
        seed: Option<u64>,
    ) -> Result<Self> {
        let classifier = Arc::new(RandomClassifier::new(targets.clone(), weights, seed)?);
        let inner = FallThrough::<()>::new(targets)
            .with_name("random")
            .with_decision_reason(random_decision_reason)
            .with_classifier(classifier);
        Ok(Self { inner })
    }

    /// Restricts weighted selection and fallback to modality-compatible targets.
    pub fn with_target_modalities(mut self, target_modalities: TargetModalities) -> Self {
        self.inner = self.inner.with_target_modalities(target_modalities);
        self
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

    async fn route(self: Arc<Self>, driver: Driver, request: Request) -> Result<Response> {
        self.inner.execute(driver, request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    use switchyard_protocol::{
        ContentBlock, ImageSource, InputModality, LlmRequest, Message, Metadata, Role,
        completion_text, text_request,
    };

    use crate::algorithms::util::affinity::AffinityRouter;
    use crate::core::testing::{echo, test_drive};
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

    fn target_set(names: &[&str]) -> Vec<ModelId> {
        names.iter().map(|name| ModelId::from(*name)).collect()
    }

    fn target_modalities(entries: &[(&str, &[InputModality])]) -> TargetModalities {
        entries
            .iter()
            .map(|(target, modalities)| {
                (ModelId::from(*target), modalities.iter().copied().collect())
            })
            .collect()
    }

    fn image_request() -> Request {
        Request {
            llm_request: LlmRequest {
                model: Some("auto".to_string()),
                messages: vec![Message {
                    role: Role::User,
                    content: vec![
                        ContentBlock::Text {
                            text: "describe this".to_string(),
                        },
                        ContentBlock::Image {
                            source: ImageSource::Url {
                                url: "https://example.test/image.png".to_string(),
                                detail: None,
                            },
                        },
                    ],
                }],
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: None,
        }
    }

    fn algorithm(names: &[&str], weights: Option<Vec<f64>>, seed: Option<u64>) -> Result<Random> {
        Random::new(target_set(names), weights, seed)
    }

    fn shared_algorithm(names: &[&str]) -> Result<Arc<dyn Algorithm>> {
        Ok(Arc::new(algorithm(names, None, None)?))
    }

    async fn selected_models(algorithm: Arc<dyn Algorithm>, count: usize) -> Result<Vec<String>> {
        let mut selected = Vec::with_capacity(count);
        for _ in 0..count {
            let (_, response) = test_drive(algorithm.clone(), request(), echo()).await?;
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

    #[tokio::test]
    async fn single_target_is_always_selected_and_called() -> Result<()> {
        let algorithm = shared_algorithm(&["only/model"])?;
        let (trace, response) = test_drive(algorithm, request(), echo()).await?;

        assert_eq!(
            response
                .llm_response
                .as_agg()
                .map(completion_text)
                .unwrap_or_default(),
            "only/model"
        );
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].selected_model_id(), "only/model");
        Ok(())
    }

    #[tokio::test]
    async fn selected_target_is_in_the_set_and_matches_the_trace() -> Result<()> {
        let names = ["a/model", "b/model", "c/model"];
        let algorithm = shared_algorithm(&names)?;

        for _ in 0..50 {
            let (trace, response) = test_drive(algorithm.clone(), request(), echo()).await?;
            let selected = response
                .llm_response
                .as_agg()
                .map(completion_text)
                .unwrap_or_default();
            assert!(
                names.contains(&selected.as_str()),
                "selected {selected} not in target set"
            );
            assert_eq!(trace[0].selected_model_id(), selected.as_str());
        }
        Ok(())
    }

    #[tokio::test]
    async fn selection_covers_all_targets_over_many_runs() -> Result<()> {
        let algorithm = shared_algorithm(&["a/model", "b/model"])?;
        let mut seen = HashSet::new();

        for _ in 0..100 {
            let (_, response) = test_drive(algorithm.clone(), request(), echo()).await?;
            seen.insert(
                response
                    .llm_response
                    .as_agg()
                    .map(completion_text)
                    .unwrap_or_default(),
            );
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
        let first: Arc<dyn Algorithm> = Arc::new(algorithm(
            &["a/model", "b/model"],
            Some(vec![1.0, 3.0]),
            Some(42),
        )?);
        let second: Arc<dyn Algorithm> = Arc::new(algorithm(
            &["a/model", "b/model"],
            Some(vec![1.0, 3.0]),
            Some(42),
        )?);

        let first_selections = selected_models(first, 1_000).await?;
        let second_selections = selected_models(second, 1_000).await?;
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
    async fn weighted_selection_is_restricted_and_renormalized_over_compatible_targets()
    -> Result<()> {
        fn modality_aware_random() -> Result<Arc<dyn Algorithm>> {
            let router = Random::new(
                target_set(&["text", "vision-a", "vision-b"]),
                Some(vec![100.0, 1.0, 3.0]),
                Some(42),
            )?
            .with_target_modalities(target_modalities(&[
                ("text", &[InputModality::Text]),
                ("vision-a", &[InputModality::Text, InputModality::Image]),
                ("vision-b", &[InputModality::Text, InputModality::Image]),
            ]));
            Ok(Arc::new(router))
        }

        async fn selections(router: Arc<dyn Algorithm>) -> Result<Vec<String>> {
            let mut selected = Vec::new();
            for _ in 0..1_000 {
                let (_, response) = test_drive(router.clone(), image_request(), echo()).await?;
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

        let first = selections(modality_aware_random()?).await?;
        let second = selections(modality_aware_random()?).await?;

        assert_eq!(first, second);
        assert!(first.iter().all(|target| target != "text"));
        let vision_b = first
            .iter()
            .filter(|target| target.as_str() == "vision-b")
            .count();
        assert!(
            (700..=800).contains(&vision_b),
            "expected a roughly 25/75 split, selected vision-b {vision_b} times"
        );
        Ok(())
    }

    #[tokio::test]
    async fn affinity_reuses_the_initial_random_selection() -> Result<()> {
        let names = ["a/model", "b/model"];
        let affinity = Arc::new(AffinityRouter::new());
        let random = Arc::new(RandomClassifier::new(
            names.iter().map(|name| ModelId::from(*name)).collect(),
            None,
            Some(42),
        )?);
        let algorithm: Arc<dyn Algorithm> = Arc::new(
            FallThrough::<()>::new(target_set(&names))
                .with_name("affinity_random")
                .with_processor(affinity.clone())
                .with_classifier(affinity.clone())
                .with_classifier(random),
        );

        let (_, first) =
            test_drive(algorithm.clone(), request_for_session("session-1"), echo()).await?;
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

        let (_, second) = test_drive(algorithm, request_for_session("session-1"), echo()).await?;
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

    #[tokio::test]
    async fn incompatible_affinity_is_bypassed_for_a_multimodal_turn() -> Result<()> {
        let names = ["text", "vision"];
        let affinity = Arc::new(AffinityRouter::new());
        let random = Arc::new(RandomClassifier::new(
            target_set(&names),
            Some(vec![1.0, 0.0]),
            Some(42),
        )?);
        let algorithm: Arc<dyn Algorithm> = Arc::new(
            FallThrough::<()>::new(target_set(&names))
                .with_name("affinity_random")
                .with_target_modalities(target_modalities(&[
                    ("text", &[InputModality::Text]),
                    ("vision", &[InputModality::Text, InputModality::Image]),
                ]))
                .with_processor(affinity.clone())
                .with_classifier(affinity)
                .with_classifier(random),
        );

        let (_, first) = test_drive(
            algorithm.clone(),
            request_for_session("session-modalities"),
            echo(),
        )
        .await?;
        let mut image = image_request();
        image.metadata = Some(Metadata {
            session_id: Some("session-modalities".to_string()),
            ..Metadata::default()
        });
        let (_, second) = test_drive(algorithm, image, echo()).await?;

        assert_eq!(
            first
                .llm_response
                .as_agg()
                .map(completion_text)
                .unwrap_or_default(),
            "text"
        );
        assert_eq!(
            second
                .llm_response
                .as_agg()
                .map(completion_text)
                .unwrap_or_default(),
            "vision"
        );
        Ok(())
    }

    #[test]
    fn rejects_invalid_weights() {
        let cases = [
            (vec![1.0], "expected 2 weights"),
            (vec![1.0, -1.0], "finite and nonnegative"),
            (vec![0.0, 0.0], "at least one weight must be positive"),
            (vec![1.0, f64::INFINITY], "finite and nonnegative"),
        ];

        for (weights, expected) in cases {
            let error = algorithm(&["a/model", "b/model"], Some(weights), None)
                .err()
                .map(|error| error.to_string())
                .unwrap_or_default();
            assert!(error.contains(expected), "unexpected error: {error}");
        }
    }

    #[test]
    fn rejects_invalid_targets() {
        let error = algorithm(&[], None, None).err();
        assert!(matches!(error, Some(LibsyError::NoTargets)));

        let error = algorithm(&["same/model", "same/model"], None, None)
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();
        assert!(error.contains("random targets must be unique"));
    }

    #[tokio::test]
    async fn decision_is_inspectable() -> Result<()> {
        let algorithm = shared_algorithm(&["only/model"])?;
        let (trace, _) = test_drive(algorithm, request(), echo()).await?;
        let decision = &trace[0];

        assert_eq!(decision.selected_model_id(), "only/model");
        assert!(decision.is_answer_call());
        Ok(())
    }
}
