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
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::algorithms::fall_through::{FallThrough, FallThroughDecision};
use crate::{
    Algorithm, Classification, Classifier, Context, Driver, LibsyError, LlmTargetSet, Request,
    Response, Result, RoutedLlmClient, Score,
};

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
            None => StdRng::from_entropy(),
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
    ) -> Result<Classification> {
        Ok(Classification::Scores(vec![Score {
            confidence: 1.0,
            target: self.select_target(),
        }]))
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

    fn count_tokens_client(&self) -> Option<Arc<dyn RoutedLlmClient>> {
        self.inner.count_tokens_client()
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
mod tests {
    use super::*;
    use std::collections::HashSet;

    use switchyard_protocol::{completion_text, text_request, text_response, Metadata};

    use crate::algorithms::AffinityRouter;
    use crate::{Decision, LlmResponse, LlmTarget, Request, RoutedLlmClient, Signals};

    /// Echoes the selected target so tests can inspect which target was called.
    struct EchoClient;

    #[async_trait]
    impl RoutedLlmClient for EchoClient {
        async fn call(
            &self,
            _ctx: Context,
            _request: Request,
            decision: Arc<dyn Decision>,
        ) -> std::result::Result<Response, switchyard_protocol::LlmClientError> {
            Ok(Response {
                llm_response: LlmResponse::Agg(text_response(None, decision.selected_model())),
                metadata: None,
            })
        }
    }

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

    fn target_set(names: &[&str]) -> LlmTargetSet {
        let targets = names
            .iter()
            .map(|name| LlmTarget {
                semantic_name: (*name).to_string(),
                llm_client: Some(Arc::new(EchoClient)),
            })
            .collect();
        LlmTargetSet::new(targets)
    }

    /// Builds a random router whose targets all share an echo client.
    fn algorithm(names: &[&str], weights: Option<Vec<f64>>, seed: Option<u64>) -> Result<Random> {
        Random::new(target_set(names), weights, seed)
    }

    fn shared_algorithm(names: &[&str]) -> Result<Arc<dyn Algorithm>> {
        Ok(Arc::new(algorithm(names, None, None)?))
    }

    async fn selected_models(algorithm: Arc<dyn Algorithm>, count: usize) -> Result<Vec<String>> {
        let mut selected = Vec::with_capacity(count);
        for _ in 0..count {
            let (_, response) = algorithm.clone().run(Context::default(), request()).await?;
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
        let (trace, response) = algorithm.run(Context::default(), request()).await?;

        assert_eq!(
            response
                .llm_response
                .as_agg()
                .map(completion_text)
                .unwrap_or_default(),
            "only/model"
        );
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].selected_model(), "only/model");
        Ok(())
    }

    #[tokio::test]
    async fn selected_target_is_in_the_set_and_matches_the_trace() -> Result<()> {
        let names = ["a/model", "b/model", "c/model"];
        let algorithm = shared_algorithm(&names)?;

        for _ in 0..50 {
            let (trace, response) = algorithm.clone().run(Context::default(), request()).await?;
            let selected = response
                .llm_response
                .as_agg()
                .map(completion_text)
                .unwrap_or_default();
            assert!(
                names.contains(&selected.as_str()),
                "selected {selected} not in target set"
            );
            assert_eq!(trace[0].selected_model(), selected.as_str());
        }
        Ok(())
    }

    #[tokio::test]
    async fn selection_covers_all_targets_over_many_runs() -> Result<()> {
        let algorithm = shared_algorithm(&["a/model", "b/model"])?;
        let mut seen = HashSet::new();

        for _ in 0..100 {
            let (_, response) = algorithm.clone().run(Context::default(), request()).await?;
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
    async fn affinity_reuses_the_initial_random_selection() -> Result<()> {
        let names = ["a/model", "b/model"];
        let affinity = Arc::new(AffinityRouter::new());
        let random = Arc::new(RandomClassifier::new(
            names.iter().map(|name| (*name).to_string()).collect(),
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

        let (_, first) = algorithm
            .clone()
            .run(Context::default(), request_for_session("session-1"))
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
            .argmax(false)?;
        assert_eq!(
            retained.map(|score| score.target),
            Some(selected.to_string())
        );

        let (_, second) = algorithm
            .run(Context::default(), request_for_session("session-1"))
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
    async fn process_signals_is_a_noop() -> Result<()> {
        let algorithm: Arc<dyn Algorithm> = Arc::new(algorithm(&["only/model"], None, None)?);
        algorithm.process_signals(Signals {}).await?;
        Ok(())
    }

    #[tokio::test]
    async fn decision_is_inspectable_and_downcasts() -> Result<()> {
        let algorithm = shared_algorithm(&["only/model"])?;
        let (trace, _) = algorithm.run(Context::default(), request()).await?;
        let decision = &trace[0];

        assert_eq!(decision.selected_model(), "only/model");
        assert!(decision
            .reasoning()
            .unwrap_or_default()
            .contains("only/model"));
        let concrete = decision
            .as_any()
            .downcast_ref::<RandomDecision>()
            .ok_or_else(|| {
                LibsyError::from(crate::DriverError::TypeMismatch {
                    expected: "RandomDecision",
                })
            })?;
        assert_eq!(concrete.selected_model, "only/model");
        Ok(())
    }
}
