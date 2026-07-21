// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Uniform random selection as a composable [`Classifier`].

use async_trait::async_trait;
use rand::seq::SliceRandom;
use switchyard_protocol::Request;

use crate::{Classification, Classifier, Score, State};

/// Boxed, thread-safe error type used across the SDK.
type BoxErr = Box<dyn std::error::Error + Send + Sync>;

/// Selects one configured routing target uniformly at random.
///
/// Place this classifier after classifiers that may abstain, such as
/// [`crate::algorithms::AffinityRouter`], to provide a guaranteed local fallback.
pub struct RandomClassifier {
    targets: Vec<String>,
}

impl RandomClassifier {
    /// Creates a random classifier over the provided target names.
    pub fn new(targets: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            targets: targets.into_iter().map(Into::into).collect(),
        }
    }

    /// Chooses one target, avoiding random-number generation for a singleton set.
    fn choose_target(&self) -> Result<String, BoxErr> {
        match self.targets.as_slice() {
            [] => Err("random classifier: no targets available".into()),
            [target] => Ok(target.clone()),
            targets => targets
                .choose(&mut rand::thread_rng())
                .cloned()
                .ok_or_else(|| "random classifier: no targets available".into()),
        }
    }
}

#[async_trait]
impl Classifier for RandomClassifier {
    async fn score(
        &self,
        _state: &mut State,
        _request: &Request,
        _driver: Option<&crate::Driver>,
    ) -> Result<Classification, BoxErr> {
        Ok(Classification::Scores(vec![Score {
            confidence: 1.0,
            target: self.choose_target()?,
        }]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use switchyard_protocol::{text_request, Request};

    fn request() -> Request {
        Request {
            llm_request: text_request(Some("auto".to_string()), "hi"),
            raw_request: None,
            metadata: None,
        }
    }

    #[tokio::test]
    async fn singleton_target_is_selected() -> Result<(), BoxErr> {
        let classifier = RandomClassifier::new(["only"]);
        let classification = classifier
            .score(&mut State::default(), &request(), None)
            .await?;
        let scores = match classification {
            Classification::Scores(scores) => scores,
            Classification::Ambiguous(_) => {
                return Err("random classifier returned ambiguity".into())
            }
        };

        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].target, "only");
        assert_eq!(scores[0].confidence, 1.0);
        Ok(())
    }

    #[tokio::test]
    async fn selected_target_belongs_to_the_configured_set() -> Result<(), BoxErr> {
        let targets = ["a", "b", "c"];
        let classifier = RandomClassifier::new(targets);
        let mut state = State::default();

        for _ in 0..50 {
            let classification = classifier.score(&mut state, &request(), None).await?;
            let selected = classification
                .argmax(false)?
                .ok_or("random classifier abstained")?;
            assert!(targets.contains(&selected.target.as_str()));
        }
        Ok(())
    }

    #[tokio::test]
    async fn empty_target_set_errors() {
        let classifier = RandomClassifier::new(Vec::<String>::new());
        assert!(classifier
            .score(&mut State::default(), &request(), None)
            .await
            .is_err());
    }
}
