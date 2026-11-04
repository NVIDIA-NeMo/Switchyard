// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::core::algorithm::{AnswerSummary, Driver};
use crate::core::state::State;
use crate::{LibsyError, Result};
use async_trait::async_trait;
use switchyard_protocol::{Category, ModelId, Request, Response};

/// One classifier's recommendation for a request, with a `[0.0, 1.0]` confidence.
///
/// A score either asks for capabilities, which a pool optimizer resolves to a
/// concrete [`ModelId`] later, or names a model outright in `pin`. The two are
/// different things and are kept apart: a capability tag is a group a deployment
/// defines, so a tag and a model may share a name without meaning the same model.
#[derive(Debug, Clone, PartialEq)]
pub struct Score {
    /// `[0.0, 1.0]` confidence in this recommendation.
    pub confidence: f64,
    /// The capability tags that would satisfy this score. Any one of them is
    /// enough (OR, not AND) — an empty list abstains from naming a need.
    pub needs: Vec<Category>,
    /// A model this score names outright, set by algorithms that have already
    /// resolved one: an affinity replay, a random split, a sub-agent's worker.
    /// It takes precedence over `needs` and bypasses capability resolution.
    pub pin: Option<ModelId>,
}

impl Score {
    /// A score asking for `needs`, for the pool to resolve.
    #[must_use]
    pub fn needing(needs: Vec<Category>, confidence: f64) -> Self {
        Self {
            confidence,
            needs,
            pin: None,
        }
    }

    /// A score naming `target` outright, for an algorithm that already resolved it.
    #[must_use]
    pub fn pinned(target: ModelId, confidence: f64) -> Self {
        Self {
            confidence,
            needs: Vec::new(),
            pin: Some(target),
        }
    }
}

/// A classifier's verdict for a request: a set of target [`Score`]s, flagged by how
/// confident the classifier is that they are decisive.
pub enum Classification {
    /// Definite recommendations; [`argmax`](Self::argmax) always yields the top target.
    Scores(Vec<Score>),
    /// Recommendations the classifier considers ambiguous; [`argmax`](Self::argmax) yields
    /// nothing unless the caller opts to ignore ambiguity.
    Ambiguous(Vec<Score>),
}

impl Classification {
    /// The top-scoring [`Score`], or `None` when the classifier abstained (an empty set).
    ///
    /// An [`Ambiguous`](Self::Ambiguous) classification also yields `None` unless
    /// `ignore_ambiguous` is set, in which case it falls back to the plain argmax.
    /// Errors if any confidence is `NaN` (an unorderable score the caller should surface).
    pub fn argmax(&self, ignore_ambiguous: bool) -> Result<Option<Score>> {
        match self {
            Classification::Scores(scores) => argmax(scores),
            Classification::Ambiguous(scores) => {
                if ignore_ambiguous {
                    argmax(scores)
                } else {
                    Ok(None)
                }
            }
        }
    }
}

/// The highest-confidence score, or `None` when the set is empty (the classifier abstained).
/// Ties keep the first. Errors on a `NaN` confidence,
/// which has no defined ordering.
fn argmax(scores: &[Score]) -> Result<Option<Score>> {
    let mut best: Option<&Score> = None;
    for score in scores.iter() {
        if score.confidence.is_nan() {
            return Err(LibsyError::AlgorithmError {
                message: format!(
                    "classifier returned NaN confidence for needs {:?}",
                    score.needs
                ),
            });
        }
        match best {
            Some(cur_best) if score.confidence > cur_best.confidence => best = Some(score),
            None => best = Some(score),
            _ => {}
        }
    }
    Ok(best.cloned())
}

/// Scores targets from the current request and the composition's state.
#[async_trait]
pub trait Classifier: Send + Sync {
    /// Score the classifier's targets given the current state and request.
    ///
    /// `driver` lets a classifier inspect runtime models and offload model calls.
    ///
    /// `request` is borrowed mutably so a classifier may rewrite it in place — inject a
    /// system prompt, drop tools, compact history. The edit is not scoped to this call:
    /// later classifiers in the cascade score the rewritten request, and it is the
    /// rewritten request that is finally sent to the selected model. Most classifiers
    /// only read it.
    async fn score(
        &self,
        state: &mut State,
        request: &mut Request,
        driver: &Driver,
    ) -> Result<(Classification, Option<Response>)>;

    /// Whether a stateful composition should request served-answer reports.
    fn tracks_answers(&self) -> bool {
        false
    }

    /// Folds the answer the host served for this session into the composition's state.
    fn observe_answer(&self, _state: &mut State, _answer: AnswerSummary) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::testing::empty_driver;
    use switchyard_protocol::{ModelId, text_request};

    /// Terse `Score` builder for the assertions below.
    fn score(need: &str, confidence: f64) -> Score {
        Score {
            needs: vec![Category::Named(need.into())],
            pin: None,
            confidence,
        }
    }

    #[test]
    fn argmax_picks_the_highest_confidence_score() -> Result<()> {
        let scores = vec![score("weak", 0.2), score("strong", 0.9), score("mid", 0.5)];
        let best = Classification::Scores(scores).argmax(false)?;
        assert_eq!(best, Some(score("strong", 0.9)));
        Ok(())
    }

    #[test]
    fn argmax_breaks_ties_by_cascade_order() -> Result<()> {
        // Equal confidence: the earlier target in cascade order wins the tie.
        let scores = vec![score("first", 0.7), score("second", 0.7)];
        let best = Classification::Scores(scores).argmax(false)?;
        assert_eq!(best, Some(score("first", 0.7)));
        Ok(())
    }

    #[test]
    fn argmax_on_an_empty_set_abstains() -> Result<()> {
        // No scores means the classifier abstained — no choice to make.
        assert_eq!(Classification::Scores(vec![]).argmax(false)?, None);
        assert_eq!(Classification::Ambiguous(vec![]).argmax(true)?, None);
        Ok(())
    }

    #[test]
    fn argmax_errors_on_nan_confidence() {
        // A NaN confidence has no defined ordering — surface it rather than guess.
        let scores = vec![score("weak", 0.3), score("strong", f64::NAN)];
        assert!(matches!(
            Classification::Scores(scores).argmax(false),
            Err(LibsyError::AlgorithmError { message })
                if message == "classifier returned NaN confidence for needs [Named(\"strong\")]"
        ));
        // A lone NaN errors too, even with nothing to compare it against.
        assert!(matches!(
            Classification::Scores(vec![score("only", f64::NAN)]).argmax(false),
            Err(LibsyError::AlgorithmError { message })
                if message == "classifier returned NaN confidence for needs [Named(\"only\")]"
        ));
    }

    #[test]
    fn ambiguous_without_ignore_makes_no_choice() -> Result<()> {
        // Ambiguous means "don't pick" unless the caller opts to ignore ambiguity.
        let scores = vec![score("strong", 0.9)];
        assert_eq!(Classification::Ambiguous(scores).argmax(false)?, None);
        Ok(())
    }

    #[test]
    fn ambiguous_with_ignore_falls_back_to_argmax() -> Result<()> {
        let scores = vec![score("weak", 0.3), score("strong", 0.8)];
        let best = Classification::Ambiguous(scores).argmax(true)?;
        assert_eq!(best, Some(score("strong", 0.8)));
        Ok(())
    }

    #[test]
    fn scores_variant_ignores_the_ambiguous_flag() -> Result<()> {
        // A definitive classification always yields its argmax, regardless of the flag.
        let scores = vec![score("a", 0.4), score("b", 0.6)];
        let with_ignore = Classification::Scores(scores.clone()).argmax(true)?;
        let without_ignore = Classification::Scores(scores).argmax(false)?;
        assert_eq!(with_ignore, without_ignore);
        assert_eq!(with_ignore, Some(score("b", 0.6)));
        Ok(())
    }

    /// Scores the request's requested model at full confidence and records that it ran.
    struct RecordingClassifier;

    #[async_trait]
    impl Classifier for RecordingClassifier {
        async fn score(
            &self,
            state: &mut State,
            request: &mut Request,
            _driver: &Driver,
        ) -> Result<(Classification, Option<Response>)> {
            state
                .extra
                .insert("ran".to_string(), crate::core::state::StateValue::Count(1));
            let need = request.model_id().unwrap_or(ModelId::from("auto"));
            Ok((
                Classification::Scores(vec![Score {
                    needs: vec![Category::Named(need.as_str().into())],
                    pin: None,
                    confidence: 1.0,
                }]),
                None,
            ))
        }
    }

    #[tokio::test]
    async fn classifier_reads_request_and_mutates_state() -> Result<()> {
        let mut state = State::default();
        let mut request = Request {
            llm_request: text_request(Some("strong".to_string()), "hi"),
            raw_request: None,
            metadata: None,
        };
        let (classification, _) = RecordingClassifier
            .score(&mut state, &mut request, &empty_driver())
            .await?;
        assert_eq!(
            classification.argmax(false)?.map(|s| s.needs),
            Some(vec![Category::Named("strong".into())])
        );
        assert_eq!(
            state.extra.get("ran"),
            Some(&crate::core::state::StateValue::Count(1))
        );
        Ok(())
    }

    /// Rewrites the request's model, then scores the rewritten value.
    struct RewritingClassifier;

    #[async_trait]
    impl Classifier for RewritingClassifier {
        async fn score(
            &self,
            _state: &mut State,
            request: &mut Request,
            _driver: &Driver,
        ) -> Result<(Classification, Option<Response>)> {
            request.llm_request.model = Some("rewritten".to_string());
            Ok((
                Classification::Scores(vec![Score {
                    needs: vec![Category::Named("rewritten".into())],
                    pin: None,
                    confidence: 1.0,
                }]),
                None,
            ))
        }
    }

    #[tokio::test]
    async fn classifier_rewrites_the_request_in_place() -> Result<()> {
        let mut state = State::default();
        let mut request = Request {
            llm_request: text_request(Some("auto".to_string()), "hi"),
            raw_request: None,
            metadata: None,
        };

        RewritingClassifier
            .score(&mut state, &mut request, &empty_driver())
            .await?;

        // The rewrite outlives the call: later classifiers in the cascade score this value,
        // and it is what reaches the model.
        assert_eq!(request.model_id().as_deref(), Some("rewritten"));
        Ok(())
    }
}
