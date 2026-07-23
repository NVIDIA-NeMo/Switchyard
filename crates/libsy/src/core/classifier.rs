// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{Driver, State};
use async_trait::async_trait;
use switchyard_protocol::Request;

/// One classifier's recommendation of a routing `target`, with a `[0.0, 1.0]` confidence.
#[derive(Debug, Clone, PartialEq)]
pub struct Score {
    /// `[0.0, 1.0]` confidence in `target`.
    pub confidence: f64,
    /// The target (model / tier) being recommended.
    pub target: String,
}

type BoxErr = Box<dyn std::error::Error + Send + Sync>;

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
    pub fn argmax(&self, ignore_ambiguous: bool) -> Result<Option<Score>, BoxErr> {
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
fn argmax(scores: &[Score]) -> Result<Option<Score>, BoxErr> {
    let mut best: Option<&Score> = None;
    for score in scores.iter() {
        if score.confidence.is_nan() {
            return Err(format!("argmax: NaN confidence for target '{}'", score.target).into());
        }
        match best {
            Some(cur_best) if score.confidence > cur_best.confidence => best = Some(score),
            None => best = Some(score),
            _ => {}
        }
    }
    Ok(best.cloned())
}

/// Scores each of the classifier's targets given State the current Request
#[async_trait]
pub trait Classifier: Send + Sync {
    /// Score the classifier's targets given the current state and request.
    /// driver is optional. It is used to offload model calls
    async fn score(
        &self,
        state: &mut State,
        request: &Request,
        driver: Option<&Driver>,
    ) -> Result<Classification, BoxErr>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StateValue;
    use switchyard_protocol::text_request;

    /// Terse `Score` builder for the assertions below.
    fn score(target: &str, confidence: f64) -> Score {
        Score {
            target: target.to_string(),
            confidence,
        }
    }

    #[test]
    fn argmax_picks_the_highest_confidence_score() -> Result<(), BoxErr> {
        let scores = vec![score("weak", 0.2), score("strong", 0.9), score("mid", 0.5)];
        let best = Classification::Scores(scores).argmax(false)?;
        assert_eq!(best, Some(score("strong", 0.9)));
        Ok(())
    }

    #[test]
    fn argmax_breaks_ties_by_cascade_order() -> Result<(), BoxErr> {
        // Equal confidence: the earlier target in cascade order wins the tie.
        let scores = vec![score("first", 0.7), score("second", 0.7)];
        let best = Classification::Scores(scores).argmax(false)?;
        assert_eq!(best.map(|s| s.target), Some("first".to_string()));
        Ok(())
    }

    #[test]
    fn argmax_on_an_empty_set_abstains() -> Result<(), BoxErr> {
        // No scores means the classifier abstained — no choice to make.
        assert_eq!(Classification::Scores(vec![]).argmax(false)?, None);
        assert_eq!(Classification::Ambiguous(vec![]).argmax(true)?, None);
        Ok(())
    }

    #[test]
    fn argmax_errors_on_nan_confidence() {
        // A NaN confidence has no defined ordering — surface it rather than guess.
        let scores = vec![score("weak", 0.3), score("strong", f64::NAN)];
        assert!(Classification::Scores(scores).argmax(false).is_err());
        // A lone NaN errors too, even with nothing to compare it against.
        assert!(Classification::Scores(vec![score("only", f64::NAN)])
            .argmax(false)
            .is_err());
    }

    #[test]
    fn ambiguous_without_ignore_makes_no_choice() -> Result<(), BoxErr> {
        // Ambiguous means "don't pick" unless the caller opts to ignore ambiguity.
        let scores = vec![score("strong", 0.9)];
        assert_eq!(Classification::Ambiguous(scores).argmax(false)?, None);
        Ok(())
    }

    #[test]
    fn ambiguous_with_ignore_falls_back_to_argmax() -> Result<(), BoxErr> {
        let scores = vec![score("weak", 0.3), score("strong", 0.8)];
        let best = Classification::Ambiguous(scores).argmax(true)?;
        assert_eq!(best, Some(score("strong", 0.8)));
        Ok(())
    }

    #[test]
    fn scores_variant_ignores_the_ambiguous_flag() -> Result<(), BoxErr> {
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
            request: &Request,
            _driver: Option<&Driver>,
        ) -> Result<Classification, BoxErr> {
            // Stash a marker in `extra` to prove state is threaded mutably.
            state.extra.insert("ran".to_string(), StateValue::Count(1));
            let target = request.requested_model().unwrap_or("auto").to_string();
            Ok(Classification::Scores(vec![Score {
                target,
                confidence: 1.0,
            }]))
        }
    }

    #[tokio::test]
    async fn classifier_reads_request_and_mutates_state() -> Result<(), BoxErr> {
        let mut state = State::default();
        let request = Request {
            llm_request: text_request(Some("strong".to_string()), "hi"),
            raw_request: None,
            metadata: None,
        };
        // A `None` driver is valid: the classifier scored without offloading a model call.
        let classification = RecordingClassifier
            .score(&mut state, &request, None)
            .await?;
        assert_eq!(
            classification.argmax(false)?.map(|s| s.target),
            Some("strong".to_string())
        );
        assert!(matches!(state.extra.get("ran"), Some(StateValue::Count(1))));
        Ok(())
    }
}
