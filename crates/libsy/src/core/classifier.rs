// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::core::algorithm::Driver;
use crate::{LibsyError, Result};
use async_trait::async_trait;
use switchyard_protocol::{Request, Response};

/// One classifier's recommendation of a routing `target`, with a `[0.0, 1.0]` confidence.
#[derive(Debug, Clone, PartialEq)]
pub struct Score {
    /// `[0.0, 1.0]` confidence in `target`.
    pub confidence: f64,
    /// The target (model / tier) being recommended.
    pub target: String,
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
                    "classifier returned NaN confidence for target {:?}",
                    score.target
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
pub trait Classifier<S = ()>: Send + Sync {
    /// Stable tier represented by `selected_model`, when this classifier defines one.
    fn routing_tier(&self, _selected_model: &str) -> Option<&'static str> {
        None
    }

    /// Drops retained routing state when `target` was unavailable for `request`.
    ///
    /// Stateless classifiers do not need to implement this hook.
    fn target_unavailable(&self, _request: &Request, _target: &str) {}

    /// Score the classifier's targets given the current state and request.
    ///
    /// When present, `driver` lets a classifier offload model calls. It is `None`
    /// when the classifier is evaluated outside an algorithm run.
    ///
    /// `request` is borrowed mutably so a classifier may rewrite it in place — inject a
    /// system prompt, drop tools, compact history. The edit is not scoped to this call:
    /// later classifiers in the cascade score the rewritten request, and it is the
    /// rewritten request that is finally sent to the selected model. Most classifiers
    /// only read it.
    async fn score(
        &self,
        state: &mut S,
        request: &mut Request,
        driver: Option<&Driver>,
    ) -> Result<(Classification, Option<Response>)>;
}

#[cfg(test)]
#[path = "classifier_tests.rs"]
mod tests;
