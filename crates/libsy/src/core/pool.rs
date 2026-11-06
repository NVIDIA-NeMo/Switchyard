// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The pool optimizer: resolves signals' [`Score`]s to one concrete [`ModelId`],
//! so an algorithm's signals stay in terms of what a request needs and never have to
//! carry the model catalog themselves.

use switchyard_protocol::{Category, ModelId};

use crate::core::classifier::{Classification, Score};
use crate::{LibsyError, Result};

/// One model in the pool and the capability tags it provides.
#[derive(Debug, Clone)]
pub struct PoolModel {
    /// The model this entry resolves to.
    pub target: ModelId,
    /// Capability tags this model satisfies. A need is satisfied if any of
    /// these tags appear in it (OR, not AND).
    pub provides: Vec<Category>,
}

/// A deployment's model catalog, shared by every `auto` route.
#[derive(Debug, Clone)]
pub struct PoolContext {
    /// Models available to this pool, in declared order. Order is the
    /// tie-break among candidates that equally satisfy the winning need.
    pub models: Vec<PoolModel>,
    /// Capability to fall back to when no score names a need.
    pub default: Category,
}

/// Resolves signal scores to one [`ModelId`].
///
/// Takes the highest-confidence [`Score`] (ties keep the first, same as
/// [`Classification::argmax`]); falls back to `pool.default` when the classifiers
/// abstained or the winner named no need. Then picks the first pool model, in
/// declared order, that provides one of the winning needs. Errors if nothing in
/// the pool satisfies it.
pub fn select(scores: &[Score], pool: &PoolContext) -> Result<ModelId> {
    select_with_history(scores, pool, &[])
}

/// Prefers a previously selected candidate only among models satisfying the winning need.
pub fn select_with_history(
    scores: &[Score],
    pool: &PoolContext,
    used: &[ModelId],
) -> Result<ModelId> {
    let winner = Classification::Scores(scores.to_vec()).argmax(false)?;
    // A pinned score names its model outright; the pool resolves capabilities only.
    if let Some(score) = &winner
        && let Some(target) = &score.pin
    {
        return pool
            .models
            .iter()
            .find(|model| &model.target == target)
            .map(|model| model.target.clone())
            .ok_or_else(|| LibsyError::TargetNotFound {
                target: target.clone(),
            });
    }
    let needs = match winner {
        Some(score) if !score.needs.is_empty() => score.needs,
        _ => vec![pool.default.clone()],
    };

    let satisfies = |model: &&PoolModel| model.provides.iter().any(|tag| needs.contains(tag));
    pool.models
        .iter()
        .filter(satisfies)
        .find(|model| used.contains(&model.target))
        .or_else(|| pool.models.iter().find(satisfies))
        .map(|model| model.target.clone())
        .ok_or_else(|| LibsyError::AlgorithmError {
            message: format!("no pool model provides any of {needs:?}"),
        })
}

/// Interprets a judge's efficient-model solve estimate at the selection boundary.
/// A rejected efficient estimate asks for capable with its complementary probability.
pub(crate) fn probability_need(mut score: Score, threshold: f64) -> Score {
    if score.confidence < threshold && (threshold - score.confidence).abs() > f64::EPSILON {
        score.needs = vec![Category::Capable];
        score.confidence = 1.0 - score.confidence;
    }
    score
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(models: &[(&str, &[Category])], default: Category) -> PoolContext {
        PoolContext {
            models: models
                .iter()
                .map(|(target, provides)| PoolModel {
                    target: ModelId::from(*target),
                    provides: provides.to_vec(),
                })
                .collect(),
            default,
        }
    }

    fn score(needs: &[Category], confidence: f64) -> Score {
        Score {
            needs: needs.to_vec(),
            pin: None,
            confidence,
        }
    }

    #[test]
    fn pins_remain_literal_and_must_belong_to_the_pool() -> Result<()> {
        let pool = pool(
            &[
                ("chosen", &[Category::Efficient]),
                ("other", &[Category::Named("chosen".into())]),
            ],
            Category::Efficient,
        );
        assert_eq!(
            select(&[Score::pinned("chosen".into(), 1.0)], &pool)?,
            ModelId::from("chosen")
        );
        assert!(select(&[Score::pinned("missing".into(), 1.0)], &pool).is_err());
        Ok(())
    }

    #[test]
    fn warm_candidates_never_override_the_required_capability() -> Result<()> {
        let pool = pool(
            &[
                ("first", &[Category::Capable]),
                ("warm", &[Category::Capable]),
                ("cheap", &[Category::Efficient]),
            ],
            Category::Efficient,
        );
        let scores = [score(&[Category::Capable], 0.8)];
        assert_eq!(
            select_with_history(&scores, &pool, &["warm".into()])?,
            ModelId::from("warm")
        );
        assert_eq!(
            select_with_history(&scores, &pool, &["cheap".into()])?,
            ModelId::from("first")
        );
        assert_eq!(
            select_with_history(&scores, &pool, &["removed".into()])?,
            ModelId::from("first")
        );
        Ok(())
    }

    #[test]
    fn probability_threshold_is_interpreted_at_selection() {
        let raw = score(&[Category::Efficient], 0.83);
        let capable = probability_need(raw.clone(), 0.9);
        assert_eq!(capable.needs, [Category::Capable]);
        assert!((capable.confidence - 0.17).abs() < 1e-9);
        assert_eq!(probability_need(raw, 0.8).needs, [Category::Efficient]);
    }

    #[test]
    fn selects_the_model_providing_the_winning_need() -> Result<()> {
        let pool = pool(
            &[
                ("sol", &[Category::Capable]),
                ("luna", &[Category::Efficient]),
            ],
            Category::Capable,
        );
        let scores = vec![score(&[Category::Efficient], 0.9)];
        assert_eq!(select(&scores, &pool)?, ModelId::from("luna"));
        Ok(())
    }

    #[test]
    fn or_semantics_match_any_provided_tag_first_in_declared_order() -> Result<()> {
        let pool = pool(
            &[
                ("sol", &[Category::Capable]),
                ("terra", &[Category::Capable, Category::Judge]),
            ],
            Category::Capable,
        );
        let scores = vec![score(&[Category::Judge], 1.0)];
        assert_eq!(select(&scores, &pool)?, ModelId::from("terra"));
        Ok(())
    }

    #[test]
    fn falls_back_to_the_pool_default_when_scores_are_empty() -> Result<()> {
        let pool = pool(
            &[
                ("sol", &[Category::Capable]),
                ("luna", &[Category::Efficient]),
            ],
            Category::Capable,
        );
        assert_eq!(select(&[], &pool)?, ModelId::from("sol"));
        Ok(())
    }

    #[test]
    fn falls_back_to_the_pool_default_when_the_winner_names_no_need() -> Result<()> {
        let pool = pool(&[("sol", &[Category::Capable])], Category::Capable);
        let scores = vec![score(&[], 0.5)];
        assert_eq!(select(&scores, &pool)?, ModelId::from("sol"));
        Ok(())
    }

    #[test]
    fn higher_confidence_need_wins_among_multiple_signals() -> Result<()> {
        let pool = pool(
            &[
                ("sol", &[Category::Capable]),
                ("luna", &[Category::Efficient]),
            ],
            Category::Capable,
        );
        let scores = vec![
            score(&[Category::Efficient], 0.4),
            score(&[Category::Capable], 0.8),
        ];
        assert_eq!(select(&scores, &pool)?, ModelId::from("sol"));
        Ok(())
    }

    #[test]
    fn nan_confidence_propagates_argmaxs_error() {
        let pool = pool(&[("sol", &[Category::Capable])], Category::Capable);
        let scores = vec![score(&[Category::Capable], f64::NAN)];
        assert!(matches!(
            select(&scores, &pool),
            Err(LibsyError::AlgorithmError { .. })
        ));
    }

    #[test]
    fn errors_when_no_pool_model_provides_the_winning_need() {
        let pool = pool(&[("sol", &[Category::Capable])], Category::Capable);
        let scores = vec![score(&[Category::Efficient], 0.9)];
        assert!(matches!(
            select(&scores, &pool),
            Err(LibsyError::AlgorithmError { message })
                if message == "no pool model provides any of [Efficient]"
        ));
    }
}
