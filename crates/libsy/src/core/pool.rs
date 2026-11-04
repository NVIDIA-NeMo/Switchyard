// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The pool optimizer: resolves a cascade's [`Score`]s to one concrete [`ModelId`],
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

/// A deployment's model catalog, shared by every `capability_router` route.
#[derive(Debug, Clone)]
pub struct PoolContext {
    /// Models available to this pool, in declared order. Order is the
    /// tie-break among candidates that equally satisfy the winning need.
    pub models: Vec<PoolModel>,
    /// Capability to fall back to when no score names a need.
    pub default: Category,
}

/// Resolves a cascade's scores to one [`ModelId`].
///
/// Takes the highest-confidence [`Score`] (ties keep the first, same as
/// [`Classification::argmax`]); falls back to `pool.default` when the classifiers
/// abstained or the winner named no need. Then picks the first pool model, in
/// declared order, that provides one of the winning needs. Errors if nothing in
/// the pool satisfies it.
pub fn select(scores: &[Score], pool: &PoolContext) -> Result<ModelId> {
    let winner = Classification::Scores(scores.to_vec()).argmax(false)?;
    // A pinned score names its model outright; the pool resolves capabilities only.
    if let Some(score) = &winner
        && let Some(target) = &score.pin
    {
        return Ok(target.clone());
    }
    let needs = match winner {
        Some(score) if !score.needs.is_empty() => score.needs,
        _ => vec![pool.default.clone()],
    };

    pool.models
        .iter()
        .find(|model| model.provides.iter().any(|tag| needs.contains(tag)))
        .map(|model| model.target.clone())
        .ok_or_else(|| LibsyError::AlgorithmError {
            message: format!("no pool model provides any of {needs:?}"),
        })
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
