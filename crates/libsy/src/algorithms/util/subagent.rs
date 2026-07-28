// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sub-agent override as a single SDK component.
//!
//! [`SubagentOverride`] scores one fixed worker target for requests carrying delegated
//! sub-agent work ([`Metadata::is_subagent_work`]) and abstains for everything else, so a
//! cascade falls through to its later classifiers on ordinary traffic.
//!
//! It is stateless and holds only the worker's *name*: the
//! [`FallThrough`](crate::algorithms::FallThrough) cascade resolves that name against its
//! target set. Keeping the policy independent of any memory of past decisions is what lets
//! it compose with a stateful classifier such as
//! [`AffinityRouter`](crate::algorithms::AffinityRouter) — the override decides *which*
//! target delegated work belongs on, affinity decides *how long* a decision lives, and
//! neither needs to know about the other.

use async_trait::async_trait;

use crate::{Classification, Classifier, Driver, Metadata, Request, Result, Score, State};

/// Scores a fixed worker target for delegated sub-agent work; abstains otherwise.
pub struct SubagentOverride {
    /// Name of the worker target, resolved by the cascade against its target set.
    worker: String,
}

impl SubagentOverride {
    /// Creates an override scoring `worker` for delegated sub-agent work.
    ///
    /// `worker` must name a target in the cascade's set, or routing a sub-agent request
    /// fails with [`LibsyError::TargetNotFound`](crate::LibsyError::TargetNotFound).
    pub fn new(worker: impl Into<String>) -> Self {
        Self {
            worker: worker.into(),
        }
    }
}

#[async_trait]
impl Classifier for SubagentOverride {
    async fn score(
        &self,
        _state: &mut State,
        request: &mut Request,
        _driver: Option<&Driver>,
    ) -> Result<Classification> {
        // Delegated *work* only. A harness maintenance turn (e.g. Codex `compact`) carries
        // sub-agent lineage but is not delegated work, so it abstains and routes normally.
        let is_delegated_work = request
            .metadata
            .as_ref()
            .is_some_and(Metadata::is_subagent_work);
        Ok(Classification::Scores(if is_delegated_work {
            vec![Score {
                confidence: 1.0,
                target: self.worker.clone(),
            }]
        } else {
            Vec::new()
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use switchyard_protocol::text_request;

    fn request(headers: &[(&str, &str)]) -> Request {
        let metadata = (!headers.is_empty()).then(|| {
            Metadata::from_headers(
                &headers
                    .iter()
                    .map(|(name, value)| ((*name).to_string(), (*value).to_string()))
                    .collect::<BTreeMap<_, _>>(),
            )
        });
        Request {
            llm_request: text_request(Some("auto".to_string()), "hi"),
            raw_request: None,
            metadata,
        }
    }

    /// Scores `headers` through the override, returning the winning target if it scored.
    async fn selected(headers: &[(&str, &str)]) -> Result<Option<String>> {
        let mut state = State::default();
        let classification = SubagentOverride::new("worker")
            .score(&mut state, &mut request(headers), None)
            .await?;
        Ok(classification.argmax(false)?.map(|score| score.target))
    }

    #[tokio::test]
    async fn requests_without_metadata_abstain() -> Result<()> {
        assert_eq!(selected(&[]).await?, None);
        Ok(())
    }

    #[tokio::test]
    async fn subagent_work_scores_the_worker() -> Result<()> {
        // Claude Code child-agent lineage.
        let claude = &[
            ("x-claude-code-session-id", "root"),
            ("x-claude-code-agent-id", "child-1"),
        ];
        assert_eq!(selected(claude).await?, Some("worker".to_string()));

        // Codex delegated-work kinds.
        assert_eq!(
            selected(&[("x-openai-subagent", "review")]).await?,
            Some("worker".to_string())
        );
        assert_eq!(
            selected(&[("x-openai-subagent", "collab_spawn")]).await?,
            Some("worker".to_string())
        );
        Ok(())
    }

    #[tokio::test]
    async fn harness_maintenance_turns_abstain() -> Result<()> {
        assert_eq!(selected(&[("x-openai-subagent", "compact")]).await?, None);
        assert_eq!(
            selected(&[("x-switchyard-is-subagent", "false")]).await?,
            None
        );
        Ok(())
    }

    #[tokio::test]
    async fn delegated_work_is_scored_definitively() -> Result<()> {
        // Confidence 1.0 under `Scores` (never `Ambiguous`), so the cascade stops here
        // rather than consulting later classifiers.
        let mut state = State::default();
        let classification = SubagentOverride::new("worker")
            .score(
                &mut state,
                &mut request(&[("x-openai-subagent", "review")]),
                None,
            )
            .await?;
        match classification {
            Classification::Scores(scores) => {
                assert_eq!(scores.len(), 1);
                assert_eq!(scores[0].confidence, 1.0);
            }
            Classification::Ambiguous(_) => panic!("override must score definitively"),
        }
        Ok(())
    }
}
