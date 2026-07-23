// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Random router built on the [`Algorithm`] interfaces.
//!
//! Selects one target from the set uniformly at random and calls it. This is the
//! simplest possible routing algorithm and the reference for the single-call
//! shape: one `driver.call_llm_target` inside `create_run_task`. Weighted selection
//! can be layered on later; the set defines the candidates.

use std::sync::Arc;

use async_trait::async_trait;
use rand::seq::SliceRandom;

use crate::{
    Algorithm, Context, Decision, Driver, LibsyError, LlmTargetSet, Request, Response, Result,
};

/// Decision produced by [`Random`]: which target was chosen and why.
pub struct RandomDecision {
    /// The randomly selected target/model.
    pub selected_model: String,
    /// Human-readable explanation of the choice.
    pub reasoning: String,
}

impl Decision for RandomDecision {
    fn selected_model(&self) -> &str {
        &self.selected_model
    }

    fn reasoning(&self) -> Option<&str> {
        Some(&self.reasoning)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Uniform random router over a target set.
pub struct Random {
    target_set: LlmTargetSet,
}

impl Random {
    /// Creates a router over `target_set`.
    ///
    /// Wrap it in an [`Arc`] and drive it with [`Algorithm::run`] or
    /// [`Algorithm::run_stream`].
    pub fn new(target_set: LlmTargetSet) -> Self {
        Self { target_set }
    }
}

#[async_trait]
impl Algorithm for Random {
    fn name(&self) -> &str {
        "random"
    }

    async fn create_run_task(
        self: Arc<Self>,
        ctx: Context,
        driver: Driver,
        request: Request,
    ) -> Result<Response> {
        // Scope the non-Send ThreadRng before the await so the future remains Send.
        let target = {
            let mut rng = rand::thread_rng();
            self.target_set
                .targets()
                .choose(&mut rng)
                .ok_or(LibsyError::NoTargets)?
                .clone()
        };

        let selected = target.semantic_name.clone();
        let decision: Arc<dyn Decision> = Arc::new(RandomDecision {
            reasoning: format!("random routing selected target '{selected}'"),
            selected_model: selected,
        });

        driver.info(ctx.clone(), Arc::clone(&decision)).await?;
        driver
            .call_llm_target(ctx, &target, request, decision)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    use switchyard_protocol::{completion_text, text_request, text_response};

    use crate::{LlmResponse, LlmTarget, Request, RoutedLlmClient, Signals};

    /// Echoes the selected target so tests can inspect which target was called.
    struct EchoClient;

    #[async_trait]
    impl RoutedLlmClient for EchoClient {
        async fn call(
            &self,
            _ctx: Context,
            _request: Request,
            decision: Arc<dyn Decision>,
        ) -> std::result::Result<Response, Box<dyn std::error::Error + Send + Sync>> {
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

    /// Builds a random router whose targets all share an echo client.
    fn algorithm(names: &[&str]) -> Random {
        let targets = names
            .iter()
            .map(|name| LlmTarget {
                semantic_name: (*name).to_string(),
                llm_client: Some(Arc::new(EchoClient)),
            })
            .collect();
        Random::new(LlmTargetSet::new(targets))
    }

    fn shared_algorithm(names: &[&str]) -> Arc<dyn Algorithm> {
        Arc::new(algorithm(names))
    }

    #[tokio::test]
    async fn single_target_is_always_selected_and_called() -> Result<()> {
        let algorithm = shared_algorithm(&["only/model"]);
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
        let algorithm = shared_algorithm(&names);

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
        let algorithm = shared_algorithm(&["a/model", "b/model"]);
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
    async fn empty_target_set_errors() {
        let algorithm = shared_algorithm(&[]);
        let error = algorithm.run(Context::default(), request()).await.err();
        assert!(matches!(error, Some(LibsyError::NoTargets)));
    }

    #[tokio::test]
    async fn process_signals_is_a_noop() -> Result<()> {
        Arc::new(algorithm(&["only/model"]))
            .process_signals(Signals {})
            .await?;
        Ok(())
    }

    #[tokio::test]
    async fn decision_is_inspectable_and_downcasts() -> Result<()> {
        let algorithm = shared_algorithm(&["only/model"]);
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
