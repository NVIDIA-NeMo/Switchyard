// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Random router built on the [`Algorithm`] interfaces.
//!
//! Selects one target from the set uniformly at random and calls it. This is the
//! simplest possible routing algorithm and the reference for the single-call
//! shape: one `driver.call_llm_target` inside `create_run_task`. (Weighted selection could
//! be layered on later; the set defines the candidates.)

use std::error::Error;
use std::sync::Arc;

use async_trait::async_trait;
use rand::seq::SliceRandom;

use libsy::{Algorithm, Context, Decision, Driver, LlmTargetSet, Request, Response, Signals};

/// Decision produced by [`RandomOrchAlgo`]: which target was chosen and why.
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
pub struct RandomOrchAlgo {
    target_set: LlmTargetSet,
}

impl RandomOrchAlgo {
    /// Create a router over `target_set`. Wrap it in an
    /// [`Arc`](std::sync::Arc) and drive it with
    /// [`run`](libsy::Algorithm::run) or
    /// [`run_stream`](libsy::Algorithm::run_stream).
    pub fn new(target_set: LlmTargetSet) -> Self {
        Self { target_set }
    }
}

#[async_trait]
impl Algorithm for RandomOrchAlgo {
    async fn create_run_task(
        self: Arc<Self>,
        _ctx: Context,
        driver: Driver,
        request: Request,
    ) -> Result<Response, Box<dyn Error + Send + Sync>> {
        // Select a target uniformly at random. Scope the RNG so the non-Send
        // `ThreadRng` is dropped before the await below, keeping the returned
        // future `Send` (required by the `Algorithm` bound).
        let target = {
            let mut rng = rand::thread_rng();
            self.target_set
                .targets()
                .choose(&mut rng)
                .ok_or("no targets available")?
                .clone()
        };

        // Route by target semantic name; the caller's client (or offload host) maps
        // it to the provider model id when it serves or offloads the call.
        let selected = target.semantic_name.clone();
        let decision: Arc<dyn Decision> = Arc::new(RandomDecision {
            reasoning: format!("random routing selected target '{selected}'"),
            selected_model: selected,
        });

        // Publish the decision to the stream, then offload the call.
        driver.info(decision.clone()).await?;
        driver.call_llm_target(&target, request, decision).await
    }

    async fn process_signals(
        self: Arc<Self>,
        _signals: Signals,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Random routing is stateless, so agent-system signals are ignored.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libsy::{
        LlmClient, LlmRequest, LlmResponse, LlmTarget, Message, Response, Role, RoutedRequest,
    };
    use std::collections::HashSet;

    /// Echoes back the target name it was called with, so a test can tell which
    /// target the algo selected.
    struct EchoClient;

    #[async_trait]
    impl LlmClient for EchoClient {
        async fn call(
            &self,
            routed: RoutedRequest,
        ) -> Result<Response, Box<dyn Error + Send + Sync>> {
            Ok(Response {
                llm_response: LlmResponse {
                    model: Some(routed.decision.selected_model().to_string()),
                    ..LlmResponse::default()
                },
                metadata: None,
            })
        }
    }

    fn request() -> Request {
        Request {
            llm_request: LlmRequest {
                model: Some("auto".to_string()),
                messages: vec![Message::text(Role::User, "hi")],
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: None,
        }
    }

    /// Build a random-routing algorithm over `names`; every target echoes its name.
    fn orch(names: &[&str]) -> Arc<dyn Algorithm> {
        Arc::new(algo(names))
    }

    fn algo(names: &[&str]) -> RandomOrchAlgo {
        let targets: Vec<LlmTarget> = names
            .iter()
            .map(|name| LlmTarget {
                semantic_name: name.to_string(),
                llm_client: Some(Arc::new(EchoClient)),
            })
            .collect();
        RandomOrchAlgo::new(LlmTargetSet::new(targets))
    }

    #[tokio::test]
    async fn single_target_is_always_selected_and_called(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let orch = orch(&["only/model"]);
        let (trace, response) = orch.clone().run(Context::default(), request()).await?;
        assert_eq!(response.llm_response.model.as_deref(), Some("only/model"));
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].selected_model(), "only/model");
        Ok(())
    }

    #[tokio::test]
    async fn selected_target_is_in_the_set_and_matches_the_trace(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let names = ["a/model", "b/model", "c/model"];
        let orch = orch(&names);
        for _ in 0..50 {
            let (trace, response) = orch.clone().run(Context::default(), request()).await?;
            let selected = response.llm_response.model.unwrap_or_default();
            assert!(
                names.contains(&selected.as_str()),
                "selected {selected} not in target set"
            );
            // The trace records the same target that was actually called.
            assert_eq!(trace[0].selected_model(), selected.as_str());
        }
        Ok(())
    }

    #[tokio::test]
    async fn selection_covers_all_targets_over_many_runs(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let orch = orch(&["a/model", "b/model"]);
        let mut seen = HashSet::new();
        for _ in 0..100 {
            let (_, response) = orch.clone().run(Context::default(), request()).await?;
            seen.insert(response.llm_response.model.unwrap_or_default());
        }
        // 100 uniform draws over two targets: both should appear (miss ~ 2^-99).
        assert_eq!(
            seen.len(),
            2,
            "expected both targets to be selected, saw {seen:?}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn empty_target_set_errors() {
        let orch = orch(&[]);
        assert!(orch
            .clone()
            .run(Context::default(), request())
            .await
            .is_err());
    }

    #[tokio::test]
    async fn process_signals_is_a_noop() -> Result<(), Box<dyn Error + Send + Sync>> {
        let algo = algo(&["only/model"]);
        Arc::new(algo).process_signals(Signals {}).await?;
        Ok(())
    }

    #[tokio::test]
    async fn decision_is_inspectable_and_downcasts() -> Result<(), Box<dyn Error + Send + Sync>> {
        let orch = orch(&["only/model"]);
        let (trace, _) = orch.clone().run(Context::default(), request()).await?;
        let decision = &trace[0];
        // Uniform, algo-agnostic access via the trait — no concrete type needed.
        assert_eq!(decision.selected_model(), "only/model");
        assert!(decision
            .reasoning()
            .unwrap_or_default()
            .contains("only/model"));
        // Escape hatch: downcast to the concrete decision when the algo is known.
        let concrete = decision
            .as_any()
            .downcast_ref::<RandomDecision>()
            .ok_or("expected a RandomDecision")?;
        assert_eq!(concrete.selected_model, "only/model");
        Ok(())
    }
}
