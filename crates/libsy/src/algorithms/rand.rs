// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Random router built on the [`Algorithm`] interfaces.
//!
//! Selects one target from the set uniformly at random and calls it. This is the
//! simplest possible routing algorithm and the reference for the single-call
//! shape: one `driver.call_llm_target` inside `create_run_task`. Weighted selection
//! can be layered on later; the set defines the candidates.

use std::error::Error;
use std::sync::Arc;

use async_trait::async_trait;
use rand::seq::SliceRandom;

use crate::affinity::Affinity;
use crate::{Algorithm, Context, Decision, Driver, LlmTargetSet, Request, Response};

/// Decision produced by [`RandomAlgo`]: which target was chosen and why.
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
pub struct RandomAlgo {
    target_set: LlmTargetSet,
    affinity: Option<Arc<dyn Affinity>>,
}

impl RandomAlgo {
    /// Creates a router over `target_set`.
    ///
    /// Wrap it in an [`Arc`] and drive it with [`Algorithm::run`] or
    /// [`Algorithm::run_stream`].
    pub fn new(target_set: LlmTargetSet) -> Self {
        Self {
            target_set,
            affinity: None,
        }
    }

    /// Retains selections according to `affinity` while preserving per-request
    /// random routing for requests the policy does not key.
    pub fn with_affinity(mut self, affinity: Arc<dyn Affinity>) -> Self {
        self.affinity = Some(affinity);
        self
    }

    /// Selects one configured target name uniformly at random.
    fn choose_target(&self) -> Result<String, Box<dyn Error + Send + Sync>> {
        if let [target] = self.target_set.targets() {
            return Ok(target.semantic_name.clone());
        }
        // Scope the non-Send ThreadRng so callers may await after this returns.
        let selected = {
            let mut rng = rand::thread_rng();
            self.target_set
                .targets()
                .choose(&mut rng)
                .ok_or("no targets available")?
                .semantic_name
                .clone()
        };
        Ok(selected)
    }
}

#[async_trait]
impl Algorithm for RandomAlgo {
    async fn create_run_task(
        self: Arc<Self>,
        ctx: Context,
        driver: Driver,
        request: Request,
    ) -> Result<Response, Box<dyn Error + Send + Sync>> {
        let (selected, reasoning) = match self.affinity.as_ref() {
            Some(affinity) => match affinity.assignment(&request) {
                Some(selected) => {
                    let reasoning = format!("affinity reused target '{selected}'");
                    (selected, reasoning)
                }
                None => {
                    let proposed = self.choose_target()?;
                    let applies = affinity.key(&request).is_some();
                    let selected = affinity.retain(&request, proposed.clone());
                    let reasoning = if !applies {
                        format!("random routing selected target '{selected}'")
                    } else if selected == proposed {
                        format!("random routing selected and retained target '{selected}'")
                    } else {
                        format!("affinity retained concurrent target '{selected}'")
                    };
                    (selected, reasoning)
                }
            },
            None => {
                let selected = self.choose_target()?;
                let reasoning = format!("random routing selected target '{selected}'");
                (selected, reasoning)
            }
        };
        let target = self.target_set.get_target(&selected)?;
        let decision: Arc<dyn Decision> = Arc::new(RandomDecision {
            reasoning,
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

    use crate::affinity::{Affinity, SessionAffinity, SubAgentAffinity};
    use crate::{
        AgentContext, LlmResponse, LlmTarget, Metadata, Request, RoutedLlmClient, Signals,
    };

    /// Echoes the selected target so tests can inspect which target was called.
    struct EchoClient;

    #[async_trait]
    impl RoutedLlmClient for EchoClient {
        async fn call(
            &self,
            _ctx: Context,
            _request: Request,
            decision: Arc<dyn Decision>,
        ) -> Result<Response, Box<dyn Error + Send + Sync>> {
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
    fn algorithm(names: &[&str]) -> RandomAlgo {
        let targets = names
            .iter()
            .map(|name| LlmTarget {
                semantic_name: (*name).to_string(),
                llm_client: Some(Arc::new(EchoClient)),
            })
            .collect();
        RandomAlgo::new(LlmTargetSet::new(targets))
    }

    fn shared_algorithm(names: &[&str]) -> Arc<dyn Algorithm> {
        Arc::new(algorithm(names))
    }

    fn shared_algorithm_with_affinity(
        names: &[&str],
        affinity: Arc<dyn Affinity>,
    ) -> Arc<dyn Algorithm> {
        Arc::new(algorithm(names).with_affinity(affinity))
    }

    fn affinity_request(is_subagent: bool, agent_id: &str, task_id: &str) -> Request {
        Request {
            llm_request: text_request(Some("auto".to_string()), "hi"),
            raw_request: None,
            metadata: Some(Metadata {
                session_id: Some("session-1".to_string()),
                agent_id: Some(agent_id.to_string()),
                task_id: Some(task_id.to_string()),
                agent_context: Some(Box::new(AgentContext {
                    is_subagent,
                    ..AgentContext::default()
                })),
                ..Metadata::default()
            }),
        }
    }

    #[tokio::test]
    async fn single_target_is_always_selected_and_called(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
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
    async fn selected_target_is_in_the_set_and_matches_the_trace(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
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
    async fn selection_covers_all_targets_over_many_runs(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
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
        assert!(algorithm.run(Context::default(), request()).await.is_err());
    }

    #[tokio::test]
    async fn process_signals_is_a_noop() -> Result<(), Box<dyn Error + Send + Sync>> {
        Arc::new(algorithm(&["only/model"]))
            .process_signals(Signals {})
            .await?;
        Ok(())
    }

    #[tokio::test]
    async fn decision_is_inspectable_and_downcasts() -> Result<(), Box<dyn Error + Send + Sync>> {
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
            .ok_or("expected a RandomDecision")?;
        assert_eq!(concrete.selected_model, "only/model");
        Ok(())
    }

    #[tokio::test]
    async fn subagent_affinity_reuses_selection_across_task_changes(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let algorithm = shared_algorithm_with_affinity(
            &["a/model", "b/model"],
            Arc::new(SubAgentAffinity::new()),
        );
        let (_, first) = Arc::clone(&algorithm)
            .run(
                Context::default(),
                affinity_request(true, "child-1", "task-1"),
            )
            .await?;
        let (trace, second) = algorithm
            .run(
                Context::default(),
                affinity_request(true, "child-1", "task-2"),
            )
            .await?;

        assert_eq!(first.selected_model(), second.selected_model());
        assert!(trace[0]
            .reasoning()
            .unwrap_or_default()
            .contains("affinity reused"));
        Ok(())
    }

    #[tokio::test]
    async fn single_target_subagent_is_retained_without_reselection(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let algorithm =
            shared_algorithm_with_affinity(&["only/model"], Arc::new(SubAgentAffinity::new()));
        Arc::clone(&algorithm)
            .run(
                Context::default(),
                affinity_request(true, "child-1", "task-1"),
            )
            .await?;
        let (trace, response) = algorithm
            .run(
                Context::default(),
                affinity_request(true, "child-1", "task-2"),
            )
            .await?;

        assert_eq!(
            response
                .llm_response
                .as_agg()
                .map(completion_text)
                .unwrap_or_default(),
            "only/model"
        );
        assert!(trace[0]
            .reasoning()
            .unwrap_or_default()
            .contains("affinity reused"));
        Ok(())
    }

    #[tokio::test]
    async fn subagent_affinity_does_not_retain_root_selection(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let affinity = Arc::new(SubAgentAffinity::new());
        let algorithm = shared_algorithm_with_affinity(
            &["a/model", "b/model"],
            Arc::clone(&affinity) as Arc<dyn Affinity>,
        );
        Arc::clone(&algorithm)
            .run(
                Context::default(),
                affinity_request(false, "root-1", "task-1"),
            )
            .await?;

        assert!(affinity
            .assignment(&affinity_request(false, "root-1", "task-2"))
            .is_none());
        let (trace, _) = algorithm
            .run(
                Context::default(),
                affinity_request(false, "root-1", "task-2"),
            )
            .await?;
        assert!(trace[0]
            .reasoning()
            .unwrap_or_default()
            .starts_with("random routing selected"));
        Ok(())
    }

    #[tokio::test]
    async fn session_affinity_reuses_selection_for_different_agents(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let algorithm = shared_algorithm_with_affinity(
            &["a/model", "b/model"],
            Arc::new(SessionAffinity::new()),
        );
        let (_, first) = Arc::clone(&algorithm)
            .run(
                Context::default(),
                affinity_request(false, "agent-a", "task-1"),
            )
            .await?;
        let (_, second) = algorithm
            .run(
                Context::default(),
                affinity_request(false, "agent-b", "task-2"),
            )
            .await?;

        assert_eq!(first.selected_model(), second.selected_model());
        Ok(())
    }

    #[tokio::test]
    async fn concurrent_first_subagent_turns_use_one_assignment(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let algorithm = shared_algorithm_with_affinity(
            &["a/model", "b/model"],
            Arc::new(SubAgentAffinity::new()),
        );
        let first = Arc::clone(&algorithm).run(
            Context::default(),
            affinity_request(true, "child-1", "task-1"),
        );
        let second = algorithm.run(
            Context::default(),
            affinity_request(true, "child-1", "task-1"),
        );
        let (first, second) = tokio::join!(first, second);
        let (_, first) = first?;
        let (_, second) = second?;

        assert_eq!(first.selected_model(), second.selected_model());
        Ok(())
    }
}
