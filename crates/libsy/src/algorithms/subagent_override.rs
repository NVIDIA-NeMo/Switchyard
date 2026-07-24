// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sub-agent override combinator built on the [`Algorithm`] interfaces.
//!
//! Wraps any algorithm without changing its behavior for normal traffic. A
//! request whose [`Metadata`] marks delegated sub-agent work
//! ([`Metadata::is_subagent_work`]) is served by one fixed worker target —
//! keeping a sub-agent loop on an intentional, cache-compatible target —
//! while every other request delegates to the wrapped algorithm. The wrapped
//! algorithm never learns about harnesses or lineage headers, and a worker
//! failure surfaces as a normal target error rather than re-entering the
//! wrapped algorithm.

use std::sync::Arc;

use async_trait::async_trait;

use crate::{Algorithm, Context, Decision, Driver, LlmTarget, Metadata, Request, Response, Result};

/// Decision produced by [`SubagentOverride`] when it routes to the worker target.
pub struct SubagentDecision {
    /// The fixed worker target selected for the sub-agent request.
    pub selected_model: String,
    /// Human-readable explanation of the override.
    pub reasoning: String,
}

impl Decision for SubagentDecision {
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

/// Routes delegated sub-agent work to a fixed worker target; delegates the rest.
pub struct SubagentOverride {
    inner: Arc<dyn Algorithm>,
    worker: LlmTarget,
}

impl SubagentOverride {
    /// Wraps `inner`, sending recognized sub-agent work to `worker` instead.
    ///
    /// Wrap it in an [`Arc`] and drive it with [`Algorithm::run`] or
    /// [`Algorithm::run_stream`].
    pub fn new(inner: Arc<dyn Algorithm>, worker: LlmTarget) -> Self {
        Self { inner, worker }
    }
}

#[async_trait]
impl Algorithm for SubagentOverride {
    fn name(&self) -> &str {
        "subagent_override"
    }

    async fn create_run_task(
        self: Arc<Self>,
        ctx: Context,
        driver: Driver,
        request: Request,
    ) -> Result<Response> {
        let is_subagent_work = request
            .metadata
            .as_ref()
            .is_some_and(Metadata::is_subagent_work);
        if !is_subagent_work {
            return Arc::clone(&self.inner)
                .create_run_task(ctx, driver, request)
                .await;
        }

        let selected = self.worker.semantic_name.clone();
        let decision: Arc<dyn Decision> = Arc::new(SubagentDecision {
            reasoning: format!("sub-agent work routed to fixed worker target '{selected}'"),
            selected_model: selected,
        });
        driver.info(ctx.clone(), Arc::clone(&decision)).await?;
        driver
            .call_llm_target(ctx, &self.worker, request, decision)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use switchyard_protocol::{completion_text, text_request, text_response};

    use crate::algorithms::Random;
    use crate::{LlmResponse, LlmTargetSet, RoutedLlmClient};

    /// Echoes the selected target so tests can inspect which target was called.
    struct EchoClient;

    #[async_trait]
    impl RoutedLlmClient for EchoClient {
        async fn call(
            &self,
            _ctx: Context,
            _request: Request,
            decision: Arc<dyn Decision>,
        ) -> std::result::Result<Response, switchyard_protocol::LlmClientError> {
            Ok(Response {
                llm_response: LlmResponse::Agg(text_response(None, decision.selected_model())),
                metadata: None,
            })
        }
    }

    fn target(name: &str) -> LlmTarget {
        LlmTarget {
            semantic_name: name.to_string(),
            llm_client: Some(Arc::new(EchoClient)),
        }
    }

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

    /// Wraps single-target random routing so the inner selection is deterministic.
    fn algorithm() -> Arc<dyn Algorithm> {
        let inner: Arc<dyn Algorithm> =
            Arc::new(Random::new(LlmTargetSet::new(vec![target("orchestrator")])));
        Arc::new(SubagentOverride::new(inner, target("worker")))
    }

    async fn selected_model(headers: &[(&str, &str)]) -> Result<String> {
        let (trace, response) = algorithm()
            .run(Context::default(), request(headers))
            .await?;
        let selected = response
            .llm_response
            .as_agg()
            .map(completion_text)
            .unwrap_or_default();
        assert_eq!(
            trace.last().map(|d| d.selected_model().to_string()),
            Some(selected.clone())
        );
        Ok(selected)
    }

    #[tokio::test]
    async fn requests_without_metadata_delegate_to_the_wrapped_algorithm() -> Result<()> {
        assert_eq!(selected_model(&[]).await?, "orchestrator");
        Ok(())
    }

    #[tokio::test]
    async fn subagent_work_is_routed_to_the_worker_target() -> Result<()> {
        // Claude Code child-agent lineage.
        let claude = &[
            ("x-claude-code-session-id", "root"),
            ("x-claude-code-agent-id", "child-1"),
        ];
        assert_eq!(selected_model(claude).await?, "worker");

        // Codex delegated-work kinds.
        assert_eq!(
            selected_model(&[("x-openai-subagent", "review")]).await?,
            "worker"
        );
        assert_eq!(
            selected_model(&[("x-openai-subagent", "collab_spawn")]).await?,
            "worker"
        );
        Ok(())
    }

    #[tokio::test]
    async fn harness_maintenance_turns_stay_on_the_wrapped_algorithm() -> Result<()> {
        assert_eq!(
            selected_model(&[("x-openai-subagent", "compact")]).await?,
            "orchestrator"
        );
        assert_eq!(
            selected_model(&[("x-switchyard-is-subagent", "false")]).await?,
            "orchestrator"
        );
        Ok(())
    }
}
