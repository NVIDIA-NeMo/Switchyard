// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Direct parent routing with an optional delegated-work route.

use std::sync::Arc;

use switchyard_protocol::{ModelId, Request};

use super::fall_through::{DefaultTarget, FallThrough};
use super::subagent::{SubagentRouter, SubagentRouterConfig};
use crate::core::algorithm::{Algorithm, Driver};
use crate::{Result, RoutingOutcome};

/// Backwards-compatible name for [`SubagentRouterConfig`].
pub use super::subagent::SubagentRouterConfig as PassthroughSubagentConfig;

/// Routes parent traffic directly and optionally routes delegated sub-agent work.
pub struct Passthrough {
    route: Arc<dyn Algorithm>,
}

/// Complete construction settings for [`Passthrough`].
pub struct PassthroughConfig {
    /// Target used for parent and harness-maintenance traffic.
    pub parent_target: ModelId,
    /// Optional delegated-work routing.
    pub subagent: Option<SubagentRouterConfig>,
}

impl Passthrough {
    /// Creates direct parent routing, optionally with a decision gate for sub-agents.
    ///
    /// A `new_session` classifier retains its first decision by `session + agent`; an
    /// `every_request` gate decides each delegated request independently. An abstaining gate uses
    /// the child default. Root and harness-maintenance traffic continue to the parent target.
    ///
    /// # Errors
    ///
    /// Returns an error when the delegated-work routing configuration is invalid.
    pub fn new(config: PassthroughConfig) -> Result<Self> {
        let parent_target = config.parent_target;
        let parent: Arc<dyn Algorithm> = Arc::new(
            FallThrough::new(vec![parent_target.clone()])
                .with_name("passthrough")
                .with_classifier(Arc::new(DefaultTarget::new(parent_target))),
        );
        let route: Arc<dyn Algorithm> = match config.subagent {
            Some(subagent) => Arc::new(SubagentRouter::new(parent, subagent)?),
            None => parent,
        };
        Ok(Self { route })
    }
}

#[async_trait::async_trait]
impl Algorithm for Passthrough {
    fn name(&self) -> &str {
        "passthrough"
    }

    async fn route(self: Arc<Self>, driver: Driver, request: Request) -> Result<RoutingOutcome> {
        self.route.clone().route(driver, request).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{Passthrough, PassthroughConfig, PassthroughSubagentConfig};
    use crate::core::algorithm::Algorithm;
    use crate::core::testing::{echo, test_drive};
    use switchyard_protocol::{Metadata, ModelId, Request, completion_text, text_request};

    fn request(metadata: Option<Metadata>) -> Request {
        Request {
            llm_request: text_request(Some("auto".to_string()), "hi"),
            raw_request: None,
            metadata,
        }
    }

    fn child(agent_id: &str) -> Request {
        request(Some(Metadata {
            session_id: Some("session-1".to_string()),
            agent_id: Some(agent_id.to_string()),
            is_subagent: true,
            is_delegated_work: true,
            ..Metadata::default()
        }))
    }

    #[tokio::test]
    async fn test_passthrough() -> crate::Result<()> {
        const MODEL_ID: &str = "testing/passthrough";
        let request = Request {
            llm_request: text_request(Some("auto".to_string()), "hi"),
            raw_request: None,
            metadata: None,
        };
        let algorithm: Arc<dyn Algorithm> = Arc::new(Passthrough::new(PassthroughConfig {
            parent_target: ModelId::from(MODEL_ID),
            subagent: None,
        })?);
        let (selected_model, response) = test_drive(algorithm, request, echo()).await?;

        assert_eq!(
            response
                .llm_response
                .as_agg()
                .map(completion_text)
                .unwrap_or_default(),
            MODEL_ID
        );
        assert_eq!(selected_model, MODEL_ID);
        Ok(())
    }

    #[tokio::test]
    async fn fixed_subagent_gate_routes_parent_and_child() -> crate::Result<()> {
        let router = Arc::new(Passthrough::new(PassthroughConfig {
            parent_target: ModelId::from("parent"),
            subagent: Some(PassthroughSubagentConfig::fixed_target("worker")),
        })?);

        let (parent, _) = test_drive(router.clone(), request(None), echo()).await?;
        let (child, _) = test_drive(router, child("child-1"), echo()).await?;

        assert_eq!(parent, "parent");
        assert_eq!(child, "worker");
        Ok(())
    }
}
