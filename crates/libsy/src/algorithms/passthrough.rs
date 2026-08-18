// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Single-target routing for direct model calls and integration diagnostics.

use std::sync::Arc;

use switchyard_protocol::{ModelId, Request};

use crate::core::algorithm::{Algorithm, Driver};
use crate::{Result, RoutingOutcome};

/// Routing algorithm that always selects one configured target.
pub struct Passthrough {
    target: ModelId,
}

impl Passthrough {
    /// Creates an algorithm that always selects `target`.
    pub fn new(target: impl Into<ModelId>) -> Self {
        Passthrough {
            target: target.into(),
        }
    }
}

#[async_trait::async_trait]
impl Algorithm for Passthrough {
    fn name(&self) -> &str {
        "passthrough"
    }

    async fn route(self: Arc<Self>, _driver: Driver, request: Request) -> Result<RoutingOutcome> {
        tracing::info!(target = %self.target, "passthrough selected target");
        Ok(RoutingOutcome::route_to(
            self.target.clone(),
            Vec::new(),
            request,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::Passthrough;
    use crate::core::algorithm::Algorithm;
    use crate::core::testing::{echo, test_drive};
    use switchyard_protocol::{Request, completion_text, text_request};

    #[tokio::test]
    async fn test_passthrough() -> crate::Result<()> {
        const MODEL_ID: &str = "testing/passthrough";
        let request = Request {
            llm_request: text_request(Some("auto".to_string()), "hi"),
            raw_request: None,
            metadata: None,
        };
        let algorithm: Arc<dyn Algorithm> = Arc::new(Passthrough::new(MODEL_ID));
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
}
