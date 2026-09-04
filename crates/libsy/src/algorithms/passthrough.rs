// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Single-target routing for direct model calls and integration diagnostics.

use std::sync::Arc;

use switchyard_protocol::{Category, Request};

use crate::core::algorithm::{Algorithm, Driver};
use crate::{LibsyError, Result, RoutingOutcome};

/// Routing algorithm that always selects one configured target.
#[derive(Default)]
pub struct Passthrough {}

#[async_trait::async_trait]
impl Algorithm for Passthrough {
    fn name(&self) -> &str {
        "passthrough"
    }

    async fn route(self: Arc<Self>, driver: Driver, request: Request) -> Result<RoutingOutcome> {
        let Some(target) = driver.models_for(Category::Any).first() else {
            return Err(LibsyError::NoTargets);
        };
        tracing::info!(target = %target, "passthrough selected target");
        Ok(RoutingOutcome::route_to(
            target.clone(),
            Vec::new(),
            request,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::Passthrough;
    use crate::core::testing::echo;
    use crate::core::{algorithm::Algorithm, testing::test_drive_with_models};
    use switchyard_protocol::{Category, Request, completion_text, text_request};

    #[tokio::test]
    async fn test_passthrough() -> crate::Result<()> {
        const MODEL_ID: &str = "testing/passthrough";
        let request = Request {
            llm_request: text_request(Some("auto".to_string()), "hi"),
            raw_request: None,
            metadata: None,
        };
        let algorithm: Arc<dyn Algorithm> = Arc::new(Passthrough::default());
        let models = Category::to_map(Category::Any, &[MODEL_ID]);
        let (selected_model, response) =
            test_drive_with_models(algorithm, request, models, echo()).await?;

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
