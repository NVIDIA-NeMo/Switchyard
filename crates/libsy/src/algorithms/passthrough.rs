// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Single-target routing for direct model calls and integration diagnostics.

use std::sync::Arc;

use switchyard_protocol::{Request, Response};

use crate::Result;
use crate::core::algorithm::{Algorithm, Driver, LlmTarget};
use switchyard_protocol::{Context, Decision};

/// Routing algorithm that always calls one configured target.
pub struct Passthrough {
    target: LlmTarget,
}

impl Passthrough {
    /// Creates an algorithm that always calls `target`.
    pub fn new(target: LlmTarget) -> Self {
        Passthrough { target }
    }
}

/// Decision emitted before [`Passthrough`] calls its configured target.
pub struct PassthroughDecision {
    model_id: String,
}

impl Decision for PassthroughDecision {
    fn selected_model(&self) -> &str {
        &self.model_id
    }
    fn reasoning(&self) -> Option<&str> {
        // There is no decision to take, so no reasoning
        None
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[async_trait::async_trait]
impl Algorithm for Passthrough {
    fn name(&self) -> &str {
        "passthrough"
    }

    async fn create_run_task(
        self: Arc<Self>,
        ctx: Context,
        driver: Driver,
        request: Request,
    ) -> Result<Response> {
        let decision: Arc<dyn Decision> = Arc::new(PassthroughDecision {
            model_id: self.target.semantic_name.clone(),
        });
        driver.info(ctx.clone(), decision.clone()).await?;
        driver
            .call_llm_target(ctx, &self.target, request, decision)
            .await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::Passthrough;
    use crate::core::algorithm::{Algorithm, LlmTarget};
    use switchyard_protocol::{
        Context, Decision, LlmResponse, Request, Response, RoutedLlmClient, completion_text,
        text_request, text_response,
    };

    /// Echoes the selected target so tests can inspect which target was called.
    /// TODO: Duplicated from rand.rs
    struct EchoClient;

    #[async_trait::async_trait]
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

    #[tokio::test]
    async fn test_passthrough() -> crate::Result<()> {
        const MODEL_ID: &str = "testing/passthrough";
        let request = Request {
            llm_request: text_request(Some("auto".to_string()), "hi"),
            raw_request: None,
            metadata: None,
        };
        let algorithm: Arc<dyn Algorithm> = Arc::new(Passthrough::new(LlmTarget {
            semantic_name: MODEL_ID.to_string(),
            llm_client: Some(Arc::new(EchoClient)),
        }));
        let (trace, response) = algorithm.run(Context::default(), request).await?;

        assert_eq!(
            response
                .llm_response
                .as_agg()
                .map(completion_text)
                .unwrap_or_default(),
            MODEL_ID
        );
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].selected_model(), MODEL_ID);
        Ok(())
    }
}
