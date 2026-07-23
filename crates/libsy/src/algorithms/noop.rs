// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! No-op router. Always returns a hard coded response. Does not route to a backend.
//! For testing.

use std::sync::Arc;

use switchyard_protocol::{
    AggLlmResponse, ContentBlock, LlmResponse, Request, Response, ResponseOutput, Role, StopReason,
};

use crate::{Algorithm, Context, Decision, Driver, Result};

/// A routing algorithm that does not route. It returns a hard-coded response.
pub struct Noop {}

/// How [`Noop`] records which model it chose. This will be the model on the Request if any,
/// otherwise a hard coded placeholder. Neither is actually used.
pub struct NoopDecision {
    model: String,
}

impl Decision for NoopDecision {
    fn selected_model(&self) -> &str {
        &self.model
    }
    fn reasoning(&self) -> Option<&str> {
        None
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[async_trait::async_trait]
impl Algorithm for Noop {
    fn name(&self) -> &str {
        "noop"
    }

    async fn create_run_task(
        self: Arc<Self>,
        ctx: Context,
        driver: Driver,
        request: Request,
    ) -> Result<Response> {
        let model = request
            .requested_model()
            .unwrap_or("switchyard/noop")
            .to_string();
        let decision: Arc<dyn Decision> = Arc::new(NoopDecision {
            model: model.clone(),
        });
        driver.info(ctx, decision.clone()).await?;

        let llm_response = LlmResponse::Agg(AggLlmResponse {
            id: Some("switchyard-noop".to_string()),
            model: Some(model),
            outputs: vec![ResponseOutput {
                role: Role::Assistant,
                content: vec![ContentBlock::Text {
                    text: "OK".to_string(),
                }],
                stop_reason: Some(StopReason::EndTurn),
            }],
            ..Default::default()
        });
        let response = Response {
            llm_response,
            metadata: request.metadata.clone(),
        };
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use switchyard_protocol::{LlmRequest, Message, Role};

    use super::*;

    #[tokio::test]
    async fn test_noop_algo() -> Result<()> {
        const TEST_MODEL: &str = "test_noop_algo";
        let request = Request {
            llm_request: LlmRequest {
                model: Some(TEST_MODEL.to_string()),
                messages: vec![Message::text(Role::User, "hi")],
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: None,
        };

        let a: Arc<dyn Algorithm> = Arc::new(Noop {});
        let (decisions, response) = a.run(Context::default(), request).await?;
        let Some(decision) = decisions.first() else {
            panic!("Expected exactly one Decision");
        };
        assert_eq!(decision.selected_model(), TEST_MODEL);
        assert_eq!(response.selected_model(), Some(TEST_MODEL));
        Ok(())
    }
}
