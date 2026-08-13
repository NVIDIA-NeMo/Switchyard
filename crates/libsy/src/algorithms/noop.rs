// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test-only algorithm that returns a hard-coded response without calling a backend.

use std::sync::Arc;

use switchyard_protocol::{
    AggLlmResponse, ContentBlock, LlmResponse, ModelId, Request, Response, ResponseOutput, Role,
    StopReason,
};

use crate::Result;
use crate::core::algorithm::{Algorithm, Driver};
use switchyard_protocol::Decision;

/// Test helper that returns a hard-coded response without routing or model I/O.
pub struct Noop {}

#[async_trait::async_trait]
impl Algorithm for Noop {
    fn name(&self) -> &str {
        "noop"
    }

    async fn route(self: Arc<Self>, driver: Driver, request: Request) -> Result<Response> {
        let model_id = request
            .model_id()
            .unwrap_or_else(|| ModelId::from("switchyard/noop"));
        let decision: Decision = Decision::new(
            model_id.clone(),
            Some("noop returned its synthetic response".to_string()),
            true,
        );
        driver.decide(decision.clone()).await?;

        let llm_response = LlmResponse::Agg(AggLlmResponse {
            id: Some("switchyard-noop".to_string()),
            model: Some(model_id.to_string()),
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
    use crate::core::testing::{echo, test_drive};

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

        // `Noop` synthesizes its own response and never offloads a call, so `echo` is
        // never reached.
        let a: Arc<dyn Algorithm> = Arc::new(Noop {});
        let (decisions, response) = test_drive(a, request, echo()).await?;
        let Some(decision) = decisions.first() else {
            panic!("Expected exactly one Decision");
        };
        assert_eq!(decision.selected_model_id(), TEST_MODEL);
        assert!(decision.is_answer_call());
        assert_eq!(response.selected_model(), Some(TEST_MODEL));
        Ok(())
    }
}
