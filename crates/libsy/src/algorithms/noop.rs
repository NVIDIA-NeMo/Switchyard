// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test-only algorithm that returns a hard-coded response without calling a backend.

use std::sync::Arc;

use switchyard_protocol::{
    AggLlmResponse, ContentBlock, LlmResponse, ModelId, Request, Response, ResponseOutput, Role,
    StopReason,
};

use crate::core::algorithm::{Algorithm, Driver};
use crate::{Result, RoutingOutcome};

/// Test helper that returns a hard-coded response without routing or model I/O.
pub struct Noop {}

#[async_trait::async_trait]
impl Algorithm for Noop {
    fn name(&self) -> &str {
        "noop"
    }

    async fn route(self: Arc<Self>, _driver: Driver, request: Request) -> Result<RoutingOutcome> {
        let streaming = request.llm_request.stream;
        let model_id = request
            .model_id()
            .unwrap_or_else(|| ModelId::from("switchyard/noop"));
        tracing::info!(target = %model_id, "noop returned its synthetic response");
        let aggregate = AggLlmResponse {
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
        };
        let llm_response = if streaming {
            LlmResponse::Stream(aggregate.into_stream())
        } else {
            LlmResponse::Agg(aggregate)
        };
        let response = Response {
            llm_response,
            metadata: request.metadata.clone(),
        };
        Ok(RoutingOutcome::answered(model_id, request, response))
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
        let (selected_model, response) = test_drive(a, request, echo()).await?;
        assert_eq!(selected_model, TEST_MODEL);
        assert_eq!(response.selected_model(), Some(TEST_MODEL));
        Ok(())
    }

    #[tokio::test]
    async fn test_noop_algo_streams_when_requested() -> Result<()> {
        let request = Request {
            llm_request: LlmRequest {
                model: Some("streaming-noop".to_string()),
                messages: vec![Message::text(Role::User, "hi")],
                stream: true,
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: None,
        };

        let algorithm: Arc<dyn Algorithm> = Arc::new(Noop {});
        let (_, response) = test_drive(algorithm, request, echo()).await?;

        assert!(matches!(response.llm_response, LlmResponse::Stream(_)));
        Ok(())
    }
}
