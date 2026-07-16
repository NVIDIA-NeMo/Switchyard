// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! No-op router. Always returns a hard coded response. Does not route to a backend.
//! For testing.

use std::{error::Error, sync::Arc};

use switchyard_protocol::{Request, Response};

use crate::{Algorithm, Context, Decision, Driver, LlmResponse, Signals};

pub struct NoopAlgo {}

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
impl Algorithm for NoopAlgo {
    async fn create_run_task(
        self: Arc<Self>,
        ctx: Context,
        driver: Driver,
        request: Request,
    ) -> Result<Response, Box<dyn Error + Send + Sync>> {
        let model = request
            .requested_model()
            .unwrap_or("switchyard/noop")
            .to_string();
        let decision: Arc<dyn Decision> = Arc::new(NoopDecision {
            model: model.clone(),
        });
        driver.info(ctx, decision.clone()).await?;

        let json = serde_json::json!({
            "id": "switchyard-noop",
            "model": model,
            "outputs": [{
                "role": "Assistant",
                "content": [{
                    "Text": {
                        "text": "OK"
                    }
                }],
                "stop_reason": "EndTurn"
            }],
            "usage": {
                "input_tokens": null,
                "output_tokens": null,
                "total_tokens": null,
                "reasoning_tokens": null
            },
            "extensions": {
                "fields": {}
            },
            "preservation": {
                "requests": {},
                "responses": {}
        }});

        let llm_response: LlmResponse = serde_json::from_value(json)?;
        let response = Response {
            llm_response,
            metadata: request.metadata.clone(),
        };
        Ok(response)
    }

    async fn process_signals(
        self: Arc<Self>,
        _signals: Signals,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use switchyard_protocol::{LlmRequest, Message, Role};

    use super::*;

    #[tokio::test]
    async fn test_noop_algo() -> Result<(), Box<dyn Error + Send + Sync>> {
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

        let a: Arc<dyn Algorithm> = Arc::new(NoopAlgo {});
        let (decisions, response) = a.run(Context::default(), request).await?;
        let Some(decision) = decisions.first() else {
            panic!("Expected exactly one Decision");
        };
        assert_eq!(decision.selected_model(), TEST_MODEL);
        assert_eq!(response.model(), Some(TEST_MODEL));
        Ok(())
    }
}
