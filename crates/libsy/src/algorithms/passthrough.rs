// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Single-target routing for direct model calls and integration diagnostics.

use std::sync::Arc;

use switchyard_protocol::{Request, Response};

use crate::Result;
use crate::core::algorithm::{Algorithm, Driver, LlmTarget};
use switchyard_protocol::Decision;

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

#[async_trait::async_trait]
impl Algorithm for Passthrough {
    fn name(&self) -> &str {
        "passthrough"
    }

    async fn route(self: Arc<Self>, driver: Driver, request: Request) -> Result<Response> {
        let decision: Decision = Decision::new(
            self.target.semantic_name.clone(),
            Some(format!(
                "passthrough selected target '{}'",
                self.target.semantic_name
            )),
            true,
        );
        driver.info(decision.clone()).await?;
        driver.call_model(request, decision).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::Passthrough;
    use crate::core::algorithm::{Algorithm, LlmTarget};
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
        let algorithm: Arc<dyn Algorithm> = Arc::new(Passthrough::new(LlmTarget {
            semantic_name: MODEL_ID.to_string(),
        }));
        let (trace, response) = test_drive(algorithm, request, echo()).await?;

        assert_eq!(
            response
                .llm_response
                .as_agg()
                .map(completion_text)
                .unwrap_or_default(),
            MODEL_ID
        );
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].selected_model_id(), MODEL_ID);
        assert!(trace[0].is_answer_call());
        Ok(())
    }
}
