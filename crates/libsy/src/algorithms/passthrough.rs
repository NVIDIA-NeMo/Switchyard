// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Single-target routing for direct model calls and integration diagnostics.

use std::sync::Arc;

use switchyard_protocol::{ModelId, Request, Response};

use crate::core::algorithm::{Algorithm, Driver};
use crate::target_modalities::eligible_targets;
use crate::{Result, TargetModalities};
use switchyard_protocol::Decision;

/// Routing algorithm that always calls one configured target.
pub struct Passthrough {
    target: ModelId,
    target_modalities: Option<TargetModalities>,
}

impl Passthrough {
    /// Creates an algorithm that always calls `target`.
    pub fn new(target: impl Into<ModelId>) -> Self {
        Passthrough {
            target: target.into(),
            target_modalities: None,
        }
    }

    /// Validates requests against the sole target's declared input modalities.
    pub fn with_target_modalities(mut self, target_modalities: TargetModalities) -> Self {
        self.target_modalities = Some(target_modalities);
        self
    }
}

#[async_trait::async_trait]
impl Algorithm for Passthrough {
    fn name(&self) -> &str {
        "passthrough"
    }

    async fn route(self: Arc<Self>, driver: Driver, request: Request) -> Result<Response> {
        if let Some(target_modalities) = &self.target_modalities {
            let (required, eligible) = eligible_targets(
                std::slice::from_ref(&self.target),
                target_modalities,
                &request,
            )?;
            tracing::info!(
                required_modalities = ?required,
                eligible_targets = ?eligible,
                "computed modality-compatible targets"
            );
        }
        tracing::info!(target = %self.target, "passthrough selected target");
        let decision: Decision = Decision::new(self.target.clone(), true);
        driver.decide(decision.clone()).await?;
        driver
            .call_model(request, vec![self.target.clone()], true)
            .await
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::Arc;

    use super::Passthrough;
    use crate::LibsyError;
    use crate::core::algorithm::Algorithm;
    use crate::core::testing::{echo, test_drive};
    use switchyard_protocol::{
        ContentBlock, ImageSource, InputModality, LlmRequest, Message, Request, Role,
        completion_text, text_request,
    };

    #[tokio::test]
    async fn test_passthrough() -> crate::Result<()> {
        const MODEL_ID: &str = "testing/passthrough";
        let request = Request {
            llm_request: text_request(Some("auto".to_string()), "hi"),
            raw_request: None,
            metadata: None,
        };
        let algorithm: Arc<dyn Algorithm> = Arc::new(Passthrough::new(MODEL_ID));
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

    #[tokio::test]
    async fn passthrough_rejects_an_incompatible_request_before_calling() {
        const MODEL_ID: &str = "testing/text-only";
        let algorithm: Arc<dyn Algorithm> = Arc::new(
            Passthrough::new(MODEL_ID).with_target_modalities(BTreeMap::from([(
                MODEL_ID.into(),
                BTreeSet::from([InputModality::Text]),
            )])),
        );
        let request = Request {
            llm_request: LlmRequest {
                messages: vec![Message {
                    role: Role::User,
                    content: vec![ContentBlock::Image {
                        source: ImageSource::Url {
                            url: "https://example.test/image.png".to_string(),
                            detail: None,
                        },
                    }],
                }],
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: None,
        };

        let result = test_drive(algorithm, request, echo()).await;

        assert!(matches!(
            result,
            Err(LibsyError::NoCompatibleTargets { .. })
        ));
    }
}
