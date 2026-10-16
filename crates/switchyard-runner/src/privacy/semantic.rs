// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Converts semantic privacy assessments into the route's lane policy.

use std::sync::Arc;

use libsy::{Call, ClassifierResponseFormat, LibsyError, PrivacyPreflight};
use serde_json::Value;
use switchyard_protocol::{
    DecisionRequest, ModelId, Request, RoutedDecisionClient, RoutedLlmClient,
};

use super::{PrivacyDecision, PrivacyLane};

pub(crate) struct SemanticPrivacyClassifier {
    preflight: PrivacyPreflight,
    client: ClassifierClient,
    clear_threshold: f64,
}

enum ClassifierClient {
    Decision(Arc<dyn RoutedDecisionClient>),
    Llm(Arc<dyn RoutedLlmClient>),
}

impl SemanticPrivacyClassifier {
    pub(crate) fn decision(
        target: ModelId,
        client: Arc<dyn RoutedDecisionClient>,
        instructions: Option<&Value>,
        clear_threshold: f64,
    ) -> Self {
        Self {
            preflight: PrivacyPreflight::decision(target, instructions),
            client: ClassifierClient::Decision(client),
            clear_threshold,
        }
    }

    pub(crate) fn llm(
        target: ModelId,
        client: Arc<dyn RoutedLlmClient>,
        prompt: Option<&str>,
        response_format_type: ClassifierResponseFormat,
        clear_threshold: f64,
    ) -> libsy::Result<Self> {
        Ok(Self {
            preflight: PrivacyPreflight::llm(target, prompt, response_format_type)?,
            client: ClassifierClient::Llm(client),
            clear_threshold,
        })
    }

    pub(crate) async fn assess(&self, request: &Request) -> PrivacyDecision {
        let assessment = self.preflight.assess(request, |call| self.call(call)).await;
        let reason = assessment.reason_code();
        let score = assessment.clear_score();
        let (lane, reason) = match reason {
            "no_sensitive_content" if score.is_some_and(|score| score >= self.clear_threshold) => {
                (PrivacyLane::Standard, reason)
            }
            "no_sensitive_content" => (PrivacyLane::Restricted, "below_clear_threshold"),
            _ => (PrivacyLane::Restricted, reason),
        };
        PrivacyDecision::semantic(lane, reason, score, self.clear_threshold)
    }

    async fn call(&self, call: Call) -> libsy::Result<()> {
        match (&self.client, call) {
            (ClassifierClient::Llm(client), Call::Model(mut call)) => {
                // The preflight has one candidate; move its payload rather than cloning it.
                let request = std::mem::take(&mut call.request);
                let target = call.models.first().cloned().ok_or(LibsyError::NoTargets)?;
                call.respond(async move {
                    client
                        .call(request)
                        .await
                        .map_err(|error| LibsyError::client_call(target, error))
                })
                .await
            }
            (ClassifierClient::Decision(client), Call::Decision(mut call)) => {
                let request = DecisionRequest {
                    model: call.request.model.take(),
                    context: call.request.context.take(),
                    questions: std::mem::take(&mut call.request.questions),
                };
                let model = call.model.clone();
                call.respond(async move {
                    client
                        .call(request)
                        .await
                        .map_err(|error| LibsyError::client_call(model, error))
                })
                .await
            }
            _ => Err(LibsyError::AlgorithmError {
                message: "privacy preflight call does not match its configured client".into(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use async_trait::async_trait;
    use serde_json::json;
    use switchyard_protocol::{
        DecisionAnswer, DecisionKind, DecisionResponse, DecisionValue, LlmClientError, LlmResponse,
        Probability, Response, Usage, text_response,
    };

    use super::*;

    struct ReplyClient {
        reason: &'static str,
        score: f64,
        is_failed: bool,
    }

    impl ReplyClient {
        fn failure(&self) -> Result<(), LlmClientError> {
            if self.is_failed {
                Err(LlmClientError::Configuration {
                    message: "private provider diagnostic".into(),
                })
            } else {
                Ok(())
            }
        }
    }

    #[async_trait]
    impl RoutedDecisionClient for ReplyClient {
        async fn call(&self, request: DecisionRequest) -> Result<DecisionResponse, LlmClientError> {
            self.failure()?;
            let mut answers = BTreeMap::new();
            for (id, question) in request.questions {
                let DecisionKind::Choice { options } = question.kind else {
                    panic!("expected privacy choices");
                };
                let remaining = if self.reason == "no_sensitive_content" {
                    "uncertain"
                } else {
                    self.reason
                };
                answers.insert(
                    id,
                    DecisionAnswer {
                        value: DecisionValue::Choice {
                            selected: self.reason.into(),
                            probabilities: Some(
                                options
                                    .into_iter()
                                    .map(|option| {
                                        let probability = match option.id.as_str() {
                                            "no_sensitive_content" => self.score,
                                            id if id == remaining => 1.0 - self.score,
                                            _ => 0.0,
                                        };
                                        (option.id, Probability(probability))
                                    })
                                    .collect(),
                            ),
                        },
                        provider_confidence: None,
                    },
                );
            }
            Ok(DecisionResponse {
                id: None,
                model: request.model,
                answers,
                usage: Usage::default(),
            })
        }
    }

    #[async_trait]
    impl RoutedLlmClient for ReplyClient {
        async fn call(&self, _request: Request) -> Result<Response, LlmClientError> {
            self.failure()?;
            Ok(Response {
                llm_response: LlmResponse::Agg(text_response(
                    None,
                    json!({"reason_code": self.reason, "clear_score": self.score}).to_string(),
                )),
                metadata: None,
                upstream_headers: Default::default(),
            })
        }
    }

    fn classifiers(
        reason: &'static str,
        score: f64,
        is_failed: bool,
    ) -> libsy::Result<[SemanticPrivacyClassifier; 2]> {
        let client = Arc::new(ReplyClient {
            reason,
            score,
            is_failed,
        });
        Ok([
            SemanticPrivacyClassifier::decision("privacy/model".into(), client.clone(), None, 0.9),
            SemanticPrivacyClassifier::llm(
                "privacy/model".into(),
                client,
                None,
                ClassifierResponseFormat::JsonSchema,
                0.9,
            )?,
        ])
    }

    #[tokio::test]
    async fn assessments_preserve_thresholds() -> libsy::Result<()> {
        for (reason, score, lane, expected_reason) in [
            (
                "no_sensitive_content",
                0.95,
                "standard",
                "no_sensitive_content",
            ),
            (
                "no_sensitive_content",
                0.8,
                "restricted",
                "below_clear_threshold",
            ),
            ("confidential_data", 0.1, "restricted", "confidential_data"),
            ("uncertain", 0.5, "restricted", "uncertain"),
        ] {
            for classifier in classifiers(reason, score, false)? {
                let decision = classifier.assess(&Request::default()).await;
                assert_eq!(decision.lane(), lane);
                assert_eq!(decision.reason_code(), expected_reason);
                assert_eq!(decision.clear_score(), Some(score));
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn failed_clients_restrict_only_the_current_request() -> libsy::Result<()> {
        for classifier in classifiers("no_sensitive_content", 0.95, true)? {
            let decision = classifier.assess(&Request::default()).await;
            assert_eq!(decision.lane(), "restricted");
            assert_eq!(decision.reason_code(), "classifier_failed");
            assert_eq!(decision.clear_score(), None);
        }
        Ok(())
    }
}
