// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Buffered System One calls using the provider-neutral decision IR.

use std::collections::BTreeMap;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::{Url, header::HeaderValue};
use serde::Deserialize;
use serde_json::{Value, json};
use switchyard_protocol::{
    BooleanEstimate, DecisionAnswer, DecisionKind, DecisionRequest, DecisionResponse,
    DecisionValue, ModelId, Probability, ProviderConfidence, RoutedDecisionClient, ScoreValue,
    Usage,
};

use crate::client::convert_reqwest_error;
use crate::{LlmClientError, Result, metrics};

/// Serves decision requests through a System One endpoint, such as TypeSafe's Jev API.
pub struct SystemOneClient {
    client: reqwest::Client,
    endpoint: Url,
    api_key: String,
}

impl SystemOneClient {
    /// `endpoint` is the full URL, including `/v1/systemone`. Each call makes one
    /// attempt; `timeout` covers sending the request and reading the response body.
    pub fn new(endpoint: Url, api_key: String, timeout: Duration) -> Result<Self> {
        if HeaderValue::try_from(format!("Bearer {api_key}")).is_err() {
            return Err(LlmClientError::Configuration {
                message: "System One API key cannot be encoded as an HTTP header".into(),
            });
        }
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(convert_reqwest_error)?;
        Ok(Self {
            client,
            endpoint,
            api_key,
        })
    }
}

#[async_trait]
impl RoutedDecisionClient for SystemOneClient {
    async fn call(&self, request: DecisionRequest) -> Result<DecisionResponse> {
        let response = self
            .client
            .post(self.endpoint.clone())
            .bearer_auth(&self.api_key)
            .json(&encode(&request)?)
            .send()
            .await
            .inspect_err(|_| metrics::record_upstream_attempt(None))
            .map_err(convert_reqwest_error)?;
        let status = response.status();
        let body = response
            .bytes()
            .await
            .inspect_err(|_| metrics::record_upstream_attempt(None))
            .map_err(convert_reqwest_error)?;
        metrics::record_upstream_attempt(Some(status.as_u16()));
        if !status.is_success() {
            return Err(LlmClientError::UpstreamHttp {
                status,
                body: String::from_utf8_lossy(&body).replace(&self.api_key, "[REDACTED]"),
            });
        }
        let response: WireResponse =
            serde_json::from_slice(&body).map_err(|source| LlmClientError::InvalidResponse {
                source: Box::new(source),
            })?;
        if !response.answers.keys().eq(request.questions.keys()) {
            return Err(LlmClientError::ResponseTranslation(
                "System One answer keys do not match the requested question keys".into(),
            ));
        }
        let answers = response
            .answers
            .into_iter()
            .map(|(id, answer)| {
                let value = match (request.questions.get(&id).map(|q| &q.kind), answer.value) {
                    (Some(DecisionKind::Boolean { .. }), WireValue::Noul { noul }) => {
                        DecisionValue::Boolean(BooleanEstimate::ProbabilityTrue(noul))
                    }
                    (
                        Some(DecisionKind::Choice { options }),
                        WireValue::Choice {
                            choice,
                            probabilities,
                        },
                    ) if options.iter().any(|option| option.id == choice) => {
                        DecisionValue::Choice {
                            selected: choice,
                            probabilities,
                        }
                    }
                    (
                        Some(DecisionKind::Score { levels }),
                        WireValue::Score {
                            score,
                            probabilities,
                        },
                    ) if !levels.is_empty()
                        && (0.0..=(levels.len() - 1) as f64).contains(&score.0) =>
                    {
                        let probabilities = probabilities
                            .map(|mut probabilities| {
                                let invalid_rubric = || {
                                    LlmClientError::ResponseTranslation(format!(
                                        "score probabilities for {id:?} do not match its rubric"
                                    ))
                                };
                                if probabilities.len() != levels.len() {
                                    return Err(invalid_rubric());
                                }
                                // JSON keys are strings; order probabilities by the request's rubric.
                                (0..levels.len())
                                    .map(|index| {
                                        probabilities
                                            .remove(&index.to_string())
                                            .ok_or_else(invalid_rubric)
                                    })
                                    .collect()
                            })
                            .transpose()?;
                        DecisionValue::Score {
                            value: score,
                            probabilities,
                        }
                    }
                    _ => {
                        return Err(LlmClientError::ResponseTranslation(format!(
                            "answer {id:?} has an unexpected question ID, kind, or value"
                        )));
                    }
                };
                Ok((
                    id,
                    DecisionAnswer {
                        value,
                        provider_confidence: answer.confidence,
                    },
                ))
            })
            .collect::<Result<_>>()?;
        Ok(DecisionResponse {
            id: response.id,
            model: response.model,
            answers,
            usage: response.usage,
        })
    }
}

fn encode(request: &DecisionRequest) -> Result<Value> {
    let model = request
        .model
        .as_ref()
        .ok_or_else(|| LlmClientError::InvalidRequest {
            message: "System One requires a selected model".into(),
        })?;
    let mut questions = serde_json::Map::new();
    for (id, question) in &request.questions {
        let (kind, criteria) = match &question.kind {
            DecisionKind::Boolean {
                true_description,
                false_description,
            } => {
                let mut criteria = serde_json::Map::new();
                for (key, description) in [("true", true_description), ("false", false_description)]
                {
                    if let Some(description) = description {
                        criteria.insert(key.into(), description.clone());
                    }
                }
                ("noul", Value::Object(criteria))
            }
            DecisionKind::Choice { options } => {
                let mut criteria = serde_json::Map::new();
                for option in options {
                    let description = option.description.as_ref().unwrap_or(&Value::Null);
                    if criteria
                        .insert(option.id.clone(), description.clone())
                        .is_some()
                    {
                        return Err(LlmClientError::InvalidRequest {
                            message: format!("duplicate option {:?} in questions.{id}", option.id),
                        });
                    }
                }
                ("choice", Value::Object(criteria))
            }
            DecisionKind::Score { levels } => ("score", json!(levels)),
        };
        questions.insert(
            id.clone(),
            json!({
                "type": kind, "instructions": question.instructions, "criteria": criteria,
            }),
        );
    }
    Ok(json!({"model": model, "state": request.context, "questions": questions}))
}

#[derive(Deserialize)]
struct WireResponse {
    id: Option<String>,
    model: Option<ModelId>,
    answers: BTreeMap<String, WireAnswer>,
    #[serde(default)]
    usage: Usage,
}

#[derive(Deserialize)]
struct WireAnswer {
    #[serde(flatten)]
    value: WireValue,
    confidence: Option<ProviderConfidence>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WireValue {
    Noul {
        noul: Probability,
    },
    Choice {
        choice: String,
        probabilities: Option<BTreeMap<String, Probability>>,
    },
    Score {
        score: ScoreValue,
        probabilities: Option<BTreeMap<String, Probability>>,
    },
}
