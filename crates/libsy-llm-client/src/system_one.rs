// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Buffered System One calls using the provider-neutral decision IR.

use std::collections::BTreeMap;
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use switchyard_protocol::{
    BooleanEstimate, DecisionAnswer, DecisionKind, DecisionRequest, DecisionResponse,
    DecisionValue, ModelId, Probability, ProviderConfidence, RoutedDecisionClient, ScoreValue,
    Usage,
};

use crate::client::convert_reqwest_error;
use crate::{LlmClientError, Result};

/// Serves decision requests through a System One endpoint, such as TypeSafe's Jev API.
pub struct SystemOneClient {
    client: reqwest::Client,
    endpoint: String,
    api_key: String,
}

impl SystemOneClient {
    /// `endpoint` is the full URL, including `/v1/systemone`. Each call makes one
    /// attempt; `timeout` covers sending the request and reading the response body.
    pub fn new(endpoint: impl Into<String>, api_key: String, timeout: Duration) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(convert_reqwest_error)?;
        Ok(Self {
            client,
            endpoint: endpoint.into(),
            api_key,
        })
    }
}

#[async_trait]
impl RoutedDecisionClient for SystemOneClient {
    async fn call(&self, request: DecisionRequest) -> Result<DecisionResponse> {
        let response = self
            .client
            .post(&self.endpoint)
            .bearer_auth(&self.api_key)
            .json(&encode(&request)?)
            .send()
            .await
            .map_err(convert_reqwest_error)?;
        let status = response.status();
        let body = response.bytes().await.map_err(convert_reqwest_error)?;
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
    content(&request.context, false, "context")?;
    let mut questions = serde_json::Map::new();
    for (id, question) in &request.questions {
        let path = format!("questions.{id}");
        content(
            &question.instructions,
            true,
            &format!("{path}.instructions"),
        )?;
        let (kind, criteria) = match &question.kind {
            DecisionKind::Boolean {
                true_description,
                false_description,
            } => {
                let mut criteria = serde_json::Map::new();
                for (key, description) in [("true", true_description), ("false", false_description)]
                {
                    if let Some(description) = description {
                        content(description, true, &format!("{path}.{key}_description"))?;
                        criteria.insert(key.into(), description.clone());
                    }
                }
                ("noul", Value::Object(criteria))
            }
            DecisionKind::Choice { options } => {
                let mut criteria = serde_json::Map::new();
                for option in options {
                    let description = option.description.as_ref().unwrap_or(&Value::Null);
                    content(description, true, &format!("{path}.options.{}", option.id))?;
                    if criteria
                        .insert(option.id.clone(), description.clone())
                        .is_some()
                    {
                        return Err(LlmClientError::InvalidRequest {
                            message: format!("duplicate option {:?} in {path}", option.id),
                        });
                    }
                }
                ("choice", Value::Object(criteria))
            }
            DecisionKind::Score { levels } => {
                for (index, level) in levels.iter().enumerate() {
                    content(level, false, &format!("{path}.levels.{index}"))?;
                }
                ("score", json!(levels))
            }
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

fn content(value: &Value, nullable: bool, path: &str) -> Result<()> {
    if value.is_string() || value.is_object() || value.is_array() || (nullable && value.is_null()) {
        return Ok(());
    }
    Err(LlmClientError::RequestEncoding(format!(
        "System One cannot represent {path} as {}",
        value
    )))
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
