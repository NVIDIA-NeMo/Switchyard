// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenAI Decisions calls using the provider-neutral decision IR.

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

/// Serves typed questions through OpenAI's Decisions API.
pub struct OpenAiDecisionClient {
    client: reqwest::Client,
    endpoint: Url,
    api_key: String,
}

impl OpenAiDecisionClient {
    /// `endpoint` includes `/v1/decisions`. Each call makes one attempt, with
    /// `timeout` covering both the request and response body.
    pub fn new(endpoint: Url, api_key: String, timeout: Duration) -> Result<Self> {
        if HeaderValue::try_from(format!("Bearer {api_key}")).is_err() {
            return Err(LlmClientError::Configuration {
                message: "OpenAI Decisions API key cannot be encoded as an HTTP header".into(),
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
impl RoutedDecisionClient for OpenAiDecisionClient {
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
        let mut questions = request.questions;
        let mut answers = BTreeMap::new();
        for answer in response.answers {
            let invalid = || {
                LlmClientError::ResponseTranslation(format!(
                    "OpenAI Decisions answer {:?} has an unexpected question name, kind, or value",
                    answer.name,
                ))
            };
            let question = questions.remove(&answer.name).ok_or_else(invalid)?;
            let value = match (&question.kind, answer.value) {
                (DecisionKind::Boolean { .. }, WireValue::Predicate { probability }) => {
                    DecisionValue::Boolean(BooleanEstimate::ProbabilityTrue(probability))
                }
                (
                    DecisionKind::Choice { options },
                    WireValue::Choice {
                        choice,
                        probabilities,
                    },
                ) => {
                    let count = probabilities.len();
                    let probabilities: BTreeMap<_, _> = probabilities
                        .into_iter()
                        .map(|p| (p.value, p.probability))
                        .collect();
                    if count != options.len()
                        || probabilities.len() != options.len()
                        || !options.iter().any(|option| option.id == choice)
                        || options
                            .iter()
                            .any(|option| !probabilities.contains_key(&option.id))
                    {
                        return Err(invalid());
                    }
                    DecisionValue::Choice {
                        selected: choice,
                        probabilities: Some(probabilities),
                    }
                }
                (
                    DecisionKind::Score { levels },
                    WireValue::Score {
                        score,
                        probabilities,
                    },
                ) => {
                    let mut probabilities = probabilities;
                    probabilities.sort_by_key(|p| p.value);
                    if probabilities.len() != levels.len()
                        || probabilities
                            .iter()
                            .enumerate()
                            .any(|(index, p)| p.value != index)
                    {
                        return Err(invalid());
                    }
                    DecisionValue::Score {
                        value: score,
                        probabilities: Some(
                            probabilities.into_iter().map(|p| p.probability).collect(),
                        ),
                    }
                }
                (_, WireValue::Refusal) => {
                    return Err(LlmClientError::ResponseTranslation(format!(
                        "OpenAI Decisions refused question {:?}",
                        answer.name,
                    )));
                }
                _ => return Err(invalid()),
            };
            answers.insert(
                answer.name,
                DecisionAnswer {
                    value,
                    provider_confidence: answer.confidence,
                },
            );
        }
        if !questions.is_empty() {
            return Err(LlmClientError::ResponseTranslation(
                "OpenAI Decisions answer names do not match the requested question keys".into(),
            ));
        }
        let usage = response.usage;
        let cached = usage.input_tokens_details.cached_tokens;
        let created = usage.input_tokens_details.cache_write_tokens;
        Ok(DecisionResponse {
            id: None,
            model: Some(response.model),
            answers,
            usage: Usage {
                input_tokens: Some(
                    usage
                        .input_tokens
                        .saturating_sub(cached)
                        .saturating_sub(created),
                ),
                cache: Usage::cache_details(Some(cached), Some(created)),
                output_tokens: Some(usage.output_tokens),
                total_tokens: Some(usage.total_tokens),
                reasoning_tokens: Some(usage.output_tokens_details.reasoning_tokens),
                ..Usage::default()
            },
        })
    }
}

fn encode(request: &DecisionRequest) -> Result<Value> {
    let model = request
        .model
        .as_ref()
        .ok_or_else(|| LlmClientError::InvalidRequest {
            message: "OpenAI Decisions requires a selected model".into(),
        })?;
    let questions: Vec<_> = request
        .questions
        .iter()
        .map(|(name, question)| {
            let mut wire = json!({"name": name, "instructions": text(&question.instructions)});
            match &question.kind {
                DecisionKind::Boolean {
                    true_description,
                    false_description,
                } => {
                    wire["type"] = json!("predicate");
                    if true_description.is_some() || false_description.is_some() {
                        wire["instructions"] = json!(
                            json!({
                                "instructions": question.instructions,
                                "true": true_description, "false": false_description,
                            })
                            .to_string()
                        );
                    }
                }
                DecisionKind::Choice { options } => {
                    wire["type"] = json!("choice");
                    wire["choices"] = Value::Array(
                        options
                            .iter()
                            .map(|option| {
                                let mut choice = json!({"value": option.id});
                                if let Some(description) = &option.description {
                                    choice["description"] = json!(text(description));
                                }
                                choice
                            })
                            .collect(),
                    );
                }
                DecisionKind::Score { levels } => {
                    wire["type"] = json!("score");
                    wire["levels"] = Value::Array(
                        levels
                            .iter()
                            .map(|level| json!({"label": text(level)}))
                            .collect(),
                    );
                }
            }
            wire
        })
        .collect();
    Ok(json!({"model": model, "input": text(&request.context), "questions": questions}))
}

// OpenAI requires text; retain structured IR content as JSON rather than dropping it.
fn text(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| value.to_string())
}

#[derive(Deserialize)]
struct WireResponse {
    model: ModelId,
    answers: Vec<WireAnswer>,
    usage: WireUsage,
}

#[derive(Deserialize)]
struct WireAnswer {
    name: String,
    #[serde(flatten)]
    value: WireValue,
    confidence: Option<ProviderConfidence>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WireValue {
    Predicate {
        probability: Probability,
    },
    Choice {
        choice: String,
        probabilities: Vec<WireProbability<String>>,
    },
    Score {
        score: ScoreValue,
        probabilities: Vec<WireProbability<usize>>,
    },
    Refusal,
}

#[derive(Deserialize)]
struct WireProbability<T> {
    value: T,
    probability: Probability,
}

#[derive(Deserialize)]
struct WireUsage {
    input_tokens: u64,
    output_tokens: u64,
    total_tokens: u64,
    input_tokens_details: InputTokenDetails,
    output_tokens_details: OutputTokenDetails,
}

#[derive(Deserialize)]
struct InputTokenDetails {
    cached_tokens: u64,
    cache_write_tokens: u64,
}

#[derive(Deserialize)]
struct OutputTokenDetails {
    reasoning_tokens: u64,
}
