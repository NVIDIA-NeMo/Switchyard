// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Choice decisions from a vLLM model's constrained next-token distribution.

use std::{collections::BTreeMap, time::Duration};

use async_trait::async_trait;
use reqwest::{Url, header::HeaderValue};
use serde_json::{Value, json};
use switchyard_protocol::{
    DecisionAnswer, DecisionKind, DecisionRequest, DecisionResponse, DecisionValue, ModelId,
    Probability, RoutedDecisionClient, Usage,
};

use crate::{LlmClientError, Result, client::convert_reqwest_error, metrics};

/// Reads one choice question using labels A–Z, retaining the complete label distribution.
/// The vLLM server must allow at least 128 log-probabilities per generated token.
pub struct VllmChoiceClient {
    client: reqwest::Client,
    endpoint: Url,
    api_key: String,
}

impl VllmChoiceClient {
    /// `endpoint` is the complete Chat Completions URL. Requests use one generated token.
    pub fn new(endpoint: Url, api_key: String, timeout: Duration) -> Result<Self> {
        if HeaderValue::try_from(format!("Bearer {api_key}")).is_err() {
            return Err(LlmClientError::Configuration {
                message: "vLLM choice API key cannot be encoded as an HTTP header".into(),
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
impl RoutedDecisionClient for VllmChoiceClient {
    async fn call(&self, request: DecisionRequest) -> Result<DecisionResponse> {
        let invalid_request = || LlmClientError::InvalidRequest {
            message: "vllm_choice requires a model and one Choice question with 1..=26 options"
                .into(),
        };
        let model = request.model.as_ref().ok_or_else(invalid_request)?;
        let (name, question) = request
            .questions
            .first_key_value()
            .filter(|_| request.questions.len() == 1)
            .ok_or_else(invalid_request)?;
        let DecisionKind::Choice { options } = &question.kind else {
            return Err(invalid_request());
        };
        if options.is_empty() || options.len() > 26 {
            return Err(invalid_request());
        }
        let labels: Vec<_> = ('A'..='Z')
            .take(options.len())
            .map(|label| label.to_string())
            .collect();
        let choices: Vec<_> = labels.iter().zip(options).map(|(label, option)| json!({"label": label, "id": option.id, "description": option.description})).collect();
        let prompt = format!(
            "Follow the question's instructions and choose one allowed option. Reply with only its letter label.\n{}",
            json!({"context": request.context, "instructions": question.instructions, "options": choices})
        );
        let response = self
            .client
            .post(self.endpoint.clone())
            .bearer_auth(&self.api_key)
            .json(&json!({
                "model": model, "messages": [{"role": "user", "content": prompt}],
                "structured_outputs": {"choice": labels}, "temperature": 0,
                "max_tokens": 1, "logprobs": true, "top_logprobs": 128,
            }))
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
        let response: Value =
            serde_json::from_slice(&body).map_err(|source| LlmClientError::InvalidResponse {
                source: Box::new(source),
            })?;
        let invalid = || {
            LlmClientError::ResponseTranslation(
                "vLLM did not return the selected label and all choice log-probabilities".into(),
            )
        };
        let choice = &response["choices"][0];
        let selected = choice["message"]["content"]
            .as_str()
            .and_then(|selected| labels.iter().position(|label| label == selected.trim()))
            .ok_or_else(invalid)?;
        let logprobs = choice["logprobs"]["content"][0]["top_logprobs"]
            .as_array()
            .ok_or_else(invalid)?;
        let masses = labels
            .iter()
            .map(|label| {
                let values: Vec<_> = logprobs
                    .iter()
                    .filter(|entry| entry["token"].as_str() == Some(label))
                    .filter_map(|entry| entry["logprob"].as_f64())
                    .collect();
                if values.is_empty() {
                    return Err(invalid());
                }
                // Several token IDs can decode to the same letter.
                Ok(values.into_iter().map(f64::exp).sum::<f64>())
            })
            .collect::<Result<Vec<_>>>()?;
        let total: f64 = masses.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            return Err(invalid());
        }
        let probabilities = options
            .iter()
            .zip(masses)
            .map(|(option, mass)| (option.id.clone(), Probability(mass / total)))
            .collect();
        let usage = &response["usage"];
        let cached = usage["prompt_tokens_details"]["cached_tokens"].as_u64();
        Ok(DecisionResponse {
            id: response["id"].as_str().map(str::to_owned),
            model: response["model"].as_str().map(ModelId::from),
            answers: BTreeMap::from([(
                name.clone(),
                DecisionAnswer {
                    value: DecisionValue::Choice {
                        selected: options[selected].id.clone(),
                        probabilities: Some(probabilities),
                    },
                    provider_confidence: None,
                },
            )]),
            usage: Usage {
                input_tokens: usage["prompt_tokens"]
                    .as_u64()
                    .map(|tokens| tokens.saturating_sub(cached.unwrap_or(0))),
                cache: Usage::cache_details(cached, None),
                output_tokens: usage["completion_tokens"].as_u64(),
                total_tokens: usage["total_tokens"].as_u64(),
                ..Usage::default()
            },
        })
    }
}
