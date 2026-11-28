// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::{Url, header::HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use switchyard_protocol::{
    DecisionAnswer, DecisionKind, DecisionRequest, DecisionResponse, DecisionValue,
    NativeDecisionInput, RoutedDecisionClient, WireFormat,
};

use crate::client::convert_reqwest_error;
use crate::{LlmClientError, Result, metrics};

const MAX_RESPONSE_BYTES: usize = 256 * 1024;

/// Maps a single routing choice to a Foundry project Rank request.
pub struct FoundryRankClient {
    client: reqwest::Client,
    endpoint: Url,
    bearer_token_file: PathBuf,
    model_versions: BTreeMap<String, String>,
    timeout: Duration,
}

impl FoundryRankClient {
    /// The endpoint is the complete Rank URL, including any API version.
    /// The deployment supplies and rotates the bearer token in the configured file.
    pub fn new(
        endpoint: Url,
        bearer_token_file: PathBuf,
        model_versions: BTreeMap<String, String>,
        timeout: Duration,
    ) -> Result<Self> {
        let is_loopback = endpoint.host_str().is_some_and(|host| {
            host.trim_matches(['[', ']'])
                .parse::<std::net::IpAddr>()
                .is_ok_and(|ip| ip.is_loopback())
        });
        if endpoint.scheme() != "https" && !(endpoint.scheme() == "http" && is_loopback) {
            return Err(LlmClientError::Configuration {
                message: "Foundry Rank requires an HTTPS endpoint outside loopback".into(),
            });
        }
        if !endpoint.username().is_empty() || endpoint.password().is_some() || timeout.is_zero() {
            return Err(LlmClientError::Configuration {
                message: "Foundry Rank requires a URL without credentials and a positive timeout"
                    .into(),
            });
        }
        if bearer_token_file.as_os_str().is_empty() {
            return Err(LlmClientError::Configuration {
                message: "Foundry Rank bearer_token_file must not be empty".into(),
            });
        }
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .build()
            .map_err(convert_reqwest_error)?;
        Ok(Self {
            client,
            endpoint,
            bearer_token_file,
            model_versions,
            timeout,
        })
    }
}

#[async_trait]
impl RoutedDecisionClient for FoundryRankClient {
    async fn call(&self, _request: DecisionRequest) -> Result<DecisionResponse> {
        Err(invalid_request(
            "Foundry Rank requires the original native request",
        ))
    }

    fn needs_native_input(&self) -> bool {
        true
    }

    async fn call_with_native_input(
        &self,
        request: DecisionRequest,
        native: Option<&NativeDecisionInput>,
    ) -> Result<DecisionResponse> {
        tokio::time::timeout(self.timeout, self.rank(request, native))
            .await
            .map_err(|source| LlmClientError::Timeout {
                source: Box::new(source),
            })?
    }
}

impl FoundryRankClient {
    async fn rank(
        &self,
        request: DecisionRequest,
        native: Option<&NativeDecisionInput>,
    ) -> Result<DecisionResponse> {
        let native = native
            .filter(|input| input.payload.is_object())
            .ok_or_else(|| invalid_request("Foundry Rank requires the original native request"))?;
        let (name, question) = request
            .questions
            .first_key_value()
            .filter(|_| request.questions.len() == 1)
            .ok_or_else(|| invalid_request("Foundry Rank requires one routing question"))?;
        let DecisionKind::Choice { options } = &question.kind else {
            return Err(invalid_request("Foundry Rank requires a choice question"));
        };
        let mut canonical_ids = BTreeSet::new();
        for option in options {
            let id = &option.id;
            if id.is_empty()
                || id.trim() != id
                || !canonical_ids.insert(id.to_ascii_lowercase())
                || self
                    .model_versions
                    .get(id)
                    .is_none_or(|version| version.trim().is_empty())
            {
                return Err(invalid_request(
                    "Foundry Rank requires distinct model IDs with configured versions",
                ));
            }
        }
        if options.is_empty() {
            return Err(invalid_request("Foundry Rank requires a model pool"));
        }
        let format = match native.format {
            WireFormat::OpenAiChat => "chat_completions",
            WireFormat::OpenAiResponses => "responses",
            WireFormat::AnthropicMessages => "messages",
        };
        let models: Vec<_> = options.iter().map(|option| option.id.as_str()).collect();
        let body = RankRequest {
            payload_format: format,
            payload: &native.payload,
            routing_config: RoutingConfig {
                models: &models,
                require_parameters: true,
                session_affinity: SessionAffinity { strategy: "none" },
            },
        };
        let token = tokio::fs::read_to_string(&self.bearer_token_file)
            .await
            .map_err(|_| credential_error())?;
        let token = token.trim();
        if token.is_empty() {
            return Err(credential_error());
        }
        let mut auth =
            HeaderValue::try_from(format!("Bearer {token}")).map_err(|_| credential_error())?;
        auth.set_sensitive(true);
        let mut response = self
            .client
            .post(self.endpoint.clone())
            .header("authorization", auth)
            .header("foundry-features", "Rank=V1Preview")
            .json(&body)
            .send()
            .await
            .inspect_err(|_| metrics::record_upstream_attempt(None))
            .map_err(convert_reqwest_error)?;
        let status = response.status();
        metrics::record_upstream_attempt(Some(status.as_u16()));
        if matches!(
            status,
            http::StatusCode::UNAUTHORIZED | http::StatusCode::FORBIDDEN
        ) {
            return Err(LlmClientError::Configuration {
                message: "Foundry Rank rejected the configured credentials or project access"
                    .into(),
            });
        }
        let mut headers = http::HeaderMap::new();
        if !status.is_success() {
            for name in ["retry-after", "retry-after-ms"] {
                for value in response.headers().get_all(name) {
                    headers.append(name, value.clone());
                }
            }
        }
        let mut body = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(convert_reqwest_error)? {
            if chunk.len() > MAX_RESPONSE_BYTES - body.len() {
                return Err(invalid_response("ranking response exceeded 256 KiB"));
            }
            body.extend_from_slice(&chunk);
        }
        if !status.is_success() {
            let no_models = status.as_u16() == 422
                && serde_json::from_slice::<Value>(&body).is_ok_and(|value| {
                    value.pointer("/error/code").and_then(Value::as_str)
                        == Some("no_supported_models")
                });
            return Err(LlmClientError::UpstreamHttp {
                status,
                body: if no_models {
                    json!({"error": {
                        "code": "no_supported_models",
                        "message": "Foundry Rank found no compatible model in the offered pool"
                    }})
                    .to_string()
                } else {
                    "Foundry Rank rejected the request".into()
                },
                headers: Box::new(headers),
            });
        }
        let response: RankResponse = serde_json::from_slice(&body)
            .map_err(|_| invalid_response("invalid ranking result"))?;
        if response.object != "model_router.ranking" || response.id.trim().is_empty() {
            return Err(invalid_response("invalid ranking result"));
        }
        let [winner] = &response.ranking.candidates[..] else {
            return Err(invalid_response("expected one winning model"));
        };
        let selected = models
            .iter()
            .find(|id| id.eq_ignore_ascii_case(&winner.model));
        let Some(selected) = selected.filter(|_| winner.rank == 1) else {
            return Err(invalid_response(
                "winner was not a rank-1 model from the offered pool",
            ));
        };
        if let Some(version) = &winner.version
            && self.model_versions.get(*selected) != Some(version)
        {
            return Err(invalid_response(
                "winner does not match the configured model version",
            ));
        }
        tracing::debug!(
            rank_status = response.ranking.status.as_str(),
            "Foundry Rank selected a model"
        );
        Ok(DecisionResponse {
            id: Some(response.id),
            model: None,
            usage: Default::default(),
            answers: BTreeMap::from([(
                name.clone(),
                DecisionAnswer {
                    value: DecisionValue::Choice {
                        selected: (*selected).to_owned(),
                        probabilities: None,
                    },
                    provider_confidence: None,
                },
            )]),
        })
    }
}

fn invalid_request(message: &str) -> LlmClientError {
    LlmClientError::InvalidRequest {
        message: message.into(),
    }
}

fn invalid_response(message: &str) -> LlmClientError {
    LlmClientError::ResponseTranslation(format!("Foundry Rank: {message}"))
}

fn credential_error() -> LlmClientError {
    LlmClientError::Configuration {
        message: "Foundry Rank requires a readable bearer_token_file with a valid token".into(),
    }
}

#[derive(Serialize)]
struct RankRequest<'a> {
    payload_format: &'a str,
    payload: &'a Value,
    routing_config: RoutingConfig<'a>,
}

#[derive(Serialize)]
struct RoutingConfig<'a> {
    models: &'a [&'a str],
    require_parameters: bool,
    session_affinity: SessionAffinity,
}

#[derive(Serialize)]
struct SessionAffinity {
    strategy: &'static str,
}

#[derive(Deserialize)]
struct RankResponse {
    id: String,
    object: String,
    ranking: Ranking,
}

#[derive(Deserialize)]
struct Ranking {
    status: RankStatus,
    candidates: Vec<Candidate>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum RankStatus {
    Succeeded,
    Fallback,
}

impl RankStatus {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::Fallback => "fallback",
        }
    }
}

#[derive(Deserialize)]
struct Candidate {
    rank: u32,
    model: String,
    version: Option<String>,
}
