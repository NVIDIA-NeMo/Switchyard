// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! An LLM-client-style wrapper over the optimizer interfaces.
//!
//! [`RoutedClient`] hides the `feed` / `optimize` loop behind a single
//! `complete` call, turning routing (and any other optimization) into a black
//! box that can stand in for an ordinary LLM client. The actual model call is
//! delegated to a [`ModelCaller`]: integrators plug in their own transport, or
//! use the built-in [`HttpCaller`] against any OpenAI-compatible endpoint.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::error::Error;

use crate::{
    AgentApiOptAlgorithm, AgentApiOptInput, AgentApiRequest, AgentApiResponse, Decision,
    EnrichmentData,
};

/// Performs a single model call. This is the one piece of I/O the client does
/// not own — integrators implement it with their own HTTP/transport stack, or
/// use [`HttpCaller`].
#[async_trait]
pub trait ModelCaller: Send + Sync {
    /// Call the model named by `request.model` with `request.prompt`, returning
    /// its completion.
    async fn call(&self, request: AgentApiRequest) -> Result<AgentApiResponse, Box<dyn Error>>;
}

/// A blackbox, LLM-client-style front end over an optimization algorithm.
///
/// Each [`complete`](RoutedClient::complete) mints a fresh optimizer for the
/// request, drives the `feed` -> `optimize` loop to completion — performing
/// every model call the optimizer asks for via the [`ModelCaller`] — and
/// returns the final model response. Single-round routers (weighted random) and
/// multi-round ones (LLM classifier) are handled by the same loop; the caller
/// sees only "request in, response out".
pub struct RoutedClient<D> {
    algorithm: Box<dyn AgentApiOptAlgorithm<D>>,
    caller: Box<dyn ModelCaller>,
}

impl<D> RoutedClient<D> {
    /// Build a client from an algorithm and a model caller.
    pub fn new(algorithm: Box<dyn AgentApiOptAlgorithm<D>>, caller: Box<dyn ModelCaller>) -> Self {
        RoutedClient { algorithm, caller }
    }

    /// Build a client that makes model calls with the built-in [`HttpCaller`]
    /// against an OpenAI-compatible endpoint at `base_url`.
    pub fn with_http(
        algorithm: Box<dyn AgentApiOptAlgorithm<D>>,
        base_url: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        RoutedClient::new(algorithm, Box::new(HttpCaller::new(base_url, api_key)))
    }

    /// Optimize and run `request` to completion, returning the final response.
    pub async fn complete(
        &self,
        request: AgentApiRequest,
    ) -> Result<AgentApiResponse, Box<dyn Error>> {
        self.complete_with(request, EnrichmentData::default()).await
    }

    /// Like [`complete`](RoutedClient::complete), but attaches `enrichment`
    /// (session/agent/task correlation) to every input fed to the optimizer.
    pub async fn complete_with(
        &self,
        request: AgentApiRequest,
        enrichment: EnrichmentData,
    ) -> Result<AgentApiResponse, Box<dyn Error>> {
        let mut optimizer = self.algorithm.optimizer();
        optimizer
            .feed(AgentApiOptInput::Request(request), enrichment.clone())
            .await?;

        // Round-agnostic drive loop: keep performing the model calls the
        // optimizer asks for and feeding their responses back until it returns.
        let mut last: Option<AgentApiResponse> = None;
        while let Decision::ModelInference(response) = optimizer.optimize().await? {
            for req in response.requests {
                let model = req.model.clone();
                let result = self.caller.call(req).await?;
                // The Response input variant carries a request-shaped struct
                // whose `prompt` holds the completion text.
                optimizer
                    .feed(
                        AgentApiOptInput::Response(AgentApiRequest {
                            prompt: result.completion.clone(),
                            model,
                        }),
                        enrichment.clone(),
                    )
                    .await?;
                last = Some(result);
            }
        }

        last.ok_or_else(|| "optimizer returned before requesting any model call".into())
    }
}

/// Default [`ModelCaller`]: POSTs to `{base_url}/chat/completions` on any
/// OpenAI-compatible endpoint, sending the prompt as a single user message.
pub struct HttpCaller {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
}

impl HttpCaller {
    /// Create a caller targeting `base_url` (e.g. `https://api.openai.com/v1`)
    /// with the given bearer `api_key`.
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        HttpCaller {
            client: reqwest::Client::new(),
            base_url: base_url.into(),
            api_key: api_key.into(),
        }
    }
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct ChatCompletionRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: String,
}

#[async_trait]
impl ModelCaller for HttpCaller {
    async fn call(&self, request: AgentApiRequest) -> Result<AgentApiResponse, Box<dyn Error>> {
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let payload = ChatCompletionRequest {
            model: &request.model,
            messages: vec![ChatMessage {
                role: "user",
                content: &request.prompt,
            }],
        };
        let response = self
            .client
            .post(url)
            .bearer_auth(&self.api_key)
            .json(&payload)
            .send()
            .await?
            .error_for_status()?
            .json::<ChatCompletionResponse>()
            .await?;
        let completion = response
            .choices
            .into_iter()
            .next()
            .map(|choice| choice.message.content)
            .ok_or("model response contained no choices")?;
        Ok(AgentApiResponse { completion })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_class::LlmClassifierAlgorithm;
    use crate::rand::{RandomRouterAlgorithm, WeightedModel};
    use std::sync::{Arc, Mutex};

    /// A no-network caller that records requests and returns scripted answers: a
    /// parseable score for the classifier model, an answer tagged with the model
    /// otherwise.
    struct RecordingCaller {
        classifier_model: String,
        calls: Arc<Mutex<Vec<AgentApiRequest>>>,
    }

    #[async_trait]
    impl ModelCaller for RecordingCaller {
        async fn call(&self, request: AgentApiRequest) -> Result<AgentApiResponse, Box<dyn Error>> {
            let completion = if request.model == self.classifier_model {
                "0.9".to_string()
            } else {
                format!("answer from {}", request.model)
            };
            self.calls
                .lock()
                .map_err(|_| "recording lock poisoned")?
                .push(request);
            Ok(AgentApiResponse { completion })
        }
    }

    #[tokio::test]
    async fn rand_client_routes_once_and_returns_the_response() -> Result<(), Box<dyn Error>> {
        let algorithm = RandomRouterAlgorithm {
            models: vec![WeightedModel::new("frontier/model", 1.0)],
            rng_seed: Some(7),
        };
        let calls = Arc::new(Mutex::new(Vec::new()));
        let caller = RecordingCaller {
            classifier_model: "unused".to_string(),
            calls: Arc::clone(&calls),
        };
        let client = RoutedClient::new(Box::new(algorithm), Box::new(caller));

        let response = client
            .complete(AgentApiRequest {
                prompt: "hi".to_string(),
                model: "auto".to_string(),
            })
            .await?;

        let recorded = calls.lock().map_err(|_| "lock poisoned")?;
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].model, "frontier/model");
        assert_eq!(response.completion, "answer from frontier/model");
        Ok(())
    }

    #[tokio::test]
    async fn classifier_client_drives_multi_round_and_returns_routed_response(
    ) -> Result<(), Box<dyn Error>> {
        let algorithm = LlmClassifierAlgorithm {
            classifier_model: "router/classifier".to_string(),
            strong_model: "frontier/model".to_string(),
            weak_model: "cheap/model".to_string(),
            threshold: 0.5,
        };
        let calls = Arc::new(Mutex::new(Vec::new()));
        let caller = RecordingCaller {
            classifier_model: "router/classifier".to_string(),
            calls: Arc::clone(&calls),
        };
        let client = RoutedClient::new(Box::new(algorithm), Box::new(caller));

        let response = client
            .complete(AgentApiRequest {
                prompt: "prove it".to_string(),
                model: "auto".to_string(),
            })
            .await?;

        let recorded = calls.lock().map_err(|_| "lock poisoned")?;
        // Two model calls: the classifier, then the routed target.
        assert_eq!(recorded.len(), 2);
        assert_eq!(recorded[0].model, "router/classifier");
        assert!(recorded[0].prompt.contains("prove it"));
        assert_eq!(recorded[1].model, "frontier/model"); // score 0.9 >= 0.5 -> strong
                                                         // The returned response is the routed call's, not the classifier's.
        assert_eq!(response.completion, "answer from frontier/model");
        Ok(())
    }
}
