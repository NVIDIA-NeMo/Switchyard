// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

mod error;
pub use error::TypeSafeClientError;

use std::collections::BTreeMap;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use switchyard_libsy::{
    TypeSafeClassifierInput, TypeSafeOption, TypeSafeProvider, TypeSafeProviderError,
    TypeSafeVerdict,
};

/// Production API root. Override with [`TypeSafeHttpClient::with_base_url`] for
/// testing or a regional deployment.
const DEFAULT_BASE_URL: &str = "https://api.typesafe.ai";
/// Model served by default. Override with [`TypeSafeHttpClient::with_model`].
const DEFAULT_MODEL: &str = "jev-latest";
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
/// Name of the single `questions` entry sent on every request, and read back from
/// `answers` in the response.
const QUESTION_NAME: &str = "route";

/// HTTP client for TypeSafe's `/v1/systemone` endpoint (the "System One Model" /
/// Jev), implementing [`TypeSafeProvider`].
///
/// Never constructed from configuration file contents: the API key always comes
/// from an environment variable via [`TypeSafeHttpClient::from_env`], and the
/// `Debug` implementation redacts it so a logged client value cannot leak it.
#[derive(Clone)]
pub struct TypeSafeHttpClient {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
}

impl std::fmt::Debug for TypeSafeHttpClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TypeSafeHttpClient")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("api_key", &"<redacted>")
            .finish()
    }
}

impl TypeSafeHttpClient {
    /// Builds a client from an API key already in hand.
    ///
    /// Prefer [`Self::from_env`] so the key never passes through configuration
    /// file parsing.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::builder()
                .timeout(DEFAULT_TIMEOUT)
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: api_key.into(),
            model: DEFAULT_MODEL.to_string(),
        }
    }

    /// Reads the API key from the named environment variable.
    ///
    /// # Errors
    ///
    /// Returns [`TypeSafeClientError::MissingApiKey`] when the variable is unset,
    /// and [`TypeSafeClientError::EmptyApiKey`] when it is set to an empty or
    /// whitespace-only value.
    pub fn from_env(variable: &str) -> Result<Self, TypeSafeClientError> {
        let api_key = std::env::var(variable).map_err(|_| TypeSafeClientError::MissingApiKey {
            variable: variable.to_string(),
        })?;
        if api_key.trim().is_empty() {
            return Err(TypeSafeClientError::EmptyApiKey {
                variable: variable.to_string(),
            });
        }
        Ok(Self::new(api_key))
    }

    /// Overrides the API root. Defaults to TypeSafe's production endpoint.
    ///
    /// # Errors
    ///
    /// Returns [`TypeSafeClientError::InvalidBaseUrl`] when `base_url` does not
    /// parse as a URL.
    pub fn with_base_url(
        mut self,
        base_url: impl Into<String>,
    ) -> Result<Self, TypeSafeClientError> {
        let base_url = base_url.into();
        if let Err(error) = reqwest::Url::parse(&base_url) {
            return Err(TypeSafeClientError::InvalidBaseUrl {
                base_url,
                reason: error.to_string(),
            });
        }
        self.base_url = base_url;
        Ok(self)
    }

    /// Overrides the model name sent as `model`. Defaults to `"jev-latest"`.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    fn endpoint(&self) -> String {
        format!("{}/v1/systemone", self.base_url.trim_end_matches('/'))
    }
}

#[derive(Serialize)]
struct SystemOneRequest<'a> {
    state: &'a str,
    model: &'a str,
    questions: BTreeMap<&'static str, ChoiceQuestion<'a>>,
}

#[derive(Serialize)]
struct ChoiceQuestion<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    instructions: &'a str,
    criteria: BTreeMap<&'a str, &'a str>,
}

#[derive(Deserialize, Default)]
struct Usage {
    #[serde(default)]
    #[allow(dead_code)]
    // Not surfaced today; kept so a wire-shape mismatch fails to parse loudly.
    input_tokens: u64,
}

#[derive(Deserialize)]
struct ChoiceAnswer {
    choice: String,
    confidence: f64,
}

#[derive(Deserialize)]
struct SystemOneResponse {
    #[serde(default)]
    #[allow(dead_code)]
    usage: Usage,
    answers: BTreeMap<String, ChoiceAnswer>,
}

#[async_trait]
impl TypeSafeProvider for TypeSafeHttpClient {
    async fn classify(
        &self,
        input: TypeSafeClassifierInput,
        options: &[TypeSafeOption],
    ) -> Result<TypeSafeVerdict, TypeSafeProviderError> {
        let criteria = options
            .iter()
            .map(|option| (option.label.as_str(), option.description.as_str()))
            .collect::<BTreeMap<_, _>>();
        let mut questions = BTreeMap::new();
        questions.insert(
            QUESTION_NAME,
            ChoiceQuestion {
                kind: "choice",
                instructions: &input.question,
                criteria,
            },
        );
        let body = SystemOneRequest {
            state: &input.context,
            model: &self.model,
            questions,
        };

        let response = self
            .http
            .post(self.endpoint())
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|error| TypeSafeProviderError(format!("typesafe request failed: {error}")))?;

        let status = response.status();
        if !status.is_success() {
            // Deliberately drop the response body: it is provider-controlled and this error
            // is surfaced through logs and telemetry. `state` in the request can carry the
            // caller's own prompt content, so an error body that happened to echo any of it
            // back must not flow into our own logs. The status code alone is enough to
            // diagnose a provider-side failure.
            return Err(TypeSafeProviderError(format!(
                "typesafe returned HTTP {status}"
            )));
        }

        let parsed: SystemOneResponse = response.json().await.map_err(|error| {
            TypeSafeProviderError(format!("typesafe response was not valid JSON: {error}"))
        })?;

        let answer = parsed.answers.get(QUESTION_NAME).ok_or_else(|| {
            TypeSafeProviderError(format!(
                "typesafe response is missing the {QUESTION_NAME:?} answer"
            ))
        })?;

        Ok(TypeSafeVerdict {
            label: answer.choice.clone(),
            confidence: answer.confidence,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{body_partial_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn options() -> Vec<TypeSafeOption> {
        vec![
            TypeSafeOption::new("capable", "complex, multi-step work"),
            TypeSafeOption::new("efficient", "short, simple requests"),
        ]
    }

    fn input() -> TypeSafeClassifierInput {
        TypeSafeClassifierInput {
            question: "Which tier does this need?".to_string(),
            context: "[user] list files in a directory".to_string(),
        }
    }

    #[test]
    fn from_env_reports_a_missing_variable() {
        let error = TypeSafeHttpClient::from_env("SWITCHYARD_TYPESAFE_TEST_UNSET_VAR");
        assert!(matches!(
            error,
            Err(TypeSafeClientError::MissingApiKey { variable })
                if variable == "SWITCHYARD_TYPESAFE_TEST_UNSET_VAR"
        ));
    }

    #[test]
    fn debug_output_redacts_the_api_key() {
        let client = TypeSafeHttpClient::new("sk-super-secret");
        let rendered = format!("{client:?}");
        assert!(!rendered.contains("sk-super-secret"));
        assert!(rendered.contains("<redacted>"));
    }

    #[test]
    fn invalid_base_url_is_rejected() {
        let result = TypeSafeHttpClient::new("key").with_base_url("not a url");
        assert!(matches!(
            result,
            Err(TypeSafeClientError::InvalidBaseUrl { .. })
        ));
    }

    #[tokio::test]
    async fn classify_parses_a_successful_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/systemone"))
            .and(header("Authorization", "Bearer sk-test"))
            .and(body_partial_json(serde_json::json!({
                "state": "[user] list files in a directory",
                "model": "jev-latest",
                "questions": {
                    "route": {
                        "type": "choice",
                        "instructions": "Which tier does this need?",
                        "criteria": {
                            "capable": "complex, multi-step work",
                            "efficient": "short, simple requests"
                        }
                    }
                }
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "usage": {"input_tokens": 42},
                "answers": {"route": {"choice": "efficient", "confidence": 0.87}}
            })))
            .mount(&server)
            .await;

        let client = TypeSafeHttpClient::new("sk-test")
            .with_base_url(server.uri())
            .expect("mock server URI is valid");

        let verdict = client
            .classify(input(), &options())
            .await
            .expect("classify should succeed");

        assert_eq!(verdict.label, "efficient");
        assert!((verdict.confidence - 0.87).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn classify_surfaces_a_non_success_status_without_leaking_the_key() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/systemone"))
            .respond_with(ResponseTemplate::new(401).set_body_string("unauthorized"))
            .mount(&server)
            .await;

        let client = TypeSafeHttpClient::new("sk-test")
            .with_base_url(server.uri())
            .expect("mock server URI is valid");

        let error = client
            .classify(input(), &options())
            .await
            .expect_err("classify should fail on HTTP 401");

        let message = error.to_string();
        assert!(message.contains("401"));
        assert!(!message.contains("sk-test"));
    }

    #[tokio::test]
    async fn classify_errors_when_the_answer_is_missing() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/systemone"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "usage": {"input_tokens": 1},
                "answers": {}
            })))
            .mount(&server)
            .await;

        let client = TypeSafeHttpClient::new("sk-test")
            .with_base_url(server.uri())
            .expect("mock server URI is valid");

        let error = client
            .classify(input(), &options())
            .await
            .expect_err("classify should fail when the answer is absent");
        assert!(error.to_string().contains("route"));
    }

    #[test]
    fn with_model_overrides_the_default() {
        let client = TypeSafeHttpClient::new("sk-test").with_model("jev-pinned");
        assert_eq!(client.model, "jev-pinned");
    }
}
