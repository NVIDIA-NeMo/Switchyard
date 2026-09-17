// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

mod error;
pub use error::TypeSafeClientError;

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::ser::SerializeMap;
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
const QUESTION_PREFIX: &str = "route_";

/// HTTP client for TypeSafe's `/v1/systemone` endpoint (the "System One Model" /
/// Jev), implementing [`TypeSafeProvider`].
///
/// `switchyard-runner` reads the API key from an environment variable through
/// [`TypeSafeHttpClient::from_env`]. Direct library users may instead use
/// [`TypeSafeHttpClient::new`]. The `Debug` implementation redacts the key.
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
    /// parse as a URL or does not use HTTPS.
    pub fn with_base_url(
        mut self,
        base_url: impl Into<String>,
    ) -> Result<Self, TypeSafeClientError> {
        let base_url = base_url.into();
        let parsed = reqwest::Url::parse(&base_url).map_err(|error| {
            TypeSafeClientError::InvalidBaseUrl {
                base_url: base_url.clone(),
                reason: error.to_string(),
            }
        })?;
        if parsed.scheme() != "https" {
            return Err(TypeSafeClientError::InvalidBaseUrl {
                base_url,
                reason: "scheme must be https".to_string(),
            });
        }
        self.base_url = base_url;
        Ok(self)
    }

    #[cfg(test)]
    fn with_base_url_for_test(
        mut self,
        base_url: impl Into<String>,
    ) -> Result<Self, TypeSafeClientError> {
        let base_url = base_url.into();
        reqwest::Url::parse(&base_url).map_err(|error| TypeSafeClientError::InvalidBaseUrl {
            base_url: base_url.clone(),
            reason: error.to_string(),
        })?;
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
    questions: BTreeMap<String, ChoiceQuestion<'a>>,
}

#[derive(Serialize)]
struct ChoiceQuestion<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    instructions: &'a str,
    criteria: OrderedCriteria<'a>,
}

struct OrderedCriteria<'a>(Vec<(&'a str, &'a str)>);

impl Serialize for OrderedCriteria<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (label, description) in &self.0 {
            map.serialize_entry(label, description)?;
        }
        map.end()
    }
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
    probabilities: BTreeMap<String, f64>,
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
        if options.is_empty() {
            return Err(TypeSafeProviderError(
                "typesafe classification needs at least one candidate".to_string(),
            ));
        }
        let unique_labels = options
            .iter()
            .map(|option| option.label.as_str())
            .collect::<BTreeSet<_>>();
        if unique_labels.len() != options.len() {
            return Err(TypeSafeProviderError(
                "typesafe classification requires unique candidate labels".to_string(),
            ));
        }
        let orders = option_orders(options);
        let mut questions = BTreeMap::new();
        for (index, order) in orders.iter().enumerate() {
            questions.insert(
                format!("{QUESTION_PREFIX}{index}"),
                ChoiceQuestion {
                    kind: "choice",
                    instructions: &input.question,
                    criteria: OrderedCriteria(
                        order
                            .iter()
                            .map(|option| (option.label.as_str(), option.description.as_str()))
                            .collect(),
                    ),
                },
            );
        }
        let body = SystemOneRequest {
            state: &input.context,
            model: &self.model,
            questions,
        };

        let started = Instant::now();
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

        let mut probabilities = options
            .iter()
            .map(|option| (option.label.clone(), 0.0))
            .collect::<BTreeMap<_, _>>();
        for index in 0..orders.len() {
            let question = format!("{QUESTION_PREFIX}{index}");
            let answer = parsed.answers.get(&question).ok_or_else(|| {
                TypeSafeProviderError(format!(
                    "typesafe response is missing the {question:?} answer"
                ))
            })?;
            if !probabilities.contains_key(&answer.choice) {
                return Err(TypeSafeProviderError(format!(
                    "typesafe returned unknown candidate {:?}",
                    answer.choice
                )));
            }
            if !(0.0..=1.0).contains(&answer.confidence) {
                return Err(TypeSafeProviderError(
                    "typesafe returned an invalid confidence".to_string(),
                ));
            }
            let mut answer_sum = 0.0;
            for (label, total) in &mut probabilities {
                let value = answer.probabilities.get(label).ok_or_else(|| {
                    TypeSafeProviderError(format!(
                        "typesafe response is missing probability for {label:?}"
                    ))
                })?;
                if !(0.0..=1.0).contains(value) {
                    return Err(TypeSafeProviderError(format!(
                        "typesafe returned an invalid probability for {label:?}"
                    )));
                }
                answer_sum += value;
                *total += value;
            }
            if !answer_sum.is_finite() || (answer_sum - 1.0).abs() > 0.02 {
                return Err(TypeSafeProviderError(format!(
                    "typesafe probabilities for {question:?} do not sum to one"
                )));
            }
        }
        for probability in probabilities.values_mut() {
            *probability /= orders.len() as f64;
        }
        let probability_sum = probabilities.values().sum::<f64>();
        if !probability_sum.is_finite() || (probability_sum - 1.0).abs() > 0.02 {
            return Err(TypeSafeProviderError(
                "typesafe probabilities do not sum to one".to_string(),
            ));
        }
        for probability in probabilities.values_mut() {
            *probability /= probability_sum;
        }
        let selected = options
            .iter()
            .reduce(|best, candidate| {
                if probabilities[&candidate.label] > probabilities[&best.label] {
                    candidate
                } else {
                    best
                }
            })
            .ok_or_else(|| {
                TypeSafeProviderError(
                    "typesafe classification needs at least one candidate".to_string(),
                )
            })?;
        let maximum = probabilities[&selected.label];
        let uniform = 1.0 / options.len() as f64;
        let confidence = if options.len() == 1 {
            1.0
        } else {
            ((maximum - uniform) / (1.0 - uniform)).clamp(0.0, 1.0)
        };

        Ok(TypeSafeVerdict {
            label: selected.label.clone(),
            confidence,
            probabilities,
            decision_latency_ms: started.elapsed().as_millis().try_into().unwrap_or(u64::MAX),
        })
    }
}

// Return at most three deterministic, non-duplicate orders to reduce routing bias.
fn option_orders(options: &[TypeSafeOption]) -> Vec<Vec<&TypeSafeOption>> {
    let mut orders = vec![options.iter().collect::<Vec<_>>()];
    if options.len() > 1 {
        let mut rotated = options.iter().collect::<Vec<_>>();
        rotated.rotate_left(1);
        orders.push(rotated);
        let mut reversed = options.iter().collect::<Vec<_>>();
        reversed.reverse();
        if !orders.iter().any(|order| {
            order
                .iter()
                .map(|option| &option.label)
                .eq(reversed.iter().map(|option| &option.label))
        }) {
            orders.push(reversed);
        }
    }
    orders
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

    #[test]
    fn plain_http_base_url_is_rejected() {
        let result = TypeSafeHttpClient::new("key").with_base_url("http://typesafe.example");
        assert!(matches!(
            result,
            Err(TypeSafeClientError::InvalidBaseUrl { reason, .. })
                if reason == "scheme must be https"
        ));
    }

    #[tokio::test]
    async fn classify_rejects_duplicate_candidate_labels() {
        let options = [
            TypeSafeOption::new("duplicate", "first description"),
            TypeSafeOption::new("duplicate", "second description"),
        ];
        let error = TypeSafeHttpClient::new("sk-test")
            .classify(input(), &options)
            .await
            .expect_err("duplicate labels should fail before the request");

        assert!(error.to_string().contains("unique candidate labels"));
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
                    "route_0": {
                        "type": "choice",
                        "instructions": "Which tier does this need?",
                        "criteria": {
                            "capable": "complex, multi-step work",
                            "efficient": "short, simple requests"
                        }
                    },
                    "route_1": {
                        "type": "choice",
                        "instructions": "Which tier does this need?",
                        "criteria": {
                            "efficient": "short, simple requests",
                            "capable": "complex, multi-step work"
                        }
                    }
                }
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "usage": {"input_tokens": 42},
                "answers": {
                    "route_0": {
                        "choice": "efficient",
                        "confidence": 0.6,
                        "probabilities": {"capable": 0.2, "efficient": 0.8}
                    },
                    "route_1": {
                        "choice": "capable",
                        "confidence": 0.2,
                        "probabilities": {"capable": 0.6, "efficient": 0.4}
                    }
                }
            })))
            .mount(&server)
            .await;

        let client = TypeSafeHttpClient::new("sk-test")
            .with_base_url_for_test(server.uri())
            .expect("mock server URI is valid");

        let verdict = client
            .classify(input(), &options())
            .await
            .expect("classify should succeed");

        assert_eq!(verdict.label, "efficient");
        assert!((verdict.confidence - 0.2).abs() < 1e-12);
        assert!((verdict.probabilities["capable"] - 0.4).abs() < 1e-12);
        assert!((verdict.probabilities["efficient"] - 0.6).abs() < 1e-12);
        assert!(verdict.decision_latency_ms < 10_000);
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
            .with_base_url_for_test(server.uri())
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
            .with_base_url_for_test(server.uri())
            .expect("mock server URI is valid");

        let error = client
            .classify(input(), &options())
            .await
            .expect_err("classify should fail when the answer is absent");
        assert!(error.to_string().contains("route"));
    }

    #[tokio::test]
    async fn classify_rejects_a_probability_distribution_that_does_not_sum_to_one() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/systemone"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "usage": {"input_tokens": 1},
                "answers": {
                    "route_0": {
                        "choice": "capable",
                        "confidence": 0.5,
                        "probabilities": {"capable": 0.8, "efficient": 0.8}
                    },
                    "route_1": {
                        "choice": "efficient",
                        "confidence": 0.5,
                        "probabilities": {"capable": 0.2, "efficient": 0.8}
                    }
                }
            })))
            .mount(&server)
            .await;

        let client = TypeSafeHttpClient::new("sk-test")
            .with_base_url_for_test(server.uri())
            .expect("mock server URI is valid");
        let error = client
            .classify(input(), &options())
            .await
            .expect_err("classify should reject a malformed distribution");

        assert!(error.to_string().contains("do not sum to one"));
    }

    #[test]
    fn option_orders_cover_stable_permutations_without_duplicates() {
        let options = vec![
            TypeSafeOption::new("one", "first"),
            TypeSafeOption::new("two", "second"),
            TypeSafeOption::new("three", "third"),
        ];
        let orders = option_orders(&options)
            .into_iter()
            .map(|order| {
                order
                    .into_iter()
                    .map(|option| option.label.as_str())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            orders,
            [
                vec!["one", "two", "three"],
                vec!["two", "three", "one"],
                vec!["three", "two", "one"],
            ]
        );
    }

    #[test]
    fn with_model_overrides_the_default() {
        let client = TypeSafeHttpClient::new("sk-test").with_model("jev-pinned");
        assert_eq!(client.model, "jev-pinned");
    }
}
