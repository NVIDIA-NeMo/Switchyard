// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared LLM judge primitives.
//!
//! [`Judge`] owns algorithm-specific request construction and verdict parsing.
//! [`JudgeClassifier`] owns the judge model call and hands its verdict to a policy that chooses
//! the route.

use std::sync::Arc;

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde_json::Value;
use switchyard_protocol::{completion_text, AggLlmResponse};

use crate::{
    Classification, Classifier, Context, Decision, Driver, LibsyError, LlmTarget, Request, Result,
    State,
};

#[derive(Clone, Debug, Default)]
/// Prompt and structured-output contract for one judge.
pub struct JudgeConfig {
    /// Prepended instructions that define what the judge evaluates.
    pub system_prompt: String,
    /// Optional provider response format that constrains the judge verdict.
    pub response_schema: Option<Value>,
}

/// Builds and parses requests for one algorithm-specific LLM judge.
pub trait Judge: Send + Sync {
    type Verdict: DeserializeOwned + Send + Sync;

    fn build_request(&self, state: &State, request: &Request) -> Request;

    fn parse(&self, response: &AggLlmResponse) -> Result<Self::Verdict> {
        parse_json_verdict(response)
    }
}

/// Converts a parsed verdict, or an unavailable verdict, into a routing classification.
/// Consider this as a deterministic policy which can act on the signals predicted from the classifier
/// and choose the route based on the verdict.
pub trait JudgePolicy: Send + Sync {
    type Verdict: Send + Sync;

    fn classify(&self, verdict: Option<&Self::Verdict>) -> Classification;
}

/// A classifier that calls one judge target and routes through its verdict policy.
pub struct JudgeClassifier<J, P> {
    judge: J,
    target: LlmTarget,
    policy: P,
}

impl<J, P> JudgeClassifier<J, P>
where
    J: Judge,
    P: JudgePolicy<Verdict = J::Verdict>,
{
    /// Combines a judge target with a verdict policy.
    pub fn new(judge: J, target: LlmTarget, policy: P) -> Self {
        Self {
            judge,
            target,
            policy,
        }
    }
}

#[async_trait]
impl<J, P> Classifier for JudgeClassifier<J, P>
where
    J: Judge,
    P: JudgePolicy<Verdict = J::Verdict>,
{
    async fn score(
        &self,
        state: &mut State,
        request: &Request,
        driver: Option<&Driver>,
    ) -> Result<Classification> {
        // A driver is required to make the judge call. The policy owns the fail-closed fallback.
        let Some(driver) = driver else {
            return Ok(self.policy.classify(None));
        };
        let response = driver
            .call_llm_target(
                Context::default(),
                &self.target,
                self.judge.build_request(state, request),
                Arc::new(JudgeDecision {
                    model: self.target.semantic_name.clone(),
                }),
            )
            .await?;
        let aggregate = response
            .llm_response
            .into_agg()
            .await
            .map_err(|error| LibsyError::external("judge model call", error))?;
        // Bad judge JSON is not a transport failure; let the policy route its unavailable branch.
        Ok(self
            .policy
            .classify(self.judge.parse(&aggregate).ok().as_ref()))
    }
}

fn parse_json_verdict<T: DeserializeOwned>(response: &AggLlmResponse) -> Result<T> {
    // Providers sometimes wrap otherwise valid JSON in a Markdown fence.
    let completion = completion_text(response);
    serde_json::from_str(strip_json_fence(completion.trim())).map_err(|err| {
        LibsyError::AlgorithmError {
            message: format!(
                "judge reply did not parse as {}: {err}",
                std::any::type_name::<T>()
            ),
        }
    })
}

struct JudgeDecision {
    model: String,
}

impl Decision for JudgeDecision {
    fn selected_model(&self) -> &str {
        &self.model
    }

    fn reasoning(&self) -> Option<&str> {
        Some("llm judge consultation")
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn strip_json_fence(text: &str) -> &str {
    let Some(rest) = text.strip_prefix("```") else {
        return text;
    };
    let rest = rest.strip_prefix("json").unwrap_or(rest);
    let rest = rest.trim_start_matches(['\n', '\r']);
    rest.strip_suffix("```").map(str::trim).unwrap_or(rest)
}
