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

    fn to_classification(&self, verdict: Option<&Self::Verdict>) -> Classification;
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

    /// Consults the judge, yielding `None` when it is unavailable or unintelligible.
    ///
    /// A judge is an optimization, not a dependency: failing the caller's request because the
    /// judge is down would be worse than routing without it, so every failure — transport,
    /// mid-stream, or unparseable reply — is logged and folded into `None` for the policy's
    /// fail-closed branch. A closed driver stream is folded too; the algorithm's next driver
    /// call surfaces it, so nothing is masked.
    async fn verdict(
        &self,
        state: &mut State,
        request: &Request,
        driver: &Driver,
    ) -> Option<J::Verdict> {
        let judge_model = self.target.semantic_name.as_str();
        let warn = |error: &dyn std::fmt::Display| {
            tracing::warn!(
                target: "libsy",
                judge_model,
                error = %error,
                "judge verdict unavailable; routing without one"
            );
        };

        let response = driver
            .call_llm_target(
                Context::default(),
                &self.target,
                self.judge.build_request(state, request),
                Arc::new(JudgeDecision {
                    model: self.target.semantic_name.to_string(),
                }),
            )
            .await
            .inspect_err(|error| warn(error))
            .ok()?;
        let aggregate = response
            .llm_response
            .into_agg()
            .await
            .inspect_err(|error| warn(error))
            .ok()?;
        self.judge
            .parse(&aggregate)
            .inspect_err(|error| warn(error))
            .ok()
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
        // A missing driver is a broken composition, not an unavailable judge.
        let Some(driver) = driver else {
            return Err(LibsyError::AlgorithmError {
                message: format!(
                    "judge classifier for target {:?} requires a driver to call it",
                    self.target.semantic_name
                ),
            });
        };
        let verdict = self.verdict(state, request, driver).await;
        Ok(self.policy.to_classification(verdict.as_ref()))
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

#[cfg(test)]
mod tests {
    use super::*;

    use futures::StreamExt;
    use serde::Deserialize;
    use switchyard_protocol::{text_request, text_response, LlmClientError};

    use crate::{LlmResponse, LlmResponseChunk, Response, Score, Step};

    const VERDICT: &str = r#"{"ok":true}"#;

    #[derive(Debug, Deserialize, PartialEq)]
    struct TestVerdict {
        ok: bool,
    }

    struct TestJudge;

    impl Judge for TestJudge {
        type Verdict = TestVerdict;

        fn build_request(&self, _state: &State, request: &Request) -> Request {
            request.clone()
        }
    }

    /// Reports only whether a verdict arrived.
    struct TestPolicy;

    impl JudgePolicy for TestPolicy {
        type Verdict = TestVerdict;

        fn to_classification(&self, verdict: Option<&Self::Verdict>) -> Classification {
            let target = if verdict.is_some() {
                "verdict"
            } else {
                "no-verdict"
            };
            Classification::Scores(vec![Score {
                target: target.to_string(),
                confidence: 1.0,
            }])
        }
    }

    fn classifier() -> JudgeClassifier<TestJudge, TestPolicy> {
        JudgeClassifier::new(
            TestJudge,
            LlmTarget {
                semantic_name: "judge".to_string(),
                llm_client: None,
            },
            TestPolicy,
        )
    }

    fn request() -> Request {
        Request {
            llm_request: text_request(Some("auto".to_string()), "judge this"),
            raw_request: None,
            metadata: None,
        }
    }

    fn buffered(completion: &str) -> Response {
        Response {
            llm_response: LlmResponse::Agg(text_response(None, completion)),
            metadata: None,
        }
    }

    fn streamed(chunks: Vec<LlmResponseChunk>) -> Response {
        Response {
            llm_response: LlmResponse::Stream(
                futures::stream::iter(chunks.into_iter().map(Ok)).boxed(),
            ),
            metadata: None,
        }
    }

    fn streamed_then_failing(chunk: LlmResponseChunk) -> Response {
        let items = futures::stream::iter([
            Ok(chunk),
            Err(LlmClientError::Timeout {
                source: Box::new(std::io::Error::other("stream died")),
            }),
        ]);
        Response {
            llm_response: LlmResponse::Stream(items.boxed()),
            metadata: None,
        }
    }

    fn selected(classification: Classification) -> Result<String> {
        classification
            .argmax(false)?
            .map(|score| score.target)
            .ok_or_else(|| LibsyError::AlgorithmError {
                message: "policy abstained".to_string(),
            })
    }

    /// Serves the single offloaded judge call with `reply`. The stream is taken first
    /// because the driver refuses to publish a step until a consumer exists.
    async fn score_served_with(reply: Result<Response>) -> Result<String> {
        let driver = Driver::new();
        let mut steps = Box::pin(driver.stream());
        let classifier = classifier();
        let mut state = State::default();
        let request = request();

        let serve = async {
            if let Some(Ok(Step::CallLlm(call))) = steps.next().await {
                let _ = call.respond(reply);
            }
        };
        let (classification, ()) =
            tokio::join!(classifier.score(&mut state, &request, Some(&driver)), serve);
        selected(classification?)
    }

    #[tokio::test]
    async fn a_buffered_verdict_reaches_the_policy() -> Result<()> {
        assert_eq!(score_served_with(Ok(buffered(VERDICT))).await?, "verdict");
        Ok(())
    }

    #[tokio::test]
    async fn a_streamed_verdict_is_drained_before_parsing() -> Result<()> {
        let chunks = VERDICT
            .chars()
            .map(|character| LlmResponseChunk::TextDelta {
                index: 0,
                text: character.to_string(),
            })
            .collect();
        assert_eq!(score_served_with(Ok(streamed(chunks))).await?, "verdict");
        Ok(())
    }

    #[tokio::test]
    async fn an_in_band_stream_error_falls_back_to_the_policy() -> Result<()> {
        let chunks = vec![
            LlmResponseChunk::TextDelta {
                index: 0,
                text: "{\"ok\":".to_string(),
            },
            LlmResponseChunk::StreamError {
                message: "upstream exploded".to_string(),
            },
        ];
        assert_eq!(score_served_with(Ok(streamed(chunks))).await?, "no-verdict");
        Ok(())
    }

    #[tokio::test]
    async fn a_transport_failure_mid_stream_falls_back_to_the_policy() -> Result<()> {
        let partial = LlmResponseChunk::TextDelta {
            index: 0,
            text: "{\"ok\":".to_string(),
        };
        assert_eq!(
            score_served_with(Ok(streamed_then_failing(partial))).await?,
            "no-verdict"
        );
        Ok(())
    }

    #[tokio::test]
    async fn an_unparseable_reply_falls_back_to_the_policy() -> Result<()> {
        assert_eq!(
            score_served_with(Ok(buffered("sorry, I can't help with that"))).await?,
            "no-verdict"
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_failed_judge_call_falls_back_to_the_policy() -> Result<()> {
        let error = LibsyError::client_call(
            "judge",
            LlmClientError::Timeout {
                source: Box::new(std::io::Error::other("judge unreachable")),
            },
        );
        assert_eq!(score_served_with(Err(error)).await?, "no-verdict");
        Ok(())
    }

    #[tokio::test]
    async fn a_missing_driver_is_an_error_not_a_fallback() -> Result<()> {
        let error = classifier()
            .score(&mut State::default(), &request(), None)
            .await
            .err()
            .ok_or_else(|| LibsyError::AlgorithmError {
                message: "expected a missing-driver error".to_string(),
            })?;

        assert!(
            matches!(&error, LibsyError::AlgorithmError { message } if message.contains("judge")),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn fenced_replies_parse_as_verdicts() -> Result<()> {
        let judge = TestJudge;
        for reply in ["```json\n{\"ok\":true}\n```", "```\n{\"ok\":true}\n```"] {
            assert!(judge.parse(&text_response(None, reply))?.ok);
        }
        Ok(())
    }
}
