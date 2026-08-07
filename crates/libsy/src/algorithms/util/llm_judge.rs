// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared LLM judge primitives.
//!
//! [`Judge`] owns algorithm-specific request construction and verdict parsing.
//! [`JudgeClassifier`] owns the judge model call and hands its verdict to a policy that chooses
//! the route.

use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde_json::Value;
use switchyard_protocol::{
    AggLlmResponse, InstructionBlock, LlmRequest, Message, OutputParams, Role, completion_text,
};

use super::classifier_contract::ClassifierContract;
use crate::core::algorithm::{Driver, LlmTarget, RoutedRequest};
use crate::core::classifier::{Classification, Classifier};
use crate::core::state::State;
use crate::{LibsyError, Result};
use switchyard_protocol::{Context, Decision, LlmClientError, Request, Response};

/// Builds the classifier-specific message view presented to a structured judge.
pub(crate) trait ClassifierInput: Send + Sync {
    fn build_messages(&self, state: &State, request: &Request) -> Vec<Message>;
}

/// Converts one structured model response into the verdict type consumed by a policy.
pub(crate) trait VerdictDecoder: Send + Sync {
    type Verdict: DeserializeOwned + Send + Sync;

    fn decode(
        &self,
        response: &AggLlmResponse,
        contract: &ClassifierContract,
    ) -> Result<Self::Verdict>;
}

/// Deserializes a structured response directly into a typed verdict.
pub(crate) struct SerdeDecoder<V> {
    verdict: PhantomData<fn() -> V>,
}

impl<V> SerdeDecoder<V> {
    pub(crate) const fn new() -> Self {
        Self {
            verdict: PhantomData,
        }
    }
}

impl<V> VerdictDecoder for SerdeDecoder<V>
where
    V: DeserializeOwned + Send + Sync,
{
    type Verdict = V;

    fn decode(
        &self,
        response: &AggLlmResponse,
        _contract: &ClassifierContract,
    ) -> Result<Self::Verdict> {
        parse_json_verdict(response)
    }
}

/// Parses a JSON value and enforces the custom contract's compiled response schema.
pub(crate) struct JsonSchemaDecoder;

impl JsonSchemaDecoder {
    pub(crate) const fn new() -> Self {
        Self
    }
}

impl VerdictDecoder for JsonSchemaDecoder {
    type Verdict = Value;

    fn decode(
        &self,
        response: &AggLlmResponse,
        contract: &ClassifierContract,
    ) -> Result<Self::Verdict> {
        let verdict = parse_json_verdict(response)?;
        contract.validate_verdict(&verdict)?;
        Ok(verdict)
    }
}

/// Runtime limits shared by structured classifier judges.
pub(crate) struct JudgeRuntimeConfig {
    max_output_tokens: u64,
}

impl JudgeRuntimeConfig {
    pub(crate) fn new(max_output_tokens: u64) -> Result<Self> {
        if max_output_tokens == 0 {
            return Err(LibsyError::AlgorithmError {
                message: "max_output_tokens must be at least 1".to_string(),
            });
        }
        Ok(Self { max_output_tokens })
    }
}

/// Reusable structured judge assembled from an input view, contract, and verdict decoder.
pub(crate) struct StructuredJudge<I, D> {
    input: I,
    contract: ClassifierContract,
    decoder: D,
    runtime: JudgeRuntimeConfig,
}

impl<I, D> StructuredJudge<I, D> {
    pub(crate) fn new(
        input: I,
        contract: ClassifierContract,
        decoder: D,
        runtime: JudgeRuntimeConfig,
    ) -> Self {
        Self {
            input,
            contract,
            decoder,
            runtime,
        }
    }

    #[cfg(test)]
    pub(crate) fn contract(&self) -> &ClassifierContract {
        &self.contract
    }
}

impl<I, D> Judge for StructuredJudge<I, D>
where
    I: ClassifierInput,
    D: VerdictDecoder,
{
    type Verdict = D::Verdict;

    fn build_request(&self, state: &State, request: &Request) -> Request {
        let messages = self.input.build_messages(state, request);
        Request {
            llm_request: LlmRequest {
                model: request.llm_request.model.clone(),
                instructions: vec![InstructionBlock {
                    role: Role::System,
                    content: Message::text(Role::System, self.contract.system_prompt().to_string())
                        .content,
                }],
                messages,
                output: OutputParams {
                    max_output_tokens: Some(self.runtime.max_output_tokens),
                    response_format: Some(self.contract.response_format().clone()),
                },
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: request.metadata.clone(),
        }
    }

    fn parse(&self, response: &AggLlmResponse) -> Result<Self::Verdict> {
        self.decoder.decode(response, &self.contract)
    }
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
    /// fallback branch. A closed driver stream is folded too; the algorithm's next driver
    /// call surfaces it, so nothing is masked.
    async fn verdict(
        &self,
        state: &mut State,
        request: &Request,
        driver: &Driver,
    ) -> Option<J::Verdict> {
        let judge_model = self.target.semantic_name.as_str();

        let response = driver
            .call_llm(RoutedRequest {
                request: self.judge.build_request(state, request),
                decision: Arc::new(JudgeDecision {
                    model: self.target.semantic_name.to_string(),
                }),
                ctx: Context::default(),
            })
            .await
            .inspect_err(|error| report_fail_open(judge_model, error, libsy_error_reason(error)))
            .ok()?;
        let aggregate = response
            .llm_response
            .into_agg()
            .await
            .inspect_err(|error| report_fail_open(judge_model, error, client_error_reason(error)))
            .ok()?;
        self.judge
            .parse(&aggregate)
            .inspect_err(|error| report_fail_open(judge_model, error, "parse_error"))
            .ok()
    }
}

/// Logs and counts a judge failure with a bounded label that excludes message content.
fn report_fail_open(judge_model: &str, error: &dyn std::fmt::Display, reason: &'static str) {
    tracing::warn!(
        target: "libsy",
        judge_model,
        reason,
        error = %error,
        "judge verdict unavailable; routing without one"
    );
    crate::observability::record_classifier_fail_open(judge_model, reason);
}

/// Returns a bounded reason for a judge call that failed at the libsy layer.
fn libsy_error_reason(error: &LibsyError) -> &'static str {
    match error {
        LibsyError::ClientCall { source, .. } => client_error_reason(source),
        _ => "call_error",
    }
}

/// Returns a bounded reason from the error kind and HTTP status only.
fn client_error_reason(error: &LlmClientError) -> &'static str {
    match error {
        LlmClientError::Timeout { .. } => "timeout",
        LlmClientError::Transport { .. } => "transport",
        LlmClientError::UpstreamHttp { status, .. } if (500..=599).contains(status) => {
            "upstream_5xx"
        }
        LlmClientError::UpstreamHttp { .. } => "upstream_non_5xx",
        LlmClientError::InvalidResponse { .. } | LlmClientError::ResponseTranslation(_) => {
            "invalid_response"
        }
        _ => "client_error",
    }
}

#[async_trait]
impl<J, P> Classifier<State> for JudgeClassifier<J, P>
where
    J: Judge,
    P: JudgePolicy<Verdict = J::Verdict>,
{
    async fn score(
        &self,
        state: &mut State,
        request: &mut Request,
        driver: Option<&Driver>,
    ) -> Result<(Classification, Option<Response>)> {
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
        // A judge consultation is a side call, never the turn's answer.
        Ok((self.policy.to_classification(verdict.as_ref()), None))
    }
}

fn parse_json_verdict<T: DeserializeOwned>(response: &AggLlmResponse) -> Result<T> {
    // Providers sometimes wrap otherwise valid JSON in a Markdown fence.
    let reply = completion_text(response);
    serde_json::from_str(strip_json_fence(reply.trim())).map_err(|err| LibsyError::AlgorithmError {
        message: format!(
            "judge reply did not parse as {}: {err}",
            std::any::type_name::<T>()
        ),
    })
}

struct JudgeDecision {
    model: String,
}

impl Decision for JudgeDecision {
    fn selected_model(&self) -> &str {
        &self.model
    }

    fn is_routed_call(&self) -> bool {
        false
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
    use switchyard_protocol::{ContentBlock, LlmClientError, text_request, text_response};

    use crate::core::algorithm::Step;
    use crate::core::classifier::Score;
    use switchyard_protocol::{LlmResponse, LlmResponseChunk, Response};

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

    #[test]
    fn the_verdict_is_read_from_the_completion() -> Result<()> {
        // A judge's reasoning is not its answer: only `content` carries the verdict, so a
        // reply that never reached one — a run truncated mid-thought — is an error rather
        // than a guess.
        let mut response = text_response(None, VERDICT);
        if let Some(output) = response.outputs.first_mut() {
            output.content.insert(
                0,
                ContentBlock::Reasoning {
                    text: r#"{"ok":false}"#.to_string(),
                    signature: None,
                },
            );
        }
        let parsed: TestVerdict = parse_json_verdict(&response)?;
        assert_eq!(parsed, TestVerdict { ok: true });

        assert!(parse_json_verdict::<TestVerdict>(&text_response(None, "still thinking")).is_err());
        Ok(())
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
                futures::stream::iter(chunks.into_iter().map(|chunk| Ok(chunk.into()))).boxed(),
            ),
            metadata: None,
        }
    }

    fn streamed_then_failing(chunk: LlmResponseChunk) -> Response {
        let items = futures::stream::iter([
            Ok(chunk.into()),
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
        let mut request = request();

        let serve = async {
            if let Some(Ok(Step::CallLlm(call))) = steps.next().await {
                let _ = call.respond(reply);
            }
        };
        let (classification, ()) = tokio::join!(
            classifier.score(&mut state, &mut request, Some(&driver)),
            serve
        );
        let (classification, _) = classification?;
        selected(classification)
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

    #[test]
    fn client_errors_map_to_bounded_fail_open_reasons() {
        let cases = vec![
            (
                LlmClientError::Timeout {
                    source: "deadline exceeded".into(),
                },
                "timeout",
            ),
            (
                LlmClientError::Transport {
                    source: "connection refused".into(),
                },
                "transport",
            ),
            (
                LlmClientError::UpstreamHttp {
                    status: 500,
                    body: "server error".to_string(),
                },
                "upstream_5xx",
            ),
            (
                LlmClientError::UpstreamHttp {
                    status: 302,
                    body: "redirect".to_string(),
                },
                "upstream_non_5xx",
            ),
            (
                LlmClientError::InvalidResponse {
                    source: "invalid JSON".into(),
                },
                "invalid_response",
            ),
            (
                LlmClientError::General("unexpected client failure".to_string()),
                "client_error",
            ),
        ];
        for (error, expected) in cases {
            assert_eq!(client_error_reason(&error), expected);
        }

        let error = LibsyError::AlgorithmError {
            message: "driver failed".to_string(),
        };
        assert_eq!(libsy_error_reason(&error), "call_error");
    }

    #[tokio::test]
    async fn a_missing_driver_is_an_error_not_a_fallback() -> Result<()> {
        let mut request = request();
        let error = classifier()
            .score(&mut State::default(), &mut request, None)
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
