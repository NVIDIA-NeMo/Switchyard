// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One-way request-size gate for two-tier routes.
//!
//! [`RequestFitClassifier`] estimates the request's input size and escalates to the
//! strong tier when it crosses a configured waterline: the weak model would reject the
//! request (its context window is smaller) or handle it poorly (long-context quality
//! degrades faster for small models). Below the waterline the classifier abstains —
//! length says nothing about difficulty, so the cascade's next classifier decides.
//!
//! [`RequestFit`] is the assembled standalone route: the gate, then the weak tier as
//! the default target.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use switchyard_protocol::{
    ContentBlock, FileSource, ImageSource, MediaSource, ModelId, Request, Response,
};

use super::fall_through::{DefaultTarget, FallThrough};
use crate::core::algorithm::{Algorithm, Driver};
use crate::core::classifier::{Classification, Classifier, Score};
use crate::{LibsyError, Result};

/// Telemetry label for this algorithm's spans, metrics, and logs.
const ALGORITHM_NAME: &str = "request_fit";

/// Rough characters per token across common BPE tokenizers. Deliberately coarse —
/// the gate is a pre-flight check, not billing.
const CHARS_PER_TOKEN: u64 = 4;

/// Fixed cost for JSON numbers, booleans, and nulls in tool schemas and arguments.
const SCALAR_CHARS: u64 = 8;

/// Estimates the request's input size in tokens as `chars / 4`, rounded up.
///
/// Counts instructions, message content (including tool calls and results), media
/// payloads, and tool definitions. Overestimates slightly rather than missing an
/// escalation the weak tier would have rejected.
pub(crate) fn estimate_input_tokens(request: &Request) -> u64 {
    let llm = &request.llm_request;
    let mut chars = 0u64;
    for instruction in &llm.instructions {
        chars = chars.saturating_add(blocks_chars(&instruction.content));
    }
    for message in &llm.messages {
        chars = chars.saturating_add(blocks_chars(&message.content));
    }
    for tool in &llm.tools {
        chars = chars
            .saturating_add(tool.name.chars().count() as u64)
            .saturating_add(
                tool.description
                    .as_deref()
                    .map_or(0, |d| d.chars().count() as u64),
            )
            .saturating_add(value_chars(&tool.parameters));
    }
    chars.saturating_add(CHARS_PER_TOKEN - 1) / CHARS_PER_TOKEN
}

fn blocks_chars(blocks: &[ContentBlock]) -> u64 {
    blocks.iter().map(block_chars).sum()
}

fn block_chars(block: &ContentBlock) -> u64 {
    match block {
        ContentBlock::Text { text }
        | ContentBlock::Refusal { text }
        | ContentBlock::Reasoning { text, .. } => text.chars().count() as u64,
        ContentBlock::ToolCall(call) => {
            (call.name.chars().count() as u64).saturating_add(value_chars(&call.arguments))
        }
        ContentBlock::ToolResult(result) => blocks_chars(&result.content),
        ContentBlock::Image { source } => match source {
            ImageSource::Url { url, .. } => url.chars().count() as u64,
            ImageSource::Base64 { data, .. } => data.chars().count() as u64,
            ImageSource::Raw(raw) => value_chars(raw),
        },
        ContentBlock::Audio { source } | ContentBlock::Video { source } => match source {
            MediaSource::Url { url, .. } => url.chars().count() as u64,
            MediaSource::Base64 { data, .. } => data.chars().count() as u64,
            MediaSource::Raw(raw) => value_chars(raw),
        },
        ContentBlock::File { source } => match source {
            FileSource::FileId(id) => id.chars().count() as u64,
            FileSource::FileData { data, .. } => data.chars().count() as u64,
            FileSource::Raw(raw) => value_chars(raw),
        },
        ContentBlock::Unknown { raw, .. } => value_chars(raw),
    }
}

/// Sums the character cost of an untyped JSON value (tool schemas, call arguments,
/// unnormalized provider blocks), keys included.
fn value_chars(value: &Value) -> u64 {
    match value {
        Value::String(s) => s.chars().count() as u64,
        Value::Array(items) => items.iter().map(value_chars).sum(),
        Value::Object(map) => map
            .iter()
            .map(|(key, item)| (key.chars().count() as u64).saturating_add(value_chars(item)))
            .sum(),
        _ => SCALAR_CHARS,
    }
}

/// Escalates requests whose estimated input size reaches the waterline; abstains below it.
///
/// Stateless and driver-free, so it can sit in any cascade — ahead of a judge it skips
/// guaranteed failures before the judge call is paid for.
pub struct RequestFitClassifier {
    weak_target: ModelId,
    strong_target: ModelId,
    escalate_over_input_tokens: u64,
}

impl RequestFitClassifier {
    /// Creates the gate. `escalate_over_input_tokens` must be positive and the two
    /// targets must differ.
    pub fn new(
        weak_target: impl Into<ModelId>,
        strong_target: impl Into<ModelId>,
        escalate_over_input_tokens: u64,
    ) -> Result<Self> {
        let weak_target = weak_target.into();
        let strong_target = strong_target.into();
        if escalate_over_input_tokens == 0 {
            return Err(LibsyError::AlgorithmError {
                message: "escalate_over_input_tokens must be greater than zero".to_string(),
            });
        }
        if weak_target == strong_target {
            return Err(LibsyError::AlgorithmError {
                message: "weak_target and strong_target must differ".to_string(),
            });
        }
        Ok(Self {
            weak_target,
            strong_target,
            escalate_over_input_tokens,
        })
    }
}

#[async_trait]
impl<S: Send> Classifier<S> for RequestFitClassifier {
    fn routing_tier(&self, selected_model_id: &ModelId) -> Option<&'static str> {
        if *selected_model_id == self.weak_target {
            Some("weak")
        } else if *selected_model_id == self.strong_target {
            Some("strong")
        } else {
            None
        }
    }

    async fn score(
        &self,
        _state: &mut S,
        request: &mut Request,
        _driver: Option<&Driver>,
    ) -> Result<(Classification, Option<Response>)> {
        let estimate = estimate_input_tokens(request);
        if estimate >= self.escalate_over_input_tokens {
            tracing::info!(
                target = %self.strong_target,
                estimated_input_tokens = estimate,
                waterline = self.escalate_over_input_tokens,
                "request-fit gate escalated to strong tier"
            );
            return Ok((
                Classification::Scores(vec![Score {
                    target: self.strong_target.clone(),
                    confidence: 1.0,
                }]),
                None,
            ));
        }
        Ok((Classification::Ambiguous(vec![]), None))
    }
}

/// Standalone request-fit route: the gate, then the weak tier as the default target.
pub struct RequestFit {
    route: FallThrough,
}

impl RequestFit {
    /// Assembles the route. Requests below the waterline are served by `weak_target`.
    pub fn new(
        weak_target: impl Into<ModelId>,
        strong_target: impl Into<ModelId>,
        escalate_over_input_tokens: u64,
    ) -> Result<Self> {
        let weak_target = weak_target.into();
        let strong_target = strong_target.into();
        let gate = Arc::new(RequestFitClassifier::new(
            weak_target.clone(),
            strong_target.clone(),
            escalate_over_input_tokens,
        )?);
        Ok(Self {
            route: FallThrough::new(vec![weak_target.clone(), strong_target])
                .with_name(ALGORITHM_NAME)
                .with_classifier(gate)
                .with_classifier(Arc::new(DefaultTarget::new(weak_target))),
        })
    }
}

#[async_trait]
impl Algorithm for RequestFit {
    fn name(&self) -> &str {
        ALGORITHM_NAME
    }

    async fn route(self: Arc<Self>, driver: Driver, request: Request) -> Result<Response> {
        self.route.execute(driver, request).await
    }
}

#[cfg(test)]
mod tests {
    use switchyard_protocol::{
        Decision, InstructionBlock, LlmClientError, LlmRequest, Message, Role, ToolCall,
        ToolDefinition, ToolResult, completion_text, text_request,
    };

    use super::*;
    use crate::core::testing::{Serve, test_drive};

    const WATERLINE: u64 = 100;

    fn sized_request(chars: usize) -> Request {
        Request {
            llm_request: text_request(Some("auto".to_string()), "x".repeat(chars)),
            raw_request: None,
            metadata: None,
        }
    }

    fn classifier() -> RequestFitClassifier {
        RequestFitClassifier::new("weak", "strong", WATERLINE).expect("valid gate")
    }

    #[test]
    fn the_estimator_counts_message_text() {
        // 400 chars / 4 = 100 tokens.
        assert_eq!(estimate_input_tokens(&sized_request(400)), 100);
    }

    #[test]
    fn the_estimator_rounds_up_so_small_requests_still_count() {
        assert_eq!(estimate_input_tokens(&sized_request(1)), 1);
    }

    #[test]
    fn the_estimator_counts_instructions_and_tool_definitions() {
        let request = Request {
            llm_request: LlmRequest {
                instructions: vec![InstructionBlock {
                    role: Role::System,
                    content: vec![ContentBlock::Text {
                        text: "i".repeat(40),
                    }],
                }],
                messages: vec![Message::text(Role::User, "m".repeat(40))],
                tools: vec![ToolDefinition {
                    name: "n".repeat(20),
                    description: Some("d".repeat(60)),
                    parameters: serde_json::json!({"type": "object"}),
                    strict: None,
                }],
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: None,
        };
        // value_chars counts schema keys and string values, not JSON punctuation:
        // "type" (4) + "object" (6) = 10. (40 + 40 + 20 + 60 + 10) / 4 = 42.5 → 43.
        assert_eq!(estimate_input_tokens(&request), 43);
    }

    #[test]
    fn the_estimator_counts_tool_calls_and_results() {
        let request = Request {
            llm_request: LlmRequest {
                messages: vec![
                    Message {
                        role: Role::Assistant,
                        content: vec![ContentBlock::ToolCall(ToolCall {
                            id: "call-1".to_string(),
                            name: "n".repeat(10),
                            arguments: serde_json::json!({"arg": "v".repeat(30)}),
                        })],
                    },
                    Message {
                        role: Role::Tool,
                        content: vec![ContentBlock::ToolResult(ToolResult {
                            tool_call_id: "call-1".to_string(),
                            content: vec![ContentBlock::Text {
                                text: "r".repeat(50),
                            }],
                            is_error: None,
                        })],
                    },
                ],
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: None,
        };
        // call: name 10 + value_chars({"arg": "v"*30}) = key "arg" (3) + 30 = 33 → 43;
        // result: 50 chars → (43 + 50) / 4 = 23.25 → 24.
        assert_eq!(estimate_input_tokens(&request), 24);
    }

    #[test]
    fn the_estimator_counts_inline_media_payloads() {
        let request = Request {
            llm_request: LlmRequest {
                messages: vec![Message {
                    role: Role::User,
                    content: vec![ContentBlock::Image {
                        source: ImageSource::Base64 {
                            media_type: Some("image/png".to_string()),
                            data: "d".repeat(400),
                        },
                    }],
                }],
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: None,
        };
        assert_eq!(estimate_input_tokens(&request), 100);
    }

    #[tokio::test]
    async fn the_gate_abstains_below_the_waterline() -> Result<()> {
        let mut state = ();
        let mut request = sized_request(4 * (WATERLINE as usize - 1));
        let (classification, served) = classifier().score(&mut state, &mut request, None).await?;
        assert!(classification.argmax(false)?.is_none());
        assert!(served.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn the_gate_escalates_at_the_waterline_inclusively() -> Result<()> {
        let mut state = ();
        let mut request = sized_request(4 * WATERLINE as usize);
        let (classification, _) = classifier().score(&mut state, &mut request, None).await?;
        assert_eq!(
            classification.argmax(false)?.map(|s| s.target),
            Some(ModelId::from("strong"))
        );
        Ok(())
    }

    #[test]
    fn the_gate_reports_routing_tiers() {
        let classifier = classifier();
        assert_eq!(
            <RequestFitClassifier as Classifier<()>>::routing_tier(
                &classifier,
                &ModelId::from("weak")
            ),
            Some("weak")
        );
        assert_eq!(
            <RequestFitClassifier as Classifier<()>>::routing_tier(
                &classifier,
                &ModelId::from("strong")
            ),
            Some("strong")
        );
        assert_eq!(
            <RequestFitClassifier as Classifier<()>>::routing_tier(
                &classifier,
                &ModelId::from("other")
            ),
            None
        );
    }

    #[test]
    fn invalid_gate_settings_are_rejected() {
        assert!(RequestFitClassifier::new("weak", "strong", 0).is_err());
        assert!(RequestFitClassifier::new("same", "same", WATERLINE).is_err());
    }

    fn serve_by_target() -> impl Serve {
        |decision: switchyard_protocol::Decision, request: Request| async move {
            Ok(crate::core::testing::reply(format!(
                "answer from {}",
                decision.selected_model_id()
            )))
            .map(|response| Response {
                metadata: request.metadata,
                ..response
            })
        }
    }

    #[tokio::test]
    async fn a_short_request_is_served_by_the_weak_tier() -> Result<()> {
        let router = Arc::new(RequestFit::new("weak", "strong", WATERLINE)?);
        let (trace, response) = test_drive(router, sized_request(40), serve_by_target()).await?;
        assert_eq!(
            trace.last().map(|d| d.selected_model_id().as_str()),
            Some("weak")
        );
        assert_eq!(
            response.llm_response.as_agg().map(completion_text),
            Some("answer from weak".to_string())
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_long_request_is_served_by_the_strong_tier() -> Result<()> {
        let router = Arc::new(RequestFit::new("weak", "strong", WATERLINE)?);
        let (trace, response) = test_drive(router, sized_request(4000), serve_by_target()).await?;
        assert_eq!(
            trace.last().map(|d| d.selected_model_id().as_str()),
            Some("strong")
        );
        assert_eq!(
            response.llm_response.as_agg().map(completion_text),
            Some("answer from strong".to_string())
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_weak_overflow_after_the_gate_passes_is_retried_on_the_strong_tier() -> Result<()> {
        // The estimate puts this request below the waterline, so the gate abstains and the
        // weak tier is served. When the weak tier still overflows, the reactive eviction
        // kicks in and retries on the strong tier — the gate only makes overflows rare, it
        // does not remove the backstop.
        let router = Arc::new(RequestFit::new("weak", "strong", WATERLINE)?);
        let serve = |decision: Decision, _request: Request| async move {
            if decision.selected_model_id().as_str() == "weak" {
                return Err(LlmClientError::ContextWindowExceeded {
                    model: decision.selected_model_id().clone(),
                    message: "prompt is too long".to_string(),
                });
            }
            Ok(crate::core::testing::reply(format!(
                "answer from {}",
                decision.selected_model_id()
            )))
        };
        let (trace, response) = test_drive(router, sized_request(40), serve).await?;
        assert_eq!(
            trace.last().map(|d| d.selected_model_id().as_str()),
            Some("strong")
        );
        assert_eq!(
            response.llm_response.as_agg().map(completion_text),
            Some("answer from strong".to_string())
        );
        Ok(())
    }
}
