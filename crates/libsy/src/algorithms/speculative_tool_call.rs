// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Race a bounded next-tool-call prediction against the normal answer stream.

mod candidates;

use std::{collections::BTreeMap, sync::Arc, time::Duration};

use futures::{StreamExt, stream};
use serde_json::json;
use switchyard_protocol::{
    AggLlmResponse, Category, ChoiceOption, ContentBlock, DecisionKind, DecisionQuestion,
    DecisionRequest, DecisionResponse, DecisionValue, LlmResponse, LlmResponseChunk, ModelId,
    Request, Response, ResponseOutput, Role, StopReason,
};

use self::candidates::Candidate;
use crate::{Algorithm, Driver, LibsyError, Result, RoutingOutcome};

const MAX_BUFFER_BYTES: usize = 64 * 1024;
const MAX_BUFFER_EVENTS: usize = 128;

/// Limits and confidence policy for [`SpeculativeToolCall`].
#[derive(Clone, Debug)]
pub struct SpeculativeToolCallConfig {
    /// Minimum probability of the selected complete call.
    pub threshold: f64,
    /// Maximum delay before the normal stream is released.
    pub max_hold: Duration,
    /// Maximum complete calls offered to the judge, excluding abstention.
    pub max_candidates: usize,
    /// Completion-target instructions also visible to the decision model.
    pub target_system_prompt: Option<String>,
}

impl Default for SpeculativeToolCallConfig {
    fn default() -> Self {
        Self {
            threshold: 0.9,
            max_hold: Duration::from_millis(400),
            max_candidates: 64,
            target_system_prompt: None,
        }
    }
}

/// Returns a confident complete tool call, or continues the already-running normal answer.
pub struct SpeculativeToolCall {
    config: SpeculativeToolCallConfig,
}

impl SpeculativeToolCall {
    /// Rejects invalid confidence, deadline, and candidate limits.
    pub fn new(config: SpeculativeToolCallConfig) -> Result<Self> {
        if !(0.0..=1.0).contains(&config.threshold)
            || config.max_hold.is_zero()
            || !(1..=256).contains(&config.max_candidates)
        {
            return Err(LibsyError::AlgorithmError {
                message: "speculative_tool_call requires threshold in [0,1], a positive hold, and 1..=256 candidates".into(),
            });
        }
        Ok(Self { config })
    }

    fn prediction(
        &self,
        response: &DecisionResponse,
        candidates: &[Candidate],
    ) -> Option<(usize, f64)> {
        let DecisionValue::Choice {
            selected,
            probabilities: Some(probabilities),
        } = &response.answers.get("next_call")?.value
        else {
            return None;
        };
        let probability = probabilities.get(selected)?.0;
        if !(self.config.threshold..=1.0).contains(&probability) {
            return None;
        }
        candidates
            .iter()
            .position(|candidate| &candidate.id == selected)
            .map(|index| (index, probability))
    }

    fn answer(
        driver: &Driver,
        request: Request,
        normal: ModelId,
        judge: ModelId,
        prediction: (usize, f64),
        mut candidates: Vec<Candidate>,
        usage: switchyard_protocol::Usage,
    ) -> RoutingOutcome {
        let (index, probability) = prediction;
        driver.set_evidence(json!({
            "source": "decision", "decision_model": judge,
            "probability": probability, "candidate_count": candidates.len(),
        }));
        let mut call = candidates.swap_remove(index).call;
        let id = uuid::Uuid::now_v7().simple().to_string();
        call.id = format!("call_{id}");
        let aggregate = AggLlmResponse {
            id: Some(format!("spec_{id}")),
            model: Some(judge.to_string()),
            usage,
            outputs: vec![ResponseOutput {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolCall(call)],
                stop_reason: Some(StopReason::ToolUse),
                url_citations: Vec::new(),
            }],
            ..Default::default()
        };
        let mut response = Response {
            llm_response: if request.llm_request.stream {
                LlmResponse::Stream(aggregate.into_stream())
            } else {
                LlmResponse::Agg(aggregate)
            },
            metadata: request.metadata.clone(),
            upstream_headers: Default::default(),
        };
        response.set_served_model(&judge);
        response.metadata.get_or_insert_default().served_by_decision = true;
        RoutingOutcome::answered(normal, request, response)
    }
}

#[async_trait::async_trait]
impl Algorithm for SpeculativeToolCall {
    fn name(&self) -> &str {
        "speculative_tool_call"
    }

    async fn route(self: Arc<Self>, driver: Driver, request: Request) -> Result<RoutingOutcome> {
        let normal = driver.first_model_for(&Category::Any)?.clone();
        let candidates = candidates::candidates(&request.llm_request, self.config.max_candidates);
        if candidates.is_empty() {
            driver.set_evidence(json!({"source": "model", "reason": "no_candidates"}));
            return Ok(RoutingOutcome::route_to(normal, Vec::new(), request));
        }
        let judge = driver.first_model_for(&Category::Judge)?.clone();
        let mut options: Vec<_> = candidates
            .iter()
            .map(|candidate| ChoiceOption {
                id: candidate.id.clone(),
                description: Some(
                    json!({"name": candidate.call.name, "arguments": candidate.call.arguments}),
                ),
            })
            .collect();
        options.push(ChoiceOption {
            id: "abstain".into(),
            description: Some(json!("Text, clarification, multiple calls, or any action outside the candidates is needed.")),
        });
        let question = DecisionRequest {
            model: None,
            context: json!({
                "instructions": request.llm_request.instructions,
                "target_instructions": self.config.target_system_prompt,
                "messages": request.llm_request.messages,
                "tools": request.llm_request.tools,
                "tool_choice": request.llm_request.tool_choice,
            }),
            questions: BTreeMap::from([(
                "next_call".into(),
                DecisionQuestion {
                    instructions: json!(
                        "Choose the complete immediate next tool call, including every argument. Follow the conversation's instructions. Abstain if text, clarification, multiple tool calls, or any other action is needed. Never guess a missing argument."
                    ),
                    kind: DecisionKind::Choice { options },
                },
            )]),
        };
        let normal_call = driver.call_model_streaming(request.clone(), vec![normal.clone()]);
        let mut prediction = Box::pin(tokio::time::timeout(
            self.config.max_hold,
            driver.call_decision(question, judge.clone()),
        ));
        tokio::pin!(normal_call);
        let mut response = tokio::select! {
            result = &mut normal_call => match result {
                Ok(response) => response,
                Err(error) => {
                    if let Ok(Ok(response)) = prediction.await
                        && let Some(selected) = self.prediction(&response, &candidates)
                    {
                        return Ok(Self::answer(
                            &driver, request, normal, judge, selected, candidates, response.usage,
                        ));
                    }
                    return Err(error);
                }
            },
            result = &mut prediction => {
                // A timed-out future still owns its call until dropped.
                drop(prediction);
                if let Ok(Ok(response)) = result
                    && let Some(selected) = self.prediction(&response, &candidates)
                {
                    return Ok(Self::answer(
                        &driver, request, normal, judge, selected, candidates, response.usage,
                    ));
                }
                driver.set_evidence(json!({"source": "model", "reason": "prediction_rejected_or_expired"}));
                return Ok(RoutingOutcome::answered(normal, request, normal_call.await?));
            }
        };
        let LlmResponse::Stream(mut chunks) = response.llm_response else {
            driver.set_evidence(json!({"source": "model", "reason": "normal_complete"}));
            return Ok(RoutingOutcome::answered(normal, request, response));
        };
        let mut prefix = Vec::new();
        let mut bytes = 0usize;
        loop {
            tokio::select! {
                result = &mut prediction => {
                    if let Ok(Ok(predicted)) = result
                        && let Some(selected) = self.prediction(&predicted, &candidates)
                    {
                        return Ok(Self::answer(
                            &driver, request, normal, judge, selected, candidates, predicted.usage,
                        ));
                    }
                    break;
                }
                event = chunks.next() => {
                    let Some(event) = event else { break };
                    let terminal = event.as_ref().map_or(true, |event| {
                        event.normalized().iter().any(|chunk| matches!(chunk,
                            LlmResponseChunk::MessageStop { .. }
                            | LlmResponseChunk::DecodeError { .. }
                            | LlmResponseChunk::StreamError { .. }
                        ))
                    });
                    bytes = bytes.saturating_add(event.as_ref().ok()
                        .and_then(|event| serde_json::to_vec(event).ok())
                        .map_or(MAX_BUFFER_BYTES, |value| value.len()));
                    prefix.push(event);
                    if terminal || bytes >= MAX_BUFFER_BYTES || prefix.len() >= MAX_BUFFER_EVENTS {
                        break;
                    }
                }
            }
        }
        driver.set_evidence(json!({"source": "model", "reason": "normal_stream_committed"}));
        response.llm_response = LlmResponse::Stream(stream::iter(prefix).chain(chunks).boxed());
        Ok(RoutingOutcome::answered(normal, request, response))
    }
}
