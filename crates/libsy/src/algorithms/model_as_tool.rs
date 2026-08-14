// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Route a model-requested media tool call to a specialized generation model.

use std::sync::Arc;

use serde_json::json;
use switchyard_protocol::{
    AggLlmResponse, ContentBlock, Decision, LlmResponse, Metadata, ModelId, Request, Response,
    ToolCall, ToolChoice, ToolDefinition, text_request,
};

use crate::core::algorithm::{Algorithm, Driver};
use crate::{LibsyError, Result};

/// Name of the synthetic tool exposed to the primary model.
pub const GENERATE_MEDIA_TOOL_NAME: &str = "generate_media";

/// Routes a model-selected media tool call to a specialized generation model.
///
/// The primary model receives one additional `generate_media(prompt)` tool. Normal answers pass
/// through unchanged. A matching tool call becomes a second routed model call whose host client
/// is responsible for turning the prompt into media.
pub struct ModelAsTool {
    primary_target: ModelId,
    media_target: ModelId,
}

impl ModelAsTool {
    /// Creates a router backed by a reasoning model and a specialized media model.
    pub fn new(primary_target: impl Into<ModelId>, media_target: impl Into<ModelId>) -> Self {
        Self {
            primary_target: primary_target.into(),
            media_target: media_target.into(),
        }
    }
}

#[async_trait::async_trait]
impl Algorithm for ModelAsTool {
    fn name(&self) -> &str {
        "model_as_tool"
    }

    async fn route(self: Arc<Self>, driver: Driver, mut request: Request) -> Result<Response> {
        reject_reserved_tool_collision(&request)?;

        let stream_requested = request.llm_request.stream;
        let request_metadata = request.metadata.clone();
        // The tool choice must be inspected before the algorithm can decide what to return.
        request.llm_request.stream = false;
        request
            .llm_request
            .extensions
            .fields
            .remove("stream_options");
        request
            .llm_request
            .extensions
            .fields
            .insert("parallel_tool_calls".to_string(), json!(false));
        request.llm_request.tools.push(media_tool());
        request
            .llm_request
            .tool_choice
            .get_or_insert(ToolChoice::Auto);
        // Same-format replay would encode the preserved request instead of the injected tool.
        request.llm_request.preservation.requests.clear();

        tracing::info!(
            target = %self.primary_target,
            tool = GENERATE_MEDIA_TOOL_NAME,
            "offering specialized model as a tool"
        );
        let primary_decision = Decision::new(self.primary_target.clone(), true);
        driver.decide(primary_decision.clone()).await?;
        let primary_response = driver.call_model(request, primary_decision).await?;
        let Response {
            llm_response,
            metadata,
        } = primary_response;
        let aggregate = llm_response
            .into_agg()
            .await
            .map_err(|error| LibsyError::external("inspecting model-as-tool response", error))?;

        let Some(tool_call) = find_media_tool_call(&aggregate)? else {
            return Ok(response_from_aggregate(
                aggregate,
                metadata,
                stream_requested,
            ));
        };
        let prompt = media_prompt(tool_call)?;
        tracing::info!(
            target = %self.media_target,
            tool = GENERATE_MEDIA_TOOL_NAME,
            "dispatching selected specialized model tool"
        );

        let media_request = Request {
            llm_request: text_request(None, prompt),
            raw_request: None,
            metadata: request_metadata,
        };
        let media_decision = Decision::new(self.media_target.clone(), true);
        driver.decide(media_decision.clone()).await?;
        let media_response = driver.call_model(media_request, media_decision).await?;
        if !stream_requested {
            return Ok(media_response);
        }

        let Response {
            llm_response,
            metadata,
        } = media_response;
        let aggregate = llm_response
            .into_agg()
            .await
            .map_err(|error| LibsyError::external("streaming model-as-tool response", error))?;
        Ok(response_from_aggregate(aggregate, metadata, true))
    }
}

fn reject_reserved_tool_collision(request: &Request) -> Result<()> {
    if request
        .llm_request
        .tools
        .iter()
        .any(|tool| tool.name == GENERATE_MEDIA_TOOL_NAME)
    {
        return Err(LibsyError::AlgorithmError {
            message: format!("request already defines reserved tool {GENERATE_MEDIA_TOOL_NAME:?}"),
        });
    }
    Ok(())
}

fn media_tool() -> ToolDefinition {
    ToolDefinition {
        name: GENERATE_MEDIA_TOOL_NAME.to_string(),
        description: Some(
            "Generate an image with a specialized local visual model. Call this tool alone only when visual output materially improves the answer. Supply a self-contained prompt describing the scene, composition, and style."
                .to_string(),
        ),
        parameters: json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "A complete image generation prompt."
                }
            },
            "required": ["prompt"],
            "additionalProperties": false
        }),
        strict: Some(true),
    }
}

fn find_media_tool_call(response: &AggLlmResponse) -> Result<Option<&ToolCall>> {
    let tool_calls = response
        .outputs
        .iter()
        .flat_map(|output| &output.content)
        .filter_map(|block| match block {
            ContentBlock::ToolCall(tool_call) => Some(tool_call),
            _ => None,
        })
        .collect::<Vec<_>>();
    let media_call = tool_calls
        .iter()
        .copied()
        .find(|tool_call| tool_call.name == GENERATE_MEDIA_TOOL_NAME);
    if media_call.is_some() && tool_calls.len() != 1 {
        return Err(LibsyError::AlgorithmError {
            message: format!(
                "{GENERATE_MEDIA_TOOL_NAME} must be called alone; primary model returned {} tool calls",
                tool_calls.len()
            ),
        });
    }
    Ok(media_call)
}

fn media_prompt(tool_call: &ToolCall) -> Result<String> {
    tool_call
        .arguments
        .get("prompt")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|prompt| !prompt.is_empty())
        .map(str::to_string)
        .ok_or_else(|| LibsyError::AlgorithmError {
            message: format!(
                "{GENERATE_MEDIA_TOOL_NAME} call {} requires a non-empty string prompt",
                tool_call.id
            ),
        })
}

fn response_from_aggregate(
    aggregate: AggLlmResponse,
    metadata: Option<Metadata>,
    stream: bool,
) -> Response {
    Response {
        llm_response: if stream {
            LlmResponse::Stream(aggregate.into_stream())
        } else {
            LlmResponse::Agg(aggregate)
        },
        metadata,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use serde_json::json;
    use switchyard_protocol::{
        AggLlmResponse, ContentBlock, LlmRequest, LlmResponse, Message, Request, Response,
        ResponseOutput, Role, StopReason, ToolCall, ToolChoice, completion_text, text_response,
    };

    use super::{GENERATE_MEDIA_TOOL_NAME, ModelAsTool};
    use crate::core::algorithm::Algorithm;
    use crate::core::testing::test_drive;

    fn request(stream: bool) -> Request {
        Request {
            llm_request: LlmRequest {
                model: Some("auto".to_string()),
                messages: vec![Message::text(Role::User, "Make a cinematic launch image")],
                stream,
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: None,
        }
    }

    fn tool_call_response(prompt: serde_json::Value) -> Response {
        Response {
            llm_response: LlmResponse::Agg(AggLlmResponse {
                outputs: vec![ResponseOutput {
                    role: Role::Assistant,
                    content: vec![ContentBlock::ToolCall(ToolCall {
                        id: "media-1".to_string(),
                        name: GENERATE_MEDIA_TOOL_NAME.to_string(),
                        arguments: json!({"prompt": prompt}),
                    })],
                    stop_reason: Some(StopReason::ToolUse),
                }],
                ..AggLlmResponse::default()
            }),
            metadata: None,
        }
    }

    #[tokio::test]
    async fn normal_answer_passes_through_after_tool_injection() -> crate::Result<()> {
        let recorded = Arc::new(Mutex::new(None));
        let captured = Arc::clone(&recorded);
        let algorithm: Arc<dyn Algorithm> = Arc::new(ModelAsTool::new("primary", "cosmos"));
        let mut streamed_request = request(true);
        streamed_request
            .llm_request
            .extensions
            .fields
            .insert("stream_options".to_string(), json!({"include_usage": true}));
        let (trace, response) =
            test_drive(algorithm, streamed_request, move |_decision, request| {
                let captured = Arc::clone(&captured);
                async move {
                    *captured.lock().expect("recording lock") = Some(request);
                    Ok(Response {
                        llm_response: LlmResponse::Agg(text_response(
                            Some("primary".to_string()),
                            "plain answer",
                        )),
                        metadata: None,
                    })
                }
            })
            .await?;

        let request = recorded
            .lock()
            .expect("recording lock")
            .take()
            .expect("primary request");
        assert_eq!(request.llm_request.tools.len(), 1);
        assert_eq!(request.llm_request.tools[0].name, GENERATE_MEDIA_TOOL_NAME);
        assert_eq!(request.llm_request.tool_choice, Some(ToolChoice::Auto));
        assert!(!request.llm_request.stream);
        assert!(
            !request
                .llm_request
                .extensions
                .fields
                .contains_key("stream_options")
        );
        assert_eq!(
            request
                .llm_request
                .extensions
                .fields
                .get("parallel_tool_calls"),
            Some(&json!(false))
        );
        assert!(request.llm_request.preservation.requests.is_empty());
        assert_eq!(trace.len(), 1);
        assert!(trace[0].is_answer_call());
        let aggregate = response
            .llm_response
            .into_agg()
            .await
            .map_err(|error| crate::LibsyError::external("aggregating test response", error))?;
        assert_eq!(completion_text(&aggregate), "plain answer");
        Ok(())
    }

    #[tokio::test]
    async fn selected_tool_dispatches_prompt_to_media_target() -> crate::Result<()> {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&calls);
        let algorithm: Arc<dyn Algorithm> = Arc::new(ModelAsTool::new("primary", "cosmos"));
        let (trace, response) = test_drive(
            algorithm,
            request(false),
            move |decision: switchyard_protocol::Decision, request: Request| {
                let captured = Arc::clone(&captured);
                async move {
                    captured
                        .lock()
                        .expect("recording lock")
                        .push((decision.selected_model_id().to_string(), request.clone()));
                    match decision.selected_model_id().as_str() {
                        "primary" => Ok(tool_call_response(json!("A chrome robot in rain"))),
                        "cosmos" => Ok(Response {
                            llm_response: LlmResponse::Agg(text_response(
                                Some("cosmos".to_string()),
                                "Image: output.png",
                            )),
                            metadata: None,
                        }),
                        other => panic!("unexpected target {other}"),
                    }
                }
            },
        )
        .await?;

        assert_eq!(
            trace
                .iter()
                .map(|decision| decision.selected_model_id())
                .collect::<Vec<_>>(),
            ["primary", "cosmos"]
        );
        assert!(trace[0].is_answer_call());
        assert!(trace[1].is_answer_call());
        let media_prompt = {
            let calls = calls.lock().expect("recording lock");
            switchyard_protocol::prompt_text(&calls[1].1.llm_request)
        };
        assert_eq!(media_prompt, "A chrome robot in rain");
        let aggregate = response
            .llm_response
            .into_agg()
            .await
            .map_err(|error| crate::LibsyError::external("aggregating test response", error))?;
        assert!(completion_text(&aggregate).contains("output.png"));
        Ok(())
    }

    #[tokio::test]
    async fn rejects_reserved_tool_collision() {
        let mut request = request(false);
        request.llm_request.tools.push(super::media_tool());
        let algorithm: Arc<dyn Algorithm> = Arc::new(ModelAsTool::new("primary", "cosmos"));
        let result = test_drive(algorithm, request, |_decision, _request| async move {
            unreachable!("collision must fail before a model call")
        })
        .await;

        assert!(matches!(
            result,
            Err(crate::LibsyError::AlgorithmError { message }) if message.contains("reserved tool")
        ));
    }

    #[tokio::test]
    async fn rejects_empty_prompt_and_parallel_media_call() {
        for response in [tool_call_response(json!("  ")), {
            let mut response = tool_call_response(json!("A robot"));
            let LlmResponse::Agg(aggregate) = &mut response.llm_response else {
                unreachable!()
            };
            aggregate.outputs[0]
                .content
                .push(ContentBlock::ToolCall(ToolCall {
                    id: "shell-1".to_string(),
                    name: "shell".to_string(),
                    arguments: json!({"command": "pwd"}),
                }));
            response
        }] {
            let response = Mutex::new(Some(response));
            let algorithm: Arc<dyn Algorithm> = Arc::new(ModelAsTool::new("primary", "cosmos"));
            let result = test_drive(algorithm, request(false), move |_decision, _request| {
                let response = response.lock().expect("response lock").take();
                async move { Ok(response.expect("one primary call")) }
            })
            .await;
            assert!(matches!(
                result,
                Err(crate::LibsyError::AlgorithmError { .. })
            ));
        }
    }
}
