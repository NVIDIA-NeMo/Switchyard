// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Plans on one model, executes on another, then reviews the completion once.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;
use switchyard_protocol::{Category, ContentBlock, Message, Request, WireFormat};

use super::advisor_gate::{AdvisorGate, AdvisorGateConfig, GateTrigger, ReviewContext};
use super::plan_execute::{DEFAULT_PLANNING_PROMPT, ExecutionTracker};
use super::util::prompts::prepend_system_prompt;
use super::util::tool_signals::is_mutating_tool_call;
use crate::core::algorithm::{Algorithm, Driver, RoutingIdentity};
use crate::{LibsyError, Result, RoutingOutcome};

/// Terminal response pattern validated against Codex DeepSWE trajectories.
pub const DEFAULT_TERMINAL_PATTERN: &str =
    r"(?i)^\s*(?:#{1,6}\s*)?(?:\*\*)?(?:implemented|completed|done|erledigt|fertig)(?:\*\*)?\b";

/// Default request appended when the reviewer examines completed execution.
pub const DEFAULT_REVIEWER_PROMPT: &str =
    include_str!("../prompts/plan-execute/reviewer-prompt.md");

/// Default verify-first instruction prepended to reviewer feedback.
pub const DEFAULT_REDO_FEEDBACK_PREFIX: &str =
    include_str!("../prompts/plan-execute/redo-feedback-prefix.md");

const MAX_PLANNER_CHECKPOINTS: usize = 4_096;

/// Configuration for [`PlanExecuteReview`].
#[derive(Clone, Debug)]
pub struct PlanExecuteReviewConfig {
    /// System instruction prepended while the planner owns the session.
    pub planning_prompt: String,
    /// Request appended when the reviewer examines the completion.
    pub reviewer_prompt: String,
    /// Prepended to REDO feedback before execution resumes.
    pub redo_feedback_prefix: String,
    /// Pattern that identifies an executor completion response.
    pub terminal_pattern: String,
    /// Maximum output tokens for the review call.
    pub reviewer_max_tokens: u64,
    /// Lets the completion through when the review call fails.
    pub fail_open: bool,
}

impl Default for PlanExecuteReviewConfig {
    fn default() -> Self {
        Self {
            planning_prompt: DEFAULT_PLANNING_PROMPT.trim().to_string(),
            reviewer_prompt: DEFAULT_REVIEWER_PROMPT.trim().to_string(),
            redo_feedback_prefix: DEFAULT_REDO_FEEDBACK_PREFIX.trim().to_string() + "\n",
            terminal_pattern: DEFAULT_TERMINAL_PATTERN.to_string(),
            reviewer_max_tokens: 2048,
            fail_open: true,
        }
    }
}

/// Plan and execute router with one planner-aware terminal review.
pub struct PlanExecuteReview {
    planning_prompt: String,
    phase: ExecutionTracker,
    review_gate: Arc<AdvisorGate>,
    planner_checkpoints: Mutex<HashMap<RoutingIdentity, PlannerCheckpoint>>,
}

#[derive(Clone)]
/// Minimal handoff history needed to restore Sol's review prefix after compaction.
struct PlannerCheckpoint {
    messages: Vec<Message>,
    responses_input: Option<Vec<serde_json::Value>>,
}

impl PlanExecuteReview {
    /// Creates a plan, execute, and review router.
    pub fn new(config: PlanExecuteReviewConfig) -> Result<Self> {
        if config.planning_prompt.trim().is_empty() {
            return Err(LibsyError::AlgorithmError {
                message: "planning_prompt must not be empty".to_string(),
            });
        }
        if config.reviewer_prompt.trim().is_empty() {
            return Err(LibsyError::AlgorithmError {
                message: "reviewer_prompt must not be empty".to_string(),
            });
        }
        let planning_prompt_text = config.planning_prompt.clone();
        let gate_config = AdvisorGateConfig {
            reviewer_system_prompt: config.reviewer_prompt,
            reviewer_prefix_prompt: Some(planning_prompt_text.clone()),
            redo_feedback_prefix: config.redo_feedback_prefix,
            review_context: ReviewContext::ExecutionDelta,
            gate_trigger: GateTrigger::Pattern(config.terminal_pattern),
            gate_require_no_tool_call: true,
            max_reviews: 1,
            gate_stall_turns: 0,
            advisor_max_tokens: config.reviewer_max_tokens,
            fail_open: config.fail_open,
            ..AdvisorGateConfig::default()
        };
        let review_gate = Arc::new(AdvisorGate::new(gate_config)?);

        Ok(Self {
            planning_prompt: planning_prompt_text,
            phase: ExecutionTracker::new(),
            review_gate,
            planner_checkpoints: Mutex::new(HashMap::new()),
        })
    }

    /// Returns the live conversation or restores its saved handoff after compaction.
    fn review_base(&self, request: &Request) -> Option<Request> {
        let identity = RoutingIdentity::from_request(request)?;
        let request_has_mutation = request
            .llm_request
            .messages
            .iter()
            .any(message_has_mutation);
        let mut checkpoints = self.planner_checkpoints.lock();
        if request_has_mutation && !checkpoints.contains_key(&identity) {
            if checkpoints.len() >= MAX_PLANNER_CHECKPOINTS
                && let Some(evicted) = checkpoints.keys().next().cloned()
            {
                checkpoints.remove(&evicted);
            }
            checkpoints.insert(identity.clone(), planner_checkpoint(request));
        }
        let review_base = checkpoints.get(&identity).map_or_else(
            || request.clone(),
            |prefix| {
                if request_starts_with(prefix, request) {
                    request.clone()
                } else {
                    restore_planner_prefix(prefix, request)
                }
            },
        );
        Some(review_base)
    }

    /// Drops saved handoff state when the client marks the session complete.
    fn evict_checkpoint(&self, request: &Request) {
        if let Some(identity) = RoutingIdentity::from_request(request) {
            self.planner_checkpoints.lock().remove(&identity);
        }
    }
}

#[async_trait::async_trait]
impl Algorithm for PlanExecuteReview {
    fn name(&self) -> &str {
        "plan_execute_review"
    }

    async fn route(
        self: Arc<Self>,
        driver: Driver,
        mut request: Request,
    ) -> Result<RoutingOutcome> {
        if self.phase.is_executing(&request) {
            let session_final = request
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.session_final)
                == Some(true);
            let review_base = self.review_base(&request);
            let result = Arc::clone(&self.review_gate)
                .route_with_review_base(driver, request.clone(), review_base)
                .await;
            if session_final {
                self.evict_checkpoint(&request);
            }
            return result;
        }

        prepend_system_prompt(&mut request, &self.planning_prompt);
        let models = driver.models_for(&Category::Capable);
        let planner = models
            .first()
            .ok_or_else(|| LibsyError::AlgorithmError {
                message: "no models available for category Capable".to_string(),
            })?
            .clone();
        Ok(RoutingOutcome::route_to(
            planner,
            models[1..].to_vec(),
            request,
        ))
    }
}

/// Copies only conversation fields that can prefix a later review request.
fn planner_checkpoint(request: &Request) -> PlannerCheckpoint {
    PlannerCheckpoint {
        messages: request.llm_request.messages.clone(),
        responses_input: exact_responses_input(request).cloned(),
    }
}

/// Prepends the original handoff while retaining the current execution request's settings.
fn restore_planner_prefix(prefix: &PlannerCheckpoint, current: &Request) -> Request {
    let mut restored = current.clone();
    restored.llm_request.messages = prefix
        .messages
        .iter()
        .cloned()
        .chain(current.llm_request.messages.iter().cloned())
        .collect();

    let format = switchyard_protocol::FormatId::known(WireFormat::OpenAiResponses);
    let prefix_input = prefix.responses_input.as_ref();
    let current_input = restored
        .llm_request
        .preservation
        .requests
        .get_mut(&format)
        .and_then(|body| body.get_mut("input"))
        .and_then(serde_json::Value::as_array_mut);
    if let (Some(prefix_input), Some(current_input)) = (prefix_input, current_input) {
        *current_input = prefix_input
            .iter()
            .cloned()
            .chain(current_input.iter().cloned())
            .collect();
    }
    restored
}

/// Detects when the current request still carries the saved handoff without reconstruction.
fn request_starts_with(prefix: &PlannerCheckpoint, current: &Request) -> bool {
    if let (Some(prefix_input), Some(current_input)) = (
        prefix.responses_input.as_ref(),
        exact_responses_input(current),
    ) {
        return !prefix_input.is_empty() && current_input.starts_with(prefix_input);
    }

    let prefix_messages = &prefix.messages;
    let current_messages = &current.llm_request.messages;
    if prefix_messages.is_empty() || current_messages.len() < prefix_messages.len() {
        return false;
    }
    let split = prefix_messages.len() - 1;
    prefix_messages[..split] == current_messages[..split]
        && prefix_messages[split].role == current_messages[split].role
        && current_messages[split]
            .content
            .starts_with(&prefix_messages[split].content)
}

fn exact_responses_input(request: &Request) -> Option<&Vec<serde_json::Value>> {
    let format = switchyard_protocol::FormatId::known(WireFormat::OpenAiResponses);
    request
        .llm_request
        .preservation
        .requests
        .get(&format)
        .and_then(|body| body.get("input"))
        .and_then(serde_json::Value::as_array)
}

fn message_has_mutation(message: &Message) -> bool {
    message.content.iter().any(|block| {
        matches!(block, ContentBlock::ToolCall(call) if is_mutating_tool_call(&call.name, &call.arguments))
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use switchyard_protocol::{
        AggLlmResponse, Category, ContentBlock, LlmRequest, LlmResponse, Message, ModelId,
        Response, ResponseOutput, Role, ToolCall, ToolChoice, ToolResult,
    };

    use super::*;
    use crate::core::algorithm::RuntimeModels;
    use crate::core::testing::{reply, test_drive_with_models};

    fn review_models() -> RuntimeModels {
        HashMap::from([
            (Category::Capable, vec![ModelId::from("planner")]),
            (Category::Efficient, vec![ModelId::from("executor")]),
            (Category::Judge, vec![ModelId::from("planner")]),
        ])
        .into()
    }

    fn request(messages: Vec<Message>) -> Request {
        Request {
            llm_request: LlmRequest {
                model: Some("switchyard/review".to_string()),
                messages,
                ..LlmRequest::default()
            },
            metadata: Some(switchyard_protocol::Metadata {
                session_id: Some("task-1".to_string()),
                ..switchyard_protocol::Metadata::default()
            }),
            ..Request::default()
        }
    }

    #[test]
    fn default_terminal_pattern_matches_observed_completion_headers() {
        let pattern =
            regex::Regex::new(DEFAULT_TERMINAL_PATTERN).expect("default pattern is valid");

        assert!(pattern.is_match("**Completed**\n- Tests pass"));
        assert!(pattern.is_match("Implemented the requested change"));
        assert!(pattern.is_match("**Done**\n- Tests pass"));
        assert!(pattern.is_match("**Erledigt**\n- Tests bestehen"));
        assert!(pattern.is_match("**Fertig**\n- Tests bestehen"));
        assert!(!pattern.is_match("Implementation is complete, now running tests"));
    }

    #[tokio::test]
    async fn plans_then_reviews_completed_execution() {
        let algorithm: Arc<dyn Algorithm> = Arc::new(
            PlanExecuteReview::new(PlanExecuteReviewConfig::default())
                .expect("default config is valid"),
        );
        let calls = Arc::new(Mutex::new(Vec::new()));

        let planning_calls = Arc::clone(&calls);
        test_drive_with_models(
            Arc::clone(&algorithm),
            request(vec![Message::text(Role::User, "fix it")]),
            review_models(),
            move |model: ModelId, request: Request| {
                planning_calls
                    .lock()
                    .expect("call log is available")
                    .push((model.to_string(), request));
                async { Ok(reply("Plan complete")) }
            },
        )
        .await
        .expect("planning routes");

        let edit = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "edit-1".to_string(),
                name: "apply_patch".to_string(),
                arguments: serde_json::json!({"patch": "change"}),
            })],
        };
        let result = Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult(ToolResult {
                tool_call_id: "edit-1".to_string(),
                content: vec![ContentBlock::Text {
                    text: "done".to_string(),
                }],
                is_error: Some(false),
            })],
        };
        let patch_capture = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "final-patch".to_string(),
                name: "exec_command".to_string(),
                arguments: serde_json::json!({
                    "cmd": "deepswe_capture_model_patch && cat /logs/artifacts/model.patch"
                }),
            })],
        };
        let patch_result = Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult(ToolResult {
                tool_call_id: "final-patch".to_string(),
                content: vec![ContentBlock::Text {
                    text: "diff --git a/parser.rs b/parser.rs".to_string(),
                }],
                is_error: Some(false),
            })],
        };
        let execution_calls = Arc::clone(&calls);
        test_drive_with_models(
            algorithm,
            request(vec![
                Message::text(Role::User, "fix it"),
                Message::text(Role::Assistant, "Plan complete"),
                edit,
                result,
                patch_capture,
                patch_result,
            ]),
            review_models(),
            move |model: ModelId, request: Request| {
                let is_review = model.as_str() == "planner"
                    && request.llm_request.messages.last().is_some_and(|message| {
                        message.text_content("\n").is_some_and(|text| {
                            text.contains("Review Luna's completed coding task")
                        })
                    });
                execution_calls
                    .lock()
                    .expect("call log is available")
                    .push((model.to_string(), request));
                async move {
                    if is_review {
                        Ok(reply("APPROVE"))
                    } else {
                        Ok(reply("Completed the task"))
                    }
                }
            },
        )
        .await
        .expect("execution routes");

        let calls = calls.lock().expect("call log is available");
        assert_eq!(
            calls
                .iter()
                .map(|(model, _)| model.as_str())
                .collect::<Vec<_>>(),
            vec!["planner", "executor", "planner"]
        );
        let planning = &calls[0].1.llm_request;
        assert_eq!(planning.instructions.len(), 1);
        let review = &calls[2].1.llm_request;
        assert_eq!(review.instructions, planning.instructions);
        assert_eq!(
            review.messages.last().map(|message| message.role),
            Some(Role::User)
        );
        assert_eq!(review.tool_choice, Some(ToolChoice::None));
        let encoded_review = serde_json::to_string(&review.messages).expect("review serializes");
        assert!(encoded_review.contains("final-patch"));
        assert!(encoded_review.contains("diff --git a/parser.rs b/parser.rs"));
        assert!(
            review.messages[review.messages.len() - 2]
                .text_content("\n")
                .expect("completion text")
                .starts_with("Completed")
        );
    }

    #[tokio::test]
    async fn planner_review_redo_returns_feedback_to_executor() {
        let algorithm: Arc<dyn Algorithm> = Arc::new(
            PlanExecuteReview::new(PlanExecuteReviewConfig::default())
                .expect("default config is valid"),
        );
        let edit = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "edit-1".to_string(),
                name: "apply_patch".to_string(),
                arguments: serde_json::json!({"patch": "change"}),
            })],
        };
        let result = Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult(ToolResult {
                tool_call_id: "edit-1".to_string(),
                content: vec![ContentBlock::Text {
                    text: "done".to_string(),
                }],
                is_error: Some(false),
            })],
        };
        let calls = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&calls);
        let (selected, _) = test_drive_with_models(
            algorithm,
            request(vec![Message::text(Role::User, "fix it"), edit, result]),
            review_models(),
            move |model: ModelId, request: Request| {
                let last = request
                    .llm_request
                    .messages
                    .last()
                    .and_then(|message| message.text_content("\n"))
                    .unwrap_or_default();
                captured
                    .lock()
                    .expect("call log is available")
                    .push((model.to_string(), request));
                async move {
                    if last.contains("Review Luna's completed coding task") {
                        Ok(reply("REDO\nCover the missing edge case."))
                    } else if last.contains(DEFAULT_REDO_FEEDBACK_PREFIX.trim()) {
                        Ok(reply("Completed after repair"))
                    } else {
                        Ok(Response {
                            llm_response: LlmResponse::Agg(AggLlmResponse {
                                outputs: vec![ResponseOutput {
                                    role: Role::Assistant,
                                    content: vec![
                                        ContentBlock::Reasoning {
                                            text: "checked the implementation".to_string(),
                                            signature: None,
                                            details: Vec::new(),
                                        },
                                        ContentBlock::Text {
                                            text: "Completed the initial attempt".to_string(),
                                        },
                                    ],
                                    stop_reason: None,
                                }],
                                ..AggLlmResponse::default()
                            }),
                            metadata: None,
                        })
                    }
                }
            },
        )
        .await
        .expect("redo flow completes");

        assert_eq!(selected.as_str(), "executor");
        let calls = calls.lock().expect("call log is available");
        assert_eq!(
            calls
                .iter()
                .map(|(model, _)| model.as_str())
                .collect::<Vec<_>>(),
            vec!["executor", "planner", "executor"]
        );
        let redo = &calls[2].1.llm_request.messages;
        assert!(
            redo[redo.len() - 2]
                .content
                .iter()
                .any(|block| matches!(block, ContentBlock::Reasoning { .. }))
        );
        assert!(
            redo[redo.len() - 2]
                .text_content("\n")
                .expect("completion echo")
                .starts_with("Completed the initial attempt")
        );
        assert!(
            redo.last()
                .and_then(|message| message.text_content("\n"))
                .expect("redo feedback")
                .contains("Cover the missing edge case.")
        );
        assert!(
            redo.last()
                .and_then(|message| message.text_content("\n"))
                .expect("redo feedback")
                .contains("First reproduce the concern")
        );
    }

    #[test]
    fn review_restores_planner_prefix_after_execution_compaction() {
        let algorithm = PlanExecuteReview::new(PlanExecuteReviewConfig::default())
            .expect("default config is valid");
        let mut handoff = request(vec![
            Message::text(Role::User, "fix it"),
            Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::Reasoning {
                        text: "the original plan".to_string(),
                        signature: Some("planner-signature".to_string()),
                        details: Vec::new(),
                    },
                    ContentBlock::ToolCall(ToolCall {
                        id: "edit-1".to_string(),
                        name: "apply_patch".to_string(),
                        arguments: serde_json::json!({"patch": "change"}),
                    }),
                ],
            },
        ]);
        handoff.llm_request.preservation.requests.insert(
            "openai_responses".into(),
            serde_json::json!({
                "input": [
                    {"type": "message", "role": "user", "content": "fix it"},
                    {"type": "reasoning", "id": "plan", "encrypted_content": "opaque-plan"},
                    {"type": "function_call", "call_id": "edit-1", "name": "apply_patch", "arguments": "{}"}
                ]
            }),
        );
        algorithm.review_base(&handoff).expect("handoff is keyed");

        let mut compacted = request(vec![
            Message::text(
                Role::User,
                "A previous model compacted the execution context.",
            ),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolCall(ToolCall {
                    id: "later-edit".to_string(),
                    name: "apply_patch".to_string(),
                    arguments: serde_json::json!({"patch": "later change"}),
                })],
            },
        ]);
        compacted.llm_request.preservation.requests.insert(
            "openai_responses".into(),
            serde_json::json!({
                "input": [
                    {"type": "message", "role": "user", "content": "execution summary"},
                    {"type": "function_call", "call_id": "later-edit", "name": "apply_patch", "arguments": "{}"}
                ]
            }),
        );
        let restored = algorithm
            .review_base(&compacted)
            .expect("compacted request is keyed");

        let messages =
            serde_json::to_string(&restored.llm_request.messages).expect("messages serialize");
        assert!(messages.contains("fix it"));
        assert!(messages.contains("the original plan"));
        assert!(messages.contains("planner-signature"));
        assert!(messages.contains("compacted the execution context"));
        assert!(messages.contains("later-edit"));
        assert!(messages.contains("edit-1"));

        let input = restored
            .llm_request
            .preservation
            .requests
            .get(&switchyard_protocol::FormatId::known(
                WireFormat::OpenAiResponses,
            ))
            .and_then(|body| body.get("input"))
            .and_then(serde_json::Value::as_array)
            .expect("exact Responses input survives");
        assert_eq!(input.len(), 5);
        assert_eq!(input[0]["content"], "fix it");
        assert_eq!(input[1]["encrypted_content"], "opaque-plan");
        assert_eq!(input[2]["call_id"], "edit-1");
        assert_eq!(input[3]["content"], "execution summary");
        assert_eq!(input[4]["call_id"], "later-edit");
    }

    #[test]
    fn completed_session_evicts_planner_checkpoint() {
        let algorithm = PlanExecuteReview::new(PlanExecuteReviewConfig::default())
            .expect("default config is valid");
        let handoff = request(vec![Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "edit-1".to_string(),
                name: "apply_patch".to_string(),
                arguments: serde_json::json!({"patch": "change"}),
            })],
        }]);
        algorithm.review_base(&handoff).expect("handoff is keyed");
        assert_eq!(algorithm.planner_checkpoints.lock().len(), 1);

        algorithm.evict_checkpoint(&handoff);

        assert!(algorithm.planner_checkpoints.lock().is_empty());
    }

    #[test]
    fn planner_checkpoints_stay_bounded() {
        let algorithm = PlanExecuteReview::new(PlanExecuteReviewConfig::default())
            .expect("default config is valid");
        for index in 0..=MAX_PLANNER_CHECKPOINTS {
            let mut handoff = request(vec![Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolCall(ToolCall {
                    id: "edit-1".to_string(),
                    name: "apply_patch".to_string(),
                    arguments: serde_json::json!({"patch": "change"}),
                })],
            }]);
            handoff
                .metadata
                .as_mut()
                .expect("request has metadata")
                .session_id = Some(format!("task-{index}"));
            algorithm.review_base(&handoff).expect("handoff is keyed");
        }

        assert_eq!(
            algorithm.planner_checkpoints.lock().len(),
            MAX_PLANNER_CHECKPOINTS
        );
    }
}
