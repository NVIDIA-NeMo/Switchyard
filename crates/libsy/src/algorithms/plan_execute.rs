// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Starts coding tasks on a capable planner, then hands execution to an efficient model.

use std::collections::HashSet;
use std::sync::Arc;

use parking_lot::Mutex;
use switchyard_protocol::{ModelId, Request};

use super::util::prompts::{SystemPromptProcessor, TargetPrompts};
use super::util::tool_signals::ToolSignals;
use crate::core::algorithm::{Algorithm, Driver, RoutingIdentity};
use crate::core::processor::{Event, Processor};
use crate::{LibsyError, Result, RoutingOutcome};

/// Default instruction prepended while the capable model is planning.
pub const DEFAULT_PLANNING_PROMPT: &str =
    include_str!("../prompts/plan-execute/planning-system-prompt.md");

/// Maximum session latches retained by one router instance.
const MAX_EXECUTING_SESSIONS: usize = 4_096;

/// Configuration for [`PlanExecute`].
#[derive(Clone, Debug)]
pub struct PlanExecuteConfig {
    /// System instruction prepended until the first edit or write tool call.
    pub planning_prompt: String,
}

impl Default for PlanExecuteConfig {
    fn default() -> Self {
        Self {
            planning_prompt: DEFAULT_PLANNING_PROMPT.trim().to_string(),
        }
    }
}

/// Routes planning turns to a capable model and all turns after the first edit
/// to an efficient model while preserving the caller's full trajectory.
pub struct PlanExecute {
    capable: ModelId,
    efficient: ModelId,
    planning_prompt: SystemPromptProcessor,
    executing_sessions: Mutex<HashSet<RoutingIdentity>>,
}

impl PlanExecute {
    /// Creates a plan/execute router.
    ///
    /// Returns an error when the planning prompt is empty.
    pub fn new(capable: ModelId, efficient: ModelId, config: PlanExecuteConfig) -> Result<Self> {
        if config.planning_prompt.trim().is_empty() {
            return Err(LibsyError::AlgorithmError {
                message: "planning_prompt must not be empty".to_string(),
            });
        }
        let planning_prompt = SystemPromptProcessor::new(
            TargetPrompts::default().with(capable.clone(), config.planning_prompt),
        );
        Ok(Self {
            capable,
            efficient,
            planning_prompt,
            executing_sessions: Mutex::new(HashSet::new()),
        })
    }

    /// Whether this request is in execution, latching the transition for keyed sessions.
    fn is_executing(&self, request: &Request) -> bool {
        let signals = ToolSignals::from_request(request, None);
        let mutation_seen = signals.edit_count > 0 || signals.write_count > 0;
        let Some(identity) = RoutingIdentity::from_request(request) else {
            return mutation_seen;
        };

        let mut sessions = self.executing_sessions.lock();
        let executing = if mutation_seen {
            if sessions.len() >= MAX_EXECUTING_SESSIONS
                && !sessions.contains(&identity)
                && let Some(evicted) = sessions.iter().next().cloned()
            {
                sessions.remove(&evicted);
            }
            sessions.insert(identity.clone());
            true
        } else {
            sessions.contains(&identity)
        };
        if request
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.session_final)
            == Some(true)
        {
            sessions.remove(&identity);
        }
        executing
    }
}

#[async_trait::async_trait]
impl Algorithm for PlanExecute {
    fn name(&self) -> &str {
        "plan_execute"
    }

    async fn route(
        self: Arc<Self>,
        _driver: Driver,
        mut request: Request,
    ) -> Result<RoutingOutcome> {
        if self.is_executing(&request) {
            tracing::info!(target = %self.efficient, phase = "execute", "plan-execute selected target");
            Ok(RoutingOutcome::route_to(
                self.efficient.clone(),
                Vec::new(),
                request,
            ))
        } else {
            self.planning_prompt
                .process(
                    &mut (),
                    Event::Decision {
                        request: &mut request,
                        selected_model_id: &self.capable,
                    },
                )
                .await?;
            tracing::info!(target = %self.capable, phase = "plan", "plan-execute selected target");
            Ok(RoutingOutcome::route_to(
                self.capable.clone(),
                Vec::new(),
                request,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use serde_json::json;
    use switchyard_protocol::{
        ContentBlock, InstructionBlock, LlmRequest, Message, Metadata, Request, Role, ToolCall,
    };

    use super::*;
    use crate::core::testing::{reply, test_drive};

    fn algorithm() -> Arc<dyn Algorithm> {
        Arc::new(
            PlanExecute::new(
                ModelId::from("model/capable"),
                ModelId::from("model/efficient"),
                PlanExecuteConfig::default(),
            )
            .expect("default config should be valid"),
        )
    }

    fn request(messages: Vec<Message>, session_id: Option<&str>) -> Request {
        Request {
            llm_request: LlmRequest {
                model: Some("switchyard/plan-execute".to_string()),
                messages,
                ..LlmRequest::default()
            },
            metadata: session_id.map(|session_id| Metadata {
                session_id: Some(session_id.to_string()),
                ..Metadata::default()
            }),
            ..Request::default()
        }
    }

    fn tool_call(name: &str, arguments: serde_json::Value) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "call-1".to_string(),
                name: name.to_string(),
                arguments,
            })],
        }
    }

    async fn route_and_capture(
        algorithm: Arc<dyn Algorithm>,
        request: Request,
    ) -> (ModelId, Request) {
        let captured = Arc::new(Mutex::new(None));
        let capture = Arc::clone(&captured);
        let (selected, _) = test_drive(algorithm, request, move |_target, request| {
            let capture = Arc::clone(&capture);
            async move {
                *capture.lock().expect("capture lock should be available") = Some(request);
                Ok(reply("ok"))
            }
        })
        .await
        .expect("routing should succeed");
        let request = captured
            .lock()
            .expect("capture lock should be available")
            .take()
            .expect("answer request should be captured");
        (selected, request)
    }

    #[tokio::test]
    async fn initial_turn_uses_capable_model_with_planning_prefix() {
        let messages = vec![Message::text(Role::User, "fix the parser")];
        let (selected, routed) =
            route_and_capture(algorithm(), request(messages.clone(), Some("task-1"))).await;

        assert_eq!(selected, "model/capable");
        assert_eq!(routed.llm_request.messages, messages);
        assert_eq!(routed.llm_request.instructions.len(), 1);
        assert_eq!(routed.llm_request.instructions[0].role, Role::System);
        assert_eq!(
            routed.llm_request.instructions[0].content,
            vec![ContentBlock::Text {
                text: DEFAULT_PLANNING_PROMPT.trim().to_string()
            }]
        );
    }

    #[tokio::test]
    async fn read_only_tool_calls_remain_in_planning() {
        let messages = vec![
            Message::text(Role::User, "fix the parser"),
            tool_call("exec_command", json!({"cmd": "rg parser crates"})),
        ];

        let (selected, routed) =
            route_and_capture(algorithm(), request(messages.clone(), None)).await;

        assert_eq!(selected, "model/capable");
        assert_eq!(routed.llm_request.messages, messages);
        assert_eq!(routed.llm_request.instructions.len(), 1);
    }

    #[tokio::test]
    async fn first_edit_switches_to_efficient_and_keeps_the_trajectory() {
        let messages = vec![
            Message::text(Role::User, "fix the parser"),
            Message::text(Role::Assistant, "I will update the parser now."),
            tool_call("apply_patch", json!({"patch": "*** Begin Patch"})),
        ];
        let mut input = request(messages.clone(), Some("task-2"));
        input.llm_request.instructions.push(InstructionBlock {
            role: Role::Developer,
            content: vec![ContentBlock::Text {
                text: "keep the public API stable".to_string(),
            }],
        });

        let (selected, routed) = route_and_capture(algorithm(), input).await;

        assert_eq!(selected, "model/efficient");
        assert_eq!(routed.llm_request.messages, messages);
        assert_eq!(routed.llm_request.instructions.len(), 1);
        assert_eq!(routed.llm_request.instructions[0].role, Role::Developer);
    }

    #[tokio::test]
    async fn shell_file_write_switches_to_execution() {
        let messages = vec![tool_call(
            "exec_command",
            json!({"cmd": "python -c 'from pathlib import Path; Path(\"x\").write_text(\"y\")'"}),
        )];

        let (selected, routed) = route_and_capture(algorithm(), request(messages, None)).await;

        assert_eq!(selected, "model/efficient");
        assert!(routed.llm_request.instructions.is_empty());
    }

    #[tokio::test]
    async fn execution_latches_by_session_after_history_compaction() {
        let algorithm = algorithm();
        let edit = request(
            vec![tool_call("write_file", json!({"path": "src/lib.rs"}))],
            Some("task-3"),
        );
        let (selected, _) = route_and_capture(Arc::clone(&algorithm), edit).await;
        assert_eq!(selected, "model/efficient");

        let compacted = request(
            vec![Message::text(
                Role::User,
                "Continue from the compacted summary",
            )],
            Some("task-3"),
        );
        let (selected, routed) = route_and_capture(algorithm, compacted).await;

        assert_eq!(selected, "model/efficient");
        assert!(routed.llm_request.instructions.is_empty());
    }

    #[tokio::test]
    async fn final_request_uses_then_releases_the_session_latch() {
        let algorithm = algorithm();
        let edit = request(
            vec![tool_call("write_file", json!({"path": "src/lib.rs"}))],
            Some("task-4"),
        );
        let (selected, _) = route_and_capture(Arc::clone(&algorithm), edit).await;
        assert_eq!(selected, "model/efficient");

        let mut final_request = request(vec![Message::text(Role::User, "Finish")], Some("task-4"));
        final_request
            .metadata
            .as_mut()
            .expect("session metadata should exist")
            .session_final = Some(true);
        let (selected, _) = route_and_capture(Arc::clone(&algorithm), final_request).await;
        assert_eq!(selected, "model/efficient");

        let reused = request(vec![Message::text(Role::User, "New task")], Some("task-4"));
        let (selected, _) = route_and_capture(algorithm, reused).await;
        assert_eq!(selected, "model/capable");
    }

    #[test]
    fn empty_planning_prompt_is_rejected() {
        let result = PlanExecute::new(
            ModelId::from("model/capable"),
            ModelId::from("model/efficient"),
            PlanExecuteConfig {
                planning_prompt: "  ".to_string(),
            },
        );

        assert!(matches!(result, Err(LibsyError::AlgorithmError { .. })));
    }
}
