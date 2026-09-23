// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Plans coding tasks on a capable model, then hands execution to an efficient model.

use std::collections::HashSet;
use std::sync::Arc;

use parking_lot::Mutex;
use switchyard_protocol::{Category, ContentBlock, Request};

use super::util::prompts::{append_note, drop_exact_replay, prepend_system_prompt};
use super::util::tool_signals::ToolSignals;
use crate::core::algorithm::{Algorithm, Driver, RoutingIdentity};
use crate::{LibsyError, Result, RoutingOutcome};

/// Default instruction added while the capable model is planning.
pub const DEFAULT_PLANNING_PROMPT: &str =
    include_str!("../prompts/plan-execute/planning-system-prompt.md");

const MAX_EXECUTING_SESSIONS: usize = 4_096;

/// Configuration for [`PlanExecute`].
#[derive(Clone, Debug)]
pub struct PlanExecuteConfig {
    /// System instruction added until the first edit or write tool call.
    pub planning_prompt: String,
    /// Optional instruction appended to the handoff request.
    pub handoff_prompt: Option<String>,
    /// Replays visible planner reasoning summaries as assistant text at handoff.
    pub planner_reasoning_as_text: bool,
}

impl Default for PlanExecuteConfig {
    fn default() -> Self {
        Self {
            planning_prompt: DEFAULT_PLANNING_PROMPT.trim().to_string(),
            handoff_prompt: None,
            planner_reasoning_as_text: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Phase {
    Plan,
    Handoff,
    Execute,
}

/// Routes planning turns to the runtime capable model, then latches execution
/// to the runtime efficient model after the first recorded mutation.
pub struct PlanExecute {
    config: PlanExecuteConfig,
    executing_sessions: Mutex<HashSet<RoutingIdentity>>,
}

impl PlanExecute {
    /// Creates a plan/execute router.
    ///
    /// Returns an error when either configured prompt is blank.
    pub fn new(config: PlanExecuteConfig) -> Result<Self> {
        if config.planning_prompt.trim().is_empty() {
            return Err(LibsyError::AlgorithmError {
                message: "planning_prompt must not be empty".to_string(),
            });
        }
        if config
            .handoff_prompt
            .as_deref()
            .is_some_and(|prompt| prompt.trim().is_empty())
        {
            return Err(LibsyError::AlgorithmError {
                message: "handoff_prompt must not be empty".to_string(),
            });
        }
        Ok(Self {
            config,
            executing_sessions: Mutex::new(HashSet::new()),
        })
    }

    fn phase(&self, request: &Request) -> Phase {
        let signals = ToolSignals::from_request(request, None);
        let mutation_seen = signals.edit_count > 0 || signals.write_count > 0;
        let Some(identity) = RoutingIdentity::from_request(request) else {
            return if mutation_seen {
                Phase::Handoff
            } else {
                Phase::Plan
            };
        };

        let mut sessions = self.executing_sessions.lock();
        let phase = if sessions.contains(&identity) {
            Phase::Execute
        } else if mutation_seen {
            if sessions.len() >= MAX_EXECUTING_SESSIONS
                && let Some(evicted) = sessions.iter().next().cloned()
            {
                sessions.remove(&evicted);
            }
            sessions.insert(identity.clone());
            Phase::Handoff
        } else {
            Phase::Plan
        };
        if request
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.session_final)
            == Some(true)
        {
            sessions.remove(&identity);
        }
        phase
    }

    fn replay_planner_reasoning_as_text(request: &mut Request) -> usize {
        let mut converted = 0;
        for message in &mut request.llm_request.messages {
            message.content = std::mem::take(&mut message.content)
                .into_iter()
                .filter_map(|block| match block {
                    ContentBlock::Reasoning { text, .. } => {
                        converted += 1;
                        (!text.is_empty()).then_some(ContentBlock::Text { text })
                    }
                    other => Some(other),
                })
                .collect();
        }
        request
            .llm_request
            .messages
            .retain(|message| !message.content.is_empty());
        if converted > 0 {
            drop_exact_replay(request);
        }
        converted
    }

    fn route_to(driver: &Driver, category: Category, request: Request) -> Result<RoutingOutcome> {
        let models = driver.models_for(&category);
        let Some((selected, fallbacks)) = models.split_first() else {
            return Err(LibsyError::AlgorithmError {
                message: format!("no models available for category {}", category.as_str()),
            });
        };
        Ok(RoutingOutcome::route_to(
            selected.clone(),
            fallbacks.to_vec(),
            request,
        ))
    }
}

#[async_trait::async_trait]
impl Algorithm for PlanExecute {
    fn name(&self) -> &str {
        "plan_execute"
    }

    async fn route(
        self: Arc<Self>,
        driver: Driver,
        mut request: Request,
    ) -> Result<RoutingOutcome> {
        match self.phase(&request) {
            Phase::Plan => {
                prepend_system_prompt(&mut request, &self.config.planning_prompt);
                tracing::debug!(phase = "plan", "plan-execute selected capable tier");
                Self::route_to(&driver, Category::Capable, request)
            }
            Phase::Handoff => {
                let reasoning_converted = if self.config.planner_reasoning_as_text {
                    Self::replay_planner_reasoning_as_text(&mut request)
                } else {
                    0
                };
                let prompt_applied = if let Some(prompt) = &self.config.handoff_prompt {
                    append_note(&mut request, prompt);
                    true
                } else {
                    false
                };
                tracing::debug!(
                    phase = "handoff",
                    handoff_prompt_applied = prompt_applied,
                    planner_reasoning_converted = reasoning_converted,
                    "plan-execute selected efficient tier"
                );
                Self::route_to(&driver, Category::Efficient, request)
            }
            Phase::Execute => {
                tracing::debug!(phase = "execute", "plan-execute selected efficient tier");
                Self::route_to(&driver, Category::Efficient, request)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use serde_json::json;
    use switchyard_protocol::{
        ContentBlock, LlmRequest, Message, Metadata, ModelId, Request, Role, ToolCall,
    };

    use super::*;
    use crate::RuntimeModels;
    use crate::core::testing::{reply, test_drive_with_models};

    const CAPABLE: &str = "model/capable";
    const EFFICIENT: &str = "model/efficient";

    fn algorithm(config: PlanExecuteConfig) -> Arc<dyn Algorithm> {
        Arc::new(PlanExecute::new(config).expect("config should be valid"))
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

    fn models() -> RuntimeModels {
        RuntimeModels::new(HashMap::from([
            (Category::Capable, vec![ModelId::from(CAPABLE)]),
            (Category::Efficient, vec![ModelId::from(EFFICIENT)]),
        ]))
    }

    async fn route_and_capture(
        algorithm: Arc<dyn Algorithm>,
        request: Request,
    ) -> (ModelId, Request) {
        let captured = Arc::new(Mutex::new(None));
        let capture = Arc::clone(&captured);
        let (selected, _) =
            test_drive_with_models(algorithm, request, models(), move |_target, request| {
                let capture = Arc::clone(&capture);
                async move {
                    *capture.lock() = Some(request);
                    Ok(reply("ok"))
                }
            })
            .await
            .expect("routing should succeed");
        let request = captured
            .lock()
            .take()
            .expect("answer request should be captured");
        (selected, request)
    }

    #[tokio::test]
    async fn plans_then_hands_off_and_latches_execution() {
        const HANDOFF: &str = "Continue from the plan and repository evidence.";
        let algorithm = algorithm(PlanExecuteConfig {
            handoff_prompt: Some(HANDOFF.to_string()),
            planner_reasoning_as_text: true,
            ..PlanExecuteConfig::default()
        });

        let read_only = request(
            vec![tool_call(
                "exec_command",
                json!({"cmd": "rg parser crates"}),
            )],
            Some("task-1"),
        );
        let (selected, routed) = route_and_capture(Arc::clone(&algorithm), read_only).await;
        assert_eq!(selected, CAPABLE);
        assert_eq!(routed.llm_request.instructions.len(), 1);

        let first_edit = request(
            vec![Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::Reasoning {
                        text: "The parser needs a boundary check.".to_string(),
                        signature: Some("planner-signature".to_string()),
                        details: vec![json!({"type": "reasoning.encrypted", "data": "opaque"})],
                    },
                    ContentBlock::ToolCall(ToolCall {
                        id: "call-1".to_string(),
                        name: "apply_patch".to_string(),
                        arguments: json!({"patch": "*** Begin Patch"}),
                    }),
                ],
            }],
            Some("task-1"),
        );
        let (selected, routed) = route_and_capture(Arc::clone(&algorithm), first_edit).await;
        assert_eq!(selected, EFFICIENT);
        assert_eq!(
            routed.llm_request.messages[0].content[0],
            ContentBlock::Text {
                text: "The parser needs a boundary check.".to_string()
            }
        );
        assert_eq!(
            routed.llm_request.messages.last(),
            Some(&Message::text(Role::User, HANDOFF))
        );

        let mut final_request = request(
            vec![Message::text(Role::User, "Continue after compaction")],
            Some("task-1"),
        );
        final_request
            .metadata
            .as_mut()
            .expect("session metadata should exist")
            .session_final = Some(true);
        let (selected, routed) = route_and_capture(Arc::clone(&algorithm), final_request).await;
        assert_eq!(selected, EFFICIENT);
        assert!(routed.llm_request.instructions.is_empty());
        assert_eq!(routed.llm_request.messages.len(), 1);

        let reused = request(vec![Message::text(Role::User, "New task")], Some("task-1"));
        let (selected, _) = route_and_capture(algorithm, reused).await;
        assert_eq!(selected, CAPABLE);
    }

    #[tokio::test]
    async fn mutation_without_a_session_uses_the_efficient_tier() {
        let messages = vec![tool_call(
            "exec_command",
            json!({"cmd": "printf 'done\\n' > task.txt"}),
        )];

        let (selected, routed) = route_and_capture(
            algorithm(PlanExecuteConfig::default()),
            request(messages.clone(), None),
        )
        .await;

        assert_eq!(selected, EFFICIENT);
        assert_eq!(routed.llm_request.messages, messages);
        assert!(routed.llm_request.instructions.is_empty());
    }

    #[tokio::test]
    async fn editor_view_keeps_planning() {
        let messages = vec![tool_call(
            "str_replace_based_edit_tool",
            json!({"command": "view", "path": "/app/main.py"}),
        )];

        let (selected, _) = route_and_capture(
            algorithm(PlanExecuteConfig::default()),
            request(messages, None),
        )
        .await;

        assert_eq!(selected, CAPABLE);
    }

    #[test]
    fn rejects_blank_prompts() {
        for config in [
            PlanExecuteConfig {
                planning_prompt: "  ".to_string(),
                ..PlanExecuteConfig::default()
            },
            PlanExecuteConfig {
                handoff_prompt: Some("  ".to_string()),
                ..PlanExecuteConfig::default()
            },
        ] {
            assert!(matches!(
                PlanExecute::new(config),
                Err(LibsyError::AlgorithmError { .. })
            ));
        }
    }
}
