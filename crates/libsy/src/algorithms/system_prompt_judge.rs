// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Judge-selected system prompt injection from a hidden action database.

use std::collections::BTreeSet;
use std::sync::Arc;

use serde::Deserialize;
use serde_json::Value;
use switchyard_protocol::{
    AggLlmResponse, ContentBlock, InstructionBlock, LlmRequest, Message, ModelId, OutputParams,
    Request, Role, completion_text,
};

use super::util::prompts::drop_exact_replay;
use super::util::robustness::{safe_client_error, safe_error_summary};
use crate::core::algorithm::{Algorithm, Driver, RoutingOutcome};
use crate::{LibsyError, Result};

const ALGORITHM_NAME: &str = "system_prompt_judge";
const DEFAULT_MAX_OUTPUT_TOKENS: u64 = 64;
const TRAILING_JUDGE_INSTRUCTION: &str =
    "Choose a system-prompt action for the next assistant turn. Return only JSON.";

/// One hidden action the judge may select for system prompt injection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PromptInjectionAction {
    /// Stable action id the judge returns.
    pub id: String,
    /// System prompt inserted when this action is selected.
    pub prompt: String,
}

impl PromptInjectionAction {
    /// Creates one validated action entry.
    pub fn new(id: impl Into<String>, prompt: impl Into<String>) -> Result<Self> {
        let id = id.into();
        let prompt = prompt.into();
        validate_action_id(&id)?;
        if prompt.trim().is_empty() {
            return Err(algorithm_error(format!(
                "system prompt action {id:?} must not have an empty prompt"
            )));
        }
        Ok(Self {
            id,
            prompt: prompt.trim().to_string(),
        })
    }

    /// Parses a text action DB.
    ///
    /// The format is intentionally small:
    ///
    /// ```text
    /// [action_id]
    /// System prompt text to inject.
    ///
    /// [other_action]
    /// Another system prompt.
    /// ```
    pub fn parse_database(source: &str) -> Result<Vec<Self>> {
        let mut actions = Vec::new();
        let mut current_id: Option<String> = None;
        let mut current_prompt = String::new();
        let mut seen = BTreeSet::new();

        for line in source.lines() {
            let trimmed = line.trim();
            if let Some(id) = section_id(trimmed) {
                flush_action(
                    &mut actions,
                    &mut seen,
                    current_id.take(),
                    &mut current_prompt,
                )?;
                current_id = Some(id.to_string());
                continue;
            }
            if current_id.is_some() {
                current_prompt.push_str(line);
                current_prompt.push('\n');
            } else if !trimmed.is_empty() && !trimmed.starts_with('#') {
                return Err(algorithm_error(
                    "system prompt action DB content must appear under [action_id] headers",
                ));
            }
        }
        flush_action(&mut actions, &mut seen, current_id, &mut current_prompt)?;
        if actions.is_empty() {
            return Err(algorithm_error(
                "system prompt action DB must contain at least one [action_id] section",
            ));
        }
        Ok(actions)
    }
}

/// Runtime knobs for [`SystemPromptJudge`].
#[derive(Clone, Debug)]
pub struct SystemPromptJudgeConfig {
    /// Most completion tokens the judge verdict may use.
    pub max_output_tokens: u64,
}

impl Default for SystemPromptJudgeConfig {
    fn default() -> Self {
        Self {
            max_output_tokens: DEFAULT_MAX_OUTPUT_TOKENS,
        }
    }
}

/// Passthrough routing with a judge-selected hidden system prompt.
pub struct SystemPromptJudge {
    target: ModelId,
    judge_target: ModelId,
    actions: Vec<PromptInjectionAction>,
    config: SystemPromptJudgeConfig,
}

impl SystemPromptJudge {
    /// Creates a prompt-injection route.
    pub fn new(
        target: ModelId,
        judge_target: ModelId,
        actions: Vec<PromptInjectionAction>,
        config: SystemPromptJudgeConfig,
    ) -> Result<Self> {
        if actions.is_empty() {
            return Err(algorithm_error(
                "system_prompt_judge requires at least one action",
            ));
        }
        if config.max_output_tokens == 0 {
            return Err(algorithm_error("max_output_tokens must be at least 1"));
        }
        Ok(Self {
            target,
            judge_target,
            actions,
            config,
        })
    }

    async fn selected_action(
        &self,
        driver: &Driver,
        request: &Request,
    ) -> Option<&PromptInjectionAction> {
        let response = driver
            .call_model(self.judge_request(request), vec![self.judge_target.clone()])
            .await
            .inspect_err(|error| {
                tracing::warn!(
                    target: "libsy",
                    error = %safe_error_summary(error),
                    "system-prompt judge unavailable; routing without injection"
                );
            })
            .ok()?;
        let aggregate = response
            .llm_response
            .into_agg()
            .await
            .inspect_err(|error| {
                tracing::warn!(
                    target: "libsy",
                    error = %safe_client_error(error),
                    "system-prompt judge response failed; routing without injection"
                );
            })
            .ok()?;
        let action = parse_verdict(&aggregate)
            .inspect_err(|error| {
                tracing::warn!(
                    target: "libsy",
                    error = %safe_error_summary(error),
                    "system-prompt judge verdict invalid; routing without injection"
                );
            })
            .ok()?;
        if action.eq_ignore_ascii_case("none") {
            return None;
        }
        self.actions
            .iter()
            .find(|entry| entry.id == action)
            .or_else(|| {
                tracing::warn!(
                    target: "libsy",
                    action,
                    "system-prompt judge selected unknown action; routing without injection"
                );
                None
            })
    }

    fn judge_request(&self, request: &Request) -> Request {
        let mut messages = request.llm_request.messages.clone();
        messages.push(Message::text(
            Role::User,
            TRAILING_JUDGE_INSTRUCTION.to_string(),
        ));
        Request {
            llm_request: LlmRequest {
                model: request.llm_request.model.clone(),
                instructions: vec![InstructionBlock {
                    role: Role::System,
                    content: vec![ContentBlock::Text {
                        text: judge_prompt(&self.actions),
                    }],
                }],
                messages,
                output: OutputParams {
                    max_output_tokens: Some(self.config.max_output_tokens),
                    response_format: Some(serde_json::json!({"type": "json_object"})),
                },
                ..LlmRequest::default()
            },
            raw_request: None,
            metadata: request.metadata.clone(),
        }
    }
}

#[async_trait::async_trait]
impl Algorithm for SystemPromptJudge {
    fn name(&self) -> &str {
        ALGORITHM_NAME
    }

    async fn route(
        self: Arc<Self>,
        driver: Driver,
        mut request: Request,
    ) -> Result<RoutingOutcome> {
        if let Some(action) = self.selected_action(&driver, &request).await {
            tracing::info!(
                target: "libsy",
                action = action.id,
                selected_model = %self.target,
                "system-prompt judge injecting action"
            );
            request.llm_request.instructions.insert(
                0,
                InstructionBlock {
                    role: Role::System,
                    content: vec![ContentBlock::Text {
                        text: action.prompt.clone(),
                    }],
                },
            );
            drop_exact_replay(&mut request);
        }
        Ok(RoutingOutcome::route_to(
            self.target.clone(),
            Vec::new(),
            request,
        ))
    }
}

#[derive(Deserialize)]
struct JudgeVerdict {
    action: String,
}

fn section_id(line: &str) -> Option<&str> {
    line.strip_prefix('[')
        .and_then(|rest| rest.strip_suffix(']'))
        .map(str::trim)
        .filter(|id| !id.is_empty())
}

fn flush_action(
    actions: &mut Vec<PromptInjectionAction>,
    seen: &mut BTreeSet<String>,
    id: Option<String>,
    prompt: &mut String,
) -> Result<()> {
    let Some(id) = id else {
        return Ok(());
    };
    if !seen.insert(id.clone()) {
        return Err(algorithm_error(format!(
            "duplicate system prompt action id {id:?}"
        )));
    }
    actions.push(PromptInjectionAction::new(id, std::mem::take(prompt))?);
    Ok(())
}

fn validate_action_id(id: &str) -> Result<()> {
    if id.eq_ignore_ascii_case("none") {
        return Err(algorithm_error(
            "system prompt action id 'none' is reserved",
        ));
    }
    if id
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
    {
        return Ok(());
    }
    Err(algorithm_error(format!(
        "system prompt action id {id:?} may contain only ASCII letters, digits, '.', '_', or '-'"
    )))
}

fn judge_prompt(actions: &[PromptInjectionAction]) -> String {
    let mut prompt = String::from(
        "You are a routing judge. Read the agent conversation and decide whether one hidden \
         system-prompt action should be injected into the next assistant call.\n\
         Select an action only when it clearly helps the next turn. Otherwise select none.\n\
         Return exactly one JSON object: {\"action\":\"<id-or-none>\"}.\n\n\
         Hidden action DB:\n",
    );
    for action in actions {
        prompt.push_str("\n[");
        prompt.push_str(&action.id);
        prompt.push_str("]\n");
        prompt.push_str(&action.prompt);
        prompt.push('\n');
    }
    prompt
}

fn parse_verdict(response: &AggLlmResponse) -> Result<String> {
    let completion = completion_text(response);
    let text = strip_json_fence(completion.trim());
    let verdict: JudgeVerdict =
        serde_json::from_str(text).or_else(|_| parse_action_from_value(text))?;
    let action = verdict.action.trim().to_string();
    if action.is_empty() {
        return Err(algorithm_error("system-prompt judge returned empty action"));
    }
    Ok(action)
}

fn parse_action_from_value(text: &str) -> Result<JudgeVerdict> {
    let value: Value = serde_json::from_str(text).map_err(|error| {
        algorithm_error(format!(
            "system-prompt judge reply did not parse as JSON: {error}"
        ))
    })?;
    let action = value
        .get("action")
        .and_then(Value::as_str)
        .ok_or_else(|| algorithm_error("system-prompt judge JSON must contain string action"))?;
    Ok(JudgeVerdict {
        action: action.to_string(),
    })
}

fn strip_json_fence(text: &str) -> &str {
    let Some(rest) = text.strip_prefix("```") else {
        return text;
    };
    let rest = rest.strip_prefix("json").unwrap_or(rest);
    let rest = rest.trim_start_matches(['\n', '\r']);
    rest.strip_suffix("```").map(str::trim).unwrap_or(rest)
}

fn algorithm_error(message: impl Into<String>) -> LibsyError {
    LibsyError::AlgorithmError {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use parking_lot::Mutex;
    use switchyard_protocol::{LlmResponse, Response, text_request, text_response};

    use crate::core::testing::{reply, test_drive};

    #[test]
    fn parses_section_based_action_database() -> Result<()> {
        let actions = PromptInjectionAction::parse_database(
            r#"
# comments before the first section are ignored
[compile_failure]
Focus on the exact compiler error.

[stuck-loop]
Stop repeating commands and make a new plan.
"#,
        )?;
        assert_eq!(
            actions,
            vec![
                PromptInjectionAction::new(
                    "compile_failure",
                    "Focus on the exact compiler error.",
                )?,
                PromptInjectionAction::new(
                    "stuck-loop",
                    "Stop repeating commands and make a new plan.",
                )?,
            ]
        );
        Ok(())
    }

    #[tokio::test]
    async fn injects_the_prompt_selected_by_the_judge() -> Result<()> {
        let actions = vec![PromptInjectionAction::new(
            "compile_failure",
            "Focus on compiler diagnostics before editing.",
        )?];
        let algorithm: Arc<dyn Algorithm> = Arc::new(SystemPromptJudge::new(
            "target".into(),
            "judge".into(),
            actions,
            SystemPromptJudgeConfig::default(),
        )?);
        let calls = Arc::new(Mutex::new(Vec::<(String, Request)>::new()));
        let recorder = Arc::clone(&calls);
        let mut request = Request {
            llm_request: text_request(Some("route".to_string()), "nvcc failed"),
            raw_request: None,
            metadata: None,
        };
        request
            .llm_request
            .preservation
            .requests
            .insert("openai_chat".into(), serde_json::json!({"model":"route"}));

        let (selected, _) = test_drive(
            algorithm,
            request,
            move |model: ModelId, request: Request| {
                let recorder = Arc::clone(&recorder);
                async move {
                    recorder.lock().push((model.to_string(), request.clone()));
                    if model == "judge" {
                        Ok(Response {
                            llm_response: LlmResponse::Agg(text_response(
                                None,
                                r#"{"action":"compile_failure"}"#,
                            )),
                            metadata: None,
                            upstream_headers: Default::default(),
                        })
                    } else {
                        Ok(reply("ok"))
                    }
                }
            },
        )
        .await?;

        assert_eq!(selected, "target");
        let calls = calls.lock();
        assert_eq!(
            calls
                .iter()
                .map(|(model, _)| model.as_str())
                .collect::<Vec<_>>(),
            ["judge", "target"]
        );
        let judge_request = &calls[0].1;
        let judge_prompt = judge_request
            .llm_request
            .instructions
            .first()
            .and_then(|instruction| instruction.content.first())
            .and_then(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .unwrap_or_default();
        assert!(judge_prompt.contains("[compile_failure]"));
        assert!(judge_prompt.contains("Focus on compiler diagnostics before editing."));
        assert_eq!(
            judge_request
                .llm_request
                .messages
                .first()
                .and_then(|message| message.text_content("|")),
            Some("nvcc failed".to_string())
        );
        let target_request = &calls[1].1;
        let injected = target_request
            .llm_request
            .instructions
            .first()
            .and_then(|instruction| instruction.content.first())
            .and_then(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            });
        assert_eq!(
            injected,
            Some("Focus on compiler diagnostics before editing.")
        );
        assert!(
            target_request.llm_request.preservation.requests.is_empty(),
            "mutated requests must not exact-replay the inbound body"
        );
        Ok(())
    }

    #[tokio::test]
    async fn none_verdict_leaves_request_untouched() -> Result<()> {
        let algorithm: Arc<dyn Algorithm> = Arc::new(SystemPromptJudge::new(
            "target".into(),
            "judge".into(),
            vec![PromptInjectionAction::new(
                "compile_failure",
                "diagnose first",
            )?],
            SystemPromptJudgeConfig::default(),
        )?);
        let calls = Arc::new(Mutex::new(Vec::<(String, Request)>::new()));
        let recorder = Arc::clone(&calls);

        test_drive(
            algorithm,
            Request {
                llm_request: text_request(Some("route".to_string()), "hello"),
                raw_request: None,
                metadata: None,
            },
            move |model: ModelId, request: Request| {
                let recorder = Arc::clone(&recorder);
                async move {
                    recorder.lock().push((model.to_string(), request));
                    if model == "judge" {
                        Ok(reply(r#"{"action":"none"}"#))
                    } else {
                        Ok(reply("ok"))
                    }
                }
            },
        )
        .await?;

        let calls = calls.lock();
        assert_eq!(calls[1].0, "target");
        assert!(calls[1].1.llm_request.instructions.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn invalid_judge_reply_fails_open() -> Result<()> {
        let algorithm: Arc<dyn Algorithm> = Arc::new(SystemPromptJudge::new(
            "target".into(),
            "judge".into(),
            vec![PromptInjectionAction::new(
                "compile_failure",
                "diagnose first",
            )?],
            SystemPromptJudgeConfig::default(),
        )?);

        let (selected, response) = test_drive(
            algorithm,
            Request {
                llm_request: text_request(Some("route".to_string()), "hello"),
                raw_request: None,
                metadata: None,
            },
            |model: ModelId, _request: Request| async move {
                if model == "judge" {
                    Ok(reply("not json"))
                } else {
                    Ok(reply("ok"))
                }
            },
        )
        .await?;

        assert_eq!(selected, "target");
        assert_eq!(
            completion_text(response.llm_response.as_agg().unwrap()),
            "ok"
        );
        Ok(())
    }
}
