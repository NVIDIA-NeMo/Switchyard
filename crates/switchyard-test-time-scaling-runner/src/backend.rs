// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Live Harbor and model backend.

use async_trait::async_trait;
use serde::Serialize;
use serde_json::{Map, Value};
use std::path::PathBuf;
use switchyard_test_time_scaling::{
    ComparisonRequest, ComparisonResponse, Result, Rollout, RolloutRequest, ScalingBackend,
    ScalingError, Summary, Task,
};

use crate::command::call_json;
use crate::config::{CommandConfig, ModelConfig};
use crate::model_client::ModelClient;

const SUMMARY_PROMPT: &str = "You are given an agentic coding task and one recorded attempt. Produce one structured summary as a JSON object. Report evidence, not just the agent's claims. Separate commands with captured responses from proposed commands, successful outputs from errors, fixed errors from unresolved errors, files proved to exist from files only claimed, tests run after the final edit from earlier tests, and verified requirements from unchecked requirements. Preserve exact paths, functions, commands, return codes, decisive output, code changes, unresolved issues, and uncertainty. Do not infer hidden-test success. Do not include any official grader result. Output JSON only.";

#[derive(Serialize)]
struct RolloutBatch<'a> {
    task: &'a Task,
    requests: Vec<RolloutRequest>,
}

/// Adapter used by the core scaling controller.
pub struct LiveBackend {
    model_id: String,
    rollout_command: CommandConfig,
    model_config: ModelConfig,
    model: ModelClient,
}

impl LiveBackend {
    /// Creates a backend after loading the configured API key.
    pub fn new(
        model_id: String,
        rollout_command: CommandConfig,
        model_config: ModelConfig,
        call_log_path: PathBuf,
    ) -> std::result::Result<Self, String> {
        let model = ModelClient::new(model_id.clone(), &model_config, call_log_path)?;
        Ok(Self {
            model_id,
            rollout_command,
            model_config,
            model,
        })
    }
}

#[async_trait]
impl ScalingBackend for LiveBackend {
    type Output = Value;

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn run_rollouts(
        &self,
        task: &Task,
        requests: Vec<RolloutRequest>,
    ) -> Result<Vec<Rollout<Self::Output>>> {
        call_json(&self.rollout_command, &RolloutBatch { task, requests })
            .await
            .map_err(ScalingError::Backend)
    }

    async fn summarize(&self, task: &Task, rollout: &Rollout<Self::Output>) -> Result<Summary> {
        let serialized = serde_json::to_string(&rollout.output)
            .map_err(|error| ScalingError::Backend(error.to_string()))?;
        let input = truncate_middle(&serialized, self.model_config.max_summary_input_chars);
        let mut prompt = format!(
            "{SUMMARY_PROMPT}\n\nOriginal task:\n{}\n\nRecorded attempt:\n{}",
            task.prompt, input
        );

        for attempt in 1..=self.model_config.summary_content_attempts {
            let reply = self
                .model
                .complete(
                    "summary",
                    &prompt,
                    self.model_config.summary_max_tokens,
                    self.model_config.summary_temperature,
                    rollout.rollout_index as u64
                        + u64::from(rollout.iteration) * 1_000
                        + attempt as u64,
                )
                .await
                .map_err(ScalingError::Backend)?;
            if let Some(value) = parse_json_object(&reply.content) {
                return Ok(Summary {
                    id: format!("summary-{}", rollout.id),
                    rollout_id: rollout.id.clone(),
                    model_id: self.model_id.clone(),
                    value,
                    raw_response: reply.raw_response,
                    generation_attempts: attempt,
                });
            }
            prompt.push_str(&format!(
                "\n\nThe previous response was not one JSON object. Return a corrected JSON object only. Previous response:\n{}",
                truncate_middle(&reply.content, 4_000)
            ));
        }
        Err(ScalingError::Backend(format!(
            "summary for {} was not a JSON object after {} attempts",
            rollout.id, self.model_config.summary_content_attempts
        )))
    }

    async fn compare(
        &self,
        _task: &Task,
        request: ComparisonRequest,
    ) -> Result<ComparisonResponse> {
        let reply = self
            .model
            .complete(
                "comparison",
                &request.prompt,
                self.model_config.comparison_max_tokens,
                self.model_config.comparison_temperature,
                request.seed,
            )
            .await
            .map_err(ScalingError::Backend)?;
        Ok(ComparisonResponse {
            model_id: self.model_id.clone(),
            content: reply.content,
        })
    }
}

fn parse_json_object(content: &str) -> Option<Map<String, Value>> {
    let trimmed = content.trim();
    if let Ok(value) = serde_json::from_str::<Map<String, Value>>(trimmed) {
        return Some(value);
    }
    let fenced = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```JSON"))
        .or_else(|| trimmed.strip_prefix("```"))?
        .strip_suffix("```")?
        .trim();
    serde_json::from_str(fenced).ok()
}

fn truncate_middle(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let left_count = limit / 2;
    let right_count = limit - left_count;
    let left: String = text.chars().take(left_count).collect();
    let mut right: Vec<char> = text.chars().rev().take(right_count).collect();
    right.reverse();
    format!(
        "{left}\n[... middle removed by recorded summary input limit ...]\n{}",
        right.into_iter().collect::<String>()
    )
}

#[cfg(test)]
mod tests {
    use super::{parse_json_object, truncate_middle};

    #[test]
    fn truncation_keeps_both_ends() {
        assert_eq!(truncate_middle("short", 5), "short");
        let value = truncate_middle("abcdefghij", 6);
        assert!(value.starts_with("abc"));
        assert!(value.ends_with("hij"));
    }

    #[test]
    fn summary_parser_accepts_an_object_with_one_optional_fence() {
        assert_eq!(
            parse_json_object(r#"{"result":"ok"}"#).unwrap()["result"],
            "ok"
        );
        assert_eq!(
            parse_json_object("```json\n{\"result\":\"ok\"}\n```").unwrap()["result"],
            "ok"
        );
        assert!(parse_json_object("before {\"result\":\"ok\"}").is_none());
    }
}
