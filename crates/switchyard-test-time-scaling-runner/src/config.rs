// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runner configuration.

use std::path::PathBuf;

use serde::Deserialize;
use switchyard_test_time_scaling::{ExperimentManifest, ScalingConfig, Task};

/// One command invoked without a shell.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CommandConfig {
    /// Executable followed by its arguments.
    pub argv: Vec<String>,
}

/// NVIDIA-compatible chat completion settings.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// OpenAI-compatible API base URL.
    pub base_url: String,
    /// Environment variable that contains the API key.
    pub api_key_env: String,
    /// Simultaneous summary and comparison calls.
    pub max_concurrency: usize,
    /// Maximum output tokens for a summary.
    pub summary_max_tokens: usize,
    /// Maximum output tokens for a comparison.
    pub comparison_max_tokens: usize,
    /// Maximum serialized rollout characters shown to the summarizer.
    pub max_summary_input_chars: usize,
    /// Content attempts made when a summary is not a JSON object.
    pub summary_content_attempts: usize,
    /// HTTP attempts for retryable model errors.
    pub http_attempts: usize,
    /// Request timeout in seconds.
    pub request_timeout_seconds: u64,
    /// Sampling temperature used for summary calls.
    pub summary_temperature: f64,
    /// Sampling temperature used for comparison calls.
    pub comparison_temperature: f64,
    /// Whether to send the recorded logical seed to the provider.
    pub send_seed: bool,
}

/// Complete input for one task run.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    /// Task to run.
    pub task: Task,
    /// PDR and RTV settings.
    pub scaling: ScalingConfig,
    /// Replication choices and source labels.
    pub manifest: ExperimentManifest,
    /// Directory for the method record and post-run evaluation.
    pub output_dir: PathBuf,
    /// Harbor batch adapter command. It reads a request from stdin and writes rollouts to stdout.
    pub rollout_command: CommandConfig,
    /// Optional post-selection grader command. It writes rollout outcomes to stdout.
    pub evaluation_command: Option<CommandConfig>,
    /// Model API settings.
    pub model: ModelConfig,
}

impl RunConfig {
    /// Rejects missing or unsafe runner settings.
    pub fn validate(&self) -> Result<(), String> {
        validate_command(&self.rollout_command, "rollout_command")?;
        if let Some(command) = &self.evaluation_command {
            validate_command(command, "evaluation_command")?;
        }
        if self.output_dir.as_os_str().is_empty() {
            return Err("output_dir must not be empty".to_string());
        }
        if self.model.base_url.trim().is_empty() || self.model.api_key_env.trim().is_empty() {
            return Err("model base_url and api_key_env must not be empty".to_string());
        }
        if self.model.max_concurrency == 0
            || self.model.summary_max_tokens == 0
            || self.model.comparison_max_tokens == 0
            || self.model.max_summary_input_chars < 2
            || self.model.summary_content_attempts == 0
            || self.model.http_attempts == 0
            || self.model.request_timeout_seconds == 0
        {
            return Err(
                "model counts, token limits, input limit, attempts, and timeout must be positive"
                    .to_string(),
            );
        }
        if !self.model.summary_temperature.is_finite()
            || self.model.summary_temperature < 0.0
            || !self.model.comparison_temperature.is_finite()
            || self.model.comparison_temperature < 0.0
        {
            return Err("model temperatures must be finite and non-negative".to_string());
        }
        Ok(())
    }
}

fn validate_command(command: &CommandConfig, name: &str) -> Result<(), String> {
    if command.argv.is_empty() || command.argv.iter().any(|part| part.trim().is_empty()) {
        return Err(format!("{name} must contain non-empty arguments"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{CommandConfig, ModelConfig, RunConfig};
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use switchyard_test_time_scaling::{
        ExperimentManifest, MANIFEST_SCHEMA_VERSION, ReplicationMode, ScalingConfig, Task,
    };

    fn config() -> RunConfig {
        RunConfig {
            task: Task {
                id: "task".to_string(),
                benchmark: "benchmark".to_string(),
                prompt: "prompt".to_string(),
            },
            scaling: ScalingConfig::default(),
            manifest: ExperimentManifest {
                schema_version: MANIFEST_SCHEMA_VERSION,
                replication_mode: ReplicationMode::Conceptual,
                code_revision: "revision".to_string(),
                model_id: "model".to_string(),
                fields: BTreeMap::new(),
            },
            output_dir: PathBuf::from("output"),
            rollout_command: CommandConfig {
                argv: vec!["runner".to_string()],
            },
            evaluation_command: None,
            model: ModelConfig {
                base_url: "https://example.test/v1".to_string(),
                api_key_env: "TEST_KEY".to_string(),
                max_concurrency: 1,
                summary_max_tokens: 1,
                comparison_max_tokens: 1,
                max_summary_input_chars: 2,
                summary_content_attempts: 1,
                http_attempts: 1,
                request_timeout_seconds: 1,
                summary_temperature: 0.0,
                comparison_temperature: 0.0,
                send_seed: false,
            },
        }
    }

    #[test]
    fn rejects_empty_command_and_zero_limits() {
        let mut value = config();
        assert!(value.validate().is_ok());
        value.rollout_command.argv.clear();
        assert!(value.validate().is_err());
        value = config();
        value.model.max_concurrency = 0;
        assert!(value.validate().is_err());
    }
}
