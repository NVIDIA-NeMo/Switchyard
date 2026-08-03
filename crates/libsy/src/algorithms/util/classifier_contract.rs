// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prompt and structured-output contracts shared by LLM classifiers.

use serde::Deserialize;
use serde_json::Value;

use crate::{LibsyError, Result};

/// User-configurable parts of a classifier's prompt and verdict contract.
///
/// Fields are private so new contract settings can be added without breaking Rust struct literals.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct ClassifierContractConfig {
    #[serde(default)]
    prompt: Option<String>,
}

impl ClassifierContractConfig {
    /// Overrides the packaged classifier prompt.
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    /// Returns the configured prompt override.
    pub fn prompt(&self) -> Option<&str> {
        self.prompt.as_deref()
    }
}

/// Rendered prompt and response format for one classifier.
#[derive(Debug)]
pub(crate) struct ClassifierContract {
    system_prompt: String,
    response_format: Value,
}

impl ClassifierContract {
    /// Builds a contract from user settings and packaged defaults.
    ///
    /// The response format must contain `json_schema.schema`. Its inner schema replaces every
    /// `{{RESPONSE_SCHEMA}}` placeholder in the prompt, while the complete response format is
    /// retained for the model request.
    pub(crate) fn from_config(
        config: &ClassifierContractConfig,
        default_prompt: &str,
        response_format_json: &str,
    ) -> Result<Self> {
        let prompt_template = config.prompt().unwrap_or(default_prompt);
        if prompt_template.trim().is_empty() {
            return Err(LibsyError::AlgorithmError {
                message: "classifier prompt must not be empty".to_string(),
            });
        }
        let response_format: Value =
            serde_json::from_str(response_format_json).map_err(|error| {
                LibsyError::AlgorithmError {
                    message: format!("response schema is invalid: {error}"),
                }
            })?;
        let prompt_schema = response_format
            .pointer("/json_schema/schema")
            .ok_or_else(|| LibsyError::AlgorithmError {
                message: "response schema has no json_schema.schema".to_string(),
            })?;
        let prompt_schema = serde_json::to_string_pretty(prompt_schema).map_err(|error| {
            LibsyError::AlgorithmError {
                message: format!("prompt schema could not be rendered: {error}"),
            }
        })?;

        Ok(Self {
            system_prompt: prompt_template.replace("{{RESPONSE_SCHEMA}}", &prompt_schema),
            response_format,
        })
    }

    pub(crate) fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    pub(crate) fn response_format(&self) -> &Value {
        &self.response_format
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_runtime_contract_renders_its_own_schema() -> Result<()> {
        let schema = r#"{
            "type": "json_schema",
            "json_schema": {
                "name": "RiskDecision",
                "schema": {
                    "type": "object",
                    "properties": {"risk": {"type": "number"}}
                }
            }
        }"#;
        let config = ClassifierContractConfig::default()
            .with_prompt("Return a risk verdict matching:\n{{RESPONSE_SCHEMA}}");
        let contract = ClassifierContract::from_config(&config, "packaged prompt", schema)?;

        assert!(contract.system_prompt().contains("\"risk\""));
        assert!(!contract.system_prompt().contains("{{RESPONSE_SCHEMA}}"));
        assert_eq!(
            contract
                .response_format()
                .pointer("/json_schema/name")
                .and_then(Value::as_str),
            Some("RiskDecision")
        );
        Ok(())
    }

    #[test]
    fn a_contract_requires_an_inner_json_schema() {
        let error = ClassifierContract::from_config(
            &ClassifierContractConfig::default(),
            "classify",
            r#"{"type":"json"}"#,
        )
        .expect_err("missing inner schema should be rejected");

        assert!(error.to_string().contains("json_schema.schema"));
    }

    #[test]
    fn a_contract_rejects_an_empty_prompt() {
        let config = ClassifierContractConfig::default().with_prompt("  \n");
        let error = ClassifierContract::from_config(
            &config,
            "packaged prompt",
            r#"{"json_schema":{"schema":{"type":"object"}}}"#,
        )
        .expect_err("empty prompt should be rejected");

        assert!(error.to_string().contains("prompt must not be empty"));
    }
}
