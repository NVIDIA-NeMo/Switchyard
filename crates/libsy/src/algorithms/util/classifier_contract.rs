// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prompt and structured-output contracts shared by LLM classifiers.

use jsonschema::Validator;
use serde::Deserialize;
use serde_json::{Value, json};

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
    validator: Option<Validator>,
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
        let response_format: Value =
            serde_json::from_str(response_format_json).map_err(|error| {
                LibsyError::AlgorithmError {
                    message: format!("response schema is invalid: {error}"),
                }
            })?;
        Self::from_response_format(prompt_template, response_format, None)
    }

    /// Builds a provider response format around a user-supplied inner JSON Schema.
    pub(crate) fn from_inner_schema(prompt_template: &str, schema: Value) -> Result<Self> {
        if schema.get("json_schema").is_some() {
            return Err(LibsyError::AlgorithmError {
                message:
                    "response_schema must be the inner JSON Schema, not a response_format wrapper"
                        .to_string(),
            });
        }
        let validator = compile_schema(&schema)?;
        Self::from_response_format(
            prompt_template,
            json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "switchyard_classifier_response",
                    "strict": true,
                    "schema": schema,
                }
            }),
            Some(validator),
        )
    }

    fn from_response_format(
        prompt_template: &str,
        response_format: Value,
        validator: Option<Validator>,
    ) -> Result<Self> {
        if prompt_template.trim().is_empty() {
            return Err(LibsyError::AlgorithmError {
                message: "classifier prompt must not be empty".to_string(),
            });
        }
        let response_schema = response_format
            .pointer("/json_schema/schema")
            .ok_or_else(|| LibsyError::AlgorithmError {
                message: "response schema has no json_schema.schema".to_string(),
            })?
            .clone();
        let prompt_schema = serde_json::to_string_pretty(&response_schema).map_err(|error| {
            LibsyError::AlgorithmError {
                message: format!("prompt schema could not be rendered: {error}"),
            }
        })?;

        Ok(Self {
            system_prompt: prompt_template.replace("{{RESPONSE_SCHEMA}}", &prompt_schema),
            response_format,
            validator,
        })
    }

    pub(crate) fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    pub(crate) fn response_format(&self) -> &Value {
        &self.response_format
    }

    /// Validates a dynamic verdict when this contract carries a runtime schema validator.
    pub(crate) fn validate_verdict(&self, verdict: &Value) -> Result<()> {
        let Some(validator) = &self.validator else {
            return Ok(());
        };
        validator
            .validate(verdict)
            .map_err(|error| LibsyError::AlgorithmError {
                message: format!("classifier verdict did not match response_schema: {error}"),
            })
    }
}

fn compile_schema(schema: &Value) -> Result<Validator> {
    if !schema.is_object() {
        return Err(algorithm_error("response_schema must be a JSON object"));
    }
    jsonschema::meta::validate(schema).map_err(|error| {
        algorithm_error(format!(
            "response_schema is not a valid JSON Schema: {error}"
        ))
    })?;
    jsonschema::validator_for(schema)
        .map_err(|error| algorithm_error(format!("response_schema could not be compiled: {error}")))
}

fn algorithm_error(message: impl Into<String>) -> LibsyError {
    LibsyError::AlgorithmError {
        message: message.into(),
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

    #[test]
    fn a_custom_contract_wraps_and_validates_its_inner_schema() -> Result<()> {
        let contract = ClassifierContract::from_inner_schema(
            "Choose a target:\n{{RESPONSE_SCHEMA}}",
            json!({
                "type": "object",
                "$defs": {
                    "decision": {
                        "type": "object",
                        "properties": {
                            "target": {"type": "string", "enum": ["sonnet", "opus"]}
                        },
                        "required": ["target"],
                        "additionalProperties": false
                    }
                },
                "properties": {
                    "decision": {"$ref": "#/$defs/decision"}
                },
                "required": ["decision"],
                "additionalProperties": false
            }),
        )?;

        assert_eq!(
            contract
                .response_format()
                .pointer("/json_schema/name")
                .and_then(Value::as_str),
            Some("switchyard_classifier_response")
        );
        assert_eq!(
            contract
                .response_format()
                .pointer("/json_schema/strict")
                .and_then(Value::as_bool),
            Some(true)
        );
        assert!(contract.system_prompt().contains("\"target\""));
        contract.validate_verdict(&json!({"decision": {"target": "sonnet"}}))?;
        assert!(
            contract
                .validate_verdict(&json!({"decision": {"target": "unknown"}}))
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn a_provider_wrapper_is_rejected_as_an_inner_schema() {
        let error = ClassifierContract::from_inner_schema(
            "classify",
            json!({"json_schema": {"schema": {"type": "object"}}}),
        )
        .expect_err("provider wrapper should be rejected");

        assert!(error.to_string().contains("inner JSON Schema"));
    }
}
