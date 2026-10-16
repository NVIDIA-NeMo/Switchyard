// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prompt and structured-output contracts shared by LLM classifiers.

use jsonschema::Validator;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{LibsyError, Result};

/// Provider-side structured-output mode used by a classifier judge.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ClassifierResponseFormat {
    /// Send the verdict schema through the provider's strict JSON Schema wrapper.
    #[default]
    JsonSchema,
    /// Request a JSON object and enforce the verdict schema locally.
    JsonObject,
}

/// User-configurable parts of a classifier's prompt and verdict contract.
///
/// Fields are private so new contract settings can be added without breaking Rust struct literals.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct ClassifierContractConfig {
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    response_format_type: ClassifierResponseFormat,
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

    /// Selects the provider-side structured-output mode.
    pub fn with_response_format_type(
        mut self,
        response_format_type: ClassifierResponseFormat,
    ) -> Self {
        self.response_format_type = response_format_type;
        self
    }

    /// Returns the configured provider-side structured-output mode.
    pub fn response_format_type(&self) -> ClassifierResponseFormat {
        self.response_format_type
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
    /// The packaged response format must contain `json_schema.schema`. JSON Schema mode retains
    /// that wrapper for the model request; JSON Object mode moves the schema into the prompt and
    /// compiles it for local validation.
    pub(crate) fn from_config(
        config: &ClassifierContractConfig,
        default_prompt: &str,
        response_format_json: &str,
    ) -> Result<Self> {
        let prompt_template = config.prompt().unwrap_or(default_prompt);
        validate_prompt(prompt_template)?;
        let response_format: Value =
            serde_json::from_str(response_format_json).map_err(|error| {
                LibsyError::AlgorithmError {
                    message: format!("response schema is invalid: {error}"),
                }
            })?;
        let schema = response_format
            .pointer("/json_schema/schema")
            .ok_or_else(|| LibsyError::AlgorithmError {
                message: "response schema has no json_schema.schema".to_string(),
            })?;
        match config.response_format_type() {
            ClassifierResponseFormat::JsonSchema => Ok(Self::from_response_format(
                prompt_template.to_string(),
                response_format,
                None,
            )),
            ClassifierResponseFormat::JsonObject => {
                let validator = compile_schema(schema)?;
                Ok(Self::from_response_format(
                    schema_in_prompt(prompt_template, schema)?,
                    json!({"type": "json_object"}),
                    Some(validator),
                ))
            }
        }
    }

    /// Builds a contract around a user-supplied inner JSON Schema.
    ///
    /// The verdict is validated against the schema locally in both modes. JSON Schema mode sends it through the
    /// provider's strict wrapper; JSON Object mode appends it to the prompt, because the
    /// provider then guarantees only that the reply is JSON.
    pub(crate) fn from_inner_schema(
        prompt_template: &str,
        schema: Value,
        response_format_type: ClassifierResponseFormat,
    ) -> Result<Self> {
        validate_prompt(prompt_template)?;
        if schema.get("json_schema").is_some() {
            return Err(LibsyError::AlgorithmError {
                message:
                    "response_schema must be the inner JSON Schema, not a response_format wrapper"
                        .to_string(),
            });
        }
        let validator = compile_schema(&schema)?;
        match response_format_type {
            ClassifierResponseFormat::JsonSchema => Ok(Self::from_response_format(
                prompt_template.to_string(),
                json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": "switchyard_classifier_response",
                        "strict": true,
                        "schema": schema,
                    }
                }),
                Some(validator),
            )),
            ClassifierResponseFormat::JsonObject => {
                // Catches the common case of a root `type` that excludes objects: the provider
                // only returns objects, so every verdict would fail and route to the default.
                // A schema with no `type` (e.g. `{"enum": [1, 2]}`) is not checked and can
                // still never match; that is left to the author.
                let is_object_allowed = match schema.get("type") {
                    Some(Value::String(root)) => root == "object",
                    Some(Value::Array(roots)) => roots.iter().any(|root| root == "object"),
                    _ => true,
                };
                if !is_object_allowed {
                    return Err(algorithm_error(
                        "response_schema must describe a JSON object in json_object mode",
                    ));
                }
                Ok(Self::from_response_format(
                    schema_in_prompt(prompt_template, &schema)?,
                    json!({"type": "json_object"}),
                    Some(validator),
                ))
            }
        }
    }

    fn from_response_format(
        system_prompt: String,
        response_format: Value,
        validator: Option<Validator>,
    ) -> Self {
        Self {
            system_prompt,
            response_format,
            validator,
        }
    }

    pub(crate) fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    pub(crate) fn response_format(&self) -> &Value {
        &self.response_format
    }

    /// Whether the provider response must be checked against the compiled schema locally.
    pub(crate) fn validates_locally(&self) -> bool {
        self.validator.is_some()
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

pub(super) fn validate_prompt(prompt_template: &str) -> Result<()> {
    if prompt_template.trim().is_empty() {
        return Err(algorithm_error("classifier prompt must not be empty"));
    }
    if prompt_template.contains("{{RESPONSE_SCHEMA}}") {
        return Err(algorithm_error(
            "classifier prompt must not include {{RESPONSE_SCHEMA}}; remove the placeholder because Switchyard supplies the schema automatically",
        ));
    }
    Ok(())
}

/// Appends the verdict schema to a prompt for JSON Object mode.
///
/// Callers validate the template, not this result: the appended schema makes the prompt
/// non-empty and may itself contain `{{RESPONSE_SCHEMA}}` text.
fn schema_in_prompt(prompt_template: &str, schema: &Value) -> Result<String> {
    let rendered_schema = serde_json::to_string_pretty(schema).map_err(|error| {
        algorithm_error(format!("response schema could not be rendered: {error}"))
    })?;
    Ok(format!(
        "{prompt_template}\n\nReturn exactly one JSON object matching this JSON Schema:\n{rendered_schema}"
    ))
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
    fn a_runtime_contract_keeps_its_schema_out_of_the_prompt() -> Result<()> {
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
        let config = ClassifierContractConfig::default().with_prompt(
            "Return a risk verdict matching the response schema supplied with the request.",
        );
        let contract = ClassifierContract::from_config(&config, "packaged prompt", schema)?;

        assert_eq!(
            contract.system_prompt(),
            "Return a risk verdict matching the response schema supplied with the request."
        );
        assert!(!contract.system_prompt().contains("\"risk\""));
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
    fn a_contract_rejects_a_response_schema_placeholder() {
        let config = ClassifierContractConfig::default()
            .with_prompt("Return JSON matching {{RESPONSE_SCHEMA}}");
        let error = ClassifierContract::from_config(
            &config,
            "packaged prompt",
            r#"{"json_schema":{"schema":{"type":"object"}}}"#,
        )
        .expect_err("schema placeholders should be rejected");

        assert!(
            error
                .to_string()
                .contains("Switchyard supplies the schema automatically")
        );
    }

    #[test]
    fn a_custom_contract_wraps_and_validates_its_inner_schema() -> Result<()> {
        let contract = ClassifierContract::from_inner_schema(
            "Choose a target matching the response schema supplied with the request.",
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
            ClassifierResponseFormat::JsonSchema,
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
        assert!(!contract.system_prompt().contains("\"target\""));
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
        for response_format_type in [
            ClassifierResponseFormat::JsonSchema,
            ClassifierResponseFormat::JsonObject,
        ] {
            let error = ClassifierContract::from_inner_schema(
                "classify",
                json!({"json_schema": {"schema": {"type": "object"}}}),
                response_format_type,
            )
            .expect_err("a wrapped schema should error");

            assert!(
                error.to_string().contains("inner JSON Schema"),
                "{response_format_type:?}: {error}"
            );
        }
    }

    #[test]
    fn a_custom_contract_can_request_a_json_object() -> Result<()> {
        let contract = ClassifierContract::from_inner_schema(
            "Choose a target.",
            json!({
                "type": "object",
                "properties": {"target": {"type": "string", "enum": ["sonnet", "opus"]}},
                "required": ["target"],
                "additionalProperties": false
            }),
            ClassifierResponseFormat::JsonObject,
        )?;

        assert_eq!(contract.response_format(), &json!({"type": "json_object"}));
        assert!(contract.system_prompt().starts_with("Choose a target."));
        assert!(contract.system_prompt().contains("JSON Schema"));
        assert!(contract.system_prompt().contains("\"target\""));
        // The provider no longer enforces the schema, so the local validator must.
        contract.validate_verdict(&json!({"target": "sonnet"}))?;
        assert!(
            contract
                .validate_verdict(&json!({"target": "unknown"}))
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn json_object_mode_accepts_only_schemas_whose_root_allows_an_object() {
        for (schema, accepted) in [
            (json!({"type": "object"}), true),
            (json!({"type": ["object"]}), true),
            (json!({"type": ["object", "null"]}), true),
            (json!({"type": "string"}), false),
            (json!({"type": "array"}), false),
            (json!({"type": ["string", "null"]}), false),
        ] {
            let result = ClassifierContract::from_inner_schema(
                "classify",
                schema.clone(),
                ClassifierResponseFormat::JsonObject,
            );

            match result {
                Ok(_) => assert!(accepted, "{schema} should be rejected"),
                Err(error) => {
                    assert!(!accepted, "{schema}: {error}");
                    assert!(
                        error.to_string().contains("must describe a JSON object"),
                        "{schema}: {error}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_schema_that_mentions_the_placeholder_is_accepted_in_both_modes() {
        let schema = json!({
            "type": "object",
            "properties": {"target": {"type": "string"}},
            "required": ["target"],
            "description": "{{RESPONSE_SCHEMA}}"
        });
        for response_format_type in [
            ClassifierResponseFormat::JsonSchema,
            ClassifierResponseFormat::JsonObject,
        ] {
            ClassifierContract::from_inner_schema("pick", schema.clone(), response_format_type)
                .unwrap_or_else(|error| panic!("{response_format_type:?}: {error}"));
        }
    }

    #[test]
    fn an_empty_prompt_is_rejected_in_json_object_mode() {
        let error = ClassifierContract::from_inner_schema(
            " ",
            json!({"type": "object"}),
            ClassifierResponseFormat::JsonObject,
        )
        .expect_err("an empty prompt should be rejected before the schema is appended");

        assert!(
            error
                .to_string()
                .contains("classifier prompt must not be empty")
        );
    }
}
