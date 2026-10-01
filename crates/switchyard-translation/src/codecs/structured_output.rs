// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Structured-output enforcement across provider formats.

use serde_json::Value;

use crate::diagnostic::TranslationDiagnostic;
use crate::error::Result;
use crate::llm::OutputParams;
use crate::policy::TranslationPolicy;
use crate::util::push_lossy;

pub(super) fn decode_openai_schema_enforcement(format: Option<&Value>) -> Option<bool> {
    let format = format.filter(|format| format["type"] == "json_schema")?;
    Some(format["json_schema"]["strict"].as_bool().unwrap_or(false))
}

pub(super) fn encode_openai_format(
    output: &OutputParams,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Option<Value>> {
    let mut format = output.response_format.clone();
    let Some(is_enforced) = output.is_schema_enforced else {
        return Ok(format);
    };
    let schema = format
        .as_mut()
        .filter(|format| format["type"] == "json_schema")
        .and_then(|format| format.get_mut("json_schema"))
        .and_then(Value::as_object_mut);
    let Some(schema) = schema.filter(|schema| schema.get("schema").is_some_and(Value::is_object))
    else {
        if is_enforced {
            push_lossy(
                diagnostics,
                policy,
                "Structured-output schema enforcement requires a json_schema object with a schema",
            )?;
        }
        return Ok(format);
    };
    // Preserve native strict schemas, including constructs outside our checked subset.
    if is_enforced
        && schema.get("strict").and_then(Value::as_bool) != Some(true)
        && !schema
            .get("schema")
            .is_some_and(is_openai_strict_compatible)
    {
        push_lossy(
            diagnostics,
            policy,
            "Structured-output schema was left non-strict: compatibility with OpenAI strict mode could not be established",
        )?;
    } else {
        schema.insert("strict".into(), Value::Bool(is_enforced));
    }
    Ok(format)
}

// A conservative subset shared by OpenAI base and fine-tuned models. Fine-tuning
// excludes format, pattern, minLength/maxLength, minimum/maximum, multipleOf,
// and minItems/maxItems. These and unchecked constructs such as references retain
// their schema with a lossiness diagnostic.
fn is_openai_strict_compatible(schema: &Value) -> bool {
    #[derive(Default)]
    struct Size {
        properties: usize,
        enum_values: usize,
        chars: usize,
    }

    fn visit(schema: &Value, depth: usize, size: &mut Size) -> bool {
        let Some(object) = schema.as_object() else {
            return false;
        };
        if object.keys().any(|key| {
            !matches!(
                key.as_str(),
                "type"
                    | "title"
                    | "description"
                    | "properties"
                    | "required"
                    | "additionalProperties"
                    | "items"
                    | "enum"
                    | "const"
                    | "anyOf"
            )
        }) {
            return false;
        }
        if let Some(values) = object.get("enum") {
            let Some(values) = values.as_array().filter(|values| !values.is_empty()) else {
                return false;
            };
            if values
                .iter()
                .any(|value| value.is_object() || value.is_array())
            {
                return false;
            }
            let chars = values
                .iter()
                .filter_map(Value::as_str)
                .map(|value| value.chars().count())
                .sum::<usize>();
            if values.len() > 250 && chars > 15_000 {
                return false;
            }
            size.enum_values += values.len();
            size.chars += chars;
        }
        if let Some(value) = object.get("const") {
            if value.is_object() || value.is_array() {
                return false;
            }
            size.chars += value.as_str().map_or(0, |value| value.chars().count());
        }
        if let Some(choices) = object.get("anyOf") {
            // Limit siblings to annotations so every constraint is checked in a branch.
            return object
                .keys()
                .all(|key| matches!(key.as_str(), "anyOf" | "title" | "description"))
                && choices.as_array().is_some_and(|choices| {
                    !choices.is_empty() && choices.iter().all(|choice| visit(choice, depth, size))
                });
        }
        let schema_type = match &schema["type"] {
            Value::String(kind) => Some(kind.as_str()),
            Value::Array(kinds)
                if kinds.len() == 2 && kinds.iter().any(|kind| kind.as_str() == Some("null")) =>
            {
                kinds
                    .iter()
                    .filter_map(Value::as_str)
                    .find(|kind| *kind != "null")
            }
            _ => None,
        };
        match schema_type {
            Some("object") if depth < 10 => {
                let (Some(fields), Some(required)) = (
                    schema["properties"].as_object(),
                    schema["required"].as_array(),
                ) else {
                    return false;
                };
                size.properties += fields.len();
                size.chars += fields.keys().map(|key| key.chars().count()).sum::<usize>();
                schema["additionalProperties"] == false
                    && required.len() == fields.len()
                    && fields.iter().all(|(name, value)| {
                        required.iter().any(|value| value.as_str() == Some(name))
                            && visit(value, depth + 1, size)
                    })
            }
            Some("array") if depth < 10 => visit(&schema["items"], depth + 1, size),
            Some("string" | "number" | "integer" | "boolean" | "null") => true,
            _ => false,
        }
    }
    let mut size = Size::default();
    schema["type"] == "object"
        && visit(schema, 0, &mut size)
        && size.properties <= 5000
        && size.enum_values <= 1000
        && size.chars <= 120_000
}
