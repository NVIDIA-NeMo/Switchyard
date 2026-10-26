// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::{Value, json};
use switchyard_protocol::{LlmRequest, Tool, ToolCall, ToolChoice};

pub(super) struct Candidate {
    pub id: String,
    pub call: ToolCall,
}

pub(super) fn candidates(request: &LlmRequest, limit: usize) -> Vec<Candidate> {
    if matches!(
        request.tool_choice,
        Some(ToolChoice::None | ToolChoice::Raw(_))
    ) || request
        .tools
        .iter()
        .any(|tool| !matches!(tool, Tool::Function(_)))
        || request.output.response_format.is_some()
        || request
            .extensions
            .fields
            .get("n")
            .and_then(Value::as_u64)
            .is_some_and(|n| n != 1)
        || ["previous_response_id", "conversation"]
            .iter()
            .any(|key| request.extensions.fields.contains_key(*key))
    {
        return Vec::new();
    }
    let mut candidates = Vec::new();
    for tool in &request.tools {
        let Tool::Function(tool) = tool else { continue };
        if let Some(ToolChoice::Tool { name }) = &request.tool_choice
            && name != &tool.name
        {
            continue;
        }
        let Ok(schema) = jsonschema::validator_for(&tool.parameters) else {
            continue;
        };
        let Some(arguments) = arguments(&tool.parameters, limit - candidates.len()) else {
            continue;
        };
        for arguments in arguments
            .into_iter()
            .filter(|arguments| schema.is_valid(arguments))
        {
            candidates.push(Candidate {
                id: format!("call_{}", candidates.len()),
                call: ToolCall {
                    id: String::new(),
                    name: tool.name.clone(),
                    arguments,
                },
            });
        }
        if candidates.len() == limit {
            break;
        }
    }
    candidates
}

fn arguments(schema: &Value, limit: usize) -> Option<Vec<Value>> {
    let mut rows = vec![json!({})];
    if let Some(required) = schema.get("required") {
        for name in required.as_array()? {
            let name = name.as_str()?;
            let values = values(schema.get("properties")?.get(name)?, limit)?;
            if rows.len().checked_mul(values.len())? > limit {
                return None;
            }
            rows = rows
                .into_iter()
                .flat_map(|row| {
                    values.iter().map(move |value| {
                        let mut row = row.clone();
                        row[name] = value.clone();
                        row
                    })
                })
                .collect();
        }
    }
    (rows.len() <= limit).then_some(rows)
}

fn values(schema: &Value, limit: usize) -> Option<Vec<Value>> {
    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
        return (!values.is_empty() && values.len() <= limit).then(|| values.clone());
    }
    if let Some(value) = schema.get("const") {
        return Some(vec![value.clone()]);
    }
    match schema.get("type")?.as_str()? {
        "boolean" => Some(vec![json!(false), json!(true)]),
        "integer" => {
            let min = schema.get("minimum")?.as_i64()?;
            let max = schema.get("maximum")?.as_i64()?;
            let count = max.checked_sub(min)?.checked_add(1)?;
            (count > 0 && usize::try_from(count).ok()? <= limit)
                .then(|| (min..=max).map(Value::from).collect())
        }
        _ => None,
    }
}
