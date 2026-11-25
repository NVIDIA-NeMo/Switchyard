// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generate task or turn metadata with one decision call and no routing algorithm.

use std::{collections::BTreeMap, env, fs, time::Duration};

use serde::Deserialize;
use serde_json::{Value, json};
use switchyard_llm_client::{OpenAiDecisionClient, SystemOneClient};
use switchyard_protocol::{DecisionQuestion, DecisionRequest, ModelId, RoutedDecisionClient};

const USAGE: &str = "usage: assess QUESTIONS.json CONTEXT.json TASK_ID [TURN_ID]";

/// Questions plus the text prepended to every question's instructions.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Rubric {
    preamble: String,
    questions: BTreeMap<String, DecisionQuestion>,
}

fn read<T: for<'a> Deserialize<'a>>(path: &str) -> Result<T, Box<dyn std::error::Error>> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    let [questions, context, task_id, turn_id @ ..] = args.as_slice() else {
        return Err(USAGE.into());
    };
    if turn_id.len() > 1 {
        return Err(USAGE.into());
    }
    let mut rubric: Rubric = read(questions)?;
    for question in rubric.questions.values_mut() {
        let text = question
            .instructions
            .as_str()
            .ok_or("instructions must be text")?;
        question.instructions = Value::String(format!("{} {text}", rubric.preamble));
    }
    let request = DecisionRequest {
        model: Some(ModelId::new(
            env::var("DECISION_MODEL").map_err(|_| "DECISION_MODEL is not set")?,
        )),
        context: read(context)?,
        questions: rubric.questions,
    };
    let endpoint = env::var("DECISION_ENDPOINT")
        .map_err(|_| "DECISION_ENDPOINT is not set")?
        .parse()?;
    let api_key = env::var("DECISION_API_KEY").map_err(|_| "DECISION_API_KEY is not set")?;
    let timeout = Duration::from_secs(30);
    let client: Box<dyn RoutedDecisionClient> = match env::var("DECISION_FORMAT")
        .as_deref()
        .unwrap_or("system_one")
    {
        "system_one" => Box::new(SystemOneClient::new(endpoint, api_key, timeout)?),
        "openai_decisions" => Box::new(OpenAiDecisionClient::new(endpoint, api_key, timeout)?),
        _ => return Err("DECISION_FORMAT must be system_one or openai_decisions".into()),
    };
    let assessment = client.call(request).await?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "task_id": task_id,
            "turn_id": turn_id.first(),
            "assessment": assessment,
        }))?
    );
    Ok(())
}
