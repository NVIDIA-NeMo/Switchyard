// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One local routing decision over stdin/stdout for the Claude Code mod.

use std::{env, fs, io, process::ExitCode};

use serde_json::{Value, json};
use switchyard_protocol::{LlmRequest, ModelId, Request};
use switchyard_runner::Runner;

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    match decide().await {
        Ok(decision) => {
            println!("{decision}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

async fn decide() -> Result<Value, String> {
    let config = env::args().nth(1).ok_or("Expected a TOML config path")?;
    let source = fs::read_to_string(config).map_err(|error| error.to_string())?;
    let messages =
        serde_json::from_reader(io::stdin().lock()).map_err(|_| "Invalid routing messages")?;
    let runner =
        Runner::from_toml_for_model_selection(&source).map_err(|error| error.to_string())?;
    let route = runner
        .route("switchyard")
        .ok_or("Missing switchyard route")?;
    let request = Request {
        llm_request: LlmRequest {
            model: Some("switchyard".into()),
            messages,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut expected = request.llm_request.clone();
    let outcome = route
        .decide(request)
        .await
        .map_err(|error| error.execution_error_summary().kind.as_str().to_owned())?;
    expected.model = outcome.request.llm_request.model.clone();
    if outcome.response.is_some() || outcome.request.llm_request != expected {
        return Err("Routing must only select a model".into());
    }
    let decision = runner
        .describe_decision(&ModelId::from("switchyard"), &outcome)
        .ok_or("Routing selected an unresolved target")?;
    Ok(json!({ "target": decision.selected.target, "model": decision.selected.model }))
}
