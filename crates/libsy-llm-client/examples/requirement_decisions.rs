// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replay a prepared requirement decision request through the composite router's client.
use std::time::{Duration, Instant};

use serde_json::json;
use switchyard_llm_client::SystemOneClient;
use switchyard_protocol::{DecisionRequest, RoutedDecisionClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: requirement_decisions REQUEST.json RESULT.json".into());
    }
    let request: DecisionRequest = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let client = SystemOneClient::new(
        "https://openrouter.ai/api/alpha/decisions".parse()?,
        std::env::var("OPENROUTER_API_KEY")?,
        Duration::from_secs(60),
    )?;
    let started = Instant::now();
    let result = match client.call(request).await {
        Ok(response) => json!({"elapsed_ms": started.elapsed().as_millis(), "response": response}),
        Err(error) => {
            json!({"elapsed_ms": started.elapsed().as_millis(), "error": error.to_string()})
        }
    };
    std::fs::write(&args[2], serde_json::to_vec_pretty(&result)?)?;
    println!(
        "{}",
        json!({"elapsed_ms":result["elapsed_ms"], "ok":result.get("response").is_some(), "result_file":args[2]})
    );
    Ok(())
}
