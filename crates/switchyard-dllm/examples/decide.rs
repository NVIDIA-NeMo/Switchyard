// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Run a local choice decision: cargo run -p switchyard-dllm --example decide -- options.json.
use switchyard_dllm::Engine;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = std::fs::read_to_string(std::env::args().nth(1).ok_or("expected options.json")?)?;
    let engine = Engine::load(serde_json::from_str(&options)?).await?;
    let request = serde_json::from_value(serde_json::json!({
        "model": "dllm/deciffusion",
        "input": [{"role": "user", "content": [{"type": "text", "text": "What is 2 + 2?"}]}],
        "questions": {"route": {
            "instructions": "Select the model tier needed to answer the user's request.",
            "kind": {"type": "choice", "data": {"options": [
                {"id": "capable", "description": "Complex reasoning, difficult coding, or expert analysis."},
                {"id": "efficient", "description": "Simple questions and routine tasks."}
            ]}}
        }}
    }))?;
    let response = engine.decide(request).await?;
    println!("{}", serde_json::to_string_pretty(&response)?);
    engine.close().await?;
    Ok(())
}
