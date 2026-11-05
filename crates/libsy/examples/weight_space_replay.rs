// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replay causal request snapshots through the native policy with decision traces.

use std::collections::HashMap;
use std::io::{self, BufRead};
use std::time::Instant;

use serde::Deserialize;
use switchyard_libsy::{State, WeightSpaceClassifier, WeightSpaceWeights};
use switchyard_protocol::{LlmRequest, Metadata, ModelId, Request};

#[derive(Deserialize)]
struct Row {
    id: String,
    task: String,
    request: LlmRequest,
    agent_id: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let path = args
        .get(1)
        .ok_or("usage: weight_space_replay WEIGHTS [COST_WEIGHT]")?;
    let cost_weight = args
        .get(2)
        .map(|v| v.parse::<f64>())
        .transpose()?
        .unwrap_or(0.75);
    let weights: WeightSpaceWeights = serde_json::from_slice(&std::fs::read(path)?)?;
    let targets: Vec<ModelId> = ["efficient", "capable"]
        .into_iter()
        .map(ModelId::from)
        .collect();
    let classifier = WeightSpaceClassifier::new(targets.clone(), weights, cost_weight)?;
    let mut states: HashMap<String, State> = HashMap::new();
    for line in io::stdin().lock().lines() {
        let row: Row = serde_json::from_str(&line?)?;
        let state = states.entry(row.task.clone()).or_default();
        let request = Request {
            llm_request: row.request,
            metadata: Some(Metadata {
                agent_id: row.agent_id,
                ..Default::default()
            }),
            raw_request: None,
        };
        let start = Instant::now();
        let trace = classifier.decide(state, &request);
        let nanos = start.elapsed().as_nanos();
        println!(
            "{}",
            serde_json::json!({"id":row.id,"task":row.task,"trace":trace,"elapsed_ns":nanos})
        );
    }
    Ok(())
}
