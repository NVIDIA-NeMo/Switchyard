// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Computes a seeded random route without dispatching the selected target.
//!
//! Run with:
//! `cargo run -p switchyard-libsy --example decision_only`

use std::sync::Arc;

use futures::StreamExt;
use switchyard_libsy::algorithms::Random;
use switchyard_libsy::{
    Algorithm, Context, DecisionStep, LibsyError, LlmTarget, LlmTargetSet, Request, Result,
};
use switchyard_protocol::text_request;

#[tokio::main]
async fn main() -> Result<()> {
    let targets = LlmTargetSet::new(
        ["fast", "strong"]
            .into_iter()
            .map(|name| LlmTarget {
                semantic_name: name.to_string(),
                llm_client: None,
            })
            .collect(),
    );
    let algorithm: Arc<dyn Algorithm> =
        Arc::new(Random::new(targets, Some(vec![3.0, 1.0]), Some(42))?);
    let request = Request {
        llm_request: text_request(Some("auto".to_string()), "Explain Rust ownership."),
        raw_request: None,
        metadata: None,
    };

    let stream = algorithm.run_decision_stream(Context::default(), request);
    tokio::pin!(stream);
    while let Some(step) = stream.next().await {
        match step? {
            DecisionStep::CallLlm(_) => {
                return Err(LibsyError::AlgorithmError {
                    message: "random routing does not require a supporting LLM call".to_string(),
                });
            }
            DecisionStep::Decision(decision) => {
                println!("trace: {}", decision.selected_model());
            }
            DecisionStep::FinalDecision(decision) => {
                println!("selected without dispatch: {}", decision.selected_model());
            }
        }
    }
    Ok(())
}
