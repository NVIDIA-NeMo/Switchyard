// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Research agent driving the raw `run` stream with **client-less** targets.
//!
//! With no client, every `driver.call_llm_target` is offloaded as a promise the orchestrator
//! surfaces as a `CallLlm` step. The agent makes the "real" model call itself and
//! fulfills the promise — this is the offload/streaming path ("ask, don't call").
//! The classifier's two steps show up as two `model call:` lines. Run with:
//!   cargo run -p libsy --example research_agent_core

use std::error::Error;
use std::sync::Arc;

use libsy::{
    Algorithm, Context, Decision, LlmRequest, LlmResponse, LlmTarget, LlmTargetSet, Request,
    Response, Step,
};
use libsy_examples::llm_class::LlmClassifierOrchAlgo;
use tokio_stream::StreamExt;

const CLASSIFIER: &str = "classifier/model";
const STRONG: &str = "strong/model";
const WEAK: &str = "weak/model";

/// The "real" model call the agent makes to fulfill a promise. The core never
/// makes the call itself — it hands back a request and waits for the response.
/// The model to call is the routing decision's selection, read off the promise.
async fn call_model(model: &str) -> Response {
    println!("  -> model call: {model}");
    let completion = if model == CLASSIFIER {
        "0.9".to_string()
    } else {
        format!("answer from {model}")
    };
    Response {
        llm_response: LlmResponse {
            completion,
            raw_response: None,
        },
        metadata: None,
    }
}

fn targets() -> LlmTargetSet {
    // Client-less targets -> every call is offloaded via a promise.
    let target = |name: &str| LlmTarget {
        semantic_name: name.to_string(),
        llm_client: None,
    };
    LlmTargetSet::new(vec![target(CLASSIFIER), target(STRONG), target(WEAK)])
}

struct ResearchAgent {
    algo: Arc<dyn Algorithm>,
}

impl ResearchAgent {
    /// Trivial plan: one lookup per question (stub).
    fn plan(&self, question: &str) -> Vec<String> {
        vec![format!("look up: {question}")]
    }

    async fn run(&mut self, question: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
        let mut notes = Vec::new();
        for step in self.plan(question) {
            let request = Request {
                llm_request: LlmRequest {
                    inbound_model_name: "auto".to_string(),
                    prompt: step,
                },
                raw_request: None,
                metadata: None,
            };
            let stream = self.algo.clone().run_stream(Context::default(), request);
            tokio::pin!(stream);
            while let Some(update) = stream.next().await {
                match update? {
                    Step::CallLlm(call) => {
                        // Perform the model call the algorithm asked for, then fulfill.
                        let response = call_model(call.get_decision()?.selected_model()).await;
                        call.respond(Ok(response))?;
                    }
                    // Decisions stream in as the algorithm makes them.
                    Step::Decision(decision) => print_decision(decision.as_ref()),
                    Step::ReturnToAgent(response) => {
                        notes.push(response.llm_response.completion);
                    }
                }
            }
        }
        Ok(notes.join("\n"))
    }
}

/// Print one decision the algorithm recorded — uniform access via the trait.
fn print_decision(decision: &dyn Decision) {
    println!(
        "    decision: {} ({})",
        decision.selected_model(),
        decision.reasoning().unwrap_or_default()
    );
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let algo: Arc<dyn Algorithm> = Arc::new(LlmClassifierOrchAlgo::new(
        CLASSIFIER,
        STRONG,
        WEAK,
        0.5,
        targets(),
    ));

    let mut agent = ResearchAgent { algo };
    println!("{}", agent.run("what is switchyard?").await?);
    Ok(())
}
