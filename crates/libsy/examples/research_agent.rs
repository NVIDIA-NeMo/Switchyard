// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal example: dropping the `RoutedClient` wrapper into a research agent,
//! routed by a multi-round LLM classifier.
//!
//! Almost everything is a stub — the point is the *integration*. The agent makes
//! one `client.complete(..)` call per step; under the hood the wrapper runs the
//! classifier call *and then* the routed call (see the two `model call:` lines
//! per step). That multi-step routing is completely invisible to the agent. Run:
//!   cargo run -p libsy --example research_agent

use async_trait::async_trait;
use libsy::client::{ModelCaller, RoutedClient};
use libsy::llm_class::{ClassifierRoutingDecision, LlmClassifierAlgorithm};
use libsy::{AgentApiRequest, AgentApiResponse};
use std::error::Error;

const CLASSIFIER: &str = "classifier/model";
const STRONG: &str = "strong/model";
const WEAK: &str = "weak/model";

/// Stub transport. Real integrators supply their own `ModelCaller`, or use
/// `RoutedClient::with_http(algorithm, base_url, api_key)`.
struct StubCaller;

#[async_trait]
impl ModelCaller for StubCaller {
    async fn call(&self, req: AgentApiRequest) -> Result<AgentApiResponse, Box<dyn Error>> {
        // Printing each call makes the multi-step routing visible: the classifier
        // call first, then the routed model call.
        println!("  -> model call: {}", req.model);
        let completion = if req.model == CLASSIFIER {
            // The classifier is asked for a strong-win-rate score; 0.8 >= threshold
            // routes the follow-up call to the strong model.
            "0.8".to_string()
        } else {
            format!("answer from {}", req.model)
        };
        Ok(AgentApiResponse { completion })
    }
}

/// A research agent that owns a routed client and nothing routing-specific.
struct ResearchAgent {
    client: RoutedClient<ClassifierRoutingDecision>,
}

impl ResearchAgent {
    /// Trivial plan: one lookup per sub-question (stub).
    fn plan(&self, question: &str) -> Vec<String> {
        vec![format!("look up: {question}")]
    }

    /// Trivial synthesis (stub).
    fn synthesize(&self, notes: Vec<String>) -> String {
        notes.join("\n")
    }

    async fn run(&self, question: &str) -> Result<String, Box<dyn Error>> {
        let mut notes = Vec::new();
        for step in self.plan(question) {
            // The only integration point: one call from the agent's view. The
            // classifier + routed calls happen inside the wrapper. `model: "auto"`
            // is a placeholder the router replaces.
            let answer = self
                .client
                .complete(AgentApiRequest {
                    prompt: step,
                    model: "auto".to_string(),
                })
                .await?;
            notes.push(answer.completion);
        }
        Ok(self.synthesize(notes))
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    // Configure routing once. The classifier scores each request with a model call,
    // then routes to the strong or weak model — a two-round algorithm the agent is
    // oblivious to. Swapping in a one-round `RandomRouterAlgorithm` (or
    // `RoutedClient::with_http(..)`) needs no change to `ResearchAgent`.
    let algorithm = LlmClassifierAlgorithm {
        classifier_model: CLASSIFIER.to_string(),
        strong_model: STRONG.to_string(),
        weak_model: WEAK.to_string(),
        threshold: 0.5,
    };
    let client = RoutedClient::new(Box::new(algorithm), Box::new(StubCaller));

    let agent = ResearchAgent { client };
    println!("{}", agent.run("what is switchyard?").await?);
    Ok(())
}
