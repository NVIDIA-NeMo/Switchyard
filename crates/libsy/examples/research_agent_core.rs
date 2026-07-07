// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Second research-agent example — driving the CORE `feed`/`optimize` API by
//! hand (no `RoutedClient`), routed by a multi-round LLM classifier. Watch
//! `routed_call()` loop twice per request: once for the classifier call, once for
//! the routed call (the two `model call:` lines). `RoutedClient` hides exactly this.
//! Run with:
//!   cargo run -p libsy --example research_agent_core

use libsy::llm_class::{ClassifierRoutingDecision, LlmClassifierAlgorithm};
use libsy::{
    AgentApiOptAlgorithm, AgentApiOptInput, AgentApiRequest, AgentApiResponse, Decision,
    EnrichmentData,
};
use std::error::Error;

const CLASSIFIER: &str = "classifier/model";
const STRONG: &str = "strong/model";
const WEAK: &str = "weak/model";

/// Stand-in for the host's real model call. The core API never makes the call
/// itself: `optimize()` hands back an `AgentApiRequest` and the host performs
/// the call and feeds the result back ("ask, don't call"). That is what keeps
/// libsy transport- and provider-agnostic — no HTTP client in the core.
async fn call_model(req: &AgentApiRequest) -> AgentApiResponse {
    // Printed so the two rounds (classifier, then routed target) are visible.
    println!("  -> model call: {}", req.model);
    let completion = if req.model == CLASSIFIER {
        "0.8".to_string() // strong-win-rate score; >= threshold routes to strong
    } else {
        format!("answer from {}", req.model)
    };
    AgentApiResponse { completion }
}

struct ResearchAgent {
    // The agent owns the algorithm *factory*, not an optimizer: a fresh, isolated
    // optimizer is minted per request (routing state never leaks between requests).
    algorithm: Box<dyn AgentApiOptAlgorithm<ClassifierRoutingDecision>>,
}

impl ResearchAgent {
    /// Trivial plan: one lookup per question (stub).
    fn plan(&self, question: &str) -> Vec<String> {
        vec![format!("look up: {question}")]
    }

    /// The core-API equivalent of `RoutedClient::complete`: mint a per-request
    /// optimizer, feed the request, and drive `feed`/`optimize` to `Return`,
    /// making each model call the optimizer asks for. This is the reusable loop
    /// the wrapper hides — the agent's `run` just builds requests and calls it.
    async fn routed_call(
        &self,
        request: AgentApiRequest,
    ) -> Result<AgentApiResponse, Box<dyn Error>> {
        // 1. Mint a per-request optimizer and feed the inbound request. `feed`
        //    takes `&mut self`, so a session is single-owner: the host serializes
        //    inputs (here, trivially, one request).
        let mut optimizer = self.algorithm.optimizer();
        optimizer
            .feed(
                AgentApiOptInput::Request(request),
                EnrichmentData::default(),
            )
            .await?;

        // 2. Loop: the optimizer decides which model calls to make; the host makes
        //    them and feeds the responses back. `Return` ends the session. The
        //    classifier runs this loop TWICE (classify, then route) — but the loop
        //    code is identical to a single-round router's (the whole point).
        let mut last = None;
        while let Decision::ModelInference(decision) = optimizer.optimize().await? {
            // `decision.decision_reasoning` / `decision.decision_info` explain the
            // route (great for logging/eval); nothing forces you to consume them.
            for req in decision.requests {
                let model = req.model.clone();
                let answer = call_model(&req).await;
                // MISSING: the `Response` input reuses `AgentApiRequest` (completion
                // stuffed into `prompt`) instead of carrying `AgentApiResponse` — a
                // known rough edge. Token usage / latency also have nowhere to go yet.
                optimizer
                    .feed(
                        AgentApiOptInput::Response(AgentApiRequest {
                            prompt: answer.completion.clone(),
                            model,
                        }),
                        EnrichmentData::default(),
                    )
                    .await?;
                last = Some(answer);
            }
        }

        // The last response is the routed answer; the intermediate classifier call
        // is consumed inside the loop and never surfaces here.
        last.ok_or_else(|| "optimizer returned before requesting any model call".into())
    }

    async fn run(&self, question: &str) -> Result<String, Box<dyn Error>> {
        let mut notes = Vec::new();
        for step in self.plan(question) {
            // The agent builds a request (`model: "auto"` is a placeholder the
            // router replaces) and hands it off — it never drives the loop itself.
            let request = AgentApiRequest {
                prompt: step,
                model: "auto".to_string(),
            };
            notes.push(self.routed_call(request).await?.completion);
        }
        Ok(notes.join("\n"))
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    // Configure routing once. The classifier is a two-round algorithm;
    // routed_call's loop does not care how many rounds the algorithm uses.
    let algorithm = LlmClassifierAlgorithm {
        classifier_model: CLASSIFIER.to_string(),
        strong_model: STRONG.to_string(),
        weak_model: WEAK.to_string(),
        threshold: 0.5,
    };
    let agent = ResearchAgent {
        algorithm: Box::new(algorithm),
    };
    println!("{}", agent.run("what is switchyard?").await?);
    Ok(())
}
