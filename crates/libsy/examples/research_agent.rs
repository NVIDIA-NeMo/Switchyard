// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal research agent using the [`Algorithm::run`] convenience.
//!
//! Every target owns an `RoutedLlmClient`, so the agent runs each request to completion with
//! [`Algorithm::run`]: it serves each offloaded call with the routed
//! target's `default_client` and returns the final response — no stream to drive. The
//! multi-step routing (classify -> route) happens inside the classifier algorithm; the
//! agent never sees it. To drive the step stream yourself instead, use
//! `Algorithm::run_stream`. Run with:
//!   cargo run -p libsy --example research_agent

use std::sync::Arc;

use async_trait::async_trait;
use switchyard_libsy::algorithms::LlmClassifier;
use switchyard_libsy::{
    Algorithm, Context, Decision, LibsyError, LlmResponse, LlmTarget, LlmTargetSet, Request,
    Response, Result, RoutedLlmClient,
};
use switchyard_protocol::{completion_text, text_request, text_response};

const CLASSIFIER: &str = "classifier/model";
const STRONG: &str = "strong/model";
const WEAK: &str = "weak/model";

/// Stub transport. Real integrators implement `RoutedLlmClient` over their own HTTP.
struct StubClient;

#[async_trait]
impl RoutedLlmClient for StubClient {
    async fn call(
        &self,
        _ctx: Context,
        _request: Request,
        decision: Arc<dyn Decision>,
    ) -> std::result::Result<Response, switchyard_protocol::RoutedLlmClientError> {
        // The model to call is the routed decision's selection, not the inbound name.
        let model = decision.selected_model().to_string();
        println!("  -> model call: {model}");
        // The classifier returns a score; other models return an answer.
        let completion = if model == CLASSIFIER {
            "0.9".to_string()
        } else {
            format!("answer from {model}")
        };
        Ok(Response {
            llm_response: LlmResponse::Agg(text_response(None, completion)),
            metadata: None,
        })
    }
}

fn targets() -> LlmTargetSet {
    let client = Arc::new(StubClient) as Arc<dyn RoutedLlmClient>;
    let target = |name: &str| LlmTarget {
        semantic_name: name.to_string(),
        llm_client: Some(client.clone()),
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

    async fn run(&self, question: &str) -> Result<String> {
        let mut notes = Vec::new();
        for step in self.plan(question) {
            let request = Request {
                llm_request: text_request(Some("auto".to_string()), step),
                raw_request: None,
                metadata: None,
            };

            let (_trace, response) = self.algo.clone().run(Context::default(), request).await?;
            let aggregate = response
                .llm_response
                .into_agg()
                .await
                .map_err(|error| LibsyError::external_boxed("aggregating response", error))?;
            notes.push(completion_text(&aggregate));
        }
        Ok(notes.join("\n"))
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    // Configure routing once: an LLM classifier over three named targets. Swapping
    // in `Random` needs no change to the agent.
    let algo: Arc<dyn Algorithm> =
        Arc::new(LlmClassifier::new(CLASSIFIER, STRONG, WEAK, 0.5, targets()));

    let agent = ResearchAgent { algo };
    println!("{}", agent.run("what is switchyard?").await?);
    Ok(())
}
