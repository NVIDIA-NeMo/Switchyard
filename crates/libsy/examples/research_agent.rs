// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal research agent using the [`Algorithm::run`] convenience.
//!
//! Every target owns an `RoutedLlmClient`, so the agent runs each request to completion with
//! [`Algorithm::run`]: it serves each offloaded call with the routed
//! target's `default_client` and returns the final response — no stream to drive. The
//! classifier cascade runs inside `FallThrough`; the agent never sees it. To drive the step
//! stream yourself instead, use
//! `Algorithm::run_stream`. Run with:
//!   cargo run -p libsy --example research_agent

use std::sync::Arc;

use async_trait::async_trait;
use switchyard_libsy::algorithms::{FallThrough, LlmTaskClassifier};
use switchyard_libsy::{
    Algorithm, Context, Decision, LibsyError, LlmResponse, LlmTarget, LlmTargetSet, Request,
    Response, Result, RoutedLlmClient, State,
};
use switchyard_protocol::{completion_text, text_request, text_response};

const CLASSIFIER: &str = "classifier/model";
const STRONG: &str = "strong/model";
const WEAK: &str = "weak/model";
/// Lowest judge-estimated solve probability that still routes to the weak model.
const THRESHOLD: f64 = 0.5;

/// Stub transport. Real integrators implement `RoutedLlmClient` over their own HTTP.
struct StubClient;

#[async_trait]
impl RoutedLlmClient for StubClient {
    async fn call(
        &self,
        _ctx: Context,
        _request: Request,
        decision: Arc<dyn Decision>,
    ) -> std::result::Result<Response, switchyard_protocol::LlmClientError> {
        // The model to call is the routed decision's selection, not the inbound name.
        let model = decision.selected_model().to_string();
        println!("  -> model call: {model}");
        // The judge returns a structured verdict; other models return an answer.
        let completion = if model == CLASSIFIER {
            r#"{"recommended_route":"efficient","p_solve":0.9,"confidence":0.9,"abstain":false,"capability_boundary":"supported","primary_rule":"SUP-1","crux":"bounded task"}"#.to_string()
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
                .map_err(|error| LibsyError::external("aggregating response", error))?;
            notes.push(completion_text(&aggregate));
        }
        Ok(notes.join("\n"))
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let target_set = targets();
    // Resolving every target up front means an unknown name fails here rather than on the
    // first request, after the judge call has already been made.
    let classifier = target_set.get_target(CLASSIFIER)?;
    let weak = target_set.get_target(WEAK)?;
    let strong = target_set.get_target(STRONG)?;
    let algo: Arc<dyn Algorithm> = Arc::new(
        FallThrough::<State>::new_with_state(target_set).with_classifier(Arc::new(
            LlmTaskClassifier::new(classifier, weak, strong, THRESHOLD)?,
        )),
    );

    let agent = ResearchAgent { algo };
    println!("{}", agent.run("what is switchyard?").await?);
    Ok(())
}
