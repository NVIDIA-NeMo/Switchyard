// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Research agent driving the raw `run` stream with **client-less** targets.
//!
//! With no client, every `driver.call_llm_target` is offloaded as a promise the orchestrator
//! surfaces as a `CallLlm` step. The agent makes the "real" model call itself and
//! fulfills the promise — this is the offload/streaming path ("ask, don't call").
//! The judge and selected target show up as two `model call:` lines. Run with:
//!   cargo run -p libsy --example research_agent_core

use std::sync::Arc;

use switchyard_libsy::{
    Algorithm, LibsyError, LlmTarget, LlmTargetSet, LlmTaskClassifier, Result, Step,
    TaskClassifierConfig,
};
use switchyard_protocol::{
    completion_text, text_request, text_response, Context, Decision, LlmResponse, Request, Response,
};
use tokio_stream::StreamExt;

const CLASSIFIER: &str = "classifier/model";
const STRONG: &str = "strong/model";
const WEAK: &str = "weak/model";
/// Lowest judge-estimated solve probability that still routes to the weak model.
const BASE_THRESHOLD: f64 = 0.5;

/// The "real" model call the agent makes to fulfill a promise. The core never
/// makes the call itself — it hands back a request and waits for the response.
/// The model to call is the routing decision's selection, read off the promise.
async fn call_model(model: &str) -> Response {
    println!("  -> model call: {model}");
    let completion = if model == CLASSIFIER {
        r#"{"recommended_route":"efficient","p_solve":0.9,"confidence":0.9,"abstain":false,"capability_boundary":"supported","primary_rule":"SUP-1","crux":"bounded task"}"#.to_string()
    } else {
        format!("answer from {model}")
    };
    Response {
        llm_response: LlmResponse::Agg(text_response(None, completion)),
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

    async fn run(&mut self, question: &str) -> Result<String> {
        let mut notes = Vec::new();
        for step in self.plan(question) {
            let request = Request {
                llm_request: text_request(Some("auto".to_string()), step),
                raw_request: None,
                metadata: None,
            };
            let stream = self
                .algo
                .clone()
                .run_stream(Context::default(), request, None);
            tokio::pin!(stream);
            while let Some(update) = stream.next().await {
                match update? {
                    Step::CallLlm(call) => {
                        // Perform the model call the algorithm asked for, then fulfill.
                        let response = call_model(call.get_decision().selected_model()).await;
                        call.respond(Ok(response))?;
                    }
                    // Decisions stream in as the algorithm makes them.
                    Step::Decision(decision) => print_decision(decision.as_ref()),
                    Step::ReturnToAgent(response) => {
                        let aggregate =
                            response.llm_response.into_agg().await.map_err(|error| {
                                LibsyError::external("aggregating response", error)
                            })?;
                        notes.push(completion_text(&aggregate));
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
async fn main() -> Result<()> {
    let target_set = targets();
    // Resolving every target up front means an unknown name fails here rather than on the
    // first request, after the judge call has already been made.
    let classifier = target_set.get_target(CLASSIFIER)?;
    let weak = target_set.get_target(WEAK)?;
    let strong = target_set.get_target(STRONG)?;
    let algo: Arc<dyn Algorithm> = Arc::new(LlmTaskClassifier::new(
        classifier,
        weak,
        strong,
        TaskClassifierConfig {
            base_threshold: BASE_THRESHOLD,
            min_confidence: 0.0,
            capability_elevated_floor: None,
            session_affinity: false,
            message_hash_fallback: false,
            recent_turn_window: None,
        },
    )?);

    let mut agent = ResearchAgent { algo };
    println!("{}", agent.run("what is switchyard?").await?);
    Ok(())
}
