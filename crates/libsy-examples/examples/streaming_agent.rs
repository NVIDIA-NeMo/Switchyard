// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Agent that forwards a streamed response to the caller token-by-token.
//!
//! Drives [`Algorithm::run_stream`] with a **client-less** target, so each model call is
//! offloaded as a [`Step::CallLlm`]. The agent serves it with a *streaming* response
//! ([`LlmResponse::Stream`]) rather than a buffered one. That stream rides untouched
//! through the algorithm and returns as [`Step::ReturnToAgent`], where the agent drives it
//! and prints each [`LlmResponseChunk`] as it arrives — true token streaming end to end.
//! Contrast [`Algorithm::run`], which would aggregate the same stream into one buffered
//! answer. Run with:
//!   cargo run -p libsy-examples --example streaming_agent

use std::error::Error;
use std::io::Write;
use std::sync::Arc;

use futures::StreamExt;
use libsy::{
    Algorithm, Context, LlmResponse, LlmResponseChunk, LlmResponseStream, LlmTarget, LlmTargetSet,
    Request, Response, Step,
};
use libsy_examples::rand::RandomOrchAlgo;
use libsy_protocol::{completion_text, text_request};

/// The "real" model call the agent makes to fulfill an offloaded promise: a response
/// whose body is a live token stream (`MessageStart`, one `TextDelta` per token,
/// `MessageStop`). A real integrator would map an upstream provider SSE stream here.
fn streaming_response(model: &str, tokens: &[&str]) -> Response {
    let mut chunks = vec![LlmResponseChunk::MessageStart {
        id: None,
        model: Some(model.to_string()),
    }];
    for token in tokens {
        chunks.push(LlmResponseChunk::TextDelta {
            index: 0,
            text: token.to_string(),
        });
    }
    chunks.push(LlmResponseChunk::MessageStop {
        reason: Some("stop".to_string()),
    });

    let stream: LlmResponseStream = futures::stream::iter(chunks.into_iter().map(Ok)).boxed();
    Response {
        llm_response: LlmResponse::Stream(stream),
        metadata: None,
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    // One client-less target -> its call is offloaded for us to serve.
    let targets = LlmTargetSet::new(vec![LlmTarget {
        semantic_name: "stream/model".to_string(),
        llm_client: None,
    }]);
    let algo: Arc<dyn Algorithm> = Arc::new(RandomOrchAlgo::new(targets));

    let request = Request {
        llm_request: text_request(Some("auto".to_string()), "tell me about switchyard"),
        raw_request: None,
        metadata: None,
    };
    let stream = algo.run_stream(Context::default(), request);
    tokio::pin!(stream);

    while let Some(step) = stream.next().await {
        match step? {
            Step::CallLlm(call) => {
                // Serve the offloaded call with a streaming response.
                let model = call.get_decision().selected_model().to_string();
                let tokens = ["Switch", "yard ", "routes ", "LLM ", "traffic."];
                call.respond(Ok(streaming_response(&model, &tokens)))?;
            }
            Step::Decision(decision) => {
                println!(
                    "decision: {} ({})",
                    decision.selected_model(),
                    decision.reasoning().unwrap_or_default()
                );
            }
            Step::ReturnToAgent(response) => match response.llm_response {
                // The stream reached the agent untouched: print each token as it arrives.
                LlmResponse::Stream(mut chunks) => {
                    print!("agent sees: ");
                    while let Some(chunk) = chunks.next().await {
                        if let LlmResponseChunk::TextDelta { text, .. } = chunk? {
                            print!("{text}");
                            std::io::stdout().flush().ok();
                        }
                    }
                    println!();
                }
                // A buffered backend would land here instead.
                LlmResponse::Agg(agg) => {
                    println!("agent sees (buffered): {}", completion_text(&agg))
                }
            },
        }
    }
    Ok(())
}
