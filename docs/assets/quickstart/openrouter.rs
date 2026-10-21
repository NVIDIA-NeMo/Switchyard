// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, sync::Arc};
use futures::StreamExt;
use switchyard_libsy::{
    Algorithm, LibsyError, PickerMode, RuntimeModels, StageRouter, StageRouterConfig, State, Step,
};
use switchyard_protocol::{Category, LlmResponse, Request, Response, completion_text, text_request};
use switchyard_translation::{WireFormat, decode_aggregated_response, encode_request};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Start with Efficient when tool history gives no reason to change models.
    let router = Arc::new(StageRouter::new(StageRouterConfig::new(
        PickerMode::EfficientFirst, 0.5,
    ))?);
    let efficient = "z-ai/glm-5.2";
    let capable = "anthropic/claude-opus-4.8";
    let client = OpenRouterClient {
        http: reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60)).build()?,
        api_key: std::env::var("OPENROUTER_API_KEY")?,
    };
    // Assign model roles; Any lists the full pool.
    let models = Arc::new(RuntimeModels::new(HashMap::from([
        (Category::Efficient, vec![efficient.into()]),
        (Category::Capable, vec![capable.into()]),
        (Category::Any, vec![efficient.into(), capable.into()]),
    ])));
    let request = Request {
        llm_request: text_request(None, "Run the tests."),
        ..Default::default()
    };

    let mut steps = router.run_stream(request, models, State::default());
    while let Some(step) = steps.next().await {
        match step? {
            Step::CallModel(call) => {
                // Give libsy the future so pending routing calls can be canceled.
                let request = call.request.clone();
                call.respond(client.call(request)).await?;
            }
            Step::CallDecision(_) => {
                // This Stage preset makes no decision calls; other presets may need a client.
                return Err("A decision client is required for this algorithm.".into());
            }
            Step::Done(outcome) => {
                println!("Selected: {}", outcome.selected_model_id()?);
                // Stage has selected the model. Your application now makes the final call.
                let response = client.call(outcome.request).await?;
                if let LlmResponse::Agg(answer) = response.llm_response {
                    println!("Answer: {}", completion_text(&answer));
                }
            }
        }
    }
    Ok(())
}

struct OpenRouterClient {
    http: reqwest::Client,
    api_key: String,
}

impl OpenRouterClient {
    async fn call(&self, request: Request) -> switchyard_libsy::Result<Response> {
        // Convert libsy's request to the provider's wire format.
        let body = encode_request(&request.llm_request, WireFormat::OpenAiChat)
            .map_err(|error| LibsyError::external("encoding OpenRouter request", error))?;
        let response = self.http
            .post("https://openrouter.ai/api/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&body)
            .send().await
            .and_then(reqwest::Response::error_for_status)
            .map_err(|error| LibsyError::external("OpenRouter request", error))?;
        let upstream_headers = response.headers().clone();
        let body = response.json::<serde_json::Value>().await
            .map_err(|error| LibsyError::external("reading OpenRouter response", error))?;
        let answer = decode_aggregated_response(&body, WireFormat::OpenAiChat)
            .map_err(|error| LibsyError::external("decoding OpenRouter response", error))?;
        Ok(Response {
            llm_response: LlmResponse::Agg(answer),
            metadata: request.metadata,
            upstream_headers,
        })
    }
}
