// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, sync::Arc};

use async_trait::async_trait;
use parking_lot::Mutex;
use switchyard_libsy::{Random, TargetPrompts};
use switchyard_llm_client::ClientRouter;
use switchyard_protocol::{
    ContentBlock, LlmClientError, LlmResponse, ModelId, Request, Response, RoutedLlmClient,
    text_response,
};

const WEAK: &str = "weak/model";
const STRONG: &str = "strong/model";

#[derive(Default)]
struct RecordingClient {
    calls: Mutex<Vec<Request>>,
    overflow: Option<ModelId>,
}

#[async_trait]
impl RoutedLlmClient for RecordingClient {
    async fn call(&self, request: Request) -> Result<Response, LlmClientError> {
        let model = request.model_id().unwrap_or_default();
        self.calls.lock().push(request);
        if self.overflow.as_ref() == Some(&model) {
            return Err(LlmClientError::ContextWindowExceeded {
                model,
                message: "too long".to_string(),
            });
        }
        Ok(Response {
            llm_response: LlmResponse::Agg(text_response(Some(model.to_string()), "ok")),
            metadata: None,
        })
    }
}

fn instruction_text(request: &Request) -> Vec<&str> {
    request
        .llm_request
        .instructions
        .iter()
        .flat_map(|instruction| instruction.content.iter())
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn fallback_call_receives_the_new_targets_prompt() -> switchyard_libsy::Result<()> {
    let client = Arc::new(RecordingClient {
        calls: Mutex::new(Vec::new()),
        overflow: Some(ModelId::from(WEAK)),
    });
    let routed_client: Arc<dyn RoutedLlmClient> = client.clone();
    let clients = HashMap::from([
        (ModelId::from(WEAK), Arc::clone(&routed_client)),
        (ModelId::from(STRONG), routed_client),
    ]);
    let prompts = TargetPrompts::default()
        .with(WEAK, "weak prompt")
        .with(STRONG, "strong prompt");
    let algorithm = Random::new(
        vec![ModelId::from(WEAK), ModelId::from(STRONG)],
        Some(vec![1.0, 0.0]),
        Some(1),
    )?;

    switchyard_llm_client::run(
        Arc::new(algorithm),
        ClientRouter::new_with_target_prompts(clients, prompts),
        Request::default(),
        None,
    )
    .await?;

    let calls = client.calls.lock();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].model_id().as_deref(), Some(WEAK));
    assert_eq!(instruction_text(&calls[0]), ["weak prompt"]);
    assert_eq!(calls[1].model_id().as_deref(), Some(STRONG));
    assert_eq!(instruction_text(&calls[1]), ["strong prompt"]);
    Ok(())
}
