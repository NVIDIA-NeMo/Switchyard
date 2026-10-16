// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Routes judge calls back to the mod, which owns Claude's authenticated client.

use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;
use switchyard_protocol::{
    ContentBlock, LlmClientError, LlmResponse, Request, Response, RoutedLlmClient, text_response,
};
use tokio::time::{sleep, timeout};

use crate::{POLL, emit};

pub struct NativeClassifier {
    pub dir: PathBuf,
    pub request_id: u64,
    pub next_call: AtomicU64,
    pub cache_turn: bool,
}

#[derive(Deserialize)]
struct Reply {
    text: Option<String>,
    error: Option<String>,
}

#[async_trait]
impl RoutedLlmClient for NativeClassifier {
    async fn call(&self, request: Request) -> Result<Response, LlmClientError> {
        let call = self.next_call.fetch_add(1, Ordering::Relaxed);
        let path = self
            .dir
            .join(format!("judge-{}-{call}.json", self.request_id));
        let llm = request.llm_request;
        let instructions = llm
            .instructions
            .iter()
            .flat_map(|b| &b.content)
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        // model.complete accepts text, not a provider-specific structured-output field.
        // Send the library's schema as an instruction; libsy still validates the reply.
        emit(&json!({
            "id": self.request_id,
            "call": call,
            "cache_turn": self.cache_turn,
            "classifier": {
                "model": llm.model,
                "system": format!("{instructions}\nReturn only JSON matching this output format: {}", json!(llm.output.response_format)),
                "prompt": serde_json::to_string(&llm.messages).map_err(|e| LlmClientError::General(e.to_string()))?,
                "maxTokens": llm.output.max_output_tokens.unwrap_or(384),
            }
        })).map_err(LlmClientError::General)?;
        // Longer than the mod's native-call timeout, shorter than its decision deadline.
        let reply = timeout(Duration::from_secs(12), async {
            loop {
                if let Ok(text) = fs::read_to_string(&path)
                    && let Ok(reply) = serde_json::from_str::<Reply>(&text)
                {
                    break reply;
                }
                sleep(POLL).await;
            }
        })
        .await;
        let _ = fs::remove_file(path);
        let reply =
            reply.map_err(|_| LlmClientError::General("Native classifier timed out".into()))?;
        if reply.error.is_some() {
            return Err(LlmClientError::General(
                "Native classifier unavailable".into(),
            ));
        }
        let text = reply
            .text
            .ok_or_else(|| LlmClientError::General("Empty classifier reply".into()))?;
        Ok(Response {
            llm_response: LlmResponse::Agg(text_response(llm.model, text)),
            metadata: request.metadata,
            upstream_headers: Default::default(),
        })
    }
}
