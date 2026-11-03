// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configured routing with trusted requirements supplied by the Node host.

use std::collections::HashMap;

use napi_derive::napi;
use serde::Serialize;
use serde_json::{Value, json};
use switchyard_protocol::{AggLlmResponse, LlmClientError, LlmRequest, ModelId, Request};
use switchyard_runner::{Runner, RunnerError, require_target_labels};

use crate::{error, parse, routing_error};

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RunnerDecision {
    selected_model_ids: Vec<ModelId>,
    request: LlmRequest,
    fallback_request: Option<LlmRequest>,
    candidate_requests: HashMap<ModelId, LlmRequest>,
    response: Option<AggLlmResponse>,
    metadata: Option<Value>,
}

/// Reusable Runner with native clients for routing-time model calls.
#[napi]
pub struct NativeRunner {
    runner: Runner,
}

#[napi]
impl NativeRunner {
    /// Loads the same deployment configuration as Rust Runner.
    #[napi(constructor)]
    pub fn new(source: String) -> napi::Result<Self> {
        let runner = Runner::from_toml(&source).map_err(|err| error("ERR_CONFIG", err))?;
        Ok(Self { runner })
    }

    /// Admits host requirements before routing; the caller serves the final completion.
    #[napi]
    pub async fn decide(
        &self,
        request_json: String,
        required_labels: Option<Vec<String>>,
        task_id: Option<String>,
    ) -> napi::Result<String> {
        let mut request = Request {
            llm_request: parse(&request_json)?,
            ..Default::default()
        };
        switchyard_translation::normalize_compaction(&mut request.llm_request, None);
        if let Some(task_id) = task_id {
            request.metadata.get_or_insert_default().task_id = Some(task_id);
        }
        if let Some(labels) = required_labels {
            require_target_labels(&mut request, labels)
                .map_err(|err| error("ERR_INVALID_REQUEST", err))?;
        }
        let model = request
            .llm_request
            .model
            .as_deref()
            .ok_or_else(|| error("ERR_INVALID_REQUEST", "request.model is required"))?;
        let route = self
            .runner
            .route(model)
            .ok_or_else(|| error("ERR_INVALID_REQUEST", "unknown route model"))?;
        let outcome = route.decide(request).await.map_err(|err| match err {
            RunnerError::Client(LlmClientError::InvalidRequest { .. }) => {
                error("ERR_INVALID_REQUEST", err)
            }
            _ => routing_error(err),
        })?;
        let response = match outcome.response {
            Some(response) => Some(
                response
                    .llm_response
                    .into_agg()
                    .await
                    .map_err(routing_error)?,
            ),
            None => None,
        };
        let metadata = outcome.metadata.map(|metadata| {
            json!({
                "outcomeId": metadata.outcome_id(),
                "algorithm": metadata.algorithm,
                "evidence": metadata.evidence,
            })
        });
        serde_json::to_string(&RunnerDecision {
            selected_model_ids: outcome.selected_model_ids,
            request: outcome.request.llm_request,
            fallback_request: outcome.fallback_request.map(|request| request.llm_request),
            candidate_requests: outcome
                .candidate_requests
                .into_iter()
                .map(|(model, request)| (model, request.llm_request))
                .collect(),
            response,
            metadata,
        })
        .map_err(routing_error)
    }
}
