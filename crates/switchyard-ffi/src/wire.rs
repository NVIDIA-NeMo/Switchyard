// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-shaped inputs and calls for the shared routing lifecycle.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use switchyard_protocol::{LlmRequest, LlmResponse, Metadata, ModelId, Response, WireFormat};
use tokio_util::sync::CancellationToken;

use super::{Result, RunContext};

#[derive(Deserialize)]
pub(super) struct Input {
    pub(super) request: serde_json::Value,
    #[serde(flatten)]
    pub(super) context: RunContext,
    #[serde(flatten)]
    pub(super) options: Options,
}

#[derive(Deserialize)]
pub(super) struct Options {
    format: WireFormat,
    #[serde(default)]
    target_formats: HashMap<ModelId, WireFormat>,
}

#[derive(Serialize)]
pub(super) struct PreparedRequest<'a> {
    model: &'a ModelId,
    format: WireFormat,
    request: serde_json::Value,
}

impl Options {
    pub(super) fn decode_request(&self, body: &serde_json::Value) -> Result<LlmRequest> {
        Ok(switchyard_translation::decode_request(self.format, body)?)
    }

    fn format_for(&self, model: &ModelId) -> WireFormat {
        self.target_formats
            .get(model)
            .copied()
            .unwrap_or(self.format)
    }

    // Each candidate needs its own model stamp without changing the algorithm's request.
    pub(super) fn prepare<'a>(
        &self,
        models: &'a [ModelId],
        request: &LlmRequest,
        cancellation: &CancellationToken,
    ) -> Result<Vec<PreparedRequest<'a>>> {
        models
            .iter()
            .map(|model| {
                if cancellation.is_cancelled() {
                    return Err("routing cancelled".into());
                }
                let mut request = request.clone();
                switchyard_translation::prepare_request_for_target(&mut request, model, None);
                let format = self.format_for(model);
                Ok(PreparedRequest {
                    model,
                    format,
                    request: switchyard_translation::encode_request(&request, format)?,
                })
            })
            .collect()
    }

    pub(super) fn decode_response(
        &self,
        body: &serde_json::Value,
        served_model: Option<ModelId>,
        models: &[ModelId],
    ) -> Result<Response> {
        let model = served_model
            .or_else(|| match models {
                [model] => Some(model.clone()),
                _ => None,
            })
            .filter(|model| models.contains(model))
            .ok_or("host reply must identify a requested target")?;
        let response =
            switchyard_translation::decode_aggregated_response(body, self.format_for(&model))?;
        Ok(Response {
            llm_response: LlmResponse::Agg(response),
            metadata: Some(Metadata {
                served_model: Some(model),
                ..Default::default()
            }),
            upstream_headers: Default::default(),
        })
    }
}
