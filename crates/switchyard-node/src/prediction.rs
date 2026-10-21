// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Task-budget prediction requests and forecasts for the JavaScript host.

use napi_derive::napi;
use serde::Deserialize;
use serde_json::Value;
use switchyard_protocol::{DecisionResponse, ModelId};
use switchyard_token_prediction::{ResponseUsage, prediction, request};

use crate::{error, parse};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    model: ModelId,
    history: Vec<Value>,
    prior_usage: Vec<ResponseUsage>,
}

#[napi]
pub fn task_prediction_request(input_json: String) -> napi::Result<String> {
    let input: Input = parse(&input_json)?;
    serde_json::to_string(&request(input.model, &input.history, &input.prior_usage))
        .map_err(|err| error("ERR_INVALID_REQUEST", err))
}

#[napi]
pub fn task_prediction(response_json: String) -> napi::Result<String> {
    let response: DecisionResponse = parse(&response_json)?;
    let forecast = prediction(&response).map_err(|err| error("ERR_INVALID_REQUEST", err))?;
    serde_json::to_string(&forecast).map_err(|err| error("ERR_INVALID_REQUEST", err))
}
