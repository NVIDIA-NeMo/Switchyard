// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Node.js bindings for caller-owned completion dispatch and routing state.

use napi_derive::napi;
use switchyard_protocol::{ModelId, Request, text_request};
use switchyard_runner::{Runner, RunnerError};
use tokio_util::sync::CancellationToken;

// The JS wrapper turns these fixed prefixes into public Error codes.
fn error(code: &str, message: &str) -> napi::Error {
    napi::Error::from_reason(format!("{code}: {message}"))
}

#[napi]
#[derive(Default)]
pub struct Cancellation {
    token: CancellationToken,
}

#[napi]
impl Cancellation {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    #[napi]
    pub fn cancel(&self) {
        self.token.cancel();
    }
}

#[napi(object)]
pub struct Decision {
    pub target: String,
    pub model: String,
}

#[napi]
pub struct NativeRunner {
    runner: Runner,
}

#[napi]
impl NativeRunner {
    #[napi(constructor)]
    pub fn new(source: String) -> napi::Result<Self> {
        let runner = Runner::from_toml_for_model_selection(&source).map_err(|err| match err {
            RunnerError::Configuration { message, .. } => error("ERR_CONFIG", &message),
            _ => error("ERR_CONFIG", "Invalid model-selection configuration"),
        })?;
        Ok(Self { runner })
    }

    #[napi]
    pub async fn decide(
        &self,
        route_id: String,
        prompt: String,
        cancellation: &Cancellation,
    ) -> napi::Result<Decision> {
        // napi-rs retains both JS objects while this async method borrows them.
        tokio::select! {
            biased;
            _ = cancellation.token.cancelled() => Err(error("ABORT_ERR", "Routing cancelled")),
            result = select_model(&self.runner, route_id, prompt) => result,
        }
    }
}

async fn select_model(runner: &Runner, route_id: String, prompt: String) -> napi::Result<Decision> {
    let route = runner
        .route(&route_id)
        .ok_or_else(|| error("ERR_UNKNOWN_ROUTE", "Unknown route ID"))?;
    let request = Request {
        llm_request: text_request(Some(route_id.clone()), prompt),
        ..Default::default()
    };
    let mut expected = request.llm_request.clone();
    let outcome = route
        .decide(request)
        .await
        .map_err(|err| error("ERR_ROUTING", err.execution_error_summary().kind.as_str()))?;
    expected.model = outcome.request.llm_request.model.clone();
    if outcome.response.is_some() || outcome.request.llm_request != expected {
        return Err(error(
            "ERR_UNSUPPORTED_OUTCOME",
            "Routing changed the completion request or produced an answer",
        ));
    }
    let decision = runner
        .describe_decision(&ModelId::from(route_id), &outcome)
        .ok_or_else(|| {
            error(
                "ERR_UNSUPPORTED_OUTCOME",
                "Routing selected an unresolved target",
            )
        })?;
    Ok(Decision {
        target: decision.selected.target,
        model: decision.selected.model.to_string(),
    })
}
