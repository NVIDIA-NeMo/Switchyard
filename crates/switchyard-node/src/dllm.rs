// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Node access to the engine's dedicated owner thread.
use crate::{parse, routing_error};
use napi_derive::napi;

#[napi]
pub struct NativeDllmEngine {
    inner: switchyard_dllm::Engine,
}

#[napi]
impl NativeDllmEngine {
    #[napi(factory)]
    pub async fn load(options: String) -> napi::Result<Self> {
        Ok(Self {
            inner: switchyard_dllm::Engine::load(parse(&options)?)
                .await
                .map_err(routing_error)?,
        })
    }
    #[napi]
    pub async fn decide(&self, request: String) -> napi::Result<String> {
        let response = self
            .inner
            .decide(parse(&request)?)
            .await
            .map_err(routing_error)?;
        serde_json::to_string(&response).map_err(routing_error)
    }
    #[napi]
    pub async fn close(&self) -> napi::Result<()> {
        self.inner.close().await.map_err(routing_error)
    }
}
