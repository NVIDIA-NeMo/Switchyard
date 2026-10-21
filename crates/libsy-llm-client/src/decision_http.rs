// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Authenticated, buffered HTTP transport for decision providers.

use std::time::Duration;

use reqwest::{Url, header::HeaderValue};
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::client::convert_reqwest_error;
use crate::{LlmClientError, Result, metrics};

pub(crate) struct DecisionHttpClient {
    client: reqwest::Client,
    endpoint: Url,
    api_key: String,
}

impl DecisionHttpClient {
    pub(crate) fn new(endpoint: Url, api_key: String, timeout: Duration) -> Result<Self> {
        if HeaderValue::try_from(format!("Bearer {api_key}")).is_err() {
            return Err(LlmClientError::Configuration {
                message: "Decision API key cannot be encoded as an HTTP header".into(),
            });
        }
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(convert_reqwest_error)?;
        Ok(Self {
            client,
            endpoint,
            api_key,
        })
    }

    pub(crate) async fn post<T: DeserializeOwned>(&self, body: &Value) -> Result<T> {
        let response = self
            .client
            .post(self.endpoint.clone())
            .bearer_auth(&self.api_key)
            .json(body)
            .send()
            .await
            .inspect_err(|_| metrics::record_upstream_attempt(None))
            .map_err(convert_reqwest_error)?;
        let status = response.status();
        let body = response
            .bytes()
            .await
            .inspect_err(|_| metrics::record_upstream_attempt(None))
            .map_err(convert_reqwest_error)?;
        metrics::record_upstream_attempt(Some(status.as_u16()));
        if !status.is_success() {
            return Err(LlmClientError::UpstreamHttp {
                status,
                body: String::from_utf8_lossy(&body).replace(&self.api_key, "[REDACTED]"),
            });
        }
        serde_json::from_slice(&body).map_err(|source| LlmClientError::InvalidResponse {
            source: Box::new(source),
        })
    }
}
