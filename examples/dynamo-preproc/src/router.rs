// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
    sync::Mutex,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, ensure};
use serde_json::value::{RawValue, to_raw_value};
use switchyard_protocol::ModelId;
use switchyard_runner::Runner;

use crate::request;

pub const MODEL_HEADER: &str = "x-gateway-model-name";
// Match SDK 0.3's idle TTL; its retained state is reclaimed by an hourly sweep.
const SESSION_TTL: Duration = Duration::from_secs(60 * 60);
type SessionIdentity = (String, String, Option<String>);

pub struct Router {
    runner: Runner,
    sessions: Mutex<HashMap<SessionIdentity, Instant>>,
}

impl Router {
    pub fn load(routes: impl AsRef<Path>) -> Result<Self> {
        // SDK parse/build access would allow validation of decision-only options before construction.
        Ok(Self::new(Runner::load(routes)?))
    }

    pub fn new(runner: Runner) -> Self {
        Self {
            runner,
            sessions: Mutex::new(HashMap::new()),
        }
    }

    pub async fn decide(
        &self,
        body: &[u8],
        headers: &http::HeaderMap,
    ) -> Result<(Vec<u8>, String)> {
        let started = Instant::now();
        let request = {
            let raw = serde_json::from_slice(body)?;
            request::decode(&raw, headers)?
        };
        let input_model = request
            .llm_request
            .model
            .as_ref()
            .context("missing request model")?
            .as_str();
        let route_id = ModelId::from(input_model);
        let route = self
            .runner
            .route(route_id.as_str())
            .context("model is outside the configured catalog")?;
        let agent = request.metadata.as_ref().and_then(|m| m.agent_id.clone());
        ensure!(
            agent.as_ref().is_none_or(|id| id.len() <= 256),
            "agent id exceeds 256 bytes"
        );
        let is_session_final =
            request.metadata.as_ref().and_then(|m| m.session_final) == Some(true);
        let admission = if let Some(session) = request
            .metadata
            .as_ref()
            .and_then(|m| m.session_id.as_ref())
        {
            ensure!(session.len() <= 256, "session id exceeds 256 bytes");
            let identity = (route_id.as_str().to_owned(), session.clone(), agent);
            let mut sessions = self.sessions.lock().unwrap_or_else(|p| p.into_inner());
            let now = Instant::now();
            if !sessions.contains_key(&identity) && sessions.len() >= 4096 {
                sessions.retain(|_, seen| now.duration_since(*seen) < SESSION_TTL);
            }
            if !sessions.contains_key(&identity) && sessions.len() >= 4096 {
                return Err(
                    crate::server::Error::new(503, "session identity capacity exceeded").into(),
                );
            }
            // The SDK can retain session state even when decide fails or is cancelled.
            sessions.insert(identity.clone(), now);
            Some((identity, now))
        } else {
            None
        };
        let mut original_ir = request.llm_request.clone();
        original_ir.model = None;
        let outcome = tokio::time::timeout(Duration::from_secs(1), route.decide(request))
            .await
            .map_err(|_| crate::server::Error::new(504, "SDK routing deadline exceeded"))?
            .map_err(|_| crate::server::Error::new(500, "SDK routing failed"))?;
        if is_session_final && let Some((identity, seen)) = admission {
            let mut sessions = self.sessions.lock().unwrap_or_else(|p| p.into_inner());
            // A newer request for this identity must retain its admission slot.
            if sessions.get(&identity) == Some(&seen) {
                sessions.remove(&identity);
            }
        }
        ensure!(outcome.response.is_none(), "routing-only contract violated");
        let model = outcome.selected_model_id()?.clone();
        let mut selected_ir = outcome.request.llm_request;
        selected_ir.model = None;
        ensure!(
            selected_ir == original_ir,
            "router unexpectedly rewrote request semantics"
        );
        let output = replace_model(body, model.as_str())?;
        let elapsed_us = started.elapsed().as_micros() as u64;
        tracing::info!(
            %model,
            elapsed_us,
            "routing decision"
        );
        Ok((output, model.as_str().to_owned()))
    }
}

fn replace_model(body: &[u8], model: &str) -> Result<Vec<u8>, serde_json::Error> {
    // Preserve original values, including precise numbers and provider-specific fields.
    let mut fields: BTreeMap<String, &RawValue> = serde_json::from_slice(body)?;
    let model = to_raw_value(model)?;
    fields.insert("model".to_owned(), &model);
    serde_json::to_vec(&fields)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn router() -> Router {
        Router::new(Runner::from_toml(include_str!("../config/routes.toml")).unwrap())
    }
    fn neutral() -> Value {
        json!({"model":"auto", "messages":[{"role":"user", "content":"Say hello"}], "stream":true, "temperature":0.2, "custom_vendor_field":{"preserve":true}, "nvext":{"custom":"keep"}})
    }
    fn session(id: &str) -> http::HeaderMap {
        let mut h = http::HeaderMap::new();
        h.insert("x-switchyard-session-id", id.parse().unwrap());
        h
    }

    #[tokio::test]
    async fn session_capacity_recovers_after_idle_expiry() {
        let r = router();
        let quiet = serde_json::to_vec(&neutral()).unwrap();
        {
            let mut sessions = r.sessions.lock().unwrap();
            for i in 0..4096 {
                sessions.insert(("auto".into(), i.to_string(), None), Instant::now());
            }
        }
        let error = r.decide(&quiet, &session("new")).await.unwrap_err();
        assert_eq!(
            error
                .downcast_ref::<crate::server::Error>()
                .unwrap()
                .status_code,
            503
        );
        let mut final_headers = session("0");
        final_headers.insert("x-switchyard-session-final", "true".parse().unwrap());
        r.decide(&quiet, &final_headers).await.unwrap();
        assert_eq!(r.sessions.lock().unwrap().len(), 4095);
        r.decide(&quiet, &session("new")).await.unwrap();
        assert_eq!(r.sessions.lock().unwrap().len(), 4096);
        {
            let mut sessions = r.sessions.lock().unwrap();
            for seen in sessions.values_mut() {
                *seen = Instant::now() - SESSION_TTL;
            }
        }
        r.decide(&quiet, &session("0")).await.unwrap();
        r.decide(&quiet, &session("new")).await.unwrap();
        let sessions = r.sessions.lock().unwrap();
        assert_eq!(sessions.len(), 2);
        assert!(sessions.contains_key(&("auto".into(), "0".into(), None)));
        assert!(sessions.contains_key(&("auto".into(), "new".into(), None)));
    }

    #[tokio::test]
    async fn preserves_original_values_including_precise_numbers_and_nested_fields() {
        let input = br#"{
            "mo\u0064el":"auto",
            "messages":[ {"role":"user", "content":"Hello"} ],
            "large_integer":18446744073709551617,
            "precise_decimal":0.12345678901234567890123456789,
            "custom\u005ffield":{ "nested": [18446744073709551617, {"text":"\u0061"}] }
        }"#;
        let (output, model) = router()
            .decide(input, &http::HeaderMap::new())
            .await
            .unwrap();
        assert_eq!(model, "Qwen/Qwen3-0.6B");
        let original: HashMap<String, &serde_json::value::RawValue> =
            serde_json::from_slice(input).unwrap();
        let rewritten: HashMap<String, &serde_json::value::RawValue> =
            serde_json::from_slice(&output).unwrap();
        assert_eq!(original.len(), rewritten.len());
        for (key, value) in original {
            if key == "model" {
                assert_eq!(rewritten[&key].get(), r#""Qwen/Qwen3-0.6B""#);
            } else {
                assert_eq!(rewritten[&key].get(), value.get(), "changed {key}");
            }
        }
    }
}
