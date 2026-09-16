// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Named route table and server-facing route metadata.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use libsy::RoutingOutcome;
use serde_json::Value;
use switchyard_llm_client::TranslatingLlmClient;
use switchyard_protocol::{ModelId, WireFormat};

use crate::config;
use crate::{ModelCapabilities, Route, RunnerError};

/// Immutable named route table.
pub struct Runner {
    routes: Vec<(ModelId, Route)>,
    fallback_base_url: Option<String>,
    provider_api_keys: Vec<String>,
    model_properties_probes: BTreeMap<ModelId, ModelPropertiesProbe>,
}

/// Borrowed model metadata returned while listing routes.
pub struct ModelInfo<'a> {
    /// Synthetic route model identifier.
    pub id: &'a ModelId,
    /// Algorithm implementing the route.
    pub algorithm: &'a str,
    /// Statically declared route capabilities.
    pub capabilities: ModelCapabilities,
    /// Completion targets in algorithm routing order.
    pub routing_targets: &'a [DecisionTarget],
}

/// Fully resolved routing decision.
pub struct DecisionDescription {
    pub selected: DecisionTarget,
    pub fallbacks: Vec<DecisionTarget>,
}

/// Non-secret configured target details returned by the decision endpoint.
#[derive(Clone)]
pub struct DecisionTarget {
    pub target: String,
    pub model: ModelId,
    pub format: WireFormat,
    pub base_url: String,
    pub extra_body: BTreeMap<String, Value>,
}

pub(crate) struct ModelPropertiesProbe {
    model: ModelId,
    format: WireFormat,
    client: Arc<TranslatingLlmClient>,
}

impl ModelPropertiesProbe {
    pub(crate) fn new(
        model: ModelId,
        format: WireFormat,
        client: Arc<TranslatingLlmClient>,
    ) -> Self {
        Self {
            model,
            format,
            client,
        }
    }

    async fn get(&self, timeout: Duration) -> Option<Value> {
        self.client
            .get_model_properties(&self.model, self.format, timeout)
            .await
            .ok()
    }
}

impl Runner {
    /// Loads and validates a version-1 deployment TOML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, RunnerError> {
        config::load_runner(path)
    }

    /// Loads and validates a version-1 deployment TOML document.
    ///
    /// Use [`Self::load`] when the deployment is stored in a file.
    pub fn from_toml(source: &str) -> Result<Self, RunnerError> {
        config::runner_from_toml(source)
    }

    /// Builds a runner from named routes in caller-provided order.
    /// Pre-condition: There must be at least one route.
    pub fn new(routes: Vec<(ModelId, Route)>) -> Self {
        Self {
            routes,
            fallback_base_url: None,
            provider_api_keys: Vec::new(),
            model_properties_probes: BTreeMap::new(),
        }
    }

    /// Registers deployment-owned API keys for server response redaction.
    /// TOML loading registers these automatically; programmatic hosts must supply them.
    pub fn with_provider_api_keys(mut self, keys: Vec<String>) -> Self {
        self.provider_api_keys = keys;
        self
    }

    /// Returns deployment-owned secrets for the server's response redactor.
    pub fn provider_api_keys(&self) -> &[String] {
        &self.provider_api_keys
    }

    pub(crate) fn with_fallback_url(mut self, fallback_base_url: Option<String>) -> Self {
        self.fallback_base_url = fallback_base_url;
        self
    }

    pub(crate) fn with_model_properties_probes(
        mut self,
        probes: BTreeMap<ModelId, ModelPropertiesProbe>,
    ) -> Self {
        self.model_properties_probes = probes;
        self
    }

    /// Returns the route registered for a model.
    pub fn route(&self, model: &str) -> Option<&Route> {
        self.routes
            .iter()
            .find(|(id, _)| id.as_str() == model)
            .map(|(_, route)| route)
    }

    /// Iterates over configured routes in caller-provided order.
    pub fn models(&self) -> impl Iterator<Item = ModelInfo<'_>> {
        self.routes.iter().map(|(id, route)| ModelInfo {
            id,
            algorithm: route.algorithm_name(),
            capabilities: route.capabilities(),
            routing_targets: route.decision_targets(),
        })
    }

    /// Reads one route's configured local backend properties without exposing its credentials.
    pub async fn model_properties(&self, model: &str, timeout: Duration) -> Option<Value> {
        self.model_properties_probes.get(model)?.get(timeout).await
    }

    /// Returns the validated API root used for unmatched HTTP requests.
    pub fn fallback_base_url(&self) -> Option<&str> {
        self.fallback_base_url.as_deref()
    }

    /// Resolves an outcome to configured target names and non-secret client settings.
    pub fn describe_decision(
        &self,
        model: &ModelId,
        outcome: &RoutingOutcome,
    ) -> Option<DecisionDescription> {
        let route = self.route(model.as_str())?;
        let resolve = |selected: &ModelId| route.decision_target(selected);
        let mut model_ids = outcome.selected_model_ids.iter();
        Some(DecisionDescription {
            selected: resolve(model_ids.next()?)?,
            fallbacks: model_ids.map(resolve).collect::<Option<Vec<_>>>()?,
        })
    }
}
