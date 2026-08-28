// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Named route table and server-facing route metadata.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use libsy::RoutingOutcome;
use reqwest::{Body, Method, Response};
use serde_json::Value;
use switchyard_llm_client::{Backend, TranslatingLlmClient};
use switchyard_protocol::{Metadata, ModelId, WireFormat};

use crate::config;
use crate::{ModelCapabilities, Route, RunnerError};

/// Immutable named route table.
pub struct Runner {
    routes: Vec<(ModelId, Route)>,
    fallback_client: Option<FallbackClient>,
}

pub(crate) struct FallbackClient {
    pub backend: Backend,
    pub client: Arc<TranslatingLlmClient>,
}

/// Borrowed model metadata returned while listing routes.
pub struct ModelInfo<'a> {
    pub id: &'a ModelId,
    pub algorithm: &'a str,
    pub capabilities: ModelCapabilities,
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
            fallback_client: None,
        }
    }

    pub(crate) fn with_fallback_client(mut self, fallback_client: Option<FallbackClient>) -> Self {
        self.fallback_client = fallback_client;
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
        })
    }

    /// Forwards an unmatched request through the configured fallback client.
    pub async fn forward_fallback(
        &self,
        method: Method,
        path_and_query: &str,
        body: Body,
        metadata: Metadata,
    ) -> Result<Option<Response>, RunnerError> {
        let Some(fallback) = &self.fallback_client else {
            return Ok(None);
        };
        fallback
            .client
            .forward(
                &fallback.backend,
                method,
                path_and_query,
                body,
                Some(&metadata),
            )
            .await
            .map(Some)
            .map_err(Into::into)
    }

    /// Resolves an outcome to configured target names and non-secret client settings.
    pub fn describe_decision(
        &self,
        model: &ModelId,
        outcome: &RoutingOutcome,
    ) -> Option<DecisionDescription> {
        let route = self.route(model.as_str())?;
        let resolve = |selected: &ModelId| route.decision_target(selected);
        Some(DecisionDescription {
            selected: resolve(&outcome.selected_model_id)?,
            fallbacks: outcome
                .fallback_models
                .iter()
                .map(resolve)
                .collect::<Option<Vec<_>>>()?,
        })
    }
}
