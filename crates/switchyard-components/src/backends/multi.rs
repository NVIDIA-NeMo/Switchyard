// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Multi-target LLM backend dispatch.
//!
//! This backend owns the mechanical part of routing once a processor or caller
//! has selected a target: rewrite the request model, stamp typed context, and
//! delegate to the configured backend for that target. Selection policy stays
//! outside this type; processors and profiles decide which target should run.

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use switchyard_core::{
    ChatRequest, ChatRequestType, ChatResponse, LlmBackend, LlmTarget, LlmTargetId, ProxyContext,
    Result, SwitchyardError,
};

use super::{BackendSelection, BackendSelectionReason};

const DEFAULT_SUPPORTED_REQUEST_TYPES: [ChatRequestType; 4] = [
    ChatRequestType::OpenAiChat,
    ChatRequestType::OpenAiResponses,
    ChatRequestType::Anthropic,
    ChatRequestType::Gemini,
];

/// One configured upstream target and the backend that can call it.
#[derive(Clone)]
pub struct LlmTargetBackend {
    // Target metadata used for model rewriting and public stats.
    target: LlmTarget,
    // Backend implementation that executes calls for the target.
    backend: Arc<dyn LlmBackend>,
}

impl LlmTargetBackend {
    /// Creates a target/backend pair.
    pub fn new(target: LlmTarget, backend: Arc<dyn LlmBackend>) -> Self {
        Self { target, backend }
    }

    /// Returns the target metadata.
    pub fn target(&self) -> &LlmTarget {
        &self.target
    }

    /// Returns the backend configured for the target.
    pub fn backend(&self) -> &dyn LlmBackend {
        self.backend.as_ref()
    }
}

impl fmt::Debug for LlmTargetBackend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LlmTargetBackend")
            .field("target", &self.target)
            .finish_non_exhaustive()
    }
}

/// Backend that delegates each request to one of several configured targets.
#[derive(Clone)]
pub struct MultiLlmBackend {
    // Target/backends are stored in lifecycle order.
    targets: Vec<LlmTargetBackend>,
    // Request formats advertised by this backend.
    supported_request_types: Vec<ChatRequestType>,
    // Deterministic fallback when no router selected a target.
    default_target_id: Option<LlmTargetId>,
}

impl MultiLlmBackend {
    /// Creates a multi-target backend with support for all Switchyard request formats.
    pub fn new(targets: impl IntoIterator<Item = LlmTargetBackend>) -> Result<Self> {
        let targets = targets.into_iter().collect::<Vec<_>>();
        validate_targets(&targets)?;
        Ok(Self {
            targets,
            supported_request_types: DEFAULT_SUPPORTED_REQUEST_TYPES.to_vec(),
            default_target_id: None,
        })
    }

    /// Replaces the advertised request formats.
    pub fn with_supported_request_types(
        mut self,
        request_types: impl IntoIterator<Item = ChatRequestType>,
    ) -> Result<Self> {
        self.supported_request_types = normalize_request_types(request_types)?;
        Ok(self)
    }

    /// Sets the target used when no router selected a target.
    pub fn with_default_target(mut self, target_id: LlmTargetId) -> Result<Self> {
        if self.target(&target_id).is_none() {
            return Err(SwitchyardError::InvalidConfig(format!(
                "default target {target_id} is not configured; known targets: {}",
                self.known_target_ids()
            )));
        }
        self.default_target_id = Some(target_id);
        Ok(self)
    }

    /// Returns the deterministic default target, when configured.
    pub fn default_target_id(&self) -> Option<&LlmTargetId> {
        self.default_target_id.as_ref()
    }

    /// Returns configured target/backend pairs in lifecycle order.
    pub fn targets(&self) -> &[LlmTargetBackend] {
        &self.targets
    }

    /// Looks up a configured target by ID.
    pub fn target(&self, target_id: &LlmTargetId) -> Option<&LlmTargetBackend> {
        self.targets
            .iter()
            .find(|entry| &entry.target.id == target_id)
    }

    fn selected_target<'a>(
        &'a self,
        ctx: &ProxyContext,
        request: &ChatRequest,
    ) -> Result<(&'a LlmTargetBackend, BackendSelectionReason)> {
        // Explicit context selection wins because request processors are the
        // routing policy layer.
        if let Some(target_id) = ctx.selected_target() {
            let Some(target) = self.target(target_id) else {
                return Err(SwitchyardError::InvalidConfig(format!(
                    "selected target {target_id} is not configured; known targets: {}",
                    self.known_target_ids()
                )));
            };
            return Ok((target, BackendSelectionReason::ContextTarget));
        }

        // A configured default only applies when no processor selected a target.
        if let Some(target_id) = &self.default_target_id {
            let Some(target) = self.target(target_id) else {
                return Err(SwitchyardError::InvalidConfig(format!(
                    "default target {target_id} is not configured; known targets: {}",
                    self.known_target_ids()
                )));
            };
            return Ok((target, BackendSelectionReason::DefaultTarget));
        }

        // A single-target backend is unambiguous even without a router.
        if self.targets.len() == 1 {
            let Some(target) = self.targets.first() else {
                return Err(SwitchyardError::InvalidConfig(
                    "MultiLlmBackend requires at least one target".to_string(),
                ));
            };
            return Ok((target, BackendSelectionReason::SingleTarget));
        }

        // As a final convenience, match the request model to a unique target
        // model. Duplicate matches remain ambiguous.
        self.target_for_request_model(request)
    }

    /// Selects a target by matching the request model to configured target models.
    fn target_for_request_model<'a>(
        &'a self,
        request: &ChatRequest,
    ) -> Result<(&'a LlmTargetBackend, BackendSelectionReason)> {
        let Some(model) = request.model() else {
            return Err(self.missing_selection_error(None));
        };

        let matches = self
            .targets
            .iter()
            .filter(|entry| entry.target.model.as_str() == model)
            .collect::<Vec<_>>();

        match matches.as_slice() {
            [target] => Ok((*target, BackendSelectionReason::RequestModel)),
            [] => Err(self.missing_selection_error(Some(model))),
            _ => Err(SwitchyardError::InvalidConfig(format!(
                "request model {model:?} matches multiple targets; set selected_target explicitly"
            ))),
        }
    }

    /// Builds the error used when no routing signal identifies one target.
    fn missing_selection_error(&self, request_model: Option<&str>) -> SwitchyardError {
        let request_model = request_model
            .map(|model| format!(" and request model {model:?} did not match a configured target"))
            .unwrap_or_default();
        SwitchyardError::InvalidConfig(format!(
                "MultiLlmBackend has multiple targets but no selected target{request_model}; known targets: {}",
            self.known_target_ids()
        ))
    }

    /// Formats configured target IDs for diagnostics.
    fn known_target_ids(&self) -> String {
        self.targets
            .iter()
            .map(|entry| entry.target.id.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

impl fmt::Debug for MultiLlmBackend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MultiLlmBackend")
            .field(
                "targets",
                &self
                    .targets
                    .iter()
                    .map(|entry| &entry.target)
                    .collect::<Vec<_>>(),
            )
            .field("supported_request_types", &self.supported_request_types)
            .field("default_target_id", &self.default_target_id)
            .finish()
    }
}

#[async_trait]
impl LlmBackend for MultiLlmBackend {
    // Returns request formats this backend accepts before delegation.
    fn supported_request_types(&self) -> &[ChatRequestType] {
        &self.supported_request_types
    }

    // Rewrites the request model to the selected target and delegates once.
    async fn call(&self, ctx: &mut ProxyContext, request: &ChatRequest) -> Result<ChatResponse> {
        let request_type = request.request_type();
        if !self.supported_request_types.contains(&request_type) {
            return Err(SwitchyardError::UnsupportedRequestType {
                component: "MultiLlmBackend".to_string(),
                request_type,
            });
        }

        let (target, reason) = self.selected_target(ctx, request)?;
        let mut routed_request = request.clone();
        routed_request.set_model(target.target.model.as_str());

        // Stamping BackendSelection lets stats processors attribute the final
        // provider call without coupling to this backend's internals.
        let _ = ctx.insert(BackendSelection::for_target(
            target.target.id.clone(),
            target.target.model.clone(),
            request.model().map(str::to_string),
            reason,
        ));

        target.backend.call(ctx, &routed_request).await
    }

    // Starts child backends in configured order and rolls back on failure.
    async fn startup(&self) -> Result<()> {
        let mut started: Vec<&LlmTargetBackend> = Vec::new();
        for target in &self.targets {
            if let Err(error) = target.backend.startup().await {
                for started_target in started.into_iter().rev() {
                    let _ = started_target.backend.shutdown().await;
                }
                return Err(error);
            }
            started.push(target);
        }
        Ok(())
    }

    // Shuts child backends down in reverse order while preserving first error.
    async fn shutdown(&self) -> Result<()> {
        let mut first_error = None;
        for target in self.targets.iter().rev() {
            if let Err(error) = target.backend.shutdown().await {
                first_error.get_or_insert(error);
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

/// Validates constructor invariants for target lists.
fn validate_targets(targets: &[LlmTargetBackend]) -> Result<()> {
    if targets.is_empty() {
        return Err(SwitchyardError::InvalidConfig(
            "MultiLlmBackend requires at least one target".to_string(),
        ));
    }

    let mut seen = HashSet::new();
    for target in targets {
        if !seen.insert(target.target.id.clone()) {
            return Err(SwitchyardError::InvalidConfig(format!(
                "duplicate LLM target id: {}",
                target.target.id
            )));
        }
    }
    Ok(())
}

/// Deduplicates request types while preserving caller order.
fn normalize_request_types(
    request_types: impl IntoIterator<Item = ChatRequestType>,
) -> Result<Vec<ChatRequestType>> {
    let mut normalized = Vec::new();
    for request_type in request_types {
        if !normalized.contains(&request_type) {
            normalized.push(request_type);
        }
    }
    if normalized.is_empty() {
        return Err(SwitchyardError::InvalidConfig(
            "MultiLlmBackend must support at least one request type".to_string(),
        ));
    }
    Ok(normalized)
}
