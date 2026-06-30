// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Profile runtime contracts for components-v2.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use switchyard_core::{ChatRequest, ChatRequestType, ChatResponse, RequestId, Result};

use crate::decision::{RoutingDecision, RoutingRequest};

/// Explicit per-request metadata passed to v2 profiles.
///
/// Header keys are stored as lowercase ASCII names. Repeated header values are
/// preserved in the original value order because endpoints may need to pass
/// through multi-value headers without silently collapsing them.
#[derive(Clone, Default, Eq, PartialEq)]
pub struct RequestMetadata {
    /// Caller-provided request identifier, when present.
    pub request_id: Option<RequestId>,
    /// Inbound wire format recorded by the endpoint, when present.
    pub inbound_format: Option<ChatRequestType>,
    /// Request headers supplied by the caller, keyed by normalized header name.
    pub headers: BTreeMap<String, Vec<String>>,
}

/// Input object handed to v2 profiles.
///
/// This is the only request carrier in the v2 profile path. It owns the
/// provider-neutral [`ChatRequest`] plus endpoint-supplied metadata such as
/// request IDs, inbound format, and headers. Policy decisions remain local to
/// the profile implementation instead of being hidden in this input object.
#[derive(Clone, PartialEq)]
pub struct ProfileInput {
    /// Provider-neutral chat request entering the profile.
    pub request: ChatRequest,
    /// Endpoint-supplied request metadata.
    pub metadata: RequestMetadata,
}

/// Routing decision metadata emitted by profiles that select among targets.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RoutingMetadata {
    /// Upstream model selected by the router.
    pub selected_model: Option<String>,
    /// Routing tier selected by the router.
    pub selected_tier: Option<String>,
    /// Router confidence for the selected decision.
    pub confidence: Option<f64>,
    /// Stable router implementation/version label.
    pub router_version: Option<String>,
    /// Router tolerance, probability, or decision threshold.
    pub tolerance: Option<f64>,
    /// Short human-readable route rationale.
    pub rationale: Option<String>,
}

impl RoutingMetadata {
    /// Returns true when no metadata field is set.
    pub fn is_empty(&self) -> bool {
        self.selected_model.is_none()
            && self.selected_tier.is_none()
            && self.confidence.is_none()
            && self.router_version.is_none()
            && self.tolerance.is_none()
            && self.rationale.is_none()
    }
}

/// Full profile result returned to servers and other erased-profile callers.
pub struct ProfileResponse {
    /// Final backend response after profile response processing.
    pub response: ChatResponse,
    /// Optional routing metadata for response headers and audits.
    pub routing_metadata: Option<RoutingMetadata>,
}

impl ProfileResponse {
    /// Creates a profile response without routing metadata.
    pub fn new(response: ChatResponse) -> Self {
        Self {
            response,
            routing_metadata: None,
        }
    }

    /// Creates a profile response with routing metadata.
    pub fn with_routing_metadata(
        response: ChatResponse,
        routing_metadata: RoutingMetadata,
    ) -> Self {
        Self {
            response,
            routing_metadata: (!routing_metadata.is_empty()).then_some(routing_metadata),
        }
    }

    /// Splits this value into the final response and optional routing metadata.
    pub fn into_parts(self) -> (ChatResponse, Option<RoutingMetadata>) {
        (self.response, self.routing_metadata)
    }

    /// Returns the response body when this is a buffered response.
    pub fn body(&self) -> Option<&serde_json::Value> {
        self.response.body()
    }
}

impl From<ChatResponse> for ProfileResponse {
    fn from(response: ChatResponse) -> Self {
        Self::new(response)
    }
}

/// Object-safe runtime surface for a complete profile.
///
/// Servers and config-built profile maps use this trait when they need to run
/// an entire profile or request a decision without selected-backend dispatch.
/// Hook-level processed state remains on [`ProfileHooks`] so dynamic dispatch
/// does not erase profile-specific authoring APIs.
#[async_trait]
pub trait Profile: Send + Sync {
    /// Executes the complete profile flow and returns the final response.
    async fn run(&self, input: ProfileInput) -> Result<ProfileResponse>;

    /// Returns a routing decision without dispatching the selected backend.
    ///
    /// Profiles that do not implement decision-only routing return a typed
    /// unsupported-profile error by default.
    async fn decide(&self, request: RoutingRequest) -> Result<RoutingDecision> {
        request.validate()?;
        Err(switchyard_core::SwitchyardError::DecisionUnsupported {
            profile_id: request.decision_profile.profile_id,
        })
    }
}

/// Typed hook surface for profile authors and embedders.
///
/// These methods support middleware-style integrations that want Switchyard to
/// prepare a request and/or process a response without owning the transport.
/// The associated `ProcessedRequest` type is a profile-owned struct, which lets
/// profiles expose real per-call state, such as a routing decision, without a
/// generic side channel.
#[async_trait]
pub trait ProfileHooks: Send + Sync {
    /// Profile-owned request-side state.
    type ProcessedRequest: Send + Sync;

    /// Runs the profile's request-side hook.
    async fn process(&self, input: ProfileInput) -> Result<Self::ProcessedRequest>;

    /// Runs the profile's response-side hook after a backend response exists.
    async fn rprocess(
        &self,
        processed: &Self::ProcessedRequest,
        response: ChatResponse,
    ) -> Result<ChatResponse>;
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::json;
    use switchyard_core::ChatRequestType;

    use super::*;

    #[test]
    fn request_metadata_is_plain_data() {
        let metadata = RequestMetadata {
            request_id: None,
            inbound_format: Some(ChatRequestType::OpenAiChat),
            headers: BTreeMap::from([(
                "x-switchyard-trace".to_string(),
                vec!["abc123".to_string()],
            )]),
        };

        assert_eq!(
            metadata
                .headers
                .get("x-switchyard-trace")
                .map(Vec::as_slice),
            Some(&["abc123".to_string()][..])
        );
        assert_eq!(metadata.inbound_format, Some(ChatRequestType::OpenAiChat));
    }

    #[test]
    fn profile_input_is_plain_data() {
        let input = ProfileInput {
            request: ChatRequest::openai_chat(json!({
                "model": "client/model",
                "messages": [],
            })),
            metadata: RequestMetadata {
                request_id: None,
                inbound_format: None,
                headers: BTreeMap::from([(
                    "x-request-source".to_string(),
                    vec!["unit-test".to_string()],
                )]),
            },
        };

        assert_eq!(input.request.model(), Some("client/model"));
        assert_eq!(
            input
                .metadata
                .headers
                .get("x-request-source")
                .map(Vec::as_slice),
            Some(&["unit-test".to_string()][..])
        );
        assert!(input.metadata.request_id.is_none());
    }
}
