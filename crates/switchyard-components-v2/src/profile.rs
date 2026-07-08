// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Profile runtime contracts for components-v2.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use switchyard_core::{ChatRequest, ChatRequestType, ChatResponse, RequestId, Result};

use crate::decision::{DecisionContext, RoutingDecision};

/// Legacy proxy session header used by Switchyard Python endpoints.
pub const PROXY_SESSION_ID_HEADER: &str = "proxy_x_session_id";
/// Canonical Relay session header accepted alongside the proxy alias.
pub const RELAY_SESSION_ID_HEADER: &str = "x-nemo-relay-session-id";

/// Explicit per-request metadata passed to v2 profiles.
///
/// Header keys are stored as lowercase ASCII names. Repeated header values are
/// preserved in the original value order because endpoints may need to pass
/// through multi-value headers without silently collapsing them.
#[derive(Clone, Default, Eq, PartialEq)]
pub struct RequestMetadata {
    /// Canonical request/session identity resolved from endpoint headers or a Decision body.
    pub session_id: Option<String>,
    /// Caller-provided request identifier, when present.
    pub request_id: Option<RequestId>,
    /// Inbound wire format recorded by the endpoint, when present.
    pub inbound_format: Option<ChatRequestType>,
    /// Request headers supplied by the caller, keyed by normalized header name.
    pub headers: BTreeMap<String, Vec<String>>,
}

/// Resolves all values from both supported session-header aliases.
///
/// Values are trimmed before comparison. Empty values and any disagreement
/// across aliases or repeated values are rejected instead of being ignored.
pub fn session_id_from_normalized_headers(
    headers: &BTreeMap<String, Vec<String>>,
) -> Result<Option<String>> {
    let mut resolved = None;
    for name in [PROXY_SESSION_ID_HEADER, RELAY_SESSION_ID_HEADER] {
        let Some(values) = headers.get(name) else {
            continue;
        };
        if values.is_empty() {
            return Err(switchyard_core::SwitchyardError::InvalidRequest(format!(
                "{name} must contain at least one non-empty session ID"
            )));
        }
        for value in values {
            merge_session_id(&mut resolved, value, name)?;
        }
    }
    Ok(resolved)
}

/// Reconciles an explicit session ID with both normalized header aliases.
pub fn reconcile_session_id(
    explicit: Option<&str>,
    headers: &BTreeMap<String, Vec<String>>,
) -> Result<Option<String>> {
    let mut resolved = None;
    if let Some(explicit) = explicit {
        merge_session_id(&mut resolved, explicit, "session_id")?;
    }
    if let Some(header_session_id) = session_id_from_normalized_headers(headers)? {
        merge_session_id(&mut resolved, &header_session_id, "session header aliases")?;
    }
    Ok(resolved)
}

fn merge_session_id(resolved: &mut Option<String>, raw: &str, source: &str) -> Result<()> {
    let value = raw.trim();
    if value.is_empty() {
        return Err(switchyard_core::SwitchyardError::InvalidRequest(format!(
            "{source} must contain a non-empty session ID"
        )));
    }
    if let Some(existing) = resolved {
        if existing != value {
            return Err(switchyard_core::SwitchyardError::InvalidRequest(format!(
                "conflicting session IDs {existing:?} and {value:?}"
            )));
        }
        return Ok(());
    }
    *resolved = Some(value.to_string());
    Ok(())
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
    async fn decide(&self, context: DecisionContext) -> Result<RoutingDecision> {
        context.request().validate()?;
        Err(switchyard_core::SwitchyardError::DecisionUnsupported {
            profile_id: context.request().decision_profile.profile_id.clone(),
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
            session_id: Some("session-1".to_string()),
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
        assert_eq!(metadata.session_id.as_deref(), Some("session-1"));
    }

    #[test]
    fn profile_input_is_plain_data() {
        let input = ProfileInput {
            request: ChatRequest::openai_chat(json!({
                "model": "client/model",
                "messages": [],
            })),
            metadata: RequestMetadata {
                session_id: Some("session-1".to_string()),
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

    #[test]
    fn session_header_aliases_and_repeated_values_reconcile() -> Result<()> {
        let headers = BTreeMap::from([
            (
                PROXY_SESSION_ID_HEADER.to_string(),
                vec![" session-1 ".to_string(), "session-1".to_string()],
            ),
            (
                RELAY_SESSION_ID_HEADER.to_string(),
                vec!["session-1".to_string()],
            ),
        ]);

        assert_eq!(
            session_id_from_normalized_headers(&headers)?.as_deref(),
            Some("session-1")
        );
        assert_eq!(
            reconcile_session_id(Some(" session-1 "), &headers)?.as_deref(),
            Some("session-1")
        );
        Ok(())
    }

    #[test]
    fn session_header_aliases_reject_conflicts_and_empty_values() {
        for headers in [
            BTreeMap::from([(
                PROXY_SESSION_ID_HEADER.to_string(),
                vec!["session-1".to_string(), "session-2".to_string()],
            )]),
            BTreeMap::from([
                (
                    PROXY_SESSION_ID_HEADER.to_string(),
                    vec!["session-1".to_string()],
                ),
                (
                    RELAY_SESSION_ID_HEADER.to_string(),
                    vec!["session-2".to_string()],
                ),
            ]),
            BTreeMap::from([(RELAY_SESSION_ID_HEADER.to_string(), vec!["   ".to_string()])]),
            BTreeMap::from([(PROXY_SESSION_ID_HEADER.to_string(), Vec::new())]),
        ] {
            assert!(session_id_from_normalized_headers(&headers).is_err());
        }
    }
}
