// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! JSON-first routing decision contract for Relay integration.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use switchyard_core::{
    BackendFormat, ChatRequest, ChatRequestType, ProfileId, RequestId, Result, SwitchyardError,
};

use crate::{ProfileInput, RequestMetadata};

/// Schema identifier for routing requests.
pub const ROUTING_REQUEST_SCHEMA_VERSION: &str = "switchyard.routing_request.v1";

/// Schema identifier for routing decisions.
pub const ROUTING_DECISION_SCHEMA_VERSION: &str = "switchyard.routing_decision.v1";

/// Relay request-time routing request.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RoutingRequest {
    /// Versioned request schema identifier.
    pub schema_version: String,
    /// Concrete Switchyard profile selection and materialization policy.
    pub decision_profile: DecisionProfile,
    /// Normalized request identity.
    pub identity: RequestIdentity,
    /// Inbound protocol details.
    pub protocol: RequestProtocol,
    /// Cheap request summary.
    pub request_summary: RequestSummary,
    /// Optional current-request materialization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_request: Option<Value>,
    /// Routing attempt metadata.
    pub attempt: DecisionAttempt,
}

impl RoutingRequest {
    /// Validates envelope invariants shared by every decision-capable profile.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != ROUTING_REQUEST_SCHEMA_VERSION {
            return Err(SwitchyardError::InvalidRequest(format!(
                "unsupported routing request schema_version {:?}; expected {ROUTING_REQUEST_SCHEMA_VERSION:?}",
                self.schema_version
            )));
        }
        if self.attempt.routing_attempt == 0 {
            return Err(SwitchyardError::InvalidRequest(
                "routing_attempt must be at least 1".to_string(),
            ));
        }
        if self.attempt.max_routing_attempts == 0
            || self.attempt.routing_attempt > self.attempt.max_routing_attempts
        {
            return Err(SwitchyardError::InvalidRequest(
                "routing_attempt must not exceed a non-zero max_routing_attempts".to_string(),
            ));
        }
        require_non_blank(
            "decision_profile.profile_id",
            self.decision_profile.profile_id.as_str(),
        )?;
        require_non_blank("identity.session_id", &self.identity.session_id)?;
        require_non_blank("identity.request_id", &self.identity.request_id)?;
        require_non_blank("identity.harness", &self.identity.harness)?;
        require_non_blank("identity.source", &self.identity.source)?;
        for (field, value) in [
            ("protocol.inbound_profile", &self.protocol.inbound_profile),
            ("protocol.inbound_endpoint", &self.protocol.inbound_endpoint),
            (
                "protocol.desired_response_profile",
                &self.protocol.desired_response_profile,
            ),
        ] {
            require_non_blank(field, value)?;
        }
        if let Some(owner_id) = self.identity.owner_id.as_deref() {
            require_non_blank("identity.owner_id", owner_id)?;
        }
        validate_current_request_materialization(self)?;
        parse_inbound_profile(&self.protocol.inbound_profile).map(|_| ())
    }
}

/// Concrete profile selection and request materialization policy.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DecisionProfile {
    /// Required Switchyard profile ID from the loaded configuration.
    pub profile_id: ProfileId,
    /// Current-request materialization mode supplied by Relay.
    pub request_materialization: CurrentRequestMaterialization,
}

/// Current-request materialization modes.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CurrentRequestMaterialization {
    /// No request material is included.
    None,
    /// Only `request_summary` is included.
    SummaryOnly,
    /// Includes the latest user prompt.
    LatestUserPrompt,
    /// Includes a bounded recent message window.
    RecentMessageWindow,
    /// Includes annotated or canonicalized request material.
    AnnotatedRequest,
    /// Includes the full inbound request body.
    FullBody,
}

/// Normalized identity passed by Relay.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RequestIdentity {
    /// Stable session identifier.
    pub session_id: String,
    /// Per-request identifier.
    pub request_id: String,
    /// Optional turn identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    /// Optional parent scope identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_scope_id: Option<String>,
    /// Optional root scope identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root_scope_id: Option<String>,
    /// Harness or agent source.
    pub harness: String,
    /// Request source.
    pub source: String,
    /// Optional resolved owner, such as a subagent or root work owner.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner_id: Option<String>,
    /// Identity quality marker for native versus synthesized fields.
    #[serde(default)]
    pub quality: IdentityQuality,
}

/// Identity quality marker.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IdentityQuality {
    /// All required identity fields came from native request/session data.
    Native,
    /// One or more required fields were synthesized by Relay.
    #[default]
    Synthetic,
    /// Identity was supplied explicitly by Relay/plugin config.
    Explicit,
}

/// Inbound protocol metadata.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RequestProtocol {
    /// Inbound wire profile string.
    pub inbound_profile: String,
    /// Inbound endpoint path.
    pub inbound_endpoint: String,
    /// Desired response profile for the client.
    pub desired_response_profile: String,
}

/// Cheap request summary produced by Relay.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct RequestSummary {
    /// Model requested by the client.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_requested_model: Option<String>,
    /// Estimated prompt token count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_token_estimate: Option<u64>,
    /// Number of tools in the payload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_count_in_payload: Option<u64>,
    /// Whether the request includes a system prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub has_system_prompt: Option<bool>,
}

/// Routing attempt metadata.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DecisionAttempt {
    /// 1-indexed routing attempt.
    pub routing_attempt: u32,
    /// Maximum routing attempts.
    pub max_routing_attempts: u32,
}

/// Switchyard routing decision returned to Relay.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RoutingDecision {
    /// Versioned decision schema identifier.
    pub schema_version: String,
    /// Stable per-decision identifier.
    pub decision_id: String,
    /// Router implementation metadata.
    pub router: DecisionProvider,
    /// Target route selected by Switchyard.
    pub route: RoutingTarget,
    /// Optional confidence score.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    /// Optional reason code.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
    /// Optional human-readable summary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason_summary: Option<String>,
    /// Additional router metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

/// Router implementation metadata in a decision response.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DecisionProvider {
    /// Router implementation name derived from the configured profile type.
    pub name: String,
    /// Router implementation version derived from the profile runtime.
    pub version: String,
}

/// Selected Switchyard route.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RoutingTarget {
    /// Tier label such as `strong` or `weak`.
    pub tier: String,
    /// Target model to send upstream.
    pub target_model: String,
    /// Switchyard target ID.
    pub backend_id: String,
    /// Switchyard wire protocol profile.
    pub target_protocol_profile: String,
    /// Endpoint selected for the target wire protocol.
    pub target_endpoint: String,
}

/// Maps a Switchyard backend format to the Decision API wire protocol label.
pub fn route_protocol_for_format(format: BackendFormat) -> Result<&'static str> {
    match format {
        BackendFormat::OpenAi => Ok("openai_chat"),
        BackendFormat::Responses => Ok("openai_responses"),
        BackendFormat::Anthropic => Ok("anthropic_messages"),
        BackendFormat::Auto => Err(SwitchyardError::InvalidConfig(
            "cannot produce routing decision for unresolved auto backend format".to_string(),
        )),
    }
}

/// Maps a Switchyard backend format to the upstream endpoint path.
pub fn route_endpoint_for_format(format: BackendFormat) -> Result<&'static str> {
    match format {
        BackendFormat::OpenAi => Ok("/v1/chat/completions"),
        BackendFormat::Responses => Ok("/v1/responses"),
        BackendFormat::Anthropic => Ok("/v1/messages"),
        BackendFormat::Auto => Err(SwitchyardError::InvalidConfig(
            "cannot produce routing decision for unresolved auto backend format".to_string(),
        )),
    }
}

// Builds the request-only state needed by policies that can route from summaries.
pub(crate) fn summary_profile_input(request: &RoutingRequest) -> Result<ProfileInput> {
    let mut body = serde_json::Map::new();
    if let Some(model) = &request.request_summary.client_requested_model {
        body.insert("model".to_string(), Value::String(model.clone()));
    }
    profile_input(request, Value::Object(body))
}

// Requires concrete prompt content before a classifier can spend an upstream call.
pub(crate) fn materialized_profile_input(request: &RoutingRequest) -> Result<ProfileInput> {
    validate_current_request_materialization(request)?;
    if matches!(
        request.decision_profile.request_materialization,
        CurrentRequestMaterialization::None | CurrentRequestMaterialization::SummaryOnly
    ) {
        return Err(SwitchyardError::InvalidRequest(
            "llm-routing requires a materialized current_request.body".to_string(),
        ));
    }
    let body = request
        .current_request
        .as_ref()
        .and_then(|current| current.get("body"))
        .cloned()
        .ok_or_else(|| {
            SwitchyardError::InvalidRequest(
                "llm-routing requires current_request.body for prompt classification".to_string(),
            )
        })?;
    if !body.is_object() {
        return Err(SwitchyardError::InvalidRequest(
            "current_request.body must be a JSON object".to_string(),
        ));
    }
    let request_type = parse_inbound_profile(&request.protocol.inbound_profile)?;
    if !has_materialized_prompt(&body, request_type) {
        return Err(SwitchyardError::InvalidRequest(
            "llm-routing current_request.body must contain non-empty prompt material".to_string(),
        ));
    }
    profile_input(request, body)
}

fn profile_input(request: &RoutingRequest, body: Value) -> Result<ProfileInput> {
    let request_type = parse_inbound_profile(&request.protocol.inbound_profile)?;
    let chat_request = chat_request_for_type(request_type, body);
    chat_request.validate()?;
    Ok(ProfileInput {
        request: chat_request,
        metadata: RequestMetadata {
            request_id: Some(RequestId::new(request.identity.request_id.clone())?),
            inbound_format: Some(request_type),
            ..RequestMetadata::default()
        },
    })
}

// Keeps the wire field extensible as a string while accepting only documented aliases.
fn parse_inbound_profile(profile: &str) -> Result<ChatRequestType> {
    match profile {
        "openai_chat" | "openai_chat_completions" | "openai_chat_completions.v1" => {
            Ok(ChatRequestType::OpenAiChat)
        }
        "openai_responses" | "openai_responses.v1" => Ok(ChatRequestType::OpenAiResponses),
        "anthropic" | "anthropic_messages" | "anthropic_messages.v1" => {
            Ok(ChatRequestType::Anthropic)
        }
        unsupported => Err(SwitchyardError::InvalidRequest(format!(
            "unsupported inbound_profile {unsupported:?}; expected a documented OpenAI Chat, OpenAI Responses, or Anthropic Messages profile"
        ))),
    }
}

fn chat_request_for_type(request_type: ChatRequestType, body: Value) -> ChatRequest {
    match request_type {
        ChatRequestType::OpenAiChat => ChatRequest::openai_chat(body),
        ChatRequestType::OpenAiResponses => ChatRequest::openai_responses(body),
        ChatRequestType::Anthropic => ChatRequest::anthropic(body),
    }
}

fn has_materialized_prompt(body: &Value, request_type: ChatRequestType) -> bool {
    match request_type {
        ChatRequestType::OpenAiChat | ChatRequestType::Anthropic => body
            .get("messages")
            .and_then(Value::as_array)
            .is_some_and(|messages| messages.iter().any(message_has_user_prompt)),
        ChatRequestType::OpenAiResponses => body
            .get("input")
            .is_some_and(responses_input_has_user_prompt),
    }
}

fn message_has_user_prompt(message: &Value) -> bool {
    message.get("role").and_then(Value::as_str) == Some("user")
        && message
            .get("content")
            .is_some_and(content_has_prompt_material)
}

fn responses_input_has_user_prompt(input: &Value) -> bool {
    match input {
        Value::String(text) => has_non_whitespace(text),
        Value::Array(items) => items.iter().any(responses_item_has_user_prompt),
        _ => false,
    }
}

fn responses_item_has_user_prompt(item: &Value) -> bool {
    match item {
        Value::String(text) => has_non_whitespace(text),
        Value::Object(object) => match object.get("role").and_then(Value::as_str) {
            Some("user") => object
                .get("content")
                .is_some_and(content_has_prompt_material),
            Some(_) => false,
            None => {
                matches!(
                    object.get("type").and_then(Value::as_str),
                    Some("input_text" | "input_image" | "input_file")
                ) && content_part_has_prompt_material(item)
            }
        },
        _ => false,
    }
}

fn content_has_prompt_material(content: &Value) -> bool {
    match content {
        Value::String(text) => has_non_whitespace(text),
        Value::Array(parts) => parts.iter().any(content_part_has_prompt_material),
        Value::Object(_) => content_part_has_prompt_material(content),
        _ => false,
    }
}

fn content_part_has_prompt_material(part: &Value) -> bool {
    let Some(object) = part.as_object() else {
        return part.as_str().is_some_and(has_non_whitespace);
    };
    if object
        .get("text")
        .and_then(Value::as_str)
        .is_some_and(has_non_whitespace)
    {
        return true;
    }
    if object
        .get("content")
        .is_some_and(content_has_prompt_material)
    {
        return true;
    }
    if object
        .get("image_url")
        .is_some_and(reference_has_prompt_material)
        || ["file_id", "file_url", "file_data"]
            .into_iter()
            .any(|key| object.get(key).is_some_and(reference_has_prompt_material))
    {
        return true;
    }
    object.get("source").is_some_and(|source| {
        ["data", "url"]
            .into_iter()
            .any(|key| source.get(key).is_some_and(reference_has_prompt_material))
    })
}

fn reference_has_prompt_material(reference: &Value) -> bool {
    match reference {
        Value::String(value) => has_non_whitespace(value),
        Value::Object(object) => object
            .get("url")
            .or_else(|| object.get("data"))
            .or_else(|| object.get("file_id"))
            .is_some_and(reference_has_prompt_material),
        _ => false,
    }
}

fn has_non_whitespace(value: &str) -> bool {
    !value.trim().is_empty()
}

pub(crate) fn routing_target(
    target: &switchyard_core::LlmTarget,
    tier: String,
) -> Result<RoutingTarget> {
    Ok(RoutingTarget {
        tier,
        target_model: target.model.as_str().to_string(),
        backend_id: target.id.as_str().to_string(),
        target_protocol_profile: route_protocol_for_format(target.format)?.to_string(),
        target_endpoint: route_endpoint_for_format(target.format)?.to_string(),
    })
}

fn require_non_blank(field: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(SwitchyardError::InvalidRequest(format!(
            "{field} must be a non-empty string"
        )));
    }
    Ok(())
}

fn validate_current_request_materialization(request: &RoutingRequest) -> Result<()> {
    let mode = request.decision_profile.request_materialization;
    match mode {
        CurrentRequestMaterialization::None | CurrentRequestMaterialization::SummaryOnly => {
            if request.current_request.is_some() {
                return Err(SwitchyardError::InvalidRequest(format!(
                    "request_materialization {mode:?} must not include current_request"
                )));
            }
        }
        CurrentRequestMaterialization::LatestUserPrompt
        | CurrentRequestMaterialization::RecentMessageWindow
        | CurrentRequestMaterialization::AnnotatedRequest
        | CurrentRequestMaterialization::FullBody => {
            let body = request
                .current_request
                .as_ref()
                .and_then(|current| current.get("body"))
                .ok_or_else(|| {
                    SwitchyardError::InvalidRequest(format!(
                        "request_materialization {mode:?} requires current_request.body"
                    ))
                })?;
            if !body.is_object() {
                return Err(SwitchyardError::InvalidRequest(
                    "current_request.body must be a JSON object".to_string(),
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use switchyard_core::{BackendFormat, LlmTarget, LlmTargetId, ModelId};

    use crate::{NoopProfileConfig, Profile, ProfileConfig, RandomRoutingProfileConfig};

    use super::*;

    type TestResult<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

    fn target(id: &str, model: &str, format: BackendFormat) -> Result<LlmTarget> {
        let mut target = LlmTarget::new(LlmTargetId::new(id)?, ModelId::new(model)?);
        target.format = format;
        target.endpoint.base_url = Some("http://127.0.0.1:1/v1".to_string());
        Ok(target)
    }

    fn routing_request() -> Result<RoutingRequest> {
        Ok(RoutingRequest {
            schema_version: ROUTING_REQUEST_SCHEMA_VERSION.to_string(),
            decision_profile: DecisionProfile {
                profile_id: ProfileId::new("remote-random")?,
                request_materialization: CurrentRequestMaterialization::SummaryOnly,
            },
            identity: RequestIdentity {
                session_id: "session-1".to_string(),
                request_id: "request-1".to_string(),
                turn_id: Some("turn-1".to_string()),
                parent_scope_id: None,
                root_scope_id: None,
                harness: "unit-test".to_string(),
                source: "nemo-relay".to_string(),
                owner_id: None,
                quality: IdentityQuality::Native,
            },
            protocol: RequestProtocol {
                inbound_profile: "openai_chat".to_string(),
                inbound_endpoint: "/v1/chat/completions".to_string(),
                desired_response_profile: "openai_chat".to_string(),
            },
            request_summary: RequestSummary {
                client_requested_model: Some("client/model".to_string()),
                ..RequestSummary::default()
            },
            current_request: None,
            attempt: DecisionAttempt {
                routing_attempt: 1,
                max_routing_attempts: 1,
            },
        })
    }

    fn materialized_request(inbound_profile: &str, body: Value) -> Result<RoutingRequest> {
        let mut request = routing_request()?;
        request.decision_profile.request_materialization = CurrentRequestMaterialization::FullBody;
        request.protocol.inbound_profile = inbound_profile.to_string();
        request.current_request = Some(json!({"body": body}));
        Ok(request)
    }

    #[test]
    fn routing_request_contract_requires_profile_id_and_omits_router() -> TestResult {
        let encoded = serde_json::to_value(routing_request()?)?;

        assert_eq!(encoded["decision_profile"]["profile_id"], "remote-random");
        assert!(encoded["decision_profile"].get("router").is_none());

        let mut missing_profile_id = encoded;
        missing_profile_id["decision_profile"]
            .as_object_mut()
            .ok_or_else(|| SwitchyardError::Other("decision_profile was not an object".into()))?
            .remove("profile_id");
        assert!(serde_json::from_value::<RoutingRequest>(missing_profile_id).is_err());
        Ok(())
    }

    #[tokio::test]
    async fn random_decision_uses_summary_without_dispatching_selected_backend() -> TestResult {
        let profile = RandomRoutingProfileConfig {
            strong: target("strong-target", "frontier/model", BackendFormat::OpenAi)?,
            weak: target("weak-target", "cheap/model", BackendFormat::Responses)?,
            strong_probability: 1.0,
            rng_seed: Some(7),
        }
        .build()?;

        let decision = profile.decide(routing_request()?).await?;

        assert_eq!(decision.route.backend_id, "strong-target");
        assert_eq!(decision.route.target_model, "frontier/model");
        assert_eq!(decision.route.target_protocol_profile, "openai_chat");
        assert_eq!(decision.router.name, "random-routing");
        Ok(())
    }

    #[tokio::test]
    async fn decision_rejects_unknown_inbound_profile_without_fallback() -> TestResult {
        let profile = RandomRoutingProfileConfig {
            strong: target("strong-target", "frontier/model", BackendFormat::OpenAi)?,
            weak: target("weak-target", "cheap/model", BackendFormat::OpenAi)?,
            strong_probability: 1.0,
            rng_seed: Some(7),
        }
        .build()?;
        let mut request = routing_request()?;
        request.protocol.inbound_profile = "future_protocol".to_string();

        let error = profile.decide(request).await;

        assert!(
            matches!(error, Err(SwitchyardError::InvalidRequest(message)) if message.contains("unsupported inbound_profile"))
        );
        Ok(())
    }

    #[tokio::test]
    async fn llm_materialization_rejects_summary_only_before_classifier_dispatch() -> TestResult {
        let request = routing_request()?;
        let error = materialized_profile_input(&request);

        assert!(
            matches!(error, Err(SwitchyardError::InvalidRequest(message)) if message.contains("materialized current_request.body"))
        );
        Ok(())
    }

    #[test]
    fn llm_materialization_validates_wrapper_shape_and_prompt_content() -> TestResult {
        let mut request = routing_request()?;
        request.decision_profile.request_materialization = CurrentRequestMaterialization::FullBody;
        request.current_request = Some(json!({"body": []}));

        let non_object = materialized_profile_input(&request);
        assert!(
            matches!(non_object, Err(SwitchyardError::InvalidRequest(message)) if message.contains("must be a JSON object"))
        );

        request.current_request = Some(json!({"body": {"model": "client/model"}}));
        let missing_prompt = materialized_profile_input(&request);
        assert!(
            matches!(missing_prompt, Err(SwitchyardError::InvalidRequest(message)) if message.contains("non-empty prompt material"))
        );
        Ok(())
    }

    #[test]
    fn llm_materialization_rejects_promptless_message_shapes() -> TestResult {
        for (profile, body) in [
            (
                "openai_chat",
                json!({"messages": [{"role": "user", "content": " \n\t "}]}),
            ),
            (
                "openai_chat",
                json!({"messages": [{"role": "user", "content": []}]}),
            ),
            (
                "openai_chat",
                json!({"messages": [{"role": "assistant", "content": "not a user prompt"}]}),
            ),
            (
                "anthropic_messages",
                json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "  "}]}]}),
            ),
            (
                "anthropic_messages",
                json!({"messages": [{"role": "assistant", "content": "not a user prompt"}]}),
            ),
        ] {
            let request = materialized_request(profile, body)?;
            let error = materialized_profile_input(&request);
            assert!(
                matches!(error, Err(SwitchyardError::InvalidRequest(message)) if message.contains("non-empty prompt material")),
                "{profile} promptless body should be rejected"
            );
        }
        Ok(())
    }

    #[test]
    fn llm_materialization_accepts_supported_user_prompt_shapes() -> TestResult {
        for (profile, body) in [
            (
                "openai_chat",
                json!({"messages": [{"role": "user", "content": "implement this"}]}),
            ),
            (
                "openai_chat",
                json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "implement this"}]}]}),
            ),
            (
                "anthropic_messages",
                json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "implement this"}]}]}),
            ),
            (
                "anthropic_messages",
                json!({"messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "data": "abc123"}}]}]}),
            ),
            ("openai_responses", json!({"input": "implement this"})),
            (
                "openai_responses",
                json!({"input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "implement this"}]}]}),
            ),
            (
                "openai_responses",
                json!({"input": [{"type": "input_text", "text": "implement this"}]}),
            ),
        ] {
            let request = materialized_request(profile, body)?;
            assert!(
                materialized_profile_input(&request).is_ok(),
                "{profile} user prompt should be accepted"
            );
        }
        Ok(())
    }

    #[test]
    fn responses_materialization_rejects_structurally_nonempty_promptless_input() -> TestResult {
        for input in [
            json!([{"type": "message", "role": "user", "content": []}]),
            json!([{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "  "}]}]),
            json!([{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "assistant only"}]}]),
            json!([{"type": "reasoning", "summary": [{"type": "summary_text", "text": "not user input"}]}]),
            json!([{"type": "input_image", "image_url": "  "}]),
        ] {
            let request = materialized_request("openai_responses", json!({"input": input}))?;
            let error = materialized_profile_input(&request);
            assert!(
                matches!(error, Err(SwitchyardError::InvalidRequest(message)) if message.contains("non-empty prompt material"))
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn default_profile_decision_is_typed_unsupported() -> TestResult {
        let profile = NoopProfileConfig {}.build()?;
        let request = routing_request()?;
        let profile_id = request.decision_profile.profile_id.clone();

        let error = profile.decide(request).await;

        assert!(
            matches!(error, Err(SwitchyardError::DecisionUnsupported { profile_id: rejected }) if rejected == profile_id)
        );
        Ok(())
    }

    #[test]
    fn documented_inbound_aliases_parse_but_unknown_values_do_not() -> TestResult {
        for profile in [
            "openai_chat",
            "openai_chat_completions.v1",
            "openai_responses",
            "openai_responses.v1",
            "anthropic",
            "anthropic_messages.v1",
        ] {
            parse_inbound_profile(profile)?;
        }
        assert!(parse_inbound_profile("openai").is_err());
        assert!(parse_inbound_profile("chat").is_err());
        Ok(())
    }

    #[test]
    fn routing_request_round_trips_additive_fields() -> TestResult {
        let mut encoded = serde_json::to_value(routing_request()?)?;
        encoded["future_envelope_field"] = json!({"accepted": true});
        encoded["decision_profile"]["router"] = json!("legacy-relay-router");

        let decoded: RoutingRequest = serde_json::from_value(encoded)?;

        assert_eq!(decoded, routing_request()?);
        Ok(())
    }
}
