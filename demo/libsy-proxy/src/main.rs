// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Demo proxy combining Switchyard HTTP/translation with libsy agent-aware routing.
//!
//! The server accepts OpenAI Chat, Anthropic Messages, and OpenAI Responses requests.
//! libsy normalizes harness identity headers, assigns a stable agent/task to a model
//! pool target, and makes calls through Switchyard's OpenAI-compatible backend.

use std::collections::BTreeMap;
use std::error::Error;
use std::net::SocketAddr;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use libsy::agentic::{
    metadata_from_headers, AgentAwareOrchAlgo, AgentRoutingCandidate, AgentRoutingDecision,
};
use libsy::{
    Algorithm, Context, Decision, LlmClient, LlmResponse, LlmTarget, LlmTargetSet, Request,
    Response, RoutedRequest,
};
use switchyard_components::OpenAiPassthroughBackend;
use switchyard_components_v2::{Profile, ProfileInput, ProfileResponse, RoutingMetadata};
use switchyard_core::{
    ChatRequest, ChatRequestType, ChatResponse, EndpointConfig, LlmBackend, ModelId, ProxyContext,
    Result, SwitchyardError,
};
use switchyard_server::{build_switchyard_router, ProfileRegistry, ServerState};
use switchyard_translation::{TranslationEngine, TranslationPolicy, WireFormat};

const CLASSIFIER_MODEL: &str = "nvidia/deepseek-ai/deepseek-v4-flash";
const FRONTIER_MODEL: &str = "aws/anthropic/bedrock-claude-opus-4-7";
const FAST_MODEL: &str = "nvidia/deepseek-ai/deepseek-v4-flash";
const CLASSIFIER_TARGET: &str = "classifier";
const FRONTIER_TARGET: &str = "frontier";
const FAST_TARGET: &str = "fast";

const DEFAULT_BASE_URL: &str = "https://inference-api.nvidia.com/v1";
const DEFAULT_ADDR: &str = "127.0.0.1:4000";
const PROFILE_MODEL_ID: &str = "libsy-agent-aware";

type BoxErr = Box<dyn Error + Send + Sync>;

/// libsy client backed by Switchyard's OpenAI-compatible backend.
struct SwitchyardBackendClient {
    backend: Arc<OpenAiPassthroughBackend>,
    model_ids: BTreeMap<String, String>,
    translation: Arc<TranslationEngine>,
}

#[async_trait]
impl LlmClient for SwitchyardBackendClient {
    async fn call(&self, routed: RoutedRequest) -> std::result::Result<Response, BoxErr> {
        let target = routed.decision.selected_model();
        let model = self
            .model_ids
            .get(target)
            .ok_or_else(|| format!("no provider model configured for target '{target}'"))?;
        let chat_request =
            chat_request_for_call(&routed.request, model, self.translation.as_ref())?;

        let mut ctx = ProxyContext::new();
        let response = self
            .backend
            .call(&mut ctx, &chat_request)
            .await
            .map_err(|error| BoxErr::from(error.to_string()))?;
        let body = response.body().cloned().ok_or_else(|| {
            BoxErr::from("the libsy proxy demo currently requires buffered upstream responses")
        })?;
        let decoded = self
            .translation
            .decode_response(WireFormat::OpenAiChat, &body, &TranslationPolicy::default())
            .map_err(|error| BoxErr::from(error.to_string()))?;

        Ok(Response {
            llm_response: LlmResponse::Agg(decoded.response),
            metadata: routed.request.metadata,
        })
    }
}

/// Switchyard profile that delegates all routing decisions to libsy.
struct LibsyAgentAwareProfile {
    algorithm: Arc<dyn Algorithm>,
    translation: Arc<TranslationEngine>,
}

#[async_trait]
impl Profile for LibsyAgentAwareProfile {
    async fn run(&self, input: ProfileInput) -> Result<ProfileResponse> {
        let request = libsy_request(&input, self.translation.as_ref())?;
        let (trace, response) = Arc::clone(&self.algorithm)
            .run(Context::default(), request)
            .await
            .map_err(|error| SwitchyardError::Other(error.to_string()))?;
        let aggregate = response
            .llm_response
            .into_agg()
            .await
            .map_err(|error| SwitchyardError::Other(error.to_string()))?;
        let body = self
            .translation
            .encode_response(
                WireFormat::OpenAiChat,
                &aggregate,
                &TranslationPolicy::default(),
            )
            .map_err(|error| SwitchyardError::Other(error.to_string()))?
            .body;

        Ok(ProfileResponse::with_routing_metadata(
            ChatResponse::openai_completion(body),
            routing_metadata(&trace),
        ))
    }
}

/// Decode the inbound wire body and attach normalized harness metadata.
fn libsy_request(input: &ProfileInput, translation: &TranslationEngine) -> Result<Request> {
    let format = wire_format(input.request.request_type());
    let decoded = translation
        .decode_request(format, input.request.body(), &TranslationPolicy::default())
        .map_err(|error| SwitchyardError::InvalidRequest(error.to_string()))?;
    let mut metadata = metadata_from_headers(&input.metadata.headers);
    metadata
        .extra_metadata
        .get_or_insert_with(Default::default)
        .insert(
            "inbound_format".to_string(),
            inbound_format(input.request.request_type()).to_string(),
        );

    Ok(Request {
        llm_request: decoded.request,
        raw_request: Some(input.request.body().clone()),
        metadata: Some(metadata),
    })
}

/// Preserve routed provider payloads; encode classifier calls from the neutral IR.
fn chat_request_for_call(
    request: &Request,
    model: &str,
    translation: &TranslationEngine,
) -> std::result::Result<ChatRequest, BoxErr> {
    let Some(mut body) = request.raw_request.clone() else {
        let mut classifier_request = request.llm_request.clone();
        classifier_request.model = Some(model.to_string());
        classifier_request.stream = false;
        let body = translation
            .encode_request(
                WireFormat::OpenAiChat,
                &classifier_request,
                &TranslationPolicy::default(),
            )
            .map_err(|error| BoxErr::from(error.to_string()))?
            .body;
        return Ok(ChatRequest::openai_chat(body));
    };

    if let Some(object) = body.as_object_mut() {
        object.insert("model".to_string(), Value::String(model.to_string()));
    }
    let request = match request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.extra_metadata.as_ref())
        .and_then(|extra| extra.get("inbound_format"))
        .map(String::as_str)
    {
        Some("anthropic") => ChatRequest::anthropic(body),
        Some("openai_responses") => ChatRequest::openai_responses(body),
        _ => ChatRequest::openai_chat(body),
    };
    Ok(request)
}

fn inbound_format(request_type: ChatRequestType) -> &'static str {
    match request_type {
        ChatRequestType::OpenAiChat => "openai_chat",
        ChatRequestType::Anthropic => "anthropic",
        ChatRequestType::OpenAiResponses => "openai_responses",
    }
}

fn wire_format(request_type: ChatRequestType) -> WireFormat {
    match request_type {
        ChatRequestType::OpenAiChat => WireFormat::OpenAiChat,
        ChatRequestType::Anthropic => WireFormat::AnthropicMessages,
        ChatRequestType::OpenAiResponses => WireFormat::OpenAiResponses,
    }
}

/// Surface libsy's routing decision as `x-model-router-*` response headers.
fn routing_metadata(trace: &[Arc<dyn Decision>]) -> RoutingMetadata {
    let decision = trace.last();
    let agent =
        decision.and_then(|decision| decision.as_any().downcast_ref::<AgentRoutingDecision>());
    let selected_target = decision.map(|decision| decision.selected_model());
    RoutingMetadata {
        selected_model: selected_target.map(|target| match target {
            FRONTIER_TARGET => FRONTIER_MODEL.to_string(),
            FAST_TARGET => FAST_MODEL.to_string(),
            other => other.to_string(),
        }),
        selected_tier: selected_target.map(str::to_string),
        confidence: agent.and_then(|decision| decision.confidence),
        router_version: Some("libsy-agent-aware-v1".to_string()),
        tolerance: None,
        rationale: decision.and_then(|decision| decision.reasoning().map(str::to_string)),
    }
}

fn build_algorithm() -> Result<Arc<dyn Algorithm>> {
    let base_url =
        std::env::var("LIBSY_PROXY_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
    let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
        SwitchyardError::InvalidConfig(
            "ANTHROPIC_API_KEY must be set to the upstream bearer key".to_string(),
        )
    })?;
    let backend = Arc::new(OpenAiPassthroughBackend::new(EndpointConfig {
        base_url: Some(base_url),
        api_key: Some(api_key),
        timeout_secs: Some(120.0),
    })?);
    let model_ids = BTreeMap::from([
        (CLASSIFIER_TARGET.to_string(), CLASSIFIER_MODEL.to_string()),
        (FRONTIER_TARGET.to_string(), FRONTIER_MODEL.to_string()),
        (FAST_TARGET.to_string(), FAST_MODEL.to_string()),
    ]);
    let translation = Arc::new(TranslationEngine::default());
    let client = Arc::new(SwitchyardBackendClient {
        backend,
        model_ids,
        translation,
    }) as Arc<dyn LlmClient>;
    let target = |name: &str| LlmTarget {
        semantic_name: name.to_string(),
        llm_client: Some(Arc::clone(&client)),
    };
    let targets = LlmTargetSet::new(vec![
        target(CLASSIFIER_TARGET),
        target(FRONTIER_TARGET),
        target(FAST_TARGET),
    ]);

    Ok(Arc::new(AgentAwareOrchAlgo::new(
        CLASSIFIER_TARGET,
        vec![
            AgentRoutingCandidate::new(
                FRONTIER_TARGET,
                "frontier model for planning, ambiguous implementation, synthesis, and review",
            ),
            AgentRoutingCandidate::new(
                FAST_TARGET,
                "efficient model for bounded exploration, retrieval, and mechanical edits",
            ),
        ],
        FRONTIER_TARGET,
        targets,
    )))
}

#[tokio::main]
async fn main() -> Result<()> {
    let translation = Arc::new(TranslationEngine::default());
    let profile = Arc::new(LibsyAgentAwareProfile {
        algorithm: build_algorithm()?,
        translation,
    }) as Arc<dyn Profile>;
    let registry = ProfileRegistry::from_profiles([(
        ModelId::new(PROFILE_MODEL_ID)?,
        profile,
        PROFILE_MODEL_ID.to_string(),
    )])?;
    let state = ServerState::new(registry);
    let addr: SocketAddr = std::env::var("LIBSY_PROXY_ADDR")
        .unwrap_or_else(|_| DEFAULT_ADDR.to_string())
        .parse()
        .map_err(|error: std::net::AddrParseError| {
            SwitchyardError::InvalidConfig(error.to_string())
        })?;
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|error| SwitchyardError::Other(error.to_string()))?;
    let bound_addr = listener
        .local_addr()
        .map_err(|error| SwitchyardError::Other(error.to_string()))?;

    println!("libsy-proxy listening on http://{bound_addr}");
    println!("  routing (libsy agent-aware): classifier={CLASSIFIER_MODEL}");
    println!("                               frontier={FRONTIER_MODEL}");
    println!("                               fast={FAST_MODEL}");
    println!(
        "  send model \"{PROFILE_MODEL_ID}\" to /v1/chat/completions, /v1/messages, or /v1/responses"
    );

    axum::serve(listener, build_switchyard_router(state))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .map_err(|error| SwitchyardError::Other(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use switchyard_components_v2::RequestMetadata;
    use switchyard_protocol::{prompt_text, text_request, Metadata};

    #[test]
    fn routed_call_preserves_provider_body_and_rewrites_model() -> std::result::Result<(), BoxErr> {
        let request = Request {
            llm_request: text_request(Some("libsy-agent-aware".to_string()), "inspect"),
            raw_request: Some(json!({
                "model": "libsy-agent-aware",
                "input": "inspect",
                "tools": [{"type": "function", "name": "shell"}],
                "stream": false,
            })),
            metadata: Some(Metadata {
                extra_metadata: Some(BTreeMap::from([(
                    "inbound_format".to_string(),
                    "openai_responses".to_string(),
                )])),
                ..Metadata::default()
            }),
        };

        let routed =
            chat_request_for_call(&request, "provider/model", &TranslationEngine::default())?;
        assert_eq!(routed.request_type(), ChatRequestType::OpenAiResponses);
        assert_eq!(
            routed.body().get("model").and_then(Value::as_str),
            Some("provider/model")
        );
        assert!(routed.body().get("tools").is_some());
        Ok(())
    }

    #[test]
    fn decodes_responses_input_and_normalizes_headers() -> Result<()> {
        let input = ProfileInput {
            request: ChatRequest::openai_responses(json!({
                "model": "libsy-agent-aware",
                "input": [{
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "current subtask"}]
                }],
                "stream": false,
            })),
            metadata: RequestMetadata {
                headers: BTreeMap::from([(
                    "thread-id".to_string(),
                    vec!["child-agent".to_string()],
                )]),
                ..RequestMetadata::default()
            },
        };

        let request = libsy_request(&input, &TranslationEngine::default())?;
        assert_eq!(prompt_text(&request.llm_request), "current subtask");
        assert_eq!(
            request
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.agent_id.as_deref()),
            Some("child-agent")
        );
        Ok(())
    }

    #[test]
    fn classifier_call_uses_a_buffered_synthetic_request() -> std::result::Result<(), BoxErr> {
        let request = Request {
            llm_request: text_request(Some("auto".to_string()), "classify this"),
            raw_request: None,
            metadata: None,
        };

        let classifier =
            chat_request_for_call(&request, "classifier/model", &TranslationEngine::default())?;
        assert_eq!(classifier.request_type(), ChatRequestType::OpenAiChat);
        assert_ne!(
            classifier.body().get("stream").and_then(Value::as_bool),
            Some(true)
        );
        Ok(())
    }

    #[test]
    fn response_metadata_exposes_provider_model_and_logical_tier() {
        let decision: Arc<dyn Decision> = Arc::new(AgentRoutingDecision {
            selected_model: FAST_TARGET.to_string(),
            reason: "bounded lookup".to_string(),
            task_kind: Some("research".to_string()),
            confidence: Some(0.9),
            agent_id: Some("child-1".to_string()),
            cache_hit: false,
        });

        let metadata = routing_metadata(&[decision]);
        assert_eq!(metadata.selected_model.as_deref(), Some(FAST_MODEL));
        assert_eq!(metadata.selected_tier.as_deref(), Some(FAST_TARGET));
        assert_eq!(metadata.confidence, Some(0.9));
    }
}
