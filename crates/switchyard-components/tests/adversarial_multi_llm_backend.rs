// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Adversarial tests for the Rust multi-target LLM backend.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::{json, Value};
use switchyard_components::{
    BackendSelection, BackendSelectionReason, LlmTargetBackend, MultiLlmBackend,
};
use switchyard_core::{
    BackendFormat, ChatRequest, ChatRequestType, ChatResponse, LlmBackend, LlmTarget, LlmTargetId,
    ModelId, ProxyContext, Result, SwitchyardError,
};

/// One backend call observed by the recording backend.
#[derive(Clone, Debug, PartialEq)]
struct ObservedCall {
    /// Name of the backend that handled the request.
    backend_name: &'static str,
    /// Request wire type received by the child backend.
    request_type: ChatRequestType,
    /// Request model visible to the child backend.
    model: Option<String>,
    /// Full request body visible to the child backend.
    body: Value,
    /// Context-selected target visible at delegation time.
    ctx_selected_target: Option<LlmTargetId>,
    /// Context-selected model visible at delegation time.
    ctx_selected_model: Option<ModelId>,
}

/// Mutex-backed shared vector used for test observations.
#[derive(Clone)]
struct Shared<T>(Arc<Mutex<Vec<T>>>);

impl<T> Default for Shared<T> {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(Vec::new())))
    }
}

impl<T: Clone> Shared<T> {
    /// Appends one observation with a typed test error on poisoned mutexes.
    fn push(&self, value: T) -> Result<()> {
        match self.0.lock() {
            Ok(mut values) => {
                values.push(value);
                Ok(())
            }
            Err(_) => Err(SwitchyardError::Other(
                "shared test log mutex poisoned".to_string(),
            )),
        }
    }

    /// Returns a cloned copy of all observations.
    fn values(&self) -> Result<Vec<T>> {
        match self.0.lock() {
            Ok(values) => Ok(values.clone()),
            Err(_) => Err(SwitchyardError::Other(
                "shared test log mutex poisoned".to_string(),
            )),
        }
    }
}

/// Backend fixture that records calls and optional lifecycle failures.
struct RecordingBackend {
    /// Stable backend name for assertions.
    name: &'static str,
    /// Captured backend calls.
    calls: Shared<ObservedCall>,
    /// Captured startup/shutdown events.
    events: Shared<String>,
    /// Request types this fixture advertises.
    supported_request_types: &'static [ChatRequestType],
    /// Optional call error.
    call_error: Option<&'static str>,
    /// Optional startup error.
    startup_error: Option<&'static str>,
    /// Optional shutdown error.
    shutdown_error: Option<&'static str>,
}

impl RecordingBackend {
    /// Creates a recording backend with all request types enabled.
    fn new(name: &'static str, calls: Shared<ObservedCall>, events: Shared<String>) -> Self {
        Self {
            name,
            calls,
            events,
            supported_request_types: &ALL_REQUEST_TYPES,
            call_error: None,
            startup_error: None,
            shutdown_error: None,
        }
    }

    /// Overrides the advertised request types.
    fn with_supported_request_types(
        mut self,
        supported_request_types: &'static [ChatRequestType],
    ) -> Self {
        self.supported_request_types = supported_request_types;
        self
    }

    /// Configures a call-time backend error.
    fn with_call_error(mut self, message: &'static str) -> Self {
        self.call_error = Some(message);
        self
    }

    /// Configures a startup failure.
    fn with_startup_error(mut self, message: &'static str) -> Self {
        self.startup_error = Some(message);
        self
    }

    /// Configures a shutdown failure.
    fn with_shutdown_error(mut self, message: &'static str) -> Self {
        self.shutdown_error = Some(message);
        self
    }
}

static ALL_REQUEST_TYPES: [ChatRequestType; 3] = [
    ChatRequestType::OpenAiChat,
    ChatRequestType::OpenAiResponses,
    ChatRequestType::Anthropic,
];
static OPENAI_CHAT_ONLY: [ChatRequestType; 1] = [ChatRequestType::OpenAiChat];

#[async_trait]
impl LlmBackend for RecordingBackend {
    fn supported_request_types(&self) -> &[ChatRequestType] {
        self.supported_request_types
    }

    async fn call(&self, ctx: &mut ProxyContext, request: &ChatRequest) -> Result<ChatResponse> {
        self.calls.push(ObservedCall {
            backend_name: self.name,
            request_type: request.request_type(),
            model: request.model().map(str::to_string),
            body: request.body().clone(),
            ctx_selected_target: selected_target(ctx).cloned(),
            ctx_selected_model: selected_model(ctx).cloned(),
        })?;
        if let Some(message) = self.call_error {
            return Err(SwitchyardError::Backend(message.to_string()));
        }
        Ok(ChatResponse::openai_completion(json!({
            "backend": self.name,
            "model": request.model(),
        })))
    }

    async fn startup(&self) -> Result<()> {
        self.events.push(format!("{}:startup", self.name))?;
        if let Some(message) = self.startup_error {
            return Err(SwitchyardError::Backend(message.to_string()));
        }
        Ok(())
    }

    async fn shutdown(&self) -> Result<()> {
        self.events.push(format!("{}:shutdown", self.name))?;
        if let Some(message) = self.shutdown_error {
            return Err(SwitchyardError::Backend(message.to_string()));
        }
        Ok(())
    }
}

/// Builds an OpenAI-format test target.
fn target(id: &'static str, model: &'static str) -> LlmTarget {
    let mut target = LlmTarget::new(LlmTargetId::from_static(id), ModelId::from_static(model));
    target.format = BackendFormat::OpenAi;
    target
}

/// Builds a target/backend pair for `MultiLlmBackend`.
fn target_backend(
    id: &'static str,
    model: &'static str,
    backend: RecordingBackend,
) -> LlmTargetBackend {
    LlmTargetBackend::new(target(id, model), Arc::new(backend))
}

/// Builds an OpenAI Chat request fixture.
fn request(model: &str) -> ChatRequest {
    ChatRequest::openai_chat(json!({
        "model": model,
        "messages": [{"role": "user", "content": "preserve me"}],
        "temperature": 0.2
    }))
}

/// Builds an Anthropic request fixture.
fn anthropic_request(model: &str) -> ChatRequest {
    ChatRequest::anthropic(json!({
        "model": model,
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "preserve anthropic"}],
        "metadata": {"kept": true}
    }))
}

/// Builds a Responses API request fixture.
fn responses_request(model: &str) -> ChatRequest {
    ChatRequest::openai_responses(json!({
        "model": model,
        "input": "preserve responses",
        "metadata": {"kept": true}
    }))
}

/// Returns the backend selection stamped by `MultiLlmBackend`.
fn selection(ctx: &ProxyContext) -> Result<&BackendSelection> {
    ctx.get::<BackendSelection>()
        .ok_or_else(|| SwitchyardError::Other("multi-LLM selection should be recorded".to_string()))
}

/// Returns the selected model stamped in context.
fn selected_model(ctx: &ProxyContext) -> Option<&ModelId> {
    ctx.get::<BackendSelection>()
        .map(|selection| &selection.model)
}

/// Returns the selected target stamped in context.
fn selected_target(ctx: &ProxyContext) -> Option<&LlmTargetId> {
    ctx.get::<BackendSelection>()
        .and_then(|selection| selection.target_id.as_ref())
}

// Default support should cover all inbound wire formats in stable order.
#[test]
fn default_supported_request_types_are_all_wire_formats_in_stable_order() -> Result<()> {
    let calls = Shared::default();
    let events = Shared::default();
    let backend = MultiLlmBackend::new([target_backend(
        "only-target",
        "served-model",
        RecordingBackend::new("only", calls, events),
    )])?;

    assert_eq!(
        backend.supported_request_types(),
        &[
            ChatRequestType::OpenAiChat,
            ChatRequestType::OpenAiResponses,
            ChatRequestType::Anthropic,
            ChatRequestType::Gemini,
        ]
    );
    Ok(())
}

// Custom request type support should de-dupe without reordering caller input.
#[test]
fn custom_supported_request_types_are_deduped_without_reordering() -> Result<()> {
    let calls = Shared::default();
    let events = Shared::default();
    let backend = MultiLlmBackend::new([target_backend(
        "only-target",
        "served-model",
        RecordingBackend::new("only", calls, events),
    )])?
    .with_supported_request_types([
        ChatRequestType::Anthropic,
        ChatRequestType::OpenAiChat,
        ChatRequestType::Anthropic,
        ChatRequestType::OpenAiChat,
        ChatRequestType::OpenAiResponses,
    ])?;

    assert_eq!(
        backend.supported_request_types(),
        &[
            ChatRequestType::Anthropic,
            ChatRequestType::OpenAiChat,
            ChatRequestType::OpenAiResponses,
        ]
    );
    Ok(())
}

// Context-selected targets should win and cloned requests should keep caller state intact.
#[tokio::test]
async fn context_selected_target_wins_and_request_is_cloned() -> Result<()> {
    let strong_calls = Shared::default();
    let weak_calls = Shared::default();
    let events = Shared::default();
    let backend = MultiLlmBackend::new([
        target_backend(
            "strong-target",
            "strong-model",
            RecordingBackend::new("strong", strong_calls.clone(), events.clone()),
        ),
        target_backend(
            "weak-target",
            "weak-model",
            RecordingBackend::new("weak", weak_calls.clone(), events.clone()),
        ),
    ])?;
    let original = request("client-model");
    let mut ctx = ProxyContext::new();
    ctx.set_selected_target(LlmTargetId::from_static("weak-target"));

    let response = backend.call(&mut ctx, &original).await?;

    assert_eq!(
        response.body(),
        Some(&json!({"backend": "weak", "model": "weak-model"}))
    );
    assert_eq!(original.model(), Some("client-model"));
    assert_eq!(
        selected_target(&ctx),
        Some(&LlmTargetId::from_static("weak-target"))
    );
    assert_eq!(
        selected_model(&ctx),
        Some(&ModelId::from_static("weak-model"))
    );
    assert_eq!(
        selection(&ctx)?.reason,
        BackendSelectionReason::ContextTarget
    );
    assert_eq!(
        selection(&ctx)?.original_model.as_deref(),
        Some("client-model")
    );
    assert!(strong_calls.values()?.is_empty());

    let weak = weak_calls.values()?;
    assert_eq!(weak.len(), 1);
    assert_eq!(weak[0].backend_name, "weak");
    assert_eq!(weak[0].request_type, ChatRequestType::OpenAiChat);
    assert_eq!(weak[0].model.as_deref(), Some("weak-model"));
    assert_eq!(weak[0].body["messages"][0]["content"], "preserve me");
    assert_eq!(
        weak[0].ctx_selected_target,
        Some(LlmTargetId::from_static("weak-target"))
    );
    assert_eq!(
        weak[0].ctx_selected_model,
        Some(ModelId::from_static("weak-model"))
    );
    Ok(())
}

// Explicit context target selection should override request model matches.
#[tokio::test]
async fn context_selected_target_wins_even_when_request_model_matches_another_target() -> Result<()>
{
    let strong_calls = Shared::default();
    let weak_calls = Shared::default();
    let events = Shared::default();
    let backend = MultiLlmBackend::new([
        target_backend(
            "strong-target",
            "strong-model",
            RecordingBackend::new("strong", strong_calls.clone(), events.clone()),
        ),
        target_backend(
            "weak-target",
            "weak-model",
            RecordingBackend::new("weak", weak_calls.clone(), events),
        ),
    ])?;
    let mut ctx = ProxyContext::new();
    ctx.set_selected_target(LlmTargetId::from_static("weak-target"));

    backend.call(&mut ctx, &request("strong-model")).await?;

    assert!(strong_calls.values()?.is_empty());
    let weak = weak_calls.values()?;
    assert_eq!(weak.len(), 1);
    assert_eq!(weak[0].model.as_deref(), Some("weak-model"));
    assert_eq!(
        selection(&ctx)?.reason,
        BackendSelectionReason::ContextTarget
    );
    assert_eq!(
        selection(&ctx)?.original_model.as_deref(),
        Some("strong-model")
    );
    Ok(())
}

// A single configured target should route without any selector processor.
#[tokio::test]
async fn single_target_fallback_routes_without_selector() -> Result<()> {
    let calls = Shared::default();
    let events = Shared::default();
    let backend = MultiLlmBackend::new([target_backend(
        "only-target",
        "served-model",
        RecordingBackend::new("only", calls.clone(), events),
    )])?;
    let mut ctx = ProxyContext::new();

    backend.call(&mut ctx, &request("client-model")).await?;

    assert_eq!(
        selected_target(&ctx),
        Some(&LlmTargetId::from_static("only-target"))
    );
    assert_eq!(
        selected_model(&ctx),
        Some(&ModelId::from_static("served-model"))
    );
    assert_eq!(
        selection(&ctx)?.reason,
        BackendSelectionReason::SingleTarget
    );
    let calls = calls.values()?;
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].model.as_deref(), Some("served-model"));
    Ok(())
}

// Non-object request bodies should be repaired only in the delegated clone.
#[tokio::test]
async fn single_target_fallback_recovers_non_object_request_body_without_mutating_original(
) -> Result<()> {
    let calls = Shared::default();
    let events = Shared::default();
    let backend = MultiLlmBackend::new([target_backend(
        "only-target",
        "served-model",
        RecordingBackend::new("only", calls.clone(), events),
    )])?;
    let original = ChatRequest::openai_chat(json!("not an object"));
    let mut ctx = ProxyContext::new();

    backend.call(&mut ctx, &original).await?;

    assert_eq!(original.body(), &json!("not an object"));
    let calls = calls.values()?;
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].body, json!({"model": "served-model"}));
    assert_eq!(selection(&ctx)?.original_model, None);
    Ok(())
}

// Anthropic request bodies should keep their wire shape while the model is rewritten.
#[tokio::test]
async fn routing_preserves_anthropic_payload_shape_while_rewriting_model() -> Result<()> {
    let calls = Shared::default();
    let events = Shared::default();
    let backend = MultiLlmBackend::new([target_backend(
        "anthropic-target",
        "served-claude",
        RecordingBackend::new("anthropic", calls.clone(), events),
    )])?;
    let original = anthropic_request("client-claude");
    let mut ctx = ProxyContext::new();

    backend.call(&mut ctx, &original).await?;

    assert_eq!(original.model(), Some("client-claude"));
    let calls = calls.values()?;
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].request_type, ChatRequestType::Anthropic);
    assert_eq!(calls[0].model.as_deref(), Some("served-claude"));
    assert_eq!(calls[0].body["max_tokens"], 128);
    assert_eq!(
        calls[0].body["messages"][0]["content"],
        "preserve anthropic"
    );
    assert_eq!(calls[0].body["metadata"], json!({"kept": true}));
    Ok(())
}

// Responses request bodies should keep their wire shape while the model is rewritten.
#[tokio::test]
async fn routing_preserves_responses_payload_shape_while_rewriting_model() -> Result<()> {
    let calls = Shared::default();
    let events = Shared::default();
    let backend = MultiLlmBackend::new([target_backend(
        "responses-target",
        "served-responses",
        RecordingBackend::new("responses", calls.clone(), events),
    )])?;
    let original = responses_request("client-responses");
    let mut ctx = ProxyContext::new();

    backend.call(&mut ctx, &original).await?;

    assert_eq!(original.model(), Some("client-responses"));
    let calls = calls.values()?;
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].request_type, ChatRequestType::OpenAiResponses);
    assert_eq!(calls[0].model.as_deref(), Some("served-responses"));
    assert_eq!(calls[0].body["input"], "preserve responses");
    assert_eq!(calls[0].body["metadata"], json!({"kept": true}));
    Ok(())
}

// Multi-LLM dispatch should not reject a selected child based on its native format list.
#[tokio::test]
async fn selected_child_backend_receives_request_even_when_its_direct_formats_do_not_match(
) -> Result<()> {
    let calls = Shared::default();
    let events = Shared::default();
    let backend = MultiLlmBackend::new([target_backend(
        "translated-target",
        "served-model",
        RecordingBackend::new("translated", calls.clone(), events)
            .with_supported_request_types(&OPENAI_CHAT_ONLY),
    )])?;
    let mut ctx = ProxyContext::new();

    backend
        .call(&mut ctx, &anthropic_request("client-claude"))
        .await?;

    let calls = calls.values()?;
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].request_type, ChatRequestType::Anthropic);
    assert_eq!(calls[0].model.as_deref(), Some("served-model"));
    assert_eq!(
        calls[0].ctx_selected_target,
        Some(LlmTargetId::from_static("translated-target"))
    );
    Ok(())
}

// Request model should select a unique matching target when context is empty.
#[tokio::test]
async fn request_model_selects_unique_target_when_context_is_empty() -> Result<()> {
    let strong_calls = Shared::default();
    let weak_calls = Shared::default();
    let events = Shared::default();
    let backend = MultiLlmBackend::new([
        target_backend(
            "strong-target",
            "strong-model",
            RecordingBackend::new("strong", strong_calls.clone(), events.clone()),
        ),
        target_backend(
            "weak-target",
            "weak-model",
            RecordingBackend::new("weak", weak_calls.clone(), events),
        ),
    ])?;
    let mut ctx = ProxyContext::new();

    backend.call(&mut ctx, &request("strong-model")).await?;

    assert_eq!(
        selected_target(&ctx),
        Some(&LlmTargetId::from_static("strong-target"))
    );
    assert_eq!(
        selection(&ctx)?.reason,
        BackendSelectionReason::RequestModel
    );
    assert_eq!(strong_calls.values()?.len(), 1);
    assert!(weak_calls.values()?.is_empty());
    Ok(())
}

// Configured default targets should handle model-less requests.
#[tokio::test]
async fn configured_default_target_routes_when_no_selector_ran() -> Result<()> {
    let strong_calls = Shared::default();
    let weak_calls = Shared::default();
    let events = Shared::default();
    let backend = MultiLlmBackend::new([
        target_backend(
            "strong-target",
            "strong-model",
            RecordingBackend::new("strong", strong_calls.clone(), events.clone()),
        ),
        target_backend(
            "weak-target",
            "weak-model",
            RecordingBackend::new("weak", weak_calls.clone(), events),
        ),
    ])?
    .with_default_target(LlmTargetId::from_static("strong-target"))?;
    let mut ctx = ProxyContext::new();
    let request = ChatRequest::openai_chat(json!({
        "messages": [{"role": "user", "content": "no model from client"}]
    }));

    backend.call(&mut ctx, &request).await?;

    assert_eq!(
        selected_target(&ctx),
        Some(&LlmTargetId::from_static("strong-target"))
    );
    assert_eq!(
        selected_model(&ctx),
        Some(&ModelId::from_static("strong-model"))
    );
    assert_eq!(
        selection(&ctx)?.reason,
        BackendSelectionReason::DefaultTarget
    );
    assert_eq!(strong_calls.values()?.len(), 1);
    assert!(weak_calls.values()?.is_empty());
    Ok(())
}

// Configured default targets intentionally override request-model selection.
#[tokio::test]
async fn configured_default_target_overrides_request_model_selection() -> Result<()> {
    let strong_calls = Shared::default();
    let weak_calls = Shared::default();
    let events = Shared::default();
    let backend = MultiLlmBackend::new([
        target_backend(
            "strong-target",
            "strong-model",
            RecordingBackend::new("strong", strong_calls.clone(), events.clone()),
        ),
        target_backend(
            "weak-target",
            "weak-model",
            RecordingBackend::new("weak", weak_calls.clone(), events),
        ),
    ])?
    .with_default_target(LlmTargetId::from_static("strong-target"))?;
    let mut ctx = ProxyContext::new();

    backend.call(&mut ctx, &request("weak-model")).await?;

    assert_eq!(
        selected_target(&ctx),
        Some(&LlmTargetId::from_static("strong-target"))
    );
    assert_eq!(
        selection(&ctx)?.reason,
        BackendSelectionReason::DefaultTarget
    );
    assert_eq!(
        selection(&ctx)?.original_model.as_deref(),
        Some("weak-model")
    );
    assert_eq!(strong_calls.values()?.len(), 1);
    assert!(weak_calls.values()?.is_empty());
    Ok(())
}

// Context-selected targets still have the highest routing priority.
#[tokio::test]
async fn explicit_context_target_still_wins_over_configured_default_target() -> Result<()> {
    let strong_calls = Shared::default();
    let weak_calls = Shared::default();
    let events = Shared::default();
    let backend = MultiLlmBackend::new([
        target_backend(
            "strong-target",
            "strong-model",
            RecordingBackend::new("strong", strong_calls.clone(), events.clone()),
        ),
        target_backend(
            "weak-target",
            "weak-model",
            RecordingBackend::new("weak", weak_calls.clone(), events),
        ),
    ])?
    .with_default_target(LlmTargetId::from_static("strong-target"))?;
    let mut ctx = ProxyContext::new();
    ctx.set_selected_target(LlmTargetId::from_static("weak-target"));

    backend.call(&mut ctx, &request("client-model")).await?;

    assert_eq!(
        selection(&ctx)?.reason,
        BackendSelectionReason::ContextTarget
    );
    assert!(strong_calls.values()?.is_empty());
    assert_eq!(weak_calls.values()?.len(), 1);
    Ok(())
}

// Ambiguous model-less requests should fail before any backend sees them.
#[tokio::test]
async fn model_less_request_with_multiple_targets_fails_without_mutating_context() -> Result<()> {
    let calls = Shared::default();
    let events = Shared::default();
    let backend = MultiLlmBackend::new([
        target_backend(
            "left-target",
            "left-model",
            RecordingBackend::new("left", calls.clone(), events.clone()),
        ),
        target_backend(
            "right-target",
            "right-model",
            RecordingBackend::new("right", calls.clone(), events),
        ),
    ])?;
    let mut ctx = ProxyContext::new();
    let request = ChatRequest::openai_chat(json!({
        "messages": [{"role": "user", "content": "no model"}]
    }));

    let Err(error) = backend.call(&mut ctx, &request).await else {
        return Err(SwitchyardError::Other(
            "model-less multi-target request should fail".to_string(),
        ));
    };

    assert!(matches!(error, SwitchyardError::InvalidConfig(_)));
    assert_eq!(selected_target(&ctx), None);
    assert_eq!(selected_model(&ctx), None);
    assert!(ctx.get::<BackendSelection>().is_none());
    assert!(calls.values()?.is_empty());
    Ok(())
}

// Duplicate model names are legal when an explicit target disambiguates them.
#[tokio::test]
async fn duplicate_models_route_successfully_when_target_is_explicit() -> Result<()> {
    let left_calls = Shared::default();
    let right_calls = Shared::default();
    let events = Shared::default();
    let backend = MultiLlmBackend::new([
        target_backend(
            "left-target",
            "shared-model",
            RecordingBackend::new("left", left_calls.clone(), events.clone()),
        ),
        target_backend(
            "right-target",
            "shared-model",
            RecordingBackend::new("right", right_calls.clone(), events),
        ),
    ])?;
    let mut ctx = ProxyContext::new();
    ctx.set_selected_target(LlmTargetId::from_static("right-target"));

    backend.call(&mut ctx, &request("shared-model")).await?;

    assert!(left_calls.values()?.is_empty());
    let right = right_calls.values()?;
    assert_eq!(right.len(), 1);
    assert_eq!(right[0].backend_name, "right");
    assert_eq!(
        selection(&ctx)?.reason,
        BackendSelectionReason::ContextTarget
    );
    assert_eq!(
        selection(&ctx)?.target_id,
        Some(LlmTargetId::from_static("right-target"))
    );
    Ok(())
}

// Successful explicit routing should replace stale context selection before delegation.
#[tokio::test]
async fn explicit_target_overwrites_stale_selected_model_and_selection_before_delegation(
) -> Result<()> {
    let calls = Shared::default();
    let events = Shared::default();
    let backend = MultiLlmBackend::new([target_backend(
        "fresh-target",
        "fresh-model",
        RecordingBackend::new("fresh", calls.clone(), events),
    )])?;
    let mut ctx = ProxyContext::new();
    ctx.set_selected_target(LlmTargetId::from_static("fresh-target"));
    let _ = ctx.insert(BackendSelection {
        target_id: Some(LlmTargetId::from_static("stale-target")),
        model: ModelId::from_static("stale-model"),
        original_model: Some("stale-client".to_string()),
        reason: BackendSelectionReason::RequestModel,
    });

    backend.call(&mut ctx, &request("client-model")).await?;

    assert_eq!(
        selected_model(&ctx),
        Some(&ModelId::from_static("fresh-model"))
    );
    let calls = calls.values()?;
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0].ctx_selected_model,
        Some(ModelId::from_static("fresh-model"))
    );
    let selection = selection(&ctx)?;
    assert_eq!(
        selection.target_id,
        Some(LlmTargetId::from_static("fresh-target"))
    );
    assert_eq!(selection.model, ModelId::from_static("fresh-model"));
    assert_eq!(selection.original_model.as_deref(), Some("client-model"));
    assert_eq!(selection.reason, BackendSelectionReason::ContextTarget);
    Ok(())
}

// Unknown selected targets should fail before delegation.
#[tokio::test]
async fn unknown_selected_target_fails_before_any_backend_call() -> Result<()> {
    let calls = Shared::default();
    let events = Shared::default();
    let backend = MultiLlmBackend::new([target_backend(
        "known-target",
        "known-model",
        RecordingBackend::new("known", calls.clone(), events),
    )])?;
    let mut ctx = ProxyContext::new();
    ctx.set_selected_target(LlmTargetId::from_static("missing-target"));

    let Err(error) = backend.call(&mut ctx, &request("known-model")).await else {
        return Err(SwitchyardError::Other(
            "unknown selected target should fail".to_string(),
        ));
    };

    assert!(matches!(error, SwitchyardError::InvalidConfig(_)));
    assert!(calls.values()?.is_empty());
    Ok(())
}

// Failed routing should leave any existing context selection untouched.
#[tokio::test]
async fn failed_target_selection_does_not_replace_existing_context_selection() -> Result<()> {
    let calls = Shared::default();
    let events = Shared::default();
    let backend = MultiLlmBackend::new([target_backend(
        "known-target",
        "known-model",
        RecordingBackend::new("known", calls.clone(), events),
    )])?;
    let mut ctx = ProxyContext::new();
    ctx.set_selected_target(LlmTargetId::from_static("missing-target"));
    let stale = BackendSelection {
        target_id: Some(LlmTargetId::from_static("stale-target")),
        model: ModelId::from_static("stale-model"),
        original_model: Some("stale-client-model".to_string()),
        reason: BackendSelectionReason::RequestModel,
    };
    let _ = ctx.insert(stale.clone());

    let Err(error) = backend.call(&mut ctx, &request("known-model")).await else {
        return Err(SwitchyardError::Other(
            "unknown selected target should fail".to_string(),
        ));
    };

    assert!(matches!(error, SwitchyardError::InvalidConfig(_)));
    assert_eq!(ctx.get::<BackendSelection>(), Some(&stale));
    assert!(calls.values()?.is_empty());
    Ok(())
}

// Multiple possible targets without a unique selector should be rejected.
#[tokio::test]
async fn multiple_targets_without_a_unique_selection_are_rejected() -> Result<()> {
    let calls = Shared::default();
    let events = Shared::default();
    let backend = MultiLlmBackend::new([
        target_backend(
            "left-target",
            "left-model",
            RecordingBackend::new("left", calls.clone(), events.clone()),
        ),
        target_backend(
            "right-target",
            "right-model",
            RecordingBackend::new("right", calls.clone(), events),
        ),
    ])?;
    let mut ctx = ProxyContext::new();

    let Err(error) = backend.call(&mut ctx, &request("client-model")).await else {
        return Err(SwitchyardError::Other(
            "ambiguous request should fail".to_string(),
        ));
    };

    assert!(matches!(error, SwitchyardError::InvalidConfig(_)));
    assert!(calls.values()?.is_empty());
    Ok(())
}

// Duplicate model names require target-level disambiguation.
#[tokio::test]
async fn duplicate_models_require_explicit_target_selection() -> Result<()> {
    let calls = Shared::default();
    let events = Shared::default();
    let backend = MultiLlmBackend::new([
        target_backend(
            "left-target",
            "shared-model",
            RecordingBackend::new("left", calls.clone(), events.clone()),
        ),
        target_backend(
            "right-target",
            "shared-model",
            RecordingBackend::new("right", calls.clone(), events),
        ),
    ])?;
    let mut ctx = ProxyContext::new();

    let Err(error) = backend.call(&mut ctx, &request("shared-model")).await else {
        return Err(SwitchyardError::Other(
            "duplicate model selection should fail".to_string(),
        ));
    };

    assert!(matches!(error, SwitchyardError::InvalidConfig(_)));
    assert!(calls.values()?.is_empty());
    Ok(())
}

// Successful calls should replace stale typed backend-selection metadata.
#[tokio::test]
async fn successful_call_replaces_stale_typed_selection_extension() -> Result<()> {
    let calls = Shared::default();
    let events = Shared::default();
    let backend = MultiLlmBackend::new([target_backend(
        "only-target",
        "fresh-model",
        RecordingBackend::new("only", calls, events),
    )])?;
    let mut ctx = ProxyContext::new();
    let _ = ctx.insert(BackendSelection {
        target_id: Some(LlmTargetId::from_static("stale-target")),
        model: ModelId::from_static("stale-model"),
        original_model: None,
        reason: BackendSelectionReason::RequestModel,
    });

    backend.call(&mut ctx, &request("client-model")).await?;

    let selection = selection(&ctx)?;
    assert_eq!(
        selection.target_id,
        Some(LlmTargetId::from_static("only-target"))
    );
    assert_eq!(selection.model, ModelId::from_static("fresh-model"));
    assert_eq!(selection.original_model.as_deref(), Some("client-model"));
    assert_eq!(selection.reason, BackendSelectionReason::SingleTarget);
    Ok(())
}

// Unsupported request types should fail before child backend delegation.
#[tokio::test]
async fn unsupported_request_type_is_rejected_before_delegation() -> Result<()> {
    let calls = Shared::default();
    let events = Shared::default();
    let backend = MultiLlmBackend::new([target_backend(
        "only-target",
        "served-model",
        RecordingBackend::new("only", calls.clone(), events),
    )])?
    .with_supported_request_types([ChatRequestType::OpenAiChat])?;
    let mut ctx = ProxyContext::new();
    let request = ChatRequest::anthropic(json!({
        "model": "served-model",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "hello"}]
    }));

    let Err(error) = backend.call(&mut ctx, &request).await else {
        return Err(SwitchyardError::Other(
            "unsupported request type should fail".to_string(),
        ));
    };

    assert!(matches!(
        error,
        SwitchyardError::UnsupportedRequestType { .. }
    ));
    assert!(calls.values()?.is_empty());
    Ok(())
}

// Child backend errors should propagate without trying other targets.
#[tokio::test]
async fn selected_backend_error_propagates_and_does_not_try_fallback_target() -> Result<()> {
    let first_calls = Shared::default();
    let second_calls = Shared::default();
    let events = Shared::default();
    let backend = MultiLlmBackend::new([
        target_backend(
            "first-target",
            "first-model",
            RecordingBackend::new("first", first_calls.clone(), events.clone())
                .with_call_error("upstream exploded"),
        ),
        target_backend(
            "second-target",
            "second-model",
            RecordingBackend::new("second", second_calls.clone(), events),
        ),
    ])?;
    let mut ctx = ProxyContext::new();
    ctx.set_selected_target(LlmTargetId::from_static("first-target"));

    let Err(error) = backend.call(&mut ctx, &request("client-model")).await else {
        return Err(SwitchyardError::Other(
            "selected backend error should propagate".to_string(),
        ));
    };

    assert!(matches!(error, SwitchyardError::Backend(message) if message == "upstream exploded"));
    let first_calls = first_calls.values()?;
    assert_eq!(first_calls.len(), 1);
    assert_eq!(first_calls[0].model.as_deref(), Some("first-model"));
    assert!(second_calls.values()?.is_empty());
    assert_eq!(
        selected_target(&ctx),
        Some(&LlmTargetId::from_static("first-target"))
    );
    assert_eq!(
        selected_model(&ctx),
        Some(&ModelId::from_static("first-model"))
    );
    assert_eq!(
        selection(&ctx)?.reason,
        BackendSelectionReason::ContextTarget
    );
    Ok(())
}

// Startup rollback should keep shutting down started backends even if rollback fails.
#[tokio::test]
async fn startup_failure_rolls_back_all_started_backends_even_when_shutdowns_fail() -> Result<()> {
    let calls = Shared::default();
    let events = Shared::default();
    let backend = MultiLlmBackend::new([
        target_backend(
            "first-target",
            "first-model",
            RecordingBackend::new("first", calls.clone(), events.clone())
                .with_shutdown_error("first rollback failed"),
        ),
        target_backend(
            "second-target",
            "second-model",
            RecordingBackend::new("second", calls.clone(), events.clone())
                .with_shutdown_error("second rollback failed"),
        ),
        target_backend(
            "third-target",
            "third-model",
            RecordingBackend::new("third", calls, events.clone())
                .with_startup_error("third startup failed"),
        ),
    ])?;

    let Err(error) = backend.startup().await else {
        return Err(SwitchyardError::Other("startup should fail".to_string()));
    };

    assert!(
        matches!(error, SwitchyardError::Backend(message) if message == "third startup failed")
    );
    assert_eq!(
        events.values()?,
        vec![
            "first:startup".to_string(),
            "second:startup".to_string(),
            "third:startup".to_string(),
            "second:shutdown".to_string(),
            "first:shutdown".to_string(),
        ]
    );
    Ok(())
}

// Normal lifecycle should start forward and shut down in reverse order.
#[tokio::test]
async fn successful_startup_and_shutdown_use_forward_then_reverse_order() -> Result<()> {
    let calls = Shared::default();
    let events = Shared::default();
    let backend = MultiLlmBackend::new([
        target_backend(
            "first-target",
            "first-model",
            RecordingBackend::new("first", calls.clone(), events.clone()),
        ),
        target_backend(
            "second-target",
            "second-model",
            RecordingBackend::new("second", calls, events.clone()),
        ),
    ])?;

    backend.startup().await?;
    backend.shutdown().await?;

    assert_eq!(
        events.values()?,
        vec![
            "first:startup".to_string(),
            "second:startup".to_string(),
            "second:shutdown".to_string(),
            "first:shutdown".to_string(),
        ]
    );
    Ok(())
}

// Startup failure should roll back only the backends that actually started.
#[tokio::test]
async fn startup_rolls_back_already_started_backends() -> Result<()> {
    let calls = Shared::default();
    let events = Shared::default();
    let backend = MultiLlmBackend::new([
        target_backend(
            "first-target",
            "first-model",
            RecordingBackend::new("first", calls.clone(), events.clone()),
        ),
        target_backend(
            "second-target",
            "second-model",
            RecordingBackend::new("second", calls.clone(), events.clone())
                .with_startup_error("startup failed"),
        ),
        target_backend(
            "third-target",
            "third-model",
            RecordingBackend::new("third", calls, events.clone()),
        ),
    ])?;

    let Err(error) = backend.startup().await else {
        return Err(SwitchyardError::Other("startup should fail".to_string()));
    };

    assert!(matches!(error, SwitchyardError::Backend(_)));
    assert_eq!(
        events.values()?,
        vec![
            "first:startup".to_string(),
            "second:startup".to_string(),
            "first:shutdown".to_string(),
        ]
    );
    Ok(())
}

// Shutdown should run every backend in reverse and return the first observed failure.
#[tokio::test]
async fn shutdown_runs_all_backends_in_reverse_and_returns_first_error() -> Result<()> {
    let calls = Shared::default();
    let events = Shared::default();
    let backend = MultiLlmBackend::new([
        target_backend(
            "first-target",
            "first-model",
            RecordingBackend::new("first", calls.clone(), events.clone())
                .with_shutdown_error("first shutdown failed"),
        ),
        target_backend(
            "second-target",
            "second-model",
            RecordingBackend::new("second", calls, events.clone())
                .with_shutdown_error("second shutdown failed"),
        ),
    ])?;

    let Err(error) = backend.shutdown().await else {
        return Err(SwitchyardError::Other("shutdown should fail".to_string()));
    };

    assert!(
        matches!(error, SwitchyardError::Backend(message) if message == "second shutdown failed")
    );
    assert_eq!(
        events.values()?,
        vec!["second:shutdown".to_string(), "first:shutdown".to_string()]
    );
    Ok(())
}

// Public accessors should expose targets without coupling tests to storage internals.
#[test]
fn public_accessors_return_targets_and_backends_without_leaking_storage_shape() -> Result<()> {
    let calls = Shared::default();
    let events = Shared::default();
    let backend = MultiLlmBackend::new([
        target_backend(
            "first-target",
            "first-model",
            RecordingBackend::new("first", calls.clone(), events.clone()),
        ),
        target_backend(
            "second-target",
            "second-model",
            RecordingBackend::new("second", calls, events),
        ),
    ])?;

    assert_eq!(backend.targets().len(), 2);
    assert_eq!(
        backend.targets()[0].target().id,
        LlmTargetId::from_static("first-target")
    );
    assert_eq!(
        backend
            .target(&LlmTargetId::from_static("second-target"))
            .ok_or_else(|| SwitchyardError::Other("second target should exist".to_string()))?
            .target()
            .model,
        ModelId::from_static("second-model")
    );
    assert!(backend
        .target(&LlmTargetId::from_static("missing-target"))
        .is_none());
    assert_eq!(
        backend.targets()[0].backend().supported_request_types(),
        &ALL_REQUEST_TYPES
    );
    Ok(())
}

// Constructor validation should reject empty target lists and duplicate IDs.
#[test]
fn configuration_rejects_empty_targets_duplicate_ids_and_empty_supported_types() -> Result<()> {
    let calls = Shared::default();
    let events = Shared::default();

    let Err(empty_targets) = MultiLlmBackend::new([]) else {
        return Err(SwitchyardError::Other(
            "empty target list should fail".to_string(),
        ));
    };
    assert!(matches!(empty_targets, SwitchyardError::InvalidConfig(_)));

    let Err(duplicate_ids) = MultiLlmBackend::new([
        target_backend(
            "same-target",
            "left-model",
            RecordingBackend::new("left", calls.clone(), events.clone()),
        ),
        target_backend(
            "same-target",
            "right-model",
            RecordingBackend::new("right", calls.clone(), events.clone()),
        ),
    ]) else {
        return Err(SwitchyardError::Other(
            "duplicate target IDs should fail".to_string(),
        ));
    };
    assert!(matches!(duplicate_ids, SwitchyardError::InvalidConfig(_)));

    let backend = MultiLlmBackend::new([target_backend(
        "only-target",
        "only-model",
        RecordingBackend::new("only", calls, events),
    )])?;
    let Err(invalid_default) = backend
        .clone()
        .with_default_target(LlmTargetId::from_static("missing-target"))
    else {
        return Err(SwitchyardError::Other(
            "unknown default target should fail".to_string(),
        ));
    };
    assert!(matches!(invalid_default, SwitchyardError::InvalidConfig(_)));
    let Err(empty_supported_types) = backend.with_supported_request_types([]) else {
        return Err(SwitchyardError::Other(
            "empty supported request types should fail".to_string(),
        ));
    };
    assert!(matches!(
        empty_supported_types,
        SwitchyardError::InvalidConfig(_)
    ));
    Ok(())
}
