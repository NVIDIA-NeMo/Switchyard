// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Intake payload construction.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Map, Value};
use switchyard_core::{
    ChatRequest, ChatRequestType, ChatResponse, ChatResponseType, ProxyContext, Result,
    StreamEvent, SwitchyardError,
};
use switchyard_translation::{TranslationEngine, TranslationPolicy, WireFormat};

use crate::backends::BackendSelection;
use crate::intake::config::{IntakeFormat, IntakeSinkConfig};
use crate::intake::context::{IntakeRequestState, RequestMetadata, SubModelCall};
use crate::request_processors::RandomRoutingDecision;
use crate::stats::{estimate_model_cost, usage_from_body, StatsRouteLabel};
use crate::telemetry::switchyard_version;

/// Placeholder model label used when no backend selected model is available.
pub const UNKNOWN_MODEL: &str = "unknown";
/// Synthetic stream IDs should not be used as external intake IDs.
pub const SYNTHETIC_STREAM_RESPONSE_IDS: &[&str] = &[
    "chatcmpl-intake-stream",
    "chatcmpl-switchyard-stream",
    "msg_switchyard_stream",
    "resp_switchyard_stream",
];

/// Snapshot of request context needed to construct one intake payload.
#[derive(Clone, Debug)]
pub struct IntakePayloadContext {
    /// Request metadata extracted from headers or typed context.
    pub request_metadata: RequestMetadata,
    /// Request start timestamp in milliseconds.
    pub started_at_ms: Option<i64>,
    /// Original inbound wire format.
    pub inbound_format: Option<ChatRequestType>,
    /// Client session ID, if available.
    pub session_id: Option<String>,
    /// Request snapshot captured by `IntakeRequestProcessor`.
    pub request_snapshot: Option<ChatRequest>,
    /// Backend-selected served model, if known.
    pub served_model: Option<String>,
    /// Routing label to include in the intake payload.
    pub routing: Option<IntakeRoutingMetadata>,
    /// Response end timestamp in milliseconds.
    pub ended_at_ms: Option<i64>,
}

impl IntakePayloadContext {
    /// Extracts the intake payload context from the typed proxy context.
    pub fn from_proxy_context(ctx: &ProxyContext, ended_at_ms: Option<i64>) -> Self {
        let request_state = ctx.get::<IntakeRequestState>();
        Self {
            request_metadata: ctx.get::<RequestMetadata>().cloned().unwrap_or_default(),
            started_at_ms: request_state.map(|state| state.started_at_ms),
            inbound_format: request_state.map(|state| state.inbound_format),
            session_id: request_state.and_then(|state| state.session_id.clone()),
            request_snapshot: request_state.and_then(|state| state.request_snapshot.clone()),
            served_model: ctx
                .get::<BackendSelection>()
                .map(|selection| selection.model.as_str().to_string())
                .filter(|model| !model.is_empty()),
            routing: routing_from_context(ctx),
            ended_at_ms,
        }
    }
}

/// Minimal routing metadata attached to intake payloads.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IntakeRoutingMetadata {
    /// Router family that produced the decision.
    pub router_type: String,
    /// Tier or route label selected by the router.
    pub routed_to: String,
}

/// Builds Intake chat-completions payloads from normalized request/response data.
#[derive(Clone)]
pub struct IntakePayloadBuilder {
    /// Sink config values needed in the payload context block.
    config: IntakeSinkConfig,
    /// Translation engine used to normalize all formats to OpenAI Chat.
    translation: Arc<TranslationEngine>,
    /// Translation policy used for request and response normalization.
    policy: TranslationPolicy,
}

impl IntakePayloadBuilder {
    /// Creates a payload builder for one intake sink config.
    pub fn new(config: IntakeSinkConfig) -> Self {
        Self {
            config,
            translation: Arc::new(TranslationEngine::default()),
            policy: TranslationPolicy::default(),
        }
    }

    /// Returns the intake sink config backing this builder.
    pub fn config(&self) -> &IntakeSinkConfig {
        &self.config
    }

    /// Returns the request snapshot or a processor error if intake state is missing.
    pub fn request_from_state<'a>(&self, ctx: &'a IntakePayloadContext) -> Result<&'a ChatRequest> {
        ctx.request_snapshot.as_ref().ok_or_else(|| {
            SwitchyardError::Processor("missing intake request snapshot in context".to_string())
        })
    }

    /// Builds a payload from native Switchyard request and response values.
    pub fn build(
        &self,
        ctx: &IntakePayloadContext,
        request_snapshot: &ChatRequest,
        response: &ChatResponse,
        stream: bool,
    ) -> Result<Value> {
        let openai_request = self.translate_request_to_openai(request_snapshot)?;
        let openai_response = self.translate_response_to_openai(response)?;
        self.build_from_openai(ctx, openai_request, openai_response, stream, None)
    }

    /// Builds a payload from an already-normalized OpenAI Chat response body.
    pub fn build_from_openai_response_body(
        &self,
        ctx: &IntakePayloadContext,
        request_snapshot: &ChatRequest,
        openai_response: Value,
        stream: bool,
    ) -> Result<Value> {
        let openai_request = self.translate_request_to_openai(request_snapshot)?;
        self.build_from_openai(ctx, openai_request, openai_response, stream, None)
    }

    /// Builds an anonymous intake record for one routing-strategy sub-model call.
    ///
    /// The record carries only the router's own model, token usage, and route
    /// label — never message content. It reuses the turn's session, task, and
    /// timestamps but overrides the routing label with this call's own. `index`
    /// yields a per-call id suffix so several calls in one turn stay distinct as
    /// separate flat documents instead of overwriting one another.
    pub fn build_submodel_record(
        &self,
        ctx: &IntakePayloadContext,
        call: &SubModelCall,
        index: usize,
    ) -> Result<Value> {
        let openai_request = json!({ "model": call.model });
        let openai_response = json!({
            "model": call.model,
            "object": "chat.completion",
            "choices": [],
            "usage": {
                "prompt_tokens": call.prompt_tokens,
                "completion_tokens": call.completion_tokens,
                "total_tokens": call.prompt_tokens.saturating_add(call.completion_tokens),
                "prompt_tokens_details": { "cached_tokens": call.cached_tokens },
            },
        });
        let mut call_ctx = ctx.clone();
        call_ctx.routing = Some(IntakeRoutingMetadata {
            router_type: call.router_type.clone(),
            routed_to: call.routed_to.clone(),
        });
        // A routing call's latency isn't the turn's, so drop the start stamp:
        // no misleading latency_ms is derived and created_at falls back to the
        // turn end (≈ now), matching the Python record this replaces.
        call_ctx.started_at_ms = None;
        let discriminator = format!("{}-{index}", call.routed_to);
        self.build_from_openai(
            &call_ctx,
            openai_request,
            openai_response,
            false,
            Some(&discriminator),
        )
    }

    /// Assembles the final intake API JSON payload.
    ///
    /// `id_discriminator` makes a flat document's `_id` unique among
    /// records built for the same turn (same session + timestamp); the primary
    /// per-turn record passes `None`.
    fn build_from_openai(
        &self,
        ctx: &IntakePayloadContext,
        openai_request: Value,
        openai_response: Value,
        stream: bool,
        id_discriminator: Option<&str>,
    ) -> Result<Value> {
        let mut request_entry = object_from_value(openai_request, "OpenAI intake request")?;
        normalize_logged_stream_request(&mut request_entry, stream);
        request_entry.insert(
            "switchyard".to_string(),
            Value::Object(self.switchyard_request_metadata(ctx, stream)),
        );
        let mut response_entry = object_from_value(openai_response, "OpenAI intake response")?;
        strip_synthetic_response_id(&mut response_entry);

        // Metadata-only unless content capture is explicitly enabled.
        if !self.config.capture_content {
            redact_prompt_content(&mut request_entry, &mut response_entry);
        }

        let session_id = ctx
            .session_id
            .clone()
            .filter(|session_id| !session_id.is_empty());

        let mut payload = Map::new();
        payload.insert("request".to_string(), Value::Object(request_entry));
        payload.insert("response".to_string(), Value::Object(response_entry));
        add_cost_fields(&mut payload);
        if let Some(evaluation_context) = self.evaluation_context(ctx) {
            payload.insert(
                "evaluation_context".to_string(),
                Value::Object(evaluation_context),
            );
        }
        if let Some(session_id) = session_id {
            payload.insert("session_id".to_string(), Value::String(session_id));
        }
        payload.insert(
            "provider".to_string(),
            Value::String("switchyard".to_string()),
        );
        let payload = Value::Object(payload);
        let flat = self
            .config()
            .target
            .as_ref()
            .is_some_and(|target| target.format == IntakeFormat::FlatDocument);
        if flat {
            return Ok(to_flat_document(
                &payload,
                ctx.ended_at_ms,
                id_discriminator,
            ));
        }
        Ok(payload)
    }

    /// Normalizes the captured request into OpenAI Chat payload shape.
    fn translate_request_to_openai(&self, request: &ChatRequest) -> Result<Value> {
        match request.request_type() {
            ChatRequestType::OpenAiChat => Ok(request.body().clone()),
            source => self
                .translation
                .translate_request(
                    request_wire_format(source),
                    WireFormat::OpenAiChat,
                    request.body(),
                    &self.policy,
                )
                .map(|output| output.body)
                .map_err(|error| {
                    SwitchyardError::Processor(format!(
                        "failed to translate intake request to OpenAI Chat: {error}"
                    ))
                }),
        }
    }

    /// Normalizes buffered responses into OpenAI Chat completion shape.
    fn translate_response_to_openai(&self, response: &ChatResponse) -> Result<Value> {
        let Some(body) = response.body() else {
            return Err(SwitchyardError::Processor(
                "intake payload builder requires a buffered response".to_string(),
            ));
        };
        match response.response_type() {
            ChatResponseType::OpenAiCompletion => Ok(body.clone()),
            ChatResponseType::OpenAiResponsesCompletion => self
                .translation
                .translate_response(
                    WireFormat::OpenAiResponses,
                    WireFormat::OpenAiChat,
                    body,
                    &self.policy,
                )
                .map(|output| output.body)
                .map_err(|error| {
                    SwitchyardError::Processor(format!(
                        "failed to translate Responses intake response to OpenAI Chat: {error}"
                    ))
                }),
            ChatResponseType::AnthropicCompletion => self
                .translation
                .translate_response(
                    WireFormat::AnthropicMessages,
                    WireFormat::OpenAiChat,
                    body,
                    &self.policy,
                )
                .map(|output| output.body)
                .map_err(|error| {
                    SwitchyardError::Processor(format!(
                        "failed to translate Anthropic intake response to OpenAI Chat: {error}"
                    ))
                }),
            ChatResponseType::OpenAiStream
            | ChatResponseType::OpenAiResponsesStream
            | ChatResponseType::AnthropicStream => Err(SwitchyardError::Processor(
                "intake payload builder requires a buffered response".to_string(),
            )),
        }
    }

    /// Adds Switchyard-specific request metadata under the request body.
    fn switchyard_request_metadata(
        &self,
        ctx: &IntakePayloadContext,
        stream: bool,
    ) -> Map<String, Value> {
        let mut metadata = Map::new();
        metadata.insert("version".to_string(), Value::String(switchyard_version()));
        metadata.insert(
            "inbound_format".to_string(),
            ctx.inbound_format
                .map(|request_type| Value::String(request_type_value(request_type).to_string()))
                .unwrap_or(Value::Null),
        );
        metadata.insert("stream".to_string(), Value::Bool(stream));
        metadata.insert(
            "user_id".to_string(),
            Value::String(self.config.user_id.clone()),
        );
        if let Some(session_id) = ctx
            .session_id
            .as_deref()
            .filter(|session_id| !session_id.is_empty())
        {
            metadata.insert(
                "session_id".to_string(),
                Value::String(session_id.to_string()),
            );
        }
        metadata.insert(
            "created_at".to_string(),
            Value::String(created_at_iso(ctx.started_at_ms, ctx.ended_at_ms)),
        );
        if let Some(latency_ms) = ctx
            .started_at_ms
            .zip(ctx.ended_at_ms)
            .map(|(started, ended)| ended - started)
        {
            metadata.insert("latency_ms".to_string(), Value::from(latency_ms));
        }
        if let Some(routing) = &ctx.routing {
            metadata.insert(
                "routing".to_string(),
                json!({
                    "router_type": routing.router_type.as_str(),
                    "routed_to": routing.routed_to.as_str(),
                }),
            );
        }
        metadata
    }

    /// Returns the intake task name, defaulting to chat.
    fn task_name(&self, ctx: &IntakePayloadContext) -> String {
        ctx.request_metadata
            .intake
            .task
            .clone()
            .unwrap_or_else(|| "chat".to_string())
    }

    /// Builds top-level Intake evaluation context from request labels.
    fn evaluation_context(&self, ctx: &IntakePayloadContext) -> Option<Map<String, Value>> {
        let evaluation_run_id = ctx
            .session_id
            .as_deref()
            .filter(|s| !s.is_empty())?
            .to_string();
        let test_case_id = self.task_name(ctx);
        let mut evaluation_context = Map::new();
        evaluation_context.insert(
            "evaluation_run_id".to_string(),
            Value::String(evaluation_run_id),
        );
        evaluation_context.insert("test_case_id".to_string(), Value::String(test_case_id));
        Some(evaluation_context)
    }
}

/// Returns current wall-clock time in milliseconds since Unix epoch.
pub fn now_millis() -> i64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => {
            let millis = duration.as_millis();
            if millis > i64::MAX as u128 {
                i64::MAX
            } else {
                millis as i64
            }
        }
        Err(_) => 0,
    }
}

/// Returns the stable intake string value for a request wire type.
pub fn request_type_value(request_type: ChatRequestType) -> &'static str {
    match request_type {
        ChatRequestType::OpenAiChat => "openai_chat",
        ChatRequestType::OpenAiResponses => "openai_responses",
        ChatRequestType::Anthropic => "anthropic",
    }
}

/// Stream format captured by intake.
#[derive(Clone, Copy, Debug)]
pub enum IntakeStreamFormat {
    /// OpenAI Chat Completions stream.
    OpenAiChat,
    /// OpenAI Responses API stream.
    OpenAiResponses,
    /// Anthropic Messages stream.
    Anthropic,
}

/// Format-specific stream capture state.
pub enum IntakeStreamCapture {
    /// OpenAI Chat stream capture.
    OpenAiChat(OpenAiChatStreamCapture),
    /// OpenAI Responses stream capture.
    OpenAiResponses(ResponsesStreamCapture),
    /// Anthropic stream capture.
    Anthropic(AnthropicStreamCapture),
}

impl IntakeStreamCapture {
    /// Creates a capture adapter for the outbound stream format.
    pub fn new(format: IntakeStreamFormat, served_model: Option<&str>) -> Self {
        match format {
            IntakeStreamFormat::OpenAiChat => {
                Self::OpenAiChat(OpenAiChatStreamCapture::new(served_model))
            }
            IntakeStreamFormat::OpenAiResponses => {
                Self::OpenAiResponses(ResponsesStreamCapture::new(served_model))
            }
            IntakeStreamFormat::Anthropic => {
                Self::Anthropic(AnthropicStreamCapture::new(served_model))
            }
        }
    }

    /// Observes one stream event while preserving the caller-visible stream.
    pub fn observe(&mut self, event: &StreamEvent) {
        match self {
            Self::OpenAiChat(capture) => capture.observe(event),
            Self::OpenAiResponses(capture) => capture.observe(event),
            Self::Anthropic(capture) => capture.observe(event),
        }
    }

    /// Converts captured stream state into an OpenAI Chat completion body.
    pub fn finish(self) -> Result<Value> {
        match self {
            Self::OpenAiChat(capture) => Ok(capture.finish()),
            Self::OpenAiResponses(capture) => capture.finish(),
            Self::Anthropic(capture) => Ok(capture.finish()),
        }
    }
}

/// Reconstructs an OpenAI Chat completion from OpenAI stream chunks.
pub fn openai_chat_response_from_stream(
    events: &[StreamEvent],
    served_model: Option<&str>,
) -> Value {
    let mut capture = OpenAiChatStreamCapture::new(served_model);
    for event in events {
        capture.observe(event);
    }
    capture.finish()
}

/// Reconstructs an OpenAI Chat completion from Anthropic stream events.
pub fn anthropic_response_from_stream(events: &[StreamEvent], served_model: Option<&str>) -> Value {
    let mut capture = AnthropicStreamCapture::new(served_model);
    for event in events {
        capture.observe(event);
    }
    capture.finish()
}

/// Reconstructs an OpenAI Chat completion from Responses stream events.
pub fn responses_response_from_stream(
    events: &[StreamEvent],
    served_model: Option<&str>,
) -> Result<Value> {
    let mut capture = ResponsesStreamCapture::new(served_model);
    for event in events {
        capture.observe(event);
    }
    capture.finish()
}

/// Capture state for OpenAI Responses streaming.
pub struct ResponsesStreamCapture {
    /// Backend-selected model fallback.
    served_model: Option<String>,
    /// Response ID from the stream, if observed.
    response_id: Option<String>,
    /// Response creation timestamp from the stream, if observed.
    created_at: Option<f64>,
    /// Accumulated output text.
    content: String,
    /// Tool calls keyed by Responses output index.
    tool_calls: BTreeMap<usize, ResponsesToolCallState>,
    /// Usage block captured from response metadata.
    usage: Option<Value>,
    /// Full completed response when the stream provides one.
    completed_response: Option<Value>,
}

impl ResponsesStreamCapture {
    /// Creates empty Responses stream capture state.
    fn new(served_model: Option<&str>) -> Self {
        Self {
            served_model: served_model.map(str::to_string),
            response_id: None,
            created_at: None,
            content: String::new(),
            tool_calls: BTreeMap::new(),
            usage: None,
            completed_response: None,
        }
    }

    /// Merges one Responses stream event into the capture state.
    fn observe(&mut self, event: &StreamEvent) {
        let StreamEvent::Json(value) = event else {
            return;
        };
        if let Some(response) = value
            .get("response")
            .filter(|response| response.is_object())
        {
            self.capture_response_metadata(response);
        }

        match value.get("type").and_then(Value::as_str) {
            Some("response.output_text.delta") => {
                if let Some(delta) = value.get("delta").and_then(Value::as_str) {
                    self.content.push_str(delta);
                }
            }
            Some("response.output_item.added" | "response.output_item.done") => {
                self.merge_output_item(
                    value.get("item"),
                    coerce_usize(value.get("output_index")).unwrap_or(0),
                );
            }
            Some("response.function_call_arguments.delta") => {
                let tool_call = self
                    .tool_calls
                    .entry(coerce_usize(value.get("output_index")).unwrap_or(0))
                    .or_default();
                if let Some(delta) = value.get("delta").and_then(Value::as_str) {
                    tool_call.arguments.push_str(delta);
                }
            }
            Some("response.completed") => {
                if let Some(response) = value.get("response").cloned() {
                    self.completed_response = Some(response);
                }
            }
            _ => {}
        }
    }

    /// Captures response-level metadata from in-progress or completed events.
    fn capture_response_metadata(&mut self, response: &Value) {
        if let Some(response_id) = response.get("id").and_then(Value::as_str) {
            self.response_id = Some(response_id.to_string());
        }
        if let Some(model) = response.get("model").and_then(Value::as_str) {
            self.served_model = Some(model.to_string());
        }
        if let Some(created_at) = response
            .get("created_at")
            .or_else(|| response.get("created"))
            .and_then(Value::as_f64)
        {
            self.created_at = Some(created_at);
        }
        if let Some(usage) = response.get("usage").filter(|usage| usage.is_object()) {
            self.usage = Some(usage.clone());
        }
    }

    /// Merges a Responses output item into the final synthetic response.
    fn merge_output_item(&mut self, item: Option<&Value>, output_index: usize) {
        let Some(item) = item.and_then(Value::as_object) else {
            return;
        };
        match item.get("type").and_then(Value::as_str) {
            Some("message") => {
                let text = extract_responses_message_text(item.get("content"));
                if !text.is_empty() && self.content.is_empty() {
                    self.content = text;
                }
            }
            Some("function_call") => {
                let tool_call = self.tool_calls.entry(output_index).or_default();
                if let Some(id) = item
                    .get("id")
                    .or_else(|| item.get("call_id"))
                    .and_then(Value::as_str)
                {
                    tool_call.id = Some(id.to_string());
                }
                if let Some(call_id) = item.get("call_id").and_then(Value::as_str) {
                    tool_call.call_id = Some(call_id.to_string());
                }
                if let Some(name) = item.get("name").and_then(Value::as_str) {
                    tool_call.name = Some(name.to_string());
                }
                if let Some(arguments) = item.get("arguments").and_then(Value::as_str) {
                    tool_call.arguments = arguments.to_string();
                }
            }
            _ => {}
        }
    }

    /// Finishes capture and translates Responses output back to OpenAI Chat.
    fn finish(self) -> Result<Value> {
        let response =
            self.completed_response.unwrap_or_else(|| {
                let mut output = Vec::new();
                if !self.content.is_empty() {
                    output.push(json!({
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": self.content}],
                    }));
                }
                output.extend(self.tool_calls.into_iter().enumerate().filter_map(
                    |(fallback_index, (_, call))| call.into_response_item(fallback_index),
                ));
                let mut response = json!({
                    "id": self.response_id.unwrap_or_else(|| "resp_switchyard_stream".to_string()),
                    "object": "response",
                    "created_at": self.created_at.unwrap_or(0.0),
                    "status": "completed",
                    "model": self.served_model.unwrap_or_else(|| UNKNOWN_MODEL.to_string()),
                    "output": output,
                    "parallel_tool_calls": false,
                    "tool_choice": "auto",
                    "tools": [],
                });
                if let (Value::Object(object), Some(usage)) = (&mut response, self.usage) {
                    object.insert("usage".to_string(), usage);
                }
                response
            });

        TranslationEngine::default()
            .translate_response(
                WireFormat::OpenAiResponses,
                WireFormat::OpenAiChat,
                &response,
                &TranslationPolicy::default(),
            )
            .map(|output| output.body)
            .map_err(|error| {
                SwitchyardError::Processor(format!(
                    "failed to translate Responses stream for intake: {error}"
                ))
            })
    }
}

#[derive(Default)]
struct ResponsesToolCallState {
    /// Stable item ID when the stream provides one.
    id: Option<String>,
    /// Provider call ID when separate from item ID.
    call_id: Option<String>,
    /// Function name.
    name: Option<String>,
    /// Incrementally assembled JSON arguments string.
    arguments: String,
}

impl ResponsesToolCallState {
    /// Converts captured tool-call state into a Responses function_call item.
    fn into_response_item(self, fallback_index: usize) -> Option<Value> {
        (self.name.is_some() || !self.arguments.is_empty()).then(|| {
            let id = self
                .id
                .or_else(|| self.call_id.clone())
                .unwrap_or_else(|| format!("fc_switchyard_{fallback_index}"));
            json!({
                "type": "function_call",
                "id": id,
                "call_id": self.call_id.unwrap_or_else(|| format!("fc_switchyard_{fallback_index}")),
                "name": self.name.unwrap_or_default(),
                "arguments": self.arguments,
                "status": "completed",
            })
        })
    }
}

/// Converts Switchyard request type into translation wire format.
fn request_wire_format(request_type: ChatRequestType) -> WireFormat {
    match request_type {
        ChatRequestType::OpenAiChat => WireFormat::OpenAiChat,
        ChatRequestType::OpenAiResponses => WireFormat::OpenAiResponses,
        ChatRequestType::Anthropic => WireFormat::AnthropicMessages,
    }
}

/// Ensures a normalized request or response is a JSON object.
fn object_from_value(value: Value, label: &str) -> Result<Map<String, Value>> {
    match value {
        Value::Object(object) => Ok(object),
        other => Err(SwitchyardError::Processor(format!(
            "{label} must be a JSON object, got {other:?}"
        ))),
    }
}

/// Adds queryable cost fields for models with known Switchyard pricing.
fn add_cost_fields(payload: &mut Map<String, Value>) {
    let Some(response) = payload.get("response") else {
        return;
    };
    let Some(model) = response.get("model").and_then(Value::as_str) else {
        return;
    };
    let usage = usage_from_body(response);
    if usage.is_zero() {
        return;
    }
    let breakdown = estimate_model_cost(
        model,
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.cached_tokens,
        usage.cache_creation_tokens,
    );
    if breakdown.total_cost <= 0.0 {
        return;
    }
    payload.insert("cost_usd".to_string(), json!(breakdown.total_cost));
    payload.insert("cost_input_usd".to_string(), json!(breakdown.input_cost));
    payload.insert("cost_output_usd".to_string(), json!(breakdown.output_cost));
    payload.insert(
        "cost_details".to_string(),
        json!({
            "base_input": breakdown.base_input_cost,
            "cached_input": breakdown.cached_input_cost,
            "cache_write": breakdown.cache_write_cost,
        }),
    );
}

/// Logs completed stream responses as non-streaming chat-completion captures.
fn normalize_logged_stream_request(request_entry: &mut Map<String, Value>, stream: bool) {
    if stream || matches!(request_entry.get("stream"), Some(Value::Bool(true))) {
        request_entry.insert("stream".to_string(), Value::Bool(false));
    }
}

/// Synthetic stream response IDs would collide in Intake's span ID field.
fn strip_synthetic_response_id(response_entry: &mut Map<String, Value>) {
    let Some(response_id) = response_entry.get("id").and_then(Value::as_str) else {
        return;
    };
    if SYNTHETIC_STREAM_RESPONSE_IDS.contains(&response_id) {
        response_entry.remove("id");
    }
}

/// Strips prompt/response text so intake captures metadata only; model/usage stay.
fn redact_prompt_content(
    request_entry: &mut Map<String, Value>,
    response_entry: &mut Map<String, Value>,
) {
    for key in [
        "messages",
        "system",
        "prompt",
        "input",
        "tools",
        "tool_choice",
        "functions",
        "function_call",
    ] {
        request_entry.remove(key);
    }
    if let Some(Value::Array(choices)) = response_entry.get_mut("choices") {
        for choice in choices.iter_mut() {
            let Some(choice) = choice.as_object_mut() else {
                continue;
            };
            choice.remove("logprobs");
            for shape in ["message", "delta"] {
                if let Some(part) = choice.get_mut(shape).and_then(Value::as_object_mut) {
                    // Allowlist: keep only role; drop content and any other field.
                    part.retain(|key, _| key == "role");
                }
            }
        }
    }
}

/// Extracts routing metadata from typed context extensions.
fn routing_from_context(ctx: &ProxyContext) -> Option<IntakeRoutingMetadata> {
    if let Some(decision) = ctx.get::<RandomRoutingDecision>() {
        return Some(IntakeRoutingMetadata {
            router_type: "random".to_string(),
            routed_to: decision.tier.as_str().to_string(),
        });
    }
    if let Some(label) = ctx.get::<StatsRouteLabel>() {
        return route_label_metadata(label.0.trim());
    }
    None
}

/// Converts a non-empty route label into custom intake routing metadata.
fn route_label_metadata(label: &str) -> Option<IntakeRoutingMetadata> {
    (!label.is_empty()).then(|| IntakeRoutingMetadata {
        router_type: "custom".to_string(),
        routed_to: label.to_string(),
    })
}

/// Coerces JSON token counters into signed integers like Python did.
fn coerce_i64(value: Option<&Value>) -> Option<i64> {
    match value {
        Some(Value::Bool(value)) => Some(i64::from(*value)),
        Some(Value::Number(value)) => value
            .as_i64()
            .or_else(|| value.as_u64().and_then(|value| i64::try_from(value).ok()))
            .or_else(|| value.as_f64().map(|value| value as i64)),
        _ => None,
    }
}

/// Coerces JSON indexes into `usize` for stream reconstruction maps.
fn coerce_usize(value: Option<&Value>) -> Option<usize> {
    value
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

/// Converts streamed tool-call arguments to their string representation.
fn json_argument_string(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(value)) => value.clone(),
        Some(Value::Object(_)) => value
            .and_then(|value| serde_json::to_string(value).ok())
            .unwrap_or_default(),
        _ => String::new(),
    }
}

/// Pulls visible text out of a Responses message output item.
fn extract_responses_message_text(content: Option<&Value>) -> String {
    let Some(parts) = content.and_then(Value::as_array) else {
        return String::new();
    };
    let mut text = String::new();
    for part in parts {
        let Some(part) = part.as_object() else {
            continue;
        };
        if !matches!(
            part.get("type").and_then(Value::as_str),
            Some("output_text" | "text")
        ) {
            continue;
        }
        if let Some(part_text) = part.get("text").and_then(Value::as_str) {
            text.push_str(part_text);
        }
    }
    text
}

/// Chooses the OpenAI finish reason for synthesized chat completions.
fn default_openai_finish_reason(has_tools: bool) -> Value {
    Value::String(if has_tools { "tool_calls" } else { "stop" }.to_string())
}

/// Maps Anthropic stop reasons onto OpenAI Chat finish reasons.
fn anthropic_stop_reason_to_openai(reason: Value) -> Value {
    let Some(reason) = reason.as_str() else {
        return reason;
    };
    Value::String(
        match reason {
            "tool_use" => "tool_calls",
            "end_turn" => "stop",
            "max_tokens" => "length",
            other => other,
        }
        .to_string(),
    )
}

/// Flattens a chat-completions intake payload into a top-level, type-prefixed
/// (`s_`/`l_`/`f_`/`b_`/`text_`) flat document, since flat posting endpoints
/// only index top-level fields. The raw record is kept in
/// `text_switchyard_record_json`.
pub fn to_flat_document(
    payload: &Value,
    ts_created_ms: Option<i64>,
    id_discriminator: Option<&str>,
) -> Value {
    let request = payload.get("request");
    let response = payload.get("response");
    let switchyard = request.and_then(|value| value.get("switchyard"));
    let routing = switchyard.and_then(|value| value.get("routing"));
    let usage = response.and_then(|value| value.get("usage"));
    let session_id = payload.get("session_id").and_then(Value::as_str);
    let ts = ts_created_ms.unwrap_or_default();

    let mut doc = Map::new();
    let id = match session_id {
        Some(session_id) => format!("{session_id}-{ts}"),
        None => format!("switchyard-{ts}"),
    };
    // Several records can share one session + timestamp (the primary turn plus
    // each routing sub-model call); the discriminator keeps their `_id`s unique.
    let id = match id_discriminator {
        Some(discriminator) => format!("{id}-{discriminator}"),
        None => id,
    };
    doc.insert("_id".to_string(), Value::String(id));
    doc.insert("ts_created".to_string(), Value::from(ts));
    doc.insert(
        "s_source".to_string(),
        Value::String("switchyard".to_string()),
    );
    doc.insert(
        "s_record_type".to_string(),
        Value::String("switchyard_request".to_string()),
    );
    doc.insert("l_schema_version".to_string(), Value::from(1));

    insert_str(&mut doc, "s_switchyard_session_id", session_id);
    insert_str(
        &mut doc,
        "s_switchyard_user_id",
        switchyard
            .and_then(|value| value.get("user_id"))
            .and_then(Value::as_str),
    );
    insert_str(
        &mut doc,
        "s_switchyard_served_model",
        response
            .and_then(|value| value.get("model"))
            .and_then(Value::as_str),
    );
    insert_str(
        &mut doc,
        "s_switchyard_requested_model",
        request
            .and_then(|value| value.get("model"))
            .and_then(Value::as_str),
    );
    insert_str(
        &mut doc,
        "s_switchyard_inbound_format",
        switchyard
            .and_then(|value| value.get("inbound_format"))
            .and_then(Value::as_str),
    );
    insert_str(
        &mut doc,
        "s_switchyard_router_type",
        routing
            .and_then(|value| value.get("router_type"))
            .and_then(Value::as_str),
    );
    insert_str(
        &mut doc,
        "s_switchyard_routed_to",
        routing
            .and_then(|value| value.get("routed_to"))
            .and_then(Value::as_str),
    );
    doc.insert(
        "b_switchyard_routed".to_string(),
        Value::Bool(routing.is_some()),
    );

    insert_long(
        &mut doc,
        "l_switchyard_input_tokens",
        usage.and_then(|value| value.get("prompt_tokens")),
    );
    insert_long(
        &mut doc,
        "l_switchyard_output_tokens",
        usage.and_then(|value| value.get("completion_tokens")),
    );
    insert_long(
        &mut doc,
        "l_switchyard_total_tokens",
        usage.and_then(|value| value.get("total_tokens")),
    );
    insert_long(
        &mut doc,
        "l_switchyard_cached_tokens",
        usage
            .and_then(|value| value.get("prompt_tokens_details"))
            .and_then(|value| value.get("cached_tokens")),
    );
    insert_long(
        &mut doc,
        "l_switchyard_latency_ms",
        switchyard.and_then(|value| value.get("latency_ms")),
    );

    insert_float(&mut doc, "f_switchyard_cost_usd", payload.get("cost_usd"));
    insert_float(
        &mut doc,
        "f_switchyard_cost_input_usd",
        payload.get("cost_input_usd"),
    );
    insert_float(
        &mut doc,
        "f_switchyard_cost_output_usd",
        payload.get("cost_output_usd"),
    );

    if let Ok(raw) = serde_json::to_string(payload) {
        doc.insert(
            "text_switchyard_record_json".to_string(),
            Value::String(raw),
        );
    }

    Value::Object(doc)
}

/// Inserts a non-empty string field, skipping absent values.
fn insert_str(doc: &mut Map<String, Value>, key: &str, value: Option<&str>) {
    if let Some(value) = value.filter(|value| !value.is_empty()) {
        doc.insert(key.to_string(), Value::String(value.to_string()));
    }
}

/// Inserts an integer field, skipping absent / non-integer values.
fn insert_long(doc: &mut Map<String, Value>, key: &str, value: Option<&Value>) {
    if let Some(number) = value.and_then(Value::as_i64) {
        doc.insert(key.to_string(), Value::from(number));
    }
}

/// Inserts a float field, skipping absent / non-numeric values.
fn insert_float(doc: &mut Map<String, Value>, key: &str, value: Option<&Value>) {
    if let Some(number) = value.and_then(Value::as_f64) {
        doc.insert(key.to_string(), Value::from(number));
    }
}

/// Chooses the best timestamp for the intake context creation time.
fn created_at_iso(started_at_ms: Option<i64>, ended_at_ms: Option<i64>) -> String {
    started_at_ms
        .or(ended_at_ms)
        .map(ms_to_iso)
        .unwrap_or_else(|| ms_to_iso(now_millis()))
}

/// Formats milliseconds since epoch in Python-compatible UTC ISO form.
fn ms_to_iso(value_ms: i64) -> String {
    let seconds = value_ms.div_euclid(1000);
    let millis = value_ms.rem_euclid(1000);
    let days = seconds.div_euclid(86_400);
    let seconds_of_day = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let hour = seconds_of_day / 3600;
    let minute = seconds_of_day % 3600 / 60;
    let second = seconds_of_day % 60;
    if millis == 0 {
        format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}+00:00")
    } else {
        format!(
            "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millis:03}000+00:00"
        )
    }
}

/// Converts days since Unix epoch to a civil date without chrono.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp + if mp < 10 { 3 } else { -9 };
    let year = y + if m <= 2 { 1 } else { 0 };
    (year, m as u32, d as u32)
}

/// Capture state for OpenAI Chat Completions streaming.
pub struct OpenAiChatStreamCapture {
    /// Backend-selected model fallback.
    served_model: Option<String>,
    /// Completion ID from stream chunks.
    id: Option<String>,
    /// Model reported by stream chunks.
    model: Option<String>,
    /// Creation timestamp from stream chunks.
    created: Option<i64>,
    /// Accumulated visible assistant content.
    content: String,
    /// Accumulated reasoning content when providers stream it separately.
    reasoning_content: String,
    /// Tool calls keyed by OpenAI stream index.
    tool_calls: Vec<OpenAiToolCallState>,
    /// Last non-null finish reason seen in the stream.
    finish_reason: Option<Value>,
    /// First usage block seen in the stream.
    usage: Option<Value>,
}

impl OpenAiChatStreamCapture {
    /// Creates empty OpenAI Chat stream capture state.
    fn new(served_model: Option<&str>) -> Self {
        Self {
            served_model: served_model.map(str::to_string),
            id: None,
            model: None,
            created: None,
            content: String::new(),
            reasoning_content: String::new(),
            tool_calls: Vec::new(),
            finish_reason: None,
            usage: None,
        }
    }

    /// Merges one OpenAI Chat stream chunk into the capture state.
    fn observe(&mut self, event: &StreamEvent) {
        let StreamEvent::Json(value) = event else {
            return;
        };
        if self.id.is_none() {
            self.id = value.get("id").and_then(Value::as_str).map(str::to_string);
        }
        if self.model.is_none() {
            self.model = value
                .get("model")
                .and_then(Value::as_str)
                .map(str::to_string);
        }
        if self.created.is_none() {
            self.created = value.get("created").and_then(Value::as_i64);
        }
        if let Some(usage) = value.get("usage").filter(|usage| usage.is_object()) {
            self.usage = Some(usage.clone());
        }
        let Some(choice) = value
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
        else {
            return;
        };
        if let Some(delta) = choice
            .get("delta")
            .and_then(|delta| delta.get("content"))
            .and_then(Value::as_str)
        {
            self.content.push_str(delta);
        }
        for reasoning_key in ["reasoning_content", "reasoning"] {
            if let Some(reasoning) = choice
                .get("delta")
                .and_then(|delta| delta.get(reasoning_key))
                .and_then(Value::as_str)
            {
                self.reasoning_content.push_str(reasoning);
            }
        }
        if let Some(tool_calls) = choice
            .get("delta")
            .and_then(|delta| delta.get("tool_calls"))
            .and_then(Value::as_array)
        {
            for tool_call in tool_calls {
                self.merge_tool_call(tool_call);
            }
        }
        if let Some(reason) = choice
            .get("finish_reason")
            .filter(|reason| !reason.is_null())
        {
            self.finish_reason = Some(reason.clone());
        }
    }

    /// Merges a possibly-partial tool-call delta by stream index.
    fn merge_tool_call(&mut self, tool_call: &Value) {
        let index = tool_call
            .get("index")
            .and_then(Value::as_u64)
            .and_then(|index| usize::try_from(index).ok())
            .unwrap_or(self.tool_calls.len());
        while self.tool_calls.len() <= index {
            self.tool_calls.push(OpenAiToolCallState::default());
        }
        let existing = &mut self.tool_calls[index];
        if let Some(id) = tool_call
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
        {
            existing.id = Some(id.to_string());
        }
        let Some(function) = tool_call.get("function").and_then(Value::as_object) else {
            return;
        };
        if let Some(name) = function
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
        {
            existing.name = Some(name.to_string());
        }
        if let Some(arguments) = function
            .get("arguments")
            .and_then(Value::as_str)
            .filter(|arguments| !arguments.is_empty())
        {
            existing.arguments.push_str(arguments);
        }
    }

    /// Builds a buffered OpenAI Chat completion from captured stream state.
    fn finish(self) -> Value {
        let has_tool_calls = self.tool_calls.iter().any(OpenAiToolCallState::has_payload);
        let content = if self.content.is_empty() {
            Value::Null
        } else {
            Value::String(self.content)
        };
        let mut message = json!({
            "role": "assistant",
            "content": content,
        });
        if !self.reasoning_content.is_empty() {
            if let Value::Object(object) = &mut message {
                object.insert(
                    "reasoning_content".to_string(),
                    Value::String(self.reasoning_content),
                );
            }
        }
        if has_tool_calls {
            if let Value::Object(object) = &mut message {
                object.insert(
                    "tool_calls".to_string(),
                    Value::Array(
                        self.tool_calls
                            .into_iter()
                            .enumerate()
                            .filter_map(|(index, tool_call)| tool_call.into_openai(index))
                            .collect(),
                    ),
                );
            }
        }
        let mut response = json!({
            "id": self.id.unwrap_or_else(|| "chatcmpl-switchyard-stream".to_string()),
            "object": "chat.completion",
            "created": self.created.unwrap_or(0),
            "model": self
                .model
                .or(self.served_model)
                .unwrap_or_else(|| UNKNOWN_MODEL.to_string()),
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": self
                    .finish_reason
                    .unwrap_or_else(|| default_openai_finish_reason(has_tool_calls)),
            }],
        });
        if let Some(usage) = self.usage {
            if let Value::Object(object) = &mut response {
                object.insert("usage".to_string(), usage);
            }
        }
        response
    }
}

#[derive(Default)]
struct OpenAiToolCallState {
    /// Provider tool-call ID.
    id: Option<String>,
    /// Function name.
    name: Option<String>,
    /// Incrementally assembled JSON arguments string.
    arguments: String,
}

impl OpenAiToolCallState {
    /// Returns true when this partial tool call carries useful data.
    fn has_payload(&self) -> bool {
        self.id.is_some() || self.name.is_some() || !self.arguments.is_empty()
    }

    /// Converts captured tool-call state into OpenAI Chat tool_call shape.
    fn into_openai(self, index: usize) -> Option<Value> {
        self.has_payload().then(|| {
            json!({
                "id": self.id.unwrap_or_else(|| format!("call_switchyard_{index}")),
                "type": "function",
                "function": {
                    "name": self.name.unwrap_or_default(),
                    "arguments": self.arguments,
                },
            })
        })
    }
}

/// Capture state for Anthropic Messages streaming.
pub struct AnthropicStreamCapture {
    /// Backend-selected model fallback.
    served_model: Option<String>,
    /// Message ID from the stream.
    id: Option<String>,
    /// Model reported by the stream.
    model: Option<String>,
    /// Content blocks keyed by Anthropic content index.
    content_blocks: BTreeMap<usize, AnthropicContentBlockState>,
    /// Input tokens from usage events.
    input_tokens: i64,
    /// Output tokens from usage events.
    output_tokens: i64,
    /// Cache-read input tokens from usage events.
    cache_read_input_tokens: i64,
    /// Cache-creation input tokens from usage events.
    cache_creation_input_tokens: i64,
    /// Anthropic stop reason from the stream.
    stop_reason: Option<Value>,
    /// Whether any usage event was seen.
    saw_usage: bool,
}

impl AnthropicStreamCapture {
    /// Creates empty Anthropic stream capture state.
    fn new(served_model: Option<&str>) -> Self {
        Self {
            served_model: served_model.map(str::to_string),
            id: None,
            model: None,
            content_blocks: BTreeMap::new(),
            input_tokens: 0,
            output_tokens: 0,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
            stop_reason: None,
            saw_usage: false,
        }
    }

    /// Merges one Anthropic Messages stream event into the capture state.
    fn observe(&mut self, event: &StreamEvent) {
        let StreamEvent::Json(value) = event else {
            return;
        };
        match value.get("type").and_then(Value::as_str) {
            Some("message_start") => {
                if let Some(message) = value.get("message") {
                    if self.id.is_none() {
                        self.id = message
                            .get("id")
                            .and_then(Value::as_str)
                            .map(str::to_string);
                    }
                    if self.model.is_none() {
                        self.model = message
                            .get("model")
                            .and_then(Value::as_str)
                            .map(str::to_string);
                    }
                    if let Some(usage) = message.get("usage") {
                        self.merge_usage(usage);
                    }
                }
            }
            Some("content_block_start") => {
                let index = coerce_usize(value.get("index")).unwrap_or(0);
                let Some(block) = value.get("content_block").and_then(Value::as_object) else {
                    return;
                };
                if block.get("type").and_then(Value::as_str) == Some("tool_use") {
                    self.content_blocks.insert(
                        index,
                        AnthropicContentBlockState {
                            kind: AnthropicContentKind::ToolUse,
                            text: String::new(),
                            id: block.get("id").and_then(Value::as_str).map(str::to_string),
                            name: block
                                .get("name")
                                .and_then(Value::as_str)
                                .map(str::to_string),
                            arguments: json_argument_string(block.get("input")),
                        },
                    );
                    return;
                }
                self.content_blocks.insert(
                    index,
                    AnthropicContentBlockState {
                        kind: AnthropicContentKind::Text,
                        text: block
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        id: None,
                        name: None,
                        arguments: String::new(),
                    },
                );
            }
            Some("content_block_delta") => {
                let index = coerce_usize(value.get("index")).unwrap_or(0);
                let Some(delta) = value.get("delta").and_then(Value::as_object) else {
                    return;
                };
                let block = self
                    .content_blocks
                    .entry(index)
                    .or_insert_with(AnthropicContentBlockState::text);
                if delta.get("type").and_then(Value::as_str) == Some("input_json_delta") {
                    block.kind = AnthropicContentKind::ToolUse;
                    if let Some(partial_json) = delta.get("partial_json").and_then(Value::as_str) {
                        block.arguments.push_str(partial_json);
                    }
                    return;
                }
                if block.kind != AnthropicContentKind::Text {
                    return;
                }
                if let Some(text) = delta.get("text").and_then(Value::as_str) {
                    block.text.push_str(text);
                    return;
                }
                if let Some(thinking) = delta.get("thinking").and_then(Value::as_str) {
                    block.text.push_str(thinking);
                }
            }
            Some("message_delta") => {
                if let Some(usage) = value.get("usage") {
                    self.merge_usage(usage);
                }
                if let Some(reason) = value
                    .get("delta")
                    .and_then(|delta| delta.get("stop_reason"))
                    .filter(|reason| !reason.is_null())
                {
                    self.stop_reason = Some(reason.clone());
                }
            }
            _ => {}
        }
    }

    /// Merges Anthropic usage counters, preserving the latest delta semantics.
    fn merge_usage(&mut self, usage: &Value) {
        if !usage.is_object() {
            return;
        }
        self.saw_usage = true;
        if let Some(value) = coerce_i64(usage.get("input_tokens")) {
            self.input_tokens = value;
        }
        if let Some(value) = coerce_i64(usage.get("output_tokens")) {
            self.output_tokens = value;
        }
        if let Some(value) = coerce_i64(usage.get("cache_read_input_tokens")) {
            self.cache_read_input_tokens = value;
        }
        if let Some(value) = coerce_i64(usage.get("cache_creation_input_tokens")) {
            self.cache_creation_input_tokens = value;
        }
    }

    /// Builds a buffered OpenAI Chat completion from captured Anthropic state.
    fn finish(self) -> Value {
        let prompt_tokens = self
            .input_tokens
            .saturating_add(self.cache_read_input_tokens)
            .saturating_add(self.cache_creation_input_tokens);
        let content = self
            .content_blocks
            .values()
            .filter(|block| block.kind == AnthropicContentKind::Text)
            .map(|block| block.text.as_str())
            .collect::<String>();
        let tool_calls = self
            .content_blocks
            .values()
            .filter(|block| block.kind == AnthropicContentKind::ToolUse)
            .enumerate()
            .map(|(index, block)| block.to_openai_tool_call(index))
            .collect::<Vec<_>>();
        let has_tool_calls = !tool_calls.is_empty();
        let mut message = json!({
            "role": "assistant",
            "content": if content.is_empty() { Value::Null } else { Value::String(content) },
        });
        if has_tool_calls {
            if let Value::Object(object) = &mut message {
                object.insert("tool_calls".to_string(), Value::Array(tool_calls));
            }
        }
        let mut response = json!({
            "id": self.id.unwrap_or_else(|| "msg_switchyard_stream".to_string()),
            "object": "chat.completion",
            "model": self
                .model
                .or(self.served_model)
                .unwrap_or_else(|| UNKNOWN_MODEL.to_string()),
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": self
                    .stop_reason
                    .map(anthropic_stop_reason_to_openai)
                    .unwrap_or_else(|| default_openai_finish_reason(has_tool_calls)),
            }],
        });
        if self.saw_usage {
            let mut usage = Map::new();
            usage.insert("prompt_tokens".to_string(), Value::from(prompt_tokens));
            usage.insert(
                "completion_tokens".to_string(),
                Value::from(self.output_tokens),
            );
            usage.insert(
                "total_tokens".to_string(),
                Value::from(prompt_tokens.saturating_add(self.output_tokens)),
            );
            if self.cache_read_input_tokens != 0 || self.cache_creation_input_tokens != 0 {
                usage.insert(
                    "prompt_tokens_details".to_string(),
                    json!({
                        "cached_tokens": self.cache_read_input_tokens,
                        "cache_creation_tokens": self.cache_creation_input_tokens,
                    }),
                );
            }
            if let Value::Object(object) = &mut response {
                object.insert("usage".to_string(), Value::Object(usage));
            }
        }
        response
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum AnthropicContentKind {
    /// Text or thinking block that should become assistant content.
    Text,
    /// Tool-use block that should become an OpenAI tool call.
    ToolUse,
}

struct AnthropicContentBlockState {
    /// Captured block kind.
    kind: AnthropicContentKind,
    /// Accumulated text content for text blocks.
    text: String,
    /// Anthropic tool-use ID for tool blocks.
    id: Option<String>,
    /// Tool name for tool-use blocks.
    name: Option<String>,
    /// Incrementally assembled tool input JSON.
    arguments: String,
}

impl AnthropicContentBlockState {
    /// Creates an empty text block for deltas that arrive before a start event.
    fn text() -> Self {
        Self {
            kind: AnthropicContentKind::Text,
            text: String::new(),
            id: None,
            name: None,
            arguments: String::new(),
        }
    }

    /// Converts an Anthropic tool-use block into OpenAI Chat tool-call shape.
    fn to_openai_tool_call(&self, index: usize) -> Value {
        json!({
            "id": self
                .id
                .clone()
                .unwrap_or_else(|| format!("toolu_switchyard_{index}")),
            "type": "function",
            "function": {
                "name": self.name.clone().unwrap_or_default(),
                "arguments": self.arguments.clone(),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use switchyard_core::Result;

    // Timestamp formatting should match the Python intake payloads exactly.
    #[test]
    fn formats_python_style_utc_iso_timestamps() -> Result<()> {
        assert_eq!(ms_to_iso(1_700_000_000_000), "2023-11-14T22:13:20+00:00");
        assert_eq!(
            ms_to_iso(1_700_000_001_840),
            "2023-11-14T22:13:21.840000+00:00"
        );
        Ok(())
    }
}
